'''
Adapted from https://github.com/TNTLFreiburg/pytorch_patternnet
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import patterns
import time
import PatternLayers


class PatternNet(torch.nn.Module):

    def __init__(self, layers_lst):
        super(PatternNet, self).__init__()

        # initialize variables for later use
        self.lst_layers = layers_lst

        # save change from conv to linear layers to know when to reshape
        # get first conv or linear layer
        cur_layer_ind = 0
        while not (self.lst_layers[cur_layer_ind].__class__.__name__ in \
                ["Conv2d","Linear"]) and cur_layer_ind < len(self.lst_layers):

            cur_layer_ind+=1
        # if only the last layer is conv or linear, do not need to find next layer
        if cur_layer_ind < len(self.lst_layers) -1:
            cur_layer = self.lst_layers[cur_layer_ind].__class__.__name__
            next_layer_ind = cur_layer_ind + 1
            while cur_layer_ind < len(self.lst_layers) -1:
                while not (self.lst_layers[next_layer_ind].__class__.__name__ in \
                        ["Conv2d","Linear"]) and next_layer_ind < len(self.lst_layers):
                    next_layer_ind+=1
                next_layer = self.lst_layers[next_layer_ind].__class__.__name__
                # now check if there's a change between conv and linear layers
                if cur_layer == "Conv2d" and next_layer == "Linear":
                    self.reshape_ind = next_layer_ind
                cur_layer_ind = next_layer_ind
                cur_layer = next_layer
                next_layer_ind = cur_layer_ind +1
        
        #self.lst_layers.append(PatternLayers.PatternLastLayer())

        # initialize the backward layers
        self.backward_layers = []
        #Automatically add last_layer
        #self.backward_layers.append(PatternLayers.PatternLastLayer())
        for layer in self.lst_layers[::-1]: #[::-1]:
            if layer.__class__.__name__ == "Conv2d":
                self.backward_layers.append(PatternLayers.PatternConv2d(layer))
            if layer.__class__.__name__ == "Linear":
                self.backward_layers.append(PatternLayers.PatternLinear(layer))
            if layer.__class__.__name__ == "MaxPool2d":
                self.backward_layers.append(PatternLayers.PatternMaxPool2d(layer))
            if layer.__class__.__name__ == "AvgPool2d":
                self.backward_layers.append(PatternLayers.PatternAvgPool2d(layer))
            if layer.__class__.__name__ == "ReLU":
                self.backward_layers.append(PatternLayers.PatternReLU())
            if layer.__class__.__name__ == "LeakyReLU":
                self.backward_layers.append(PatternLayers.PatternLeakyReLU(layer.negative_slope))
        


    def compute_signal(self, img, only_biggest_value=True):
        """ Additional method to compute the signal for a given image. """
        return self.forward(img, only_biggest_value)

    def forward(self, img, only_biggest_value=False, index=None):

        output, _, indices, switches = self.forward_with_extras(img)
        
        if index!=None:
            y = torch.zeros(output.size(), requires_grad=False, dtype=torch.double)
            y[:,index] = output[:,index]
            #print(index, y.shape)          

        elif not only_biggest_value:
            y = output
            #print(y.shape)
        # use only highest valued output
        else:
            y = torch.zeros(output.size(), requires_grad=False, dtype=torch.double)
            #print(y.type())
            max_v, max_i = torch.max(output.data, dim=1)
            #print(max_v.type())
            y[range(y.shape[0]), max_i] = max_v

        ind_cnt = 0
        switch_cnt = 0

        # go through all layers and apply their backward pass functionality
        for ind, layer in enumerate(self.backward_layers):
            '''
            print(layer.__class__.__name__)
            print(y.shape)
            print(y.mean())
            print(y.min())
            print(y.max())
            '''
            if layer.__class__.__name__ == "PatternReLU":
                mask = indices[ind_cnt]
                y = layer.backward(y, mask)
                ind_cnt += 1
            elif layer.__class__.__name__ == "PatternLeakyReLU":
                mask = indices[ind_cnt]
                y = layer.backward(y, mask)
                ind_cnt += 1
            elif layer.__class__.__name__ == "PatternLastLayer":
                y = layer.backward(y)

            elif layer.__class__.__name__ == "PatternMaxPool2d":
                y = layer.backward(y, switches[switch_cnt])
                switch_cnt += 1
            else:
                # if other layer than linear or conv, could theoretically
                # be applied here without noticing
                y = layer.backward(y)
                
                # check if reshape is necessary
                #print(len(self.lst_layers) - ind, self.reshape_ind + 1)
                if len(self.lst_layers) - ind == self.reshape_ind + 1:
                    s = self._reshape_size_in
                    y.data = y.data.view(-1, s[1], s[2], s[3])
 
        return y

    def forward_with_extras(self, imgs):
        """
        Performs one forward pass through the network given at initialization
        (only convolutional, linear, pooling and ReLU layer). Additionally to
        the final output the input and output to each convolutional and linear
        layer, the switches of pooling layers and the indices of the values
        that are set to zero of each ReLU layer, are returned.
        """

        output = Variable(imgs, requires_grad=False)

        layers = []
        layers_wo_bias = []
        cnt = 0
        indices = []
        switches = []

        for ind, layer in enumerate(self.backward_layers[::-1]):
            #print(layer.forward_layer)
            if layer.__class__.__name__ == "PatternConv2d":
                # save input to layer
                layers.append({})
                #print('PatternConv2d')
                layers[cnt]["inputs"] = output.data
                # apply forward layer
                output, output_wo_bias = layer(output)
                # save output of layer
                layers[cnt]["outputs"] = output.data
                # save output without bias
                layers_wo_bias.append(output_wo_bias)
                cnt += 1
            elif layer.__class__.__name__ == "PatternLinear":
                # save input to layer
                layers.append({})
                layers[cnt]["inputs"] = output.data
                # apply layer
                output, output_wo_bias = layer(output)
                # save output of layer
                layers[cnt]["outputs"] = output.data
                # save output without bias
                layers_wo_bias.append(output_wo_bias)
                cnt += 1
            elif layer.__class__.__name__ == "PatternMaxPool2d":
                # set return indices to true to get the switches
                # apply layer
                output, switch = layer(output)
                # save switches
                switches.append(switch)
            elif layer.__class__.__name__ == "PatternAvgPool2d":
                output = layer(output)
            elif layer.__class__.__name__ == "PatternReLU":
                # save indices smaller zero
                output, inds = layer(output)
                indices.append(inds)
            elif layer.__class__.__name__ == "PatternLeakyReLU":
                # save indices smaller zero
                output, inds = layer(output)
                indices.append(inds)
            elif layer.__class__.__name__ == "PatternLastLayer":
                # save indices smaller zero
                output = layer(output)
        
            # add view between convolutional and linear sequential
            if ind == self.reshape_ind-1:  # layer before the first linear layer
                    self._reshape_size_in = output.shape
                    output = output.view(-1,self.lst_layers[ind+1].in_features)


        return (
            output,
            (layers, layers_wo_bias),
            indices[::-1],
            switches[::-1],
        )

    def compute_statistics(self, imgs, use_bias=False):
        """ Initializes statistics if no statistics were computed
            before. Otherwise updates the already computed statistics.
        """

        # get the layer outputs
        a, outputs, _, _ = self.forward_with_extras(imgs)
        layer_outputs = outputs[0]
        layer_outputs_wo_bias = outputs[1]
        print('Forward with extra done')
        # cnt for layers with params
        cnt = 0
        for layer in self.backward_layers[::-1]:
            print(layer.__class__.__name__)
            if layer.__class__.__name__ in ["PatternConv2d", "PatternLinear"]:
                layer.compute_statistics(layer_outputs[cnt]["inputs"],
                                         layer_outputs[cnt]["outputs"],
                                         layer_outputs_wo_bias[cnt])
                cnt += 1


    def compute_patterns(self):

            for layer in self.backward_layers[::-1]:
                if layer.__class__.__name__ in ["PatternConv2d", "PatternLinear"]:
                    layer.compute_patterns()


    def set_patterns(self, pattern_type="relu"):
        """ pattern_type can either be A_plus or A_linear
        """
        for layer in self.backward_layers[::-1]:
            if layer.__class__.__name__ in ["PatternConv2d", "PatternLinear"]:
                layer.set_patterns(pattern_type=pattern_type)