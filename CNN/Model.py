import sys
import math

import torch
import torch.nn as nn
from dropblock import DropBlock2D

class MJONet(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

		#Convolutional layer one
        self.add_module('conv0', nn.Conv2d(num_channels, 30, kernel_size=5, padding=(2, 2), stride=1, bias=False))
        self.drop_block0 = DropBlock2D(block_size=3, drop_prob=0.1)
        self.add_module('act0', nn.LeakyReLU(negative_slope=0.003, inplace=True))
        self.add_module('pool0', nn.AvgPool2d(kernel_size=3, stride=3, padding=1))

		#Convolutional layer two
        self.add_module('conv1', nn.Conv2d(30, 40, kernel_size=5, padding=(2, 2), stride=1, bias=False))
        self.drop_block1 = DropBlock2D(block_size=3, drop_prob=0.3)
        self.add_module('act1', nn.LeakyReLU(negative_slope=0.003, inplace=True))
        self.add_module('pool1', nn.AvgPool2d(kernel_size=3, stride=3, padding=1))

 		# #Convolutional layer three
        self.add_module('conv2', nn.Conv2d(40, 60, kernel_size=3, padding=1, stride=1, bias=False))
        self.drop_block2 = DropBlock2D(block_size=3, drop_prob=0.3)
        self.add_module('act2', nn.LeakyReLU(negative_slope=0.003, inplace=True))

		#Fully connected layer
        self.add_module('linear1', nn.Linear(1920, 200))
        self.add_module('dp1', nn.Dropout(p = 0.3))
        self.add_module('linear2', nn.Linear(200,4))

		#Randomly initialize the weights of the CNN
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

                
    def forward(self, x):
        x_init_shape = x.shape

        x = self.pool0(self.act0(self.drop_block0(self.conv0(x))))
        x = self.pool1(self.act1(self.drop_block1(self.conv1(x))))
        x = self.act2(self.drop_block2(self.conv2(x)))
        
        x = x.view(x_init_shape[0],-1)
        x = self.dp1(self.linear1(x))
        x = self.linear2(x)
        
        zeros = torch.tensor([0.0], dtype = torch.double)
        
        mu = x[:,:2]
        
        #Rescale output for sigma1/2 to ensure they are positive
        sigma1 = torch.log(1 + torch.exp(x[:,2]))
        sigma2 = torch.log(1 + torch.exp(x[:,3]))
        
        #Setting sigma1 and 2 into a covariance matrix
        cov = torch.zeros(x_init_shape[0], 2, 2)
        cov[:,0,0] = sigma1
        cov[:,1,1] = sigma2
        
        return mu.type(torch.DoubleTensor), cov.type(torch.DoubleTensor)