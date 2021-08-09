import torch
import torch.nn as nn
import PatternNetworks
import PatternLayers
import sys
sys.path.append('../../CNN/')
from Model import MJONet
from Dataset import MJODataset

'''
Compute the attention vectors using the train dataset.
Only 2000 samples at a time are fed to the network to avoid computation crashes
'''

T_OUTPUT = 10
ModelDir = '/network/aopp/preds0/pred/users/delaunay/ModelSaves/Lag'
SaveDir = '/network/aopp/preds0/pred/users/delaunay/Data/'

#Load the train dataset and the CNN model
dataloader = torch.load('/network/aopp/preds0/pred/users/delaunay/Data/Datasets/Train_'+str(T_OUTPUT)+'.pt')
X0_train, y0_train, amp0_train =  dataloader.dataset.X0, dataloader.dataset.y0, dataloader.dataset.amp0

m = torch.load(ModelDir + str(T_OUTPUT)+'/model_'+str(T_OUTPUT)+'.pt')
model = nn.Sequential(m.conv0, m.act0, m.pool0, m.conv1, m.act1, m.pool1, m.conv2, m.act2, m.linear1, m.linear2)
model.layers = list(model.children())

#Create PatternNet and start computing the attribution vectors
patternNet = PatternNetworks.PatternNet(model.layers)
torch.save(patternNet, ModelDir + str(T_OUTPUT) +'/patternNet_'+str(T_OUTPUT)+'.pt')

print('Computing statistics - Start')

for i in range(y0_train.shape[0]//2000):
    print('Updating statistics - ' + str(i+1) + '/' + str(y0_train.shape[0]//2000))
    patternNet = torch.load(ModelDir + str(T_OUTPUT) + '/patternNet_'+str(T_OUTPUT)+'.pt')
    patternNet.compute_statistics(X0_train[2000*i:2000*(i+1)])
    torch.save(patternNet, ModelDir + str(T_OUTPUT) +'/patternNet_'+str(T_OUTPUT)+'.pt')
        
print('Last update')
patternNet = torch.load(ModelDir + str(T_OUTPUT) + '/patternNet_'+str(T_OUTPUT)+'.pt')
patternNet.compute_statistics(X0_train[2000*(y0_train.shape[0]//2000):])
torch.save(patternNet, ModelDir + str(T_OUTPUT) +'/patternNet_'+str(T_OUTPUT)+'.pt')

print('Statistics computed')