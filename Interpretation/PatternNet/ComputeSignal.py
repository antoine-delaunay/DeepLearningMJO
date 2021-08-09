import torch
import PatternNetworks
import PatternLayers
from tqdm import tqdm
import sys
sys.path.append('../../CNN/')
from Model import MJONet
from Dataset import MJODataset

'''
Load the trained PatternNet and use it to compute the signals of the train and test dataset
'''

T_OUTPUT = 10
ModelDir = '/network/aopp/preds0/pred/users/delaunay/ModelSaves/Lag'
DataDir = '/network/aopp/preds0/pred/users/delaunay/Data/Datasets/'
SaveDir = '/network/aopp/preds0/pred/users/delaunay/Data/'

#Load the train dataset and the patternet model
dataloader = torch.load(DataDir + 'Train_'+str(T_OUTPUT)+'.pt')
X0_train, y0_train, amp0_train =  dataloader.dataset.X0, dataloader.dataset.y0, dataloader.dataset.amp0
n_train = y0_train.shape[0]

#Load the test dataset
dataloader = torch.load(DataDir + 'Test_'+str(T_OUTPUT)+'.pt')
X0_test, y0_test, amp0_test =  dataloader.dataset.X0, dataloader.dataset.y0, dataloader.dataset.amp0
n_test = y0_test.shape[0]

#Load the patternet
patternet = torch.load(ModelDir + str(T_OUTPUT) + '/patternNet_'+str(T_OUTPUT)+'.pt')

#Compute and set patterns
patternet.compute_patterns()
print('Patterns computed')

patternet.set_patterns()
print('Patterns set') 

#Compute the signals of the train dataset
print('Compute the signals of the train dataset')
signal1_train = torch.zeros(n_train, 7, 17, 144)
signal2_train = torch.zeros(n_train, 7, 17, 144)

for i in tqdm(range(n_train//100)):
    signal1_train[100*i: 100*(i+1),:,:,:] = patternet(X0_train[100*i:100*(i+1)], index = 0).detach()
    signal2_train[100*i: 100*(i+1),:,:,:] = patternet(X0_train[100*i:100*(i+1)], index = 1).detach()

signal1_train[100*(n_train//100):n_train, :,:,:] = patternet(X0_train[100*(n_train//100):], index = 0)
signal2_train[100*(n_train//100):n_train, :,:,:] = patternet(X0_train[100*(n_train//100):], index = 1)

torch.save(signal1_train, SaveDir + 'Forecasts/CNN/Signals/signal1_train_' + str(T_OUTPUT) + '.pt')
torch.save(signal2_train, SaveDir + 'Forecasts/CNN/Signals/signal2_train_' + str(T_OUTPUT) + '.pt')

#Compute the signals of the test dataset
print('Compute the signals of the test dataset')
signal1_test = torch.zeros(n_test, 7, 17, 144)
signal2_test = torch.zeros(n_test, 7, 17, 144)

for i in tqdm(range(n_test//100)):
    signal1_test[100*i: 100*(i+1),:,:,:] = patternet(X0_test[100*i:100*(i+1)], index = 0).detach()
    signal2_test[100*i: 100*(i+1),:,:,:] = patternet(X0_test[100*i:100*(i+1)], index = 1).detach()

signal1_test[100*(n_test//100):, :,:,:] = patternet(X0_test[100*(n_test//100):], index = 0)
signal2_test[100*(n_test//100):, :,:,:] = patternet(X0_test[100*(n_test//100):], index = 1)

torch.save(signal1_test, SaveDir + 'Forecasts/CNN/Signals/signal1_test_' + str(T_OUTPUT) + '.pt')
torch.save(signal2_test, SaveDir + 'Forecasts/CNN/Signals/signal2_test_' + str(T_OUTPUT) + '.pt')
