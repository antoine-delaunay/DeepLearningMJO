from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

from Model import MJONet
from Dataset import MJODataset

torch.manual_seed(0)

T_OUTPUT = 10    #Forecast lead time
num_channels = 7 #number of input fields, default: 7, change if datasets with less/more input variables

DatasetsDir = '/network/aopp/preds0/pred/users/delaunay/Data/Datasets/'
ModelDir = '/network/aopp/preds0/pred/users/delaunay/ModelSaves/Lag' + str(T_OUTPUT)+'/'

model = MJONet(num_channels).double()
train_dataloader = torch.load(DatasetsDir + 'Train_' + str(T_OUTPUT) +'.pt')
test_dataloader = torch.load(DatasetsDir + 'Test_' + str(T_OUTPUT) + '.pt')
    
def train(model, train_dataloader, test_dataloader, n_epochs= 35):
    optimizer =  torch.optim.Adam(model.parameters(), weight_decay = 0.01)     
    model.train(True)
    
    loss_train = np.zeros(n_epochs)
    loss_test = np.zeros(n_epochs)
    test_len = test_dataloader.dataset[:][0].shape[0]
    
    best_loss = 10e+3
    best_epoch = 0 
    
    for epoch_num in range(n_epochs):
        model.train(True)
        epoch_train_loss = 0.0
        nProcessed = 0
        
        #Forward pass with all the batches
        for data in tqdm(train_dataloader):
            optimizer.zero_grad()
            inputs,_,_,_, targets = data

            mu, cov = model(inputs)   
            
            #Computing the negative log-likelihood
            dist = MultivariateNormal(mu, covariance_matrix = cov)
            loss = -(dist.log_prob(targets)).mean()
            epoch_train_loss += loss.item()*targets.shape[0]
            loss.backward()

            optimizer.step()

            nProcessed += targets.shape[0]   
        
        #Average loss on the train epoch
        epoch_train_loss /= nProcessed
        loss_train[epoch_num] = epoch_train_loss
        
        #At each epoch, we compute the negative log-likelihood on the test dataset as well
        model.train(False)
        epoch_test_loss = 0.0
        nProcessed = 0
        preds = torch.empty(test_len, 2)
        observations = torch.empty(test_len, 2)
        j = 0
        
        for data in tqdm(test_dataloader):
            inputs,_,_,_, targets = data

            mu, cov = model(inputs)   
            dist = MultivariateNormal(mu, covariance_matrix = cov)

            epoch_test_loss += -(dist.log_prob(targets)).mean().item()*targets.shape[0]
            
            preds[j : j + targets.shape[0]][:] = mu.detach()
            observations[j : j + targets.shape[0]][:] = targets.detach()
            j+=targets.shape[0]
              
            nProcessed += targets.shape[0]
        
        epoch_test_loss = epoch_test_loss/nProcessed
        
        loss_test[epoch_num] = epoch_test_loss
        
        print('Train - Epoch: {:.0f}'.format(epoch_num+1))
        print('Train - Loss: {:.8f}'.format(epoch_train_loss))
        print('Test - Loss: {:.8f}'.format(epoch_test_loss) + '\n')
                
        #Save the model if the test loss is better
        if epoch_test_loss<best_loss:
          best_loss = epoch_test_loss
          best_epoch = epoch_num+1
          torch.save(model, ModelDir + 'model_'+str(T_OUTPUT) + '.pt')
    
    #Store the train and test losses in a table
    results = np.zeros((n_epochs,2))
    results[:,0], results[:,1] = loss_train, loss_test
    df = pd.DataFrame(data=results)
    df.columns = ['Train Loss', 'Test Loss']
    df.to_csv(ModelDir +'Results_' + str(T_OUTPUT) + '.txt', index = False)
       
    return loss_train, loss_test
 
if __name__ == '__main__':
    print("\n T_OUTPUT = {:.0f}".format(T_OUTPUT))
    train(model, train_dataloader, test_dataloader, n_epochs = 35)
