import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from properscoring import crps_gaussian

import sys
sys.path.append('../CNN/')

from Dataset import MJODataset
from Model import MJONet

DataDir = '/network/aopp/preds0/pred/users/delaunay/Data/Datasets/'
ModelDir = '/network/aopp/preds0/pred/users/delaunay/ModelSaves/Lag' 
PredsDir = '/network/aopp/preds0/pred/users/delaunay/Data/Forecasts/CNN/Preds/'
PlotsDir = '/network/aopp/preds0/pred/users/delaunay/Analysis/Performance/Plots/'
ResultsDir = '/network/aopp/preds0/pred/users/delaunay/Analysis/Performance/Results/'

n_members = 10  #Choose number of members for the MC-Dropout
seeds = [20*i for i in range(n_members)]

T_OUTPUTS = [1,3,5,10,15,20,25,30,35]
n_bins = 5 #Number of bins for the RMS Error / Spread diagram


def RMSE(mu, targets):
    '''Compute the RMSE'''
    return torch.sqrt(((mu[:,0] -targets[:,0])**2 + (mu[:,1]-targets[:,1])**2).mean(dim=0))

def BivariateCorrelation(preds, targets):
    '''Compute the Bivariate Correlation'''
    num = (torch.mul(preds[:,0], targets[:,0]) + torch.mul(preds[:,1], targets[:,1])).sum()
    denom1 = (torch.norm(preds, dim = 1).square().sum()).sqrt()
    denom2 = (torch.norm(targets, dim = 1).square().sum()).sqrt()
    return num/(denom1*denom2)

def Amplitude(x):
    return torch.norm(x, dim = 1)

def AmpError(mu, targets):
    '''Compute the amplitude error'''
    return (Amplitude(mu) - Amplitude(targets)).mean(dim =0)
    
def PhaseError(mu, targets):
    '''Compute the phase error'''
    num = (torch.mul(targets[:,0],mu[:,1]) - torch.mul(targets[:,1], mu[:,0]))
    denom = (torch.mul(mu[:,0], mu[:,0]) + torch.mul(mu[:,1], mu[:,1]))
    return torch.atan(num/denom).mean(dim = 0)*180.0/np.pi

def CRPS(mu, cov, targets):
    '''Compute the CRPS'''
    return np.array([crps_gaussian(targets[i,0], mu=mu[i,0], sig = cov[i,0,0].sqrt()) + crps_gaussian(targets[i,1], mu=mu[i,1], sig = cov[i,1,1].sqrt()) for i in range(targets.shape[0])]).mean()
    
def LogScore(mu, cov, targets):
    '''Compute the Log-score/Ignorance-score'''
    dist = MultivariateNormal(mu, covariance_matrix = cov)
    loss = -(dist.log_prob(targets)).mean()
    return loss
        
def ErrorDrop(preds_mu_mean, targets, spread_epistemic, spread_aleatoric, spread_total, T_OUTPUT):
    '''Compute the Error Drop. First sort the data according to the predicted uncertainty / rmse.
    Then remove the alpha percent most uncertain data gradually and compute the rmse of the 1-alpha percent remaining meanwhile.
    For each RMM, the error drop is the ratio of the rmse between the 5% most certain forecasts and all the forecasts' rmse.
    '''
    mse = (preds_mu_mean - targets).square()
    
    #"spread" arrays = arrays of forecasted uncertainty in the chronological order
    #"uncertainty" arrays = arrays of forecasted uncertainty according to the sorting order
    
    spread_epistemic = torch.from_numpy(spread_epistemic)    
    spread_aleatoric = torch.from_numpy(spread_aleatoric)    
    spread_total = torch.from_numpy(spread_total)    
        
    #Sorting forecasts according to RMM1 predicted spread
    uncertainty_epi = torch.zeros(preds_mu_mean.shape[0], 4)
    uncertainty_alea = torch.zeros(preds_mu_mean.shape[0], 4)    
    uncertainty_total = torch.zeros(preds_mu_mean.shape[0], 4)
    uncertainty_mse = torch.zeros(preds_mu_mean.shape[0], 4)
    
    sort_epi = torch.sort(spread_epistemic[:,0], descending=True)
    sort_alea = torch.sort(spread_aleatoric[:,0], descending=True)
    sort_total = torch.sort(spread_total[:,0], descending=True) 
        
    uncertainty_epi[:,0], uncertainty_epi[:,1] = sort_epi[0], (preds_mu_mean[sort_epi[1], 0] - targets[sort_epi[1], 0]).square()    
    uncertainty_alea[:,0], uncertainty_alea[:,1] = sort_alea[0], (preds_mu_mean[sort_alea[1], 0] - targets[sort_alea[1], 0]).square()
    uncertainty_total[:,0], uncertainty_total[:,1] = sort_total[0], (preds_mu_mean[sort_total[1], 0] - targets[sort_total[1], 0]).square()
            
    #Sorting forecasts according to RMM2 predicted spread    
    sort_epi = torch.sort(spread_epistemic[:,1], descending=True)
    sort_alea = torch.sort(spread_aleatoric[:,1], descending=True)
    sort_total = torch.sort(spread_total[:,1], descending=True)    
    uncertainty_epi[:,2], uncertainty_epi[:,3] = sort_epi[0], (preds_mu_mean[sort_epi[1], 1] - targets[sort_epi[1], 1]).square()
    uncertainty_alea[:,2], uncertainty_alea[:,3] = sort_alea[0], (preds_mu_mean[sort_alea[1], 1] - targets[sort_alea[1], 1]).square()
    uncertainty_total[:,2], uncertainty_total[:,3] = sort_total[0], (preds_mu_mean[sort_total[1], 1] - targets[sort_total[1], 1]).square()

    #Sorting forecasts according to the RMM1/2 MSE
    sort_mse = torch.sort(mse[:,0], descending = True)  
    uncertainty_mse[:,0], uncertainty_mse[:,1] = sort_mse[0], mse[sort_mse[1],0]
    sort_mse = torch.sort(mse[:,1], descending = True)  
    uncertainty_mse[:,2], uncertainty_mse[:,3] = sort_mse[0], mse[sort_mse[1],1]
    
    #For each alpha from 0 to 0.95, compute the RMSE of the remaining forecasts 1- alpha forecasts.
    results = torch.zeros(20,8)
    alphas = [i*0.05 for i in range(20)]
    test = torch.zeros(20)
    for i, alpha in enumerate(alphas):
        idx = int(preds_mu_mean.shape[0]*alpha)
        results[i,0], results[i,1], results[i,2], results[i,3] = uncertainty_epi[idx:,1].mean(dim=0).sqrt(),uncertainty_alea[idx:,1].mean(dim=0).sqrt(), uncertainty_total[idx:,1].mean(dim=0).sqrt(), uncertainty_mse[idx:,1].mean(dim=0).sqrt()
       
        results[i,4], results[i,5], results[i,6], results[i,7] = uncertainty_epi[idx:,3].mean(dim=0).sqrt(),uncertainty_alea[idx:,3].mean(dim=0).sqrt(), uncertainty_total[idx:,3].mean(dim=0).sqrt(), uncertainty_mse[idx:,3].mean(dim=0).sqrt()
    
    #Compute the error drop for RMM1/2 (last point divided by the first point of the oracle)
    errdrop_1 = results[-1,2]/results[0,2]
    errdrop_2 = results[-1,6]/results[0,6]
    
    #return the average error drop for RMM1 and 2
    return 0.5*(errdrop_1 + errdrop_2)
     
def RmseSpread(preds_mu_mean, targets, spread_epistemic, spread_aleatoric, spread_total, T_OUTPUT):
    '''
    Compute 5 bins and their associated RMS Error and RMS Spread following T. N. Palmer (2008).
    Spreads estimates are already unbiased (cf. Predictions function).
    '''

    ensemble_mean_error = (preds_mu_mean - targets).square().detach().numpy()
    
    rmspread = np.zeros((n_bins,6))
    rmse = np.zeros((n_bins,6))
    
    #For each type of spread and for RMM1/2, create 5 equally populated bins and associate each forecast to one of them
    qc0 = pd.qcut(spread_epistemic[:,0], n_bins)
    qc1 = pd.qcut(spread_epistemic[:,1], n_bins)
    qc2 = pd.qcut(spread_aleatoric[:,0], n_bins)
    qc3 = pd.qcut(spread_aleatoric[:,1], n_bins)    
    qc4 = pd.qcut(spread_total[:,0], n_bins)
    qc5 = pd.qcut(spread_total[:,1], n_bins)
        
    bins0, codes0 = qc0.categories, qc0.codes
    bins1, codes1 = qc1.categories, qc1.codes
    bins2, codes2 = qc2.categories, qc2.codes
    bins3, codes3 = qc3.categories, qc3.codes
    bins4, codes4 = qc4.categories, qc4.codes
    bins5, codes5 = qc5.categories, qc5.codes
           
    #For each bin and type of spread, compute the RMS Error and RMS Spread
    for j in range(n_bins):
      idx0 = np.where(codes0 == j)
      idx1 = np.where(codes1 == j)
      idx2 = np.where(codes2 == j)
      idx3 = np.where(codes3 == j)
      idx4 = np.where(codes4 == j)
      idx5 = np.where(codes5 == j)
      
      rmspread[j,0] = np.sqrt((spread_epistemic[idx0, 0]).mean())
      rmspread[j,1] = np.sqrt((spread_epistemic[idx1, 1]).mean())
      rmspread[j,2] = np.sqrt((spread_aleatoric[idx2, 0]).mean())
      rmspread[j,3] = np.sqrt((spread_aleatoric[idx3, 1]).mean())
      rmspread[j,4] = np.sqrt((spread_total[idx4, 0]).mean())
      rmspread[j,5] = np.sqrt((spread_total[idx5, 1]).mean())
          
      rmse[j,0] = np.sqrt(ensemble_mean_error[idx0,0].mean())
      rmse[j,1] = np.sqrt(ensemble_mean_error[idx1,1].mean())
      rmse[j,2] = np.sqrt(ensemble_mean_error[idx2,0].mean())
      rmse[j,3] = np.sqrt(ensemble_mean_error[idx3,1].mean())
      rmse[j,4] = np.sqrt(ensemble_mean_error[idx4,0].mean())
      rmse[j,5] = np.sqrt(ensemble_mean_error[idx5,1].mean())
             
    #Save the results
    np.save(ResultsDir +'rmse_CNN_'+str(T_OUTPUT)+'.npy', rmse)
    np.save(ResultsDir +'rmspread_CNN_'+str(T_OUTPUT)+'.npy', rmspread)    
    
    return

def Predictions():
    '''Compute the forecasts on the test set using the Monte-Carlo Dropout.
        For each lead day (T_OUTPUT), the test set is fed into the network n_members times.
        Each time, some weights are randomly deactivated (seed changed) and forecasts computed with the "new" network.
        An ensemble of n_members forecasts is therefore obtained
    '''
        
    for i, T_OUTPUT in enumerate(T_OUTPUTS):
      print("T_OUTPUT = " +str(T_OUTPUT))
      #Load TEST dataset
      test_dataloader = torch.load(DataDir+'Test_'+str(T_OUTPUT)+'.pt')
      X, y, amp0 = test_dataloader.dataset.X0, test_dataloader.dataset.y_T_OUTPUT.detach(), test_dataloader.dataset.amp0.detach()
      
      model = torch.load(ModelDir + str(T_OUTPUT) + '/model_' + str(T_OUTPUT)+'.pt')
      preds_mu = torch.zeros(len(seeds), y.shape[0], 2)
      preds_cov = torch.zeros(len(seeds),y.shape[0], 2,2)
      
      #n_members times, some weights are deactivated and the inputs are given to the network to obtain a forecasts
      for j, seed in tqdm(enumerate(seeds)):
        torch.manual_seed(seed)
        model.train(True)
        
        mu, cov = model(X)    
        preds_mu[j, :, :] = mu.detach()
        preds_cov[j,:,:,:] = cov.detach()
      
      #preds_mu_mean is the average forecast of the ensemble
      preds_mu_mean = preds_mu.mean(dim=0)
      
      #Compute the epistemic variance (variance of the ensemble means)
      spread_epi = torch.var(preds_mu, dim=0, unbiased = True)
      temp_alea = preds_cov.mean(dim=0)
      #Compute the aleatoric variance (mean of the aleatoric ensemble variances)
      spread_alea = torch.zeros((temp_alea.shape[0],2))
      spread_alea[:,0], spread_alea[:,1] = temp_alea[:,0,0], temp_alea[:,1,1]
      #Compute the total variance (sum of epistemic and aleatoric)
      spread_total = spread_epi + spread_alea
          
      #Save the forecasts    
      torch.save(preds_mu, PredsDir + 'mu_ens_'+str(T_OUTPUT) +'.pt')
      torch.save(preds_mu_mean, PredsDir + 'mu_' + str(T_OUTPUT) + '.pt')
      torch.save(spread_epi, PredsDir + 'cov_epi_' + str(T_OUTPUT) + '.pt')
      torch.save(spread_alea, PredsDir + 'cov_alea_' + str(T_OUTPUT) + '.pt')
      torch.save(spread_total, PredsDir + 'cov_' + str(T_OUTPUT) + '.pt')
      torch.save(y, PredsDir + 'targets_' + str(T_OUTPUT) + '.pt')
      torch.save(amp0, PredsDir + 'amp0_' + str(T_OUTPUT) + '.pt')   


def Predictions_Train():
    '''Same function as Predictions, with MC-Dropout, but applied on the TRAIN dataset'''
    
    for i, T_OUTPUT in enumerate(T_OUTPUTS):
      print("T_OUTPUT = " +str(T_OUTPUT))
      #Load TRAIN dataset      
      test_dataloader = torch.load(DataDir+'Train_'+str(T_OUTPUT)+'.pt')
      X, y, amp0 = test_dataloader.dataset.X0, test_dataloader.dataset.y_T_OUTPUT.detach(), test_dataloader.dataset.amp0.detach()
      
      model = torch.load(ModelDir + str(T_OUTPUT) + '/model_' + str(T_OUTPUT)+'.pt')
      preds_mu = torch.zeros(len(seeds), y.shape[0], 2)
      preds_cov = torch.zeros(len(seeds),y.shape[0], 2,2)
      
      #n_members times, some weights are deactivated and the inputs are given to the network to obtain a forecasts
      for j, seed in tqdm(enumerate(seeds)):
        torch.manual_seed(seed)
        model.train(True)
        
        mu, cov = torch.zeros((y.shape[0], 2)), torch.zeros((y.shape[0], 2, 2))
        
        #The TRAIN dataset is fed 2000 samples at a time to avoid crashes
        for k in tqdm(range(y.shape[0]//2000)):
            temp1, temp2 = model(X[2000*k:2000*(k+1)])
            mu[2000*k:2000*(k+1)], cov[2000*k:2000*(k+1)] = temp1.detach(), temp2.detach()
            
        mu[2000*(y.shape[0]//2000):], cov[2000*(y.shape[0]//2000):] = model(X[2000*(y.shape[0]//2000):])    

        preds_mu[j, :, :] = mu.detach()
        preds_cov[j,:,:,:] = cov.detach()
      
      #preds_mu_mean is the average forecast of the ensemble
      preds_mu_mean = preds_mu.mean(dim=0)
      
      #Compute the epistemic variance (variance of the ensemble means)
      spread_epi = torch.var(preds_mu, dim=0, unbiased = True)
      temp_alea = preds_cov.mean(dim=0)
      
      #Compute the aleatoric variance (mean of the aleatoric ensemble variances)
      spread_alea = torch.zeros((temp_alea.shape[0],2))
      spread_alea[:,0], spread_alea[:,1] = temp_alea[:,0,0], temp_alea[:,1,1]
      #Compute the total variance (sum of epistemic and aleatoric)
      spread_total = spread_epi + spread_alea 
      
      #Save the forecasts    
      torch.save(preds_mu, PredsDir + 'mu_ens_train_'+str(T_OUTPUT) +'.pt')
      torch.save(preds_mu_mean, PredsDir + 'mu_train_' + str(T_OUTPUT) + '.pt')
      torch.save(spread_epi, PredsDir + 'cov_epi_train_' + str(T_OUTPUT) + '.pt')
      torch.save(spread_alea, PredsDir + 'cov_alea_train_' + str(T_OUTPUT) + '.pt')
      torch.save(spread_total, PredsDir + 'cov_train_' + str(T_OUTPUT) + '.pt')
      torch.save(y, PredsDir + 'targets_train_' + str(T_OUTPUT) + '.pt')
      torch.save(amp0, PredsDir + 'amp0_train_' + str(T_OUTPUT) + '.pt')


def main():
    '''
    For each lead time, compute the log-score, the RMSE/Spread bins, and the Error-Drop
    '''
    results = torch.zeros((len(T_OUTPUTS), 7))
    
    for i, T_OUTPUT in tqdm(enumerate(T_OUTPUTS)):
        
        #Load the forecasts obtained with "Predictions" or "Predictions_Train"
        mu = torch.load(PredsDir + 'mu_'+ str(T_OUTPUT)+'.pt').detach()
        mu_ens = torch.load(PredsDir + 'mu_ens_'+ str(T_OUTPUT)+'.pt').detach()
        spread_epistemic = torch.load(PredsDir + 'cov_epi_'+ str(T_OUTPUT)+'.pt').detach()
        spread_aleatoric = torch.load(PredsDir + 'cov_alea_'+ str(T_OUTPUT)+'.pt').detach()        
        spread_total = torch.load(PredsDir + 'cov_' + str(T_OUTPUT) + '.pt').detach()
        targets = torch.load(PredsDir + 'targets_' + str(T_OUTPUT) + '.pt').detach()
        amp0 = torch.load(PredsDir + 'amp0_' + str(T_OUTPUT) + '.pt').detach()
        
        #Only events with initial amplitude greater than 1.0
        idx = np.where(amp0>=1.0)[0]
        mu, mu_ens, spread_epistemic, spread_aleatoric, spread_total, targets, amp0 = mu[idx,:], mu_ens[:,idx,:], spread_epistemic[idx,:], spread_aleatoric[idx,:], spread_total[idx,:], targets[idx,:], amp0[idx]

        #Put the predicted spread in a covariance matrix for the log-score computation
        cov_matrix = torch.zeros(spread_total.shape[0],2,2)
        cov_matrix[:,0,0], cov_matrix[:,1,1] = spread_total[:,0], spread_total[:,1]
         
        spread_epistemic = spread_epistemic.numpy()
        spread_aleatoric = spread_aleatoric.numpy()
        spread_total = spread_total.numpy()
        
        #Compute the metrics and the bins  
        results[i,0] = RMSE(mu, targets)
        results[i,1] = BivariateCorrelation(mu, targets)
        results[i,2] = AmpError(mu, targets)
        results[i,3] = PhaseError(mu, targets)
        results[i,4] = CRPS(mu, cov_matrix, targets)
        results[i,5] = LogScore(mu, cov_matrix, targets.type(torch.FloatTensor))
        results[i,6] = ErrorDrop(mu, targets, spread_epistemic, spread_aleatoric, spread_total, T_OUTPUT)
        RmseSpread(mu, targets, spread_epistemic, spread_aleatoric, spread_total, T_OUTPUT)
 
    #Save the results in a dataframe            
    df = pd.DataFrame(results.numpy())
    df.columns = ['RMSE', 'Bivariate Correlation','Amplitude Error', 'Phase Error', 'CRPS', 'Log Score', 'Error Drop']
    df.to_csv(ResultsDir + 'Results_CNN.txt', index = False)
    print(df.head())
    
    return
    
if __name__ == '__main__':
    Predictions_Train()
    Predictions()
    main()