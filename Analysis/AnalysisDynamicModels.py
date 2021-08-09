import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

'''
This file compute the Log-score and Error-drop for each lead time for a given dynamical model.
The model's name needs to be specified at the beginning. 
A specific file in the same folder does the same job for the CNN.
'''
model = 'BOM' #BOM, CNRM, ECMWF, HMCR
FilesDir = '/network/aopp/preds0/pred/users/delaunay/Data/Forecasts/' + model +'/EnsMembers/'
PlotsDir = '/network/aopp/preds0/pred/users/delaunay/Analysis/Performance/Plots/'
ResultsDir = '/network/aopp/preds0/pred/users/delaunay/Analysis/Performance/Results/'
T_OUTPUTS = [1,3,5,10,15,20,25,30,35]

def LogScore(mu, cov, targets):
    '''Compute the Log-score'''
    dist = MultivariateNormal(mu, covariance_matrix = cov)
    loss = -(dist.log_prob(targets)).mean()
    return np.array(loss)
    
def ErrorDrop(preds_mean, targets, spread_epistemic, T_OUTPUT):
    '''Compute the Error Drop. First sort the data according to the predicted uncertainty / rmse.
    Then remove the alpha percent most uncertain data gradually and compute the rmse of the 1-alpha percent remaining meanwhile.
    For each RMM, the error drop is the ratio of the rmse between the 5% most certain forecasts and all the forecasts' rmse.
    '''
    
    spread_epistemic = torch.from_numpy(spread_epistemic)  
    mse = (preds_mean - targets).square()
    
    #Sorting uncertainties
    #"spread" arrays = arrays of forecasted uncertainty in the chronological order
    #"uncertainty" arrays = arrays of forecasted uncertainty according to the sorting order
    
    #Sorting forecasts according to RMM1/2 predicted spread
    uncertainty_epi = torch.zeros(preds_mean.shape[0], 4)
    sort_epi = torch.sort(spread_epistemic[:,0], descending=True)
    uncertainty_epi[:,0], uncertainty_epi[:,1] = sort_epi[0], (preds_mean[sort_epi[1], 0] - targets[sort_epi[1], 0]).square()
    sort_epi = torch.sort(spread_epistemic[:,1], descending=True)
    uncertainty_epi[:,2], uncertainty_epi[:,3] = sort_epi[0], (preds_mean[sort_epi[1], 1] - targets[sort_epi[1], 1]).square()
  
    #Sorting forecasts according to RMM1/2 mse
    uncertainty_mse = torch.zeros(preds_mean.shape[0], 4)
    sort_mse = torch.sort(mse[:,0], descending = True)  
    uncertainty_mse[:,0], uncertainty_mse[:,1] = sort_mse[0], mse[sort_mse[1],0]
    sort_mse = torch.sort(mse[:,1], descending = True)  
    uncertainty_mse[:,2], uncertainty_mse[:,3] = sort_mse[0], mse[sort_mse[1],1]
    
    #For each alpha from 0 to 0.95, compute the RMSE of the remaining forecasts 1- alpha forecasts.
    results = torch.zeros(20,4)
    alphas = [i*0.05 for i in range(20)]
    for i, alpha in enumerate(alphas):
        idx = int(preds_mean.shape[0]*alpha)
        results[i,0], results[i,1] = uncertainty_epi[idx:,1].mean(dim=0).sqrt(), uncertainty_epi[idx:,3].mean(dim=0).sqrt()
        results[i,2], results[i,3] = uncertainty_mse[idx:,1].mean(dim=0).sqrt(), uncertainty_mse[idx:,3].mean(dim=0).sqrt()
    
    #Compute the error drop for RMM1/2 (last point divided by the first point of the oracle)
        
    errdrop_1 = results[-1,0]/results[0,0]
    errdrop_2 = results[-1,1]/results[0,1]
    
    #return the average error drop for RMM1 and 2
    return 0.5*(errdrop_1 + errdrop_2)
  
def RmseSpread(preds_mean, targets, spread_epistemic, T_OUTPUT, n_members):
    '''
    Compute 5 bins and their associated RMS Error and RMS Spread following T. N. Palmer (2008).
    spread estimates are already unbiased (cf. Predictions function).
    '''
    ensemble_mean_error = (preds_mean - targets).square().detach().numpy()
    
    #For RMM1/2, create 5 equally populated bins and associate each forecast to one of them    
    qc0 = pd.qcut(spread_epistemic[:,0], 5)
    qc1 = pd.qcut(spread_epistemic[:,1], 5)
        
    bins0, codes0 = qc0.categories, qc0.codes
    bins1, codes1 = qc1.categories, qc1.codes
        
    rmspread = np.zeros((5,2))
    rmse = np.zeros((5,2))

    #For each bin, compute the RMS Error and RMS Spread    
    for j in range(5):
      idx0 = np.where(codes0 == j)
      idx1 = np.where(codes1 == j)
      rmspread[j,0] = np.sqrt((spread_epistemic[idx0, 0]).mean())
      rmspread[j,1] = np.sqrt((spread_epistemic[idx1, 1]).mean())
          
      rmse[j,0] = np.sqrt(ensemble_mean_error[idx0,0].mean())
      rmse[j,1] = np.sqrt(ensemble_mean_error[idx1,1].mean())

    #Save the results        
    np.save(ResultsDir + 'rmse_'+model+'_'+str(T_OUTPUT)+'.npy', rmse)
    np.save(ResultsDir + 'rmspread_'+model+'_'+str(T_OUTPUT)+'.npy', rmspread)
      
    return

def main():
    '''
    For each lead time, compute the log-score, the RMSE/Spread bins, and the Error-Drop
    '''
    
    print(model)
    results = np.zeros((len(T_OUTPUTS), 2))
    for i,T_OUTPUT in tqdm(enumerate(T_OUTPUTS)):
        df = pd.read_csv(FilesDir+model+str(T_OUTPUT)+'.txt')
        n_members = int((df.shape[1]-5)/2)
        
        #Only events with initial amplitude greater than 1.0
        df['Amp0'] = np.sqrt((df['RMM1_0']**2+df['RMM2_0']**2))
        df = df.loc[df['Amp0']>=1.0]
        preds = torch.zeros((n_members, df.shape[0], 2))
     
        for j in range(df.shape[0]):
          temp = df.iloc[j, 5:-1].to_numpy()
          
          for k in range(n_members):
            preds[k,j,0], preds[k,j,1] = temp[2*k], temp[2*k+1]
          
        preds_mean = preds.mean(dim=0)
        targets = torch.from_numpy(df[['RMM1_obs', 'RMM2_obs']].to_numpy())
        
        #Compute the spread from the ensemble members and put it into a covariance matrix
        spread_epistemic = torch.var(preds, unbiased=True, dim=0).detach().numpy() 
        
        cov = torch.zeros(targets.shape[0],2,2)
        cov[:,0,0], cov[:,1,1] = torch.from_numpy(spread_epistemic[:,0]), torch.from_numpy(spread_epistemic[:,1])
        cov = cov.type(torch.DoubleTensor)
          
        #Compute the Log-score, the Error Drop, and the bins  
        results[i,0] = LogScore(preds_mean, cov, targets)
        results[i,1] = ErrorDrop(preds_mean, targets, spread_epistemic, T_OUTPUT)
        RmseSpread(preds_mean, targets, spread_epistemic, T_OUTPUT, n_members)
        
    #Save the results in a dataframe       
    df = pd.DataFrame(data=results)
    df.columns =['Log Score', 'Error Drop']
    df.to_csv(ResultsDir+'Results_'+model+'.txt', index=False)
    print(df.head())
    
    return

if __name__ == '__main__':
    main()