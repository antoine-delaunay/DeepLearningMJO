import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from tqdm import tqdm
from Model import MJONet
from Dataset import MJODataset
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt

features = ['3','sst', 'shum', 'hgt', 'dlwrf', 'full']
features_names_plot = ['Standard', 'Standard\n + SST', 'Standard\n + SHUM400', "Standard\n + " + r'$\Phi$850', 'Standard\n + DLR', 'All\n features']
T_OUTPUT = 10

DataDir = '/network/aopp/preds0/pred/users/delaunay/Data/Datasets/'
ModelDir = '/network/aopp/preds0/pred/users/delaunay/ModelSaves/Lag' 
PlotsDir = '/network/aopp/preds0/pred/users/delaunay/Analysis/Performance/Plots/'
ResultsDir = '/network/aopp/preds0/pred/users/delaunay/Analysis/Performance/Results/'

seeds = [20*i for i in range(10)]
n_members = len(seeds)
results = torch.zeros(len(features))

def LogScore(mu, cov, targets):
    dist = MultivariateNormal(mu, covariance_matrix = cov)
    loss = -(dist.log_prob(targets)).mean()
    return loss
 
#Compute the forecasts and the loss for the test sets on all the different combinations of variables
 
for i, feature in enumerate(features):
    print(feature)
    test_dataloader = torch.load(DataDir+'Test_'+feature + '_'+ str(T_OUTPUT)+'.pt')
    X, y, amp0, y_T_OUTPUT = test_dataloader.dataset.X0, test_dataloader.dataset.y_T_OUTPUT.detach(), test_dataloader.dataset.amp0.detach(), test_dataloader.dataset.y_T_OUTPUT.detach()
      
    model = torch.load(ModelDir + str(T_OUTPUT) + '/model_'+ feature + '_' + str(T_OUTPUT)+'.pt')
    preds_mu = torch.zeros(len(seeds), y.shape[0], 2)
    preds_cov = torch.zeros(len(seeds),y.shape[0], 2,2)
    
    #MC-Dropout
    for j, seed in tqdm(enumerate(seeds)):
        torch.manual_seed(seed)
        model.train(True)
        
        mu, cov = model(X)    
        preds_mu[j, :, :] = mu.detach()
        preds_cov[j,:,:,:] = cov.detach()
    
    #Average the ensemble forecasts + compute the total spread
    preds_mu_mean = preds_mu.mean(dim=0)
    spread_epi = torch.var(preds_mu, dim=0, unbiased = True)
    temp_alea = preds_cov.mean(dim=0)
    spread_alea = torch.zeros((temp_alea.shape[0],2))
    spread_alea[:,0], spread_alea[:,1] = temp_alea[:,0,0], temp_alea[:,1,1]
    spread_total = spread_epi + spread_alea

    cov_matrix = torch.zeros(spread_total.shape[0],2,2)
    cov_matrix[:,0,0], cov_matrix[:,1,1] = spread_total[:,0], spread_total[:,1]
    idx = np.where(amp0>=1.0)[0]
    preds_mu_mean, cov_matrix, y_T_OUTPUT = preds_mu_mean[idx], cov_matrix[idx], y_T_OUTPUT[idx]
    
    #Compute the log-score for this combination of features
    results[i] = LogScore(preds_mu_mean, cov_matrix.to(torch.float64), y_T_OUTPUT)
    
torch.save(results, ResultsDir + 'CompareFeatures_' + str(T_OUTPUT) + '.pt')

#Plot the results
plt.figure()
plt.barh(features_names_plot, results)
plt.xlim(2.0,2.4)
plt.xlabel('Log-Score')
plt.ylabel('Features')
plt.title('Comparison of the features')
plt.tight_layout(pad=0.4)
plt.savefig(PlotsDir + 'CompareFeatures_'+str(T_OUTPUT)+'.png')