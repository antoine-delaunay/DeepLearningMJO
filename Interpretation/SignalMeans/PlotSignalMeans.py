import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import sys
sys.path.append('../../CNN/')
from Dataset import MJODataset

T_OUTPUT = 10
lon = np.linspace(-180, 180, 144)
lat = np.linspace(-20, 20, 17)
lon2d, lat2d = np.meshgrid(lon, lat)

levels_means = [0.0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64]

DataDir = '/network/aopp/preds0/pred/users/delaunay/Data/'
PlotDir =  '/network/aopp/preds0/pred/users/delaunay/Analysis/Interpretation/Plots/Lag' + str(T_OUTPUT)+'/SignalMeans/'

#LOADING DATA
print('Loading Data')

dataloader = torch.load(DataDir + 'Datasets/Test_'+str(T_OUTPUT) + '.pt')
amp0 = dataloader.dataset.amp0.detach().numpy()
phase0 = dataloader.dataset.phase0.detach().numpy()

S1 = torch.load(DataDir + 'Forecasts/CNN/Signals/signal1_test_' + str(T_OUTPUT) + '.pt').detach().numpy()
S2 = torch.load(DataDir + 'Forecasts/CNN/Signals/signal2_test_' + str(T_OUTPUT) + '.pt').detach().numpy()

S1 = abs(S1)
S2 = abs(S2)

for i in range(7):
    S1[:,i,:,:] = (S1[:,i,:,:] - S1[:,i,:,:].min())/(S1[:,i,:,:].max() - S1[:,i,:,:].min())
    S2[:,i,:,:] = (S2[:,i,:,:] - S2[:,i,:,:].min())/(S2[:,i,:,:].max() - S2[:,i,:,:].min())

SM1, SM2 = np.zeros((8,17,144)), np.zeros((8,17,144))

#COMPUTING SIGNAL MEANS
print('Computing Signal Means')

for i in range(8):
    idx = np.where((phase0 ==i+1) & (amp0>=1.0))[0]
    #Averagint over the days and then over the features (order does not matter)
    temp1 = S1[idx,:,:,:].mean(axis=0)
    temp2 = S2[idx,:,:,:].mean(axis=0)

    SM1[i,:,:] = temp1.mean(axis=0)
    SM2[i,:,:] = temp2.mean(axis=0)
    
#PLOTTING SIGNAL MEANS
print('Plotting Signal Means')
fig,ax = plt.subplots(ncols=2,nrows=8,figsize=(24,16),
                        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}, constrained_layout=True)
  
idx0 = np.unravel_index(SM1[:,:,:].argmax(), SM1[:,:,:].shape)[0]
idx1 = np.unravel_index(SM2[:,:,:].argmax(), SM2[:,:,:].shape)[0]
  
for i in range(8):
    ax[i,0].coastlines()
    ax[i,1].coastlines()
    im0 = ax[i,0].contourf(lon2d, lat2d, SM1[i,:,:], cmap='Oranges', levels=levels_means)
    im1 = ax[i,1].contourf(lon2d, lat2d, SM2[i,:,:], cmap='Oranges', levels=levels_means)

    ax[i,0].set_title('Signal RMM1 - Phase ' + str(int(i+1)), fontsize=15)
    ax[i,1].set_title('Signal RMM2 - Phase ' + str(int(i+1)), fontsize=15)
    
    ax[i,0].set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax[i,0].set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax[i,0].tick_params(bottom=False)
    
    ax[i,1].set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax[i,1].set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax[i,1].tick_params(bottom=False)
      
    if i == idx0:
      temp0 = im0
    if i == idx1:
      temp1 = im1
  
fig.colorbar(temp1, ax=ax[7,:],shrink = 0.6, pad = 0.01, orientation='horizontal')
  
fig.suptitle('' , fontsize=10)
fig.savefig(PlotDir + 'SM_' + str(T_OUTPUT) + '.png')
plt.close()