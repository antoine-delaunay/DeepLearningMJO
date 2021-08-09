import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

models = ['ECMWF', 'HMCR', 'CNRM', 'BOM']
model_markers = ['x','^','o','v']
model_colors = ['blue', 'olive', 'green', 'red']

T_OUTPUT = 10
PlotsDir = '/network/aopp/preds0/pred/users/delaunay/Analysis/Performance/Plots/'
ResultsDir = '/network/aopp/preds0/pred/users/delaunay/Analysis/Performance/Results/'

fig,ax = plt.subplots(ncols=2,nrows=1,figsize=(12, 5), constrained_layout=True)

#For each dynamical model, get the values of the rms spread and rms error of each bin and plot them
for i, model in enumerate(models):
    rmspread = np.load(ResultsDir +'rmspread_'+model+'_'+str(T_OUTPUT)+'.npy')
    rmse = np.load(ResultsDir +'rmse_'+model+'_'+str(T_OUTPUT)+'.npy')
    ax[0].scatter(rmspread[np.isfinite(rmspread[:,0]),0], rmse[np.isfinite(rmspread[:,0]),0], s=30, marker = model_markers[i], color = model_colors[i])
    ax[1].scatter(rmspread[np.isfinite(rmspread[:,1]),1], rmse[np.isfinite(rmspread[:,1]),1], s=30, marker = model_markers[i], color = model_colors[i])

#Do the same for the CNN (aleatoric and total spread)
rmspread = np.load(ResultsDir + 'rmspread_CNN_'+str(T_OUTPUT)+'.npy')
rmse = np.load(ResultsDir + 'rmse_CNN_'+str(T_OUTPUT)+'.npy')

ax[0].scatter(rmspread[np.isfinite(rmspread[:,2]),2], rmse[np.isfinite(rmspread[:,2]),2], s=30, marker = 's', color = 'darkviolet')
ax[0].scatter(rmspread[np.isfinite(rmspread[:,4]),4], rmse[np.isfinite(rmspread[:,4]),4], s=30, marker = '+', color = 'black')

ax[1].scatter(rmspread[np.isfinite(rmspread[:,3]),3], rmse[np.isfinite(rmspread[:,3]),3], s=30, marker = 's', color = 'darkviolet')
ax[1].scatter(rmspread[np.isfinite(rmspread[:,5]),5], rmse[np.isfinite(rmspread[:,5]),5], s=30, marker = '+', color = 'black')

ax[0].legend(['ECMWF', 'HMCR', 'CNRM', 'BOM','CNN - aleatoric', 'CNN - total'])
ax[0].plot(np.linspace(0.0,1.2,1000), np.linspace(0.0,1.2,1000), color='black', linestyle='--')
ax[0].set_xlim((0.0,1.15))
ax[0].set_ylim((0.0,1.15))
ax[0].set_xlabel('RMS Spread')
ax[0].set_ylabel('RMS Error')
ax[0].set_title(r'\textbf{a.} RMM1')#)RMS Error / Spread diagram - RMM1 - '+ str(T_OUTPUT) + ' days forecast')

ax[1].legend(['ECMWF', 'HMCR', 'CNRM', 'BOM','CNN - aleatoric', 'CNN - total'])
ax[1].plot(np.linspace(0.0,1.2,1000), np.linspace(0.0,1.2,1000), color='black', linestyle='--')
ax[1].set_xlim((0.0,1.15))
ax[1].set_ylim((0.0,1.15))
ax[1].set_xlabel('RMS Spread')
ax[1].set_ylabel('RMS Error')
ax[1].set_title(r'\textbf{b.} RMM2')#)RMS Error / Spread diagram - RMM1 - '+ str(T_OUTPUT) + ' days forecast')

fig.savefig(PlotsDir+'rmspread_error_'+str(T_OUTPUT)+'.png', bbox_inches='tight')
plt.close()