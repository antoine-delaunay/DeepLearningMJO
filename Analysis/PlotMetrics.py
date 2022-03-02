import torch
import torch.nn as nn
import numpy as np
import matplotlib.font_manager
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
rc('text', usetex=True)

mpl.rcParams['font.size'] = 9
mpl.rcParams['mathtext.default'] = 'regular'

ResultsDir = '/network/aopp/preds0/pred/users/delaunay/Analysis/Performance/Results/'
PlotsDir = '/network/aopp/preds0/pred/users/delaunay/Analysis/Performance/Plots/'
models = ['ECMWF', 'HMCR','CNRM', 'BOM', 'CNN']

RMSE, BC, AmpError, PhaseError, CRPS, LogScore, ErrorDrop = [], [], [], [], [], [], []
time = [1, 3, 5, 10, 15, 20, 25, 30, 35]

#Get for each model the value of the metrics at different lead times
for i, model in enumerate(models):
    df = pd.read_csv(ResultsDir +'Results_'+model+'.txt')
    RMSE.append(df['RMSE'].to_numpy())
    BC.append(df['Bivariate Correlation'].to_numpy())
    AmpError.append(df['Amplitude Error'].to_numpy())
    PhaseError.append(df['Phase Error'].to_numpy())
    CRPS.append(df['CRPS'].to_numpy())
    LogScore.append(df['Log Score'].to_numpy())
    ErrorDrop.append(df['Error Drop'].to_numpy())
    
#Plot metrics except Bivariate Correlation
fig,ax = plt.subplots(ncols=3,nrows=2,figsize=(10.8, 6), constrained_layout=True)

ax[0,0].plot(time, RMSE[0], color = 'blue', marker = 'x', markersize = 4.4)
ax[0,0].plot(time, RMSE[1], color = 'olive', marker = '^', markersize = 4.4)
ax[0,0].plot(time, RMSE[2], color = 'green', marker = 'o', markersize = 4.4)
ax[0,0].plot(time, RMSE[3], color = 'red' , marker = 'v',  markersize = 4.4)
ax[0,0].plot(time, RMSE[4], color = 'black', marker = '+', markersize = 4.4)

ax[0,1].plot(time, AmpError[0], color = 'blue', marker = 'x', markersize = 4.4)
ax[0,1].plot(time, AmpError[1], color = 'olive', marker = '^', markersize = 4.4)
ax[0,1].plot(time, AmpError[2], color = 'green', marker = 'o', markersize = 4.4)
ax[0,1].plot(time, AmpError[3], color = 'red' , marker = 'v',  markersize = 4.4)
ax[0,1].plot(time, AmpError[4], color = 'black', marker = '+', markersize = 4.4)

ax[0,2].plot(time, PhaseError[0], color = 'blue', marker = 'x', markersize = 4.4)
ax[0,2].plot(time, PhaseError[1], color = 'olive', marker = '^', markersize = 4.4)
ax[0,2].plot(time, PhaseError[2], color = 'green', marker = 'o', markersize = 4.4)
ax[0,2].plot(time, PhaseError[3], color = 'red' , marker = 'v',  markersize = 4.4)
ax[0,2].plot(time, PhaseError[4], color = 'black', marker = '+', markersize = 4.4)

ax[1,0].plot(time, CRPS[0], color = 'blue', marker = 'x', markersize = 4.4)
ax[1,0].plot(time, CRPS[1], color = 'olive', marker = '^', markersize = 4.4)
ax[1,0].plot(time, CRPS[2], color = 'green', marker = 'o', markersize = 4.4)
ax[1,0].plot(time, CRPS[3], color = 'red' , marker = 'v',  markersize = 4.4)
ax[1,0].plot(time, CRPS[4], color = 'black', marker = '+', markersize = 4.4)

ax[1,1].plot(time, LogScore[0], color = 'blue', marker = 'x', markersize = 4.4)
ax[1,1].plot(time, LogScore[1], color = 'olive', marker = '^', markersize = 4.4)
ax[1,1].plot(time, LogScore[2], color = 'green', marker = 'o', markersize = 4.4)
ax[1,1].plot(time, LogScore[3], color = 'red' , marker = 'v',  markersize = 4.4)
ax[1,1].plot(time, LogScore[4], color = 'black', marker = '+', markersize = 4.4)
ax[1,1].set_ylim(0,18)

ax[1,2].plot(time, ErrorDrop[0], color = 'blue', marker = 'x', markersize = 4.4)
ax[1,2].plot(time, ErrorDrop[1], color = 'olive', marker = '^', markersize = 4.4)
ax[1,2].plot(time, ErrorDrop[2], color = 'green', marker = 'o', markersize = 4.4)
ax[1,2].plot(time, ErrorDrop[3], color = 'red' , marker = 'v', markersize = 4.4)
ax[1,2].plot(time, ErrorDrop[4], color = 'black', marker = '+', markersize = 4.4)

ax[0,0].legend(models, fontsize=8)
ax[0,1].legend(models, fontsize=8)
ax[0,2].legend(models, fontsize=8)
ax[1,0].legend(models, fontsize=8)
ax[1,1].legend(models, fontsize=8)
ax[1,2].legend(models, fontsize=8)

ax[0,0].set_xlabel('Lead time (days)', fontsize=12)
ax[0,1].set_xlabel('Lead time (days)', fontsize=12)
ax[0,2].set_xlabel('Lead time (days)', fontsize=12)
ax[1,0].set_xlabel('Lead time (days)', fontsize=12)
ax[1,1].set_xlabel('Lead time (days)', fontsize=12)
ax[1,2].set_xlabel('Lead time (days)', fontsize=12)

ax[0,0].set_ylabel('RMSE', fontsize=12)
ax[0,1].set_ylabel('Amplitude Error', fontsize=12)
ax[0,2].set_ylabel('Phase Error', fontsize=12)
ax[1,0].set_ylabel('CRPS', fontsize=12)
ax[1,1].set_ylabel('Ignorance-score', fontsize=12)
ax[1,2].set_ylabel('Error Drop', fontsize=12)

ax[0,0].set_title(r"\textbf{a. }")
ax[0,1].set_title(r"\textbf{b. }")
ax[0,2].set_title(r"\textbf{c. }")
ax[1,0].set_title(r"\textbf{d. }")
ax[1,1].set_title(r"\textbf{e. }")
ax[1,2].set_title(r"\textbf{f. }")

plt.savefig(PlotsDir +'Metrics.png', bbox_inches='tight')
plt.close()

#Plot Bivariate Correlation
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(6,4), constrained_layout=True)

ax.plot(time, BC[0], color = 'blue', marker = 'x', markersize = 4.4)
ax.plot(time, BC[1], color = 'olive', marker = '^', markersize = 4.4)
ax.plot(time, BC[2], color = 'green', marker = 'o', markersize = 4.4)
ax.plot(time, BC[3], color = 'red' , marker = 'v',  markersize = 4.4)
ax.plot(time, BC[4], color = 'black', marker = '+', markersize = 4.4)

ax.legend(models, fontsize=8)
ax.set_xlabel('Lead time (days)', fontsize=12)
ax.set_ylabel('Bivariate Correlation', fontsize=12)

plt.savefig(PlotsDir +'BC.png', bbox_inches='tight')
plt.close()
