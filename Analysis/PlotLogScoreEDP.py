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

mpl.rcParams['font.size'] = 7
mpl.rcParams['mathtext.default'] = 'regular'

ResultsDir = '/network/aopp/preds0/pred/users/delaunay/Analysis/Performance/Results/'
PlotsDir = '/network/aopp/preds0/pred/users/delaunay/Analysis/Performance/Plots/'
models = ['ECMWF', 'HMCR','CNRM', 'BOM', 'CNN']

LogScore, ErrorDrop = [], []
time = [1, 3, 5, 10, 15, 20, 25, 30, 35]

#Get for each model the value of the log-score and error drop at different lead times
for i, model in enumerate(models):
    df = pd.read_csv(ResultsDir +'Results_'+model+'.txt')
    LogScore.append(df['Log Score'].to_numpy())
    ErrorDrop.append(df['Error Drop'].to_numpy())
    
#Plot 
fig,ax = plt.subplots(ncols=2,nrows=1,figsize=(7.2, 3), constrained_layout=True)
ax[0].plot(time, LogScore[0], color = 'blue', marker = 'x', markersize = 4.4)
ax[0].plot(time, LogScore[1], color = 'olive', marker = '^', markersize = 4.4)
ax[0].plot(time, LogScore[2], color = 'green', marker = 'o', markersize = 4.4)
ax[0].plot(time, LogScore[3], color = 'red' , marker = 'v',  markersize = 4.4)
ax[0].plot(time, LogScore[4], color = 'black', marker = '+', markersize = 4.4)
ax[0].set_ylim(0,18)

ax[1].plot(time, ErrorDrop[0], color = 'blue', marker = 'x', markersize = 4.4)
ax[1].plot(time, ErrorDrop[1], color = 'olive', marker = '^', markersize = 4.4)
ax[1].plot(time, ErrorDrop[2], color = 'green', marker = 'o', markersize = 4.4)
ax[1].plot(time, ErrorDrop[3], color = 'red' , marker = 'v', markersize = 4.4)
ax[1].plot(time, ErrorDrop[4], color = 'black', marker = '+', markersize = 4.4)

ax[0].legend(models)
ax[1].legend(models)
ax[0].set_xlabel('Lead time (days)')
ax[1].set_xlabel('Lead time (days)')
ax[0].set_ylabel('Log-score')
ax[1].set_ylabel('Error Drop')
ax[0].set_title(r"\textbf{a. } Log-score")
ax[1].set_title(r"\textbf{b. } Error drop")

plt.savefig(PlotsDir +'LogScore_EDP.png', bbox_inches='tight')
plt.close()
