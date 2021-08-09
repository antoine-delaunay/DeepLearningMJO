import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib import rc
rc('text', usetex=True)
import cartopy.crs as ccrs
from tqdm import tqdm

import sys
sys.path.append('../../CNN/')
from Dataset import MJODataset

#mpl.rcParams['font.size'] = 7
mpl.rcParams['mathtext.default'] = 'regular'

PLOT_DIAGRAM = True
T_OUTPUT = 10
idx1 = 9797
idx2 = 132

n_days = 10
levels_signal = np.linspace(0.0,0.6,10)
levels_means = [np.linspace(-36,36,10), np.linspace(-13,13,10), np.linspace(-70,70,10), np.linspace(-3.4,3.4,10), 
                np.linspace(0,0.0026,10), np.linspace(13577,15434,10), np.linspace(1e+6,1.59e+6,10)]

DataDir = '/network/aopp/preds0/pred/users/delaunay/Data/'
PlotDir =  '/network/aopp/preds0/pred/users/delaunay/Analysis/Interpretation/Plots/Lag' + str(T_OUTPUT)+'/MaritimeContinent/'

features = ['UA200', 'UA850', 'OLR', 'SST', 'SHUM400', 'Z850', 'DLR']
units = [r'UA200 - $m.s^{-1}$', r'UA850 - $m.s^{-1}$', r'OLR - $W.m^{-2}$', r'SST - °K', r'SHUM400', r'Z850 - $m^{2}.s^{-2}$', r'DLR - $J.m^{-2}$']

lon = np.linspace(-180, 180, 144)
lat = np.linspace(-20, 20, 17)
lon2d, lat2d = np.meshgrid(lon, lat)

def PlotMaps(XM1, XM2, SM1_1, SM1_2, SM2_1, SM2_2, phase, feature_idx, levels_means, levels_signal):
    '''
    Plot composite and signal maps associated with the forecasts of all days starting in the given phase
    '''
    idx_max1 = np.unravel_index(XM1[:,feature_idx,:,:].argmax(), XM1.shape)[0]
    idx_max2 = np.unravel_index(XM2[:,feature_idx,:,:].argmax(), XM2.shape)[0]

    feature = features[feature_idx]
    
    if feature_idx<=3:
        cmap_mean = 'bwr'
    else:
        cmap_mean = 'RdYlBu_r'

    fig = plt.figure(figsize=(12,5), constrained_layout=True)
    gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[1, 0.1, 1, 1, 0.1])
    
    ax00 = fig.add_subplot(gs[0, 0], projection= ccrs.PlateCarree(central_longitude=180)) 
    ax00.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax00.set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax00.tick_params(bottom=False)
    ax00.coastlines()
    ax00.set_title(r"\textbf{a. } Decaying event", loc = 'left')
    im00 = ax00.contourf(lon2d, lat2d, XM1[phase, feature_idx,:,:], cmap=cmap_mean, levels = levels_means[feature_idx], extend ='both')
    
    ax01 = fig.add_subplot(gs[0, 1], projection= ccrs.PlateCarree(central_longitude=180)) 
    ax01.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax01.set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax01.tick_params(bottom=False)
    ax01.coastlines()
    ax01.set_title(r"\textbf{b. } Propagating event", loc = 'left')
    im01 = ax01.contourf(lon2d, lat2d, XM2[phase, feature_idx,:,:], cmap=cmap_mean, levels = levels_means[feature_idx], extend ='both')
    
    ax1 = fig.add_subplot(gs[1, :], visible=False)
    cb1 = fig.colorbar(im01, ax = ax1, shrink = 0.5, pad = -0.2, orientation = 'horizontal')
    cb1.ax.set_title(units[feature_idx])#,fontsize=8)
    
    ax20 = fig.add_subplot(gs[2, 0], projection = ccrs.PlateCarree(central_longitude=180))
    ax20.coastlines()
    ax20.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax20.set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax20.tick_params(bottom=False)
    ax20.set_title(r"\textbf{c. }", loc = 'left')
    im20 = ax20.contourf(lon2d, lat2d, SM1_1[phase, feature_idx], cmap='RdYlBu_r', levels = levels_signal, extend = 'max')
    
    ax21 = fig.add_subplot(gs[2, 1], projection = ccrs.PlateCarree(central_longitude=180))
    ax21.coastlines()
    ax21.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax21.set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax21.tick_params(bottom=False)
    ax21.set_title(r"\textbf{d. }", loc = 'left')    
    im21 = ax21.contourf(lon2d, lat2d, SM1_2[phase, feature_idx], cmap='RdYlBu_r', levels = levels_signal, extend = 'max')
    
    ax30 = fig.add_subplot(gs[3, 0], projection = ccrs.PlateCarree(central_longitude=180))
    ax30.coastlines()
    ax30.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax30.set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax30.tick_params(bottom=False)
    ax30.set_title(r"\textbf{e. }", loc = 'left')    
    im30 = ax30.contourf(lon2d, lat2d, SM2_1[phase, feature_idx], cmap='RdYlBu_r', levels = levels_signal, extend = 'max')
    
    ax31 = fig.add_subplot(gs[3, 1], projection = ccrs.PlateCarree(central_longitude=180))
    ax31.coastlines()
    ax31.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax31.set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax31.tick_params(bottom=False)
    ax31.set_title(r"\textbf{f. }", loc = 'left')    
    im31 = ax31.contourf(lon2d, lat2d, SM2_2[phase, feature_idx], cmap='RdYlBu_r', levels = levels_signal, extend = 'max')
    
    ax4 = fig.add_subplot(gs[4, :], visible=False)
    cb4 = fig.colorbar(im31, ax = ax4, shrink = 0.5, pad = -0.2, orientation = 'horizontal')
    cb4.ax.set_title("Arbitrary unit")
    
    if feature_idx >=4: #Write the color bar numbers in scientific writing when necessary
        cb1.formatter.set_powerlimits((0, 2))
        cb1.update_ticks()

    fig.suptitle('', fontsize = 10)
    fig.savefig(PlotDir + feature + '_MC_'+str(phase+1)+'_Propa_Decaying.png')
    plt.close()
    return


if __name__ == '__main__':
    #LOADING DATA
    print('Loading Data')

    train = torch.load(DataDir + 'Datasets/Train_'+str(T_OUTPUT) + '.pt')
    X0_train = train.dataset.X0.detach().numpy()
    y0_train = train.dataset.y0.detach().numpy()
    time0_train = train.dataset.time0
    amp0_train = train.dataset.amp0.detach().numpy()
    phase0_train = train.dataset.phase0.detach().numpy()

    test = torch.load(DataDir + 'Datasets/Test_'+str(T_OUTPUT) + '.pt')
    X0_test = test.dataset.X0.detach().numpy()
    y0_test = test.dataset.y0.detach().numpy()
    time0_test = test.dataset.time0
    amp0_test = test.dataset.amp0.detach().numpy()
    phase0_test = test.dataset.phase0.detach().numpy()

    mu_train_1 = torch.load(DataDir + 'Forecasts/CNN/Preds/mu_train_1.pt').detach().numpy()
    mu_train_3 = torch.load(DataDir + 'Forecasts/CNN/Preds/mu_train_3.pt').detach().numpy()
    mu_train_5 = torch.load(DataDir + 'Forecasts/CNN/Preds/mu_train_5.pt').detach().numpy()
    mu_train_10 = torch.load(DataDir + 'Forecasts/CNN/Preds/mu_train_10.pt').detach().numpy()
    mu_train_15 = torch.load(DataDir + 'Forecasts/CNN/Preds/mu_train_15.pt').detach().numpy()
    mu_train_20 = torch.load(DataDir + 'Forecasts/CNN/Preds/mu_train_20.pt').detach().numpy()
    mu_train_25 = torch.load(DataDir + 'Forecasts/CNN/Preds/mu_train_25.pt').detach().numpy()

    mu_test_1 = torch.load(DataDir + 'Forecasts/CNN/Preds/mu_1.pt').detach().numpy()
    mu_test_3 = torch.load(DataDir + 'Forecasts/CNN/Preds/mu_3.pt').detach().numpy()
    mu_test_5 = torch.load(DataDir + 'Forecasts/CNN/Preds/mu_5.pt').detach().numpy()
    mu_test_10 = torch.load(DataDir + 'Forecasts/CNN/Preds/mu_10.pt').detach().numpy()
    mu_test_15 = torch.load(DataDir + 'Forecasts/CNN/Preds/mu_15.pt').detach().numpy()
    mu_test_20 = torch.load(DataDir + 'Forecasts/CNN/Preds/mu_20.pt').detach().numpy()
    mu_test_25 = torch.load(DataDir + 'Forecasts/CNN/Preds/mu_25.pt').detach().numpy()

    min_all = torch.load(DataDir + 'InputFeatures/ERA5/min.yrs1979-2019.20S-20N.pt').detach().numpy()
    max_all = torch.load(DataDir + 'InputFeatures/ERA5/max.yrs1979-2019.20S-20N.pt').detach().numpy()

    S1_train = torch.load(DataDir + 'Forecasts/CNN/Signals/signal1_train_' + str(T_OUTPUT) + '.pt').detach().numpy()
    S2_train = torch.load(DataDir + 'Forecasts/CNN/Signals/signal2_train_' + str(T_OUTPUT) + '.pt').detach().numpy()

    S1_test = torch.load(DataDir + 'Forecasts/CNN/Signals/signal1_test_' + str(T_OUTPUT) + '.pt').detach().numpy()
    S2_test = torch.load(DataDir + 'Forecasts/CNN/Signals/signal2_test_' + str(T_OUTPUT) + '.pt').detach().numpy()

    S1_train = abs(S1_train)
    S2_train = abs(S2_train)

    S1_test = abs(S1_test)
    S2_test = abs(S2_test)

    #RESCALING
    print('Rescaling')

    for i in range(7):
        mini1 = S1_train[:,i,:,:].min()
        maxi1 = S1_train[:,i,:,:].max()

        mini2 = S2_train[:,i,:,:].min()
        maxi2 = S2_train[:,i,:,:].max()
        
        S1_train[:,i,:,:] = (S1_train[:,i,:,:] - mini1)/(maxi1 - mini1)
        S2_train[:,i,:,:] = (S2_train[:,i,:,:] - mini2)/(maxi2 - mini2)

        S1_test[:,i,:,:] = (S1_test[:,i,:,:] - mini1)/(maxi1 - mini1)
        S2_test[:,i,:,:] = (S2_test[:,i,:,:] - mini2)/(maxi2 - mini2)
          
    X_norm_train = np.copy(X0_train)
    X0_train = min_all + (max_all - min_all)*X_norm_train

    X_norm_test = np.copy(X0_test)
    X0_test = min_all + (max_all - min_all)*X_norm_test
    
    '''
    We are looking for a given starting phase (e.g. forecasts starting in phase 3), and considering that idx1/2 is the first day in this phase.
    Hence, we assume that all the next forecasts of these MJO events starting in this phase will lie within : idx1 - idx1 + n_days + 5
    '''
    
    X1, phase1, S1_1, S2_1, amp0_train = X0_train[idx1: idx1+n_days+6], phase0_train[idx1 : idx1+n_days+6], S1_train[idx1:idx1+n_days+6], S2_train[idx1:idx1+n_days+6], amp0_train[idx1:idx1+n_days+6]
    X2, phase2, S1_2, S2_2, amp0_test = X0_test[idx2: idx2+n_days+6], phase0_test[idx2 : idx2+n_days+6], S1_test[idx2:idx2+n_days+6], S2_test[idx2:idx2+n_days+6], amp0_test[idx2:idx2+n_days+6]

    #COMPUTE COMPOSITE AND SIGNAL MAPS
    print('Compute composite and signal maps')
    XM1,XM2 = np.zeros((8,7,17,144)), np.zeros((8,7,17,144))
    SM1_1,SM1_2 = np.zeros((8,7,17,144)),np.zeros((8,7,17,144))
    SM2_1,SM2_2 = np.zeros((8,7,17,144)),np.zeros((8,7,17,144))

    for i in range(8):
        idx_ph1 = np.where(phase1==i+1)[0]
        idx_ph2 = np.where(phase2==i+1)[0]
        
        XM1[i,:,:,:] = X1[idx_ph1].mean(axis=0)
        XM2[i,:,:,:] = X2[idx_ph2].mean(axis=0)
       
        SM1_1[i,:,:,:] = S1_1[idx_ph1].mean(axis=0)
        SM2_1[i,:,:,:] = S2_1[idx_ph1].mean(axis=0)

        SM1_2[i,:,:,:] = S1_2[idx_ph2].mean(axis=0)
        SM2_2[i,:,:,:] = S2_2[idx_ph2].mean(axis=0)

    #PLOT COMPOSITE MAPS
    print('Plot composite maps')
    
    PlotMaps(XM1, XM2, SM1_1, SM1_2, SM2_1, SM2_2, 2, 1, levels_means, levels_signal)
    PlotMaps(XM1, XM2, SM1_1, SM1_2, SM2_1, SM2_2, 2, 2, levels_means, levels_signal)
    PlotMaps(XM1, XM2, SM1_1, SM1_2, SM2_1, SM2_2, 2, 4, levels_means, levels_signal)
    
    if PLOT_DIAGRAM == True:
        '''
        Select all events in initial phase 3 with amplitude above 1.0 in the selected time 
        frame and plots all the day-1,3,...,10 forecasts starting on these days and observations 
        from the first day in the given phase up to day-20.
        We also make the same assumption as before
        '''
        
        print('Plot Diagram')
        fig,ax = plt.subplots(ncols=2,nrows=1,figsize=(15, 7))
        
        preds1, preds2 = np.zeros((n_days+6, 5, 2)), np.zeros((n_days+6, 5, 2))
        for i in range(n_days+6):
            preds1[i,:,:] = np.array([y0_train[idx1+i], mu_train_1[idx1+i], mu_train_3[idx1+i], mu_train_5[idx1+i], mu_train_10[idx1+i]])#, mu_train_15[idx1+i], mu_20[idx1+i], mu_25[idx1+i]])
            preds2[i,:,:] = np.array([y0_test[idx2+i], mu_test_1[idx2+i], mu_test_3[idx2+i], mu_test_5[idx2+i], mu_test_10[idx2+i]])#, mu_test_15[idx1+i], mu_20[idx1+i], mu_25[idx1+i]])
        
        idx_scatter1 = [idx1, idx1+1, idx1+3, idx1+5, idx1+10, idx1+15, idx1+20]#, idx1+25]
        idx_scatter2 = [idx2, idx2+1, idx2+3, idx2+5, idx2+10, idx2+15, idx2+20]#, idx2 +25]
        
        ax[0].plot(y0_train[idx1:idx1+20+1,0], y0_train[idx1:idx1+20+1,1], marker = '.', markersize = 5,color = 'royalblue') # markevery = np.array([0,1,3,5,10,15,20])
        ax[1].plot(y0_test[idx2:idx2+20+1,0], y0_test[idx2:idx2+20+1,1], marker = '.', markersize = 5, color = 'royalblue') #markevery = np.array([0,1,3,5,10,15,20])

        idx = np.where((phase1==3) & (amp0_train>=1.0))[0]
        colors = cm.get_cmap('Oranges')(np.linspace(0.35,1, idx.shape[0]))
        
        for i, j in enumerate(idx):
            ax[0].plot(preds1[j,:,0], preds1[j,:,1], color = colors[i], marker = '.', markersize = 5)
        
        idx = np.where((phase2==3) & (amp0_test>=1.0))[0] #Replace amp0_test>=1.0 by >=0.0 when looking at initial events in phase 4 for the decaying event because it has decayed
        colors = cm.get_cmap('Oranges')(np.linspace(0.35,1, idx.shape[0]))
        
        for i, j in enumerate(idx):    
            ax[1].plot(preds2[j,:,0], preds2[j,:,1], color = colors[i], marker = '.', markersize = 5)

        labels = ['0', '1', '3', '5', '10', '15', '20']#, '25']

        for i in range(len(labels)): #7
            ax[0].annotate(labels[i], (y0_train[idx_scatter1[i],0]+0.09, y0_train[idx_scatter1[i], 1]), color = 'royalblue')
            ax[1].annotate(labels[i], (y0_test[idx_scatter2[i],0]+0.09, y0_test[idx_scatter2[i], 1]), color = 'royalblue')
            
        for i in range(2):
            ax[i].plot(np.linspace(-3.0, -1.0/np.sqrt(2), 100), np.linspace(-3.0,-1.0/np.sqrt(2),100), color = 'grey', linestyle = '--')
            ax[i].plot(np.linspace(-3.0, -1.0/np.sqrt(2), 100), np.linspace(3.0,1.0/np.sqrt(2),100), color = 'grey', linestyle = '--')
            ax[i].plot(np.linspace(-3.0, -1.0, 100), np.zeros(100), color = 'grey', linestyle = '--')
            ax[i].plot(np.zeros(100), np.linspace(-3.0,-1.0,100), color = 'grey', linestyle = '--')
            
            ax[i].plot(np.linspace(1.0/np.sqrt(2), 3.0, 100), np.linspace(1.0/np.sqrt(2),3.0,100), color = 'grey', linestyle = '--')
            ax[i].plot(np.linspace(1/np.sqrt(2), 3.0, 100), np.linspace(-1.0/np.sqrt(2),-3.0,100), color = 'grey' , linestyle = '--')
            ax[i].plot(np.linspace(1.0, 3.0, 100), np.zeros(100), color = 'grey', linestyle = '--')
            ax[i].plot(np.zeros(100), np.linspace(1.0,3.0,100), color = 'grey', linestyle = '--')    
            
            circle = plt.Circle((0, 0), 1.0, color='grey',linestyle = '--', fill = False)
            ax[i].add_patch(circle)
            
            ax[i].text(-2.0, -1.0, '1', fontsize = 'large')
            ax[i].text(-1.0, -2.0, '2', fontsize = 'large')
            ax[i].text(1.0, -2.0, '3', fontsize = 'large')
            ax[i].text(2.0, -1.0, '4', fontsize = 'large')
            ax[i].text(2.0, 1.0, '5', fontsize = 'large')
            ax[i].text(1.0, 2.0, '6', fontsize = 'large')
            ax[i].text(-1.0, 2.0, '7', fontsize = 'large')
            ax[i].text(-2.0, 1.0, '8', fontsize = 'large')
            
            ax[i].text(0.0, -2.8, 'Indian Ocean', fontsize = 'large', horizontalalignment = 'center')
            ax[i].text(0.0, 2.8, 'Western Pacific', fontsize = 'large', horizontalalignment = 'center')
            ax[i].text(2.8, 0.0, 'Maritime Continent', fontsize = 'large', verticalalignment = 'center', rotation='vertical')
            ax[i].text(-2.8, 0.0, 'Eastern Pacific / Africa', fontsize = 'large', verticalalignment = 'center', rotation='vertical')
            
            ax[i].legend(['Observations', 'Forecasts'], loc = 'upper right')
            ax[i].set_xlim(-3.0,3.0)
            ax[i].set_ylim(-3.0,3.0)
            ax[i].set_xlabel('RMM1')
            ax[i].set_ylabel('RMM2')
            
        ax[0].set_title(r"\textbf{a. } Decaying event", loc ='left')
        ax[1].set_title(r"\textbf{b. } Propagating event", loc = 'left')
        fig.savefig(PlotDir + "Diagram.png", bbox_inches='tight')
        plt.close()
