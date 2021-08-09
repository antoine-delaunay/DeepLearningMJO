import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from tqdm import tqdm
from scipy import stats

import sys
sys.path.append('../../CNN/')
from Dataset import MJODataset

 
T_OUTPUT = 10
lon = np.linspace(-180, 180, 144)
lat = np.linspace(-20, 20, 17)
lon2d, lat2d = np.meshgrid(lon, lat)
features = ['UA200', 'UA850', 'OLR', 'SST', 'SHUM400', 'Z850', 'DLR']
units = [r'UA200 - $m.s^{-1}$', r'UA850 - $m.s^{-1}$', r'OLR - $W.m^{-2}$', r'SST - °K', r'SHUM400', r'Z850 - $m^{2}.s^{-2}$', r'DLR - $J.m^{-2}$']

DataDir = '/network/aopp/preds0/pred/users/delaunay/Data/'
PlotDir =  '/network/aopp/preds0/pred/users/delaunay/Analysis/Interpretation/Plots/Lag' + str(T_OUTPUT)+'/Uncertainty/'
season = "winter"   #"summer" or "winter"

def Plot1Phase(X, XM, significance, feature_idx, phase, levels_means, levels_anoms, n_days):
    '''
    Plot composites and anomalies predictability maps of 1 given phase
    '''
    
    feature = features[feature_idx]
    
    if feature_idx<=3:
        cmap = 'bwr'
    else:
        cmap = 'RdYlBu_r'

    fig = plt.figure(figsize=(17,5), constrained_layout=True)
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 0.1, 1, 0.1])
    
    ax0 = fig.add_subplot(gs[0, :], projection= ccrs.PlateCarree(central_longitude=180)) 
    ax0.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax0.set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax0.tick_params(bottom=False)
    ax0.coastlines()
    
    im0 = ax0.contourf(lon2d, lat2d, XM[phase, feature_idx,:,:], cmap=cmap, levels = levels_means[feature_idx])
    ax0.set_title('Composite map', fontsize=15)
    
    ax1 = fig.add_subplot(gs[1, :], visible=False)
    cb1 = fig.colorbar(im0, ax = ax1, shrink = 0.5, pad = -0.2, orientation = 'horizontal')
    cb1.ax.set_title(units[feature_idx])
    
    ax2 = fig.add_subplot(gs[2, 0], projection = ccrs.PlateCarree(central_longitude=180))
    ax2.coastlines()
    ax2.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax2.set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax2.tick_params(bottom=False)
    ax2.coastlines()

    im2 = ax2.contourf(lon2d, lat2d, X[0, phase, feature_idx], cmap='RdYlBu_r', levels = levels_anoms[feature_idx])
    ax2.set_title('Weak - Predictable (' + str(n_days[0,phase]) + ' days) vs. Unpredictable (' + str(n_days[1,phase]) + ' days)', fontsize=15)
    ax2.contourf(lon2d,lat2d, significance[0,phase,feature_idx,:,:], alpha=0,levels=[0.0,0.5,1],hatches=[None,'..'])

    ax3 = fig.add_subplot(gs[2, 1], projection = ccrs.PlateCarree(central_longitude=180))
    ax3.coastlines()
    ax3.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax3.set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax3.tick_params(bottom=False)
    ax3.coastlines()
    im3 = ax3.contourf(lon2d, lat2d, X[1, phase, feature_idx], cmap='RdYlBu_r', levels = levels_anoms[feature_idx])
    ax3.set_title('Strong - Predictable (' + str(n_days[2,phase]) + ' days) vs. Unpredictable (' + str(n_days[3,phase]) + ' days)', fontsize=15)
    ax3.contourf(lon2d,lat2d, significance[2,phase,feature_idx,:,:], alpha=0,levels=[0.0,0.5,1],hatches=[None,'..'])

    ax4 = fig.add_subplot(gs[3, :], visible=False)
    cb4 = fig.colorbar(im2, ax = ax4, shrink = 0.5, pad = -0.2, orientation = 'horizontal')
    cb4.ax.set_title(units[feature_idx])
    
    if feature_idx >=4:  #Write the color bar numbers in scientific writing when necessary
        cb1.formatter.set_powerlimits((0, 2))
        cb1.update_ticks()
        cb4.formatter.set_powerlimits((0, 2))
        cb4.update_ticks()    
    fig.suptitle(r'\textbf{'+chr(97+phase)+'}'+' - ' + feature + ' - Phase ' + str(phase+1), fontsize=15)
    fig.savefig(PlotDir + feature+'_phase'+str(phase+1)+'_'+str(T_OUTPUT)+'.png')
    plt.close()
    return


def Plot2Phases(X, XM, significance, feature_idx, phase1, phase2, levels_means, levels_anoms, n_days):
    '''
    Plot composites and anomalies predictability maps of 2 phases at the same time (phase1 in the left column and phase2 in the right column)
    '''
    feature = features[feature_idx]
    
    if feature_idx<=3:
        cmap_mean = 'bwr'
    else:
        cmap_mean = 'RdYlBu_r'

    fig = plt.figure(figsize=(17,7), constrained_layout=True)
    gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[1, 0.1, 1, 1, 0.1])
    
    ax00 = fig.add_subplot(gs[0, 0], projection= ccrs.PlateCarree(central_longitude=180)) 
    ax00.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax00.set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax00.tick_params(bottom=False)
    ax00.coastlines()
    ax00.set_title(r"\textbf{a. } Phase " + str(phase1 +1) + ' - composite map', loc = 'left')

    im00 = ax00.contourf(lon2d, lat2d, XM[phase1, feature_idx,:,:], cmap=cmap_mean, levels = levels_means[feature_idx])
    
    ax01 = fig.add_subplot(gs[0, 1], projection= ccrs.PlateCarree(central_longitude=180)) 
    ax01.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax01.set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax01.tick_params(bottom=False)
    ax01.coastlines()
    ax01.set_title(r"\textbf{b. } Phase " + str(phase2 +1) + ' - composite map', loc = 'left')
    
    im01 = ax01.contourf(lon2d, lat2d, XM[phase2, feature_idx,:,:], cmap=cmap_mean, levels = levels_means[feature_idx])
    
    ax1 = fig.add_subplot(gs[1, :], visible=False)
    cb1 = fig.colorbar(im01, ax = ax1, shrink = 0.5, pad = -0.2, orientation = 'horizontal')
    cb1.ax.set_title(units[feature_idx])#,fontsize=8)
    
    ax20 = fig.add_subplot(gs[2, 0], projection = ccrs.PlateCarree(central_longitude=180))
    ax20.coastlines()
    ax20.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax20.set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax20.tick_params(bottom=False)
    ax20.set_title(r"\textbf{c. }"+'Weak - Predictable (' + str(n_days[0,phase1]) + ' days) vs. Unpredictable (' + str(n_days[1,phase1]) + ' days)', loc = 'left')

    im20 = ax20.contourf(lon2d, lat2d, X[0, phase1, feature_idx], cmap='RdYlBu_r', levels = levels_anoms[feature_idx])
    ax20.contourf(lon2d,lat2d, significance[0,phase1,feature_idx,:,:], alpha=0,levels=[0.0,0.5,1],hatches=[None,'..'])
    
    ax21 = fig.add_subplot(gs[2, 1], projection = ccrs.PlateCarree(central_longitude=180))
    ax21.coastlines()
    ax21.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax21.set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax21.tick_params(bottom=False)
    ax21.set_title(r"\textbf{d. }"+'Weak - Predictable (' + str(n_days[0,phase2]) + ' days) vs. Unpredictable (' + str(n_days[1,phase2]) + ' days)', loc = 'left')    
    im21 = ax21.contourf(lon2d, lat2d, X[0, phase2, feature_idx], cmap='RdYlBu_r', levels = levels_anoms[feature_idx])
    ax21.contourf(lon2d,lat2d, significance[0,phase2,feature_idx,:,:], alpha=0,levels=[0.0,0.5,1],hatches=[None,'..'])
    
    
    ax30 = fig.add_subplot(gs[3, 0], projection = ccrs.PlateCarree(central_longitude=180))
    ax30.coastlines()
    ax30.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax30.set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax30.tick_params(bottom=False)
    ax30.set_title(r"\textbf{e. }"+'Strong - Predictable (' + str(n_days[2,phase1]) + ' days) vs. Unpredictable (' + str(n_days[3,phase1]) + ' days)', loc = 'left')    
    im30 = ax30.contourf(lon2d, lat2d, X[1, phase1, feature_idx], cmap='RdYlBu_r', levels = levels_anoms[feature_idx])
    ax30.contourf(lon2d,lat2d, significance[2,phase1,feature_idx,:,:], alpha=0,levels=[0.0,0.5,1],hatches=[None,'..'])
    
    ax31 = fig.add_subplot(gs[3, 1], projection = ccrs.PlateCarree(central_longitude=180))
    ax31.coastlines()
    ax31.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax31.set_xticklabels(['30°E', '60°E', '90°E', '120°E', '150°E', '180°E', '150°W', '120°W', '90°W', '60°W', '30°W'])
    ax31.tick_params(bottom=False)
    ax31.set_title(r"\textbf{f. }"+'Strong - Predictable (' + str(n_days[2,phase2]) + ' days) vs. Unpredictable (' + str(n_days[3,phase2]) + ' days)', loc = 'left')    
    im31 = ax31.contourf(lon2d, lat2d, X[1, phase2, feature_idx], cmap='RdYlBu_r', levels = levels_anoms[feature_idx])
    ax31.contourf(lon2d,lat2d, significance[2,phase2,feature_idx,:,:], alpha=0,levels=[0.0,0.5,1],hatches=[None,'..'])
        
    ax4 = fig.add_subplot(gs[4, :], visible=False)
    cb4 = fig.colorbar(im31, ax = ax4, shrink = 0.5, pad = -0.2, orientation = 'horizontal')
    cb4.ax.set_title(units[feature_idx])
    
    if feature_idx >=4: #Write the color bar numbers in scientific writing when necessary
        cb1.formatter.set_powerlimits((0, 2))
        cb1.update_ticks()
        cb4.formatter.set_powerlimits((0, 2))
        cb4.update_ticks()        

    fig.suptitle('', fontsize = 10)
    fig.savefig(PlotDir+ feature+'_'+str(phase1+1)+'_'+str(phase2+1)+'_uncertainty.png')
    plt.close()
    return


if __name__ == '__main__':
    print("T_OUTPUT = " + str(T_OUTPUT))
    
    '''DATALOADING'''
    print('Dataloading')

    dataloader = torch.load(DataDir + 'Datasets/Train_'+str(T_OUTPUT) + '.pt')
    time0_train, amp0_train = dataloader.dataset.time0, dataloader.dataset.amp0.detach().numpy()
    month0_train = np.array([time0_train[i].month for i in range(len(time0_train))])
    phase0_train = dataloader.dataset.phase0.detach()
    cov_train = torch.load(DataDir + 'Forecasts/CNN/Preds/cov_train_' + str(T_OUTPUT) + '.pt').detach()
    X0_train, y0_train, y_T_OUTPUT_train = dataloader.dataset.X0.detach(), dataloader.dataset.y0.detach().numpy(), dataloader.dataset.y_T_OUTPUT.detach()
    mu_train = torch.load(DataDir + 'Forecasts/CNN/Preds/mu_train_' + str(T_OUTPUT) + '.pt').detach()

    dataloader = torch.load(DataDir + 'Datasets/Test_'+str(T_OUTPUT) + '.pt')
    time0_test, amp0_test = dataloader.dataset.time0, dataloader.dataset.amp0.detach().numpy()
    month0_test = np.array([time0_test[i].month for i in range(len(time0_test))])
    phase0_test = dataloader.dataset.phase0.detach()
    cov_test = torch.load(DataDir + 'Forecasts/CNN/Preds/cov_' + str(T_OUTPUT) + '.pt').detach()
    mu_test = torch.load(DataDir + 'Forecasts/CNN/Preds/mu_' + str(T_OUTPUT) + '.pt').detach()

    X0_test, y0_test, y_T_OUTPUT_test = dataloader.dataset.X0.detach(), dataloader.dataset.y0.detach().numpy(), dataloader.dataset.y_T_OUTPUT.detach()

    month0, amp0 = np.concatenate((month0_train, month0_test), axis=0), np.concatenate((amp0_train, amp0_test), axis=0)

    X0, y0, y_T_OUTPUT, phase0 = torch.cat((X0_train, X0_test),0), np.concatenate((y0_train, y0_test), axis=0), np.concatenate((y_T_OUTPUT_train, y_T_OUTPUT_test), axis=0), torch.cat((phase0_train, phase0_test), 0)
    cov = torch.cat((cov_train, cov_test),0)
    mu = torch.cat((mu_train, mu_test),0).numpy()

    '''PUT NORMALIZED INPUTS BACK TO THE ORIGINAL SCALES'''
    print('Put normalized inputs back to the original scales')
    min_all = torch.load(DataDir + 'InputFeatures/ERA5/min.yrs1979-2019.20S-20N.pt')
    max_all = torch.load(DataDir + 'InputFeatures/ERA5/max.yrs1979-2019.20S-20N.pt')

    X_norm = torch.clone(X0)
    X0 = min_all + (max_all - min_all)*X_norm

    #Choose season
    if season == 'winter':
        idx = np.where((month0<=4) | (month0>=11))[0]
        phase0, X0, cov, amp0, y_T_OUTPUT, mu = phase0[idx], X0[idx], cov[idx], amp0[idx], y_T_OUTPUT[idx], mu[idx]

    elif season == 'summer':
        idx = np.where((month0>=5) & (month0<=10))[0]
        phase0, X0, cov, amp0, y_T_OUTPUT, mu = phase0[idx], X0[idx], cov[idx], amp0[idx], y_T_OUTPUT[idx], mu[idx]


    '''COMPUTE QUANTILES AND ASSOCIATED MAPS'''
    print('Compute quantiles and associated maps')
    print('Define predictable and unpredictable events')
    q30, q70 = torch.zeros(2), torch.zeros(2)
    q30[0], q30[1], q70[0], q70[1] = torch.quantile(cov[:,0], 0.3), torch.quantile(cov[:,1], 0.3), torch.quantile(cov[:,0], 0.7), torch.quantile(cov[:,1], 0.7)

    #Predictable 
    idx0 = np.where(cov[:,0]<=q30[0])[0]
    idx1 = np.where(cov[:,1]<=q30[1])[0]
    idx2 = np.where(cov[:,0]>=q70[0])[0]
    idx3 = np.where(cov[:,1]>=q70[1])[0]
    idx4 = np.where((cov[:,0]<=q30[0]) & (cov[:,1]<=q30[1]))[0]
    idx5 = np.where((cov[:,0]>=q70[0]) & (cov[:,1]>=q70[1]))[0]

    #Unpredictable
    idx_wp = np.where(((cov[:,0]<=q30[0]) & (cov[:,1]<=q30[1])) & (amp0<=1.0))[0]
    idx_wu = np.where(((cov[:,0]>=q70[0]) & (cov[:,1]>=q70[1])) & (amp0<=1.0))[0]
    idx_sp = np.where(((cov[:,0]<=q30[0]) & (cov[:,1]<=q30[1])) & (amp0>=1.0))[0]
    idx_su = np.where(((cov[:,0]>=q70[0]) & (cov[:,1]>=q70[1])) & (amp0>=1.0))[0]

    ampT = np.linalg.norm(y_T_OUTPUT, axis=1)

    n_wpT = np.where(ampT[idx_wp]>=1.0)[0].shape[0]/idx_wp.shape[0]
    n_wuT = np.where(ampT[idx_wu]>=1.0)[0].shape[0]/idx_wu.shape[0]
    n_spT = np.where(ampT[idx_sp]>=1.0)[0].shape[0]/idx_sp.shape[0]
    n_suT = np.where(ampT[idx_su]>=1.0)[0].shape[0]/idx_su.shape[0]

    ph_wp, ph_wu, ph_sp, ph_su = phase0[idx_wp], phase0[idx_wu], phase0[idx_sp], phase0[idx_su]
    X0_wp, X0_wu, X0_sp, X0_su = X0[idx_wp], X0[idx_wu], X0[idx_sp], X0[idx_su]
    mu_wp, mu_wu, mu_sp, mu_su = mu[idx_wp], mu[idx_wu], mu[idx_sp], mu[idx_su]

    X = np.zeros((4,8,7,17,144))
    for i in range(8):
        idx = np.where(ph_wp==i+1)[0]
        X[0,i,:,:,:] = X0_wp[idx].mean(axis=0)
        idx = np.where(ph_wu==i+1)[0]
        X[1,i,:,:,:] = X0_wu[idx].mean(axis=0)
        idx = np.where(ph_sp==i+1)[0]
        X[2,i,:,:,:] = X0_sp[idx].mean(axis=0)
        idx = np.where(ph_su==i+1)[0]
        X[3,i,:,:,:] = X0_su[idx].mean(axis=0)

    #Composite average maps (all forecasts with amp0>=1.0 are considered)
    XM = torch.zeros(8,7,17,144)
    for i in range(8):
        idx = np.where(phase0==i+1)[0]
        XM[i,:,:,:] = X0[idx].mean(dim=0)

    #Compute anomalies maps between predictable and unpredictable events
    X_ = np.zeros((2,8,7,17,144))
    X_[0,:,:,:,:] = X[0,:,:,:,:] - X[1,:,:,:,:]
    X_[1,:,:,:,:] = X[2,:,:,:,:] - X[3,:,:,:,:]

    t_stats, p_values = np.ones((4, 8, 7, 17, 144)), np.ones((4, 8, 7, 17, 144))

    n_days = np.zeros((4,8), dtype = np.int32)

    '''COMPUTE SIGNIFICANCE'''
    print('Compute significance')
    
    ''' For each point of the map and feature, we compute the significance 
        of the difference between weak(strong) predictable and weak (strong) unpredictable forecasts
        two-tailed student's t-test with 95% of confidence level
    '''

    for p in tqdm(range(8)):
        idx = np.where(phase0==p+1)[0]
        idx_wp = np.where(ph_wp==p+1)[0]
        idx_wu = np.where(ph_wu==p+1)[0]
        idx_sp = np.where(ph_sp==p+1)[0] 
        idx_su = np.where(ph_su==p+1)[0] 
        
        n_days[0,p] = idx_wp.shape[0]
        n_days[1,p] = idx_wu.shape[0]
        n_days[2,p] = idx_sp.shape[0]
        n_days[3,p] = idx_su.shape[0]
            
        for k in range(7):
            for i in range(17):
                for j in range(144):
                     t_stats[0,p,k,i,j], p_values[0,p,k,i,j] = stats.ttest_ind(X0_wp[idx_wp,k,i,j], X0_wu[idx_wu,k,i,j], axis=0)      
                     t_stats[1,p,k,i,j], p_values[1,p,k,i,j] = t_stats[0,p,k,i,j], p_values[0,p,k,i,j]
                     t_stats[2,p,k,i,j], p_values[2,p,k,i,j] = stats.ttest_ind(X0_sp[idx_sp,k,i,j], X0_su[idx_su,k,i,j], axis=0)      
                     t_stats[3,p,k,i,j], p_values[3,p,k,i,j] = t_stats[2,p,k,i,j], p_values[2,p,k,i,j]

        
    significance = np.zeros((4,8,7,17,144))

    idx_significance = np.where(p_values<=0.05)
    significance[idx_significance] = 1.0

    np.save(DataDir + 'Forecasts/CNN/Significance/significance_predictability_alea_'+season +'_'+str(T_OUTPUT)+'.npy', significance)
    
    '''COMPUTE CONTOUR LEVELS'''
    print('Compute contour levels')
    levels_anoms = []
    levels_means = []

    for i in range(7):
        temp = torch.from_numpy(X_[:,:,i,:,:])
        idx = np.where(temp.isnan()==False)
        levels_anoms.append(np.linspace(temp[idx].min(), temp[idx].max(), 10))

    for i in range(7):
        temp = XM[:,i,:,:]
        idx = np.where(temp.isnan()==False)
        levels_means.append(np.linspace(temp[idx].min(), temp[idx].max(), 10))

    '''PLOTTING'''
    print('Plotting')

    for i in tqdm(range(7)):
        for j in range(8):
            Plot1Phase(X_, XM, significance, i, j, levels_means, levels_anoms, n_days)

    Plot2Phases(X_, XM, significance, 2, 2, 6, levels_means, levels_anoms, n_days)        
    Plot2Phases(X_, XM, significance, 4, 2, 6, levels_means, levels_anoms, n_days)        
    Plot2Phases(X_, XM, significance, 5, 2, 6, levels_means, levels_anoms, n_days)