import torch
import torch.nn as nn
import numpy as np
import netCDF4 as nc
from torch.utils.data import Dataset, DataLoader
from Dataset import MJODataset

DataDir = '/network/aopp/preds0/pred/users/delaunay/Data/InputFeatures/ERA5/'
DatasetDir = '/network/aopp/preds0/pred/users/delaunay/Data/Datasets/'

torch.manual_seed(0)
batch_size = 50
T_OUTPUT = 10
idx_start = 120
idx_end = 14944

print('T_OUTPUT = ' + str(T_OUTPUT))

'''LOADING NETCDF4 DATASETS'''
print('Loading netCDF4 datasets')
data_ua200= nc.Dataset(DataDir + 'ua200.day.anomalies.yrs1979-2019.20S-20N.nc')
data_ua850 = nc.Dataset(DataDir + 'ua850.day.anomalies.yrs1979-2019.20S-20N.nc')
data_rlut = nc.Dataset(DataDir + 'rlut.day.anomalies.yrs1979-2019.20S-20N.nc')
data_sst = nc.Dataset(DataDir + 'tos.day.anomalies.yrs1979-2019.20S-20N.nc')
data_hgt850 = nc.Dataset(DataDir + 'z850.day.means.yrs1979-2019.20S-20N.nc')
data_shum400 = nc.Dataset(DataDir + 'shum400.day.means.yrs1979-2019.20S-20N.nc')
data_dlr = nc.Dataset(DataDir + 'dlr.day.means.yrs1979-2019.20S-20N.nc')
data_rmm = nc.Dataset(DataDir + 'MJO_PC_INDEX_1979_2019.nc')
time = nc.num2date(data_rmm['time'][idx_start:idx_end], units = data_rmm['time'].units, only_use_cftime_datetimes=False, only_use_python_datetimes = True)

'''FILLING MISSING VALUES WITH ZEROS'''
print('Filling missing values with zeros')
sst_np = data_sst['tos'][:]
sst_np[np.where(sst_np.mask==True)] = 0.0
dlr_np = data_dlr['strd'][:,:,:]
dlr_np[np.where(dlr_np.mask==True)] = 0.0

'''CONVERTING TO TORCH TENSORS'''
print('Converting to torch tensors')
#DATA INPUTS
ua200 = torch.tensor(data_ua200['ua200_anom'][idx_start:idx_end, :, :])
ua850 = torch.tensor(data_ua850['ua850_anom'][idx_start:idx_end, :, :])
rlut = torch.tensor(data_rlut['rlut_anom'][idx_start:idx_end, :, :])
sst = torch.tensor(sst_np[idx_start:idx_end, :, :])
hgt850 = torch.tensor(data_hgt850['z'][idx_start:idx_end, :, :])
shum400 = torch.tensor(data_shum400['q'][idx_start:idx_end, :, :])
dlr = torch.tensor(dlr_np[idx_start: idx_end, :, :])

#RMM OUTPUTS
rmm1_T_OUTPUT = torch.tensor(data_rmm['PC1'][idx_start + T_OUTPUT:idx_end + T_OUTPUT])
rmm2_T_OUTPUT = torch.tensor(data_rmm['PC2'][idx_start + T_OUTPUT:idx_end + T_OUTPUT])
rmm1 = torch.tensor(data_rmm['PC1'][idx_start: idx_end])
rmm2 = torch.tensor(data_rmm['PC2'][idx_start: idx_end])
amp0 = torch.norm(torch.cat((rmm1.unsqueeze(1), rmm2.unsqueeze(1)),1), dim=1)

train_len = int(0.8*rmm1_T_OUTPUT.shape[0])
test_len = rmm1_T_OUTPUT.shape[0] - train_len

'''COMPUTING INITIAL PHASE'''
print('Computing initial phase')
phase_rad = np.arctan2(rmm2, rmm1)

pi = np.pi
phase0 = torch.zeros(phase_rad.shape[0], dtype = torch.int32)

for i in range(phase0.shape[0]):
    if (phase_rad[i]>= -pi and phase_rad[i]< -3*pi/4):
        phase0[i] = 1
    elif (phase_rad[i]>= -3*pi/4 and phase_rad[i]< -pi/2):
        phase0[i] = 2
    elif (phase_rad[i] >= -pi/2 and phase_rad[i] < -pi/4):
        phase0[i] = 3
    elif (phase_rad[i] >= -pi/4 and phase_rad[i] < 0.0):
        phase0[i] = 4
    elif (phase_rad[i] >= 0.0 and phase_rad[i] < pi/4):
        phase0[i] = 5
    elif (phase_rad[i] >= pi/4 and phase_rad[i] < pi/2):
        phase0[i] = 6
    elif (phase_rad[i] >= pi/2 and phase_rad[i] < 3*pi/4):
        phase0[i] = 7
    elif (phase_rad[i] >= 3*pi/4 and phase_rad[i] < pi):
        phase0[i] = 8   


'''COMPUTING MINS AND MAX FOR RESCALING'''
print('Computing min and max for rescaling')
min_ua200 = torch.min(ua200, dim = 0).values
max_ua200 = torch.max(ua200, dim = 0).values
    
min_ua850 = torch.min(ua850, dim = 0).values
max_ua850 = torch.max(ua850, dim = 0).values
    
min_rlut = torch.min(rlut, dim = 0).values
max_rlut = torch.max(rlut, dim = 0).values

min_sst = torch.min(sst, dim = 0).values
max_sst = torch.max(sst, dim = 0).values

min_hgt850 = torch.min(hgt850, dim = 0).values
max_hgt850 = torch.max(hgt850, dim = 0).values

min_shum400 = torch.min(shum400, dim = 0).values
max_shum400 = torch.max(shum400, dim = 0).values

min_dlr = torch.min(dlr, dim = 0).values
max_dlr = torch.max(dlr, dim = 0).values

min_all = torch.cat((min_ua200.unsqueeze(0), min_ua850.unsqueeze(0), min_rlut.unsqueeze(0),
            min_sst.unsqueeze(0), min_shum400.unsqueeze(0), min_hgt850.unsqueeze(0), min_dlr.unsqueeze(0)),0)
max_all = torch.cat((max_ua200.unsqueeze(0), max_ua850.unsqueeze(0), max_rlut.unsqueeze(0),
            max_sst.unsqueeze(0), max_shum400.unsqueeze(0), max_hgt850.unsqueeze(0),max_dlr.unsqueeze(0)),0)

'''MIN-MAX SCALING'''
print('Min-Max scaling')

ua200 = (ua200 - min_ua200)/(max_ua200- min_ua200)
ua850 = (ua850 - min_ua850)/(max_ua850 - min_ua850)
rlut = (rlut - min_rlut)/(max_rlut - min_rlut)
hgt850 = (hgt850 - min_hgt850)/(max_hgt850 - min_hgt850)
shum400 = (shum400 - min_shum400)/(max_shum400 - min_shum400)

#Avoid to have nan values when max = min =0.0 (e.g. point located on land for sst)
zero = torch.tensor([0.0], dtype = torch.float)
sst = torch.where(((sst - min_sst)/(max_sst - min_sst)).isnan()==False, (sst - min_sst)/(max_sst - min_sst), sst)

zero2 = torch.tensor([0.0], dtype = torch.float)
dlr = torch.where(((dlr - min_dlr)/(max_dlr - min_dlr)).isnan()==False, (dlr - min_dlr)/(max_dlr - min_dlr), dlr)

'''CONCATENATION'''
print('Concatenation')
X0 = torch.cat((ua200.unsqueeze(1), ua850.unsqueeze(1), rlut.unsqueeze(1), sst.unsqueeze(1), shum400.unsqueeze(1), hgt850.unsqueeze(1), dlr.unsqueeze(1)), 1)
y0 = torch.cat((rmm1.unsqueeze(1),rmm2.unsqueeze(1)), 1)

y_T_OUTPUT = torch.cat((rmm1_T_OUTPUT.unsqueeze(1),rmm2_T_OUTPUT.unsqueeze(1)),1)

X0 = X0.type(torch.DoubleTensor)
y0 = y0.type(torch.DoubleTensor)

y_T_OUTPUT = y_T_OUTPUT.type(torch.DoubleTensor)

'''SAVE FINAL DATASETS'''
print('Saving final datasets')
train_dataset = MJODataset(time[:train_len], X0[:train_len].clone(), y0[:train_len].clone(), 
                amp0[:train_len].clone(), phase0[:train_len].clone(), y_T_OUTPUT[:train_len].clone())
test_dataset = MJODataset(time[train_len:], X0[train_len:].clone(), y0[train_len:].clone(),
                amp0[train_len:].clone(), phase0[train_len:].clone(), y_T_OUTPUT[train_len:].clone())

train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

torch.save(train, DatasetDir + "Train_"+ str(T_OUTPUT)+".pt")
torch.save(test, DatasetDir + "Test_" + str(T_OUTPUT) +".pt")

'''SAVE MIN AND MAX'''
torch.save(min_all, DataDir + 'min.yrs1979-2019.20S-20N.pt')
torch.save(max_all, DataDir + 'max.yrs1979-2019.20S-20N.pt')

'''CLOSING NETCDF4 DATASETS'''
data_ua200.close()
data_ua850.close()
data_rlut.close()
data_sst.close()
data_rmm.close()
data_shum400.close()
data_hgt850.close()
data_dlr.close()