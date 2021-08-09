import pandas as pd
import numpy as np
import os
from datetime import datetime
import netCDF4 as nc
from tqdm import tqdm
from properscoring import crps_ensemble

'''
This file loads the daily forecasts S2S text files, preprocess them
to make them "csv-readable" and store the forecast of each lead time in a specific table
'''

model = "CNRM"  #HMCR, ECMWF, CNRM, BOM
n_members = 10 #10, 11, 10, 10

InputDir_str = "/network/aopp/preds0/pred/users/delaunay/Data/DynamicModels/" + model +"/RawData/"
InputDir = os.fsencode(InputDir_str)
OutputDir = '/network/aopp/preds0/pred/users/delaunay/Data/Forecasts/' + model + '/EnsMembers/'

data_rmm = nc.Dataset('/network/aopp/preds0/pred/users/delaunay/Data/InputFeatures/ERA5/MJO_PC_INDEX_1979_2019.nc')
time = nc.num2date(data_rmm['time'][:], units = data_rmm['time'].units, only_use_cftime_datetimes=False, only_use_python_datetimes = True)
time2 = np.array([time[i].date() for i in range(len(time))])
data_model = []

n_days = 35

for file in tqdm(os.listdir(InputDir)):
     filename = os.fsdecode(file)
     if filename.endswith(".txt"): 
     
        #Preprocess the S2S files (change the columns spaces)
         fin = open(InputDir_str+filename, "rt")
         data = fin.read()
         data = data.replace('  ', ' ')
         fin.close()
         fin = open(InputDir_str+filename, "wt")
         fin.write(data)
         fin.close()
        

        #Read the files and append the data of each reforecast date to a list      
         df_file = pd.read_csv(InputDir_str+filename,sep=' ', header=0 )

         date_str = df_file.columns.to_list()[3][:-2]
         date = datetime.strptime(date_str, '%Y%m%d').date()
         df_file.columns = ['Forecast day', 'prod', 'rf', 'RMM1', 'RMM2', 'Amplitude', 'Phase']
         forecasts = df_file[['Forecast day','RMM1','RMM2']].loc[((df_file['prod']=='pf')|(df_file['prod']=='cf')) & (df_file['Forecast day']<=n_days*24)]
         forecasts_np = np.zeros((n_days, n_members, 2))
        
         for i in range(n_days):
             temp = forecasts[['RMM1','RMM2']].loc[forecasts['Forecast day'] == (i+1)*24].to_numpy()[:n_members,:]
             forecasts_np[i,:,:] = temp
         
         app = [date, forecasts_np]
         data_model.append(app)

#For each lead time, create a dataframe and store the forecasts from the previous list
for j in tqdm(range(n_days)):
    data = []
    for i in range(len(data_model)):
        idx = np.where(time2 == data_model[i][0])[0][0]
        if idx<=(data_rmm['PC1'].shape[0] - 40): #Keep initial days where we have forecasts (i.e. not at the end of the dataset)
          temp = np.zeros(n_members*2 + 4) #day0 obs, forecast day j , obs day j
          temp[0],  temp[1] = data_rmm['PC1'][idx], data_rmm['PC2'][idx]
          temp[2], temp[3] = data_rmm['PC1'][idx+j+1], data_rmm['PC2'][idx+j+1]
          
          for k in range(n_members):
             temp[4+2*k], temp[4+2*k+1] = data_model[i][1][j][k][0], data_model[i][1][j][k][1]
          
          append = [data_model[i][0]]
          for k in range(temp.shape[0]):
              append.append(temp[k])
          data.append(append)
        
    columns = ['Date_day0', 'RMM1_0','RMM2_0','RMM1_obs','RMM2_obs']
    for i in range(n_members):
        columns.append('RMM1_pf'+str(i+1))
        columns.append('RMM2_pf' + str(i+1))
        
    data_pd = pd.DataFrame(data=data, columns = columns)
    data_pd = data_pd.sort_values('Date_day0')
    data_pd.to_csv(OutputDir+ model + str(j+1)+".txt", index=False)