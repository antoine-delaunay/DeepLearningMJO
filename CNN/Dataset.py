import torch
from torch.utils.data import Dataset, DataLoader
   
'''
Class containing all the necessary information to train, test the CNN and analyze its forecasts with PyTorch
time0, X0, y0, amp0, phase 0 are the date, inputs fields, observed RMMs, observed RMM amplitude, observed RMM phase (from 1 to 8) on day 0.
y_T_OUTPUT is the observed RMMs on the forecast lead day.
'''

class MJODataset(Dataset):
    def __init__(self, time0, X0, y0, amp0, phase0, y_T_OUTPUT):
        self.time0 = time0
        self.X0 = X0
        self.y0 = y0
        self.amp0 = amp0
        self.phase0 = phase0
        self.y_T_OUTPUT = y_T_OUTPUT
    
    def __getitem__(self, index):
        return(self.X0[index], self.y0[index], self.amp0[index], self.phase0[index], self.y_T_OUTPUT[index])
    
    def __len__(self):
        return self.y0.shape[0]
    