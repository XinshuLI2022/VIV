import pandas as pd
import numpy as np
import torch
import random
import copy
from scipy.stats import norm
from torch.utils.data import Dataset, DataLoader

def get_var_df(df,var):
    var_cols = [c for c in df.columns if c.startswith(var)]
    return df[var_cols].to_numpy()

def cat(data_list, axis=1):
    try:
        output=torch.cat(data_list,axis)
    except:
        output=np.concatenate(data_list,axis)

    return output

class Data(object):
    def __init__(self, train, valid, test, num):
        self.train = train
        self.valid = valid
        self.test  = test
        self.num   = num

    def transform(self, load_dict):
        if load_dict['type'] == 'numpy' or load_dict['type'] == 'np':
            try:
                self.numpy()
            except:
                pass

            return self.train

        if load_dict['type'] == 'tensor':
            try:
                self.tensor()
            except:
                pass
        
        if load_dict['type'] == 'double':
            try:
                self.double()
            except:
                pass

        if load_dict['GPU']:
            try:
                self.cuda()
            except:
                pass

        loader = self.get_loader(load_dict)
        return loader

    def tensor(self):
        try:
            self.train.to_tensor()
            self.valid.to_tensor()
            self.test.to_tensor()
        except:
            pass

    def double(self):
        try:
            self.train.to_double()
            self.valid.to_double()
            self.test.to_double()
        except:
            pass

    def cpu(self):
        try:
            self.train.to_cpu()
            self.valid.to_cpu()
            self.test.to_cpu()
        except:
            pass

    def detach(self):
        try:
            self.train.detach()
            self.valid.detach()
            self.test.detach()
        except:
            pass

    def numpy(self):
        try:
            self.train.to_numpy()
            self.valid.to_numpy()
            self.test.to_numpy()
        except:
            pass

    def cuda(self,n=0,type='float'):
        if type == 'float':
            try:
                self.tensor()
            except:
                pass
        elif type == 'double':
            try:
                self.double()
            except:
                pass
        
        try:
            self.train.to_cuda(n)
            self.valid.to_cuda(n)
            self.test.to_cuda(n)
        except:
            pass
    
    def get_loader(self, load_dict, data=None):
        if data is None:
            data = self.train
        loader = DataLoader(data, batch_size=load_dict['batch_size'])
        return loader

    def split(self, split_ratio=0.5, data=None):
        if data is None: data = self.train
        self.data1 = copy.deepcopy(data)
        self.data2 = copy.deepcopy(data)

        split_num = int(data.length * split_ratio)
        self.data1.split(0, split_num)
        self.data2.split(split_num, data.length)

        return self.data1, self.data2

class CausalDataset(Dataset):
    def __init__(self, df, variables = ['u','x','v','z','p','m','t','y','f','c'], observe_vars=['z', 't']):
        if not 'c' in variables: variables.append('c')

        self.length = len(df)
        self.variables = variables
        
        for var in variables:
            exec(f'self.{var}=get_var_df(df, \'{var}\')')
        
        observe_list = []
        for item in observe_vars:
            exec(f'observe_list.append(self.{item})')
        self.c = np.concatenate(observe_list, axis=1)
            
    def to_cpu(self):
        for var in self.variables:
            exec(f'self.{var} = self.{var}.cpu()')
            
    def to_cuda(self,n=0):
        for var in self.variables:
            exec(f'self.{var} = self.{var}.cuda({n})')
    
    def to_tensor(self):
        if type(self.t) is np.ndarray:
            for var in self.variables:
                exec(f'self.{var} = torch.Tensor(self.{var})')
        else:
            for var in self.variables:
                exec(f'self.{var} = self.{var}.float()')
            
    def to_double(self):
        if type(self.t) is np.ndarray:
            for var in self.variables:
                exec(f'self.{var} = torch.Tensor(self.{var}).double()')
        else:
            for var in self.variables:
                exec(f'self.{var} = self.{var}.double()')

            
    def to_numpy(self):
        try:
            self.detach()
        except:
            pass
        try:
            self.to_cpu()
        except:
            pass

        for var in self.variables:
            exec(f'self.{var} = self.{var}.numpy()')

    def shuffle(self):
        idx = list(range(self.length))
        random.shuffle(idx)
        for var in self.variables:
            try:
                exec(f'self.{var} = self.{var}[idx]')
            except:
                pass
    
    def split(self, start, end):
        for var in self.variables:
            try:
                exec(f'self.{var} = self.{var}[start:end]')
            except:
                pass

        self.length = end - start
            
    def to_pandas(self):
        var_list = []
        var_dims = []
        var_name = []
        for var in self.variables:
            exec(f'var_list.append(self.{var})')
            exec(f'var_dims.append(self.{var}.shape[1])')
        for i in range(len(self.variables)):
            for d in range(var_dims[i]):
                var_name.append(self.variables[i]+str(d))
        df = pd.DataFrame(np.concatenate(var_list, axis=1),columns=var_name)
        return df
    
    def detach(self):
        for var in self.variables:
            exec(f'self.{var} = self.{var}.detach()')
        
    def __getitem__(self, idx):
        var_dict = {}
        for var in self.variables:
            exec(f'var_dict[\'{var}\']=self.{var}[idx]')
        
        return var_dict

    def __len__(self):
        return self.length