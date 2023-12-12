from itertools import product
import numpy as np
from numpy.random import default_rng
from pandas import DataFrame
import os
import pandas as pd
from utils import CausalDataset, Data, cat, set_seed
np.random.seed(42)

def h(t):
    return 2 * ((t - 5) ** 4 / 600 + np.exp(-4 * (t - 5) ** 2) + t / 10 - 2)

def f(p, t, s):
    return 100 + (10 + p) * s * h(t) - 2 * p
def norm(x):
    return (x - x.mean()) / x.std()
  
def generate_IHDP(num,data, rho=0.5, alpha=0, beta=1, seed=2023):

    rng=default_rng(seed)
    emotion = data[:,0:1] # feature X3 ,i.e.,a1 
    time = data[:,1:2] # feature X2, i.e., c1
    cost = data[:,2:3] # should be feature X1, i.e., z1
    noise_price = rng.normal(0, 1.0, (num,1)) # to create U, unobserved confounder
    noise_demand = rho * noise_price + rng.normal(0, np.sqrt(1 - rho ** 2), (num,1)) # to create U, unobserved confounder
    price = 25 + (beta * cost + 3) * h(time) + alpha * cost + noise_price # t
    # normalize t 
    price_norm = norm(price)

    price_intervention = np.linspace(np.percentile(price, 2.5),np.percentile(price, 97.5),num).reshape(-1, 1)

    # normalize tt

    price_intervention_norm = norm(price_intervention)


    structural = f(price, time, emotion).astype(float) #g
    ## normalize g
    structural = norm(structural)

    outcome = (structural + noise_demand).astype(float) #y

    structural_t = f(price_intervention, time, emotion).astype(float) #g
    structural_t = norm(structural_t)

    outcome_t = (structural_t + noise_demand).astype(float) #y

    
    g_t0 = f(price-price, time, emotion).astype(float)
    g_t0 = norm(g_t0)
    y_t0 = (g_t0 + noise_demand).astype(float)
 
    numpys = [noise_price,noise_demand,  time, emotion, cost,time, emotion, price_norm, g_t0,g_t0,y_t0,  structural,structural, outcome,price_intervention_norm,structural_t,structural_t,outcome_t]
    
    data = DataFrame(np.concatenate(numpys, axis=1),
                          columns=['u1','u2','x1','x2','i1','c1','a1','t1','m1','m2','m3','g1','v1','y1','e1','b1','f1','h1'])
    return data

class Generator(object):
    def __init__(self, gens_dict, G=False):
        
        self.num = gens_dict['num']
        self.num_reps = gens_dict['reps']
        self.seed = gens_dict['seed']
        self.dataDir = gens_dict['dataDir']
        self.rho = gens_dict['rho']
        self.alpha = gens_dict['alpha']
        self.beta = gens_dict['beta']
        self.data = gens_dict['data']
   

        if not os.path.exists(self.dataDir + '/1/train.csv') or G:

            print('Next, run dataGenerator: ')
            
            for exp in range(self.num_reps):
                seed = exp * 527 + self.seed
                print(f'Generate {self.data} datasets - {exp}/{self.num_reps}. ')
 
                dataPath = 'Data/Causal/Twins/twins.csv'
                df = pd.read_csv(dataPath)
                df = df[df['dbirwt_1'] < 2000]
                df = df.dropna()
                df_va = df.values[:,4:]
                num_ = len(df)
                data_df = generate_IHDP(num=num_,data=df_va,rho=self.rho, alpha=self.alpha, beta=self.beta, seed=seed)
                train_num = int(num_*0.56)
                valid_num = int(num_*0.24)
                test_num = int(num_*0.2)
                train_df = data_df[:train_num]
                valid_df = data_df[train_num:train_num+valid_num]
                test_df = data_df[train_num+valid_num:]

                data_path = self.dataDir + '/{}/'.format(exp)
                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                
                train_df.to_csv(data_path + '/train.csv', index=False)
                valid_df.to_csv(data_path + '/val.csv', index=False)
                test_df.to_csv(data_path + '/test.csv', index=False)
            
            print('-'*30)

        
    def get_exp(self, exp, num=0):

        subDir = self.dataDir + f'/{exp}/'

        self.train_df = pd.read_csv(subDir+'train.csv')
        self.val_df   = pd.read_csv(subDir+'val.csv')
        self.test_df  = pd.read_csv(subDir+'test.csv')

        if not (num > 0 and num < len(self.train_df)):
            num = len(self.train_df)

        train = CausalDataset(self.train_df[:num], variables = ['u','x','i','z','c','a','t','m','g','v','y','e','b','f','h'])
        val   = CausalDataset(self.val_df[:num],   variables = ['u','x','i','z','c','a','t','m','g','v','y','e','b','f','h'])
        test  = CausalDataset(self.test_df[:num],  variables = ['u','x','i','z','c','a','t','m','g','v','y','e','b','f','h'])

        return Data(train, val, test, num)