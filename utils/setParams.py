import os
import json

def set_dir(data, fn, setting):
    data_setting = f'{data}'
    if not fn == '':
        data_setting = f'{data_setting}/{fn}'
    if not setting == 0:
        data_setting = f'{data_setting}/{setting}'

    targetDir = f'./Data/data/'
    configDir = f'./Data/config/'
    dataDir   = f'./Data/data/{data_setting}/'
    resultDir = f'./Data/results/{data_setting}/'
    evalDir = f'./Data/results/{data_setting}/Eval/'
    logDir = f'./Data/results/{data_setting}/log/'
    os.makedirs(os.path.dirname(targetDir), exist_ok=True)
    os.makedirs(os.path.dirname(dataDir), exist_ok=True)
    os.makedirs(os.path.dirname(configDir), exist_ok=True)
    os.makedirs(os.path.dirname(resultDir), exist_ok=True)
    os.makedirs(os.path.dirname(evalDir), exist_ok=True)
    os.makedirs(os.path.dirname(logDir), exist_ok=True)

    return targetDir, dataDir, configDir, resultDir, evalDir, logDir

class trainParams(object):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.targetDir, self.dataDir, self.configDir, self.resultDir, self.evalDir, self.logDir \
                = set_dir(args.data, args.rho, f'{args.num}_{args.alpha}_{args.beta}')
        self.Params_dict = {}
        self.args = args
        
        self.set_Params(keep_train=False)
        
    def set_Params(self, args=None, keep_train=True):
        if args is None:
            args = self.args
        else:
            self.args = args

        self.args_dict = vars(self.args)  
        self.Params_dict['args']=self.args_dict

        self.set_dirs()
        self.Params_dict['Dirs']=self.Dirs_dict

        self.set_gens()
        self.Params_dict['gens']=self.gens_dict

        self.set_log()
        self.Params_dict['log']=self.log_dict

        if not keep_train:
            self.set_train()
            self.Params_dict['train']=self.train_dict

    def set_train(self):
        self.train_dict = {}

        self.train_dict['deepgmm'] = {}
        self.train_dict['deepgmm']['reps'] = self.Params_dict['args']['reps']
        self.train_dict['deepgmm']['seed'] = self.Params_dict['args']['seed']
        self.train_dict['deepgmm']['x_dim'] = self.Params_dict['args']['mX']
        self.train_dict['deepgmm']['t_dim'] = 1
        self.train_dict['deepgmm']['covariate_weight_decay'] = 0.0
        self.train_dict['deepgmm']['instrumental_weight_decay'] = 0.0
        self.train_dict['deepgmm']['learning_rate'] = 0.005
        self.train_dict['deepgmm']['verbose'] = 1
        self.train_dict['deepgmm']['show_per_epoch'] = 5
        self.train_dict['deepgmm']['lam2'] = 0.1
        self.train_dict['deepgmm']['epochs'] = 100
        self.train_dict['deepgmm']['batch_size']= 1000

        self.train_dict['agmm'] = {}
        self.train_dict['agmm']['reps'] = self.Params_dict['args']['reps']
        self.train_dict['agmm']['seed'] = self.Params_dict['args']['seed']
        self.train_dict['agmm']['u_dim'] = 1
        self.train_dict['agmm']['x_dim'] = self.Params_dict['args']['mX']
        self.train_dict['agmm']['t_dim'] = 1
        self.train_dict['agmm']['covariate_weight_decay'] = 0.0
        self.train_dict['agmm']['instrumental_weight_decay'] = 0.0
        self.train_dict['agmm']['learning_rate'] = 0.005
        self.train_dict['agmm']['verbose'] = 1
        self.train_dict['agmm']['show_per_epoch'] = 5
        self.train_dict['agmm']['lam2'] = 0.1
        self.train_dict['agmm']['epochs'] = 100
        self.train_dict['agmm']['batch_size']= 1000

        self.train_dict['onesiv'] = {}
        self.train_dict['onesiv']['reps'] = self.Params_dict['args']['reps']
        self.train_dict['onesiv']['seed'] = self.Params_dict['args']['seed']
        self.train_dict['onesiv']['dropout'] = 0.5
        self.train_dict['onesiv']['x_dim'] = self.Params_dict['args']['mX']
        self.train_dict['onesiv']['t_dim'] = 1
        self.train_dict['onesiv']['learning_rate'] = 0.005
        self.train_dict['onesiv']['beta1'] = 0.9
        self.train_dict['onesiv']['beta2'] = 0.999
        self.train_dict['onesiv']['eps']   = 1e-8
        self.train_dict['onesiv']['w1'] = 0.0017
        self.train_dict['onesiv']['w2'] = 1.0
        self.train_dict['onesiv']['verbose'] = 1
        self.train_dict['onesiv']['show_per_epoch'] = 10
        self.train_dict['onesiv']['epochs'] = 30
        self.train_dict['onesiv']['batch_size']= 1000

        self.train_dict['nn'] = {}
        self.train_dict['nn']['reps'] = self.Params_dict['args']['reps']
        self.train_dict['nn']['seed'] = self.Params_dict['args']['seed']
        self.train_dict['nn']['u_dim'] = 1
        self.train_dict['nn']['x_dim'] = self.Params_dict['args']['mX']
        self.train_dict['nn']['t_dim'] = 1
        self.train_dict['nn']['covariate_weight_decay'] = 0.0
        self.train_dict['nn']['instrumental_weight_decay'] = 0.0
        self.train_dict['nn']['learning_rate'] = 0.005
        self.train_dict['nn']['verbose'] = 1
        self.train_dict['nn']['show_per_epoch'] = 5
        self.train_dict['nn']['lam2'] = 0.1
        self.train_dict['nn']['epochs'] = 100
        self.train_dict['nn']['batch_size']= 1000

        self.train_dict['deepiv'] = {}
        self.train_dict['deepiv']['seed'] = self.Params_dict['args']['seed']
        self.train_dict['deepiv']['num'] = self.Params_dict['args']['num']
        self.train_dict['deepiv']['n_components'] = 10
        self.train_dict['deepiv']['reps'] = self.Params_dict['args']['reps']
        self.train_dict['deepiv']['epochs'] = self.Params_dict['args']['epochs']
        self.train_dict['deepiv']['batch_size'] = self.Params_dict['args']['batch_size']
        self.train_dict['deepiv']['dropout'] = self.Params_dict['args']['dropout']
        self.train_dict['deepiv']['layers'] = self.Params_dict['args']['layers']
        self.train_dict['deepiv']['activation'] = self.Params_dict['args']['activation']
        self.train_dict['deepiv']['t_loss'] = 'mixture_of_gaussians'
        self.train_dict['deepiv']['y_loss'] = 'mse'
        self.train_dict['deepiv']['samples_per_batch'] = 2
        self.train_dict['deepiv']['x_dim'] = self.Params_dict['args']['mX']

        self.train_dict['dfiv'] = {}
        self.train_dict['dfiv']['seed'] = self.Params_dict['args']['seed']
        self.train_dict['dfiv']['num'] = self.Params_dict['args']['num']
        self.train_dict['dfiv']['reps'] = self.Params_dict['args']['reps']
        self.train_dict['dfiv']['epochs'] = self.Params_dict['args']['epochs']
        self.train_dict['dfiv']['batch_size'] = self.Params_dict['args']['batch_size']
        self.train_dict['dfiv']['dropout'] = self.Params_dict['args']['dropout']
        self.train_dict['dfiv']['GPU'] = self.Params_dict['args']['GPU']
        self.train_dict['dfiv']['t_loss'] = 'mse'
        self.train_dict['dfiv']['y_loss'] = 'mse'
        self.train_dict['dfiv']['intercept'] = True
        self.train_dict['dfiv']['split_ratio'] = 0.5
        self.train_dict['dfiv']['lam1'] = 0.1
        self.train_dict['dfiv']['lam2'] = 0.1
        self.train_dict['dfiv']['stage1_iter'] = 20
        self.train_dict['dfiv']['stage2_iter'] = 1
        self.train_dict['dfiv']['covariate_iter'] = 20
        self.train_dict['dfiv']['treatment_weight_decay'] = 0.0
        self.train_dict['dfiv']['instrumental_weight_decay'] = 0.0
        self.train_dict['dfiv']['covariate_weight_decay'] = 0.1
        self.train_dict['dfiv']['verbose'] = 1
        self.train_dict['dfiv']['show_per_epoch'] = 5
        self.train_dict['dfiv']['t_dim'] = 1
        self.train_dict['dfiv']['x_dim'] = self.Params_dict['args']['mX']\

        return self.train_dict
    
    def get_train(self):
        self.train_dict = self.Params_dict['train']
        return self.train_dict

    def set_log(self):
        self.log_dict = {}
        self.log_dict['logDir'] = self.logDir
        self.log_dict['clear'] = self.Params_dict['args']['clear']
        return self.log_dict

    def get_log(self):
        self.log_dict = self.Params_dict['log']
        return self.log_dict

    def set_gens(self):

        self.gens_dict = {}
        

        
        if self.Params_dict['args']['data'] == "demand" or "IHDP" or "Twins":
            
            self.gens_dict['num'] = self.Params_dict['args']['num']
            self.gens_dict['data'] = self.Params_dict['args']['data']
            self.gens_dict['reps'] = self.Params_dict['args']['reps']
            self.gens_dict['seed'] = self.Params_dict['args']['seed']
            self.gens_dict['rho'] = self.Params_dict['args']['rho']
            self.gens_dict['alpha'] = self.Params_dict['args']['alpha']
            self.gens_dict['beta'] = self.Params_dict['args']['beta']
            self.gens_dict['dataDir'] = self.dataDir
    
        else:
            self.gens_dict['data'] = self.Params_dict['args']['data']
            self.gens_dict['fn'] = self.Params_dict['args']['fn']
            self.gens_dict['num'] = self.Params_dict['args']['num']
            self.gens_dict['seed'] = self.Params_dict['args']['seed']
            self.gens_dict['reps'] = self.Params_dict['args']['reps']
            self.gens_dict['mX'] = self.Params_dict['args']['mX']
            self.gens_dict['x_fn'] = self.Params_dict['args']['x_fn']
            self.gens_dict['y_fn'] = self.Params_dict['args']['y_fn']
            self.gens_dict['dataDir'] = self.dataDir
            self.gens_dict['mV'] = self.Params_dict['args']['mV']
            self.gens_dict['mZ'] = self.Params_dict['args']['mZ']
            self.gens_dict['mC'] = self.Params_dict['args']['mC']
            self.gens_dict['mA'] = self.Params_dict['args']['mA']
            self.gens_dict['rep_dim'] = self.Params_dict['args']['rep_dim']
            self.gens_dict['mU'] = self.Params_dict['args']['mU']
            self.gens_dict['VX'] = self.Params_dict['args']['VX']
            self.gens_dict['one'] = self.Params_dict['args']['one']
        return self.gens_dict

    def get_gens(self):
        self.gens_dict = self.Params_dict['gens']
        return self.gens_dict

    def set_dirs(self):
        self.Dirs_dict = {}
        self.Dirs_dict['targetDir'] = self.targetDir
        self.Dirs_dict['dataDir'] = self.dataDir
        self.Dirs_dict['configDir'] = self.configDir
        self.Dirs_dict['resultDir'] = self.resultDir
        self.Dirs_dict['evalDir'] = self.evalDir
        self.Dirs_dict['logDir'] = self.logDir

        return self.Dirs_dict

    def get_dirs(self):
        self.Dirs_dict = self.Params_dict['Dirs']
        self.targetDir = self.Dirs_dict['targetDir']
        self.dataDir = self.Dirs_dict['dataDir']
        self.configDir = self.Dirs_dict['configDir']
        self.resultDir = self.Dirs_dict['resultDir']
        self.evalDir = self.Dirs_dict['evalDir']
        self.logDir = self.Dirs_dict['logDir']
        return self.Dirs_dict

    def get_args(self):
        return self.args_dict
    
    def load_json(self, file):
        if not file.endswith('.json'):
            file = file + '.json'
        with open(f'{self.configDir}{file}', 'r') as f:
            self.Params_dict = json.load(f)

        self.get_dirs()
        self.get_args()
        self.get_gens()
        self.get_log()

    def save_json(self, file=None):
        self.set_Params()

        if file is None:
            file = '{}_{}_{}'.format(self.Params_dict['args']['data'], self.Params_dict['args']['rho'], self.Params_dict['args']['num'])    
        if not file.endswith('.json'):
            file = file + '.json'
        json_data = json.dumps(self.Params_dict,indent=4)
        with open(f'{self.configDir}{file}', 'w') as f:
            f.write(json_data)
    