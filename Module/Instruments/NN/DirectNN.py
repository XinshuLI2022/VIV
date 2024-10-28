import torch
from torch import nn
from utils import trainEnv, trainParams, Log, cat
import numpy as np
class Trainer(object):
    def __init__(self, data, train_dict, device="cuda:0"):
        data.cuda()
        self.data = data
        self.device = device

        self.z_dim = data.train.z.shape[1] 
        self.x_dim = data.train.x.shape[1] 
        self.t_dim = 1
        self.num_domain = 2
        self.instrumental_weight_decay = 0.0
        self.covariate_weight_decay = 0.0
        self.learning_rate = 0.005
        self.build_net()

        self.verbose = 1
        self.show_per_epoch = 5
        self.lam2 = 0.1
        self.n_epoch = 100
        self.batch_size = 1000

        self.train()

    def build_net(self):
        self.instrumental_net = nn.Sequential(nn.Linear(self.z_dim+self.x_dim, 1280),
                                      nn.ReLU(),
                                      nn.Linear(1280, 320),
                                      nn.BatchNorm1d(320),
                                      nn.ReLU(),
                                      nn.Linear(320, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 1))

        self.covariate_net = nn.Sequential(nn.Linear(self.x_dim+self.t_dim, 1280),
                                      nn.ReLU(),
                                      nn.Linear(1280, 320),
                                      nn.BatchNorm1d(320),
                                      nn.ReLU(),
                                      nn.Linear(320, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 1))

        self.instrumental_net.to(self.device)
        self.covariate_net.to(self.device)

        self.instrumental_opt = torch.optim.Adam(self.instrumental_net.parameters(),lr=self.learning_rate,weight_decay=self.instrumental_weight_decay)
        self.covariate_opt = torch.optim.Adam(self.covariate_net.parameters(),lr=self.learning_rate,weight_decay=self.covariate_weight_decay)

        self.loss_fn4t = torch.nn.MSELoss()
        self.loss_fn4y = torch.nn.MSELoss()

    def train(self, verbose=None, show_per_epoch=None):
        if verbose is None or show_per_epoch is None:
            verbose, show_per_epoch = self.verbose, self.show_per_epoch

        self.lam2 *= self.data.train.length

        for exp in range(self.n_epoch//5):
            self.instrumental_update(self.data.train, verbose)

            if verbose >= 1 and (exp % show_per_epoch == 0 or exp == self.n_epoch - 1):
                print(type(self.data.train.z))
                train_t_hat = self.instrumental_net(cat([self.data.train.x,self.data.train.z])).detach()
                
                loss_train = self.loss_fn4t(train_t_hat, self.data.train.t)

                print("Epoch {} ended: {:.4f}.".format(exp, loss_train))
                
        self.data.train.t = self.instrumental_net(cat([self.data.train.x,self.data.train.z])).detach()

        for exp in range(self.n_epoch):
            self.covariate_update(self.data.train, verbose)

            if verbose >= 1 and (exp % show_per_epoch == 0 or exp == self.n_epoch - 1):
                eval_train = self.evaluate(self.data.train)
                eval_valid = self.evaluate(self.data.valid)
                eval_test  = self.evaluate(self.data.test)

                print(f"Epoch {exp} ended:")
                print(f"Train: {eval_train}. ")
                print(f"Valid: {eval_valid}. ")
                print(f"Test : {eval_test}. ")

    def instrumental_update(self, data, verbose):
        loader = self.data.get_loader({'batch_size':self.batch_size}, data)

        for idx, inputs in enumerate(loader):
            x = inputs['x'].to(self.device)
            t = inputs['t'].to(self.device)
            z = inputs['z'].to(self.device)

            t_hat = self.instrumental_net(cat([x,z]))

            loss = self.loss_fn4t(t_hat, t)

            self.instrumental_opt.zero_grad()
            loss.backward()
            self.instrumental_opt.step()

            if verbose >= 2:
                print('Batch {} - loss: {:.4f}'.format(idx, loss))

    def covariate_update(self, data, verbose):
        loader = self.data.get_loader({'batch_size':self.batch_size}, data)

        for idx, inputs in enumerate(loader):
            x = inputs['x'].to(self.device)
            t = inputs['t'].to(self.device)
            y = inputs['y'].to(self.device)

            y_hat = self.covariate_net(cat([x,t]))

            loss = self.loss_fn4y(y_hat, y)

            self.covariate_opt.zero_grad()
            loss.backward()
            self.covariate_opt.step()

            if verbose >= 2:
                print('Batch {} - loss: {:.4f}'.format(idx, loss))

    def estimation(self, data):
        self.covariate_net.train(False)

        y0_hat = self.covariate_net(cat([data.x,data.t-data.t]))
        y_hat = self.covariate_net(cat([data.x,data.t]))
        yt_hat = self.covariate_net(cat([data.x,data.e]))
        return y0_hat, y_hat,yt_hat

    def evaluate(self, data):
        y0_hat, y_hat,yt_hat = self.estimation(data)

        loss_y = self.loss_fn4y(y_hat, data.y)
        loss_g = self.loss_fn4y(y_hat, data.g)
        loss_v = self.loss_fn4y(y_hat, data.v)
        loss_yt = self.loss_fn4y(yt_hat, data.h)
        loss_vt = self.loss_fn4y(yt_hat, data.f)
        loss_gt = self.loss_fn4y(yt_hat, data.b)
        loss_g0 = self.loss_fn4y(y0_hat, data.m[:,0])
        loss_v0 = self.loss_fn4y(y0_hat, data.m[:,1])
        loss_y0 = self.loss_fn4y(y0_hat, data.m[:,2])

        eval_str = 'loss_y: {:.4f}, loss_g: {:.4f}, loss_v: {:.4f},loss_yt: {:.4f}, loss_gt: {:.4f}, loss_vt: {:.4f},loss_y0: {:.4f}, loss_g0: {:.4f}, loss_v0: {:.4f}'\
            .format(loss_y,loss_g,loss_v,loss_yt,loss_gt,loss_vt,loss_y0,loss_g0,loss_v0)
        return eval_str

def run(exp, data, train_dict, log, device, resultDir, others):

    print(f"Run {exp}/{train_dict['reps']}")
    
    trainer = Trainer(data, train_dict)

    return trainer.estimation