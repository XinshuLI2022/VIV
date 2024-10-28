import torch
from torch import nn
import numpy as np
from utils import set_seed, cat

############# Define Networks ################
def build_net(t_input_dim, z_input_dim, x_input_dim):
    treatment_net = nn.Sequential(nn.Linear(t_input_dim, 16),
                                  nn.ReLU(),
                                  nn.Linear(16, 1))

    instrumental_net = nn.Sequential(nn.Linear(z_input_dim+x_input_dim, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.BatchNorm1d(32))

    covariate_net = nn.Sequential(nn.Linear(x_input_dim, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, 32),
                                  nn.BatchNorm1d(32),
                                  nn.ReLU(),
                                  nn.Linear(32, 16),
                                  nn.ReLU())
    
    return treatment_net, instrumental_net, covariate_net

################## Define loss #################################
def fit_linear(target: torch.Tensor, feature: torch.Tensor, reg: float = 0.0):
    nData, nDim = feature.size()
    A = torch.matmul(feature.t(), feature)
    device = feature.device
    A = A + reg * torch.eye(nDim, device=device)
    A_inv = torch.inverse(A)
    if target.dim() == 2:
        b = torch.matmul(feature.t(), target)
        weight = torch.matmul(A_inv, b)
    else:
        b = torch.einsum("nd,n...->d...", feature, target)
        weight = torch.einsum("de,d...->e...", A_inv, b)
    return weight

def linear_reg_pred(feature: torch.Tensor, weight: torch.Tensor):
    if weight.dim() == 2:
        return torch.matmul(feature, weight)
    else:
        return torch.einsum("nd,d...->n...", feature, weight)

def linear_reg_loss(target: torch.Tensor, feature: torch.Tensor, reg: float):
    weight = fit_linear(target, feature, reg)
    pred = linear_reg_pred(feature, weight)
    return torch.norm((target - pred)) ** 2 + reg * torch.norm(weight) ** 2

############ Define Utils #####################
def add_const_col(mat: torch.Tensor):
    n_data = mat.size()[0]
    device = mat.device
    return torch.cat([mat, torch.ones((n_data, 1), device=device)], dim=1)

def augment_z_feature(feature, add_intercept):
    if add_intercept: feature = add_const_col(feature)
    return feature

def augment_tx_feature(feature, feature_tmp, add_intercept):
    if add_intercept: feature = add_const_col(feature)
    if add_intercept: feature_tmp = add_const_col(feature_tmp)
    feature = outer_prod(feature, feature_tmp)
    feature = torch.flatten(feature, start_dim=1)
    return feature

def outer_prod(mat1: torch.Tensor, mat2: torch.Tensor):
    mat1_shape = tuple(mat1.size())
    mat2_shape = tuple(mat2.size())
    assert mat1_shape[0] == mat2_shape[0]
    nData = mat1_shape[0]
    aug_mat1_shape = mat1_shape + (1,) * (len(mat2_shape) - 1)
    aug_mat1 = torch.reshape(mat1, aug_mat1_shape)
    aug_mat2_shape = (nData,) + (1,) * (len(mat1_shape) - 1) + mat2_shape[1:]
    aug_mat2 = torch.reshape(mat2, aug_mat2_shape)
    return aug_mat1 * aug_mat2

class DFIVTrainer(object):

    def __init__(self, data, train_dict, device="cuda:0"):

        data.cuda()
        data.split(train_dict["split_ratio"])
        self.data = data
        
        self.t_loss = train_dict['t_loss']
        self.y_loss = train_dict['y_loss']
        self.gpu_flg = train_dict['GPU'] and torch.cuda.is_available()
        self.add_intercept = train_dict['intercept']
        self.n_epoch = train_dict["epochs"]
        self.lam1 = train_dict["lam1"]
        self.lam2 = train_dict["lam2"]
        self.stage1_iter = train_dict["stage1_iter"]
        self.stage2_iter = train_dict["stage2_iter"]
        self.covariate_iter = train_dict["covariate_iter"]
        self.split_ratio = train_dict["split_ratio"]
        self.treatment_weight_decay = train_dict["treatment_weight_decay"]
        self.instrumental_weight_decay = train_dict["instrumental_weight_decay"]
        self.covariate_weight_decay = train_dict["covariate_weight_decay"]
        self.verbose = train_dict["verbose"]
        self.show_per_epoch = train_dict["show_per_epoch"]

        self.treatment_net, self.instrumental_net, self.covariate_net = build_net(train_dict['t_dim'], data.train.z.shape[1], data.train.x.shape[1])
        if self.gpu_flg:
            self.treatment_net.to(device)
            self.instrumental_net.to(device)
            self.covariate_net.to(device)
        self.treatment_opt = torch.optim.Adam(self.treatment_net.parameters(),weight_decay=self.treatment_weight_decay)
        self.instrumental_opt = torch.optim.Adam(self.instrumental_net.parameters(),weight_decay=self.instrumental_weight_decay)
        self.covariate_opt = torch.optim.Adam(self.covariate_net.parameters(),weight_decay=self.covariate_weight_decay)
    
    def train(self, verbose=None, show_per_epoch=None):
        if verbose is None or show_per_epoch is None:
            verbose, show_per_epoch = self.verbose, self.show_per_epoch

        self.lam1 *= self.data.data1.length
        self.lam2 *= self.data.data2.length

        for exp in range(self.n_epoch):
            self.stage1_update(self.data.data1, verbose)
            self.covariate_update(self.data.data1, self.data.data2, verbose)
            self.stage2_update(self.data.data1, self.data.data2, verbose)
            if exp % show_per_epoch == 0 or exp == self.n_epoch - 1:
                if verbose >= 1: 
                    _,pred_tx2y,_ = self.estimation4tx(self.data.valid)
                    mse_y = ((pred_tx2y - self.data.valid.y) ** 2).mean()
                    mse_g = ((pred_tx2y - self.data.valid.g) ** 2).mean()
                    print(f"Epoch {exp} ended: {mse_y}, {mse_g}. ")

    def stage1_update(self, train_1st, verbose):
        self.instrumental_net.train(True)
        self.treatment_net.train(False)
        self.covariate_net.train(False)

        treatment_feature = self.treatment_net(train_1st.t).detach()
        for i in range(self.stage1_iter):
            self.instrumental_opt.zero_grad()
            instrumental_feature = self.instrumental_net(cat([train_1st.z,train_1st.x]))
            feature = augment_z_feature(instrumental_feature, self.add_intercept)
            loss = linear_reg_loss(treatment_feature, feature, self.lam1)
            loss.backward()
            if verbose >= 2: print(f"stage1 learning: {loss.item()}")
            self.instrumental_opt.step()

    def covariate_update(self, train_1st, train_2nd, verbose):
        self.instrumental_net.train(False)
        self.treatment_net.train(False)
        self.covariate_net.train(True)

        instrumental_1st_feature = self.instrumental_net(cat([train_1st.z,train_1st.x])).detach()
        instrumental_2nd_feature = self.instrumental_net(cat([train_2nd.z,train_2nd.x])).detach()
        treatment_1st_feature = self.treatment_net(train_1st.t).detach()

        feature_1st = augment_z_feature(instrumental_1st_feature, self.add_intercept)
        feature_2nd = augment_z_feature(instrumental_2nd_feature, self.add_intercept)
        self.stage1_weight = fit_linear(treatment_1st_feature, feature_1st, self.lam1)
        predicted_treatment_feature_2nd = linear_reg_pred(feature_2nd, self.stage1_weight).detach()

        for i in range(self.covariate_iter):
            self.covariate_opt.zero_grad()
            covariate_feature = self.covariate_net(train_2nd.x)
            feature = augment_tx_feature(predicted_treatment_feature_2nd, covariate_feature, self.add_intercept)
            loss = linear_reg_loss(train_2nd.y, feature, self.lam2)
            loss.backward()
            if verbose >= 2: print(f"update covariate: {loss.item()}")
            self.covariate_opt.step()

    def stage2_update(self, train_1st, train_2nd, verbose):
        self.instrumental_net.train(False)
        self.treatment_net.train(True)
        self.covariate_net.train(False)
        
        instrumental_1st_feature = self.instrumental_net(cat([train_1st.z,train_1st.x])).detach()
        instrumental_2nd_feature = self.instrumental_net(cat([train_2nd.z,train_2nd.x])).detach()
        covariate_2nd_feature = self.covariate_net(train_2nd.x).detach()

        for i in range(self.stage2_iter):
            self.treatment_opt.zero_grad()
            treatment_1st_feature = self.treatment_net(train_1st.t)

            feature_1st = augment_z_feature(instrumental_1st_feature, self.add_intercept)
            feature_2nd = augment_z_feature(instrumental_2nd_feature, self.add_intercept)
            self.stage1_weight = fit_linear(treatment_1st_feature, feature_1st, self.lam1)
            predicted_treatment_feature = linear_reg_pred(feature_2nd, self.stage1_weight)

            feature = augment_tx_feature(predicted_treatment_feature, covariate_2nd_feature, self.add_intercept)
            self.stage2_weight = fit_linear(train_2nd.y, feature, self.lam2)
            pred = linear_reg_pred(feature, self.stage2_weight)
            loss = torch.norm((train_2nd.y - pred)) ** 2 + self.lam2 * torch.norm(self.stage2_weight) ** 2

            loss.backward()
            if verbose >= 2: print(f"stage2 learning: {loss.item()}")
            self.treatment_opt.step()

    def estimation4tx(self, data, update_weight1=False, update_weight2=False):
        self.instrumental_net.train(False)
        self.treatment_net.train(False)
        self.covariate_net.train(False)

        # instrumental_feature = self.instrumental_net(cat([data.z,data.x])).detach()
        treatment_feature = self.treatment_net(data.t).detach()
        treatment_feature_0 = self.treatment_net(data.t-data.t).detach()
        treatment_feature_t = self.treatment_net(data.e).detach()
        covariate_feature = self.covariate_net(data.x).detach()

        # feature_stage1 = augment_z_feature(instrumental_feature, self.add_intercept)
        # if update_weight1: self.stage1_weight = fit_linear(treatment_feature, feature_stage1, self.lam1)
        # predicted_treatment_feature = linear_reg_pred(feature_stage1, self.stage1_weight)

        feature_stage2_tx2y = augment_tx_feature(treatment_feature, covariate_feature, self.add_intercept)
        if update_weight2: self.stage2_weight = fit_linear(data.y, feature_stage2_tx2y, self.lam2)
        pred_tx2y = linear_reg_pred(feature_stage2_tx2y, self.stage2_weight)

        feature_stage2_0x2y = augment_tx_feature(treatment_feature_0, covariate_feature, self.add_intercept)
        if update_weight2: self.stage2_weight = fit_linear(data.y, feature_stage2_0x2y, self.lam2)
        pred_0x2y = linear_reg_pred(feature_stage2_0x2y, self.stage2_weight)

        feature_stage2_ttx2y = augment_tx_feature(treatment_feature_t, covariate_feature, self.add_intercept)
        if update_weight2: self.stage2_weight = fit_linear(data.y, feature_stage2_ttx2y, self.lam2)
        pred_ttx2y = linear_reg_pred(feature_stage2_ttx2y, self.stage2_weight)

        return pred_0x2y, pred_tx2y, pred_ttx2y

    def estimation4zx(self, data, update_weight1=False, update_weight2=False):
        self.instrumental_net.train(False)
        self.treatment_net.train(False)
        self.covariate_net.train(False)

        instrumental_feature = self.instrumental_net(cat([data.z,data.x])).detach()
        treatment_feature = self.treatment_net(data.t).detach()
        covariate_feature = self.covariate_net(data.x).detach()

        feature_stage1 = augment_z_feature(instrumental_feature, self.add_intercept)
        if update_weight1: self.stage1_weight = fit_linear(treatment_feature, feature_stage1, self.lam1)
        predicted_treatment_feature = linear_reg_pred(feature_stage1, self.stage1_weight)

        feature_stage2_zx2y = augment_tx_feature(predicted_treatment_feature, covariate_feature, self.add_intercept)
        if update_weight2: self.stage2_weight = fit_linear(data.y, feature_stage2_zx2y, self.lam2)
        pred_zx2y = linear_reg_pred(feature_stage2_zx2y, self.stage2_weight)

        return pred_zx2y

def run(exp, data, train_dict, log, device, resultDir, others):
    set_seed(train_dict['seed'])
    print(f"Run {exp}/{train_dict['reps']}")
    
    trainer = DFIVTrainer(data, train_dict)
    trainer.train()

    return trainer.estimation4tx