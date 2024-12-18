from typing import Dict, Any, Optional, List
import torch
from torch import nn
import logging
from pathlib import Path
import copy

from sklearn.model_selection import train_test_split
import numpy as np

from .model import DeepGMMModel
from .dataClass import TrainDataSet, TrainDataSetTorch, TestDataSetTorch, TestDataSet
from utils import set_seed, cat

logger = logging.getLogger()

def build_net_for_demand(z_dim, x_dim, t_dim):
    response_net = nn.Sequential(nn.Linear(t_dim + x_dim, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, 1))

    dual_net = nn.Sequential(nn.Linear(z_dim + x_dim, 128),
                             nn.ReLU(),
                             nn.Linear(128, 64),
                             nn.ReLU(),
                             nn.Linear(64, 1))

    return response_net, dual_net

class DeepGMMTrainer(object):

    def __init__(self, data_list: List, net_list: List, train_params: Dict[str, Any],
                 gpu_flg: bool = False, dump_folder: Optional[Path] = None):
        self.data_list = data_list
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        if self.gpu_flg:
            logger.info("gpu mode")
        # configure training params
        self.dual_iter: int = train_params["dual_iter"]
        self.primal_iter: int = train_params["primal_iter"]
        self.n_epoch: int = train_params["n_epoch"]

        # build networks
        networks = net_list
        self.primal_net: nn.Module = networks[0]
        self.dual_net: nn.Module = networks[1]
        self.primal_weight_decay = train_params["primal_weight_decay"]
        self.dual_weight_decay = train_params["dual_weight_decay"]

        if self.gpu_flg:
            self.primal_net.to("cuda:0")
            self.dual_net.to("cuda:0")

        self.primal_opt = torch.optim.Adam(self.primal_net.parameters(),
                                           weight_decay=self.primal_weight_decay,
                                           lr=0.0005, betas=(0.5, 0.9))
        self.dual_opt = torch.optim.Adam(self.dual_net.parameters(),
                                         weight_decay=self.dual_weight_decay,
                                         lr=0.0025, betas=(0.5, 0.9))

        # build monitor
        self.monitor = None

    def train(self, rand_seed: int = 42, verbose: int = 0, epoch_show: int = 20) -> float:
        """

        Parameters
        ----------
        rand_seed: int
            random seed
        verbose : int
            Determine the level of logging
        Returns
        -------
        oos_result : float
            The performance of model evaluated by oos
        """
        train_data = self.data_list[0]
        test_data = self.data_list[2]
        if train_data.covariate is not None:
            train_data = TrainDataSet(treatment=np.concatenate([train_data.treatment, train_data.covariate], axis=1),
                                      structural=train_data.structural,
                                      covariate=None,
                                      instrumental=train_data.instrumental,
                                      outcome=train_data.outcome)
            test_data = TestDataSet(treatment=np.concatenate([test_data.treatment, test_data.covariate], axis=1),
                                     covariate=None,
                                     structural=test_data.structural)

        train_data_t = TrainDataSetTorch.from_numpy(train_data)
        test_data_t = TestDataSetTorch.from_numpy(test_data)
        if self.gpu_flg:
            train_data_t = train_data_t.to_gpu()
            test_data_t = test_data_t.to_gpu()

        for t in range(self.n_epoch):
            self.dual_update(train_data_t, verbose)
            self.primal_update(train_data_t, verbose)
            if t % epoch_show == 0 or t == self.n_epoch - 1:
                print(f"Epoch {t} ended")
                if verbose >= 1:
                    logger.info(f"Epoch {t} ended")
                    mdl = DeepGMMModel(self.primal_net, self.dual_net)
                    logger.info(f"test error {mdl.evaluate_t(test_data_t).data.item()}")

        mdl = DeepGMMModel(self.primal_net, self.dual_net)
        oos_loss: float = mdl.evaluate_t(test_data_t).data.item()
        logger.info(f"test_loss:{oos_loss}")
        logger.info(f"target variance: {np.var(train_data.outcome)}")
        return oos_loss

    def dual_update(self, train_data_t: TrainDataSetTorch, verbose: int):
        self.dual_net.train(True)
        self.primal_net.train(False)
        with torch.no_grad():
            epsilon = train_data_t.outcome - self.primal_net(train_data_t.treatment)
        for t in range(self.dual_iter):
            self.dual_opt.zero_grad()
            moment = torch.mean(self.dual_net(train_data_t.instrumental) * epsilon)
            reg = 0.25 * torch.mean((self.dual_net(train_data_t.instrumental) * epsilon) ** 2)
            loss = -moment + reg
            if verbose >= 2:
                logger.info(f"dual loss:{loss.data.item()}")
            loss.backward()
            self.dual_opt.step()

    def primal_update(self, train_data_t: TrainDataSetTorch, verbose: int):
        self.dual_net.train(False)
        self.primal_net.train(True)
        with torch.no_grad():
            dual = self.dual_net(train_data_t.instrumental)
        for t in range(self.primal_iter):
            self.primal_opt.zero_grad()
            epsilon = train_data_t.outcome - self.primal_net(train_data_t.treatment)
            loss = torch.mean(dual * epsilon)
            if verbose >= 2:
                logger.info(f"primal loss:{loss.data.item()}")
            loss.backward()
            self.primal_opt.step()

def run(exp, data, train_dict, log, device, resultDir, others):
    set_seed(train_dict['seed'])
    print(f"Run {exp}/{train_dict['reps']}")

    data.numpy()

    response_net, dual_net = build_net_for_demand(data.train.z.shape[1],data.train.x.shape[1],train_dict['t_dim'])
    net_list = [response_net, dual_net]

    train_data = TrainDataSet(treatment=np.concatenate([data.train.t, data.train.x],1),
                                instrumental=np.concatenate([data.train.z, data.train.x],1),
                                covariate=None,
                                outcome=data.train.y,
                                structural=data.train.v)

    val_data = TrainDataSet(treatment=np.concatenate([data.valid.t, data.valid.x],1),
                            instrumental=None,
                            covariate=None,
                            outcome=data.valid.y,
                            structural=data.valid.v)

    test_data = TestDataSet(treatment=np.concatenate([data.test.t, data.test.x],1),
                            instrumental=None,
                            covariate=None,
                            outcome=None,
                            structural=data.test.v)

    data_list = [train_data, val_data, test_data]

    train_config = {"primal_iter": 1, 
                    "dual_iter": 5, 
                    "n_epoch": 300, 
                    "primal_weight_decay": 0.0, 
                    "dual_weight_decay": 0.0}

    use_gpu = False
    one_dump_dir = resultDir

    trainer = DeepGMMTrainer(data_list, net_list, train_config, use_gpu, one_dump_dir)
    test_loss = trainer.train(rand_seed=42, verbose=1, epoch_show=5)

    def estimation(data):
        input0 = torch.Tensor(np.concatenate([data.t-data.t, data.x],1))
        point0 = response_net(input0).detach().numpy()

        input = torch.Tensor(np.concatenate([data.t, data.x],1))
        point = response_net(input).detach().numpy()

        inputt = torch.Tensor(np.concatenate([data.e, data.x],1))
        pointt = response_net(inputt).detach().numpy()

        return point0, point,pointt

    return estimation