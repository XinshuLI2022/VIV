import torch
from torch import nn
import numpy as np
from .net import AGMM
from utils import log_metrics, set_seed, cat

# def logger(learner, adversary, epoch, writer, Z_train, T_train, Y_train, Z_val, T_val, Y_val, T_test_tens, G_val):
#     writer.add_histogram('learner', learner[-1].weight, epoch)
#     writer.add_histogram('adversary', adversary[-1].weight, epoch)
#     log_metrics(Z_train, T_train, Y_train, Z_val, T_val, Y_val, T_test_tens, learner, adversary, epoch, writer, true_of_T=G_val, mode='tx')

def run(exp, data, train_dict, log, device, resultDir, others):
    set_seed(train_dict['seed'])
    print(f"Run {exp}/{train_dict['reps']}")

    n_x = data.train.x.shape[1]
    n_z = data.train.z.shape[1]
    n_t = train_dict['t_dim']
    p = 0.1 # dropout prob of dropout layers throughout notebook
    n_hidden = 100 # width of hidden layers throughout notebook
    g_features = 100

    learner = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t+n_x, n_hidden), nn.LeakyReLU(),
                            nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.ReLU(),
                            nn.Dropout(p=p), nn.Linear(n_hidden, 1))

    # For any method that uses an unstructured adversary test function f(z)
    adversary_fn = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_z+n_x, n_hidden), nn.LeakyReLU(),
                                nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.ReLU(),
                                nn.Dropout(p=p), nn.Linear(n_hidden, 1))

    ############################### define agmm. #########################################
    learner_lr = 1e-4
    adversary_lr = 1e-4
    learner_l2 = 1e-3
    adversary_l2 = 1e-4
    adversary_norm_reg = 1e-3
    n_epochs = 300
    bs = 100
    sigma = 2.0/g_features
    n_centers = 100

    data.numpy()

    Z_train, T_train, Y_train, G_train = map(lambda x: torch.Tensor(x).to(device), (np.concatenate([data.train.z, data.train.x],1), np.concatenate([data.train.t, data.train.x],1), data.train.y, data.train.g))
    # Z_val, T_val, Y_val, G_val = map(lambda x: torch.Tensor(x).to(device), (np.concatenate([data.valid.z, data.valid.x],1), np.concatenate([data.valid.t, data.valid.x],1), data.valid.y, data.valid.g))
    T_test_tens = torch.Tensor(np.concatenate([data.test.t, data.test.x],1)).to(device)
    G_test_tens = torch.Tensor(data.test.g).to(device)

    agmm = AGMM(learner, adversary_fn).fit(Z_train, T_train, Y_train, T_test_tens, 
                                            learner_lr=learner_lr, adversary_lr=adversary_lr,
                                            learner_l2=learner_l2, adversary_l2=adversary_l2,
                                            n_epochs=n_epochs, bs=bs, 
                                            results_dir=resultDir, device=device, verbose=0)

    def estimation(data, model='final'):
        input0 = torch.Tensor(np.concatenate([data.t-data.t, data.x],1)).to(device)
        point0 = agmm.predict(input0, model=model)

        input = torch.Tensor(np.concatenate([data.t, data.x],1)).to(device)
        point = agmm.predict(input, model=model)

        inputt = torch.Tensor(np.concatenate([data.e, data.x],1)).to(device)
        pointt = agmm.predict(inputt, model=model)

        return point0, point,pointt

    return estimation