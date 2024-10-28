import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools


def log_metrics(Z, T, Y, Z_val, T_val, Y_val, T_test, learner, adversary, epoch, writer, true_of_T=None, true_of_T_test=None, loss='moment', mode='t'):
    y_pred = learner(T)
    y_pred_val = learner(T_val)
    if loss == 'moment':
        writer.add_scalar('moment', torch.mean((Y - y_pred) * adversary(Z)), epoch)

        writer.add_scalar('moment_val', torch.mean((Y_val - y_pred_val) * adversary(Z_val)),epoch)
    if loss == 'kernel':
        psi = (Y - y_pred) / Y.shape[0]
        writer.add_scalar('kernel_loss',
                          (psi.T @ adversary(Z, Z) @ psi)[0][0],
                          epoch)
        psi_val = (Y_val - y_pred_val) / Y_val.shape[0]
        writer.add_scalar('kernel_loss_val',
                          (psi_val.T @ adversary(
                              Z_val, Z_val) @ psi_val)[0][0],
                          epoch)

    try:
        R2train = 1 - np.mean((true_of_T.cpu().numpy().flatten() - y_pred.cpu().data.numpy().flatten())
                            ** 2) / np.var(true_of_T.cpu().numpy().flatten())

        myR2train = 1 - np.mean((true_of_T.cpu().numpy().flatten() - y_pred.cpu().data.numpy().flatten())
                                ** 2) / np.var(Y.cpu().numpy().flatten())

        MSEtrain = np.mean((true_of_T.cpu().numpy().flatten() -
                            y_pred.cpu().data.numpy().flatten())**2)
        writer.add_scalar('MSEtrain', MSEtrain, epoch)

        writer.add_scalar('R2train', R2train, epoch)

        writer.add_scalar('myR2train', myR2train, epoch)
        # select 3 points from the set of test points
        test_points = T_test[[0, 50, 99]]

        if mode == 't':
            learned_function_values = dict(zip(list(test_points[:, 0].cpu().numpy().flatten().astype('str')),
                                                list(learner(test_points).cpu().detach().numpy().flatten())))
        else:
            learned_function_values = dict(zip(list((test_points[:,0:1] * test_points[:,1:2]).cpu().numpy().flatten().astype('str')),
                                            list(learner(test_points).cpu().detach().numpy().flatten())))
        writer.add_scalars('function', learned_function_values, epoch)
    except:
        pass

