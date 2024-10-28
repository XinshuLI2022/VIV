import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from utils import cat


def evaluation(data, estimation, resultDir, method, mode, exp):

    
    mu0_data, mu_data, mut_data = estimation(data)
    try:
        plot_data = cat([data.g, data.v, data.y, mu_data]).detach().cpu().numpy()
        plot_data_t = cat([data.b, data.f, data.h, mut_data]).detach().cpu().numpy()
        plot_data_0 = cat([data.m, mu0_data]).detach().cpu().numpy()
        plots = cat([data.m[:,1:2], data.v, data.f, mu0_data, mu_data,mut_data]).detach().cpu().numpy()
    except:
        plot_data = cat([data.g, data.v, data.y, mu_data])
        plot_data_t = cat([data.b, data.f, data.h, mut_data])
        plot_data_0 = cat([data.m, mu0_data])
        plots = cat([data.m[:,1:2], data.v, data.f, mu0_data, mu_data,mut_data])
    
    mse = ((plot_data - plot_data[:, -1:]) ** 2).mean(0)
    mse_t = ((plot_data_t - plot_data_t[:, -1:]) ** 2).mean(0)
    mse_0 = ((plot_data_0 - plot_data_0[:, -1:]) ** 2).mean(0)
    print("{}-MSE - g(t)-fn(t,x): {:.4f}, f(t,x)-fn(t,x): {:.4f}, f(t,x)+u-fn(t,x):{:.4f}.".format(mode, mse[0],mse[1],mse[2]))
    print("{}-MSE - g(t)-fn(t,x): {:.4f}, f(t,x)-fn(t,x): {:.4f}, f(t,x)+u-fn(t,x):{:.4f}.".format(mode, mse_t[0],mse_t[1],mse_t[2]))
    print("{}-MSE - g(t)-fn(t,x): {:.4f}, f(t,x)-fn(t,x): {:.4f}, f(t,x)+u-fn(t,x):{:.4f}.".format(mode, mse_0[0],mse_0[1],mse_0[2]))


    return cat([mse_0[:3], mse[:3],mse_t[:3]],0).reshape(1,-1)
