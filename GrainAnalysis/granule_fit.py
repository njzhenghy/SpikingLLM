import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import FSNeuronPlus, FSNeuronDecoupled

class _HeavisideSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta: float):
        ctx.save_for_backward(x); ctx.beta = beta
        return (x > 0).to(x.dtype)
    @staticmethod
    def backward(ctx, grad_out):
        # print("here")
        (x,) = ctx.saved_tensors; beta = ctx.beta
        sig = torch.sigmoid(beta * x)
        return grad_out * (beta * sig * (1 - sig)), None


def train_grain_fitting(X, T, num_grains, beta, lr, epoch, genotype=None):
    fs1 = FSNeuronPlus(T=T, num_grains=num_grains, beta=beta, genotype=genotype)
    error_list, y_list = [], []
    optimizer = torch.optim.Adam([fs1.d], lr=lr)
    for i in range(epoch):
        y, S = fs1(X, None, True)
        # l1_reg = 0.01 * torch.norm(S, p=1)
        error = F.mse_loss(y, X, reduction='mean')
        y_list.append(y)
        error_list.append(error.data)
        optimizer.zero_grad()
        error.backward(retain_graph=True)
        optimizer.step()
        print(fs1.get_base_list())
        print(fs1.d.data, error)
        # print(error)
    
    return error_list 


def run_and_plot_experiments(experiments_dict, save_path='./results.png'):
    X, S = None, None
    results = {}
    
    for name, config in experiments_dict.items():
        print(config)
        X = torch.abs(torch.normal(mean=0.0, std=1.0, size=(10000,1))*5)
        S = torch.zeros((config['T'], X.shape[0]))
        results[name] = train_grain_fitting(X, **config)
    
    plt.figure(figsize=(12, 8))
    for name, error_history in results.items():
        config = experiments_dict[name]
        label = f"{name} (T={config['T']}, grains={config['num_grains']})"
        plt.plot(error_history, label=label, linewidth=2)
    
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Decoupled Grain Fitting Experiments Comparison')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.show()
    
    return results

experiments = {
    # 'single_grain_T_4': {'T': 4, 'num_grains': 1, 'beta': 10, 'lr': 1e-3, 'epoch': 1000},
    # 'double_grain_T_4': {'T': 4, 'num_grains': 2, 'beta': 10, 'lr': 1e-3, 'epoch': 1000},
    # 'single_grain_T_8': {'T': 8, 'num_grains': 1, 'beta': 10, 'lr': 1e-3, 'epoch': 1000},
    # 'double_grain_T_8': {'T': 8, 'num_grains': 2, 'beta': 10, 'lr': 1e-3, 'epoch': 1000, 'genotype': [0, 0, 1, 1, 1, 1, 1, 1]},
    # 'double_grain_T_8': {'T': 8, 'num_grains': 2, 'beta': 10, 'lr': 1e-3, 'epoch': 1000, 'genotype': [0, 0, 0, 1, 1, 1, 1, 1]},
    # 'double_grain_T_8': {'T': 8, 'num_grains': 2, 'beta': 10, 'lr': 1e-3, 'epoch': 1000, 'genotype': [0, 0, 0, 0, 1, 1, 1, 1]},
    # 'double_grain_T_8': {'T': 8, 'num_grains': 2, 'beta': 10, 'lr': 1e-3, 'epoch': 1000, 'genotype': [0, 0, 0, 0, 0, 1, 1, 1]},
    # 'double_grain_T_8': {'T': 8, 'num_grains': 2, 'beta': 10, 'lr': 1e-3, 'epoch': 1000, 'genotype': [0, 0, 0, 0, 0, 0, 1, 1]},
    'triple_grain_T_8_[2,2,4]': {'T': 8, 'num_grains': 3, 'beta': 10, 'lr': 1e-3, 'epoch': 1000, 'genotype': [0, 0, 1, 1, 2, 2, 2, 2],},
    'triple_grain_T_8_[2,3,3]': {'T': 8, 'num_grains': 3, 'beta': 10, 'lr': 1e-3, 'epoch': 1000, 'genotype': [0, 0, 1, 1, 1, 2, 2, 2],},
    'triple_grain_T_8_[2,4,2]': {'T': 8, 'num_grains': 3, 'beta': 10, 'lr': 1e-3, 'epoch': 1000, 'genotype': [0, 0, 1, 1, 1, 1, 2, 2],},
    'triple_grain_T_8_[3,2,3]': {'T': 8, 'num_grains': 3, 'beta': 10, 'lr': 1e-3, 'epoch': 1000, 'genotype': [0, 0, 0, 1, 1, 2, 2, 2],},
    'triple_grain_T_8_[3,3,2]': {'T': 8, 'num_grains': 3, 'beta': 10, 'lr': 1e-3, 'epoch': 1000, 'genotype': [0, 0, 0, 1, 1, 1, 2, 2],},
    'triple_grain_T_8_[4,2,2]': {'T': 8, 'num_grains': 3, 'beta': 10, 'lr': 1e-3, 'epoch': 1000, 'genotype': [0, 0, 0, 0, 1, 1, 2, 2],},
    'triple_grain_T_8_[2,2,2,2]': {'T': 8, 'num_grains': 4, 'beta': 10, 'lr': 1e-3, 'epoch': 1000, 'genotype': [0, 0, 1, 1, 2, 2, 3, 3],},
}

all_results = run_and_plot_experiments(experiments, './T=4-8_grain=1-3.png')