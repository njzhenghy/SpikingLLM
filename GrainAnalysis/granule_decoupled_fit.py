import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import FSNeuronDecoupled

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

    fs_neuron = FSNeuronDecoupled(T=T, num_grains=num_grains, beta=beta, genotype=genotype)
    
    optimizer_d = torch.optim.Adam([fs_neuron.d], lr=lr)
    optimizer_h_theta = torch.optim.Adam([fs_neuron.h, fs_neuron.theta], lr=lr)
    
    error_list= []

    for i in range(epoch):
        mode = "h_theta" if i % 2 == 0 else "d"
        
        if mode == "h_theta":
            optimizer_h_theta.zero_grad()

            fs_neuron.d.requires_grad_(False)
            fs_neuron.h.requires_grad_(True)
            fs_neuron.theta.requires_grad_(True)
            
            y, S = fs_neuron(X, None, True)
            error = F.mse_loss(y, X, reduction='mean')
            error_list.append(error.item())

            error.backward(retain_graph=True)
            optimizer_h_theta.step()

            fs_neuron.d.requires_grad_(True)
            print(f"error={error}, mode={mode}")
            
        if mode == "d":
            optimizer_d.zero_grad()

            fs_neuron.d.requires_grad_(True)
            fs_neuron.h.requires_grad_(False)
            fs_neuron.theta.requires_grad_(False)
            
            y, S = fs_neuron(X, None, True)
            error = F.mse_loss(y, X, reduction='mean')
            error_list.append(error.item())
            
            error.backward(retain_graph=True)
            optimizer_d.step()

            fs_neuron.h.requires_grad_(True)
            fs_neuron.theta.requires_grad_(True)

            print(f"error={error}, mode={mode}")
        
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
    'triple_grain_T_8_[2,2,4]': {'T': 8, 'num_grains': 3, 'beta': 10, 'lr': 5e-3, 'epoch': 2000, 'genotype': [0, 0, 1, 1, 2, 2, 2, 2],},
    'triple_grain_T_8_[2,3,3]': {'T': 8, 'num_grains': 3, 'beta': 10, 'lr': 5e-3, 'epoch': 2000, 'genotype': [0, 0, 1, 1, 1, 2, 2, 2],},
    'triple_grain_T_8_[2,4,2]': {'T': 8, 'num_grains': 3, 'beta': 10, 'lr': 5e-3, 'epoch': 2000, 'genotype': [0, 0, 1, 1, 1, 1, 2, 2],},
    'triple_grain_T_8_[3,2,3]': {'T': 8, 'num_grains': 3, 'beta': 10, 'lr': 5e-3, 'epoch': 2000, 'genotype': [0, 0, 0, 1, 1, 2, 2, 2],},
    'triple_grain_T_8_[3,3,2]': {'T': 8, 'num_grains': 3, 'beta': 10, 'lr': 5e-3, 'epoch': 2000, 'genotype': [0, 0, 0, 1, 1, 1, 2, 2],},
    'triple_grain_T_8_[4,2,2]': {'T': 8, 'num_grains': 3, 'beta': 10, 'lr': 5e-3, 'epoch': 2000, 'genotype': [0, 0, 0, 0, 1, 1, 2, 2],},
    'triple_grain_T_8_[2,2,2,2]': {'T': 8, 'num_grains': 4, 'beta': 10, 'lr': 5e-3, 'epoch': 2000, 'genotype': [0, 0, 1, 1, 2, 2, 3, 3],},
}

all_results = run_and_plot_experiments(experiments, './Decoupled_T=4-8_grain=1-3.png')