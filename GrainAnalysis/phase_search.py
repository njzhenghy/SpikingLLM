import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, os
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse as parser
from utils import FSNeuronDecoupled, FSNeuronDecoupledSoftMax
import torch.optim as optim
import argparse, sys 
from typing import List

parser = argparse.ArgumentParser(description='Phase Search Training')
parser.add_argument('--epoch_outer', type=int, default=20, help='Number of epochs to train')
parser.add_argument('--epoch_inner', type=int, default=50, help='Number of epochs to train')
parser.add_argument('--T', type=int, default=8)
parser.add_argument('--num_grains', type=int, default=3)
parser.add_argument('--log_interval', type=int, default=1)
parser.add_argument('--beta', type=float, default=10.)
parser.add_argument('--step_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--start', type=int, default=-1)
parser.add_argument('--end', type=int, default=32)
parser.add_argument('--model_name', type=str, default='Llama-2-7B-hf')

args = parser.parse_args()

class PhaseSearch(nn.Module):
    def __init__(self, X, T: int, num_grains: int = 3, beta : float = 1.0, epoch_inner : int = 1000, device : str = 'cuda:0', tau = None, mode = None):
        super().__init__()
        self.T = T
        self.num_grains = num_grains
        self.genotype = self.__generate_genotype(T, num_grains)
        self.num_edges = len(self.genotype)
        self.beta = beta
        self.epoch = epoch_inner
        self.device = device
        self.fs_neurons = nn.ModuleList()
        self.tau = tau
        self.mode = mode
        self.architecture = self.__get_architecture(self.genotype)
        if self.mode == None:
            for i in range(self.num_edges):
                fs_neuron = FSNeuronDecoupled(T=self.T, num_grains=self.num_grains,
                                beta=self.beta, genotype=self.genotype[i], tau=self.tau)
                fs_neuron.to(device)
                self.fs_neurons.append(fs_neuron)

        elif self.mode == "SoftMax":
            for i in range(self.num_edges):
                fs_neuron = FSNeuronDecoupledSoftMax(T=self.T, num_grains=self.num_grains,
                                beta=self.beta, genotype=self.genotype[i], tau=self.tau)
                fs_neuron.to(device)
                self.fs_neurons.append(fs_neuron)


    def get_params(self):
        if self.mode == "SoftMax":
            h_params, theta_params, d_params, v0_params, arch_params = [], [], [], [], list(self.architecture)
            
            for fs_neuron in self.fs_neurons:
                h_params.append(fs_neuron.h)
                theta_params.append(fs_neuron.theta)
                d_params.append(fs_neuron.d)
                v0_params.append(fs_neuron.v0)
            
            return h_params, theta_params, d_params, v0_params, arch_params
        else:
            h_params, theta_params, d_params, arch_params = [], [], [], list(self.architecture)

            for fs_neuron in self.fs_neurons:
                h_params.append(fs_neuron.h)
                theta_params.append(fs_neuron.theta)
                d_params.append(fs_neuron.d)
            
            return h_params, theta_params, d_params, arch_params
    
    def get_arch_params(self):
        return list(self.architecture)
    
    def freeze_params(self, params_to_freeze):
        for param in params_to_freeze:
            param.requires_grad = False
    
    def unfreeze_params(self, params_to_unfreeze):
        for param in params_to_unfreeze:
            param.requires_grad = True


    def __generate_genotype(self, T: int, num_grains: int) -> List[List[int]]:
        """
        Args: T=8, num_grains=3
        Outputs: [[2, 2, 4], [2, 3, 3], [2, 4, 2], [3, 2, 3], [3, 3, 2], [4, 2, 2]]
        """
        if num_grains * 2 > T:
            raise ValueError()
        genotype = []

        def _generate(current: List[int], remaining_T: int, remaining_grains: int):
            if len(current) == num_grains:
                if remaining_T == 0:
                    result = []
                    for i, count in enumerate(current):
                        result.extend([i] * count)
                    genotype.append(list(current))
                return
                
            min_size, max_size = 2, remaining_T - 2 * (remaining_grains - 1)
                
            for size in range(min_size, max_size + 1):
                _generate(current + [size], remaining_T - size, remaining_grains - 1)    
            
        _generate([], T, num_grains)

        assignment = []
        for element in genotype:
            element_list = []
            for grain_idx, steps in enumerate(element):
                element_list.extend([grain_idx] * steps)
            assignment.append(element_list)

        return assignment


    def __get_architecture(self, genotype):
        architecture = nn.ParameterList()
        arch_weights = []
        
        for i in range(self.num_edges):
            random_offset = torch.normal(mean=0.05, std=0.05, size=(1,))
            random_offset = torch.clamp(random_offset, min=0, max=0.1)

            if self.mode == "SoftMax":
                arch_param = nn.Parameter(
                    0.01 * torch.randn(1).abs(),
                    requires_grad=True
                )
            else:
                arch_param = nn.Parameter(
                    torch.ones(1) * 0.1 + random_offset * 0.02,
                    requires_grad=True
                )
            architecture.append(arch_param)
        
        for i, arch_param in enumerate(architecture):
            arch_weights.append(arch_param.item())
        print(f"arch_weights:{arch_weights}")

        return architecture
    
        
    def get_optimal_adaptive_T(self):
        arch_list = list(self.architecture)
        index = arch_list.index(max(arch_list))
        return self.genotype[index]
    

    def forward(self, X):
        losses, outputs = [], []

        for i, fs_neuron in enumerate(self.fs_neurons):
            Y, S = fs_neuron(X, None, True)
            loss_i = F.mse_loss(Y, X, reduction='mean')
            losses.append(loss_i)
            outputs.append(Y)
            
        arch_params = torch.stack(list(self.architecture)).squeeze()
        arch_weights = F.softmax(arch_params, dim=0)
            

        output = torch.zeros_like(X)
        for i, Y in enumerate(outputs):
            output += arch_weights[i] * Y
        
        total_loss = F.mse_loss(output, X, reduction='mean') + torch.sum(torch.stack(losses) * arch_weights) * 10
            
        # print(f"FS losses weighted sum: {torch.sum(torch.stack(losses) * arch_weights).item():.6f}")
            
        return total_loss, self.architecture

  
def train_phase_search(X, args, device, tau = None, mode = None):
    model = PhaseSearch(X, args.T, args.num_grains, args.beta, args.epoch_inner, device, tau = tau, mode=mode)
    model.to(device)
        
    best_error, best_T = torch.inf, None
    loss, _ = model(X)
    if loss.item() <= best_error:
        best_error = loss.item()
        best_T = model.get_optimal_adaptive_T()

    if mode == "SoftMax":
        theta_params, h_params, d_params, v0_params, arch_params = model.get_params()
    
        theta_h_optimizer = optim.Adam(h_params + theta_params, lr=args.lr)
        d_optimizer = optim.Adam(d_params, lr=args.lr) 
        v0_optimizer = optim.Adam(v0_params, lr=args.lr)
        arch_optimizer = optim.Adam(arch_params, lr=args.lr * 10)
        
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

        for epoch in range(args.epoch_outer):
            # optimize h, theta
            model.freeze_params(arch_params + d_params)
            model.unfreeze_params(theta_params + h_params)

            for inner_epoch in range(args.epoch_inner):
                theta_h_optimizer.zero_grad()
                theta_h_optimizer.zero_grad()

                loss, _ = model(X)
                loss.backward()
                theta_h_optimizer.step()
                v0_optimizer.step()
                
            # optimize d
            model.freeze_params(arch_params + theta_params + h_params)
            model.unfreeze_params(d_params)

            for inner_epoch in range(args.epoch_inner):
                d_optimizer.zero_grad()
                loss, _ = model(X)
                loss.backward()
                d_optimizer.step()
                v0_optimizer.step()

            model.freeze_params(theta_params + h_params + d_params)
            model.unfreeze_params(arch_params)
            
            arch_optimizer.zero_grad()
            loss, architecture = model(X)
            loss.backward()
            arch_optimizer.step()
            if loss.item() <= best_error:
                best_error = loss.item()
                best_T = model.get_optimal_adaptive_T()
            
            print(f"Epoch {epoch+1}/{args.epoch_outer}, Loss: {loss.item():.6f}")

                
            arch_weights = []
            for i, arch_param in enumerate(architecture):
                arch_weights.append(arch_param.item())
            
            print(f"arch_weights:{arch_weights}")
            print(f"optim T:{best_T}")

        return model.num_grains, best_T

    else:

        theta_params, h_params, d_params, arch_params = model.get_params()
        
        theta_h_optimizer = optim.Adam(h_params + theta_params, lr=args.lr)
        d_optimizer = optim.Adam(d_params, lr=args.lr) 
        arch_optimizer = optim.Adam(arch_params, lr=args.lr * 10)
        
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

        for epoch in range(args.epoch_outer):
            # optimize h, theta
            model.freeze_params(arch_params + d_params)
            model.unfreeze_params(theta_params + h_params)

            for inner_epoch in range(args.epoch_inner):
                theta_h_optimizer.zero_grad()
                theta_h_optimizer.zero_grad()

                loss, _ = model(X)
                loss.backward()
                theta_h_optimizer.step()
                
            # optimize d
            model.freeze_params(arch_params + theta_params + h_params)
            model.unfreeze_params(d_params)

            for inner_epoch in range(args.epoch_inner):
                d_optimizer.zero_grad()
                loss, _ = model(X)
                loss.backward()
                d_optimizer.step()

            model.freeze_params(theta_params + h_params + d_params)
            model.unfreeze_params(arch_params)
            
            arch_optimizer.zero_grad()
            loss, architecture = model(X)
            loss.backward()
            arch_optimizer.step()
            
            print(f"Epoch {epoch+1}/{args.epoch_outer}, Loss: {loss.item():.6f}")

            if loss.item() <= best_error:
                best_error = loss.item()
                best_T = model.get_optimal_adaptive_T()
                
            arch_weights = []
            for i, arch_param in enumerate(architecture):
                arch_weights.append(arch_param.item())
            
            print(f"arch_weights:{arch_weights}")
            print(f"optim T:{best_T}")

    return model.num_grains, best_T

    
def train_search(args):
    
    base_keys = [
        "post_attention_layernorm.output",
        "input_layernorm.output",
        "self_attn.o_proj.input",
        "mlp.down_proj.input",
        "self_attn.q_Identity.input",
        "self_attn.k_Identity.input",
        "self_attn.v_Identity.input",
        "self_attn.softmax_Identity.input",
        "mlp.up_proj.output",
        "mlp.silu_Identity.input",
    ]
    activation_value = []
    output_dict = {}

    # tau_dict = torch.load(f'../tau_dict.pth', map_location="cuda:0")
    tau, mode = 1.0, None

    for i in range(args.start, args.end):
        # search model.norm.output
        if i == -1:
            activation = torch.load(f'./activation_dir/{args.model_name}/layers/down_model.norm.output.pth', map_location="cuda:0")
            
            activation_value = activation[f"model.norm.output"]
            # tau = tau_dict[f"model.norm.output"]  

            print(f"load down_model.norm.output.pth success")

            X = activation_value.view(-1).to(device="cuda:0").unsqueeze(dim=1)
            num_grains, T_list = train_phase_search(X, args, device, tau, mode)

            print(f'num_grains={num_grains}, T_list={T_list}')
            output_dict[f"model.norm.output"]= T_list
            
            
            continue
             
        activation = torch.load(f'./activation_dir/{args.model_name}/layers/down_activation_stat_{i}.pth', map_location="cuda:0")
        print(f"load down_activation_stat_{i}.pth success")
        
        for bk in base_keys:
            activation_value = activation[f"model.layers.{i}.{bk}"]
            # tau = tau_dict[f"model.layers.{i}.{bk}"]  

            X = activation_value.view(-1).to(device="cuda:0").unsqueeze(dim=1)
            
            if "softmax" in bk:
                mode = "SoftMax"

            num_grains, T_list = train_phase_search(X, args, device, tau, mode)
            print("----------------------------------------------------------------")
            print(f"model.layers.{i}.{bk}")
            print(f'num_grains={num_grains}, T_list={T_list}')

            output_dict[f"model.layers.{i}.{bk}"]=(args.num_grains, T_list) 
    
    save_path = f'./arch_dir/{args.model_name}-T-{args.T}-grains-{args.num_grains}/search_arch.pth'
    if not os.path.exists(save_path):
        os.makedirs(save_path) 

    
    torch.save(output_dict, f'./arch_dir/{args.model_name}-T-{args.T}-grains-{args.num_grains}/search_arch.pth')
    print("save success")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device(f"{args.device}:{args.gpu}")
    elif torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        sys.exit(2)

    train_search(args)
    