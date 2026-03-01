import torch
import torch.nn as nn
from utils import FSNeuronDecoupled, LASNeuron, FSNeuronDecoupledSoftMax
from utils import train_fs_decoupled, train_fs_baseline, train_fs_decoupled_softmax
import torch.nn.functional as F
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Phase Retrain Args')
parser.add_argument('--epoch_inner', type=int, default=2000, help='Number of epochs to train')
parser.add_argument('--T', type=int, default=8)
parser.add_argument('--num_grains', type=int, default=3)
parser.add_argument('--log_interval', type=int, default=1)
parser.add_argument('--beta', type=float, default=1.)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--start', type=int, default=-1)
parser.add_argument('--end', type=int, default=32)
parser.add_argument('--tau', type=float, default=4)

args = parser.parse_args()

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

def retrain(args, mode=None):
    if mode == "Test":
        if args.num_grains == 1:
            genotype = [0, 0, 0, 0, 0, 0, 0, 0]
        if args.num_grains == 2:
            genotype = [0, 0, 0, 0, 1, 1, 1, 1]
        if args.num_grains == 3:
            genotype = [0, 0, 0, 1, 1, 1, 2, 2]

    output_dict = {}
    tau = torch.load(f'../tau_dict.pth', map_location="cuda:0")
    
    # retrain base_operations
    for i in range(args.start, args.end):
        # Compute ModelNorm
        if i == -1:
            key_name = "model.norm.output"
            activation = torch.load(f'./activation_dir/down_model.norm.output.pth', map_location="cuda:0")
            arch = torch.load(f'./arch_dir/search_arch_model_norm.pth', map_location="cuda:0")
            activation_value = activation[key_name]

            X = activation_value.view(-1).to(args.device).unsqueeze(dim=1).abs()
            # genotype = arch[key_name][2]
            genotype = arch[0][2]
            tau_value = tau[key_name]   
            
            fs_neuron_Ours = FSNeuronDecoupled(T=args.T, num_grains=args.num_grains,
                                genotype=genotype, tau=tau_value).to(args.device)
            fs_neuron_baseline = LASNeuron(T=args.T, tau=tau_value).to(args.device)

            best_error, best_base, best_h, best_theta = train_fs_decoupled(X, args, fs_neuron_Ours)
            print(f"model_name: {key_name}")
            print(f"Ours: best_error={best_error}, base={best_base}, best_h={best_h}, best_theta={best_theta}")

            best_error_baseline, best_base_baseline = train_fs_baseline(X, args, fs_neuron_baseline)
            print(f"Baseline: best_error={best_error_baseline}, base={best_base_baseline}")

            output_dict[key_name] = (args.num_grains, genotype, best_base, tau_value, best_h, best_theta)
            continue

        activation = torch.load(f'./activation_dir/down_activation_stat_{i}.pth', map_location="cuda:0")
        arch_list = torch.load(f'./arch_dir/search_arch{i}.pth', map_location="cuda:0")
        print("load activation and search_arch success")

        # base opertaions
        for element in arch_list:
            best_error_baseline = 0
            key_name, _, genotype = element # key_name, num_grains, genotype = element
            tau_value = tau[key_name]
            print("--------------------------------------------------------------------------------------")
            print(f"model_name: {key_name}")
                
            if "softmax" not in key_name:
                fs_neuron_baseline = LASNeuron(T=args.T, tau=tau_value).to(args.device)
                fs_neuron_Ours = FSNeuronDecoupled(T=args.T, num_grains=args.num_grains,
                                genotype=genotype, tau=tau_value).to(args.device)
                
                activation_value = activation[key_name]
                X = activation_value.view(-1).to(args.device).unsqueeze(dim=1).abs()
                
                best_error_baseline, best_base_baseline = train_fs_baseline(X, args, fs_neuron_baseline)
                best_error, best_base, best_h, best_theta = train_fs_decoupled(X, args, fs_neuron_Ours)

                print(f"Baseline: best_error={best_error_baseline}, base={best_base_baseline}")
                print(f"Ours: best_error={best_error}, base={best_base}, best_h={best_h}, best_theta={best_theta}")

                output_dict[key_name] = (args.num_grains, genotype, best_base, tau_value, best_h, best_theta)
                # print(num_grains, genotype, best_base, tau_value, best_h, best_theta)
            else:
                fs_neuron_baseline = LASNeuron(T=args.T, tau=tau_value).to(args.device)
                fs_neuron_Ours = FSNeuronDecoupledSoftMax(T=len(genotype), num_grains=args.num_grains,
                            genotype=genotype, tau=tau_value).to(args.device)
         
                activation_value = activation[key_name]
                X = activation_value.view(-1).to(args.device).unsqueeze(dim=1).abs()

                best_error_baseline, best_base_baseline = train_fs_baseline(X, args, fs_neuron_baseline)
                best_error, best_base, best_h, best_theta, best_v0 = train_fs_decoupled_softmax(X, args, fs_neuron_Ours)

                print(f"Baseline: best_error={best_error_baseline}, base={best_base_baseline}")
                print(f"Ours: best_error={best_error}, base={best_base}, best_h={best_h}, best_theta={best_theta}, best_v0={best_v0}")

                output_dict[key_name] = (args.num_grains, genotype, best_base, tau_value, best_h, best_theta, best_v0.float())

    torch.save(output_dict, f'./retrain_decoupled/search_arch_grains={args.num_grains}-9-10-1315-tau.pth')
    print("save success")

if __name__ == "__main__":
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 若用 CUDA
    np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mode = "Train" # mode = "Test"
    retrain(args, mode)