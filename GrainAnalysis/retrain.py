import torch
import torch.nn as nn
from utils import FSNeuronPlus
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description='Phase Retrain Args')
parser.add_argument('--epoch_inner', type=int, default=2000, help='Number of epochs to train')
parser.add_argument('--time_step', type=int, default=4)
parser.add_argument('--num_grains', type=int, default=2)
parser.add_argument('--log_interval', type=int, default=1)
parser.add_argument('--beta', type=float, default=10.)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=32)
parser.add_argument('--tau', type=float, default=4)

args = parser.parse_args()

def train_fs(X, args, fs_neuron):
    best_error, best_base = torch.inf, None
    optimizer = torch.optim.Adam([fs_neuron.d], lr=args.lr)
   
    for i in range(args.epoch_inner):
        y, S = fs_neuron(X, None, True)
        # l1_reg = 0.01 * torch.norm(S, p=1)
        error = F.mse_loss(y, X, reduction='mean')
        if error.item() < best_error:
            best_error, best_base = error.item(), fs_neuron.get_base_list()
        optimizer.zero_grad()
        error.backward(retain_graph=True)
        optimizer.step()

    return best_error, best_base

def train_fs_baseline(X, args, fs_neuron):
    best_error, best_base = torch.inf, None
   
    for i in range(1):
        y, S = fs_neuron(X, None, True)
        # l1_reg = 0.01 * torch.norm(S, p=1)
        error = F.mse_loss(y, X, reduction='mean')
        if error.item() < best_error:
            best_error, best_base = error.item(), fs_neuron.get_base_list()

    return best_error, best_base

def retrain(args, fs_neuron=None):
    tau = 1.
    model_norm_key = "model.norm.output"
    activation = torch.load(f'./activation_dir/model.norm.output.pth', map_location="cuda:0")
    tau = torch.load(f'../tau_dict.pth', map_location="cuda:0")

    output_dict = torch.load(f'/home/ubuntu/PhaseSNN/GrainAnalysis/retrain/search_arch_grains=2.pth', map_location="cuda:0")
    
    # retrain model.norm.output

        # test codes is here
    test_grains = 3
    if test_grains == 1:
        key_name, num_grains, genotype = model_norm_key, 1, [0, 0, 0, 0, 0, 0, 0, 0]
    if test_grains == 2:
        key_name, num_grains, genotype = model_norm_key, 2, [0, 0, 0, 0, 1, 1, 1, 1]
    if test_grains == 3:
        key_name, num_grains, genotype = model_norm_key, 3, [0, 0, 0, 1, 1, 1, 2, 2]
    
    tau_value = tau[key_name]

    # num_grains, genotype = 2, [0, 0, 0, 0, 1, 1, 1, 1]  # for testing

    fs_neuron = FSNeuronPlus(T=len(genotype), num_grains=num_grains,
                                genotype=genotype, tau=tau_value).to(args.device)

    fs_neuron_baseline = FSNeuronPlus(T=len(genotype), num_grains=1,
                                genotype=None, tau=tau_value).to(args.device)

    activation_value = activation[key_name]
    X = activation_value.view(-1).to(args.device).unsqueeze(dim=1).abs()
    
    best_error, best_base = train_fs(X, args, fs_neuron)
    best_base_values = [phase.item() for phase in best_base]
    print(f"model_name: {key_name}")
    print(f"Ours: best_error={best_error}, base={best_base}")

    best_error_baseline, best_base_baseline = train_fs_baseline(X, args, fs_neuron_baseline)
    best_base_values_baseline = [phase.item() for phase in best_base_baseline]
    print(f"Baseline: best_error={best_error_baseline}, base={best_base_baseline}")

    output_phase = torch.tensor(best_base_values)
    output_dict[key_name] = (num_grains, genotype, output_phase, tau_value)
    # torch.save(output_dict, f'./retrain/search_arch_9-4-1748.pth')
    # torch.save(output_dict, f'./retrain/search_arch_grains={num_grains}.pth')
    # exit(0)

    # retrain base_operations
    patterns = ["q_Identity", "k_Identity", "post_attention_layernorm"]

    activation_value = []


    for i in range(args.start, args.end): 
        activation = torch.load(f'./activation_dir/down_activation_stat_{i}.pth', map_location="cuda:0")
        input_list = torch.load(f'./arch_dir/search_arch{i}.pth', map_location="cuda:0")
        print("load activation and search_arch success")

        # genotype = None 
        for element in input_list:
            key_name, _, _ = element # key_name, num_grains, genotype = element

            # num_grains, genotype = 2, [0, 0, 0, 0, 1, 1, 1, 1]  # for testing
            tau_value = tau[key_name]

            #if fs_neuron == None:
            fs_neuron = FSNeuronPlus(T=len(genotype), num_grains=num_grains,
                            genotype=genotype, tau=tau_value).to(args.device)
            fs_neuron_baseline = FSNeuronPlus(T=len(genotype), num_grains=1,
                            genotype=None, tau=tau_value).to(args.device)
         
            activation_value = activation[key_name]
            X = activation_value.view(-1).to(args.device).unsqueeze(dim=1).abs()

            # if any(pattern in key_name for pattern in patterns):
            #     tau = torch.max(X)
            #     X /= tau
            #     print(f"name={key_name}, mean={torch.mean(X)}, std={torch.std(X)}, max={tau}, min={torch.min(X)}")
            # else:
            #     tau = 1

            best_error, base = train_fs(X, args, fs_neuron)
            best_base_values = [base.item() for base in best_base]
            print(f"model_name: {key_name}")
            print(f"Ours: best_error={best_error}, base={base}")

            best_error_baseline, best_base_baseline = train_fs_baseline(X, args, fs_neuron_baseline)
            best_base_values_baseline = [base.item() for base in best_base_baseline]
            print(f"Baseline: best_error={best_error_baseline}, base={best_base_baseline}")

            output_base = torch.tensor(best_base_values)
            output_dict[key_name] = (num_grains, genotype, output_base, tau_value)
            # print(num_grains, genotype, output_base, tau)

    torch.save(output_dict, f'./retrain/search_arch_grains={num_grains}.pth')
    print("save success")

if __name__ == "__main__":
    retrain(args)