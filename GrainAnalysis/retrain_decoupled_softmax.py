import torch
import torch.nn as nn
from utils import FSNeuronDecoupled, FSNeuronPlus, FSNeuronDecoupledSoftmax
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Phase Retrain Args')
parser.add_argument('--epoch_inner', type=int, default=2000, help='Number of epochs to train')
parser.add_argument('--time_step', type=int, default=4)
parser.add_argument('--num_grains', type=int, default=2)
parser.add_argument('--log_interval', type=int, default=1)
parser.add_argument('--beta', type=float, default=1.)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=32)
parser.add_argument('--tau', type=float, default=4)

args = parser.parse_args()


def plot_histograms(X, y, key_name, epoch=None, prefix=""):
    """绘制输入X和输出y的直方图对比"""
    plt.figure(figsize=(12, 5))
    
    # 将数据移到CPU并转换为numpy
    X_np = X.cpu().numpy().flatten()
    y_np = y.detach().cpu().numpy().flatten()
    
    # 绘制输入X的直方图
    plt.subplot(1, 2, 1)
    plt.hist(X_np, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Input X Distribution\n{key_name}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'Mean: {X_np.mean():.3e}\nStd: {X_np.std():.3e}\nMax: {X_np.max():.3e}\nMin: {X_np.min():.3e}'
    plt.text(0.65, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 绘制输出y的直方图
    plt.subplot(1, 2, 2)
    plt.hist(y_np, bins=50, alpha=0.7, color='red', edgecolor='black')
    plt.title(f'Output y Distribution\n{key_name}')
    plt.xlabel('Value')
    # plt.xlim((0, 0.05))
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'Mean: {y_np.mean():.3e}\nStd: {y_np.std():.3e}\nMax: {y_np.max():.3e}\nMin: {y_np.min():.3e}'
    plt.text(0.65, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 保存图像
    if epoch is not None:
        filename = f'./visualization/{prefix}_{key_name.replace("/", "_")}_epoch_{epoch}.png'
    else:
        filename = f'./visualization/{prefix}_{key_name.replace("/", "_")}_final.png'
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"直方图已保存: {filename}")


def train_fs_decoupled(X, args, fs_neuron):
    best_error= torch.inf
    
    optimizer_d = torch.optim.Adam([fs_neuron.d], lr=args.lr)
    optimizer_h_theta = torch.optim.Adam([fs_neuron.h, fs_neuron.theta, fs_neuron.v0], lr=1e-4)
    
    error_list= []
    y = None
    y, S = fs_neuron(X, None, True)
    error = F.mse_loss(y, X, reduction='mean')
    error_value = error.item()

    if error_value < best_error:
        best_error = error_value
        best_base, best_h, best_theta = fs_neuron.get_base_reset_thd_list()
        best_v0 = fs_neuron.v0.data
        # print(f"New best error={best_error}, best_base={best_base}, best_h={best_h}, best_theta={best_theta}")


    for i in range(args.epoch_inner // 500):
        mode = "h_theta" if i % 2 == 0 else "d"

        for j in range(100):
            optimizer_h_theta.zero_grad()

            fs_neuron.d.requires_grad_(False)
            fs_neuron.h.requires_grad_(True)
            fs_neuron.theta.requires_grad_(True)
            fs_neuron.v0.requires_grad_(True)
                
            y, S = fs_neuron(X, None, True)
            error = F.mse_loss(y, X, reduction='mean')
            # print(error)
            error.backward()
            optimizer_h_theta.step()

            fs_neuron.d.requires_grad_(True)
            # print(f"error={error}, mode={mode}")
                
        for k in range(100):
            optimizer_d.zero_grad()

            fs_neuron.d.requires_grad_(True)
            fs_neuron.h.requires_grad_(False)
            fs_neuron.theta.requires_grad_(False)
            fs_neuron.v0.requires_grad_(False)
            
            y, S = fs_neuron(X, None, True)
            error = F.mse_loss(y, X, reduction='mean')
            # print(error)
            error.backward()

            optimizer_d.step()

            fs_neuron.h.requires_grad_(True)
            fs_neuron.theta.requires_grad_(True)
            fs_neuron.v0.requires_grad_(True)

            # print(f"error={error}, mode={mode}")

        error_final = F.mse_loss(y, X, reduction='mean')
        error_value = error_final.item()

        if error_value < best_error:
            best_error = error_value
            best_base, best_h, best_theta = fs_neuron.get_base_reset_thd_list()
            best_v0 = fs_neuron.v0.data
            
        
    plot_histograms(X, y, key_name="retrain_decoupled_layernorm", prefix="decoupled")

    return best_error, best_base, best_h, best_theta, best_v0


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

    # output_dict = torch.load(f'/home/ubuntu/PhaseSNN/GrainAnalysis/retrain_decoupled/search_arch_grains=2-1714.pth', map_location="cuda:0")
    output_dict = torch.load(f'/home/ubuntu/PhaseSNN/GrainAnalysis/retrain_decoupled/search_arch_grains=2-floor_tau.pth', map_location="cuda:0")
    
    # retrain model.norm.output

        # test codes is here
    test_grains = 2
    if test_grains == 1:
        key_name, num_grains, genotype = model_norm_key, 1, [0, 0, 0, 0, 0, 0, 0, 0]
    if test_grains == 2:
        key_name, num_grains, genotype = model_norm_key, 2, [0, 0, 0, 0, 1, 1, 1, 1]
    if test_grains == 3:
        key_name, num_grains, genotype = model_norm_key, 3, [0, 0, 0, 1, 1, 1, 2, 2]
    
    # torch.save(output_dict, f'./retrain/search_arch_grains={num_grains}.pth')
    # exit(0)

    # retrain base_operations

    activation_value = []

    for i in range(args.start, args.end): 
        activation = torch.load(f'./activation_dir/down_activation_stat_{i}.pth', map_location="cuda:0")
        input_list = torch.load(f'./arch_dir/search_arch{i}.pth', map_location="cuda:0")
        print("load activation and search_arch success")

        # genotype = None 
        for element in input_list:
            key_name, _, _ = element # key_name, num_grains, genotype = element

            # num_grains, genotype = 2, [0, 0, 0, 0, 1, 1, 1, 1]  # for testing
            if "softmax"  not in key_name:
                continue
            
            tau_value = tau[key_name]

            #if fs_neuron == None:
            fs_neuron = FSNeuronDecoupledSoftmax(T=len(genotype), num_grains=num_grains,
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

            best_error, best_base, best_h, best_theta, best_v0 = train_fs_decoupled(X, args, fs_neuron)

            print(f"model_name: {key_name}")
            print(f"Ours: best_error={best_error}, base={best_base}, best_h={best_h}, best_theta={best_theta}, best_v0={best_v0}")

            best_error_baseline, best_base_baseline = train_fs_baseline(X, args, fs_neuron_baseline)
            best_base_values_baseline = [base.item() for base in best_base_baseline]
            print(f"Baseline: best_error={best_error_baseline}, base={best_base_baseline}")

            output_dict[key_name] = (num_grains, genotype, best_base, tau_value, best_h, best_theta, best_v0.float())
            # print(num_grains, genotype, best_base, tau_value, best_h, best_theta)


    # torch.save(output_dict, f'./retrain_decoupled/search_arch_grains=2-1714.pth')
    # torch.save(output_dict, f'./retrain_decoupled/search_arch_grains=2-floor_tau.pth')
    print("save success")

if __name__ == "__main__":
    retrain(args)