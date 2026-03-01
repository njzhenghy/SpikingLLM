import itertools
from typing import List
from collections import defaultdict
import torch 
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt


def generate_genotype(T: int, num_grains: int) -> List[List[int]]:
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


class FSNeuron(nn.Module):
    def __init__(self, T: int, num_grains: int = 2, beta : float = 10.0, genotype : list = None):
        """
        Args:
            T: 总时间步数
            num_grains: 粒度数量 (默认2个粒度)
        """
        super().__init__()
        self.T = T
        self.num_grains = num_grains
        
        self.d = nn.Parameter(torch.ones(num_grains))
        self.beta = beta
        
        self.grain_assignment = self._assign_grains(genotype)
        
        
    def _heaviside_ste(self, x, beta=10.0):
        return _HeavisideSTE.apply(x, beta)
    
    def _assign_grains(self, genotype=None):
        assignment = []
        steps_per_grain = self.T // self.num_grains
        remainder = self.T % self.num_grains
        
        for grain_idx in range(self.num_grains):
            steps = steps_per_grain + (1 if grain_idx < remainder else 0)
            assignment.extend([grain_idx] * steps)
        
        if genotype is not None:
            assignment = genotype

        return assignment
    
    def get_grain_power(self, t: int):
        grain_idx = self.grain_assignment[t]
        grain_start = self.grain_assignment.index(grain_idx)
        position_in_grain = t - grain_start
        return torch.pow(self.d[grain_idx], -(position_in_grain + 1))
    
    def forward(self, x, spikes: torch.Tensor = None, return_spikes: bool = False):
        y = torch.zeros_like(x)

        if spikes is None and x is not None:
            v = copy.deepcopy(x)
            spikes = []
            for t in range(self.T):
                power_val = self.get_grain_power(t)
                s_t = self._heaviside_ste(v - power_val, self.beta)
                spikes.append(s_t)
                v = v - power_val * s_t
            spikes = torch.stack(spikes, dim=0).squeeze()
            
        for t in range(self.T):
            power_val = self.get_grain_power(t)
            y = y + power_val * spikes[t].unsqueeze(-1)

        if return_spikes:
            return y, spikes
        return y


class LASNeuron(nn.Module):
    def __init__(self, T: int, beta : float = 1.0, tau : float = 1.0):
        """
        Args:
            T: 总时间步数
            num_grains: 粒度数量 (默认2个粒度)
        """
        super().__init__()
        self.T = T
        self.num_grains = 1
        self.d = nn.Parameter(torch.ones(self.num_grains)*2)
        self.tau = tau
        self.phase_list = {}
        self.beta = beta
        self.grain_assignment = [0] * self.T

    def _heaviside_ste(self, x, beta=10.0):
        return _HeavisideSTE.apply(x, beta)
    
    def get_grain_power(self, t: int):
        grain_idx = self.grain_assignment[t]
        grain_start = 0
        position_in_grain = t - grain_start
        self.phase_list[t] = torch.pow(self.d[grain_idx], -(position_in_grain + 1)) * self.tau
        return self.phase_list[t]

    def get_base_list(self):
        return self.d.data

    def forward(self, x, spikes: torch.Tensor = None, return_spikes: bool = False):
        y = torch.zeros_like(x)

        if spikes is None and x is not None:
            v = copy.deepcopy(x)
            spikes = []
            for t in range(self.T):
                power_val = self.get_grain_power(t)
                s_t = self._heaviside_ste(v - power_val, self.beta)
                spikes.append(s_t)
                v = v - power_val * s_t
            spikes = torch.stack(spikes, dim=0).squeeze()
            
        for t in range(self.T):
            power_val = self.get_grain_power(t)
            y = y + power_val * spikes[t].unsqueeze(-1)

        if return_spikes:
            return y, spikes
        else:
            return y


class FSNeuronDecoupled(nn.Module):
    def __init__(self, T: int, num_grains: int = 2, beta : float = 10.0, genotype : list = None, tau : float = 1.0):
        """
        Args:
            T: 总时间步数
            num_grains: 粒度数量 (默认2个粒度)
        """
        super().__init__()
        self.T = T
        self.num_grains = num_grains
        
        self.d = nn.Parameter(torch.ones(num_grains)*2)
        self.h = nn.Parameter(torch.ones(num_grains)*2)
        self.theta = nn.Parameter(torch.ones(num_grains)*2)

        self.phase_list = {}
        self.beta = beta
        self.tau = tau
        # self.tau = math.floor(tau + 0.5)
        self.grain_assignment = self._assign_grains(genotype)
        
        
    def _heaviside_ste(self, x, beta=10.0):
        return _HeavisideSTE.apply(x, beta)
    
    def _assign_grains(self, genotype=None):
        assignment = []
        steps_per_grain = self.T // self.num_grains
        remainder = self.T % self.num_grains
        
        for grain_idx in range(self.num_grains):
            steps = steps_per_grain + (1 if grain_idx < remainder else 0)
            assignment.extend([grain_idx] * steps)
        
        if genotype is not None:
            assignment = genotype

        return assignment
    
    def get_grain_power(self, t: int, mode='d'):
        grain_idx = self.grain_assignment[t]
        grain_start = 0
        position_in_grain = t - grain_start

        if mode == 'd':
            self.phase_list[t] = torch.pow(self.d[grain_idx], -(position_in_grain + 1)) * self.tau
        if mode == 'h':
            self.phase_list[t] = torch.pow(self.h[grain_idx], -(position_in_grain + 1)) * self.tau
        if mode == 'theta':
            self.phase_list[t] = torch.pow(self.theta[grain_idx], -(position_in_grain + 1)) * self.tau

        return self.phase_list[t]

    def get_base_reset_thd_list(self):
        return self.d.data, self.h.data, self.theta.data
    
    def forward(self, x, spikes: torch.Tensor = None, return_spikes: bool = False):
        y = torch.zeros_like(x)

        if spikes is None and x is not None:
            v = copy.deepcopy(x)
            spikes = []
            for t in range(self.T):
                h = self.get_grain_power(t ,mode='h')
                theta = self.get_grain_power(t ,mode='theta')
                s_t = self._heaviside_ste(v - theta, self.beta)
                spikes.append(s_t)
                v = v - h * s_t
            spikes = torch.stack(spikes, dim=0).squeeze()
            
        for t in range(self.T):
            d = self.get_grain_power(t ,mode='d')
            y = y + d * spikes[t].unsqueeze(-1)

        if return_spikes:
            return y, spikes
        return y


class FSNeuronDecoupledSoftMax(nn.Module):
    def __init__(self, T: int, num_grains: int = 2, beta : float = 10.0, genotype : list = None, tau : float = 1.0):
        """
        Args:
            T: 总时间步数
            num_grains: 粒度数量 (默认2个粒度)
        """
        super().__init__()
        self.T = T
        self.num_grains = num_grains
        
        self.d = nn.Parameter(torch.ones(num_grains)*2)
        self.phase_list = {}
        self.h = nn.Parameter(torch.ones(num_grains)*2)
        self.theta = nn.Parameter(torch.ones(num_grains)*2)
        self.beta = beta
        self.tau = tau
        # self.tau = math.floor(tau + 0.5)
        self.v0 = nn.Parameter(torch.tensor(0.5 * self.tau * 2 ** (-self.T)))

        if genotype is None:
            genotype = [0] * self.T
            num_grains = 1
            self.d = nn.Parameter(torch.ones(num_grains)*2)

        self.grain_assignment = self._assign_grains(genotype)
        
        
    def _heaviside_ste(self, x, beta=10.0):
        return _HeavisideSTE.apply(x, beta)
    
    def _assign_grains(self, genotype=None):
        assignment = []
        steps_per_grain = self.T // self.num_grains
        remainder = self.T % self.num_grains
        
        for grain_idx in range(self.num_grains):
            steps = steps_per_grain + (1 if grain_idx < remainder else 0)
            assignment.extend([grain_idx] * steps)
        
        if genotype is not None:
            assignment = genotype

        return assignment
    
    def get_grain_power(self, t: int, mode='d'):
        grain_idx = self.grain_assignment[t]
        grain_start = 0
        position_in_grain = t - grain_start

        if mode == 'd':
            return torch.pow(self.d[grain_idx], -(position_in_grain + 1)) * self.tau
        if mode == 'h':
            return torch.pow(self.h[grain_idx], -(position_in_grain + 1)) * self.tau
        if mode == 'theta':
            return torch.pow(self.theta[grain_idx], -(position_in_grain + 1)) * self.tau

    def get_base_reset_thd_list(self):
        return self.d.data, self.h.data, self.theta.data
    
    def forward(self, x, spikes: torch.Tensor = None, return_spikes: bool = False):
        y = torch.zeros_like(x)

        if spikes is None and x is not None:
            v = copy.deepcopy(x) + self.v0
            spikes = []
            for t in range(self.T):
                h = self.get_grain_power(t ,mode='h')
                theta = self.get_grain_power(t ,mode='theta')
                s_t = self._heaviside_ste(v - theta, self.beta)
                spikes.append(s_t)
                v = v - h * s_t
            spikes = torch.stack(spikes, dim=0).squeeze()
            
        for t in range(self.T):
            d = self.get_grain_power(t ,mode='d')
            y = y + d * spikes[t].unsqueeze(-1)

        if return_spikes:
            return y, spikes
        return y



def train_fs_decoupled(X, args, fs_neuron):
    best_error= torch.inf
    
    optimizer_d = torch.optim.Adam([fs_neuron.d], lr=args.lr)
    optimizer_h_theta = torch.optim.Adam([fs_neuron.h, fs_neuron.theta], lr=args.lr)


    y, S = fs_neuron(X, None, True)
    error = F.mse_loss(y, X, reduction='mean')
    error_value = error.item()

    if error_value < best_error:
        best_error = error_value
        best_base, best_h, best_theta = fs_neuron.get_base_reset_thd_list()
        # print(f"New best error={best_error}, best_base={best_base}, best_h={best_h}, best_theta={best_theta}")


    for i in range(args.epoch_inner // 500):
        mode = "h_theta" if i % 2 == 0 else "d"

        for j in range(20):
            optimizer_h_theta.zero_grad()

            fs_neuron.d.requires_grad_(False)
            fs_neuron.h.requires_grad_(True)
            fs_neuron.theta.requires_grad_(True)
                
            y, S = fs_neuron(X, None, True)
            error = F.mse_loss(y, X, reduction='mean')

            error.backward()
            optimizer_h_theta.step()

            fs_neuron.d.requires_grad_(True)
            # print(f"error={error}, mode={mode}")
                
        for k in range(20):
            optimizer_d.zero_grad()

            fs_neuron.d.requires_grad_(True)
            fs_neuron.h.requires_grad_(False)
            fs_neuron.theta.requires_grad_(False)
            
            y, S = fs_neuron(X, None, True)
            error = F.mse_loss(y, X, reduction='mean')
                
            error.backward()

            optimizer_d.step()

            fs_neuron.h.requires_grad_(True)
            fs_neuron.theta.requires_grad_(True)

            # print(f"error={error}, mode={mode}")

        error_final = F.mse_loss(y, X, reduction='mean')
        error_value = error_final.item()

        if error_value < best_error:
            best_error = error_value
            best_base, best_h, best_theta = fs_neuron.get_base_reset_thd_list()
            
        
    return best_error, best_base, best_h, best_theta


def train_fs_decoupled_softmax(X, args, fs_neuron):
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

        for _ in range(20):
            optimizer_h_theta.zero_grad()

            fs_neuron.d.requires_grad_(False); fs_neuron.h.requires_grad_(True); fs_neuron.theta.requires_grad_(True); fs_neuron.v0.requires_grad_(True)
                
            y, S = fs_neuron(X, None, True)
            error = F.mse_loss(y, X, reduction='mean')
            # print(error)
            error.backward()
            optimizer_h_theta.step()

            fs_neuron.d.requires_grad_(True)
            # print(f"error={error}, mode={mode}")
                
        for _ in range(20):
            optimizer_d.zero_grad()

            fs_neuron.d.requires_grad_(True); fs_neuron.h.requires_grad_(False); fs_neuron.theta.requires_grad_(False); fs_neuron.v0.requires_grad_(False)
            
            y, S = fs_neuron(X, None, True)
            error = F.mse_loss(y, X, reduction='mean')
            # print(error)
            error.backward()

            optimizer_d.step()

            fs_neuron.h.requires_grad_(True); fs_neuron.theta.requires_grad_(True); fs_neuron.v0.requires_grad_(True)


        error_final = F.mse_loss(y, X, reduction='mean')
        error_value = error_final.item()

        if error_value < best_error:
            best_error = error_value
            best_base, best_h, best_theta = fs_neuron.get_base_reset_thd_list()
            best_v0 = fs_neuron.v0.data
            
        
    # plot_histograms(X, y, key_name="retrain_decoupled_layernorm", prefix="decoupled")

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





if __name__ == "__main__":
    print(generate_genotype(T=8, num_grains=3))
    print(len(generate_genotype(T=8, num_grains=3)))