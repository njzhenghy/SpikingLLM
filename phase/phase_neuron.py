import torch
import torch.nn as nn
import sys
from quantize.quantizer import round_ste, clamp_ste, floor_ste


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


def heaviside_ste(x, beta=10.0):
    return _HeavisideSTE.apply(x, beta)


class FSNeuron(nn.Module):
    def __init__(self, T: int, quantized_shape, quantized_item_stat, num_grains: int = 1, beta : float = 1.0, genotype : list = None, neuron_d : torch.Tensor = None, tau : float = 1.0, neuron_h : torch.Tensor = None, neuron_theta : torch.Tensor = None, neuron_v0 : float = 0.0):
        """
        Args:
            T: 总时间步数
            num_grains: 粒度数量 (默认2个粒度)
        """
        super().__init__()
        self.T = T
        self.num_grains = num_grains
        
        if neuron_d is not None:
            self.d = nn.Parameter(neuron_d)
            self.tau = tau
            self.h = nn.Parameter(neuron_h)
            self.theta = nn.Parameter(neuron_theta) 
        else:
            self.d = nn.Parameter(torch.ones(num_grains)*2)
            self.tau = None
            self.group_size = quantized_shape[-1]
            self.find_activation_quant_param(quantized_item_stat)
            genotype = [0] * self.T
            self.h = nn.Parameter(torch.ones(num_grains)*2)
            self.theta = nn.Parameter(torch.ones(num_grains)*2) 

        self.beta = beta
        
        self.grain_assignment = self._assign_grains(genotype)

        self.v = None

        self.v0 = neuron_v0

    def _heaviside_ste(self, x, beta=10.0):
        return _HeavisideSTE.apply(x, beta)

    def _assign_grains(self, genotype=None):
        assignment = []
        steps_per_grain = self.T // self.num_grains
        remainder = self.T % self.num_grains
        
        # for grain_idx in range(self.num_grains):
        #     steps = steps_per_grain + (1 if grain_idx < remainder else 0)
        #     assignment.extend([grain_idx] * steps)
        
        if genotype is not None:
            assignment = genotype
            return assignment
        else:
            return [0, 0, 0, 0, 0, 0, 0, 0]

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
    
    def forward(self, x, spikes: torch.Tensor = None, return_spikes: bool = False):
        T, bs, n, dim1 = x.shape
        y = []

        if spikes is None and x is not None:
            self.v = torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)
            for t in range(self.T):
                self.v = self.v + x[t, ...]
            
            v_sign = torch.sign(self.v)
            self.v = self.v.abs() + self.v0

            spikes = []
            for t in range(self.T):
                h = self.get_grain_power(t, mode='h')
                theta = self.get_grain_power(t, mode='theta')
                s_t = self._heaviside_ste(self.v - theta, self.beta)
                spikes.append(s_t)
                self.v = self.v - h * s_t
            spikes = torch.stack(spikes, dim=0).reshape(T, bs, n, dim1)
            
        for t in range(self.T):
            d = self.get_grain_power(t, mode='d')
            y.append(v_sign * d * spikes[t])       

        y = torch.stack(y, dim=0).reshape(T, bs, n, dim1)
        if return_spikes:
            return y, spikes
        return y

    def find_activation_quant_param(self, quantized_item_stat):
        x = quantized_item_stat.reshape(-1,self.group_size)
        xmax = x.amax([-1], keepdim=True)   
        # scale = 2*xmax/(2**self.n_bits-1)
        # scale = 2*xmax/(2**self.n_bits)
        # scale = scale.clamp(min=5e-4, max=1e4)
        self.tau = float(xmax)

class LMHTNeuron(nn.Module):
    def __init__(self, L: int, ori, T=2, avg=True, initv=0.5):
        super(LMHTNeuron, self).__init__()
        self.scale = nn.Parameter(ori.scale, requires_grad=True)
        # self.register_parameter('scale', ori.scale)
        if ori.zero_point is not None:
            self.zero_point = nn.Parameter(ori.zero_point, requires_grad=True)
        self.v = None
        self.avg = avg
        # self.act = MultiLevelFunction.apply
        self.T = T
        self.L = L
        self.quantized_shape = ori.quantized_shape
        self.group_size = ori.group_size
        self.mode = ori.mode
        self.asym = ori.asym
        self.disable_zero_point_in_sym = ori.disable_zero_point_in_sym
        self.activation_clipping = ori.activation_clipping
        self.enable = True
        self.qmin = ori.qmin
        self.qmax = ori.qmax
        self.initv = nn.Parameter(torch.tensor(initv))
        
        self.ori = ori  # lzg添加，SNN转ANN时需要用到的原始参数

    def forward(self, x: torch.Tensor):
        """
        x: (TB, lenSeq, dim)
        """
        scale = clamp_ste(self.scale, 5e-4, 1e4)
        if self.asym or not self.disable_zero_point_in_sym:
            round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax) if self.zero_point is not None else None
        if self.asym:   # lzg添加,不修改原有逻辑的基础上,添加判断分支
            T, bs, n, dim1 = x.shape
            x_reshaped = x.reshape( T, bs, n, -1, self.group_size)
            self.v = torch.ones_like(x_reshaped[ 0, ...]) * scale * self.initv  # 7月12日版本
            #! 下面两行是swl改动的后续版本
            # self.v = torch.ones_like(x_reshaped[ 0, ...]) * scale * 0.5
            # self.v = self.v + torch.ones_like(x_reshaped[ 0, ...]) * self.initv
            I_z = round_zero_point / self.T
            I_z = I_z * scale
            spike_pot = []
            if self.avg: 
                x_s = x_reshaped.sum(dim=0) + round_zero_point * scale
                max_val = self.qmax * scale
                min_val = torch.zeros_like(max_val)
                x_s = torch.clamp(x_s, min_val, max_val)
                # x_s = clamp_ste(x_s, min_val, max_val)
                x_tmp = x_s / self.T
            for t in range(self.T):
                # max_val = self.L * scale
                # min_val = torch.zeros_like(max_val)
                # x_tmp = torch.clamp(x_reshaped[t, ...].add(I_z), min_val, max_val)
                # x_tmp = torch.clamp(x_tmp, min_val, max_val)
                if not self.avg:
                    x_tmp = x_reshaped[t, ...] + I_z
                self.v = self.v + x_tmp
                output = floor_ste(self.v / scale)
                output = output.clamp(0, self.L)
                output = output * scale
                # output = self.act(self.v, scale, self.L)
                self.v = self.v - output
                output = output - I_z
                # self.v = self.v.clamp(min=min_val, max=max_val-1e-4)
                # self.v = self.v.clamp(0, self.L*scale-1e-4)
                spike_pot.append(output)
            spike_pot = torch.stack(spike_pot, dim=0)
            if self.group_size:
                spike_pot = spike_pot.reshape(T, bs, n, dim1)
        else:
            T, bs, n, dim1 = x.shape
            x_reshaped = x.reshape(T, bs, n, -1, self.group_size)
            self.v = torch.ones_like(x_reshaped[ 0, ...]).mul(scale)*self.initv  # 7月12日版本
            #! 下面两行是swl改动的后续版本
            # self.v = torch.ones_like(x_reshaped[ 0, ...]) * scale * 0.5
            # self.v = self.v + torch.ones_like(x_reshaped[ 0, ...]) *  self.initv
            I_z = (self.L * T + 1) * scale / 2
            spike_pot = []
            if self.avg:
                x_s = x_reshaped.sum(dim=0)
                max_val = self.qmax * scale
                min_val = self.qmin * scale
                x_s = torch.clamp(x_s, min_val, max=max_val)
                # x_s = clamp_ste(x_s, min_val, max_val)
                x_tmp = x_s / self.T
            for t in range(self.T):
                if not self.avg:
                    x_tmp = x_reshaped[t, ...]
                self.v = self.v.add(x_tmp.add(I_z / self.T))
                # output = self.act(self.v, scale, self.L)
                output = floor_ste(self.v / scale)
                output = output.clamp(0, self.L)    # 注意这行，如果用下面的clamp_ste会导致训练loss差两个数量级这么恐怖！！！
                # output = clamp_ste(output, 0, self.L)
                output = output * scale
                
                self.v = self.v - output
                # output = output.sub(I_z / self.T)
                output = output - (I_z / self.T)
                spike_pot.append(output)
            spike_pot = torch.stack(spike_pot, dim=0)
            if self.group_size:
                spike_pot = spike_pot.reshape(T, bs, n, dim1)

        return spike_pot