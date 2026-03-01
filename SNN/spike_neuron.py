import torch
import torch.nn as nn
import sys
from quantize.quantizer import round_ste, clamp_ste, floor_ste

class TwoLevelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, th):
        out2 = (input >= 2. * th).float()
        out1 = (input >= 1. * th).float() * (1. - out2)
        out = out1 * th + out2 * 2. * th
        input = ((input.detach() >= 0.5 * th) * (input.detach() <= 2.5 * th)).float()       
        ctx.save_for_backward(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (tmp,) = ctx.saved_tensors
        grad_input = grad_output * tmp
        return grad_input, None


class FourLevelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, th):       
        out4 = (input >= 4. * th).float()
        out3 = (input >= 3. * th).float() * (1. - out4)
        out2 = (input >= 2. * th).float() * (1. - out4) * (1. - out3)
        out1 = (input >= 1. * th).float() * (1. - out4) * (1. - out3) * (1. - out2)
        out = out1 * th + out2 * 2. * th + out3 * 3. * th + out4 * 4. * th
        input = ((input.detach() >= 0.5 * th) * (input.detach() <= 4.5 * th)).float()
        ctx.save_for_backward(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (tmp,) = ctx.saved_tensors
        grad_input = grad_output * tmp
        return grad_input, None


class MultiLevelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, th, L, sigma=0.5):
        out = floor_ste(input / th)
        # out = out.clamp(0, L)
        out = clamp_ste(out, 0, L)
        out = out * th
        ctx.save_for_backward(input, th)
        ctx.L = L
        ctx.sigma = sigma
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, th = ctx.saved_tensors
        L = ctx.L
        x = input / th
        floor_x = floor_ste(x)
        # clamped_floor_x = floor_x.clamp(0, L)
        clamped_floor_x = clamp_ste(clamped_floor_x, 0, L)
        mask = (floor_x > 0) & (floor_x < L)
        grad_input = grad_output * mask.float()
        grad_th = grad_output * mask.float() * (clamped_floor_x - x)
        return grad_input, grad_th, None, None

    # @staticmethod
    # def backward(ctx, grad_output):
    #     input, th = ctx.saved_tensors
    #     L = ctx.L
    #     x = floor_ste(input / th) / L
    #     grad_input = torch.sigmoid(x)*torch.sigmoid(1-x)
    #     grad_th = L*torch.sigmoid(x) - torch.sigmoid(x)*torch.sigmoid(1-x)*input/th
    #     return grad_output*grad_input, grad_output*grad_th, None, None

class LMHTNeuron(nn.Module):
    def __init__(self, L: int, ori, T=2, avg=True, initv=0.5):
        super(LMHTNeuron, self).__init__()
        self.scale = nn.Parameter(ori.scale, requires_grad=True)
        # self.register_parameter('scale', ori.scale)
        if ori.zero_point is not None:
            self.zero_point = nn.Parameter(ori.zero_point, requires_grad=True)
        self.v = None
        self.avg = avg
        self.act = MultiLevelFunction.apply
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
