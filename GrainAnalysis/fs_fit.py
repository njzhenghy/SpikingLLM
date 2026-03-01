import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np
import matplotlib.pyplot as plt


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
    def __init__(self, T: int):
        super().__init__()
        self.T = T
        self.d = nn.Parameter(torch.zeros(T))
        self.beta = 0.1
        #self.reset_parameters()

    def reset_parameters(self):
        self.d.data = torch.linspace(1, 10, steps=T)  

    def forward(self, x, spikes: torch.Tensor = None, return_spikes: bool = False):
        y = torch.zeros_like(x)

        if spikes is None and x is not None:
            v = copy.deepcopy(x)
            spikes = []
            for t in range(self.T):
                s_t = heaviside_ste(v - self.d[t], self.beta)
                spikes.append(s_t)
                v = v - self.d[t] * s_t
            spikes = torch.stack(spikes, dim=0).squeeze()
            
        for t in range(self.T):
            y = y + self.d[t] * spikes[t].unsqueeze(-1)

        if return_spikes:
            return y, spikes
        return y

manual_seed = 1
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)    

epoch = 10
T = 2

# X = torch.abs(torch.normal(mean=0.0, std=1.0, size=(10000,1)))

X = torch.abs(torch.normal(mean=0.0, std=1.0, size=(10000,1)))
S = torch.zeros((T, X.shape[0]))


fs1 = FSNeuron(T=T)

error_list = []
epoch = 1000
optimizer = torch.optim.SGD([fs1.d], lr=5e-4)
for i in range(epoch):
    ls_y = []
    y, S = fs1(X, None, True)

    l2_reg = 0.01 * torch.norm(fs1.d, p=1)
    error = F.mse_loss(y, X, reduction='mean')
    ls_y.append(y)
    error_list.append(error.data)
    optimizer.zero_grad()
    error.backward()
    optimizer.step()
    # print(fs1.d.data)
    print(error, fs1.d.data)

plt.figure(figsize=(10, 6))
plt.plot(error_list)
plt.title('Fs d(t) Fitting')
print("min e:", min(error_list))
plt.savefig('./bcd2.png')