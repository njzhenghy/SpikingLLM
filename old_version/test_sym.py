# 对称
import torch
import math
from quantize.quantizer import round_ste, clamp_ste, floor_ste
from SNN.spike_neuron import MultiLevelFunction
import random
import numpy as np
import matplotlib.pyplot as plt

# # torch.Size([2, 32, 2048, 128]) torch.Size([2, 32, 2051, 128])
# D = 128
# q = torch.randn((1,1,32,2048,128))
# k = torch.randn((1,1,32,2051,128))


# S_q = q.sum(dim=0)  # (T, bsz, H, Q, D)
# S_k = k.sum(dim=0)  # (T, bsz, H, K, D)
# attn2 = torch.matmul(S_q, S_k.transpose(-2, -1))/ math.sqrt(D)

# S_q = q.cumsum(dim=0)  # (T, B, H, N, D)
# S_k = k.cumsum(dim=0)  # (T, B, H, N, D)

# # 步骤 2：向量化的批量 matmul
# # term1: S_q @ k^T
# term1 = torch.matmul(S_q, k.transpose(-1, -2))  # (T, B, H, N, N)

# # term2: q @ S_k^T
# term2 = torch.matmul(q, S_k.transpose(-1, -2))  # (T, B, H, N, N)

# # term3: q @ k^T
# term3 = torch.matmul(q, k.transpose(-1, -2))    # (T, B, H, N, N)

# # 合并计算结果并归一化
# attn_scores = (term1 + term2 - term3) / math.sqrt(D)

# print(torch.allclose(attn_scores.sum(dim=0),attn2))


# T, B, H, Q, D = query_states.shape
# K = key_states.size(3)
# is_causal=causal_mask is None and q_len > 1
# a = torch.matmul(query_states.sum(dim=0), key_states.sum(dim=0).transpose(-2, -1)) / math.sqrt(D)

# S_q = query_states.cumsum(dim=0)  # (T, B, H, N, D)
# S_k = key_states.cumsum(dim=0)  # (T, B, H, N, D)

# # term1: S_q @ k^T
# term1 = torch.matmul(S_q, key_states.transpose(-1, -2))  # (T, B, H, N, N)
# # term2: q @ S_k^T
# term2 = torch.matmul(query_states, S_k.transpose(-1, -2))  # (T, B, H, N, N)
# # term3: q @ k^T
# term3 = torch.matmul(query_states, key_states.transpose(-1, -2))    # (T, B, H, N, N)
# # 合并计算结果并归一化
# b = (term1 + term2 - term3) / math.sqrt(D)


# import torch

# # 示例初始化
# A = torch.randn(2,1, 2048, 4096)
# B = torch.randn(2,1, 2048, 4096)

# # 构造近似 C
# C = 0.5 * (A * B.sum(dim=0)) + 0.5 * (B * A.sum(dim=0))
# print(C.shape)
# # 验证近似效果
# sum_C = C.sum(dim=0)
# sum_AB = A.sum(dim=0) * B.sum(dim=0)

# error = (sum_C - sum_AB).abs().mean()
# print("平均绝对误差:", error.item())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 如果使用 GPU，还需要设置 CUDA 的种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 为了确保 CUDA 的确定性（可重复），可能需要如下设置：
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 使用示例
set_seed(42)

scale = torch.Tensor([[0.1151]])
zero_point = None
n = 6
qmin = -2 ** (n - 1)
qmax = 2 ** (n - 1) - 1
group_size = 4096
x = torch.randn((1, 2048, 4096))

bs, num, dim1 = x.shape
scale = clamp_ste(scale, 1e-4, 1e4)
round_zero_point = clamp_ste(round_ste(zero_point), qmin, qmax) if zero_point is not None else None
max_val = qmax * scale
min_val = qmin * scale
v_0 = torch.ones_like(x).mul(scale) * 0.5
x_s = x
x_int = torch.clamp(x_s, min_val, max_val)
x_int = x_int.add(v_0.detach())
x_int = floor_ste(x_int / scale)
# x_int = round_ste(x_int / scale)
x_int = x_int.clamp(qmin, max=qmax)
x_dequant = x_int
x_dequant = x_dequant.mul(scale)
if round_zero_point is not None:
    x_dequant = x_dequant.sub(round_zero_point.mul(scale))
if  group_size:
    x_dequant = x_dequant.reshape(bs, num, dim1)

T_list = [1, 2, 3, 4, 5, 8, 16, 32]
norm_witha = []
max_witha = []
norm_withoa = []
max_withoa = []
for T in T_list:
    L = math.ceil((2 ** n - 1) / T)
    avg = True
    act = MultiLevelFunction.apply
    
    X = x / T
    X.unsqueeze_(0)
    X = X.repeat(T, 1, 1, 1)
    
    x_reshaped = X.reshape(T, bs, num, -1, group_size)
    # v = torch.ones_like(x_reshaped[ 0, ...]).mul(scale)*0.5
    v = torch.ones_like(x_reshaped[0, ...]).mul(scale) * 0.5
    I_z = (L * T + 1) * scale / 2
    spike_pot = []
    if avg: 
        # x_s = x_reshaped.sum(dim=0).detach().add(round_zero_point.mul(scale))
        # x_s = x_reshaped.sum(dim=0).detach().add(other=I_z)
        x_s = x_reshaped.sum(dim=0).detach()
        max_val = qmax * scale
        min_val = qmin * scale
        # max_val = (2 ** n - 1) * scale
        # min_val = 0 * scale
        x_s = torch.clamp(x_s, min_val, max=max_val)
        x_tmp = x_s / T
    for t in range(T):
        if not avg:
            x_tmp = x_reshaped[t, ...]
        v = v.detach().add(x_tmp.detach().add(I_z / T))
        output = act(v, scale, L)
        v -= output.detach()
        output = output.detach().sub(I_z.detach() / T)
        spike_pot.append(output)
    spike_pot = torch.stack(spike_pot, dim=0)
    if group_size:
        spike_pot = spike_pot.reshape(T, bs, num, dim1)

    a = spike_pot.sum(dim=0)

    error = torch.norm(input=a-x_dequant)
    print(torch.allclose(a,x_dequant))
    norm = torch.norm(input=a-x_dequant)
    max = torch.max(torch.abs(x_dequant-a))
    
    print(a - x_dequant)
    print(norm)
    print(max)
    norm_witha.append(norm)
    max_witha.append(max)
    # print(torch.max(a-x_dequant))

    # v = torch.ones_like(x_reshaped[ 0, ...]).mul(scale)*0.5
    # v_th = scale/(T * L)
    # I_z = round_zero_point / T
    # I_z = I_z.mul(scale)
    # # I_z = round_zero_point 
    # spike_pot = []
    # avg = False
    # for t in range(T):
    #     x_tmp = x_reshaped[t, ...].add(I_z)
    #     v =  v.detach().add(x_tmp.detach())
    #     output =  act(v, scale,  L)
    #     v -= output.detach()
    #     output = output.detach().sub(I_z.detach())
    #     spike_pot.append(output)
    # spike_pot = torch.stack(spike_pot, dim=0)
    # if group_size:
    #     spike_pot = spike_pot.reshape(T, bs, num, dim1)
    # a = spike_pot.sum(dim=0)

    # print("-------------------------------------------------")
    # print(torch.allclose(a,x_dequant))
    # norm = torch.norm(a-x_dequant)
    # max = torch.max(torch.abs(x_dequant-a))
    # print(norm)
    # print(max)
    # norm_withoa.append(norm)
    # max_withoa.append(max)
    # # print(torch.max(a-x_dequant))
    # print("================================================")

# 定义数据
# T_values = [1, 2, 4, 8, 16, 32]

# norm = [1e-8, 0.307, 0.7521, 0.4855, 0.2171, 0.0028]
# max = [1e-8, 0.2171, 0.2171, 0.2171, 0.2171, 9.5367e-07]

# norm_n = [1e-8, 0.6513, 0.8408, 139.9352, 999.5336, 586.6599]
# max_n = [1e-8, 0.2171, 0.2171, 0.6513, 2.6052, 1.0855]

# 创建画布
plt.figure(figsize=(10, 6), dpi=100)

# 绘制三条曲线（带不同标记和标签）
plt.plot(T_list, norm_witha, 
         marker='o', linestyle='-', linewidth=2, markersize=8, label='norm(with average)', color='r')
plt.plot(T_list, max_witha, 
         marker='s', linestyle='-', linewidth=2, markersize=8, label='max(with average)', color='g')
# plt.plot(T_list, norm_withoa, 
#          marker='o', linestyle='--', linewidth=2, markersize=8, label='norm(without average)', color='tomato')
# plt.plot(T_list, max_withoa, 
#          marker='s', linestyle='--', linewidth=2, markersize=8, label='max(without average)', color='limegreen')

# plt.yscale('log')
# 设置坐标轴
plt.xticks(T_list)  # 强制x轴只显示指定刻度
plt.xlabel('T', fontsize=20)
plt.ylabel('Error', fontsize=20)

# plt.yscale('log')
# 添加辅助元素
plt.grid(True, linestyle='--', alpha=0.7)  # 虚线网格
plt.legend(loc='best', fontsize=20)        # 自动选择最佳图例位置

# 显示图形
plt.tight_layout()
plt.show()
plt.savefig('error.png')

# v = torch.ones_like(x_reshaped[ 0, ...]).mul(scale)*0.5
# v_th = scale/( T* L)
# I_z = round_zero_point /  T
# I_z = I_z.mul(scale)
# # I_z = round_zero_point 
# spike_pot = []
# avg = False
# x_s = x_reshaped.sum(dim=0).detach().add(round_zero_point.mul(scale))
# v =  v.detach().add(x_s.detach())
# for t in range( T):
#     # x_tmp = x_reshaped[t, ...].add(I_z)
#     # v =  v.detach().add(x_tmp.detach())
#     output =  act( v, scale,  L)
#     v -= output.detach()
#     output = output.detach().sub(I_z.detach())
#     spike_pot.append(output)
# spike_pot = torch.stack(spike_pot, dim=0)
# if group_size:
#     spike_pot = spike_pot.reshape(T, bs, num, dim1)
# a = spike_pot.sum(dim=0)

# print("-------------------------------------------------")
# print(torch.allclose(a,x_dequant))
# print(torch.norm(a-x_dequant))
# print(torch.max(x_dequant-a))
# print(torch.max(a-x_dequant))

pass