import torch
from torch.autograd import Function, gradcheck
from quantize.quantizer import round_ste, clamp_ste, floor_ste

class MultiLevelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, th, L, sigma=0.5):
        out = floor_ste(input / th)
        out = out.clamp(0, L)
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
        clamped_floor_x = floor_x.clamp(0, L)
        mask = (floor_x > 0) & (floor_x < L)
        grad_input = grad_output * mask.float()
        grad_th = grad_output * mask.float() * (clamped_floor_x - x)
        print(grad_input)
        return grad_input, grad_th, None, None

def test_fn(input, th):
    return MultiLevelFunction.apply(input, th, 32, 0.5)

for i in range(1000):
    import os
    os.system('clear')
    # input = torch.randn(32, dtype=torch.float64, requires_grad=True)
    # th = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    # 生成0.5-2之间的随机1维tensor
    input = torch.empty(10, dtype=torch.float64).uniform_(-10, 10).requires_grad_()
    th = torch.empty(1, dtype=torch.float64).uniform_(0.5, 2.0).requires_grad_()
    # print(i, "input:", input, "th:", th)
    try:
        test = gradcheck(test_fn, (input, th), eps=1e-6, atol=1e-5)
        print(f"{i}_Gradcheck passed？", test)
        if test:
            break
    except:
        continue