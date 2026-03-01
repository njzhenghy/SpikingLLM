import argparse
import logging
from argparse import Namespace
from typing import Any, Sequence, Optional, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Function
from transformers.activations import ACT2FN

from utils import clamp_ste, round_ste, floor_ste


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
        return grad_input, grad_th, None, None


class MLSWATFunction(Function):
    r"""
    == 用于动态多阈值的脉冲释放函数。 ==
    动态适应性：无论阈值间距是否相等，自动调整邻域范围。
    梯度传播合理性：仅在输入值接近阈值时传播梯度，避免离散跳跃导致的训练不稳定。
    兼容性：在等间距场景下与原函数行为一致（如 [4th, 3th, 2th, th] → 范围 [0.5th, 4.5th]）。
    """

    @staticmethod
    def forward(ctx: Any, inputTensor, threshes: Tensor):
        r"""

        :param ctx:
        :param inputTensor:
        :type inputTensor:Tensor
        :param threshes: 一维张量。多个阈值从大到小排序的。
        :type threshes:Tensor
        :return:
        :rtype:Tensor
        """
        # region ---多阈值互斥累加---
        # 掩码张量。用来排除被比较过的阈值。
        rem = torch.ones_like(inputTensor, dtype=inputTensor.dtype)
        # 初始化输出结果。
        out = torch.zeros_like(inputTensor, dtype=inputTensor.dtype)
        # 依次和每个阈值比较。
        for thresh in threshes:
            # 当前阈值的匹配掩码
            mask = (inputTensor >= thresh).float()
            # 排除已被更大阈值匹配的部分
            exclusive = mask * rem
            # 在当前迭代累加当前阈值
            out = out + exclusive * thresh
            # 更新剩余掩码，这些位置将不再被后续（更小的）阈值匹配
            rem = rem * (1.0 - mask)
        # endregion

        # region ——— 构造 backward 用的 tmp mask ———
        # 计算半步长：smallest = threshes[-1], second_smallest = threshes[-2]
        if threshes.numel() > 1:
            halfLower = (threshes[-2] - threshes[-1]) * 0.5
            halfUpper = (threshes[0] - threshes[1]) * 0.5
        else:
            # 只有一个阈值时，范围就是 [thresh/2, thresh*1.5]
            halfLower = threshes[0] * 0.5
            halfUpper = halfLower

        lower = threshes[-1] - halfLower
        upper = threshes[0] + halfUpper

        backward_mask = (
                (inputTensor.detach() >= lower) &
                (inputTensor.detach() <= upper)
        ).float()

        # 保存所有 backward 需要的张量
        ctx.save_for_backward(inputTensor, threshes, backward_mask)
        return out

    @staticmethod
    def backward(ctx, gradOutput: Tensor):
        r"""
        grad_output: 与 forward 返回的 out 同形，表示 dL/dout
        需要返回
        :param ctx:
        :param gradOutput:
        :type gradOutput: Tensor
        :return: (dL/dinputTensor, dL/dthreshes)
        :rtype: Tuple[Tensor, Tensor]
        """
        inputTensor, threshes, backward_mask = ctx.saved_tensors

        # ——— 输入梯度：原地屏蔽 = grad_output * mask ———
        gradInput = gradOutput * backward_mask

        # ——— 阈值梯度：按互斥逻辑直接累加到 grad_threshes ———
        rem = torch.ones_like(inputTensor, dtype=inputTensor.dtype)
        gradThreshes = torch.zeros_like(threshes)

        # 对每个阈值，梯度 ∂out/∂t_i = exclusive_i
        # 所以 dL/dt_i = sum( dL/dout * exclusive_i )
        for i, thresh in enumerate(threshes):
            mask = (inputTensor >= thresh).float()
            exclusive = mask * rem
            # 累加所有元素位置的梯度
            gradThreshes[i] = (gradOutput * exclusive).sum()
            rem = rem * (1.0 - mask)

        return gradInput, gradThreshes


class MLSWATNeuron(nn.Module):
    r"""
    == Multi-level Spiking with Adaptive Threshes ==

    需要注意。在定义阈值的时候，阈值是一个长度为 T 的列表。
    其中第一个元素为最低的阈值，后续的值代表的意思为：
    第 t+1 个元素的值，代表的是第 t+1 个阈值比第 t 个阈值大的部分。

    # TODO阈值的更新需要注意，在优化器执行完 step() 之后，需要手动裁切。比如：

    通过 threshes 属性，对更新后的阈值实时投影，保持其偏序关系。
    """

    def __init__(self,
                 args: Namespace,
                 scale: Tensor,
                 zero_point: Tensor,
                 nameLogger: str) -> None:
        r"""

        :param args:
        :type args:Namespace
        :param scale:
        :type scale: Tensor
        :param zero_point:
        :type zero_point:Tensor
        :param nameLogger:
        :type nameLogger:str
        """
        super().__init__()
        self.logger = logging.getLogger(nameLogger)
        self.args = args
        if args.ifLearnThreshInMLSWA:
            self.logger.info(f"指定了需要学习阈值。")
        else:
            self.logger.info(f"使用固定阈值。")

        # region 对阈值的设置。
        if args.threshes is not None:  # TODO 需要在 arguments 中单独添加该参数，区别于 args.thresh .
            # 指定阈值
            if not isinstance(args.threshes, Sequence):
                errorMsg = ValueError(f"args.threshes 要么为None，要么为浮点列表")

            elif len(args.threshes) != self.args.L:
                errorMsg = ValueError(
                    f"阈值的数量 len(args.threshes) 应该等于每个时刻可能发送的最多脉冲数 self.L -{self.args.L}。")
            else:
                errorMsg = None
                self.raw_threshes = nn.Parameter(torch.tensor(args.threshes),
                                                 requires_grad=args.ifLearnThreshInMLSWA)
        else:
            # 从 QANN 继承 scale 作为阈值。
            if scale is None:
                errorMsg = ValueError(f"当不显示指定阈值的时候，应该传入量化的 scale 用来定义阈值。")
            # elif len(scale.shape) != 1:
            #     errorMsg = ValueError(f"选择从 QANN 继承 scale 的时候，scale 的必须是1维张量。")
            # elif len(scale) != self.args.L:
            #     errorMsg = ValueError(f"选择从 QANN 继承 scale 的时候，scale 的元素个数 {len(scale)} 必须等于"
            #                           f" 每个时刻可能发送的最多脉冲数 self.L - {self.args.L}。")
            else:
                errorMsg = None
                self.raw_threshes = nn.Parameter(torch.tensor([scale
                                                               for _ in range(self.args.L)]),
                                                 requires_grad=args.ifLearnThreshInMLSWA)
        if errorMsg is not None:
            self.logger.error(errorMsg)
            raise errorMsg
        if not (self.threshes > 0).all().item():
            errorMsg = ValueError(f"阈值中的每个元素应该都是正数！")
            self.logger.info(errorMsg)
            raise errorMsg
        # endregion

        # region zero_point的设置
        if zero_point is not None:
            self.zero_point = nn.Parameter(zero_point,
                                           requires_grad=args.ifLearnZPinMSWATInMLSWA)
        else:
            errorMsg = ValueError(f"必须传入 zero_point，否则不能模拟负数激活值。")
            self.logger.info(errorMsg)
            raise errorMsg
        # endregion
        self.act = MLSWATFunction.apply

        self.v: Tensor  # 初始电压

    def forward(self, x: Tensor) -> Tensor:
        r"""

        :param x:
        :type x:Tensor
        :return:
        :rtype: Tensor
        """
        threshes = clamp_ste(self.threshes, 1e-4, 1e4)  # (L, )
        if self.args.asym or not self.args.disable_zero_point_in_sym:
            if self.zero_point:
                round_zero_point = clamp_ste(round_ste(self.zero_point), self.args.qmin, self.args.qmax)
            else:
                round_zero_point = None
        else:
            raise NotImplementedError(f"当前不支持非对称的量化的等价转换。")

        round_zero_point = round_zero_point.repeat(threshes.shape[0], )  # (L, )

        T, N, dimLast = x.shape
        x_reshaped = x.reshape(T, N, -1, self.args.sizeGroup)
        self.v = torch.ones_like(x_reshaped[0]).mul(threshes[0]) * 0.5  # (N, dimLast//G, G)
        Iz = round_zero_point / self.args.T
        Iz = Iz.mul((threshes[0] - threshes[-1]) / threshes.size(0)).sum()  # (L, )
        spikesOut = list()

        if self.args.avg_neuron:
            offsets = torch.einsum('i,i->', threshes, round_zero_point).sum(dim=0, keepdim=False)
            xS = x_reshaped.sum(dim=0).detach() + offsets
            maxVal = self.args.qmax * (threshes[0] - threshes[-1]) / threshes.size(0)
            minVal = torch.zeros_like(maxVal)
            xS = torch.clamp(xS, minVal, maxVal)
            xTemp = xS / self.args.T

        for t in range(self.args.T):
            if not self.args.avg_neuron:
                xTemp = x_reshaped[t].add(Iz)
            self.v = self.v.detach().add(xTemp.detach())
            output = self.act(self.v, threshes)

            self.v -= output.detach()
            output = output.detach().sub(Iz.detach())

            spikesOut.append(output)
        output = torch.stack(spikesOut, dim=0)
        if self.args.sizeGroup:
            output = output.reshape(T,
                                    N,
                                    dimLast)
        return output

    @property
    def threshes(self) -> Tensor:
        r"""
        softplus(x)=ln(1+e^x)，确保输出 > 0
        作用是保证实际上的 threshes 中的元素都为正数。
        并且计算真实阈值，保证各个阈值的偏序关系。
        :return: 返回满足偏序关系且从大到小排序的阈值序列。
        :rtype: Tensor
        """
        # 通过投影，确保每个增量严格 > 0。
        increments = F.softplus(self.raw_threshes)  # Tensor, requires_grad=True
        # 计算真实阈值。
        real_threshes = torch.cumsum(increments, dim=0)  # 前缀和
        # 对从小到大的阈值序列，颠倒为从大到小排序。
        real_threshes = torch.flip(real_threshes, dims=(0,))  # 反转顺序
        return real_threshes


class LMHTNeuron(nn.Module):
    r"""
    == 根据 PrefixQuant 和 LMHT 实现的整数倍脉冲神经元。==
    """

    def __init__(self,
                 args: Namespace,
                 scale: Tensor,
                 ifLearnScaleInSpike: bool,
                 zero_point: Optional[Tensor],
                 ifLearnZPInSpike: bool, ):
        r"""
        :param args:
        :type args: Namespace
        :param scale: TODO 形状需要确定：(1,1) or () 量化的时候使用 scale.data，作为脉冲阈值。脉冲神经元不用自己定义脉冲阈值。
        :type scale: Tensor
        :param ifLearnScaleInSpike:
        :type ifLearnScaleInSpike: bool
        :param zero_point: TODO 形状需要确定(1,1) or ()。量化的时候计算的 zero_point.data。
        :type zero_point:Tensor
        :param ifLearnZPInSpike:
        :type ifLearnZPInSpike: bool
        """
        super(LMHTNeuron, self).__init__()
        self.v: Tensor  # 初始电压。
        self.args = args
        self.act = MultiLevelFunction.apply

        self.scale = nn.Parameter(scale, requires_grad=ifLearnScaleInSpike)
        if zero_point is not None:
            self.zero_point = nn.Parameter(zero_point, requires_grad=ifLearnZPInSpike)
        # self.scale = nn.Parameter(scale,
        #                           requires_grad=args.ifLearnScaleInSpike)
        # self.zero_point = nn.Parameter(zero_point,
        #                                requires_grad=args.ifLearnZPInSpike)
        self.enable = True
        self.sizeGroup = 1

    def forward(self, x) -> Tensor:
        r"""

        :param x: (T, sizeB, lenSeq, dimFeat)
        :type x:Tensor
        :return:
        :rtype:Tensor
        """
        scale = clamp_ste(self.scale, 1e-4, 1e4)
        if self.args.asym or not self.args.disable_zero_point_in_sym:
            if self.zero_point:
                round_zero_point = clamp_ste(round_ste(self.zero_point), self.args.qmin, self.args.qmax)
            else:
                round_zero_point = None
        else:
            raise NotImplementedError(f"当前不支持非对称的量化的等价转换。")

        T, N, dimSeq = x.shape
        # T, B, dimSeq = x.shape
        xReshaped = x.reshape(T,
                              N,
                              -1,
                              self.sizeGroup,
                              # 1
                              )
        self.v = torch.ones_like(xReshaped[0, ...]) * scale * 0.5
        Iz = round_zero_point / self.args.T
        Iz = Iz * scale
        spikesOut = list()

        if self.args.avg_neuron:
            xS = xReshaped.sum(dim=0).detach().add(round_zero_point.mul(scale))
            maxVal = self.args.qmax * scale
            minVal = torch.zeros_like(maxVal)
            xS = torch.clamp(xS, minVal, maxVal)
            xTemp = xS / self.args.T

        for t in range(self.args.T):
            if not self.args.avg_neuron:
                xTemp = xReshaped[t, ...] + Iz
            self.v = self.v + xTemp
            output = self.act(self.v, scale, self.args.L)
            self.v = self.v - output
            output = output - Iz
            spikesOut.append(output)
        output = torch.stack(spikesOut, dim=0)
        if self.sizeGroup:
            output = output.reshape(self.args.T,
                                    N,
                                    dimSeq)

        return output


class FloorLayer(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


funcFloor = FloorLayer.apply


class QCFSWithPrefix(nn.Module):
    """
    == 用于 ANN 中的量化层 ==
    """

    def __init__(self,
                 args: Namespace,
                 numFeatures: int,
                 nameLogger: str,
                 maxFloatFeats: Tensor = None) -> None:
        r"""
        :param args:
        :type args: Namespace
        :param maxFloatFeats: TODO (sizeB, lenSeq) or (lenSeq, dimFeat)??
        :type maxFloatFeats: Tensor
        :param numFeatures: 如果是输入特征量化，则为对应原始 Linear 层的输入特征数，
                            如果为输出特征量化， 则为原始 Linear 层的输出特征数。
        :type numFeatures: int,
        :param nameLogger: 用于记录日志的 logger 的名字字符串。
        :type nameLogger: str
        """
        super().__init__()
        logger = logging.getLogger(nameLogger)
        # region 确定 scale 和 zero_point
        # TODO 这里来自于 Prefix 量化中为 input 计算的静态分支的 scale 和 zero_point，
        #  它那里的 scale 和 zero_point 都是可学习的。
        if args.ifMinMax:
            if maxFloatFeats is None:
                errorMessage = ValueError(
                    f"希望使用特征来计算 scale 和 zero_point 的时候，必须传入具体的 maxFloatFeats 值！")
                logger.error(errorMessage)
                raise errorMessage
            else:
                maxFeat = maxFloatFeats.amax(dim=-1, keepdim=True)
                self.scale = nn.Parameter(
                    (2 * maxFeat / (2 ** args.bitsForQuant - 1)).clamp(min=1e-4,
                                                                       max=1e4),
                    requires_grad=args.ifLearnScale)
                self.zero_point = nn.Parameter(
                    (2 ** args.bitsForQuant - 1) - 1 * torch.ones_like(self.scale),
                    requires_grad=args.ifLearnZP
                )
        else:
            self.scale = nn.Parameter(
                # torch.ones(size=()),
                torch.tensor(1 / (2 ** args.bitsForQuant - 1)),
                requires_grad=args.ifLearnScale
            )
            self.zero_point = nn.Parameter(
                torch.zeros(size=()),
                requires_grad=args.ifLearnZP
            )

        # endregion

        # region 用于前传中的超参
        self.shapeQuantized = [1, numFeatures]
        self.sizeGroup = numFeatures
        self.t: int = args.bitsForQuant  # 2^N -1
        # self.t = args.L * args.T
        self.phi: float = args.phi
        self.beta: float = args.betaForZP  # TODO 在argument 中设置，默认值 0.5，范围[0,1]
        self.gamma: float = args.gammaForZP  # TODO 在argument 中设置，默认值 0.5，范围[0,1]
        self.queryMin: float = 0
        self.queryMax: float = 2 ** args.bitsForQuant - 1
        # endregion

    def forward(self, x) -> Tensor:
        r"""
        :param x: (sizeBatch, lenSeq, dimFeat)
        :type x: Tensor
        :return: (sizeBatch, lenSeq, dimFeat)
        :rtype: Tensor
        """
        round_zero_point = clamp_ste(
            round_ste(self.zero_point),
            self.queryMin,
            self.queryMax)
        x = floor_ste(x / self.scale)
        x = x.add(round_zero_point)
        x = x.clamp(self.queryMin, self.queryMax)
        x = x.sub(round_zero_point)
        x = x.mul(self.scale)
        return x


# region正负脉冲的算子 by Gt

class MultiLevelFunctionPN(Function):
    @staticmethod
    def forward(ctx, input, th, L):
        k = (input / th).clamp(-L, L)
        # 进行标记
        # 对正部分应用floor, 负部分为0
        k_pos = torch.where(k > 0, k.floor(), torch.tensor(0.0))
        # 对负部分应用ceil, 正部分为0
        k_neg = torch.where(k < 0, k.ceil(), torch.tensor(0.0))
        k = k_pos + k_neg  # 合并正负部分
        out = k * th
        # k = ((input / th).floor() + zero_point).clamp(0, L)

        mask = ((input.detach() >= (L + 1) * th) *
                (input.detach() <= (L + 1) * th)).float()
        ctx.save_for_backward(mask)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors
        grad_input = grad_output * mask
        return grad_input, None, None


class LMHTNeuronPN(nn.Module):
    def __init__(self,
                 args: Namespace,
                 scale: float = 1.):
        r"""

        :param L:
        :param T:
        :param th:
        :param initial_mem:
        :param sigma:
        :param scale: 这里的 scale 应该是前一层数据记录的 scale？
        """
        super().__init__()
        self.v_threshold = nn.Parameter(torch.tensor([args.Thresh]),
                                        requires_grad=True)
        self.v: Tensor
        self.initial_mem = args.initial_mem * args.Thresh
        self.L = args.L
        self.T = args.T
        self.scale = scale
        self.averageT = args.averageT
        self.act = MultiLevelFunctionPN.apply

    def forward(self, x):
        self.v = torch.zeros(x.shape[1:])
        spike_pot = list()
        if self.averageT:
            x_mean_along_T = x.mean(dim=0, keepdim=False)
        else:
            x_mean_along_T = None
        for t in range(self.T):
            if self.averageT:
                self.v += x_mean_along_T
            else:
                self.v += x[t]
            output = self.act(self.v,
                              self.v_threshold,
                              self.L,
                              self.sigma)
            self.v -= output
            spike_pot.append(output)
        return torch.stack(spike_pot, dim=0)


class LMHTNeuronWithTriSpike(nn.Module):
    def __init__(self, L, T=2, th=1.0, initial_mem=0.0, sigma=0.5, scale=1.0, pulse_pos=3, pulse_neg=-2):
        super(LMHTNeuronWithTriSpike, self).__init__()
        self.v_threshold = nn.Parameter(torch.tensor([th]), requires_grad=True)
        self.v = None
        self.initial_mem = initial_mem * th
        self.L = L
        self.T = T
        self.sigma = sigma
        self.scale = scale
        self.pulse_pos = pulse_pos
        self.pulse_neg = pulse_neg
        # 继续使用多级量化激活函数
        self.act = MultiLevelFunction.apply

    def forward(self, x):
        self.v = torch.ones_like(x[0]) * self.initial_mem
        spike_pot = []

        for t in range(self.T):
            self.v = self.v + x[t]
            output = self.act(self.v, self.v_threshold, self.L, self.sigma)

            # 发射三元脉冲：正脉冲、负脉冲或无脉冲
            output_spikes = torch.zeros_like(output)
            output_spikes[output >= self.v_threshold] = self.pulse_pos
            output_spikes[output <= -self.v_threshold] = self.pulse_neg

            # 重置膜电位
            self.v = self.v - output_spikes

            spike_pot.append(output_spikes)

        return torch.stack(spike_pot, dim=0)


# endregion


# region FSNeuron

def get_spike_function(surrogate_type: str = 'sigmoid', slope: float = 1.0) -> Type[Function]:
    """
    Factory to generate surrogate-gradient-based spike functions.

    :param surrogate_type: One of ['triangular', 'sigmoid', 'fast_sigmoid', 'arctangent', 'hard_sigmoid']
    :param slope: Slope factor for smooth surrogates (like sigmoid)
    :return: callable spike function
    """
    surrogate_type = surrogate_type.lower()

    class SurrogateSpikeFunction(Function):
        @staticmethod
        def forward(ctx, v_scaled):
            ctx.save_for_backward(v_scaled)
            return v_scaled > 0

        @staticmethod
        def backward(ctx, grad_output):
            v_scaled, = ctx.saved_tensors
            if surrogate_type == 'triangular':
                grad = torch.clamp(1 - v_scaled.abs(), min=0.0)
            elif surrogate_type == 'sigmoid':
                s = slope * v_scaled
                grad = slope * torch.sigmoid(s) * (1 - torch.sigmoid(s))
            elif surrogate_type == 'fast_sigmoid':
                grad = slope / ((1 + slope * v_scaled.abs()) ** 2)
            elif surrogate_type == 'arctangent':
                grad = 1 / (1 + (slope * v_scaled).pow(2)) * slope
            elif surrogate_type == 'hard_sigmoid':
                grad = ((v_scaled > -1.0) & (v_scaled < 1.0)).float() * 0.5 * slope
            else:
                raise ValueError(f"Unsupported surrogate type: {surrogate_type}")
            return grad_output * grad

    return SurrogateSpikeFunction


class SpikeFunction(Function):
    @staticmethod
    def forward(ctx, v_scaled) -> Tensor:
        ctx.save_for_backward(v_scaled)
        return (v_scaled > 0).to(v_scaled.dtype)

    @staticmethod
    def backward(ctx, grad_output) -> Tensor:
        v_scaled, = ctx.saved_tensors
        dz_dv = torch.clamp(1 - v_scaled.abs(), min=0.0)
        grad_input = grad_output * dz_dv
        return grad_input


class FSNeuron(nn.Module):
    def __init__(self,
                 K,
                 args: argparse.Namespace,
                 nameLogger: str) -> Tensor:
        r"""
        :param K: number of time steps
        :type K: int
        :param args:
        :type args: argparse.Namespace
        :param nameLogger:
        :type nameLogger: str
        :return
        :rtype Tensor
        """
        super().__init__()
        self.K = K
        self.logger = logging.getLogger(nameLogger)

        # Learnable parameters: (K,) shape
        self.h = nn.Parameter(torch.abs(torch.randn(K)), requires_grad=True)
        self.d = nn.Parameter(torch.abs(torch.randn(K)), requires_grad=True)
        self.T = nn.Parameter(torch.randn(K), requires_grad=True)

        # for tracking
        self.print_spikes = args.print_spike  # False
        self.print_n_neurons = args.print_n_neurons  # False
        self.print_mean_stddev = args.print_mean_stddev  # False
        self.return_reg = args.return_reg  # False
        # self.spike_func = get_spike_function(args.typeGradInFS, 5).apply
        self.spike_func = SpikeFunction.apply

    def forward(self, x: torch.Tensor):
        r"""

        :param x: (B,L,D)
        :type x: Tensor
        :return: (B,L,D)
        :rtype: Tensor
        """
        if self.print_n_neurons:
            self.logger.debug(f'Number of neurons: {x[0].numel()}')

        if self.print_mean_stddev:
            self.logger.debug("Mean:", x.mean().item(), "Std:", x.std().item())
        v = x
        # z = torch.zeros_like(x)
        out = torch.zeros_like(x)
        # v_reg = torch.tensor(0., device=x.device, dtype=x.dtype)
        # z_reg = torch.tensor(0., device=x.device, dtype=x.dtype)

        for t in range(self.K):
            # v_scaled = (v - self.T[t]) / (v.abs() + 1)
            v_scaled = v - self.T[t]
            z = self.spike_func(v_scaled)
            # v_reg = v_reg + torch.square(F.relu(v_scaled.abs() - 1)).mean()
            # z_reg = z_reg + z.mean()
            if self.print_spikes:
                self.logger.debug(f"Spikes at t={t}:", z.sum().item())
            out = out + z * self.d[t]
            v = v - z * self.h[t]

        # if self.return_reg:
        #     return out, v_reg, z_reg
        # else:
        return out


class FS_MI(nn.Module):
    def __init__(self, num_params):
        super(FS_MI, self).__init__()
        self.h = nn.Parameter(torch.abs(torch.randn(num_params)))  # 重置幅度
        self.d = nn.Parameter(torch.abs(torch.randn(num_params)))  # 放电强度
        self.T = nn.Parameter(torch.randn(num_params))  # 阈值
        self.spike_func = SpikeFunction.apply

    def forward(self, x):
        """
        输入 x: shape (T, B, D) 或 (T, N)
        """
        T_steps = x.shape[0]
        v = torch.zeros_like(x[0])  # 初始化电压 v (B, D)
        outputs = []

        for t in range(T_steps):
            v = v + x[t]  # 每个时间步都“充电”
            spikes = self.spike_function(v - self.T[t])  # 判断是否超过阈值
            outputs.append(spikes * self.d[t])  # 输出加权脉冲
            v = v - spikes * self.h[t]  # 放电后降低电压（reset）

        return torch.stack(outputs, dim=0)  # (T, B, D)


# endregion

class TETLoss(nn.Module):
    def __init__(self,
                 nameLogger: str,
                 cali_with: str,
                 T: int
                 ):
        r"""
        :param nameLogger
        :type nameLogger: str
        :param cali_with: 'SNN' or 'Ori'
        :type cali_with: str
        :param T:
        :type T: int
        """
        super(TETLoss, self).__init__()
        self.logger = logging.getLogger(nameLogger)
        self.cali_with = cali_with
        self.T = T
        self.criterion = nn.MSELoss(
            # label_smoothing=optsLoss["label_smoothing"]
        )
        self.logger.info(f"使用的校准特征为： {self.cali_with}; 每个时刻使用的损失为：{type(self.criterion)}.")

    def forward(self,
                logits: Tensor,
                target: Tensor) -> Tensor:
        r"""

        :param logits:(T, B, ***)
        :type logits: Tensor
        :param target:(T, B, ***) or (B, ***)
        :type target: Tensor
        :return:
        :rtype: Tensor
        """
        assert self.T == logits.shape[0], f"特征的第 0 个维度(长度为 {logits.shape[0]} 应该等于网络的latency ({self.T})"
        loss = 0.
        for t in range(self.T):
            if self.cali_with == "Ori":
                loss += self.criterion(logits[t], target)
            elif self.cali_with == "SNN":
                loss += self.criterion(logits[t], target[t])
            else:
                raise NotImplementedError(f"检查 args.cali_with ({self.cali_with}),"
                                          f"必须为：SNN 或者 Ori.")
        if self.cali_with == "SNN":
            loss /= self.T
        return loss


class ActSWL(nn.Module):
    def __init__(self,
                 T: int,
                 hidden_act: str) -> None:
        r"""

        :param T:
        :type T: int
        :param hidden_act:
        :type hidden_act: str
        """
        super(ActSWL, self).__init__()
        self.act_fn = ACT2FN[hidden_act]
        self.T = T

    def forward(self,
                x: Tensor) -> Tensor:
        r"""

        :param x: (T, B, L, D)
        :type  x: Tensor
        :return:
        """
        X = torch.zeros_like(x[0])
        Y_pre = 0
        Out = []
        for t in range(self.T):
            X = X + x[t]
            Y = self.act_fn(X)
            Out.append(Y - Y_pre)
            Y_pre = Y
        act_out = torch.stack(Out, dim=0)
        return act_out
