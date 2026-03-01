import torch
import torch.nn as nn
from torch.nn import functional as F
from SNN.spike_neuron import LMHTNeuron
from transformers.models.llama.configuration_llama import LlamaConfig
import copy
from quantize.int_linear_fake import quantSdpaLlamaAttention
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding,apply_rotary_pos_emb,rotate_half,LlamaRMSNorm,repeat_kv
import utils.hadamard_utils as hadamard_utils
import math
from transformers.activations import ACT2FN
import math
from transformers.models.llama.modeling_llama import LLAMA_START_DOCSTRING, LlamaPreTrainedModel, LLAMA_INPUTS_DOCSTRING, LlamaDecoderLayer
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

class snnLinear2(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module,
        T,
        avg = True
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight) # trainable
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.T = T
    
    def forward(self, input: torch.Tensor):
        BT, D, L = input.shape
        bsz = BT // self.T
        input = input.view(self.T, bsz, D, L)
        bias = self.bias

        out = self.fwd_func(input, self.weight, bias, **self.fwd_kwargs)
        
        return out.sum(dim=0)


class snnLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module,
        T,
        avg=True
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight) # trainable
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = org_module.use_weight_quant
        self.use_act_quant = org_module.use_act_quant
        self.wbits = org_module.wbits
        self.input_bits = org_module.input_bits
        self.output_bits = org_module.output_bits
        self.online_full_had=org_module.online_full_had
        self.use_temporary_parameter=org_module.use_temporary_parameter
        # if self.use_act_quant and self.input_bits < 16:
        L = math.ceil((2**self.input_bits - 1)/T)
        # L = math.ceil((2**self.input_bits)/T)
        self.avg = avg
        if self.use_act_quant and self.input_bits < 16:
            self.input_quantizer = LMHTNeuron(L, org_module.input_quantizer, T=T, avg=self.avg)
        self.weight_quantizer = org_module.weight_quantizer
        if self.online_full_had:
            self.fp32_had = org_module.fp32_had
            self.had_K = org_module.had_K
            self.K = org_module.K
    
    def forward(self, input: torch.Tensor):
        input_dtype = input.dtype

        # Rotate, if needed
        if self.online_full_had:
            if self.fp32_had: # Full Hadamard in FP32
                input = hadamard_utils.matmul_hadU_cuda(input.float(), self.had_K, self.K).to(input_dtype)
            else: # Full Hadamard in FP16
                input = hadamard_utils.matmul_hadU_cuda(input, self.had_K, self.K)
                
        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        bias = self.bias
            
        if self.use_weight_quant and self.wbits < 16:
            weight = self.weight_quantizer(weight)

        if self.use_act_quant and self.input_bits < 16:
            input = self.input_quantizer(input)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        return out


class snnIdentity(nn.Module):
    def __init__(
        self,
        org_module, T, avg=True
    ):
        super().__init__()
        # de-activate the quantized forward default
        # self.use_act_quant = org_module.use_act_quant
        # initialize quantizer
        # self.i_cluster_counts = org_module.use_act_quant
        # self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params)
        # self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params)
        # self.matmul_func = matmul_func
        # self.use_weight_quant = org_module.use_weight_quant
        self.use_act_quant = org_module.use_act_quant
        # self.wbits = org_module.wbits
        # self.input_bits = org_module.input_bits
        self.output_bits = org_module.output_bits
        # self.online_full_had=org_module.use_act_quant
        # self.use_temporary_parameter=org_module.use_act_quant
        # self.disable_act_quant = org_module.use_act_quant
        self.out_features = org_module.out_features
        L = math.ceil((2**self.output_bits - 1)/T)
        # L = math.ceil((2**self.output_bits)/T)
        self.avg = avg
        if self.use_act_quant and self.output_bits < 16:
            self.input_quantizer = LMHTNeuron(L, org_module.input_quantizer, T=T, avg=self.avg)

    def forward(self, x):
        out = x
        if self.use_act_quant and self.output_bits < 16:
            out = self.input_quantizer(out)
        return out


class ActSWL(nn.Module):
    def __init__(self,
                 T: int,
                 hidden_act: str) -> None:
        r"""

        :param T:
        :type T: int
        :param hidden_act: 默认值 "silu"
        :type hidden_act: str
        """
        super(ActSWL, self).__init__()
        self.act_fn = ACT2FN[hidden_act]
        self.T = T

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
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


def seq_matmul(A, B):
    S_q = A.cumsum(dim=0)
    S_k = B.cumsum(dim=0)
    # term1: S_q @ k^T
    term1 = torch.matmul(S_q, B) 
    # term2: q @ S_k^T
    term2 = torch.matmul(A, S_k)  
    # term3: q @ k^T
    term3 = torch.matmul(A, B)    
    # 合并计算结果并归一化
    attn_scores = term1 + term2 - term3
    return attn_scores
   
class snnSdpaLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, org_module, config, T, avg=True, args=None):
        super().__init__()
        self.config = config
        self.layer_idx = org_module.layer_idx
        # if self.layer_idx is None:
        #     logger.warning_once(
        #         f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
        #         "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
        #         "when creating this class."
        #     )
        self.avg = avg
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.T = T
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.rotary_emb = copy.deepcopy(org_module.rotary_emb)

        self.k_proj = snnLinear(
            org_module.k_proj, T, self.avg
        )
        self.v_proj = snnLinear(
            org_module.v_proj, T, self.avg
        )
        self.q_proj = snnLinear(
            org_module.q_proj, T, self.avg
        )
        self.o_proj = snnLinear(
            org_module.o_proj, T, self.avg
        )
        self.flag = True
        if self.flag:
            self.q_Identity = snnIdentity(org_module.q_Identity, T, self.avg)
            self.k_Identity = snnIdentity(org_module.k_Identity,T, self.avg)
            self.v_Identity = snnIdentity(org_module.v_Identity, T, self.avg)
            self.softmax_Identity = snnIdentity(org_module.softmax_Identity, T, self.avg)
            # self.qkt_matmul = QuantMatMul(
            #     args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul
            # )
            # self.pv_matmul = QuantMatMul(
            #     args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul
            # )
        
        self.use_weight_quant = False
        self.use_act_quant = False
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bszT, q_len, _ = hidden_states.size()
        bsz = bszT//self.T
        T = self.T
        hidden_states = hidden_states.view(T, bsz, q_len, -1)
        

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(-1, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(-1, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(-1, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        
        q1, k1 = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        cos_unsq = cos.unsqueeze(1)
        sin_unsq = sin.unsqueeze(1)
        query_states = (query_states * cos_unsq) + (rotate_half(query_states) * sin_unsq)
        key_states = (key_states * cos_unsq) + (rotate_half(key_states) * sin_unsq)

        # query_states = query_states.view(T, bsz, self.num_heads, q_len, self.head_dim)
        key_states = key_states.view(T, bsz, self.num_key_value_heads, q_len, self.head_dim)
        value_states = value_states.view(T, bsz, self.num_key_value_heads, q_len, self.head_dim)
        
        # # In case static cache is used, it is an instance attribute.
        # past_key_value = getattr(self, "past_key_value", past_key_value)
        # past_key_value_T = [copy.deepcopy(past_key_value) for _ in range(T)]
        # if past_key_value is not None:
        #     key_states_tmp = []
        #     value_states_tmp = []
        #     for t in range(T):
        #         key_t = key_states[t]
        #         value_t = value_states[t]
        #         # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #         cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #         key_states_, value_states_ = past_key_value_T[t].update(key_t, value_t, self.layer_idx, cache_kwargs)
        #         key_states_tmp.append(key_states_)
        #         value_states_tmp.append(value_states_)
        #     key_states = torch.stack(key_states_tmp, dim=0)
        #     value_states = torch.stack(value_states_tmp, dim=0)
        # T, B, H, Q, D = key_states.shape
        # key_states = key_states.view(-1, H, Q, D)
        # value_states = value_states.view(-1, H, Q, D)
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # key_states = key_states.sum(dim=0)
        # value_states = value_states.sum(dim=0)
        # past_key_value = getattr(self, "past_key_value", past_key_value)
        # if past_key_value is not None:
        #     # print("use cache")
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # key_states /= self.T
        # key_states.unsqueeze_(0)
        # key_states = key_states.repeat(self.T, 1, 1, 1, 1)
        # value_states /= self.T
        # value_states.unsqueeze_(0)
        # value_states = value_states.repeat(self.T, 1, 1, 1, 1)
        key_states = key_states.permute(1, 0, 2, 3, 4)
        value_states = value_states.permute(1, 0, 2, 3, 4)
        past_key_value = getattr(self, "past_key_value", past_key_value)
        if past_key_value is not None:
            # print("use cache")
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        key_states = key_states.permute(1, 0, 2, 3, 4)
        value_states = value_states.permute(1, 0, 2, 3, 4)
        T, B, H, Q, D = key_states.shape
        key_states = key_states.contiguous().view(-1, H, Q, D)
        value_states = value_states.contiguous().view(-1, H, Q, D)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        # if query_states.device.type == "cuda" and causal_mask is not None:
        #     query_states = query_states.contiguous()
        #     key_states = key_states.contiguous()
        #     value_states = value_states.contiguous()
        
        # In case we are not compiling, we may set `causal_mask` to None, which is required to dispatch to SDPA's Flash Attention 2 backend, rather
        # relying on the `is_causal` argument.
        if self.flag:
            # print(query_states.shape)
            # print(key_states.shape)
            # print(value_states.shape)
            BT, H, Q, D = key_states.shape
            # query_states = query_states.contiguous()
            # key_states = key_states.contiguous()
            # value_states = value_states.contiguous()
            
            query_states = query_states.transpose(1, 2).contiguous().view(T, bsz, q_len, -1)
            key_states = key_states.transpose(1, 2).contiguous().view(T, bsz, Q, -1)
            value_states = value_states.transpose(1, 2).contiguous().view(T, bsz, Q, -1)
            query_states = self.q_Identity(query_states)
            value_states = self.v_Identity(value_states)
            key_states = self.k_Identity(key_states)
            query_states = query_states.view(T, bsz, q_len, self.num_heads, self.head_dim).transpose(2, 3)
            key_states = key_states.view(T, bsz, Q, H, D).transpose(2, 3)
            value_states = value_states.view(T, bsz, Q, H, D).transpose(2, 3)
        
        # attn_output = manual_scaled_dot_product_attention(
        # # attn_output = F.scaled_dot_product_attention(
        #     query_states,
        #     key_states,
        #     value_states,
        #     attn_mask=causal_mask,
        #     dropout_p=self.attention_dropout if self.training else 0.0,
        #     is_causal=causal_mask is None and q_len > 1,
        # )
        
        T, B, H, Q, D = query_states.shape
        K = key_states.size(3)
        is_causal=causal_mask is None and q_len > 1
        # a = torch.matmul(query_states.sum(dim=0), key_states.sum(dim=0).transpose(-2, -1)) / math.sqrt(D)
        
        # attn_scores = seq_matmul(query_states, key_states.transpose(-2, -1))
        # attn_scores = attn_scores / math.sqrt(D)
        attn_scores = seq_matmul(query_states, key_states.transpose(-2, -1) / math.sqrt(D))
        T, B, H, Q, K = attn_scores.shape
        attn_scores = attn_scores.view(-1, H, Q, K)

        # 应用 attention mask
        if causal_mask is not None:
            if causal_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))
            else:
                attn_scores = attn_scores + causal_mask

        # 应用 causal mask（上三角部分设为 -inf）
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(Q, K, dtype=torch.bool, device=query_states.device), diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # softmax 计算注意力权重
        
        BT, Q, K, D = attn_scores.shape
        attn_scores = attn_scores.view(T, bsz, Q, K, D)
        X = torch.zeros_like(attn_scores[0])
        Y_pre = 0
        Out = []
        for t in range(self.T):
            X = X + attn_scores[t]
            Y = F.softmax(X, dim=-1)
            Out.append(Y-Y_pre)
            Y_pre = Y
        attn_weights = torch.stack(Out, dim=0)
        # a = Out.sum(dim=0)
        
        # attn_weights = F.softmax(attn_scores.sum(dim=0), dim=-1)
        
        # print(attn_weights.shape)
        if self.flag:
            T, B, head,d1,d2=attn_weights.shape
            attn_weights = attn_weights.view(T, bsz, head, -1 ).transpose(2,3)
            attn_weights = self.softmax_Identity(attn_weights)
            attn_weights = attn_weights.transpose(2, 3).contiguous().view(-1, head, d1, d2)
        # 应用 dropout
        dropout_p=self.attention_dropout if self.training else 0.0
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        # 输出
        # a = torch.matmul(attn_weights.sum(dim=0), value_states.sum(dim=0))
        TB, head,d1,d2=attn_weights.shape
        attn_weights = attn_weights.view(T, bsz, head, d1, d2)
        attn_output = seq_matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(2, 3).contiguous()
        attn_output = attn_output.view(T, bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)
        
        attn_output = attn_output.view(-1, q_len, self.hidden_size)

        return attn_output, None, past_key_value


def compute_hadamard_v2(A, B):
    """
    计算满足 sum(C_m) = (sum(A_i)) ⊙ (sum(B_j)) 的 C_m 矩阵（适用于 A, B 为 (T, bsz, Q, K)）
    
    参数:
        A: torch.Tensor, 形状为 (T, bsz, Q, K)
        B: torch.Tensor, 形状为 (T, bsz, Q, K)
    
    返回:
        torch.Tensor: 形状为 (T, bsz, Q, K)，包含所有 C_m 矩阵
    """
    T, bsz, Q, K = A.shape

    # 计算所有 A_i ⊙ B_j 外积组合 (T, T, bsz, Q, K)
    A_exp = A.unsqueeze(1)  # (T, 1, bsz, Q, K)
    B_exp = B.unsqueeze(0)  # (1, T, bsz, Q, K)
    outer_hadamard = A_exp * B_exp  # (T, T, bsz, Q, K)

    # 对角项：A_m ⊙ B_m
    diag_terms = torch.diagonal(outer_hadamard, dim1=0, dim2=1)  # (bsz, Q, K, T)
    diag_terms = diag_terms.permute(3, 0, 1, 2)  # (T, bsz, Q, K)

    # 非对角项的平均 (A_i ⊙ B_m + A_m ⊙ B_i)/2
    transposed = outer_hadamard.permute(1, 0, 2, 3, 4)  # (T, T, bsz, Q, K)
    off_diag_terms = (outer_hadamard + transposed) / 2  # (T, T, bsz, Q, K)

    # 构造 mask 以去除对角项
    mask = ~torch.eye(T, dtype=torch.bool, device=A.device)  # (T, T)
    off_diag_sum = off_diag_terms[mask].view(T, T-1, bsz, Q, K).sum(dim=1)  # (T, bsz, Q, K)

    # 最终组合
    C = diag_terms + off_diag_sum  # (T, bsz, Q, K)

    # 验证
    # sum_C = C.sum(dim=0)  # (bsz, Q, K)
    # sum_A = A.sum(dim=0)  # (bsz, Q, K)
    # sum_B = B.sum(dim=0)  # (bsz, Q, K)
    # expected = sum_A * sum_B  # (bsz, Q, K)
    # print(torch.allclose(sum_C, expected, atol=1e-5))
    # assert torch.allclose(sum_C, expected, atol=1e-5), "验证失败"

    return C


def compute_hadamard_v3(A, B):
    """
    计算满足 sum(C_m) = (sum(A_i)) ⊙ (sum(B_j)) 的 C_m 矩阵（适用于 A, B 为 (T, bsz, Q, K)）
    
    参数:
        A: torch.Tensor, 形状为 (T, bsz, Q, K)
        B: torch.Tensor, 形状为 (T, bsz, Q, K)
    
    返回:
        torch.Tensor: 形状为 (T, bsz, Q, K)，包含所有 C_m 矩阵
    """
    T, bsz, Q, K = A.shape

    # 计算所有 A_i ⊙ B_j 外积组合 (T, T, bsz, Q, K)
    A_exp = A.unsqueeze(1)  # (T, 1, bsz, Q, K)
    B_exp = B.unsqueeze(0)  # (1, T, bsz, Q, K)
    outer_hadamard = A_exp.float() * B_exp.float()  # (T, T, bsz, Q, K)

    # 对角项：A_m ⊙ B_m
    diag_terms = torch.diagonal(outer_hadamard, dim1=0, dim2=1)  # (bsz, Q, K, T)
    diag_terms = diag_terms.permute(3, 0, 1, 2)  # (T, bsz, Q, K)

    # 非对角项的平均 (A_i ⊙ B_m + A_m ⊙ B_i)/2
    transposed = outer_hadamard.permute(1, 0, 2, 3, 4)  # (T, T, bsz, Q, K)
    off_diag_terms = (outer_hadamard + transposed) / 2  # (T, T, bsz, Q, K)

    # 构造 mask 以去除对角项
    mask = ~torch.eye(T, dtype=torch.bool, device=A.device)  # (T, T)
    off_diag_sum = off_diag_terms[mask].view(T, T-1, bsz, Q, K).sum(dim=1)  # (T, bsz, Q, K)

    # 最终组合
    C = diag_terms + off_diag_sum  # (T, bsz, Q, K)

    # 验证
    # sum_C = C.sum(dim=0)  # (bsz, Q, K)
    # sum_A = A.sum(dim=0)  # (bsz, Q, K)
    # sum_B = B.sum(dim=0)  # (bsz, Q, K)
    # expected = sum_A * sum_B  # (bsz, Q, K)
    # print(torch.allclose(sum_C, expected, atol=1e-5))
    # assert torch.allclose(sum_C, expected, atol=1e-5), "验证失败"

    return C.half()

class snnLlamaMLP(nn.Module):
    def __init__(self, org_module, config, T, avg=True):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.avg = avg
        self.gate_proj = snnLinear(
            org_module.gate_proj, T, self.avg
        ) 
        self.up_proj = snnLinear(
            org_module.up_proj, T, self.avg
        ) 
        self.down_proj = snnLinear(
            org_module.down_proj, T, self.avg
        ) 
        self.act_fn = ACT2FN[config.hidden_act]
        self.T = T

    def forward(self, x):
        T= self.T
        BT, D, L  = x.shape
        bsz = BT // T
        x = x.view(T, bsz, D, L)
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            out_up = self.up_proj(x)
            out_g = self.gate_proj(x)
            
            # out_g = out_g.view(T, bsz, Q, K)
            X = torch.zeros_like(out_g[0])
            Y_pre = 0
            Out = []
            for t in range(self.T):
                X = X + out_g[t]
                Y = self.act_fn(X)
                Out.append(Y-Y_pre)
                Y_pre = Y
            act_out_g = torch.stack(Out, dim=0)
            out = compute_hadamard_v2(act_out_g, out_up)
            if torch.isnan(out).any() or torch.isinf(out).any():
                out = compute_hadamard_v3(act_out_g, out_up)
            # out = 0.5 * (act_out_g * out_up.sum(dim=0)) + 0.5 * (out_up * act_out_g.sum(dim=0)) 
            down_proj = self.down_proj(out)
            down_proj = down_proj.view(-1, D, L)
        return down_proj


class snnLlamaMLP_SWL(nn.Module):
    def __init__(self, org_module, config, T, avg=True):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.avg = avg
        self.gate_proj = snnLinear(
            org_module.gate_proj, T, self.avg
        )
        self.up_proj = snnLinear(
            org_module.up_proj, T, self.avg
        )
        self.down_proj = snnLinear(
            org_module.down_proj, T, self.avg
        )
        if self.config.pretraining_tp > 1:
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_func = ActSWL(T=T,
                                   hidden_act=config.hidden_act)
        self.T = T

    def forward(self, x):
        r"""
        :param x: (T*B, L, D)
        """
        # print(f"shape of input: {x.shape}")  # torch.Size([8, 2048, 4096]) 这里对的。T=2, B=4.
        T = self.T
        BT, D, L = x.shape
        bsz = BT // T
        x = x.reshape(T, bsz, D, L)
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            out_up = self.up_proj(x)
            out_g = self.gate_proj(x)
            # out = self.act_fn(out_g * out_up)
            act_out_g = self.act_func(out_g)  # (T,B,L,D)
            out = compute_hadamard_v2(act_out_g, out_up)
            # endregion
            down_proj = self.down_proj(out)
            down_proj = down_proj.reshape(-1, D, L)
        return down_proj


class snnRMSNorm(nn.Module):
    def __init__(self, 
                ori_norm,
                T=2,
                avg = True
                ):
        super().__init__()
        self.register_buffer('weight',ori_norm.weight)
        self.bias = None
        self.variance_epsilon = ori_norm.variance_epsilon
        self.use_temporary_parameter = False
        # self.use_act_quant = False
        self.output_bits = ori_norm.output_bits
        self.out_features = self.weight.shape[-1]
        # self.register_buffer('scale',ori_norm.weight)
        self.T = T
        self.avg = avg
        L = math.ceil((2**self.output_bits - 1)/T)
        # L = math.ceil((2**self.output_bits)/T)
        self.output_quantizer = LMHTNeuron(L, ori_norm.output_quantizer, T=T, avg=self.avg)


    def forward(self, x):
        BT, D, L = x.shape
        B = BT // self.T
        x = x.view(self.T, B, D, L)
        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        
        X = torch.zeros_like(x[0])
        Y_pre = 0
        Out = []
        for t in range(self.T):
            X = X + x[t]
            variance = X.pow(2).mean(-1, keepdim=True)
            Y = X * torch.rsqrt(variance + self.variance_epsilon)
            Y =  Y.to(input_dtype) * weight
            Out.append(Y-Y_pre)
            Y_pre = Y
        Out = torch.stack(Out, dim=0)   
        Out = self.output_quantizer(Out)
        Out = Out.view(-1, D, L)        
        
        # variance = x.pow(2).mean(-1, keepdim=True)
        # x = x * torch.rsqrt(variance + self.variance_epsilon)
        # x =  x.to(input_dtype) * weight
        # Out = self.output_quantizer(x)
        # Out = Out.view(-1, D, L)

        return Out


class snnEmbedding(nn.Module):
    def __init__(
        self,
        original_embedding,
        T=4,
        avg=True
    ) -> None:
        super().__init__()
        self.oriebd = original_embedding
        self.register_buffer('weight', self.oriebd.weight)
        self.fwd_func = F.embedding
        self.T = T
        self.avg = avg

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input.shape = [B, 2048]
        # input.unsqueeze_(0)
        # input = input / self.T
        # input = input.repeat(self.T,  1,  1)
        # zeros = torch.zeros(self.T - 1, *input.shape[1:], device=input.device, dtype=input.dtype)  # shape: (T-1, B, D)
        # input = torch.cat([input, zeros], dim=0)  # shape: (T, B, D)

        out = self.fwd_func(
            input,
            self.weight,
            self.oriebd.padding_idx,
            self.oriebd.max_norm,
            self.oriebd.norm_type,
            self.oriebd.scale_grad_by_freq,
            self.oriebd.sparse,
        )
        B,  L, D=out.shape
        out = out/self.T
        out.unsqueeze_(0)
        out = out.repeat(self.T, 1, 1,  1)
        # print(out.shape)
        # sys.exit(0)
        # out.shape = [4, B, 2048, 4096]
        
        # out = out.unsqueeze(0)
        # zeros = torch.zeros(self.T - 1, *out.shape[1:], device=out.device, dtype=out.dtype)  # shape: (T-1, B, L, D)
        # out = torch.cat([out, zeros], dim=0)  # shape: (T, B, L, D)
        
        return out.view(-1, L, D)