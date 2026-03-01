import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer
import utils.hadamard_utils as hadamard_utils
from typing import List, Optional, Tuple, Union
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding,apply_rotary_pos_emb,LlamaRMSNorm,repeat_kv, rotate_half
import math
from transformers.cache_utils import Cache
import copy
# from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import logging
from transformers.activations import ACT2FN
logger = logging.get_logger(__name__)


def manual_scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    B, H, Q, D = q.shape
    K = k.size(2)

    # 计算注意力分数
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # [B, H, Q, K]

    # 应用 attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
        else:
            attn_scores = attn_scores + attn_mask

    # 应用 causal mask（上三角部分设为 -inf）
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(Q, K, dtype=torch.bool, device=q.device), diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

    # softmax 计算注意力权重
    attn_weights = F.softmax(attn_scores, dim=-1)

    # 应用 dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # 输出
    output = torch.matmul(attn_weights, v)
    return output


class quantSdpaLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, org_module: nn.Module, config: LlamaConfig, args=None):
        super().__init__()
        self.config = config
        self.layer_idx = org_module.layer_idx
        # if self.layer_idx is None:
        #     logger.warning_once(
        #         f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
        #         "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
        #         "when creating this class."
        #     )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.rotary_emb = copy.deepcopy(org_module.rotary_emb)

        self.k_proj = QuantLinear(
            org_module.k_proj,
        )
        self.v_proj = QuantLinear(
            org_module.v_proj,
        )
        self.q_proj = QuantLinear(
            org_module.q_proj,
        )
        self.o_proj = QuantLinear(
            org_module.o_proj,
        )
        self.flag = True
        if self.flag:
            self.q_Identity = QuantIdentity(org_module.q_proj.out_features)
            self.k_Identity = QuantIdentity(org_module.q_proj.out_features)
            self.v_Identity = QuantIdentity(org_module.q_proj.out_features)
            self.softmax_Identity = QuantIdentity(self.num_heads)
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
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        cos_unsq = cos.unsqueeze(1)
        sin_unsq = sin.unsqueeze(1)
        query_states = (query_states * cos_unsq) + (rotate_half(query_states) * sin_unsq)
        key_states = (key_states * cos_unsq) + (rotate_half(key_states) * sin_unsq)


        # In case static cache is used, it is an instance attribute.
        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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
            B, H, Q, D = key_states.shape
            # query_states = query_states.contiguous()
            # key_states = key_states.contiguous()
            # value_states = value_states.contiguous()
            query_states = query_states.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            key_states = key_states.transpose(1, 2).contiguous().view(bsz, Q, -1)
            value_states = value_states.transpose(1, 2).contiguous().view(bsz, Q, -1)
            query_states = self.q_Identity(query_states)
            value_states = self.v_Identity(value_states)
            key_states = self.k_Identity(key_states)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(B, Q, H, D).transpose(1, 2)
            value_states = value_states.view(B, Q, H, D).transpose(1, 2)
        
        # attn_output = manual_scaled_dot_product_attention(
        # # attn_output = F.scaled_dot_product_attention(
        #     query_states,
        #     key_states,
        #     value_states,
        #     attn_mask=causal_mask,
        #     dropout_p=self.attention_dropout if self.training else 0.0,
        #     is_causal=causal_mask is None and q_len > 1,
        # )
        
        B, H, Q, D = query_states.shape
        K = key_states.size(2)
        is_causal=causal_mask is None and q_len > 1
        # 计算注意力分数
        # attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(D)  # [B, H, Q, K]
        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1) / math.sqrt(D))   # [B, H, Q, K]

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
        attn_weights = F.softmax(attn_scores, dim=-1)
        # print(attn_weights.shape)
        if self.flag:
            b,head,d1,d2=attn_weights.shape
            attn_weights = attn_weights.view(bsz,self.num_heads,-1 ).transpose(1,2)
            attn_weights = self.softmax_Identity(attn_weights)
            attn_weights = attn_weights.transpose(1, 2).contiguous().view(b, head, d1, d2)
        # 应用 dropout
        dropout_p=self.attention_dropout if self.training else 0.0
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        # 输出
        attn_output = torch.matmul(attn_weights, value_states)
        
        

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value



class quantLlamaMLP(nn.Module):
    def __init__(self, org_module: nn.Module, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = QuantLinear(
            org_module.gate_proj,
        ) 
        self.up_proj = QuantLinear(
            org_module.up_proj,
        ) 
        self.down_proj = QuantLinear(
            org_module.down_proj,
        ) 
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
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
            out = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            down_proj = self.down_proj(out)

        return down_proj


# class quantLlamaAttention2(nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""

#     def __init__(self, org_module: nn.Module, config: LlamaConfig, args=None):
#         super().__init__()
#         self.config = config
#         self.layer_idx = org_module.layer_idx
#         # if self.layer_idx is None:
#         #     logger.warning_once(
#         #         f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
#         #         "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
#         #         "when creating this class."
#         #     )

#         self.attention_dropout = config.attention_dropout
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = self.hidden_size // self.num_heads
#         self.num_key_value_heads = config.num_key_value_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#         self.max_position_embeddings = config.max_position_embeddings
#         self.rope_theta = config.rope_theta
#         self.is_causal = True

#         if (self.head_dim * self.num_heads) != self.hidden_size:
#             raise ValueError(
#                 f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
#                 f" and `num_heads`: {self.num_heads})."
#             )

#         self.rotary_emb = copy.deepcopy(org_module.rotary_emb)

#         self.k_proj = QuantLinear(
#             org_module.k_proj,
#         )
#         self.v_proj = QuantLinear(
#             org_module.v_proj,
#         )
#         self.q_proj = QuantLinear(
#             org_module.q_proj,
#         )
#         self.o_proj = QuantLinear(
#             org_module.o_proj,
#         )
        
#         self.flag = False
#         if self.flag:
#             self.q_Identity = QuantIdentity()
#             self.k_Identity = QuantIdentity()
#             self.att_weight_Identity = QuantIdentity()
#             # self.qkt_matmul = QuantMatMul(
#             #      matmul_func=torch.matmul
#             # )
#             # self.pv_matmul = QuantMatMul(
#             #      matmul_func=torch.matmul
#             # )
        
#         self.use_weight_quant = False
#         self.use_act_quant = False
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[torch.LongTensor] = None,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         bsz, q_len, _ = hidden_states.size()

#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#         past_key_value = getattr(self, "past_key_value", past_key_value)
#         cos, sin = self.rotary_emb(value_states, position_ids)
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)

#         # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
#         if self.flag:
#             query_states = self.q_Identity(query_states)
#             key_states = self.k_Identity(key_states)
#             # attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
#         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

#         if attention_mask is not None:  # no matter the length, we just slice it
#             causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#             attn_weights = attn_weights + causal_mask

#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#         if self.flag:
#             attn_weights = self.att_weight_Identity(attn_weights)
#             # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
#             # value_states = self.pv_matmul.quant_x2(value_states)
#             # attn_output = self.pv_matmul(attn_weights, value_states)
#         attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
#         attn_output = torch.matmul(attn_weights, value_states)

#         if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.transpose(1, 2).contiguous()

#         attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

#         if self.config.pretraining_tp > 1:
#             attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
#             o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
#             attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
#         else:
#             attn_output = self.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None


#         return attn_output, attn_weights, past_key_value
    
    
    
class quantLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, org_module: nn.Module, config: LlamaConfig, args=None):
        super().__init__()
        self.config = config
        self.layer_idx = org_module.layer_idx
        # if self.layer_idx is None:
        #     logger.warning_once(
        #         f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
        #         "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
        #         "when creating this class."
        #     )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.rotary_emb = copy.deepcopy(org_module.rotary_emb)

        self.k_proj = QuantLinear(
            org_module.k_proj,
        )
        self.v_proj = QuantLinear(
            org_module.v_proj,
        )
        self.q_proj = QuantLinear(
            org_module.q_proj,
        )
        self.o_proj = QuantLinear(
            org_module.o_proj,
        )
        
        self.flag = False
        if self.flag:
            self.q_Identity = QuantIdentity()
            self.k_Identity = QuantIdentity()
            self.softmax_Identity = QuantIdentity()
            # self.qkt_matmul = QuantMatMul(
            #      matmul_func=torch.matmul
            # )
            # self.pv_matmul = QuantMatMul(
            #      matmul_func=torch.matmul
            # )
        
        self.use_weight_quant = False
        self.use_act_quant = False
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if self.flag:
            query_states = self.q_Identity(query_states)
            key_states = self.k_Identity(key_states)
            # attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.flag:
            attn_weights = self.softmax_Identity(attn_weights)
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            # value_states = self.pv_matmul.quant_x2(value_states)
            # attn_output = self.pv_matmul(attn_weights, value_states)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None


        return attn_output, attn_weights, past_key_value

class QuantMatMul(nn.Module):
    def __init__(
        self,
        x1_quant_params: dict = {},
        x2_quant_params: dict = {},
        disable_act_quant=False,
        matmul_func=torch.bmm,
    ):
        super().__init__()
        # de-activate the quantized forward default
        self.use_act_quant = False
        # initialize quantizer
        self.i_cluster_counts = None
        # self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params)
        # self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params)
        self.matmul_func = matmul_func

        self.disable_act_quant = disable_act_quant


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def quant_x1(self, x1):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
        return x1

    def quant_x2(self, x2):
        if self.use_act_quant:
            x2 = self.x2_quantizer(x2)
        return x2

    def forward(self, x1, x2):
        out = self.matmul_func(x1, x2)
        return out


class QuantIdentity(nn.Module):
    def __init__(
        self,
        out_features,
    ):
        super().__init__()
        # de-activate the quantized forward default
        self.use_act_quant = False
        # initialize quantizer
        # self.i_cluster_counts = None
        # self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params)
        # self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params)
        # self.matmul_func = matmul_func
        self.use_weight_quant = False
        # self.wbits = 16
        # self.input_bits = 16
        self.output_bits = 16
        # self.online_full_had=False
        # self.use_temporary_parameter=False
        # self.disable_act_quant = disable_act_quant
        self.out_features = out_features


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def forward(self, x):
        out = x
        if self.use_act_quant and self.output_bits < 16:
            out = self.input_quantizer(out)
        return out




class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
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
        self.use_weight_quant = False
        self.use_act_quant = False
        self.wbits = 16
        self.input_bits = 16
        self.output_bits = 16
        self.online_full_had=False
        self.use_temporary_parameter=False

    
    
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

        if self.use_act_quant and self.output_bits < 16:
            out = self.output_quantizer(out)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant




