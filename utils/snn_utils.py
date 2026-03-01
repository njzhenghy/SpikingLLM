
import torch
import torch.nn as nn
from SNN.spike_layer import snnEmbedding
from utils.quant_utils import set_op_by_name
from quantize.quant_norm import QuantRMSNorm
from SNN.spike_layer import snnRMSNorm, snnSdpaLlamaAttention, snnLlamaMLP, snnLinear2
from quantize import int_linear_fake
from transformers.models.llama import LlamaModel
import copy


def replicate_past_key_values(past_key_values, T):
    """
    将原有的 past_key_values 沿时间维度复制 T 次，组成新的 [T, B, H, L, D] 结构。
    """
    new_past_kv = []
    for key, value in past_key_values:
        # key, value: [B, H, L, D]
        tmp_key = key[0].clone()/T
        tmp_value = value[0].clone()/T
        key_repeat = torch.stack([tmp_key for _ in range(T)], dim=0)       # [T, B, H, L, D]
        value_repeat = torch.stack([tmp_value for _ in range(T)], dim=0)   # [T, B, H, L, D]
        # key_repeat = torch.stack([key.clone() if t == 0 else torch.zeros_like(key) for t in range(T)], dim=0)
        # value_repeat = torch.stack([value.clone() if t == 0 else torch.zeros_like(value) for t in range(T)], dim=0)
        key_repeat = key_repeat.permute(1, 0, 2, 3, 4) # [B, T, H, L, D]
        value_repeat = value_repeat.permute(1, 0, 2, 3, 4)
        new_past_kv.append((key_repeat, value_repeat))
        
    return tuple(new_past_kv)


def wrap_to_snn_model(model, T, avg):
    '''
    replace nn.Linear and norm layer to correspond quantization counterparts
    '''
    # T = args.T
    # avg = args.avg_neuron
    # L = args.L
    for name, module in model.named_modules():
        
        # skip lm_head quantization
        if 'lm_head' in name:
            lm_head = snnLinear2(module, T=T)
            set_op_by_name(model, name, lm_head)
            del module
        # skip quantization of norm for lm_head
        # elif 'model.norm' in name:
        #     continue
        # if 'input_quantizer' in name and module.quant_type=='activation':
        #     neuron = LMHTNeuron(T)
        #     neuron.scale = module.scale
        #     set_op_by_name(model, name, neuron)
        #     del module
        # elif isinstance(module,(RMSN, LlamaRMSNorm)):
        #     quantnorm = QuantRMSNorm(module)
        #     set_op_by_name(model, name, quantnorm)
        #     del module
        # elif isinstance(module, LlamaModel):
        #     snnllama = snnEmbedding(module, T=T)
        #     set_op_by_name(model, name, snnllama)
        #     del module
        elif isinstance(module, nn.Embedding):
            snnembedding = snnEmbedding(module, T=T, avg=avg)
            set_op_by_name(model, name, snnembedding)
            del module
        elif isinstance(module, QuantRMSNorm):
            snnnorm = snnRMSNorm(module,T=T, avg=avg)
            set_op_by_name(model, name, snnnorm)  
            del module 
        elif isinstance(module, int_linear_fake.quantSdpaLlamaAttention):
            snnAttention = snnSdpaLlamaAttention(module, module.config, T, avg=avg)
            set_op_by_name(model, name, snnAttention)  
            del module
        elif isinstance(module, int_linear_fake.quantLlamaMLP):
            snnMLP = snnLlamaMLP(module, module.config, T, avg=avg)
            set_op_by_name(model, name, snnMLP)  
            del module 
            
def get_spike_config(args, quant_config):
    spike_config = {}
    spike_config["wbits"] = quant_config.wbits
    spike_config["w_group_size"] = quant_config.w_group_size
    spike_config["w_asym"] = quant_config.w_asym
    spike_config["input_bits"] = quant_config.input_bits
    spike_config["input_group_size"] = quant_config.input_group_size
    spike_config["input_asym"] = quant_config.input_asym
    spike_config["input_mode"] = quant_config.input_mode
    spike_config["k_bits"] = quant_config.k_bits
    spike_config["v_bits"] = quant_config.v_bits
    spike_config["output_asym"] = quant_config.output_asym
    spike_config["output_bits"] = quant_config.output_bits
    spike_config["output_mode"] = quant_config.output_mode
    spike_config["kv_group_size"] = quant_config.kv_group_size
    spike_config["kv_asym"] = quant_config.kv_asym
    spike_config["k_pre_rope"] = quant_config.k_pre_rope
    spike_config["kv_mode"] = quant_config.kv_mode
    spike_config["down_online_had"] = quant_config.down_online_had
    spike_config["qk_online_had"] = quant_config.qk_online_had
    spike_config["real_quant"] = quant_config.real_quant    
    spike_config["set_prefixed_tokens"] = quant_config.set_prefixed_tokens  
    spike_config["activation_clipping"] = quant_config.activation_clipping    
    spike_config["T"] = args.T
    spike_config["avg_neuron"] = args.avg_neuron
    return spike_config
     
def snn_wrap_to_quant_model(model):
    '''
    replace nn.Linear and norm layer to correspond quantization counterparts
    '''
    for name, module in model.named_modules():
        # skip lm_head quantization
        if 'lm_head' in name:
            lm_head = nn.Linear(in_features = module.in_features, out_features = module.out_features, bias = module.bias)
            lm_head.weight = nn.Parameter(module.weight.data.clone())
            set_op_by_name(model, name, lm_head)
            del module
        elif isinstance(module, snnEmbedding):
            embedding = nn.Embedding(num_embeddings=module.oriebd.num_embeddings, embedding_dim=4096, padding_idx=module.oriebd.padding_idx, max_norm=module.oriebd.max_norm, norm_type=module.oriebd.norm_type, scale_grad_by_freq=module.oriebd.scale_grad_by_freq, sparse=module.oriebd.sparse)
            embedding.weight = nn.Parameter(module.weight.data.clone())
            set_op_by_name(model, name, embedding)
            del module
        elif isinstance(module, snnRMSNorm):
            norm = QuantRMSNorm(module)
            for identity_name in ['input_layernorm', 'post_attention_layernorm', 'norm']:
                module.output_quantizer.ori.scale = module.output_quantizer.scale # 用SNN的scale覆盖ANN的scale
                setattr(norm, 'output_quantizer', module.output_quantizer.ori)
                setattr(norm.output_quantizer, 'initv', module.output_quantizer.initv)
            set_op_by_name(model, name, norm)
            del module 
        elif isinstance(module, snnSdpaLlamaAttention):
            attention = int_linear_fake.quantSdpaLlamaAttention(module, module.config)
            for proj_name in ['k_proj', 'v_proj', 'q_proj', 'o_proj']:
                setattr(getattr(attention, proj_name), 'weight_quantizer', getattr(module, proj_name).weight_quantizer)
            for identity_name in ['o_proj', 'q_Identity', 'k_Identity', 'v_Identity', 'weight_Identity']:
                identity_module = getattr(module, identity_name)
                if hasattr(identity_module, 'input_quantizer'):
                    identity_module.input_quantizer.ori.scale = identity_module.input_quantizer.scale   # 用SNN的scale覆盖ANN的scale
                    setattr(getattr(attention, identity_name), 'input_quantizer', identity_module.input_quantizer.ori)
                    setattr(getattr(attention, identity_name).input_quantizer, 'initv', identity_module.input_quantizer.initv)
            set_op_by_name(model, name, attention)
            del module 
        elif isinstance(module, snnLlamaMLP):
            quantMLP = int_linear_fake.quantLlamaMLP(module, module.config)
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                setattr(getattr(quantMLP, proj_name), 'weight_quantizer', getattr(module, proj_name).weight_quantizer)
            module.down_proj.input_quantizer.ori.scale = module.down_proj.input_quantizer.scale
            setattr(quantMLP.down_proj, 'input_quantizer', module.down_proj.input_quantizer.ori)
            setattr(quantMLP.down_proj.input_quantizer, 'initv', module.down_proj.input_quantizer.initv)
            set_op_by_name(model, name, quantMLP)
            del module