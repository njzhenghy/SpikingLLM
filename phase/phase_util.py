import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaAttention, LlamaMLP, LlamaSdpaAttention, LlamaDecoderLayer
from utils.quant_utils import set_op_by_name
from phase.phase_layer import phaseSnnRMSNorm, phaseSnnSdpaLlamaAttention, phaseSnnLlamaMLP, phaseSnnLinear2, phaseSnnEmbedding, phaseSnnLinear, phaseSnnIdentity
from utils.model_utils import RMSN, get_kv_cache
from quantize.quantizer import UniformAffineQuantizer
import utils.hadamard_utils as hadamard_utils
from phase.phase_neuron import FSNeuron
import functools
from tqdm import tqdm
from utils.rotation_utils import QKRotationWrapper


def set_phase_model_time_step(model, T, logger):
    for name, module in model.named_modules():
        if hasattr(module, "T"):
            module.T = T
            #logger.info(f'set {name} T={T}')


def get_act_stat(model, dataloader, accumulate_type='max', prefixed_tokens=None, online_had=False):
    model.eval()
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    model_dim = model.config.hidden_size
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    device = next(model.parameters()).device
    act_stat = {}
    prefixed_length = len(prefixed_tokens) if prefixed_tokens is not None else 0

    if online_had:
        had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
    # ????????????????????????????
    # def stat_tensor(name, tensor, type):
    #     hidden_dim = tensor.shape[-1]
    #     tensor = tensor.view(-1, hidden_dim).abs().detach().unsqueeze (0)
    #     # ema_factor = 0.99
    #     # if accumulate_type == 'max':
    #     #     comming_max = torch.max(tensor, dim=0)[0].float().cpu()
    #     # elif accumulate_type == 'mean':
    #     #     comming_max = torch.mean(tensor, dim=0).float().cpu()
    #     key_name = f"{name}.{type}"
    #     if key_name in act_stat:
    #         act_stat[key_name] = torch.cat((act_stat[key_name], tensor.half()), 0)
    #     else:
    #         act_stat[key_name] = tensor.half()
    # ????????????????????????????
    def stat_tensor(name, tensor, type):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        ema_factor = 0.99
        if accumulate_type == 'max':
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        elif accumulate_type == 'mean':
            comming_max = torch.mean(tensor, dim=0).float().cpu()
        key_name = f"{name}.{type}"
        if key_name in act_stat:
            act_stat[key_name] = ema_factor * act_stat[key_name] + (1-ema_factor) * comming_max
        else:
            act_stat[key_name] = comming_max
    # ????????????????????????????
    def stat_input_hook(m, x, y, name):
        if 'apply_rotary_pos_emb_qk_rotation_wrapper' in name:
            input_Q = x[0].transpose(1, 2).flatten(-2)
            input_K = x[1].transpose(1, 2).flatten(-2)
            output_Q = y[0].transpose(1, 2).flatten(-2)
            output_K = y[1].transpose(1, 2).flatten(-2)
            if prefixed_length > 0:
                input_Q = input_Q[:,prefixed_length:, ]
                input_K = input_K[:,prefixed_length:, ]
                output_Q = output_Q[:,prefixed_length:, ]
                output_K = output_K[:,prefixed_length:, ]
            stat_tensor(name, input_Q, 'input_Q')
            stat_tensor(name, input_K, 'input_K')
            stat_tensor(name, output_Q, 'output_Q')
            stat_tensor(name, output_K, 'output_K')
        else:
            if isinstance(x, tuple):
                x = x[0]
            if prefixed_length > 0:
                if 'norm' in name:
                    x_ = x[:,prefixed_length:, ]
                    y_ = y[:,prefixed_length:, ]
                else:
                    x_ = x[:,:,prefixed_length:, ]
                    y_ = y[:,:,prefixed_length:, ]                    
            else:
                x_,y_ = x, y
            if online_had and 'down_proj' in name:
                x_ = hadamard_utils.matmul_hadU_cuda(x_, had_K, K)
            stat_tensor(name, x_, 'input')
            stat_tensor(name, y_, 'output')

    hooks = []
    for name, m in model.named_modules():
        # if isinstance(m, nn.Linear):
        if isinstance(m, (phaseSnnLinear,phaseSnnRMSNorm,phaseSnnIdentity,QKRotationWrapper)):
            # print(m)
            # if isinstance(m, QuantIdentity):
            #     print("get indentity")
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(len(dataloader)), desc='obtain activation stat'):
        data = dataloader[i][0]
        if prefixed_tokens is not None:
            data = torch.cat([torch.tensor([prefixed_tokens]),data],dim=1)
        model(data.to(device))

    for h in hooks:
        h.remove()

    return act_stat


def wrap_to_phase_model(model, T):
    '''
    replace nn.Linear and norm layer to correspond quantization counterparts
    '''
    for name, module in model.named_modules():
        
        if 'lm_head' in name:
            lm_head = phaseSnnLinear2(module, T=T)
            set_op_by_name(model, name, lm_head)
            del module
        elif isinstance(module, nn.Embedding):
            snnembedding = phaseSnnEmbedding(module, T=T)
            set_op_by_name(model, name, snnembedding)
            del module
        elif isinstance(module, (RMSN, LlamaRMSNorm)):
            snnnorm = phaseSnnRMSNorm(module,T=T)
            set_op_by_name(model, name, snnnorm)  
            del module 
        elif isinstance(module, LlamaSdpaAttention):
            snnAttention = phaseSnnSdpaLlamaAttention(module, module.config, T)
            set_op_by_name(model, name, snnAttention)  
            del module
        elif isinstance(module, LlamaMLP):
            snnMLP = phaseSnnLlamaMLP(module, module.config, T)
            set_op_by_name(model, name, snnMLP)  
            del module 


def register_online_had(model):
    for name, module in model.named_modules():
        if isinstance(module,phaseSnnLinear) and 'down_proj' in name:
            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
            module.online_full_had = True
            module.had_K = had_K
            module.K = K
            module.fp32_had = False


def init_weight_quantizer(args, model, logger, minmax_init=True):
    for name, module in model.named_modules():
        if isinstance(module,phaseSnnLinear):
            # layer_name = name.split('.')[-1]
            # wbits = args.special_w_quant_bit if layer_name in args.special_w_quant_layer else args.wbits
            wbits = args.wbits
            module.wbits = wbits
            if wbits >= 16:
                continue
            w_group_size=args.w_group_size
            w_asym=args.w_asym
            quantized_item_stat=module.weight if minmax_init else None
            module.use_weight_quant = True
            module.weight_quantizer = UniformAffineQuantizer(wbits, module.weight.shape,  w_asym, w_group_size,
                                                           quantized_item_stat=quantized_item_stat,
                                                           quant_type='weight',
                                                           minmax_init=minmax_init)
            sym_stat = "asymmetric" if w_asym else 'symmetric'
            logger.info(f'weight quantization: set {name} as w{wbits}g{w_group_size} {sym_stat} quantization')

tau_dict={}

def init_input_neuron(args, model, activation_stat, logger=None, neuron_parameter=None):
    for name, module in model.named_modules():
        # skip lm_head quantization
        if isinstance(module, phaseSnnLinear):
            # skip quant at norm layer
            layer_name = name.split('.')[-1]
            if layer_name in ['q_proj','k_proj','v_proj','up_proj','gate_proj'] or 'lm_head' in name:
                continue
            # if 'lm_head' in name:
            #     continue
            input_stat = activation_stat[f'{name}.input'] if activation_stat is not None else None
            quantized_shape = (1, module.in_features)
            module.use_act_quant = True

            if neuron_parameter is not None:
                num_grains, genotype, neuron_d, tau, neuron_h, neuron_theta = neuron_parameter[f'{name}.input']
                module.input_quantizer = FSNeuron(T=args.T, quantized_shape=quantized_shape, quantized_item_stat=input_stat, num_grains=num_grains, genotype=genotype, neuron_d=neuron_d, tau=tau, neuron_h=neuron_h, neuron_theta=neuron_theta, spike_one=args.spike_one)
            else:
                module.input_quantizer = FSNeuron(T=args.T, quantized_shape=quantized_shape, quantized_item_stat=input_stat)
                # tau = module.input_quantizer.tau
                # tau_dict[f'{name}.input'] = tau

            logger.info(f'input activation neuron: set {name}')
        elif isinstance(module,(phaseSnnRMSNorm)):
            # quantization for the input of q_proj/k_proj/v_porj/up_proj/gate_proj are putted in normalization layer
            output_stat = activation_stat[f'{name}.output'] if activation_stat is not None else None
            quantized_shape = (1, module.out_features)
            module.use_act_quant = True

            if neuron_parameter is not None:
                num_grains, genotype, neuron_d, tau, neuron_h, neuron_theta = neuron_parameter[f'{name}.output']
                module.output_quantizer = FSNeuron(T=args.T, quantized_shape=quantized_shape, quantized_item_stat=output_stat, num_grains=num_grains, genotype=genotype, neuron_d=neuron_d, tau=tau, neuron_h=neuron_h, neuron_theta=neuron_theta, spike_one=args.spike_one)
            else:
                module.output_quantizer = FSNeuron(T=args.T, quantized_shape=quantized_shape, quantized_item_stat=output_stat)
                # tau = module.output_quantizer.tau
                # tau_dict[f'{name}.output'] = tau

            logger.info(f'output activation neuron: set {name}')


def init_out_neuron(args, model, activation_stat, logger=None, neuron_parameter=None):
    # for the quantization of k/v output (kv-cache quantization)
    for name, module in model.named_modules():
        if isinstance(module,phaseSnnIdentity) and ('softmax_Identity' in name or 'q_Identity' in name):
            output_stat = activation_stat[f'{name}.output'] if activation_stat is not None else None
            quantized_shape = (1,module.out_features)
            module.use_act_quant = True

            if neuron_parameter is not None:
                if 'softmax_Identity' in name:
                    num_grains, genotype, neuron_d, tau, neuron_h, neuron_theta, neuron_v0 = neuron_parameter[f'{name}.input']
                    module.input_quantizer = FSNeuron(T=args.T, quantized_shape=quantized_shape, quantized_item_stat=output_stat, num_grains=num_grains, genotype=genotype, neuron_d=neuron_d, tau=tau, neuron_h=neuron_h, neuron_theta=neuron_theta, neuron_v0=neuron_v0, spike_one=False)
                else:
                    num_grains, genotype, neuron_d, tau, neuron_h, neuron_theta = neuron_parameter[f'{name}.input']
                    module.input_quantizer = FSNeuron(T=args.T, quantized_shape=quantized_shape, quantized_item_stat=output_stat, num_grains=num_grains, genotype=genotype, neuron_d=neuron_d, tau=tau, neuron_h=neuron_h, neuron_theta=neuron_theta, spike_one=args.spike_one)
            else:
                module.input_quantizer = FSNeuron(T=args.T, quantized_shape=quantized_shape, quantized_item_stat=output_stat)
                # if ('softmax_Identity' in name):
                #     tau = module.input_quantizer.tau
                #     module.input_quantizer.v0 = 0.5 * tau* 2**(-args.T)
                # tau = module.input_quantizer.tau
                # tau_dict[f'{name}.input'] = tau

            logger.info(f'input identity neuron: set {name}')
        elif isinstance(module,phaseSnnIdentity) and not ('softmax_Identity' in name or 'q_Identity' in name):
            output_stat = activation_stat[f'{name}.output'] if activation_stat is not None else None
            quantized_shape = (1,module.out_features)
            module.use_act_quant = True

            if neuron_parameter is not None:
                if 'silu_Identity' in name:
                    num_grains, genotype, neuron_d, tau, neuron_h, neuron_theta, neuron_v0 = neuron_parameter[f'{name}.input']
                    module.input_quantizer = FSNeuron(T=args.T, quantized_shape=quantized_shape, quantized_item_stat=output_stat, num_grains=num_grains, genotype=genotype, neuron_d=neuron_d, tau=tau, neuron_h=neuron_h, neuron_theta=neuron_theta, neuron_v0=neuron_v0, spike_one=args.spike_one)
                else:
                    num_grains, genotype, neuron_d, tau, neuron_h, neuron_theta = neuron_parameter[f'{name}.input']
                    module.input_quantizer = FSNeuron(T=args.T, quantized_shape=quantized_shape, quantized_item_stat=output_stat, num_grains=num_grains, genotype=genotype, neuron_d=neuron_d, tau=tau, neuron_h=neuron_h, neuron_theta=neuron_theta, spike_one=args.spike_one)
            else:
                module.input_quantizer = FSNeuron(T=args.T, quantized_shape=quantized_shape, quantized_item_stat=output_stat)
                # tau = module.input_quantizer.tau
                # tau_dict[f'{name}.input'] = tau

            logger.info(f'input identity neuron: set {name}')
        # elif isinstance(module, phaseSnnLinear) and ('up_proj' in name):
        #     input_stat = activation_stat[f'{name}.output'] if activation_stat is not None else None
        #     quantized_shape = (1, module.out_features)
        #     module.use_act_quant = True

        #     if neuron_parameter is not None:
        #         num_grains, genotype, neuron_d, tau, neuron_h, neuron_theta = neuron_parameter[f'{name}.output']
        #         module.output_quantizer = FSNeuron(T=args.T, quantized_shape=quantized_shape, quantized_item_stat=input_stat, num_grains=num_grains, genotype=genotype, neuron_d=neuron_d, tau=tau, neuron_h=neuron_h, neuron_theta=neuron_theta)
        #     else:
        #         module.output_quantizer = FSNeuron(T=args.T, quantized_shape=quantized_shape, quantized_item_stat=input_stat)
        #         # tau = module.output_quantizer.tau
        #         # tau_dict[f'{name}.output'] = tau

        #     logger.info(f'output activation neuron: set {name}')

    # torch.save(tau_dict, '/home/ubuntu/solar/PhaseSNN/GrainAnalysis/activation_dir/Llama-2-7B-hf-8bit/tau_dict.pth')

def set_quant_state(model, act_quant: bool = False):
    for m in model.modules():
        # if isinstance(m, QuantLinear):
        if isinstance(m, (phaseSnnLinear,phaseSnnIdentity,phaseSnnRMSNorm,QKRotationWrapper)):
            m.set_quant_state(act_quant)
