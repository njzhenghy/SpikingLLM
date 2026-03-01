import os
import getpass
import sys
import random
import numpy as np
import torch
import utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map, dispatch_model
from utils.quant_utils import wrap_to_quant_model, init_weight_quantizer, init_input_quantizer, init_out_quantizer, register_online_had, init_k_quantizer, init_v_quantizer
import utils.model_utils as model_utils
import utils.rotation_utils as rotation_utils
from main import evaluate
from utils.train_utils import load_json_as_namespace,create_logger
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from utils.snn_utils import wrap_to_snn_model, replicate_past_key_values, get_spike_config
from utils.block_calibration import calibration_copy
from utils.data_utils import get_loaders
from utils.quant_utils import get_quant_config
from utils import train_utils
import copy

torch.backends.cudnn.benchmark = True

def main():
    username = getpass.getuser()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_model_path", type=str, help="model path of quantized model")
    parser.add_argument("--output_dir", default="./log/test_snn", type=str, help="direction of logging file")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--T", type=int, default=2
                        , help="time step")
    # parser.add_argument("--L", type=int, default=8, help="spike neuron")
    parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--avg_neuron", action="store_true",help="set average in spike neuron")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_memory", type=str, default="40GiB",help="The maximum memory of each GPU")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--train_size", type=int, default=512, help="Number of calibration data samples.")
    parser.add_argument("--val_size", type=int, default=64, help="Number of validation data samples.")
    parser.add_argument("--training_seqlen", type=int, default=1024, help="lenth of the training sequence.")
    parser.add_argument("--calib_dataset", type=str, default="pile", choices=["wikitext2", "ptb", "c4", "mix", "redpajama", "pile", "hellaswag"], help="Where to extract calibration data from.")
    parser.add_argument("--use_fp32", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size.")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--off_load_to_disk", action="store_true", default=False, help="save training dataset to disk, saving CPU memory but may reduce training speed")
    parser.add_argument("--loss_type", type=str, default="mse", help="")
    parser.add_argument("--training_target",type=str,default="fp_input", choices=["fp_input", "quant_input", "both"], help="what is the source of the input to obatin the training target")
    parser.add_argument("--mse_init", action="store_true", help="init step size through MSE instead of MIN-MAX")
    parser.add_argument("--neuron_lr", type=float, default=5e-6, help="lr of scale in calibaration")
    parser.add_argument("--quant_lr", type=float, default=5e-6, help="lr of scale in calibaration")
    parser.add_argument("--weight_lr", type=float, default=5e-6, help="lr of scale in calibaration")
    parser.add_argument("--min_lr_factor", type=float, default=10, help="min_lr = lr/min_lr_factor")
    parser.add_argument("--wd", type=float, default=0,help="weight decay")
    parser.add_argument("--early_stop", type=int, default=0,help="early stoping after validation loss do not decrease")
    parser.add_argument("--save_spike_dir", default=None, type=str, help="direction for saving spike model")
    parser.add_argument("--clip_grad", default=0.3, type=float)
    # parser.set_defaults(avg_neuron=True)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = create_logger(output_dir)

    quant_config = load_json_as_namespace(os.path.join(args.quant_model_path, 'prefixequant_config.json'))
    # if quant_config['set_prefixed_tokens']:
    if quant_config.set_prefixed_tokens:
        prefixed_key_values = torch.load(os.path.join(args.quant_model_path, 'prefixed_key_values.pth'))
        spike_prefixed_key_values = torch.load(os.path.join(args.quant_model_path, 'prefixed_key_values.pth'))
    else:
        prefixed_key_values = None
        spike_prefixed_key_values = None

    logger.info(args)
    # init quantized model
    config = AutoConfig.from_pretrained(args.quant_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.quant_model_path, use_fast=False, legacy=False, trust_remote_code=True)
    with init_empty_weights():
        # orig_model = AutoModelForCausalLM.from_pretrained(args.quant_model_path, config=config, device_map='cpu', torch_dtype=torch.float16, trust_remote_code=True)
        spike_model = AutoModelForCausalLM.from_pretrained(args.quant_model_path, config=config, device_map='cpu', torch_dtype=torch.float16, trust_remote_code=True)
    for param in spike_model.parameters():
        param.requires_grad = False
    
    wrap_to_quant_model(spike_model)
    # register on-line hadadamrd transformation
    if quant_config.down_online_had:
        register_online_had(spike_model)
    # wrap rope for online_had and rope output capture
    rope_function_name = model_utils.get_rope_function_name(spike_model)
    layers = model_utils.get_layers(spike_model)
    for layer in layers:
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(layer.self_attn, rope_function_name, config=spike_model.config, online_had=quant_config.qk_online_had)

            
    # init weight quantizer
    if quant_config.wbits < 16:
        logger.info('init weight quantizer')
        init_weight_quantizer(quant_config, spike_model, logger=logger, minmax_init=False)

    # init input quantizer
    if quant_config.input_bits < 16:
        logger.info('init input quantizer')
        init_input_quantizer(quant_config, spike_model, logger=logger, minmax_init=False)

    if quant_config.output_bits < 16:
        logger.info('init output quantizer')
        init_out_quantizer(quant_config, spike_model, logger=logger, minmax_init=False)

    print("Loading pre-computed quantized weights...")
    orig_model = copy.deepcopy(spike_model)
    print(spike_model)
    wrap_to_snn_model(spike_model, args.T, args.avg_neuron)
    print(spike_model)
    
    block_class_name = spike_model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(spike_model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])

    spike_prefixed_key_values = replicate_past_key_values(spike_prefixed_key_values, args.T)
    if args.epochs > 0: # epoch大于0时,进行校准
        cache_trainloader = f'{args.cache_dir}/dataloader_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_train.cache'
        cache_valloader = f'{args.cache_dir}/dataloader_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_val.cache'
        if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
            trainloader = torch.load(cache_trainloader)
            logger.info(f"load trainloader from {cache_trainloader}")
            valloader = torch.load(cache_valloader)
            logger.info(f"load valloader from {cache_valloader}")
        else:
            trainloader, valloader = get_loaders(args.calib_dataset, tokenizer, args.train_size, args.val_size, seed=args.seed, seqlen=args.training_seqlen)
        load_checkpoint_in_model(orig_model, checkpoint=args.quant_model_path, device_map=device_map, dtype=torch.float16, offload_folder=f'/cache/{username}/offload')
        load_checkpoint_in_model(spike_model, checkpoint=args.quant_model_path, device_map=device_map, dtype=torch.float16, offload_folder=f'/cache/{username}/offload')
        dispatch_model(orig_model, device_map=device_map)
        dispatch_model(spike_model, device_map=device_map)
        spike_model = calibration_copy(orig_model, spike_model, prefixed_key_values, spike_prefixed_key_values, args, trainloader, valloader, logger)
        
    else:
        load_checkpoint_in_model(spike_model, checkpoint=args.quant_model_path, device_map=device_map, dtype=torch.float16, offload_folder=f'/cache/{username}/offload')
        dispatch_model(spike_model, device_map=device_map)

    del orig_model
    
    for param in spike_model.parameters():
        param.requires_grad = False
    spike_model.half()    # to make sure same evaluation results with main
    # if args.save_spike_dir:
    #     logger.info("start saving model")
    #     spike_model.save_pretrained(args.save_spike_dir)
    #     tokenizer.save_pretrained(args.save_spike_dir)
    #     torch.save(prefixed_key_values, os.path.join(args.save_spike_dir, 'spike_prefixed_key_values.pth'))
    #     spike_config = get_spike_config(args, quant_config)
    #     # spike_config['prefixed_tokens'] = quant_config['prefixed_tokens']
    #     train_utils.save_dict_as_json(spike_config, os.path.join(args.save_spike_dir, 'spike_config.json'))
    #     logger.info(f"save model to {args.save_spike_dir} success")
    with torch.no_grad():
        evaluate(spike_model, tokenizer, spike_prefixed_key_values, args, logger)


if __name__ == "__main__":
    print(sys.argv)
    main()
