import os
import sys
import copy
import random
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map, dispatch_model
from utils.quant_utils import wrap_to_quant_model, init_weight_quantizer, init_input_quantizer, init_out_quantizer, register_online_had
from utils.data_utils import get_loaders
import utils.model_utils as model_utils
import utils.rotation_utils as rotation_utils
from utils.train_utils import load_json_as_namespace,create_logger
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from utils.snn_utils import wrap_to_snn_model, snn_wrap_to_quant_model
from errorplot import errorplot

torch.backends.cudnn.benchmark = True

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--snn_model_path", type=str, help="model path of calibration snn model")
    parser.add_argument("--output_dir", default="./log/test_snn", type=str, help="direction of logging file")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--T", type=int, default=2, help="time step")
    parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--avg_neuron", action="store_true",help="set average in spike neuron")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--max_memory", type=str, default="40GiB",help="The maximum memory of each GPU")
    parser.add_argument("--calib_dataset", type=str, default="pile", choices=["wikitext2", "ptb", "c4", "mix", "redpajama", "pile", "hellaswag"], help="Where to extract calibration data from.")
    parser.add_argument("--train_size", type=int, default=512, help="Number of calibration data samples.")
    parser.add_argument("--val_size", type=int, default=64, help="Number of validation data samples.")
    parser.add_argument("--training_seqlen", type=int, default=1024, help="lenth of the training sequence.")

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

    snn_config = load_json_as_namespace(os.path.join(args.snn_model_path, 'spike_config.json'))
    if snn_config.set_prefixed_tokens:
        # prefixed_key_values = torch.load(os.path.join(args.quant_model_path, 'prefixed_key_values.pth'))
        spike_prefixed_key_values = torch.load(os.path.join(args.snn_model_path, 'spike_prefixed_key_values.pth'))
    else:
        prefixed_key_values = None
        spike_prefixed_key_values = None

    logger.info(args)
    # prefixed_key_values = replicate_past_key_values(prefixed_key_values, args.T)
    config = AutoConfig.from_pretrained(args.snn_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.snn_model_path, use_fast=False, legacy=False, trust_remote_code=True)
    with init_empty_weights():
        spike_model = AutoModelForCausalLM.from_pretrained(args.snn_model_path, config=config, device_map='cpu', torch_dtype=torch.float16, trust_remote_code=True)
    wrap_to_quant_model(spike_model)
    
    if snn_config.down_online_had:
        register_online_had(spike_model)
    
    rope_function_name = model_utils.get_rope_function_name(spike_model)
    layers = model_utils.get_layers(spike_model)
    for layer in layers:
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn, 
                    rope_function_name, 
                    config=spike_model.config,
                    online_had=snn_config.qk_online_had)   

    # init weight quantizer
    if snn_config.wbits < 16:
        logger.info('init weight quantizer')
        init_weight_quantizer(snn_config, spike_model, logger=logger, minmax_init=False)

    # init input quantizer
    if snn_config.input_bits < 16:
        logger.info('init input quantizer')
        init_input_quantizer(snn_config, spike_model, logger=logger, minmax_init=False)

    if snn_config.output_bits < 16:
        logger.info('init output quantizer')
        init_out_quantizer(snn_config, spike_model, logger=logger, minmax_init=False)
    
    print("Loading pre-computed quantized weights...")

    print(spike_model)
    wrap_to_snn_model(spike_model, args.T, args.avg_neuron)
    print(spike_model)
    
    block_class_name = spike_model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(spike_model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    
    load_checkpoint_in_model(spike_model, checkpoint=args.snn_model_path, device_map=device_map, dtype=torch.float16)
    for param in spike_model.parameters():
        param.requires_grad = False
    spike_model.half()    # to make sure same evaluation results with main
    
    quant_model = copy.deepcopy(spike_model)
    snn_wrap_to_quant_model(quant_model)
    
    cache_trainloader = f'{args.cache_dir}/dataloader_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_train.cache'
    cache_valloader = f'{args.cache_dir}/dataloader_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_val.cache'
    if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
        trainloader = torch.load(cache_trainloader)
        logger.info(f"load trainloader from {cache_trainloader}")
        valloader = torch.load(cache_valloader)
        logger.info(f"load valloader from {cache_valloader}")
    else:
        trainloader, valloader = get_loaders(args.calib_dataset, tokenizer, args.train_size, args.val_size, seed=args.seed, seqlen=args.training_seqlen)
    dispatch_model(quant_model, device_map=device_map)
    dispatch_model(spike_model, device_map=device_map)
    # errorplot(quant_model, spike_model, prefixed_key_values, spike_prefixed_key_values, args, trainloader, valloader, logger)


if __name__ == "__main__":
    print(sys.argv)
    main()
