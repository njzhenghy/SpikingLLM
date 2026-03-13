import os
import sys
import random
import numpy as np
import torch
import time
from utils.data_utils import get_loaders, test_ppl
from quantize.block_ap import block_ap
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from quantize.int_linear_real import load_quantized_model
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.hooks import remove_hook_from_module
from utils.quant_utils import get_quant_config, check_quantizer
from utils import train_utils
import utils.model_utils as model_utils
import utils.rotation_utils as rotation_utils
torch.backends.cudnn.benchmark = True
from phase.phase_util import wrap_to_phase_model, init_out_neuron, init_weight_quantizer, init_input_neuron, register_online_had, set_phase_model_time_step, get_act_stat
from utils.snn_utils import replicate_past_key_values
# from phase.phase_neuron import rate_list
from phase.phase_calibration import calibration

@torch.no_grad()
def evaluate(model, tokenizer,prefixed_key_values, args, logger):
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    model = dispatch_model(model, device_map=device_map, skip_keys='past_key_values') # set skip_keys to avoid a bug
    prefixed_key_values = model_utils.mv_kv_cache(prefixed_key_values, model)
    results_str=""
    if args.eval_ppl:
        # datasets = ["wikitext2", "c4"]
        datasets = ["wikitext2", "c4", "redpajama", "pile"]
        # datasets = ["wikitext2"]
        ppl_results = test_ppl(args, model, tokenizer, prefixed_key_values, datasets)
        for dataset in ppl_results:
            logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')
            results_str += f"{ppl_results[dataset]:.2f} "

    if args.eval_tasks != "":
        if prefixed_key_values is not None:
            model = model_utils.WrappedPrefixCausalLM(model, prefixed_key_values)
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = args.eval_tasks.split(',')
        model = HFLM(pretrained=model, batch_size=args.eval_batch_size)
        task_manager = lm_eval.tasks.TaskManager()
        results = lm_eval.simple_evaluate(
        model=model,
        tasks=task_list,
        num_fewshot=0,
        task_manager=task_manager,
        )
        logger.info(make_table(results))
        total_acc = 0
        total_acc_with_norm = 0
        for task in ['winogrande','hellaswag','arc_challenge','arc_easy','piqa']:
            if task in task_list:
                total_acc += results['results'][task]['acc,none']
                results_str += f"{results['results'][task]['acc,none']*100:.2f} "
                if 'acc_norm,none' in results['results'][task]:
                    results_str += f"{results['results'][task]['acc_norm,none']*100:.2f} "
                    total_acc_with_norm += results['results'][task]['acc_norm,none']
                else:
                    total_acc_with_norm += results['results'][task]['acc,none']
        logger.info(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')
        logger.info(f'Average Acc (with norm): {total_acc_with_norm/len(task_list)*100:.2f}%')
        logger.info(f'Results string: {results_str.strip()}')
        # remove wrapper
        if prefixed_key_values is not None:
            model = model.model


def main():
    import argparse

    parser = argparse.ArgumentParser()
    # -----------------model setting------------------------------------
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--model_name", type=str, default=None, help="model name, for the saving of corresponding data cache")
    parser.add_argument("--cache_dir", default="/home/public/shared_hf_cache/", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="./log/", type=str, help="direction of logging file")
    parser.add_argument("--save_quant_dir", default=None, type=str, help="direction for saving quantization model")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--resume_quant", type=str, default=None, help="model path of resumed quantized model")
    # -----------------quantization setting------------------------------------
    parser.add_argument("--wbits", type=int, default=8, help="quantization bits")
    parser.add_argument("--w_group_size", type=int, default=-1, help="quantization group size")
    parser.add_argument("--w_asym", dest="w_asym", action="store_true", help="Set w_asym to True")
    # parser.add_argument("--w_sym", dest="w_asym", action="store_false", help="Set w_asym to False")
    # parser.set_defaults(w_asym=False)
    parser.add_argument("--input_bits", type=int, default=16, help="quantization bits")
    parser.add_argument("--input_group_size", type=int, default=-1, help="quantization group size")
    parser.add_argument("--input_mode", type=str, default='dynamic', help="quantization type")
    parser.add_argument("--input_asym", dest="input_asym", action="store_true", help="Set input_asym to True")
    # parser.add_argument("--input_sym", dest="input_asym", action="store_false", help="Set input_asym to False")
    # parser.set_defaults(input_asym=True)
    parser.add_argument("--output_mode", type=str, default='dynamic', help="quantization type")
    parser.add_argument("--output_asym", dest="output_asym", action="store_true", help="Set output_asym to True")
    # parser.add_argument("--output_sym", dest="output_asym", action="store_false", help="Set output_asym to False")
    # parser.set_defaults(output_asym=True)
    parser.add_argument("--k_bits", type=int, default=16, help="")
    parser.add_argument("--output_bits", type=int, default=16, help="")
    parser.add_argument("--v_bits", type=int, default=16, help="")
    parser.add_argument("--kv_group_size", type=int, default=128, help="default as head-wise")
    parser.add_argument("--k_pre_rope", action="store_true")
    parser.add_argument("--kv_mode", type=str, default='dynamic', help="quantization type")
    parser.add_argument("--kv_asym", dest="kv_asym", action="store_true", help="Set kv_asym to True")
    # parser.add_argument("--kv_sym", dest="kv_asym", action="store_false", help="Set kv_asym to False")
    # parser.set_defaults(kv_asym=False)
    parser.add_argument("--mse_init", action="store_true", help="init step size through MSE instead of MIN-MAX")
    parser.add_argument("--asym_mse_init", action="store_true", help="init step size through MSE instead of MIN-MAX")
    parser.add_argument("--skip_qk_weight_init", action="store_true")
    parser.add_argument("--block_qk_weight_init", action="store_true")
    parser.add_argument("--mse_init_size", type=int, default=8, help="sample number used in mse_init; actually, even 4 or 2 is enough")
    parser.add_argument("--fp_mse_init", action="store_true", help="use full-precision block input during the mse init process")
    # ------------------rotation and prefix setting------------------------------------
    parser.add_argument("--pre_rotate", action="store_true")
    parser.add_argument("--rotate_mode", type=str, default='hadamard')
    parser.add_argument("--down_online_had", action="store_true")
    parser.add_argument("--qk_online_had", action="store_true")
    parser.add_argument("--set_prefixed_tokens", action="store_true")
    parser.add_argument("--outlier_threshold", type=int, default=64, help="\eta in Eq.(3), indicating the oitlier threshold ratio detect outlier tokens")
    parser.add_argument("--activation_clipping", action="store_true", help="layer-wise activation clipping for dynamic quantization")
    # -----------------training setting------------------------------------
    parser.add_argument("--quant_lr", type=float, default=4e-07, help="lr of quantization parameters (s and z)")
    parser.add_argument("--weight_lr", type=float, default=4e-07, help="lr of fp weights")
    parser.add_argument("--min_lr_factor", type=float, default=10, help="min_lr = lr/min_lr_factor")
    parser.add_argument("--clip_grad", type=float, default=0.3)
    parser.add_argument("--wd", type=float, default=0, help="weight decay")
    parser.add_argument("--off_load_to_disk", action="store_true", default=False, help="save training dataset to disk, saving CPU memory but may reduce training speed")
    parser.add_argument("--use_fp32", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--early_stop", type=int, default=0, help="early stoping after validation loss do not decrease")
    parser.add_argument("--constant_wlr", action="store_true")
    parser.add_argument("--train_size", type=int, default=64, help="Number of calibration data samples.")
    parser.add_argument("--val_size", type=int, default=8, help="Number of validation data samples.")
    parser.add_argument("--training_seqlen", type=int, default=512, help="lenth of the training sequence.")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--calib_dataset",type=str, default="pile",
        choices=["wikitext2", "ptb", "c4", "mix", "redpajama", "pile"],
        help="Where to extract calibration data from.")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size.")
    parser.add_argument("--loss_type", type=str, default="mse", help="")
    parser.add_argument("--training_target", type=str, default="fp_input",
        choices=["fp_input", "quant_input", "both"],
        help="what is the source of the input to obatin the training target")
    # -----------------evaluation setting------------------------------------
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true", help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_memory", type=str, default="7GiB", help="The maximum memory of each GPU")
    # ------------------ others ------------------------------------------
    parser.add_argument("--max_outlier", type=float, default=5, help="")
    parser.add_argument("--max_item_index", type=int, default=5, help="")
    parser.add_argument("--set_outlier_zero", action="store_true")
    parser.add_argument("--modified_index", type=int, default=0, help="")
    # ------------------ ablation ------------------------------------------
    parser.add_argument("--ablate_prefix_number", type=int, default=None, help="")

    parser.add_argument("--T", type=int, default=2, help="time step")
    parser.add_argument("--neuron_path", type=str, help="neuron path")
    parser.add_argument("--spike_one", action="store_true", default=False, help="spike one")

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # if args.input_asym:
    #     assert args.input_asym and args.output_asym and not args.w_asym and not args.kv_asym # 对应非对称量化的断言
    # else:
    #     assert not (args.input_asym or args.output_asym or args.w_asym or args.kv_asym) # 对应对称量化的断言
        
    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_quant_dir:
        Path(args.save_quant_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = train_utils.create_logger(output_dir)
    logger.info(args)
    if args.model_name is None:
        args.model_name = args.model_path.split('/')[-1]
        logger.info(f"model_name is None, setting as {args.model_name}")
    if args.resume_quant:
        # directly load quantized model for evaluation
        model, tokenizer = load_quantized_model(args.resume_quant, args.wbits, args.group_size)
    else:
        # load fp quantized model
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, legacy=False, trust_remote_code=True)
        dtype = torch.float16 if not args.use_fp32 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, device_map='cpu', torch_dtype=dtype, trust_remote_code=True)
        if args.pre_rotate:
            rotation_utils.fuse_layer_norms(model)
            rotation_utils.rotate_model(model, rotate_mode=args.rotate_mode, online=args.down_online_had)
            model.half()
        for param in model.parameters():
            param.requires_grad = False
        print(model)
        wrap_to_phase_model(model, T=1)
        print(model)

        # register on-line hadadamrd transformation
        if args.pre_rotate and args.down_online_had:
            register_online_had(model)
        # wrap rope for online_had and rope output capture
        rope_function_name = model_utils.get_rope_function_name(model)
        layers = model_utils.get_layers(model)
        for layer in layers:
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                        layer.self_attn, 
                        rope_function_name, 
                        config=model.config,
                        online_had=args.qk_online_had)

        prefixed_tokens = None                
        prefixed_key_values = None
        args.prefixed_length = 0
        activation_stat = None  
        # include_static = (args.input_mode == "static" and args.input_bits < 16 ) or (args.kv_mode == "static" and (args.k_bits < 16 or args.v_bits < 16))
        if args.set_prefixed_tokens:
            from utils.stat_utils import get_prefixed_tokens
            # model and data prepaer
            if model.device.type == 'cpu':
                original_device = 'cpu'
                block_class_name = model.model.layers[0].__class__.__name__
                device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
                model = dispatch_model(model, device_map=device_map)
            else:
                original_device = 'cuda'
            cal_dataloader, _ = get_loaders(
            args.calib_dataset,
            tokenizer,
            train_size=64,
            val_size=0,
            seed=args.seed,
            seqlen=512,
            )
            # get prefixed tokens
            if args.set_prefixed_tokens:
                tick = time.time()
                prefixed_tokens = get_prefixed_tokens(cal_dataloader, model, tokenizer, args.model_name, outlier_threshold=args.outlier_threshold, activation_type='down_proj')
                logger.info(f"get {len(prefixed_tokens)} prefixed tokens; token id:{prefixed_tokens}; text: {tokenizer.decode(prefixed_tokens)}")
                logger.info(f"time to get prefixed token:{time.time()-tick:.0}s")
                model.config.prefixed_tokens = prefixed_tokens
                args.prefixed_length = len(prefixed_tokens)
                use_cache = model.config.use_cache
                model.config.use_cache = True
                if args.ablate_prefix_number is not None:
                    prefixed_tokens = prefixed_tokens[:args.ablate_prefix_number]
                    logger.info(f'ablation:set prefix as {prefixed_tokens}')
                output = model(torch.tensor([prefixed_tokens], device=model.device), return_dict=True)
                prefixed_key_values = output.past_key_values
                model.config.use_cache = use_cache
                
            # ????????????????????????????

            # cal_dataloader, _ = get_loaders(
            # args.calib_dataset,
            # tokenizer,
            # train_size=5,
            # val_size=0,
            # seed=args.seed,
            # seqlen=512,
            # )
            activation_stat = get_act_stat(model, cal_dataloader, 'max', prefixed_tokens, args.down_online_had)
            # torch.save(activation_stat, f'/home/ubuntu/solar/PhaseSNN/GrainAnalysis/activation_dir/Llama-2-7B-hf-{args.wbits}bit/activation_stat.pth')
            # exit(0)
            if original_device == 'cpu':
                remove_hook_from_module(model, recurse=True)
                model = model.cpu()
        
        # During forward propagation, the model uses t=1 for acceleration.
        set_phase_model_time_step(model=model, T=1, logger=logger)

        if args.neuron_path:
            logger.info('load neuron_parameter')
            neuron_parameter = torch.load(args.neuron_path)
        else:
            neuron_parameter = None

        # When passing through spiking neurons, t=args.T is used to approximate the spiking dynamics.
        logger.info('init input neuron')
        init_input_neuron(args, model, activation_stat, logger, neuron_parameter)

        logger.info('init output neuron')
        init_out_neuron(args, model, activation_stat, logger, neuron_parameter)

        print(model)

        spike_prefixed_key_values = replicate_past_key_values(prefixed_key_values, T=1)
        
        train_utils.cleanup_memory()

        if args.epochs > 0:
            logger.info("=== start quantization Training ===")
            tick = time.time()  

            cache_trainloader = f'{args.cache_dir}/dataloader_{args.model_name}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_train.cache'
            cache_valloader = f'{args.cache_dir}/dataloader_{args.model_name}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_val.cache'
            if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
                trainloader = torch.load(cache_trainloader)
                logger.info(f"load trainloader from {cache_trainloader}")
                valloader = torch.load(cache_valloader)
                logger.info(f"load valloader from {cache_valloader}")
            else:
                trainloader, valloader = get_loaders(
                    args.calib_dataset,
                    tokenizer,
                    args.train_size,
                    args.val_size,
                    seed=args.seed,
                    seqlen=args.training_seqlen,
                )
                torch.save(trainloader, cache_trainloader)
                torch.save(valloader, cache_valloader)

            calibration(model, prefixed_key_values, spike_prefixed_key_values, args, trainloader, valloader, logger)

            logger.info(time.time() - tick)

    model.half()
    torch.cuda.empty_cache()
    
    evaluate(model, tokenizer, spike_prefixed_key_values, args, logger)



if __name__ == "__main__":
    print(sys.argv)
    main()