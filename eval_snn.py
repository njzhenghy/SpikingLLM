import os
import sys
import random
import numpy as np
import torch
import utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map
from utils.quant_utils import wrap_to_quant_model, init_weight_quantizer, init_input_quantizer, init_out_quantizer, register_online_had, init_k_quantizer, init_v_quantizer
import utils.model_utils as model_utils
import utils.rotation_utils as rotation_utils
from main import evaluate
from utils.train_utils import load_json_as_namespace,create_logger
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from utils.snn_utils import wrap_to_snn_model, replicate_past_key_values

torch.backends.cudnn.benchmark = True



def main():
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--quant_model_path", type=str, help="model path of quantized model")
    parser.add_argument("--model_path", type=str, help="model path")
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
    parser.add_argument("--max_memory", type=str, default="25GiB",help="The maximum memory of each GPU")
    parser.add_argument("--spike_model_path", default=None, type=str, help="direction for saving spike model")

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

    # quant_config = load_json_as_namespace(os.path.join(args.quant_model_path, 'prefixequant_config.json'))
    spike_config = load_json_as_namespace(os.path.join(args.spike_model_path, 'spike_config.json'))
    # if quant_config['set_prefixed_tokens']:
    if spike_config.set_prefixed_tokens:
        prefixed_key_values = torch.load(os.path.join(args.spike_model_path, 'spike_prefixed_key_values.pth'))
    else:
        prefixed_key_values = None

    logger.info(args)
    # prefixed_key_values = replicate_past_key_values(prefixed_key_values, args.T)
    # init quantized model
    config = AutoConfig.from_pretrained(args.spike_model_path,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.spike_model_path, use_fast=False,legacy=False,trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, device_map='cpu', torch_dtype=torch.float16, trust_remote_code=True)
    wrap_to_quant_model(model)
    # register on-line hadadamrd transformation
    if spike_config.down_online_had:
        register_online_had(model)
    # wrap rope for online_had and rope output capture
    rope_function_name = model_utils.get_rope_function_name(model)
    layers = model_utils.get_layers(model)
    for layer in layers:
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn, 
                    rope_function_name, 
                    config=model.config,
                    online_had=spike_config.qk_online_had)   

    # init weight quantizer
    if spike_config.wbits < 16:
        logger.info('init weight quantizer')
        init_weight_quantizer(spike_config, model, logger=logger, minmax_init=False)

    # init input quantizer
    if spike_config.input_bits < 16:
        logger.info('init input quantizer')
        init_input_quantizer(spike_config, model, logger=logger, minmax_init=False)

    if spike_config.output_bits < 16:
        logger.info('init output quantizer')
        init_out_quantizer(spike_config, model, logger=logger, minmax_init=False)
    
    # # init kv quantizer
    # if quant_config.v_bits < 16:
    #     logger.info('init v quantizer')
    #     init_v_quantizer(quant_config, model, logger, minmax_init=False)

    # # if True:
    # if quant_config.k_bits < 16:
    #     # consistently init for wrap rope 
    #     logger.info('init k quantizer')
    #     init_k_quantizer(quant_config, model,  minmax_init=False)


    # logger.info(model.state_dict()["<具体路径>.scale"].shape) 
    # state = torch.load("your_quantized_checkpoint.pt", map_location="cpu")
    # print(state["<相应 key>"].shape)
    # model.tie_weights()
    # device_map = infer_auto_device_map(model)
    
    print("Loading pre-computed quantized weights...")
    
    # from safetensors.torch import load_file
    # ckpt_path = os.path.join(args.quant_model_path, "model-00001-of-00004.safetensors")  # 选一个分片文件
    # ckpt_weights = load_file(ckpt_path)
    # for key in ckpt_weights:
    #     if key in model.state_dict():
    #         if ckpt_weights[key].shape != model.state_dict()[key].shape:
    #             print(f"Mismatch: {key}: checkpoint {ckpt_weights[key].shape} vs model {model.state_dict()[key].shape}")
    #     else:
    #         print(f"Checkpoint key not found in model: {key}")
    # print(model)
    # wrap_to_snn_llama_model(model, args)
    print(model)
    wrap_to_snn_model(model, spike_config.T, spike_config.avg_neuron)
    print(model)
    prefixed_key_values = replicate_past_key_values(prefixed_key_values, args.T)
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    
    load_checkpoint_in_model(model,checkpoint=args.spike_model_path,device_map=device_map,dtype=torch.float16)
    for param in model.parameters():
        param.requires_grad = False
    model.half()    # to make sure same evaluation results with main
    with torch.no_grad():
        evaluate(model, tokenizer, prefixed_key_values,  args,logger)



if __name__ == "__main__":
    print(sys.argv)
    main()
