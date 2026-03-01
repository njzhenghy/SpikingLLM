import torch
import torch.nn as nn
import quantize.int_linear_fake as int_linear_fake
import quantize.int_linear_real as int_linear_real
from quantize.recon_loss import get_recon_loss
from utils.snn_utils import replicate_past_key_values
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import gc
from utils.quant_utils import (
    quant_parameters,weight_parameters,trainable_parameters,
    set_quant_state,quant_inplace,set_quant_parameters,
    set_weight_parameters,trainable_parameters_num,get_named_linears,set_op_by_name,
    mse_init)
import time
from utils.train_utils import NativeScalerWithGradNormCount
from utils.data_utils import BlockTrainDataset, SpikeBlockTrainDataset, copy_block_dataset, copy_block_dataset_to_spike
from contextlib import nullcontext
from utils.model_utils import get_kv_cache, mv_kv_cache
from SNN.spike_neuron import LMHTNeuron
from quantize.quantizer import UniformAffineQuantizer

@torch.no_grad()
def update_dataset(layer, source_dataset, target_dataset, dev, attention_mask, position_ids, prefixed_key_values):
    with torch.cuda.amp.autocast():
        for index, inps in enumerate(source_dataset):
            inps = inps.to(dev)
            if len(inps.shape)==2:
                inps = inps.unsqueeze(0)
            new_data = layer(inps, attention_mask=attention_mask, position_ids=position_ids, past_key_value=get_kv_cache(prefixed_key_values, bs=source_dataset.batch_size))[0].to('cpu')
            target_dataset.update_data(index,new_data)


@torch.no_grad()
def spike_update_dataset(layer, source_dataset, target_dataset, dev, attention_mask, position_ids, prefixed_key_values, T):
    # attention_mask = attention_mask.repeat(T, 1, 1, 1)
    with torch.cuda.amp.autocast():
        for index in range(len(source_dataset)):
            input = source_dataset[index].to(dev)
            if len(input.shape)==2:
                input = input.unsqueeze(0)
            past_key_value = get_kv_cache(prefixed_key_values, bs=input.shape[0]//T)
            new_data = layer(input, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value)[0].to('cpu')
            target_dataset.update_data(index,new_data)

def train_one_epoch(slayer, prefixed_key_values, attention_mask, position_ids, loss_func, dev, 
                    traincast, quant_inps, fp_inps_with_fp, fp_inps_with_quant, training_target, T):
    loss_list = []
    # attention_mask = attention_mask.repeat(T, 1, 1, 1)
    for index in range(len(quant_inps)):
        with traincast():
            input = quant_inps[index].to(dev)
            past_key_value = get_kv_cache(prefixed_key_values, bs=input.shape[0]//T)

            spike_out = slayer(input, attention_mask=attention_mask, position_ids=position_ids,
                                past_key_value=past_key_value)[0]

            TB, D, L = spike_out.shape
            B = TB // T
            spike_out = spike_out.view(T, B, D, L)
            spike_out = spike_out.sum(dim=0) 
            if training_target == 'fp_input':
                label = fp_inps_with_fp[index].to(dev)
                loss = loss_func(spike_out, label)

        if not math.isfinite(loss.item()):
            print("Loss is NAN, stopping training")
            with traincast():
                past_key_value1 = get_kv_cache(prefixed_key_values, bs=input.shape[0]//T)

                spike_out1 = slayer(input, attention_mask=attention_mask, position_ids=position_ids,
                                    past_key_value=past_key_value1)[0]

            raise ValueError
        loss_list.append(loss.detach().cpu())
    loss_mean = torch.stack(loss_list).mean()
    return loss_mean

@torch.no_grad()
def eval_one_epoch(slayer, prefixed_key_values, attention_mask, position_ids,
                      loss_func, dev, traincast,
                      quant_inps, fp_inps_with_fp, fp_inps_with_quant, training_target, T):
    loss_list = []
    # attention_mask = attention_mask.repeat(T, 1, 1, 1)
    for index in range(len(quant_inps)):
        with traincast():
            input = quant_inps[index].to(dev)
            past_key_value = get_kv_cache(prefixed_key_values, bs=input.shape[0]//T)
            spike_out = slayer(input, attention_mask=attention_mask,position_ids=position_ids,
                                past_key_value=past_key_value)[0]
            
            TB, D, L = spike_out.shape
            B = TB // T
            spike_out = spike_out.view(T, B, D, L)
            spike_out = spike_out.sum(dim=0)

            if training_target == 'fp_input':
                label = fp_inps_with_fp[index].to(dev)
                loss = loss_func(spike_out, label)
        loss_list.append(loss.detach().cpu())
    loss_mean = torch.stack(loss_list).mean()
    return loss_mean

def errorplot(
    model,
    spike_model,
    prefixed_key_values,
    spike_prefixed_key_values,
    args,
    trainloader,
    valloader,
    logger=None,
):
    logger.info("Starting ...")
    if args.off_load_to_disk:
        logger.info("offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefixed_key_values = mv_kv_cache(prefixed_key_values, dev=dev)
    spike_prefixed_key_values = mv_kv_cache(spike_prefixed_key_values, dev=dev)
    use_cache = model.config.use_cache
    model.config.use_cache = True

    # step 1: move embedding layer and first layer to target device, only suppress llama models now
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = torch.float16 if not args.use_fp32 else torch.float32
    traincast = torch.cuda.amp.autocast if not args.use_fp32 else nullcontext

    slayers = spike_model.model.layers
    spike_model.model.embed_tokens = spike_model.model.embed_tokens.to(dev)
    spike_model.model.norm = spike_model.model.norm.to(dev)
    slayers[0] = slayers[0].to(dev)
    
    # step 2: init dataset
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, model.config.hidden_size, args.batch_size, dtype, cache_path=args.cache_dir, off_load_to_disk=args.off_load_to_disk)
    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, model.config.hidden_size, args.batch_size, dtype, cache_path=args.cache_dir, off_load_to_disk=args.off_load_to_disk)
    
    # step 3: catch the input of the first layer 
    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None

        def forward(self, inp, **kwargs):
            self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs["attention_mask"]
            if self.position_ids is None:
                self.position_ids = kwargs["position_ids"]
            raise ValueError
    
    # step 3.1: catch the input of training set
    layers[0] = Catcher(layers[0], fp_train_inps)
    iters = len(trainloader) // args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i*args.batch_size, (i+1)*args.batch_size)], dim=0)
            try:
                model(data.to(dev), past_key_values=get_kv_cache(prefixed_key_values, bs=args.batch_size))
            except ValueError:
                pass
    position_ids = layers[0].position_ids
    attention_mask = layers[0].attention_mask
    attention_mask = attention_mask.to(dtype) if attention_mask is not None else None
    layers[0] = layers[0].module
    
    # step 3.2: catch the input of validation set
    layers[0] = Catcher(layers[0],fp_val_inps)
    iters = len(valloader) // args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([valloader[j][0] for j in range(i * args.batch_size, (i + 1) * args.batch_size)], dim=0)
            try:
                model(data.to(dev), past_key_values=get_kv_cache(prefixed_key_values, bs=args.batch_size))
            except ValueError:
                pass
    layers[0] = layers[0].module
    
    # step 4: move embedding layer and first layer to cpu
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    # step 5: copy fp input and repeat T times as the spike input, they are same at the first layer
    spike_train_inps = SpikeBlockTrainDataset(args.T, args.train_size, args.training_seqlen, model.config.hidden_size, args.batch_size, dtype, cache_path=args.cache_dir, off_load_to_disk=args.off_load_to_disk)
    # spike_train_inps.data = fp_train_inps.data.repeat(1, args.T, 1, 1)/args.T
    for index, data in enumerate(fp_train_inps):
        data = data / args.T
        data.unsqueeze(0)
        data = data.repeat(args.T, 1, 1, 1)
        # data_s = data.repeat(args.T, 1, 1)/args.T
        spike_train_inps.update_data(index, data.view(-1, args.training_seqlen, model.config.hidden_size))
    spike_val_inps = SpikeBlockTrainDataset(args.T, args.val_size, args.training_seqlen, model.config.hidden_size, args.batch_size, dtype, cache_path=args.cache_dir, off_load_to_disk=args.off_load_to_disk)
    # spike_val_inps.data = fp_val_inps.data.repeat(1, args.T, 1, 1)/args.T
    for index, data in enumerate(fp_val_inps):
        # data_s = data.repeat(args.T, 1, 1)/args.T
        # spike_val_inps.update_data(index, data_s)
        data = data / args.T
        data.unsqueeze(0)
        data = data.repeat(args.T, 1, 1, 1)
        spike_val_inps.update_data(index, data.view(-1, args.training_seqlen, model.config.hidden_size))
    
    # step 3.1: catch the input of training set
    slayers[0] = Catcher(slayers[0], spike_train_inps)
    iters = len(trainloader) // args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i*args.batch_size, (i+1)*args.batch_size)], dim=0)
            try:
                spike_model(data.to(dev), past_key_values=get_kv_cache(spike_prefixed_key_values, bs=args.batch_size))
            except ValueError:
                pass
    attention_mask_s = slayers[0].attention_mask
    attention_mask_s = attention_mask_s.to(dtype) if attention_mask_s is not None else None
    slayers[0] = slayers[0].module
    torch.cuda.empty_cache()
    
    # spike_train_inps = copy_block_dataset_to_spike(args.T, fp_train_inps)   # 代码可执行, 和上面逻辑等价
    # spike_val_inps = copy_block_dataset_to_spike(args.T, fp_val_inps)
    
    if args.training_target == 'fp_input':
        fp_train_inps_with_fp = fp_train_inps
        fp_val_inps_with_fp = fp_val_inps
        fp_train_inps_with_quant = None
        fp_val_inps_with_quant = None
    
    for param in spike_model.parameters():
        param.requires_grad = False
    try:
        
        # step 6: start training    
        loss_func = get_recon_loss(args.loss_type) 
        loss_list = []
        # writer = SummaryWriter(log_dir='logs')
        for block_index in range(len(layers)):
            logger.info(f"=== Start calibration blocks {block_index}===")
            layer = layers[block_index].to(dev)
            layer.to(dev) 

            # obtain output of full-precision model
            if args.epochs > 0 or args.mse_init:
                set_quant_state(layer, weight_quant=False, act_quant=False)
                if args.training_target == 'fp_input':
                    update_dataset(layer, fp_train_inps_with_fp, fp_train_inps_with_fp, dev, attention_mask, position_ids, prefixed_key_values)
                    update_dataset(layer, fp_val_inps_with_fp, fp_val_inps_with_fp, dev, attention_mask, position_ids, prefixed_key_values)
            
            # serarch for the optimal initialization for quantiztaion parameters
            if args.mse_init:
                logger.info("MSE init start")
                sub_train_input = spike_train_inps.get_subset(args.mse_init_size).to(dev, torch.float16) 
                one_attention_mask = None if attention_mask is None else attention_mask[0:1]
                mse_init(layer, prefixed_key_values, dev, sub_train_input, one_attention_mask, position_ids, logger, args)
                # mse_init(layer,prefixed_key_values, dev, sub_train_input, position_ids, logger, args, sub_train_gt)
                logger.info("MSE init end")

            # use spiking model
            slayer = slayers[block_index].to(dev)
            
            if args.epochs > 0:
                with torch.no_grad():
                    slayer.float()      # fp32 is also required for AMP training

                for epoch in range(1):
                    start_time = time.time()
                    train_loss = train_one_epoch(slayer, spike_prefixed_key_values, attention_mask_s, position_ids,
                                                                loss_func, dev, traincast,
                        spike_train_inps, fp_train_inps_with_fp, fp_train_inps_with_quant, args.training_target, args.T)
                    val_loss = eval_one_epoch(slayer, spike_prefixed_key_values, attention_mask_s, position_ids,
                        loss_func, dev, traincast,
                        spike_val_inps, fp_val_inps_with_fp, fp_val_inps_with_quant, args.training_target, args.T)
                    logger.info(f"blocks {block_index} epoch {epoch} train_loss:{train_loss} val_loss:{val_loss} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} time {time.time()-start_time}")
                loss_list.append(train_loss.item())

            # real smooth and quantization
            slayer.half()
            if args.epochs>0 or args.mse_init:
                # update inputs of spike model
                spike_update_dataset(slayer, spike_train_inps, spike_train_inps, dev, attention_mask_s, position_ids, spike_prefixed_key_values, args.T)
                spike_update_dataset(slayer, spike_val_inps, spike_val_inps, dev, attention_mask_s, position_ids, spike_prefixed_key_values, args.T)
            
            # move to cpu
            slayers[block_index] = slayer.to("cpu")

            torch.cuda.empty_cache()
            gc.collect()
    except ValueError:
        logger.info("-------------Loss is NAN-------------")
        pass

    # spike_model.model.norm.output_quantizer.avg = True
    logger.info(f"loss_list: {loss_list}")
    # delete cached dataset
    if args.off_load_to_disk:
        for dataset in [fp_train_inps, fp_val_inps, spike_train_inps, spike_val_inps]:
            if dataset is not None:
                dataset.clear_cache(())

    torch.cuda.empty_cache()
    gc.collect()                    

    return 
