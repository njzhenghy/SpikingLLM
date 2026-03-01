#export HF_ENDPOINT=https://hf-mirror.com
gpu_no=$1
T=$2
group_size=$3
de=${de:-""}
if [[ "$de" == "de" ]]; then
    cmd="CUDA_VISIBLE_DEVICES=$gpu_no python -m debugpy --listen 5678 --wait-for-client"
else
    cmd="CUDA_VISIBLE_DEVICES=$gpu_no python"
fi
cmd="$cmd unevenness.py \
--quant_model_path pre_symquantized_models/input_group_size_$group_size/Llama-2-7b-hf-w6a6 \
--spike_model_path pre_symquantized_spike_models/input_group_size_$group_size/Llama-2-7b-w6a6-T$2 \
--output_dir ./log/calibration_T$T/Llama-2-7b-w6a6 \
--T $T \
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
--max_memory 40GiB \
--train_size 512 \
--val_size 64 \
--training_seqlen 512 \
--calib_dataset pile \
--loss_type mse \
--epochs 30 \
--neuron_lr 5e-5 \
--quant_lr 0 \
--weight_lr 0 \
--batch_size 4 \
--wd 0"
echo $cmd
eval $cmd


# --calib_dataset pile \
