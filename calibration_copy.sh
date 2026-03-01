# export HF_ENDPOINT=https://hf-mirror.com
gpu_no=$1
T=$2
neuron_lr=$3
train_size=${4:-64}
epochs=${5:-10}
batch_size=${6:-4}
de=${de:-""}

model_name="Llama-2-7B-hf"
quant_mode="w6a6"

if [[ "$de" == "de" ]]; then
    cmd="CUDA_VISIBLE_DEVICES=$gpu_no python -m debugpy --listen 8765 --wait-for-client"
else
    cmd="CUDA_VISIBLE_DEVICES=$gpu_no python"
fi
cmd="$cmd calibration.py \
--quant_model_path ./pre_symquantized_models/$model_name-$quant_mode \
--output_dir ./log/calibration_T$T/$model_name-$quant_mode-neuron_lr-$neuron_lr-train_size-$train_size-epochs-$epochs-batch_size-$batch_size \
--T $T \
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
--max_memory 20GiB \
--train_size $train_size \
--training_seqlen 1024 \
--calib_dataset pile \
--loss_type mse \
--epochs $epochs \
--neuron_lr $neuron_lr \
--quant_lr 0 \
--weight_lr 0 \
--batch_size $batch_size \
--wd 0 \
--save_spike_dir ./pre_symquantized_spike_models_cal/calibration_T$T/$model_name-$quant_mode-neuron_lr-$neuron_lr-train_size-$train_size-epochs-$epochs-batch_size-$batch_size \
--eval_ppl"
echo $cmd
eval $cmd