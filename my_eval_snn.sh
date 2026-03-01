gpu_no=$1
T=$2
neuron_lr=$3
train_size=${4:-64}
epochs=${5:-10}
batch_size=${6:-2}
de=${de:-""}

model_name="Llama-2-7B-hf"
quant_mode="w6a6"

if [[ "$de" == "de" ]]; then
    cmd="CUDA_VISIBLE_DEVICES=$gpu_no python -m debugpy --listen 8765 --wait-for-client"
else
    cmd="CUDA_VISIBLE_DEVICES=$gpu_no python"
fi
cmd="$cmd eval_snn.py \
--model_path meta-llama/$model_name  \
--spike_model_path ./pre_symquantized_spike_models_cal/calibration_T$T/$model_name-$quant_mode-neuron_lr-$neuron_lr-train_size-$train_size-epochs-$epochs-batch_size-$batch_size \
--output_dir ./log/eval_snn_T$T/$model_name-$quant_mode-neuron_lr-$neuron_lr-train_size-$train_size-epochs-$epochs-batch_size-$batch_size \
--eval_ppl \
--T $T \
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande"
echo $cmd
eval $cmd