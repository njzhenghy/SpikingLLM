# export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/home/solar/model"

gpu_no=$1
T=$2
neuron_lr=$3
train_size=${4:-64}
epochs=${5:-10}
batch_size=${6:-4}
de=${de:-""}

model_name="Meta-Llama-3-8B"
quant_mode="w6a6"

if [[ "$de" == "de" ]]; then
    cmd="CUDA_VISIBLE_DEVICES=$gpu_no python -m debugpy --listen 8765 --wait-for-client"
else
    cmd="CUDA_VISIBLE_DEVICES=$gpu_no python"
fi
cmd="$cmd error.py \
--quant_model_path ./pre_symquantized_models/$model_name-$quant_mode \
--output_dir ./log/error \
--T $T \
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
--max_memory 40GiB \
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
--eval_ppl"
echo $cmd
eval $cmd