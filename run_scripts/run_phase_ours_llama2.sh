export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/home/public/shared_hf_cache"
gpu_no=$1
T=${2:-10}

bit=${3:-8}
group_size=${4:-256}
de=${de:-""}

model_name="Llama-2-7B-hf"
# model_name="Meta-Llama-3-8B"
quant_mode="w${bit}"

cmd="CUDA_VISIBLE_DEVICES=$gpu_no python"
if [[ "$de" == "de" ]]; then
    cmd="$cmd -m debugpy --listen 5678 --wait-for-client"
fi
hostname=$(hostname)
if [[ "$hostname" == "server30" ]]; then
    model_path=/cache/model/$model_name/
else
    model_path=meta-llama/$model_name
fi
cmd="$cmd ../phase_main.py \
--model_path $model_path \
--model_name Llama-2-7B-hf \
--output_dir ./log/snn_ours_T_$T/$model_name \
--wbits $bit \
--pre_rotate \
--down_online_had \
--qk_online_had \
--set_prefixed_tokens \
--max_memory 20GiB \
--eval_ppl \
--eval_tasks winogrande \
--neuron_path ../GrainAnalysis/retrain_dir/Llama-2-7B-hf-T-10-grains-2-lr-0.0005/retrain.pth \
--T $T"
echo $cmd
eval $cmd