gpu_no=$1
T=${2:-4}

bit=${4:-8}
group_size=${5:-256}

model_name="Meta-Llama-3-8B"
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
cmd="$cmd phase_main.py \
--model_path $model_path \
--model_name Llama-3-8B \
--output_dir ./log/snn_ours_T4/$model_name \
--wbits $bit \
--mse_init \
--pre_rotate \
--down_online_had \
--qk_online_had \
--set_prefixed_tokens \
--max_memory 20GiB \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,winogrande \
--neuron_path ./GrainAnalysis/retrain_dir/${model_name}-T-${T}-grains-8-lr-0.0008/retrain.pth \
--T $T"
echo $cmd
eval $cmd
