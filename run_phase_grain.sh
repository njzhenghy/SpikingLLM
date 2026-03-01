gpu_no=$1
T=${2:-8}
bit=${3:-6}
group_size=${4:-256}
de=${de:-""}

# model_name="Meta-Llama-3-8B"
model_name="Llama-2-7B-hf"
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
--model_name Llama-2-7B \
--output_dir ./log/symquantized_qann/$model_name-$quant_mode-$group_size \
--wbits $bit \
--mse_init \
--pre_rotate \
--down_online_had \
--qk_online_had \
--set_prefixed_tokens \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande \
--neuron_path ./GrainAnalysis/arch_dir_grains=2/search_arch_9-10-1558.pth \
--T $T"
echo $cmd
eval $cmd
