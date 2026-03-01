gpu_no=$1
group_size=${2:-256}
bit=${3:-6}
de=${de:-""}

# model_name="Meta-Llama-3-8B"
model_name="Llama-2-7B-hf"
quant_mode="w${bit}a${bit}"

cmd="CUDA_VISIBLE_DEVICES=$gpu_no python"
if [[ "$de" == "de" ]]; then
    cmd="$cmd -m debugpy --listen 8765 --wait-for-client"
fi
hostname=$(hostname)
if [[ "$hostname" == "server30" ]]; then
    model_path=/cache/model/$model_name/
else
    model_path=meta-llama/$model_name
fi
cmd="$cmd main.py \
--model_path $model_path \
--model_name Llama-2-7B \
--output_dir ./log/symquantized_qann/$model_name-$quant_mode-$group_size \
--wbits $bit \
--input_group_size $group_size \
--input_bits $bit \
--input_mode static \
--output_bits $bit \
--output_mode static \
--kv_group_size 128 \
--kv_mode static \
--mse_init \
--pre_rotate \
--down_online_had \
--qk_online_had \
--set_prefixed_tokens \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande \
--save_quant_dir ./pre_symquantized_models/$model_name-$quant_mode-$group_size"
echo $cmd
eval $cmd
