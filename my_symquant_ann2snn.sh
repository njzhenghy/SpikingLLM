gpu_no=$1
T=$2
de=${de:-""}

model_name="Meta-Llama-3-8B"
quant_mode="w6a6"

cmd="CUDA_VISIBLE_DEVICES=$gpu_no python"
if [[ "$de" == "de" ]]; then
    cmd="$cmd -m debugpy --listen 8765 --wait-for-client"
fi
cmd="$cmd ann2snn.py \
--quant_model_path ./pre_symquantized_models/$model_name-$quant_mode \
--output_dir ./log/symquantized_snn_T$T/$model_name-$quant_mode \
--T $T \
--eval_ppl \
--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande"
echo $cmd
eval $cmd

#--save_spike_dir ./pre_symquantized_spike_models/$model_name-$quant_mode-T$T"