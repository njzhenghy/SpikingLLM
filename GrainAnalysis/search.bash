#!/bin/bash
gpu_no=$1
models=("Llama-2-7B-hf" "Meta-Llama-3-8B")

for model in "${models[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu_no python phase_search.py --model_name "$model" --T 8 --num_grains 3
    CUDA_VISIBLE_DEVICES=$gpu_no python phase_search.py --model_name "$model" --T 8 --num_grains 2
    CUDA_VISIBLE_DEVICES=$gpu_no python phase_search.py --model_name "$model" --T 6 --num_grains 2
done

for model in "${models[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu_no python retrain_decoupled.py --model_name "$model" --T 8 --num_grains 3
    CUDA_VISIBLE_DEVICES=$gpu_no python retrain_decoupled.py --model_name "$model" --T 8 --num_grains 2
    CUDA_VISIBLE_DEVICES=$gpu_no python retrain_decoupled.py --model_name "$model" --T 8 --num_grains 1 --genotype '[0,0,0,0,0,0,0,0]'
    CUDA_VISIBLE_DEVICES=$gpu_no python retrain_decoupled.py --model_name "$model" --T 6 --num_grains 3 --genotype '[0,0,1,1,2,2]'
    CUDA_VISIBLE_DEVICES=$gpu_no python retrain_decoupled.py --model_name "$model" --T 6 --num_grains 2
    CUDA_VISIBLE_DEVICES=$gpu_no python retrain_decoupled.py --model_name "$model" --T 6 --num_grains 1 --genotype '[0,0,0,0,0,0]'
    CUDA_VISIBLE_DEVICES=$gpu_no python retrain_decoupled.py --model_name "$model" --T 4 --num_grains 2 --genotype '[0,0,1,1]'
    CUDA_VISIBLE_DEVICES=$gpu_no python retrain_decoupled.py --model_name "$model" --T 4 --num_grains 1 --genotype '[0,0,0,0]'
done