
# Official implement for **[Distribution-Aware Multi-Granularity Phase Coding: Towards Lower Conversion Error for Spike-Driven Large Language Models](https://openreview.net/pdf?id=meDMftHUlX)** [ICLR 2026]

![Overview](/main_figure.jpg)
## Installation
```
conda create -n prefixquant python==3.9.21

conda activate prefixquant


pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124


git clone git@github.com:Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install -e .

cd ..
pip install -r requirements.txt
```

pip install datasets==3.5.0 

## ANN-to-SNN Conversion
We provide two example commands to convert Llama-2-7B and Llama-3-8B to spiking neural networks:

```
bash run_scripts/run_phase_ours_llama2.sh
bash run_scripts/run_phase_ours_llama3.sh
```
Optional arguments:
* `--neuron_path` corresponds to the optimized phase coding base described in the paper (path located at `../GrainAnalysis/retrain_dir/`)
* `--T` corresponds to the number of time steps for spiking neurons.

Example with arguments:
```
bash run_scripts/run_phase_ours_llama2.sh 0,1
```
Note: `0,1` specifies the GPU indices (e.g., for A100 GPUs) to be used for the conversion process. 

The log file will be saved at: `run_scripts/log/*.txt`

## Parameter Configuration  
> Note: The results presented here reflect the latest optimized configuration. 
Compared to the values reported in the original paper, current results show 
improved performance due to refined hyperparameter tuning.

| Model | T | Grain | Wiki2 | Wino | ArcC | ArcE | PiQA |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| LLaMA-2-7B |  8 | 2 | 6.50 | 70.56 | 46.33 | 73.70 | 78.35 |  
| LLaMA-2-7B |  8 | 3 | 6.31 | 70.48 | 46.25 | 73.82 | 78.29 |
| LLaMA-2-7B | 10 | 2 | 5.50 | 70.48 | 46.50 | 73.91 | 78.29 |  
| LLaMA-2-7B | 10 | 3 | 5.50 | 70.48 | 46.33 | 73.86 | 78.35 |  

| Model | T | Grain | Wiki2 | Wino | ArcC | ArcE | PiQA |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| LLaMA-3-8B | 6 | 2 | 7.60 | 69.77 | 48.55 | 74.75 | 79.11 |
| LLaMA-3-8B | 6 | 3 | 7.79 | 70.80 | 48.81 | 74.96 | 79.05 |
| LLaMA-3-8B | 8 | 2 | 6.34 | 72.93 | 54.01 | 77.44 | 80.63 | 
| LLaMA-3-8B | 8 | 3 | 6.33 | 73.72 | 53.41 | 77.36 | 80.36 |

## Citation

If you find our work helpful, please consider citing our paper:

```bibtex
@inproceedings{zhengdistribution,
  title={Distribution-Aware Multi-Granularity Phase Coding: Towards Lower Conversion Error for Spike-Driven Large Language Models},
  author={Zheng, Hanyuan and Zhang, Haozhen and Chen, Tianshuo and Liu, Zhaogeng and Chang, Yi and Gu, Bin},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```