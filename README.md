
# Official PyTorch implement for Spiking LLM

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
* `--T` corresponds to the number of time steps for spiking neuronsgroup.

Example with arguments:
```
bash run_scripts/run_phase_ours_llama2.sh 0,1
```
Note: `0,1` specifies the GPU indices (e.g., for A100 GPUs) to be used for the conversion process.