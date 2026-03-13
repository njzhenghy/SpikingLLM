
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
## Quantization
We provide two example command to quantized `Llama-2-7B` and`Llama-3-8B` without fine-tuning:

```
bash my_run_symquant_llama2.sh
```
add `--epochs 20` to calibration quantized model.


add `--input_group_size 256` to divide matrix into different quantized group.

## Conversion and Evaluation
We provide a example command to convert the quantization model to SNN model
```
bash my_symquant_ann2snn.sh
```

Use the following command to get and calibrate SNN model
```
bash calibration.sh
```

Details are as follows:
- `--T`: time step.
- `--wbits`, `--input_bits`: bits for quantization
