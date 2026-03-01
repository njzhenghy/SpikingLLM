import torch
import os
import argparse

parser = argparse.ArgumentParser(description='Phase Load Args')
parser.add_argument('--model_name', type=str, default='Llama-2-7B-hf')
args = parser.parse_args()
model_name = "Llama-2-7B-hf"
# model_name = "Meta-Llama-3-8B"

def load_activation(base_keys, model_norm_key, path='activation_dir'):
    activation = torch.load(f'./activation_dir/{args.model_name}/activation_stat.pth',
                        weights_only=False, map_location="cuda:0")
    print("load success")
    ouput_dict = {}

    # load activation of base_operations
    for i in range(32): 
        for bk in base_keys:
            activation_value = activation[f"model.layers.{i}.{bk}"]
            flat = activation_value.view(-1)

            num_samples = min(100000, flat.numel())
            idx = torch.randint(0, flat.numel(), (num_samples,))
            sampled = flat[idx]

            ouput_dict[f"model.layers.{i}.{bk}"] = sampled
                
        torch.save(ouput_dict, f'./{path}/down_activation_stat_{i}.pth')
        print(f"save down_activation_stat_{i} success")

    # load activation of model_norm
    model_norm_value = activation[f"{model_norm_key}"]
    flat = model_norm_value.view(-1)

    num_samples = min(100000, flat.numel())
    idx = torch.randint(0, flat.numel(), (num_samples,))

    sampled = flat[idx]
    ouput_dict[f"{model_norm_key}"] = sampled

    torch.save(ouput_dict, f'./{path}/down_{model_norm_key}.pth')
    print(f"save {model_norm_key} success")

if __name__ == "__main__":
    base_keys = [
        "post_attention_layernorm.output",
        "input_layernorm.output",
        "self_attn.o_proj.input",
        "mlp.down_proj.input",
        "self_attn.q_Identity.input",
        "self_attn.k_Identity.input",
        "self_attn.v_Identity.input",
        "self_attn.softmax_Identity.input",
        "mlp.up_proj.output",
        "mlp.silu_Identity.input",
    ]
    model_norm_key = "model.norm.output"
    save_path = f'activation_dir/{model_name}/layers'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    load_activation(base_keys, model_norm_key, save_path)



        
