from datasets import load_dataset

# for data in ('piqa', 'hellaswag'):
#     load_dataset(data)

# load_dataset('allenai/ai2_arc', 'ARC-Challenge')   # 对应arc_easy,arc_challenge
# load_dataset('allenai/winogrande', 'winogrande_debiased')
# load_dataset('mit-han-lab/pile-val-backup')
# load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1')

# 从load_dataset中加载数据集测试
load_dataset("Rowan/hellaswag", split="train", trust_remote_code=True)
load_dataset('Rowan/hellaswag', split="validation", trust_remote_code=True)
load_dataset("mit-han-lab/pile-val-backup", split="validation")
load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample",split='train')