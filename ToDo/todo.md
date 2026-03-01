# To Do List
![zhz](https://img.shields.io/badge/genius-kakaa-orange)
1. [ ] 替换神经元
***
![zhy](https://img.shields.io/badge/iclr2025-zhy-blue)

2. [ ] 调用大模型输出，训练y=x
    - [ ] $\theta(x)$ 和 $H(x)$ 解耦
    - [ ] 架构搜索，更新架构权重 $\alpha$
    - [ ] 字典算法，代理梯度下降更新 $B$

3. 如何拟合Q, K和post_attention_layernorm?
    - 下采样更多数据
    - 分段拟合矩阵(矩阵分解, 分head拟合)
    - 初始化较大的base, 从大至小优化表示粒度
    
    base_keys = [
        "post_attention_layernorm.output",
        "input_layernorm.output",
        "self_attn.o_proj.input",
        "mlp.down_proj.input",
        "self_attn.q_Identity.input",
        "self_attn.k_Identity.input",
        "self_attn.v_Identity.input",
        "self_attn.softmax_Identity.input"
    ]
