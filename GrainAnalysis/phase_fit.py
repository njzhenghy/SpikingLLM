import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ------- 你已有的 FSNeuron（略作清理以便复用） -------
class FSNeuron(nn.Module):
    def __init__(self, T: int, num_grains: int = 2, beta: float = 1.0):
        super().__init__()
        self.T = T
        self.num_grains = num_grains
        self.d = nn.Parameter(torch.ones(num_grains))   # 可学习的base参数
        self.beta = beta
        self.grain_assignment = self._assign_grains()

    def _assign_grains(self):
        assignment = []
        steps_per_grain = self.T // self.num_grains
        remainder = self.T % self.num_grains
        for g in range(self.num_grains):
            steps = steps_per_grain + (1 if g < remainder else 0)
            assignment.extend([g] * steps)
        return assignment

    def get_grain_power(self, t: int):
        g_idx = self.grain_assignment[t]
        g_start = self.grain_assignment.index(g_idx)
        pos_in_grain = t - g_start
        return torch.pow(self.d[g_idx], -(pos_in_grain + 1))

    @staticmethod
    def heaviside_ste(x, beta=1.0):
        # 连续近似（平滑的硬阈值），前向近似Heaviside，反向给出可导近似
        return torch.sigmoid(beta * x)                   # 你也可以自定义更尖锐的 STE

    def forward(self, x, spikes: torch.Tensor = None, return_spikes: bool = False):
        y = torch.zeros_like(x)

        if spikes is None and x is not None:
            v = x.clone()
            spikes = []
            for t in range(self.T):
                p = self.get_grain_power(t)
                s_t = FSNeuron.heaviside_ste(v - p, self.beta)
                spikes.append(s_t)
                v = v - p * s_t
            spikes = torch.stack(spikes, dim=0).squeeze()

        for t in range(self.T):
            p = self.get_grain_power(t)
            y = y + p * spikes[t].unsqueeze(-1)
        return (y, spikes) if return_spikes else y


class CandidateFS(nn.Module):
    def __init__(self, T, num_grains, beta):
        super().__init__()
        self.block = FSNeuron(T=T, num_grains=num_grains, beta=beta)

    def forward(self, x):
        return self.block(x)

class FSCell(nn.Module):
    def __init__(self, candidates_cfg):
        """
        candidates_cfg: 列表，每个元素是一个dict，例如
            [{"T":8,"num_grains":2,"beta":1.0},
             {"T":8,"num_grains":3,"beta":1.0},
             {"T":8,"num_grains":2,"beta":2.0}]
        """
        super().__init__()
        self.cands = nn.ModuleList([CandidateFS(**cfg) for cfg in candidates_cfg])
        self.alpha = nn.Parameter(torch.zeros(len(self.cands)))  # 架构logits

    def forward(self, x):
        weights = F.softmax(self.alpha, dim=0)  # π_i
        outs = [cand(x) * w for cand, w in zip(self.cands, weights)]
        return torch.stack(outs, dim=0).sum(dim=0)  # 混合输出

class FSSuperNet(nn.Module):
    def __init__(self, num_layers, candidates_cfg_per_layer):
        super().__init__()
        assert len(candidates_cfg_per_layer) == num_layers
        self.layers = nn.ModuleList([
            FSCell(candidates_cfg_per_layer[i]) for i in range(num_layers)
        ])
        # 例如最后做一个线性头，可按任务替换
        self.head = nn.Identity()

    def forward(self, x):
        for cell in self.layers:
            x = cell(x)
        return self.head(x)

    # 导出genotype（每层选择权重最大的候选）
    def export_genotype(self):
        genotype = []
        for li, cell in enumerate(self.layers):
            with torch.no_grad():
                k = torch.argmax(cell.alpha).item()
            genotype.append({"layer": li, "choice_idx": k})
        return genotype

    # 参数分组：模型权重w vs 架构参数α
    def weight_parameters(self):
        for n, p in self.named_parameters():
            if "alpha" not in n:
                yield p

    def arch_parameters(self):
        for n, p in self.named_parameters():
            if "alpha" in n:
                yield p


def train_one_epoch_first_order(supernet, train_loader, val_loader, loss_fn,
                                w_opt, a_opt, device="cuda"):
    supernet.train()

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    for _ in range(len(train_loader)):
        # === 1) 用验证集更新 α（first-order：忽略 w*(α) 的高阶依赖） ===
        try:
            x_val, y_val = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            x_val, y_val = next(val_iter)

        x_val, y_val = x_val.to(device), y_val.to(device)
        a_opt.zero_grad()
        y_pred_val = supernet(x_val)
        val_loss = loss_fn(y_pred_val, y_val)
        val_loss.backward()
        a_opt.step()

        # === 2) 用训练集更新 w ===
        try:
            x_tr, y_tr = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_tr, y_tr = next(train_iter)

        x_tr, y_tr = x_tr.to(device), y_tr.to(device)
        w_opt.zero_grad()
        y_pred_tr = supernet(x_tr)
        tr_loss = loss_fn(y_pred_tr, y_tr)
        tr_loss.backward()
        w_opt.step()


def make_loader(n=10_000, batch_size=128, mean=0.0, std=1.0, scale=5.0, seed=0, shuffle=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.abs(torch.normal(mean=mean, std=std, size=(n, 1), generator=g)) * scale  # x ~ |N(0,1)|*5
    y = x.clone()  # y = x
    ds = TensorDataset(x.float(), y.float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle), ds

# 生成 train / test（同分布、不同种子）
train_loader = make_loader(seed=42)
val_loader = make_loader(seed=2025, shuffle=False)

# 假设输入是 [batch, feature] 的回归任务；把你的数据集/损失替换掉即可。
device = "cuda" if torch.cuda.is_available() else "cpu"

cands_layer0 = [
    {"T": 8, "num_grains": 2, "beta": 1.0},
    {"T": 8, "num_grains": 3, "beta": 1.0},
    {"T": 8, "num_grains": 2, "beta": 2.0},
]
cands_layer1 = [
    {"T": 8, "num_grains": 4, "beta": 1.0},
    {"T": 12,"num_grains": 3, "beta": 1.0},
    {"T": 8, "num_grains": 2, "beta": 0.5},
]

supernet = FSSuperNet(num_layers=2,
                      candidates_cfg_per_layer=[cands_layer0, cands_layer1]).to(device)

w_opt = torch.optim.Adam(supernet.weight_parameters(), lr=1e-3)
a_opt = torch.optim.Adam(supernet.arch_parameters(),  lr=3e-3)  # α学得快点通常更稳定
loss_fn = nn.MSELoss()

# train_loader / val_loader 自行提供
for epoch in range(20):
    train_one_epoch_first_order(supernet, train_loader, val_loader,
                                loss_fn, w_opt, a_opt, device)

# 导出离散架构（每层的最佳候选索引）
genotype = supernet.export_genotype()
print("Genotype:", genotype)
