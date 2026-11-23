"""2D の点（渦巻き）で DDPM を学習し、前向き・逆向き過程を図にする。

    python diffusion/train_2d.py
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from ddpm import DDPM

OUT = Path(__file__).parent / "figures"


def spiral(n):
    """渦巻き状の 2D データ。平均 0・標準偏差およそ 1 に揃える。"""
    theta = torch.sqrt(torch.rand(n)) * 3 * math.pi
    x = torch.stack([theta * torch.cos(theta), theta * torch.sin(theta)], dim=1)
    x = x + 0.15 * torch.randn_like(x)
    return (x - x.mean(0)) / x.std()


class TimeEmbedding(nn.Module):
    """時刻 t を sin/cos のベクトルにする（NeRF の位置エンコーディングと同じ考え方）。"""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("freq", torch.exp(-math.log(10000) * torch.arange(dim // 2) / (dim // 2)))

    def forward(self, t):
        a = t.float()[:, None] * self.freq[None]
        return torch.cat([a.sin(), a.cos()], dim=1)


class EpsMLP(nn.Module):
    """eps_theta(x_t, t): ノイズの乗った点と時刻を受け取り、乗っているノイズを返す。"""

    def __init__(self, hidden=128, tdim=32):
        super().__init__()
        self.temb = TimeEmbedding(tdim)
        self.net = nn.Sequential(
            nn.Linear(2 + tdim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, self.temb(t)], dim=1))


def scatter_row(points, title, path):
    """時刻ごとの点の分布を横に並べて保存する。points は {t: (N, 2)}。"""
    ts = sorted(points)
    fig, axes = plt.subplots(1, len(ts), figsize=(3 * len(ts), 3.2))
    for ax, t in zip(axes, ts):
        p = points[t]
        ax.scatter(p[:, 0], p[:, 1], s=1, alpha=0.5)
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"t = {t}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(steps=6000, batch=2048, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(exist_ok=True)

    ddpm = DDPM(device=device)
    data = spiral(100_000).to(device)
    show = [0, 100, 250, 500, 1000]  # 図に出す時刻

    # 前向き過程: データが少しずつ正規分布に崩れていく様子
    x0 = data[:3000]
    fwd = {0: x0.cpu()}
    for t in show[1:]:
        tt = torch.full((len(x0),), t - 1, device=device)  # 配列の添字は 0 始まり
        fwd[t] = ddpm.q_sample(x0, tt, torch.randn_like(x0)).cpu()
    scatter_row(fwd, "forward process q(x_t | x_0)", OUT / "forward.png")

    # 学習
    model = EpsMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses = []
    for i in range(steps):
        x = data[torch.randint(0, len(data), (batch,), device=device)]
        loss = ddpm.loss(model, x)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
        if (i + 1) % 1000 == 0:
            print(f"step {i + 1:6d}  loss {sum(losses[-1000:]) / 1000:.4f}")

    # 逆向き過程: ノイズから渦巻きが現れる様子
    model.eval()
    # 添字 t の p_step を終えた直後の点は、ノイズが t ステップ分残った状態（図の時刻 t）にあたる
    _, rev = ddpm.sample(model, (3000, 2), device, keep=show)
    scatter_row(rev, "reverse process p_theta(x_{t-1} | x_t)", OUT / "reverse.png")

    # 時刻ごとの損失: どのノイズ量でノイズを当てにくいか
    with torch.no_grad():
        per_t = []
        for t in range(0, ddpm.T, 10):
            tt = torch.full((20000,), t, device=device)
            x = data[:20000]; eps = torch.randn_like(x)
            per_t.append(((eps - model(ddpm.q_sample(x, tt, eps), tt)) ** 2).mean().item())
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(losses, lw=0.5); ax[0].set_yscale("log"); ax[0].set_xlabel("step"); ax[0].set_title("training loss")
    ax[1].plot(range(0, ddpm.T, 10), per_t); ax[1].set_xlabel("t"); ax[1].set_title("loss per timestep")
    fig.tight_layout(); fig.savefig(OUT / "loss.png", dpi=120); plt.close(fig)
    print("saved figures to", OUT)


if __name__ == "__main__":
    main()
