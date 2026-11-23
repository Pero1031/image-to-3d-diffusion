"""MNIST で DDPM を学習し、生成画像と生成の途中経過を図にする。

    python diffusion/train_mnist.py --epochs 10
"""
import argparse
import copy
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import make_grid

from ddpm import DDPM
from unet import UNet

ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "figures"


def load_mnist(device):
    """全画像を GPU に載せる。28x28 を 32x32 に広げ、画素値を [-1, 1] にする。"""
    ds = datasets.MNIST(ROOT / "data", train=True, download=True)
    x = ds.data.float() / 127.5 - 1.0          # ノイズ（平均0・分散1）と釣り合う値域にする
    x = F.pad(x, (2, 2, 2, 2), value=-1.0)     # 2 回半分にできる大きさにする
    return x[:, None].to(device), ds.targets.to(device)


def save_grid(x, path, nrow):
    grid = make_grid(x.clamp(-1, 1) * 0.5 + 0.5, nrow=nrow, padding=1, pad_value=0.3)
    plt.imsave(path, grid.permute(1, 2, 0).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=0, help="試走用。0 なら無制限")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)

    data, _ = load_mnist(device)
    ddpm = DDPM(device=device)
    model = UNet().to(device)
    # EMA: 重みの移動平均。生成にはこちらを使う（学習中の重みの揺れが画像に出にくい）
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    step, t0, losses = 0, time.time(), []
    for epoch in range(args.epochs):
        perm = torch.randperm(len(data), device=device)
        for i in range(0, len(data) - args.batch + 1, args.batch):
            loss = ddpm.loss(model, data[perm[i:i + args.batch]])
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                decay = min(0.999, (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - decay)
            losses.append(loss.item()); step += 1
            if step == args.max_steps:
                break
        print(f"epoch {epoch + 1:3d}  step {step:6d}  loss {sum(losses[-400:]) / len(losses[-400:]):.4f}"
              f"  {time.time() - t0:.0f}s", flush=True)
        if step == args.max_steps:
            break
    torch.save(ema.state_dict(), ROOT / "logs" / "mnist_ema.pt")

    # 生成: 64 枚。途中経過は先頭 8 枚について保存する
    show = [1000, 750, 500, 250, 100, 50, 0]
    t0 = time.time()
    x, traj = ddpm.sample(ema.eval(), (64, 1, 32, 32), device, keep=show)
    print(f"sampling 64 images: {time.time() - t0:.0f}s")
    save_grid(x.cpu(), OUT / "mnist_samples.png", nrow=8)
    # 行 = 時刻（上がノイズ、下が完成）、列 = 別々のサンプル
    save_grid(torch.cat([traj[t][:8] for t in show]), OUT / "mnist_reverse.png", nrow=8)
    save_grid(data[:64].cpu(), OUT / "mnist_real.png", nrow=8)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(losses, lw=0.5); ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_title("training loss")
    fig.tight_layout(); fig.savefig(OUT / "mnist_loss.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
