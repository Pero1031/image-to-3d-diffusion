"""数字を指定して生成する（classifier-free guidance）。生成は DDIM でステップ数を減らす。

    python diffusion/train_mnist_cfg.py --epochs 15
    python diffusion/train_mnist_cfg.py --sample-only     # 学習済みの重みで図だけ作り直す
"""
import argparse
import copy
import time

import torch

from ddpm import DDPM, guided
from train_mnist import OUT, ROOT, load_mnist, save_grid
from unet import UNet

CKPT = ROOT / "logs" / "mnist_cfg_ema.pt"


def train(args, ddpm, model, device):
    data, labels = load_mnist(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    step, t0, losses = 0, time.time(), []
    for epoch in range(args.epochs):
        perm = torch.randperm(len(data), device=device)
        for i in range(0, len(data) - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            y = labels[idx].clone()
            # 10% の確率でラベルを「条件なし」に置き換える。1 つのネットワークが両方の予測を学ぶ
            y[torch.rand(len(y), device=device) < 0.1] = model.n_classes
            loss = ddpm.loss(lambda x, t: model(x, t, y), data[idx])
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
    torch.save(ema.state_dict(), CKPT)
    return ema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--sample-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(exist_ok=True); (ROOT / "logs").mkdir(exist_ok=True)
    ddpm = DDPM(device=device)
    model = UNet(n_classes=10).to(device)
    if args.sample_only:
        model.load_state_dict(torch.load(CKPT, map_location=device))
    else:
        model = train(args, ddpm, model, device)
    model.eval()

    y = torch.arange(10, device=device)                 # 列 = 数字 0〜9
    noise = torch.randn(10, 1, 32, 32, device=device)   # 全条件で同じノイズから始める

    # 図 1: guidance の強さを変える（DDIM 50 ステップ）
    rows = [ddpm.ddim_sample(guided(model, y, w), noise.shape, device, 50, noise.clone()) for w in (0, 1, 3, 7)]
    save_grid(torch.cat(rows).cpu(), OUT / "mnist_cfg_scales.png", nrow=10)

    # 図 2: ステップ数を変える（guidance 3）。最下段は DDPM の 1000 ステップ
    eps = guided(model, y, 3.0)
    rows = []
    for n in (5, 10, 20, 50):
        t0 = time.time()
        rows.append(ddpm.ddim_sample(eps, noise.shape, device, n, noise.clone()))
        torch.cuda.synchronize(); print(f"DDIM {n:4d} steps: {time.time() - t0:.1f}s")
    t0 = time.time()
    torch.manual_seed(0)
    x = noise.clone()
    for t in reversed(range(ddpm.T)):
        x = ddpm.p_step(eps, x, t)
    torch.cuda.synchronize(); print(f"DDPM 1000 steps: {time.time() - t0:.1f}s")
    save_grid(torch.cat(rows + [x]).cpu(), OUT / "mnist_ddim_steps.png", nrow=10)


if __name__ == "__main__":
    main()
