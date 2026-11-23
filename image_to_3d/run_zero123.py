"""Zero-1-to-3（画像1枚と視点の変化から別視点の画像を出す拡散モデル）を刺激画像に当てる。

    python image_to_3d/run_zero123.py

仕組み（MNIST の DDPM との違いは条件の入れ方だけ）:
  - 入力画像の CLIP 埋め込みに視点の変化 4 数値を足して 1 本のベクトルにし、U-Net の cross-attention に渡す
  - 入力画像を VAE で潜在表現にし、ノイズの乗った潜在表現にチャンネル方向で連結する（U-Net の入力は 4+4 = 8 ch）
  - 条件を全部 0 にした予測と条件つきの予測を混ぜる（classifier-free guidance）
"""
import math
import time

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file
from transformers import CLIPVisionModelWithProjection

from common import ROOT, STIM, hstack, load_object

OUT = ROOT / "image_to_3d" / "figures"
OUT.mkdir(exist_ok=True)
device = "cuda"
repo = snapshot_download("bennyguo/zero123-xl-diffusers", local_files_only=True)

t0 = time.time()
kw = dict(variant="fp16", torch_dtype=torch.float32)  # 保存は半精度、計算は単精度（この GPU は半精度が速くない）
unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet", **kw).to(device).eval()
vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", **kw).to(device).eval()
clip = CLIPVisionModelWithProjection.from_pretrained(repo, subfolder="image_encoder", **kw).to(device).eval()
proj = torch.nn.Linear(768 + 4, 768)
sd = load_file(f"{repo}/clip_camera_projection/diffusion_pytorch_model.safetensors")
proj.load_state_dict({k.split(".")[-1]: v.float() for k, v in sd.items()})
proj.to(device)
sched = DDIMScheduler.from_pretrained(repo, subfolder="scheduler")
print(f"load: {time.time() - t0:.0f}s")

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)


@torch.no_grad()
def novel_views(image, azimuths, elevation=0.0, steps=50, scale=3.0, seed=0):
    """image: 256x256 の PIL（白背景）。azimuths: 入力視点からの水平回転（度）のリスト。"""
    n = len(azimuths)
    x = torch.from_numpy(np.asarray(image)).float().div(127.5).sub(1).permute(2, 0, 1)[None].to(device)

    # 条件 1: CLIP 埋め込み + 視点の変化（仰角の差、方位角の差の sin/cos、距離の差）
    c = torch.nn.functional.interpolate(x * 0.5 + 0.5, size=224, mode="bicubic", antialias=True)
    emb = clip((c - CLIP_MEAN) / CLIP_STD).image_embeds                       # (1, 768)
    cam = torch.tensor([[math.radians(-elevation), math.sin(math.radians(a)), math.cos(math.radians(a)), 0.0]
                        for a in azimuths], device=device)
    ctx = proj(torch.cat([emb.expand(n, -1), cam], dim=1))[:, None]           # (n, 1, 768)
    # 条件 2: 入力画像の潜在表現
    cat = vae.encode(x).latent_dist.mode().expand(n, -1, -1, -1)              # (n, 4, 32, 32)

    g = torch.Generator(device).manual_seed(seed)
    z = torch.randn(n, 4, 32, 32, device=device, generator=g)
    sched.set_timesteps(steps)
    for t in sched.timesteps:
        zin = torch.cat([torch.cat([z, cat], 1), torch.cat([z, torch.zeros_like(cat)], 1)])
        eps_c, eps_u = unet(zin, t, encoder_hidden_states=torch.cat([ctx, torch.zeros_like(ctx)])).sample.chunk(2)
        eps = eps_u + scale * (eps_c - eps_u)                                  # classifier-free guidance
        z = sched.step(eps, t, z).prev_sample
    out = vae.decode(z / vae.config.scaling_factor).sample.clamp(-1, 1)
    out = ((out.permute(0, 2, 3, 1) + 1) * 127.5).round().byte().cpu().numpy()
    return [Image.fromarray(o) for o in out]


azimuths = [0, 30, 90, 180, 270]  # 0 は入力と同じ視点（再現できるかの確認）
for coef in ("016", "128", "512"):
    image = load_object(STIM / f"stim01_studio/stim01_rot000_coef{coef}.png", 256, bg=1.0)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    views = novel_views(image, azimuths)
    print(f"coef {coef}: {len(azimuths)} views {time.time() - t0:.0f}s, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    hstack([image] + views, OUT / f"zero123_coef{coef}.png")
