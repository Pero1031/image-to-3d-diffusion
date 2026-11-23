"""TripoSR（画像1枚から 3D を直接出すモデル）を刺激画像に当て、別視点の画像を出す。

    python image_to_3d/run_triposr.py
"""
import sys
import time
import types

import torch

from common import ROOT, STIM, hstack, load_object

# メッシュ化と背景除去の部品は使わないので、空のモジュールで代用する（Windows でビルドが難しいため）
for name in ("torchmcubes", "rembg"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["torchmcubes"].marching_cubes = None
sys.path.insert(0, str(ROOT / "data/third_party/TripoSR"))
from tsr.system import TSR  # noqa: E402

OUT = ROOT / "image_to_3d" / "figures"
OUT.mkdir(exist_ok=True)
device = "cuda"

t0 = time.time()
model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
model.renderer.set_chunk_size(8192)
model.to(device)
print(f"load: {time.time() - t0:.0f}s")

for coef in ("016", "128", "512"):  # 濁りが少ない・中間・多い
    image = load_object(STIM / f"stim01_studio/stim01_rot000_coef{coef}.png", 512, bg=0.5)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        codes = model([image], device=device)
    t1 = time.time()
    views = model.render(codes, n_views=6, height=256, width=256, return_type="pil")[0]
    print(f"coef {coef}: 3D {t1 - t0:.1f}s, render 6 views {time.time() - t1:.1f}s, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    hstack([image.resize((256, 256))] + views, OUT / f"triposr_coef{coef}.png")
