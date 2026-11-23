"""TripoSR（画像1枚から 3D を直接出すモデル）を刺激画像に当て、別視点の画像を出す。

    python image_to_3d/run_triposr.py
"""
import sys
import time
import types

import torch

from common import AZ, COEFS, CONDS, GEN, ROOT, load_object, stim_path, tag

# メッシュ化と背景除去の部品は使わないので、空のモジュールで代用する（Windows でビルドが難しいため）
for name in ("torchmcubes", "rembg"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["torchmcubes"].marching_cubes = None
sys.path.insert(0, str(ROOT / "data/third_party/TripoSR"))
from tsr.system import TSR  # noqa: E402

device = "cuda"

t0 = time.time()
model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
model.renderer.set_chunk_size(8192)
model.to(device)
print(f"load: {time.time() - t0:.0f}s")

# 作った 3D を 90 度刻みの 4 視点から描く（AZ と同じ並び）。このモデルに乱数はない
for cond in CONDS:
    out = GEN / tag(cond)
    out.mkdir(parents=True, exist_ok=True)
    for coef in COEFS:
        image = load_object(stim_path(cond, coef), 512, bg=0.5)
        with torch.no_grad():
            codes = model([image], device=device)
        views = model.render(codes, n_views=len(AZ), height=256, width=256, return_type="pil")[0]
        for az, v in zip(AZ, views):
            v.save(out / f"triposr_coef{coef}_az{az:03d}.png")
    print(tag(cond), flush=True)
