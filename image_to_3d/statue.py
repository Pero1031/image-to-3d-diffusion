"""ろうの像と不透明の像（同じ形・同じ照明、材質だけ違う多視点データ）で、裏側の比較を行う。

    python image_to_3d/statue.py run        Zero-1-to-3 で別視点を生成
    python image_to_3d/statue.py analyze    正解と比べる

データは別のレンダラーで作った既存のもの（読み取りのみ）。カメラは像のまわりの輪に 9 度刻みで 40 台あり、
入力の視点から 10・20・30 台先が、ちょうど 90・180・270 度まわった正解の視点になる。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from common import GEN, ROOT, SEEDS, luminance, object_mask, stats

DATA = Path("D:/GoLD_rayt/out/studies/gs_gloss")
# 400 px の組を使う。256 px の組は像が小さく輪郭が粗いため、生成された別視点が破綻した（断片化、像の重複）
SETS = {"wax": "wax_thin_v2", "opaque": "opaque_sharp_v2"}
RING, N_RING = 0, 40                       # いちばん低い輪（仰角 15 度）、40 視点
INPUTS = tuple(range(0, N_RING, 5))        # 入力に使う視点（45 度おきに 8 つ）
STEPS = {0: 0, 90: 10, 180: 20, 270: 30}   # 回転角 -> 何台先か
UNSEEN = (90, 180, 270)
OUT = GEN / "statue_v2"


def load_view(material, index, size=256, bg=1.0, ratio=0.85):
    """像を正方形に切り出して背景色 bg に合成した画像と、物体の領域（縁を 3 画素削ったもの）。

    輪郭は少しぼかして合成する。2 値のままのギザギザな輪郭を入力すると、生成結果が破綻したため。"""
    name = f"view_{RING}_{index % N_RING:03d}"
    rgb = np.asarray(Image.open(DATA / SETS[material] / "train" / f"{name}.png").convert("RGB"), np.float32) / 255
    mask_im = Image.open(DATA / "masks" / "mask_train" / f"{name}.png").convert("L")
    a = np.asarray(mask_im) > 127
    soft = np.asarray(mask_im.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
    ys, xs = np.where(a)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) / ratio)
    oy, ox = (side - (y1 - y0)) // 2, (side - (x1 - x0)) // 2
    canvas = np.full((side, side, 3), bg, np.float32)
    m = np.zeros((side, side), np.uint8)
    al = soft[y0:y1, x0:x1, None]
    canvas[oy:oy + y1 - y0, ox:ox + x1 - x0] = rgb[y0:y1, x0:x1] * al + bg * (1 - al)
    m[oy:oy + y1 - y0, ox:ox + x1 - x0] = a[y0:y1, x0:x1] * 255
    image = Image.fromarray((canvas * 255).round().astype(np.uint8)).resize((size, size), Image.LANCZOS)
    mask = np.asarray(Image.fromarray(m).resize((size, size), Image.LANCZOS)) > 250
    return image, ndimage.binary_erosion(mask, iterations=3)


def run():
    import run_zero123 as z
    OUT.mkdir(parents=True, exist_ok=True)
    for material in SETS:
        for i in INPUTS:
            image, _ = load_view(material, i)
            for seed in SEEDS:
                if (OUT / f"{material}_in{i:02d}_az270_seed{seed}.png").exists():
                    continue
                t0 = time.time()
                for az, v in zip(STEPS, z.novel_views(image, list(STEPS), seed=seed)):
                    v.save(OUT / f"{material}_in{i:02d}_az{az:03d}_seed{seed}.png")
                print(f"{material} input {i:02d} seed {seed}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run()
    else:
        import statue_analyze  # noqa: F401
