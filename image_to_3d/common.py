"""入力画像の準備。半透明の刺激画像（RGBA、背景は透明）を、モデルが期待する形にする。"""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
STIM = Path("D:/kohinata/Blender/stimuli_final")


def load_object(path, size, bg, ratio=0.85):
    """物体が画像の ratio を占める正方形に切り出し、背景色 bg（0〜1）に合成して size にする。"""
    im = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
    ys, xs = np.where(im[..., 3] > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) / ratio)
    canvas = np.zeros((side, side, 4), np.float32)
    oy, ox = (side - (y1 - y0)) // 2, (side - (x1 - x0)) // 2
    canvas[oy:oy + y1 - y0, ox:ox + x1 - x0] = im[y0:y1, x0:x1]
    rgb = canvas[..., :3] * canvas[..., 3:] + bg * (1 - canvas[..., 3:])
    return Image.fromarray((rgb * 255).round().astype(np.uint8)).resize((size, size), Image.LANCZOS)


def hstack(images, path):
    w, h = images[0].size
    out = Image.new("RGB", (w * len(images), h))
    for i, im in enumerate(images):
        out.paste(im.convert("RGB"), (i * w, 0))
    out.save(path)
