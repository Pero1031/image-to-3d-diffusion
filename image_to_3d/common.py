"""実験条件、入力画像の準備、物体内の明るさの統計。"""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
STIM = Path("D:/kohinata/Blender/stimuli_final")   # 既存の刺激画像（RGBA、背景は透明）
BLEND_DIR = Path("D:/kohinata/Blender")
BLENDER = "D:/Blender/blender.exe"
GT, GEN = ROOT / "data/gt_views", ROOT / "data/gen_views"

LIGHTS = {"studio": ("brown_photostudio_06_4k.hdr", 1.0),          # (照明ファイル, PNG にするときの露出倍率)
          "rustig_koppie": ("rustig_koppie_4k.hdr", 0.51)}
# (形, 照明, 照明の回転角)。入力画像はすべて既存の刺激画像
CONDS = [("stim01", "studio", 0), ("stim01", "studio", 180),
         ("stim01", "rustig_koppie", 315), ("stim01", "rustig_koppie", 225), ("stim01", "rustig_koppie", 135),
         ("stim02", "studio", 0)]
COEFS = ("016", "128", "512")     # 濁りが少ない・中間・多い
AZ = (0, 90, 180, 270)            # 入力視点からの水平回転（度）
UNSEEN = (90, 180, 270)
SEEDS = (0, 1, 2)


def tag(cond):
    return "{}_{}_rot{:03d}".format(*cond)


def stim_path(cond, coef):
    shape, light, rot = cond
    return STIM / f"{shape}_{light}" / f"{shape}_rot{rot:03d}_coef{coef}.png"


def load_object(path, size, bg, ratio=0.85, with_mask=False):
    """物体が画像の ratio を占める正方形に切り出し、背景色 bg（0〜1）に合成して size にする。

    with_mask=True なら、アルファから作った物体の領域（縁を 3 画素削ったもの）も返す。"""
    im = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
    ys, xs = np.where(im[..., 3] > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) / ratio)
    canvas = np.zeros((side, side, 4), np.float32)
    oy, ox = (side - (y1 - y0)) // 2, (side - (x1 - x0)) // 2
    canvas[oy:oy + y1 - y0, ox:ox + x1 - x0] = im[y0:y1, x0:x1]
    rgb = canvas[..., :3] * canvas[..., 3:] + bg * (1 - canvas[..., 3:])
    out = Image.fromarray((rgb * 255).round().astype(np.uint8)).resize((size, size), Image.LANCZOS)
    if not with_mask:
        return out
    from scipy import ndimage
    alpha = Image.fromarray((canvas[..., 3] * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    return out, ndimage.binary_erosion(np.asarray(alpha) > 250, iterations=3)


def luminance(rgb8):
    c = np.asarray(rgb8, np.float32)[..., :3] / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)   # sRGB -> 線形
    return lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def object_mask(im):
    """白背景の画像から物体の領域を取る。画像の縁につながった白い領域を背景とみなす。

    アルファのない生成画像用。物体の白く飛んだ部分が輪郭に接していると、そこは背景に入ってしまう。"""
    from scipy import ndimage
    white = (np.asarray(im)[..., :3] > 240).all(-1)
    lab, _ = ndimage.label(white)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return ndimage.binary_erosion(~np.isin(lab, border[border > 0]), iterations=3)


def stats(im, mask=None):
    """(平均輝度, コントラスト = 標準偏差 / 平均)。物体の内側だけで計算する。"""
    y = luminance(im)[object_mask(im) if mask is None else mask]
    return float(y.mean()), float(y.std() / y.mean())
