"""切り抜きの確認用の一覧を作る。赤 = 手作業の輪郭、緑 = 自動（isnet）。"""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from segment import AOIS, IMAGES, OUT, manual_mask

MODEL = "isnet-general-use"
rows = list(csv.DictReader(open(OUT / "report.csv", encoding="utf-8")))
aois = json.load(open(AOIS, encoding="utf-8"))


def outline(m, width=5):
    return m & ~ndimage.binary_erosion(m, iterations=width)


def tile(name, size=300):
    im = np.asarray(Image.open(IMAGES / name).convert("RGB")).copy()
    auto = np.asarray(Image.open(OUT / MODEL / (Path(name).stem + ".png"))) > 0
    man = manual_mask(aois.get(name, []), im.shape[1::-1])
    if man is not None:
        im[outline(man)] = (255, 0, 0)
    im[outline(auto)] = (0, 255, 0)
    return Image.fromarray(im).resize((size, size))


def sheet(names, path, cols=6):
    out = Image.new("RGB", (300 * cols, 300 * -(-len(names) // cols)), "white")
    for i, n in enumerate(names):
        out.paste(tile(n), (300 * (i % cols), 300 * (i // cols)))
    out.save(path)


man = [r for r in rows if r["has_manual"] == "True"]
worst = sorted(man, key=lambda r: float(r[f"{MODEL}_iou"]))[:12]
print("lowest IoU:", [(r["image"], round(float(r[f"{MODEL}_iou"]), 3)) for r in worst[:6]])
sheet([r["image"] for r in worst], OUT / "preview_lowest_iou.png")

# 写真の端に接している（はみ出している可能性がある）もの
b_auto = sorted(rows, key=lambda r: -float(r[f"{MODEL}_border"]))
touch_auto = [r for r in b_auto if float(r[f"{MODEL}_border"]) > 0.005]
touch_man = [r for r in man if float(r["manual_border"]) > 0.005]
print(f"touching the image border: auto {len(touch_auto)}/{len(rows)}, manual {len(touch_man)}/{len(man)}")
print("largest:", [(r["image"], round(float(r[f"{MODEL}_border"]), 3)) for r in touch_auto[:8]])
if touch_auto:
    sheet([r["image"] for r in touch_auto[:18]], OUT / "preview_touching_border.png")
area = np.array([float(r[f"{MODEL}_area"]) for r in rows])
print(f"object area fraction: min {area.min():.2f}, median {np.median(area):.2f}, max {area.max():.2f}")
