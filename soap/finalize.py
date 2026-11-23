"""最終的な物体領域を決める。基本は isnet、影を物体に含めてしまった写真だけ個別に差し替える。

    python soap/finalize.py

出力: data/soap_masks/final/<画像名>.png と、全 122 枚の確認用一覧 data/soap_masks/final_all.png
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from segment import AOIS, IMAGES, OUT, manual_mask

DEFAULT = "isnet-general-use"
# 目で確認して isnet が影を含めていた写真。u2net が正しければ u2net、どちらも誤りなら手作業の輪郭を使う
OVERRIDE = {"soap25_IMG_5428": "u2net", "soap53_IMG_8380": "u2net",
            "soap47_IMG_7363": "manual", "soap57_IMG_9125": "manual"}

aois = json.load(open(AOIS, encoding="utf-8"))
names = sorted(p.name for p in IMAGES.glob("*.jpg"))
(OUT / "final").mkdir(exist_ok=True)
sheet = Image.new("RGB", (150 * 14, 150 * -(-len(names) // 14)), "white")
for i, name in enumerate(names):
    stem = Path(name).stem
    src = OVERRIDE.get(stem, DEFAULT)
    image = np.asarray(Image.open(IMAGES / name).convert("RGB")).copy()
    if src == "manual":
        m = manual_mask(aois.get(name, []), image.shape[1::-1])
        if m is None:
            print(f"NO MANUAL CONTOUR for {name}: keeping {DEFAULT} (shadow included)")
            src = DEFAULT
    if src != "manual":
        m = np.asarray(Image.open(OUT / src / (stem + ".png"))) > 0
    Image.fromarray(m.astype(np.uint8) * 255).save(OUT / "final" / (stem + ".png"))
    image[m & ~ndimage.binary_erosion(m, iterations=6)] = (0, 255, 0)
    sheet.paste(Image.fromarray(image).resize((150, 150)), (150 * (i % 14), 150 * (i // 14)))
sheet.save(OUT / "final_all.png")
print(len(names), "masks written")
