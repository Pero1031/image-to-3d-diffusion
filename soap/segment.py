"""石けんの写真 122 枚の物体領域を自動で切り抜き、手作業の輪郭（61 枚）と比べる。

    python soap/segment.py

出力（すべて data/ の下、git の管理外）:
    data/soap_masks/<model>/<画像名>.png   自動の物体領域
    data/soap_masks/report.csv             画像ごとの一致度と、画像の縁に接している割合
    data/soap_masks/preview_*.png          確認用の一覧
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).parent.parent
os.environ.setdefault("U2NET_HOME", str(ROOT / "data" / "rembg"))   # モデルの保存先
from rembg import new_session, remove  # noqa: E402

IMAGES = Path("D:/project/vision-experiments/material_aoi_pilot/images")
AOIS = Path("D:/project/vision-experiments/Tobii_pro/aois.json")
OUT = ROOT / "data" / "soap_masks"
MODELS = ("u2net", "isnet-general-use")


def manual_mask(aois, size):
    """手作業の輪郭（多角形）を塗りつぶした領域。輪郭がなければ None。"""
    im = Image.new("L", size, 0)
    found = False
    for a in aois:
        if a.get("name") != "contour":
            continue
        for poly in ([a["points"]] if a["shape"] == "polygon" else a["polygons"]):
            ImageDraw.Draw(im).polygon([tuple(p) for p in poly], fill=255)
            found = True
    return np.asarray(im) > 0 if found else None


def auto_mask(image, session):
    """背景除去モデルの出力を 2 値にし、いちばん大きい連結領域だけを残して穴を埋める。"""
    m = np.asarray(remove(image, session=session, only_mask=True)) > 127
    lab, n = ndimage.label(m)
    if n > 1:
        m = lab == 1 + np.argmax(ndimage.sum(m, lab, range(1, n + 1)))
    return ndimage.binary_fill_holes(m)


def border_fraction(m):
    """画像の外周の画素のうち、物体領域に入っている割合。0 より大きければ物体が写真の端に接している。"""
    edge = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    return float(edge.mean())


def main():
    with open(AOIS, encoding="utf-8") as f:
        aois = json.load(f)
    names = sorted(p.name for p in IMAGES.glob("*.jpg"))
    sessions = {m: new_session(m) for m in MODELS}
    rows = []
    for i, name in enumerate(names):
        image = Image.open(IMAGES / name).convert("RGB")
        man = manual_mask(aois.get(name, []), image.size)
        row = {"image": name, "has_manual": man is not None,
               "manual_border": border_fraction(man) if man is not None else ""}
        for model, session in sessions.items():
            m = auto_mask(image, session)
            (OUT / model).mkdir(parents=True, exist_ok=True)
            Image.fromarray(m.astype(np.uint8) * 255).save(OUT / model / (Path(name).stem + ".png"))
            row[f"{model}_area"] = float(m.mean())
            row[f"{model}_border"] = border_fraction(m)
            row[f"{model}_iou"] = float((m & man).sum() / (m | man).sum()) if man is not None else ""
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(names)}", flush=True)
    with open(OUT / "report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    for model in MODELS:
        iou = np.array([r[f"{model}_iou"] for r in rows if r["has_manual"]])
        print(f"{model:18s} IoU vs manual (n={len(iou)}): mean {iou.mean():.3f}, median {np.median(iou):.3f}, "
              f"min {iou.min():.3f}, below 0.9: {(iou < 0.9).sum()}, below 0.8: {(iou < 0.8).sum()}")


if __name__ == "__main__":
    main()
