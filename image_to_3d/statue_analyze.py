"""ろうの像と不透明の像: モデルが作った裏側を、正解と比べる。statue.py analyze から呼ばれる。

セル = 入力の視点（8 つ）。各セルで、見えていない側（90・180・270 度）の平均輝度とコントラストを平均する。
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import ROOT, SEEDS, object_mask, stats
from statue import INPUTS, OUT, STEPS, UNSEEN, load_view

FIG = ROOT / "image_to_3d" / "figures"


def truth(material, i, az):
    return np.array(stats(*load_view(material, i + STEPS[az])))


def model(material, i, az):
    """(乱数 3 回の平均, 乱数間の標準偏差)"""
    v = []
    for seed in SEEDS:
        im = Image.open(OUT / f"{material}_in{i:02d}_az{az:03d}_seed{seed}.png").convert("RGB")
        v.append(stats(im, object_mask(im)))
    return np.mean(v, axis=0), np.std(v, axis=0, ddof=1)


def unseen(fn, material, i):
    return np.mean([fn(material, i, az) for az in UNSEEN], axis=0)


T = {m: np.array([unseen(truth, m, i) for i in INPUTS]) for m in ("wax", "opaque")}
M = {m: np.array([unseen(lambda a, b, c: model(a, b, c)[0], m, i) for i in INPUTS]) for m in ("wax", "opaque")}
seed_sd = np.mean([[model(m, i, az)[1] for az in UNSEEN] for m in ("wax", "opaque") for i in INPUTS], axis=(0, 1))
F = {m: np.array([truth(m, i, 0) for i in INPUTS]) for m in ("wax", "opaque")}     # 入力画像（正面）
M0 = {m: np.array([model(m, i, 0)[0] for i in INPUTS]) for m in ("wax", "opaque")}  # 回転 0 度の出力（再現の確認）

print(f"{len(INPUTS)} input views; values are mean luminance / contrast\n")
print(f"{'':26s}{'front (input)':>18s}{'model at 0 deg':>18s}{'unseen views':>18s}")
for name, fr, m0, un in (("truth, wax", F["wax"], None, T["wax"]), ("truth, opaque", F["opaque"], None, T["opaque"]),
                         ("model, wax input", F["wax"], M0["wax"], M["wax"]),
                         ("model, opaque input", F["opaque"], M0["opaque"], M["opaque"])):
    z = "" if m0 is None else "{:.3f} / {:.3f}".format(*m0.mean(0))
    print(f"{name:26s}{'%.3f / %.3f' % tuple(fr.mean(0)):>18s}{z:>18s}{'%.3f / %.3f' % tuple(un.mean(0)):>18s}")

d_true, d_model, d_front = T["wax"] - T["opaque"], M["wax"] - M["opaque"], F["wax"] - F["opaque"]
summary = {"n": len(INPUTS), "seed_sd": seed_sd.tolist()}
print(f"\ndifferences wax - opaque over the {len(INPUTS)} input views (seed SD of one cell: {seed_sd[0]:.3f} / {seed_sd[1]:.3f})")
for k, name in enumerate(("luminance", "contrast")):
    slope = float((d_true[:, k] * d_model[:, k]).sum() / (d_true[:, k] ** 2).sum())
    r_front, r_truth = np.corrcoef(d_model[:, k], d_front[:, k])[0, 1], np.corrcoef(d_model[:, k], d_true[:, k])[0, 1]
    closer = int((np.abs(M["wax"][:, k] - T["opaque"][:, k]) < np.abs(M["wax"][:, k] - T["wax"][:, k])).sum())
    summary[name] = {"true_diff": float(d_true[:, k].mean()), "model_diff": float(d_model[:, k].mean()),
                     "front_diff": float(d_front[:, k].mean()), "slope": slope, "r_model_front": float(r_front),
                     "r_model_truth": float(r_truth), "wax_output_closer_to_opaque_truth": closer}
    print(f"{name:10s} front {d_front[:, k].mean():+.3f}   truth unseen {d_true[:, k].mean():+.3f}   "
          f"model unseen {d_model[:, k].mean():+.3f}   slope {slope:+.2f}   r(model, front) {r_front:+.2f}   "
          f"r(model, truth) {r_truth:+.2f}   wax output closer to opaque truth in {closer}/{len(INPUTS)}")
with open(FIG / "statue_summary.json", "w") as f:
    json.dump(summary, f, indent=1)

# 例の図: 入力視点 0。行 = 出どころ、列 = 回転角
i = INPUTS[0]
rows = [("truth, wax", lambda az: load_view("wax", i + STEPS[az])[0]),
        ("truth, opaque", lambda az: load_view("opaque", i + STEPS[az])[0]),
        ("model, wax input", lambda az: Image.open(OUT / f"wax_in{i:02d}_az{az:03d}_seed0.png")),
        ("model, opaque input", lambda az: Image.open(OUT / f"opaque_in{i:02d}_az{az:03d}_seed0.png"))]
fig, axes = plt.subplots(len(rows), len(STEPS), figsize=(2.2 * len(STEPS), 2.3 * len(rows)))
for r, (name, get) in enumerate(rows):
    for c, az in enumerate(STEPS):
        axes[r, c].imshow(get(az)); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == 0:
            axes[r, c].set_title(f"azimuth {az} deg", fontsize=9)
        if c == 0:
            axes[r, c].set_ylabel(name, fontsize=8)
fig.suptitle("Statue: wax and opaque, same shape and light", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "statue_example.png", dpi=90); plt.close(fig)
