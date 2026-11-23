"""全条件を集計する。モデルの出力は、半透明の正解と不透明の対照のどちらに近いか。

    python image_to_3d/analyze_all.py

比べる量は、見えていない側（90・180・270 度）の物体内の平均輝度とコントラストを 3 視点で平均したもの。
方位角の正の向きがモデルと Blender で逆でも、90 度と 270 度が入れ替わるだけなので、この平均は変わらない。
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import AZ, COEFS, CONDS, GEN, GT, ROOT, SEEDS, UNSEEN, load_object, object_mask, stats, tag

OUT = ROOT / "image_to_3d" / "figures"
OUT.mkdir(exist_ok=True)
SOURCES = ("translucent", "opaque", "zero123", "triposr")
LABEL = {"translucent": "Blender, translucent (truth)", "opaque": "Blender, opaque (control)",
         "zero123": "Zero-1-to-3", "triposr": "TripoSR"}


def image(source, cond, coef, az, seed=0):
    """(画像, 物体の領域)。Blender の画像はアルファから、生成画像は白背景から領域を取る。"""
    t = tag(cond)
    if source == "translucent":
        return load_object(GT / t / f"gt_coef{coef}_az{az:03d}.png", 256, bg=1.0, with_mask=True)
    if source == "opaque":
        return load_object(GT / t / f"gt_opaque_coef{coef}_az{az:03d}.png", 256, bg=1.0, with_mask=True)
    name = f"zero123_coef{coef}_az{az:03d}_seed{seed}.png" if source == "zero123" else f"triposr_coef{coef}_az{az:03d}.png"
    im = Image.open(GEN / t / name).convert("RGB")
    return im, object_mask(im)


def unseen(source, cond, coef, seed=0):
    """見えていない 3 視点の (平均輝度, コントラスト) の平均。"""
    return np.mean([stats(*image(source, cond, coef, az, seed)) for az in UNSEEN], axis=0)


# cells[source] は (条件 x 濁り, 2)。Zero-1-to-3 は乱数 3 回の平均。seed_sd は乱数間の標準偏差
keys = [(cond, coef) for cond in CONDS for coef in COEFS]
cells, seed_sd = {}, []
for s in SOURCES:
    if s == "zero123":
        per_seed = np.array([[unseen(s, c, k, seed) for seed in SEEDS] for c, k in keys])   # (n, seeds, 2)
        cells[s], seed_sd = per_seed.mean(1), per_seed.std(1)
    else:
        cells[s] = np.array([unseen(s, c, k) for c, k in keys])

print(f"{len(keys)} cells = {len(CONDS)} conditions x {len(COEFS)} densities; unseen views {UNSEEN}")
print(f"{'':34s}" + "".join(f"{LABEL[s][:22]:>26s}" for s in SOURCES))
for i, (cond, coef) in enumerate(keys):
    print(f"{tag(cond):28s}{coef:>5s} " + "".join("        {:.3f} / {:.3f}   ".format(*cells[s][i]) for s in SOURCES))

summary = {"n_cells": len(keys), "zero123_seed_sd_mean": seed_sd.mean(0).tolist()}
print("\nmean absolute difference from each reference (luminance / contrast), and how many cells are closer to opaque")
for m in ("zero123", "triposr"):
    d_t, d_o = np.abs(cells[m] - cells["translucent"]), np.abs(cells[m] - cells["opaque"])
    closer = (d_o < d_t).sum(0)
    summary[m] = {"mad_to_translucent": d_t.mean(0).tolist(), "mad_to_opaque": d_o.mean(0).tolist(),
                  "cells_closer_to_opaque": closer.tolist()}
    print(f"{LABEL[m]:12s} to translucent {d_t.mean(0)[0]:.3f} / {d_t.mean(0)[1]:.3f}   "
          f"to opaque {d_o.mean(0)[0]:.3f} / {d_o.mean(0)[1]:.3f}   "
          f"closer to opaque: luminance {closer[0]}/{len(keys)}, contrast {closer[1]}/{len(keys)}")
print("Zero-1-to-3 SD across seeds (mean over cells): {:.3f} / {:.3f}".format(*seed_sd.mean(0)))
d = np.abs(cells["translucent"] - cells["opaque"]).mean(0)
print(f"translucent vs opaque (how far apart the two references are): {d[0]:.3f} / {d[1]:.3f}")
summary["mad_translucent_vs_opaque"] = d.tolist()
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=1)

# 正解との差の向き: 見えていない側で、モデルは正解より暗いか、コントラストが高いか
# 照明の向きで分ける。front = カメラ側から光、side = 横から、back = 逆光
GROUP = {("studio", 0): "front", ("rustig_koppie", 315): "front", ("rustig_koppie", 225): "side",
         ("studio", 180): "back", ("rustig_koppie", 135): "back"}
group = np.array([GROUP[c[1], c[2]] for c, _ in keys])
print()
print("direction of the error against the translucent truth, and distance to each reference by lighting")
for m in ("zero123", "triposr"):
    darker, higher = cells[m][:, 0] < cells["translucent"][:, 0], cells[m][:, 1] > cells["translucent"][:, 1]
    summary[m]["darker_than_truth"], summary[m]["higher_contrast_than_truth"] = int(darker.sum()), int(higher.sum())
    print(f"{LABEL[m]:12s} darker than truth {darker.sum()}/{len(keys)}, higher contrast than truth {higher.sum()}/{len(keys)}")
    for g in ("front", "side", "back"):
        i = group == g
        d_t = np.abs(cells[m][i] - cells["translucent"][i]).mean(0)
        d_o = np.abs(cells[m][i] - cells["opaque"][i]).mean(0)
        summary[m][g] = {"n": int(i.sum()), "mad_to_translucent": d_t.tolist(), "mad_to_opaque": d_o.tolist()}
        print(f"    {g:5s} (n={i.sum():2d})  to translucent {d_t[0]:.3f} / {d_t[1]:.3f}   to opaque {d_o[0]:.3f} / {d_o[1]:.3f}")
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=1)

# 図 1: 各セルで、モデルの値を 2 つの基準と比べる散布図。対角線に近いほどその基準と一致
fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
for r, (k, name) in enumerate(((0, "mean luminance"), (1, "contrast"))):
    for c, (m, ref) in enumerate((("zero123", "translucent"), ("zero123", "opaque"),
                                  ("triposr", "translucent"), ("triposr", "opaque"))):
        ax = axes[r, c]
        for coef, color in zip(COEFS, ("C0", "C2", "C3")):
            idx = [i for i, key in enumerate(keys) if key[1] == coef]
            err = seed_sd[idx, k] if m == "zero123" else None
            ax.errorbar(cells[ref][idx, k], cells[m][idx, k], yerr=err, fmt="o", ms=4, color=color,
                        label=f"density {coef}")
        lim = (0, max(cells[ref][:, k].max(), cells[m][:, k].max()) * 1.1)
        ax.plot(lim, lim, "k:", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel(LABEL[ref]); ax.set_ylabel(LABEL[m]); ax.set_title(name, fontsize=9)
axes[0, 0].legend(fontsize=7)
fig.suptitle("Unseen views (90, 180, 270 deg): model output vs. the two references")
fig.tight_layout(); fig.savefig(OUT / "summary_scatter.png", dpi=110); plt.close(fig)

# 図 2: 条件ごとの画像の一覧（濁りが少ない条件と多い条件）。行 = 出どころ、列 = 視点
for cond in CONDS:
    for coef in (COEFS[0], COEFS[-1]):
        fig, axes = plt.subplots(len(SOURCES), len(AZ), figsize=(2.2 * len(AZ), 2.3 * len(SOURCES)))
        for r, s in enumerate(SOURCES):
            for c, az in enumerate(AZ):
                im, mask = image(s, cond, coef, az)
                ax = axes[r, c]
                ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel("mean {:.2f}  contrast {:.2f}".format(*stats(im, mask)), fontsize=7)
                if r == 0:
                    ax.set_title(f"azimuth {az} deg", fontsize=9)
                if c == 0:
                    ax.set_ylabel(LABEL[s], fontsize=8)
        fig.suptitle(f"{tag(cond)}, density {coef}", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / f"views_{tag(cond)}_coef{coef}.png", dpi=90); plt.close(fig)
