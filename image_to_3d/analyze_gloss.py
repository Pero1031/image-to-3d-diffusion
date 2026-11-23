"""生成された裏側は、光沢のある物体として描かれているか。

    python image_to_3d/analyze_gloss.py

光沢の手がかりは、別プロジェクト（gloss-perception-lab）のハイライト解析をそのまま使う:
ハイライト領域 H = 多重スケールの中心周辺応答の積（光沢マップ P）が閾値を超える部分。
閾値は、人の光沢評定つきデータセットの球の画像で決めた値を流用する（画像は同じ 256 px）。
    coverage  ハイライトが物体に占める割合
    contrast  ハイライトとその周囲の輝度のコントラスト
    sharp     ハイライトの縁の鋭さ
    P_mean    光沢マップの物体内の平均
比べるのは、正面の明るさを揃えられた 10 条件の、見えていない側（90・180・270 度）。
"""
import json
import sys

import numpy as np
import torch
from PIL import Image

from common import COEFS, CONDS, GEN, GT, SEEDS, UNSEEN, load_object, object_mask, tag

LAB = "D:/ML/gloss-perception-lab"
sys.path.insert(0, LAB)
from scripts.highlight_cues import cues_for_batch, erode  # noqa: E402
from src.models.features import GlossFeatures  # noqa: E402

KEYS = ("coverage", "contrast", "sharp_grad", "P_mean")
device = "cuda"
feat = GlossFeatures(32, sigma_step=1).to(device)
with open(f"{LAB}/runs/highlight_cues/cues_q95.json") as f:
    thr = json.load(f)["threshold"]


def cues(im, mask):
    x = torch.from_numpy(np.asarray(im.convert("RGB"))).float().div(255).permute(2, 0, 1)[None].to(device)
    obj = erode(torch.from_numpy(mask)[None].to(device), 10)[0]      # 輪郭から 10 px 内側だけを使う（元の解析と同じ）
    with torch.no_grad():
        c, _, _ = cues_for_batch(x, feat, obj, thr)
    return np.array([c[k].item() for k in KEYS])


def blender(pattern, cond, coef, az):
    return cues(*load_object(GT / tag(cond) / pattern.format(coef, az), 256, bg=1.0, with_mask=True))


def generated(name, cond, coef, az):
    out = []
    for seed in SEEDS:
        im = Image.open(GEN / tag(cond) / f"{name}_coef{coef}_az{az:03d}_seed{seed}.png").convert("RGB")
        out.append(cues(im, object_mask(im)))
    return np.nanmean(out, axis=0)


SOURCES = {"truth, translucent": lambda c, k, a: blender("gt_coef{}_az{:03d}.png", c, k, a),
           "truth, opaque": lambda c, k, a: blender("gt_opaque_coef{}_az{:03d}.png", c, k, a),
           "model, translucent input": lambda c, k, a: generated("zero123", c, k, a),
           "model, opaque input": lambda c, k, a: generated("zero123op", c, k, a)}

cells = []
for cond in CONDS:
    with open(GT / tag(cond) / "opaque_albedo.json") as f:
        info = json.load(f)
    cells += [(cond, k) for k in COEFS if info["matched"][k]]

# values[source] = (cells, 2 (front, unseen), cues)
values = {}
for name, fn in SOURCES.items():
    values[name] = np.array([[fn(c, k, 0), np.nanmean([fn(c, k, a) for a in UNSEEN], axis=0)] for c, k in cells])
    print(name, "done", flush=True)

print(f"\n{len(cells)} matched cells; mean over cells. NaN = no highlight region found")
for part, j in (("front view (0 deg)", 0), ("unseen views (90, 180, 270 deg)", 1)):
    print(f"\n{part}\n{'':28s}" + "".join(f"{k:>12s}" for k in KEYS))
    for name in SOURCES:
        print(f"{name:28s}" + "".join(f"{v:>12.4g}" for v in np.nanmean(values[name][:, j], axis=0)))

print("\npaired differences over cells, unseen views: mean (cells with a positive difference / cells where defined)")
pairs = (("model: translucent input - opaque input", "model, translucent input", "model, opaque input"),
         ("truth: translucent - opaque", "truth, translucent", "truth, opaque"),
         ("model(translucent in) - truth translucent", "model, translucent input", "truth, translucent"))
summary = {}
for label, a, b in pairs:
    d = values[a][:, 1] - values[b][:, 1]
    ok = ~np.isnan(d)
    summary[label] = {k: {"mean": float(np.nanmean(d[:, i])), "positive": int((d[:, i] > 0).sum()), "n": int(ok[:, i].sum())}
                      for i, k in enumerate(KEYS)}
    print(f"{label:44s}" + "".join(f"{np.nanmean(d[:, i]):>+11.3g} ({(d[:, i] > 0).sum()}/{ok[:, i].sum()})" for i in range(len(KEYS))))
with open(GT.parent.parent / "image_to_3d" / "figures" / "gloss_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
