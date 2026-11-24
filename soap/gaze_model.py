"""視線とモデルの特徴: 人が物体のどこを見たかと、モデルの特徴のどこに半透明感の情報があるかを比べる。

    python soap/gaze_model.py        （先に soap/patch_features.py）

視線: 参加者 5 人の注視（P003 は物体の中心ばかりを見る傾向がほかの 5 人よりはっきり強く、外れ値になるので
      除く）。画像の外の注視は捨てる。注視ごとに 1 と数え、σ = 30 画素（元の 1024 画素の画像で）でぼかす。
特徴: TripoSR の DINO のパッチ (32 x 32) と、Zero-1-to-3 の CLIP の最終層のパッチ (16 x 16)。被験者が見た
      写真そのもの（背景あり）から取る。評定は Part 3 と同じ（6 人の標準化した評定の平均）。

実行前に決めたこと:
問い (a)  見たところの特徴で、評定をよりよく予測できるか。物体の中のパッチを次の重みで平均し、Part 3 と同じ
          交差検証（石けんごとに評価側へ回すリッジ回帰）で評定を予測する。
            一様          物体の中を均等に
            視線          その画像の注視の密度
            典型          ほかの石けんの注視を、物体の外接矩形に合わせて写したもの。注視の数はその画像と同じ。
                          中心に寄る・上に寄るといった、どの画像にも共通の見方を表す（20 回引き直す）
            見ていない所  物体の中で、注視の密度が低いところほど大きな重み
          判断: 視線 > 典型 なら、人はその画像の中で半透明感の情報があるところを見ている。
                視線 > 見ていない所 も同じ向きの証拠。差がなければ、視線は読み出しに何も足さない。
問い (b)  モデルが半透明に見ているところを、人は見ているか。一様の読み出し（その石けんを除いて学習）を
          パッチごとに当てると、画像の予測をパッチに分けた地図になる（物体の中での平均 = 画像の予測）。
          物体の中で標準化し、注視点での平均（NSS）を求める。0 = 物体の中のでたらめな点と同じ。
          典型の見方でも NSS は 0 からずれうるので、典型の注視での期待値を引いた値を主な指標にする。
          明るいところを見ているだけでも正になりうるので、輝度の地図の NSS と、地図から輝度で説明できる
          部分を除いた残りの NSS も求める。
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import LeaveOneGroupOut

from ratings import ROOT, cv_predict, load, ridge

FIXATIONS = Path("D:/project/vision-experiments/Tobii_pro/data/master_fixations.csv")
IMAGES = Path("D:/project/vision-experiments/material_aoi_pilot/images")
MASKS = ROOT / "data" / "soap_masks" / "final"
OUT = ROOT / "soap" / "figures"
EXCLUDE = {"P003"}
SIZE, BASE, SIGMA = 1024, 128, 30.0       # 画像の大きさ、密度を作る格子、ぼかし（いずれも元の画像の画素）
GRID = {"dino": 32, "clip": 16}
N_DRAWS = 20
rng = np.random.default_rng(0)

data, names, groups, Z, y = load()
pf = np.load(ROOT / "data" / "soap_patch_features.npz")
assert list(pf["names"]) == names
ids = sorted(set(groups))
index = {g: np.where(groups == g)[0] for g in ids}


def down(a, g):
    """正方形の配列を g x g に面積平均で縮める。"""
    k = a.shape[0] // g
    return a.reshape(g, k, g, k).mean((1, 3))


def density(xy, g):
    """注視点 (n, 2)（元の画像の画素）-> ぼかした密度 (g, g)。"""
    h = np.histogram2d(xy[:, 1], xy[:, 0], bins=BASE, range=[[0, SIZE], [0, SIZE]])[0] if len(xy) else np.zeros((BASE, BASE))
    return down(ndimage.gaussian_filter(h, SIGMA / (SIZE / BASE)), g)


def luminance(rgb):
    c = rgb / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722])


# ---------------------------------------------------------------- 注視、物体の領域、輝度
rows = list(csv.DictReader(open(FIXATIONS, encoding="utf-8-sig")))
people = sorted({r["participant_id"] for r in rows} - EXCLUDE)
fix = {n: [] for n in names}
dropped = 0
for r in rows:
    if r["participant_id"] in EXCLUDE or not r["x_img_px"] or r["image_name"] not in fix:
        continue
    x, yy = float(r["x_img_px"]), float(r["y_img_px"])
    if 0 <= x < SIZE and 0 <= yy < SIZE:
        fix[r["image_name"]].append((x, yy))
    else:
        dropped += 1
fix = [np.array(fix[n]).reshape(-1, 2) for n in names]
masks = [np.asarray(Image.open(MASKS / (Path(n).stem + ".png"))) > 0 for n in names]
lum = [luminance(np.asarray(Image.open(IMAGES / n).convert("RGB"), np.float64)) for n in names]
boxes = []
for m in masks:
    ys, xs = np.where(m)
    boxes.append((xs.min(), xs.max() + 1, ys.min(), ys.max() + 1))
on_obj = np.mean(np.concatenate([m[f[:, 1].astype(int), f[:, 0].astype(int)] for m, f in zip(masks, fix)]))
print(f"participants used: {people} (excluded {sorted(EXCLUDE)}); fixations per image: median "
      f"{np.median([len(f) for f in fix]):.0f}, range {min(len(f) for f in fix)}-{max(len(f) for f in fix)}; "
      f"outside the image, dropped: {dropped}; on the object: {on_obj:.2f}")

# 典型の見方: 注視を物体の外接矩形で正規化した座標 (u, v)。ほかの石けんの注視から引く
uv = [np.column_stack([(f[:, 0] - b[0]) / (b[1] - b[0]), (f[:, 1] - b[2]) / (b[3] - b[2])]) for f, b in zip(fix, boxes)]


def to_image(u, b):
    xy = np.column_stack([b[0] + u[:, 0] * (b[1] - b[0]), b[2] + u[:, 1] * (b[3] - b[2])])
    return xy[(xy[:, 0] >= 0) & (xy[:, 0] < SIZE) & (xy[:, 1] >= 0) & (xy[:, 1] < SIZE)]


pool_other = [np.concatenate([uv[j] for j in range(len(names)) if groups[j] != groups[i]]) for i in range(len(names))]


# ---------------------------------------------------------------- (a) 重みを変えてパッチを平均する
def pooled(F, W):
    """F: (n, g, g, C)、W: (n, g, g) -> (n, C)。"""
    return np.einsum("nhwc,nhw->nc", F, W) / W.sum((1, 2))[:, None]


def r_of(X):
    pred = cv_predict(X.astype(np.float64), y, groups)
    return float(np.corrcoef(pred, y)[0, 1]), pred


def boot_diff(pa, pb_list, n=2000):
    """石けんを単位に再標本化した r の差（pa の r - pb_list の各 r の平均）の 95% 区間。"""
    out = []
    for _ in range(n):
        i = np.concatenate([index[g] for g in rng.choice(ids, len(ids))])
        ra = np.corrcoef(pa[i], y[i])[0, 1]
        out.append(ra - np.mean([np.corrcoef(pb[i], y[i])[0, 1] for pb in pb_list]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


summary = {"participants": people, "excluded": sorted(EXCLUDE), "fixations_on_object": float(on_obj)}
maps_for_fig = {}
for feat, g in GRID.items():
    F = pf[feat].astype(np.float32)
    M = np.array([down(m.astype(np.float64), g) for m in masks])
    G = np.array([density(f, g) for f in fix])
    W = {"uniform": M, "gaze": M * G,
         "unfixated": M * (1 - G / np.array([(G[i] * (M[i] > 0)).max() for i in range(len(names))])[:, None, None])}
    res, preds = {}, {}
    for k, w in W.items():
        res[k], preds[k] = r_of(pooled(F, w))
    draws = []
    for _ in range(N_DRAWS):
        Wd = np.array([M[i] * density(to_image(pool_other[i][rng.integers(0, len(pool_other[i]), len(fix[i]))], boxes[i]), g)
                       for i in range(len(names))])
        draws.append(r_of(pooled(F, Wd)))
    r_typ = np.array([d[0] for d in draws])
    a = {"r": res, "typical_r_median": float(np.median(r_typ)), "typical_r_range": [float(r_typ.min()), float(r_typ.max())],
         "gaze_above_typical_draws": int((res["gaze"] > r_typ).sum()),
         "gaze_minus_uniform_ci": boot_diff(preds["gaze"], [preds["uniform"]]),
         "gaze_minus_unfixated_ci": boot_diff(preds["gaze"], [preds["unfixated"]]),
         "gaze_minus_typical_ci": boot_diff(preds["gaze"], [d[1] for d in draws])}
    print(f"\n=== {feat} patches ({g} x {g}) ===")
    print(f"(a) r: uniform {res['uniform']:.3f}, gaze {res['gaze']:.3f}, unfixated {res['unfixated']:.3f}, "
          f"typical {a['typical_r_median']:.3f} [{a['typical_r_range'][0]:.3f}, {a['typical_r_range'][1]:.3f}] "
          f"(gaze above {a['gaze_above_typical_draws']}/{N_DRAWS} draws)")
    print(f"    gaze - uniform 95% CI {a['gaze_minus_uniform_ci']}, gaze - unfixated {a['gaze_minus_unfixated_ci']}, "
          f"gaze - typical {a['gaze_minus_typical_ci']}")

    # ------------------------------------------------------------ (b) 予測をパッチに分けた地図と注視
    S = np.zeros((len(names), g, g))
    Xu = pooled(F, M).astype(np.float64)
    for tr, te in LeaveOneGroupOut().split(Xu, y, groups):
        model = ridge().fit(Xu[tr], y[tr])
        sc, rg = model[0], model[1]
        coef = rg.coef_ / sc.scale_
        S[te] = F[te] @ coef + (rg.intercept_ - coef @ sc.mean_)
        assert np.allclose((S[te] * M[te]).sum((1, 2)) / M[te].sum((1, 2)), model.predict(Xu[te]), atol=1e-3)
    L = np.array([down(l, g) for l in lum])
    typical = np.array([density(to_image(pool_other[i], boxes[i]), g) for i in range(len(names))])
    nss = {k: np.zeros(len(names)) for k in ("readout", "luminance", "readout_minus_luminance")}
    exp = {k: np.zeros(len(names)) for k in nss}
    for i in range(len(names)):
        obj = M[i] >= 0.5
        cells = np.clip((fix[i] * g / SIZE).astype(int), 0, g - 1)
        hit = obj[cells[:, 1], cells[:, 0]]
        s, l = S[i][obj], L[i][obj]
        res_s = s - np.polyval(np.polyfit(l, s, 1), l)            # 輝度で説明できる部分を除いた残り
        for k, v in (("readout", s), ("luminance", l), ("readout_minus_luminance", res_s)):
            zmap = np.zeros((g, g)); zmap[obj] = (v - v.mean()) / v.std()
            nss[k][i] = zmap[cells[hit, 1], cells[hit, 0]].mean()
            exp[k][i] = (typical[i][obj] * zmap[obj]).sum() / typical[i][obj].sum()
    b = {}
    for k in nss:
        dlt = nss[k] - exp[k]
        bt = [np.mean(dlt[np.concatenate([index[q] for q in rng.choice(ids, len(ids))])]) for _ in range(2000)]
        b[k] = {"nss": float(nss[k].mean()), "expected_typical": float(exp[k].mean()), "difference": float(dlt.mean()),
                "ci": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                "images_above_typical": int((dlt > 0).sum()),
                "r_difference_with_rating": float(np.corrcoef(dlt, y)[0, 1])}
        print(f"(b) {k:24s} NSS {b[k]['nss']:+.3f}, typical {b[k]['expected_typical']:+.3f}, difference "
              f"{b[k]['difference']:+.3f} [{b[k]['ci'][0]:+.3f}, {b[k]['ci'][1]:+.3f}], above typical in "
              f"{b[k]['images_above_typical']}/{len(names)}; r of the difference with the rating {b[k]['r_difference_with_rating']:+.2f}")
    b["r_readout_luminance_within_object"] = float(np.mean(
        [np.corrcoef(S[i][M[i] >= 0.5], L[i][M[i] >= 0.5])[0, 1] for i in range(len(names))]))
    print(f"    mean within-object correlation of the readout map with luminance: {b['r_readout_luminance_within_object']:+.2f}")
    summary[feat] = {"grid": g, "pooling": a, "maps": b}
    maps_for_fig[feat] = (S, M, res, r_typ, a)

with open(OUT / "gaze_model_summary.json", "w") as f:
    json.dump(summary, f, indent=1)

# ---------------------------------------------------------------- 図 1: 重みのかけ方ごとの予測
fig, ax = plt.subplots(figsize=(7.5, 3.4))
labels = ["uniform", "gaze", "typical gaze layout\n(20 draws)", "unfixated"]
for j, (feat, col) in enumerate((("dino", "C2"), ("clip", "C0"))):
    S, M, res, r_typ, a = maps_for_fig[feat]
    off = -0.12 if j == 0 else 0.12
    ax.scatter([res["uniform"], res["gaze"]], [0 + off, 1 + off], color=col, zorder=3,
               label=f"{'TripoSR DINO' if feat == 'dino' else 'Zero-1-to-3 CLIP'} patches")
    ax.scatter(r_typ, np.full(len(r_typ), 2 + off), color=col, s=8, alpha=0.5)
    ax.scatter([res["unfixated"]], [3 + off], color=col, zorder=3)
ax.set_yticks(range(4), labels, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("cross-validated r with the mean rating (held-out soaps)")
ax.set_title("pooling the patches inside the object with different weights", fontsize=9)
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "gaze_pooling.png", dpi=120); plt.close(fig)

# ---------------------------------------------------------------- 図 2: 例（評定の高い写真と低い写真）
S, M, _, _, _ = maps_for_fig["dino"]
order = np.argsort(y)
pick = [order[-1], order[-2], order[1], order[0]]
lim = np.percentile(np.abs(np.concatenate([S[i][M[i] >= 0.5] for i in pick])), 98)    # 4 枚で共通の色の範囲
fig, axes = plt.subplots(2, len(pick), figsize=(3 * len(pick), 6.2))
for c, i in enumerate(pick):
    im = np.asarray(Image.open(IMAGES / names[i]).convert("RGB"))
    axes[0, c].imshow(im); axes[0, c].scatter(fix[i][:, 0], fix[i][:, 1], s=10, c="red", edgecolors="white", linewidths=0.4)
    axes[0, c].set_title(f"{names[i]}\nrating {y[i]:+.2f}", fontsize=8)
    smap = np.ma.masked_where(M[i] < 0.5, S[i])
    axes[1, c].imshow(im, alpha=0.35)
    axes[1, c].imshow(smap, extent=(0, SIZE, SIZE, 0), cmap="coolwarm", vmin=-lim, vmax=lim, alpha=0.85,
                      interpolation="nearest")
    axes[1, c].scatter(fix[i][:, 0], fix[i][:, 1], s=10, c="k", edgecolors="white", linewidths=0.4)
    for r_ in range(2):
        axes[r_, c].set_xticks([]); axes[r_, c].set_yticks([])
axes[0, 0].set_ylabel("photo and fixations (5 observers)", fontsize=8)
axes[1, 0].set_ylabel(f"DINO readout per patch\n(red > 0 > blue, one scale, +-{lim:.1f})", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "gaze_examples.png", dpi=90); plt.close(fig)

# ---------------------------------------------------------------- 図 3: 注視点での地図の値（典型の見方との差）
fig, ax = plt.subplots(figsize=(8, 3.2))
kinds =[("readout", "readout map"), ("luminance", "luminance"), ("readout_minus_luminance", "readout map,\nluminance removed")]
for j, (feat, col) in enumerate((("dino", "C2"), ("clip", "C0"))):
    for k, (key, _) in enumerate(kinds):
        v = summary[feat]["maps"][key]
        ax.errorbar(v["difference"], k + (-0.12 if j == 0 else 0.12), xerr=[[v["difference"] - v["ci"][0]], [v["ci"][1] - v["difference"]]],
                    fmt="o", color=col, capsize=3, label=("TripoSR DINO" if feat == "dino" else "Zero-1-to-3 CLIP") if k == 0 else None)
ax.axvline(0, color="gray", lw=0.8)
ax.set_yticks(range(len(kinds)), [k[1] for k in kinds], fontsize=8); ax.invert_yaxis()
ax.set_xlabel("map value at fixations minus that expected from the typical gaze layout\n(SD within the object)",
              fontsize=9)
ax.legend(fontsize=7); ax.set_title("do observers look where the map is high?", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "gaze_nss.png", dpi=120); plt.close(fig)
