"""材質だけが違う入力の組で、モデルが作る裏側が変わるかを調べる。

    python image_to_3d/analyze_pairs.py

入力の組: 半透明の刺激画像と、同じ形・照明で正面の平均輝度を揃えた不透明の対照。
指標（実行前に決めたもの）: 見えていない側（90・180・270 度）の統計について
    本物の差   = 半透明の正解 - 不透明の正解
    モデルの差 = 半透明を入力したときの出力 - 不透明を入力したときの出力
を条件ごとに求め、モデルの差を本物の差に回帰した傾き（原点を通る）を見る。
    傾き 0 = 入力の材質を無視している、1 = 材質の違いを正しく反映している。
予想（実行前）: 傾きは 0 に近い。
正面の明るさを揃えられなかった条件（逆光など）は、組として成り立たないので主な集計から外す。
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import AZ, COEFS, CONDS, GEN, GT, ROOT, SEEDS, UNSEEN, load_object, object_mask, stats, tag

OUT = ROOT / "image_to_3d" / "figures"


def truth(kind, cond, coef):
    name = "gt_coef{}_az{:03d}.png" if kind == "translucent" else "gt_opaque_coef{}_az{:03d}.png"
    return np.mean([stats(*load_object(GT / tag(cond) / name.format(coef, az), 256, bg=1.0, with_mask=True))
                    for az in UNSEEN], axis=0)


def model(name, cond, coef):
    """(乱数ごとの値 (seeds, 2))。name は zero123（半透明を入力）か zero123op（不透明を入力）。"""
    out = []
    for seed in SEEDS:
        ims = [Image.open(GEN / tag(cond) / f"{name}_coef{coef}_az{az:03d}_seed{seed}.png").convert("RGB")
               for az in UNSEEN]
        out.append(np.mean([stats(im, object_mask(im)) for im in ims], axis=0))
    return np.array(out)


keys, matched = [], []
for cond in CONDS:
    with open(GT / tag(cond) / "opaque_albedo.json") as f:
        info = json.load(f)
    for coef in COEFS:
        keys.append((cond, coef)); matched.append(info["matched"][coef])
matched = np.array(matched)

d_true = np.array([truth("translucent", c, k) - truth("opaque", c, k) for c, k in keys])
m_t = np.array([model("zero123", c, k) for c, k in keys])        # (cells, seeds, 2)
m_o = np.array([model("zero123op", c, k) for c, k in keys])
d_model = m_t.mean(1) - m_o.mean(1)
# 乱数だけで生じる差の目安: 乱数 3 回の平均どうしの差の標準誤差
noise = np.sqrt(m_t.var(1, ddof=1) / len(SEEDS) + m_o.var(1, ddof=1) / len(SEEDS))

print(f"{'cell':34s}{'matched':>8s}{'true diff (lum / con)':>26s}{'model diff (lum / con)':>26s}{'noise':>18s}")
for i, (cond, coef) in enumerate(keys):
    print(f"{tag(cond):28s}{coef:>5s} {str(bool(matched[i])):>8s}"
          f"{d_true[i, 0]:>+14.3f} / {d_true[i, 1]:+.3f}{d_model[i, 0]:>+14.3f} / {d_model[i, 1]:+.3f}"
          f"{noise[i, 0]:>10.3f} / {noise[i, 1]:.3f}")

summary = {}
for label, sel in (("matched", matched), ("all", np.ones(len(keys), bool))):
    x, y = d_true[sel], d_model[sel]
    slope = (x * y).sum(0) / (x * x).sum(0)
    same_sign = (np.sign(x) == np.sign(y)).sum(0)
    summary[label] = {"n": int(sel.sum()), "slope": slope.tolist(),
                      "mean_abs_true_diff": np.abs(x).mean(0).tolist(),
                      "mean_abs_model_diff": np.abs(y).mean(0).tolist(),
                      "mean_noise": noise[sel].mean(0).tolist(), "same_sign": same_sign.tolist()}
    print(f"\n{label} cells (n={sel.sum()}): slope luminance {slope[0]:+.2f}, contrast {slope[1]:+.2f}   "
          f"|true diff| {np.abs(x).mean(0)[0]:.3f} / {np.abs(x).mean(0)[1]:.3f}   "
          f"|model diff| {np.abs(y).mean(0)[0]:.3f} / {np.abs(y).mean(0)[1]:.3f}   "
          f"noise {noise[sel].mean(0)[0]:.3f} / {noise[sel].mean(0)[1]:.3f}   "
          f"same sign {same_sign[0]}/{sel.sum()}, {same_sign[1]}/{sel.sum()}")
with open(OUT / "pairs_summary.json", "w") as f:
    json.dump(summary, f, indent=1)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
for k, name in enumerate(("mean luminance", "contrast")):
    ax = axes[k]
    for sel, fmt, lab in ((matched, "o", "front view matched"), (~matched, "x", "front view not matched")):
        ax.errorbar(d_true[sel, k], d_model[sel, k], yerr=noise[sel, k], fmt=fmt, ms=5, label=lab)
    lim = np.abs(np.concatenate([d_true[:, k], d_model[:, k]])).max() * 1.15
    ax.plot([-lim, lim], [-lim, lim], "k:", lw=0.8, label="slope 1: tracks the material")
    ax.axhline(0, color="gray", lw=0.8, label="slope 0: ignores the material")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("truth: translucent - opaque"); ax.set_ylabel("Zero-1-to-3: translucent input - opaque input")
    ax.set_title(f"{name}, unseen views")
axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "pairs_scatter.png", dpi=110); plt.close(fig)

# 例: 正面からのスタジオ照明、濁りが少ない条件。行 = 入力の種類、列 = 視点
cond, coef = CONDS[0], COEFS[0]
rows = [("truth, translucent", lambda az: load_object(GT / tag(cond) / f"gt_coef{coef}_az{az:03d}.png", 256, 1.0)),
        ("truth, opaque", lambda az: load_object(GT / tag(cond) / f"gt_opaque_coef{coef}_az{az:03d}.png", 256, 1.0)),
        ("model, translucent input", lambda az: Image.open(GEN / tag(cond) / f"zero123_coef{coef}_az{az:03d}_seed0.png")),
        ("model, opaque input", lambda az: Image.open(GEN / tag(cond) / f"zero123op_coef{coef}_az{az:03d}_seed0.png"))]
fig, axes = plt.subplots(len(rows), len(AZ), figsize=(2.2 * len(AZ), 2.3 * len(rows)))
for r, (name, get) in enumerate(rows):
    for c, az in enumerate(AZ):
        axes[r, c].imshow(get(az)); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == 0:
            axes[r, c].set_title(f"azimuth {az} deg", fontsize=9)
        if c == 0:
            axes[r, c].set_ylabel(name, fontsize=8)
fig.suptitle(f"{tag(cond)}, density {coef}", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "pairs_example.png", dpi=90); plt.close(fig)

# 追加の解析: モデルの裏側の差は、本物の裏側の差と、入力（正面）の差のどちらに従うか。
# 「正面の見えを裏側へ持ち越しているだけ」なら、正面の差との相関のほうが高くなる。
from common import stim_path  # noqa: E402


def front(path):
    return np.array(stats(*load_object(path, 256, bg=1.0, with_mask=True)))


d_front = np.array([front(stim_path(c, k)) - front(GT / tag(c) / f"gt_opaque_coef{k}_az000.png") for c, k in keys])
copy = {}
print("\nmatched cells: does the model's far-side difference follow the truth or the front view of the input?")
for i, name in enumerate(("luminance", "contrast")):
    f, m, t = d_front[matched, i], d_model[matched, i], d_true[matched, i]
    copy[name] = {"max_abs_front_diff": float(np.abs(f).max()), "r_model_front": float(np.corrcoef(m, f)[0, 1]),
                  "r_model_truth": float(np.corrcoef(m, t)[0, 1]), "r_front_truth": float(np.corrcoef(f, t)[0, 1]),
                  "same_sign_as_front": int((np.sign(f) == np.sign(m)).sum())}
    print(f"{name:10s} largest front difference {np.abs(f).max():.3f}   r(model, front) {copy[name]['r_model_front']:+.2f}   "
          f"r(model, truth) {copy[name]['r_model_truth']:+.2f}   r(front, truth) {copy[name]['r_front_truth']:+.2f}   "
          f"same sign as front {copy[name]['same_sign_as_front']}/{matched.sum()}")
summary["front_copy"] = copy
with open(OUT / "pairs_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
