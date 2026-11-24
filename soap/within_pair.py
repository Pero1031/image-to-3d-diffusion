"""同じ石けんの写真 2 枚の評定の差を、特徴から予測できるかを調べる。

    python soap/within_pair.py

動機: 写真ごとの評定のばらつきのほとんどは石けんどうしの差で、石けんの種類（透明な石けん、乳白色の
石けん）を見分ける特徴でも評定は予測できてしまう。同じ石けんの 2 枚は、物体は同じで照明（拡散光か、
逆光・斜光か）だけが違う。その評定の差まで予測できれば、読み出しは物体の種類ではなく画像の見えに
基づいている。

2 つの方法:
    方法 1  predict.py と同じ読み出し（個体ごとの交差検証、両方の写真が評価側）の予測について、
            2 枚の予測の差と評定の差を比べる。石けんどうしの差で学習した読み出しが、そのまま同じ
            石けんの中の差も追うかを見る。
    方法 2  2 枚の特徴の差から評定の差を予測するように学習する（1 組ずつ評価側に回す）。
指標: 予測の差と評定の差の相関（原点まわり。2 枚の順番を入れ替えても値が変わらない）。
偶然の水準は組ごとに差の符号を入れ替えて求め、上限は評定者どうしの差の一致度から求める。
"""
import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV

from ratings import ALPHAS, ROOT, cv_predict, load

OUT = ROOT / "soap" / "figures"
rng = np.random.default_rng(0)

data, names, groups, Z, y = load()
ids = sorted(set(groups))
P = np.array([[i for i in range(len(names)) if groups[i] == g] for g in ids])   # (61, 2) 名前順
d = y[P[:, 0]] - y[P[:, 1]]


def r0(a, b):
    """原点まわりの相関。組の向きを入れ替えると a, b の符号が両方変わるので、値は向きによらない。"""
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


# --- 評定の差の信頼性: 6 人を 3 人ずつに分け、それぞれの平均から出した差どうしの相関 ---
D = Z[:, P[:, 0]] - Z[:, P[:, 1]]
half = [np.corrcoef(np.nanmean(D[list(a)], 0), np.nanmean(D[[i for i in range(6) if i not in a]], 0))[0, 1]
        for a in itertools.combinations(range(6), 3) if 0 in a]
sh = float(np.mean(half)); rel = 2 * sh / (1 + sh)
print(f"share of rating variance within objects: {((y[P] - y[P].mean(1, keepdims=True)) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}")
print(f"within-pair difference: mean |d| = {np.abs(d).mean():.3f}, split-half r = {sh:.3f}, "
      f"reliability = {rel:.3f}, ceiling r = {np.sqrt(rel):.2f}")
# 追加（結果を見てから加えたもの）: 差の向きの一致。差が小さい組では、評定者どうしでも向きが揃うとは限らない
large = np.abs(d) > np.median(np.abs(d))
agree = np.mean([np.sign(np.nanmean(D[list(a)], 0)) == np.sign(np.nanmean(D[[i for i in range(6) if i not in a]], 0))
                 for a in itertools.combinations(range(6), 3) if 0 in a], axis=0)      # 組ごとの、折半どうしの一致率
rater_sign = {"small": float(agree[~large].mean()), "large": float(agree[large].mean())}
print(f"two halves of the raters agree on the direction: {rater_sign['small']:.2f} of pairs with a small difference, "
      f"{rater_sign['large']:.2f} with a large one (split at the median |d| = {np.median(np.abs(d)):.2f})")


def method2(X, target):
    """組の特徴の差 -> 評定の差。切片なし（向きを入れ替えると両方の符号が変わるだけなので）。1 組ずつ評価側。"""
    pred = np.zeros(len(target))
    for i in range(len(target)):
        tr = np.arange(len(target)) != i
        s = np.sqrt((X[tr] ** 2).mean(0)) + 1e-12          # 原点まわりの大きさで揃える
        pred[i] = RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(X[tr] / s, target[tr]).predict(X[i:i + 1] / s)[0]
    return pred


def summarize(p, refit=None, n_boot=2000, n_flip=10000, n_refit=200):
    r = r0(p, d)
    boot = [r0(p[i], d[i]) for i in (rng.integers(0, len(d), len(d)) for _ in range(n_boot))]
    if refit is None:      # 予測は評価側の評定に依存しないので、評定の差の符号だけを入れ替えればよい
        null = [r0(p, d * rng.choice([-1, 1], len(d))) for _ in range(n_flip)]
    else:                  # 学習にも評定の差を使うので、符号を入れ替えて学習からやり直す
        null = []
        for _ in range(n_refit):
            dd = d * rng.choice([-1, 1], len(d))
            null.append(r0(refit(dd), dd))
    return {"r": r, "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "null_95": float(np.percentile(null, 95)), "p": float((1 + np.sum(np.array(null) >= r)) / (1 + len(null))),
            "same_sign_small": int((np.sign(p) == np.sign(d))[~large].sum()),
            "same_sign_large": int((np.sign(p) == np.sign(d))[large].sum()),
            "slope": float((p * d).sum() / (p * p).sum()),           # 評定の差 = slope x 予測の差
            "mean_abs_pred": float(np.abs(p).mean())}


sets = ["color", "stats14", "stats14+color",
        "zero123_clip_bg", "zero123_clip_nobg",
        "zero123_unet_t100_bg", "zero123_unet_t100_nobg", "zero123_unet_t500_bg", "zero123_unet_t500_nobg",
        "zero123_unet_t900_bg", "zero123_unet_t900_nobg", "triposr_dino_bg", "triposr_dino_nobg"]
feat = {s: data[s].astype(np.float64) for s in sets if s in data.files}
feat["stats14+color"] = np.concatenate([data["stats14"], data["color"]], 1).astype(np.float64)

res, p1, p2 = {}, {}, {}
print(f"\n{'features':26s}{'between r':>10s}{'M1 r':>8s}{'95% CI':>15s}{'chance':>8s}{'p':>7s}{'sign s/l':>10s}"
      f"{'M2 r':>8s}{'95% CI':>15s}{'chance':>8s}{'p':>7s}")
for s in sets:
    X = feat[s]
    pred = cv_predict(X, y, groups)
    p1[s] = pred[P[:, 0]] - pred[P[:, 1]]
    Xd = X[P[:, 0]] - X[P[:, 1]]
    p2[s] = method2(Xd, d)
    m1, m2 = summarize(p1[s]), summarize(p2[s], refit=lambda dd, Xd=Xd: method2(Xd, dd))
    res[s] = {"between_r": float(np.corrcoef(pred[P].mean(1), y[P].mean(1))[0, 1]), "method1": m1, "method2": m2}
    print(f"{s:26s}{res[s]['between_r']:>10.3f}{m1['r']:>8.3f}{'[%.2f, %.2f]' % tuple(m1['ci']):>15s}"
          f"{m1['null_95']:>8.2f}{m1['p']:>7.3f}{'%d/%d' % (m1['same_sign_small'], m1['same_sign_large']):>10s}"
          f"{m2['r']:>8.3f}{'[%.2f, %.2f]' % tuple(m2['ci']):>15s}{m2['null_95']:>8.2f}{m2['p']:>7.3f}", flush=True)
print(f"(sign s/l: pairs whose predicted difference has the rated direction, among the {(~large).sum()} pairs with a "
      f"small and the {large.sum()} with a large rating difference)")

# モデルの特徴は、画像統計より同じ石けんの中の差をよく追うか（組を単位にした対の再標本化）
comp = {}
for m, pp in (("method1", p1), ("method2", p2)):
    for s in ("zero123_clip_bg", "zero123_clip_nobg", "triposr_dino_bg"):
        diff = []
        for _ in range(2000):
            i = rng.integers(0, len(d), len(d))
            diff.append(r0(pp[s][i], d[i]) - r0(pp["stats14+color"][i], d[i]))
        comp[f"{m}:{s}-stats14+color"] = {"diff": r0(pp[s], d) - r0(pp["stats14+color"], d),
                                          "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
for k, v in comp.items():
    print(f"{k:45s} {v['diff']:+.3f}  95% CI [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]")

# 追加: 画像統計の各指標と評定の関係。写真どうし（ほとんどが石けんどうしの差）と、同じ石けんの 2 枚の差とで比べる
cues = {}
print(f"\n{'image statistic':32s}{'r across photos':>17s}{'r within soaps':>16s}")
for j, s in enumerate(data["stat_names"]):
    v = data["stats14"][:, j].astype(np.float64)
    cues[str(s)] = {"across": float(np.corrcoef(v, y)[0, 1]), "within": float(np.corrcoef(v[P[:, 0]] - v[P[:, 1]], d)[0, 1])}
    print(f"{s:32s}{cues[str(s)]['across']:>+17.2f}{cues[str(s)]['within']:>+16.2f}")

with open(OUT / "within_pair_summary.json", "w") as f:
    json.dump({"n_pairs": len(d), "split_half_r": sh, "reliability": rel, "ceiling_r": float(np.sqrt(rel)),
               "mean_abs_diff": float(np.abs(d).mean()), "median_abs_diff": float(np.median(np.abs(d))),
               "n_small": int((~large).sum()), "n_large": int(large.sum()), "rater_halves_same_sign": rater_sign,
               "results": res, "comparisons": comp, "image_statistic_cues": cues}, f, indent=1)

# --- 図: 左 = 評定の差と予測の差（方法 1）、右 = 特徴ごとの相関 ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
ax = axes[0]
for s, c, lab in (("stats14+color", "0.5", "image statistics + colour"),
                  ("zero123_clip_nobg", "C1", "Zero-1-to-3 CLIP, object on white")):
    sgn = np.sign(d)                            # 評定が高いほうを先にそろえて描く
    ax.scatter(p1[s] * sgn, d * sgn, s=14, color=c, label=f"{lab} (r = {res[s]['method1']['r']:.2f})")
lim = max(np.abs(d).max(), np.abs(p1["zero123_clip_nobg"]).max(), np.abs(p1["stats14+color"]).max()) * 1.1
ax.plot([-lim, lim], [-lim, lim], "k:", lw=0.8); ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
ax.axhline(np.median(np.abs(d)), color="gray", lw=0.6, ls="--")
ax.text(-lim * 0.97, np.median(np.abs(d)), "median", fontsize=7, va="bottom", color="gray")
ax.set_xlim(-lim, lim); ax.set_ylim(0, lim)
ax.set_xlabel("predicted difference (photo with higher rating - other)"); ax.set_ylabel("rating difference")
ax.set_title("same soap, two photos (method 1)", fontsize=9); ax.legend(fontsize=7, loc="upper left")
ax = axes[1]
for m, off, mk in (("method1", -0.17, "o"), ("method2", 0.17, "s")):
    for i, s in enumerate(sets):
        v = res[s][m]
        color = "0.5" if not s.startswith(("zero123", "triposr")) else ("C0" if s.endswith("_bg") else "C1")
        ax.errorbar(v["r"], i + off, xerr=[[v["r"] - v["ci"][0]], [v["ci"][1] - v["r"]]], fmt=mk, ms=4,
                    color=color, capsize=2, mfc=color if m == "method1" else "white")
ax.axvline(np.sqrt(rel), color="k", ls=":", lw=1); ax.axvline(0, color="gray", lw=0.8)
ax.set_yticks(range(len(sets)), sets, fontsize=7); ax.invert_yaxis()
ax.set_xlabel("r between predicted and rated difference (61 soaps)")
ax.set_title("filled: readout trained across soaps (1), open: trained on differences (2); dotted: ceiling",
             fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "within_pair.png", dpi=120); plt.close(fig)
