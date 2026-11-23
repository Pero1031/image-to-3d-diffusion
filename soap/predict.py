"""特徴から人の半透明感の評定を予測できるかを、個体ごとの交差検証で調べる。

    python soap/predict.py

目標: 参加者ごとに標準化した評定の、写真ごとの平均。
方法: 標準化 -> リッジ回帰（正則化の強さは学習側の中で選ぶ）。同じ石けんの 2 枚は必ず同じ側に入れる。
指標: 評価側の予測と評定の相関 r。信頼区間は個体を単位にした再標本化、偶然の水準は評定を個体間で入れ替えて求める。
"""
import csv
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
RATINGS = Path("D:/project/vision-experiments/Tobii_pro/data/master_trial_summary.csv")
OUT = ROOT / "soap" / "figures"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

data = np.load(ROOT / "data" / "soap_features.npz")
names = list(data["names"])
groups = np.array([n.split("_")[0] for n in names])            # 個体 ID（soap<N>）

# --- 評定: 参加者ごとに標準化し、写真ごとに平均 ---
rows = list(csv.DictReader(open(RATINGS, encoding="utf-8-sig")))
people = sorted({r["participant_id"] for r in rows})
R = np.full((len(people), len(names)), np.nan)
for r in rows:
    R[people.index(r["participant_id"]), names.index(r["image_name"])] = float(r["rating"])
Z = (R - np.nanmean(R, 1, keepdims=True)) / np.nanstd(R, 1, keepdims=True)
y = np.nanmean(Z, 0)
# 評定者どうしの一致度: 6 人を 3 人ずつに分けた平均どうしの相関（全 10 通りの平均）と、6 人分に補正した値
half = [np.corrcoef(np.nanmean(Z[list(a)], 0), np.nanmean(Z[[i for i in range(len(people)) if i not in a]], 0))[0, 1]
        for a in itertools.combinations(range(len(people)), len(people) // 2) if 0 in a]
split_half = float(np.mean(half))
reliability = 2 * split_half / (1 + split_half)
print(f"{len(people)} raters, {len(names)} images, {len(set(groups))} objects; missing ratings: {int(np.isnan(R).sum())}")
print(f"split-half r = {split_half:.3f}, reliability of the 6-rater mean = {reliability:.3f} "
      f"(a perfect predictor would reach about r = {np.sqrt(reliability):.2f})")
pair = np.array([y[groups == g] for g in sorted(set(groups))])
print(f"same object, two photos: r = {np.corrcoef(pair[:, 0], pair[:, 1])[0, 1]:.3f}")


def cv_predict(X, target):
    """個体を 1 つずつ評価側に回す交差検証。評価側の予測を返す。"""
    pred = np.zeros(len(target))
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
    for tr, te in LeaveOneGroupOut().split(X, target, groups):
        pred[te] = model.fit(X[tr], target[tr]).predict(X[te])
    return pred


def evaluate(X, n_boot=2000, n_perm=200):
    pred = cv_predict(X, y)
    r = np.corrcoef(pred, y)[0, 1]
    ids = sorted(set(groups))
    index = {g: np.where(groups == g)[0] for g in ids}
    boot = []
    for _ in range(n_boot):                       # 個体を単位に再標本化
        i = np.concatenate([index[g] for g in rng.choice(ids, len(ids))])
        boot.append(np.corrcoef(pred[i], y[i])[0, 1])
    null = []
    for _ in range(n_perm):                       # 評定を個体ごと入れ替える（同じ個体の 2 枚は一緒に動かす）
        perm = dict(zip(ids, rng.permutation(ids)))
        yp = np.empty_like(y)
        for g in ids:
            yp[index[g]] = y[index[perm[g]]]
        null.append(np.corrcoef(cv_predict(X, yp), yp)[0, 1])
    p = (1 + np.sum(np.array(null) >= r)) / (1 + n_perm)
    return {"r": float(r), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "null_95": float(np.percentile(null, 95)), "p": float(p), "dim": int(X.shape[1])}, pred


sets = ["color", "stats14",
        "zero123_clip_bg", "zero123_clip_nobg",
        "zero123_unet_t100_bg", "zero123_unet_t100_nobg", "zero123_unet_t500_bg", "zero123_unet_t500_nobg",
        "zero123_unet_t900_bg", "zero123_unet_t900_nobg", "triposr_dino_bg", "triposr_dino_nobg"]
results, preds = {}, {}
print(f"\n{'features':28s}{'dim':>6s}{'r':>8s}{'95% CI':>18s}{'chance 95%':>12s}{'p':>8s}")
for s in sets:
    results[s], preds[s] = evaluate(data[s].astype(np.float64))
    v = results[s]
    print(f"{s:28s}{v['dim']:>6d}{v['r']:>8.3f}{'[%.2f, %.2f]' % tuple(v['ci']):>18s}{v['null_95']:>12.3f}{v['p']:>8.3f}", flush=True)

# 画像統計にモデルの特徴を足すと良くなるか（モデルの特徴が、画像統計にない情報を持っているか）
best = max((s for s in sets if s not in ("color", "stats14")), key=lambda s: results[s]["r"])
combo = np.concatenate([data["stats14"], data["color"]], 1).astype(np.float64)
results["stats14+color"], preds["stats14+color"] = evaluate(combo)
print(f"{'stats14+color':28s}{combo.shape[1]:>6d}{results['stats14+color']['r']:>8.3f}"
      f"{'[%.2f, %.2f]' % tuple(results['stats14+color']['ci']):>18s}")
# 個体を単位にした対の再標本化で、最良のモデル特徴と画像統計の差の信頼区間を出す
ids = sorted(set(groups)); index = {g: np.where(groups == g)[0] for g in ids}
diff = []
for _ in range(2000):
    i = np.concatenate([index[g] for g in rng.choice(ids, len(ids))])
    diff.append(np.corrcoef(preds[best][i], y[i])[0, 1] - np.corrcoef(preds["stats14+color"][i], y[i])[0, 1])
d_ci = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
print(f"\nbest model features: {best}; r difference to stats14+color = "
      f"{results[best]['r'] - results['stats14+color']['r']:+.3f}, 95% CI [{d_ci[0]:+.2f}, {d_ci[1]:+.2f}]")

with open(OUT / "predict_summary.json", "w") as f:
    json.dump({"n_raters": len(people), "n_images": len(names), "n_objects": len(ids), "split_half_r": split_half,
               "reliability": reliability, "results": results, "best_model_features": best,
               "best_minus_stats_ci": d_ci}, f, indent=1)

order = ["color", "stats14", "stats14+color"] + [s for s in sets if s not in ("color", "stats14")]
fig, ax = plt.subplots(figsize=(8, 4.6))
for i, s in enumerate(order):
    v = results[s]
    color = "0.5" if not s.startswith(("zero123", "triposr")) else ("C0" if s.endswith("_bg") else "C1")
    ax.errorbar(v["r"], i, xerr=[[v["r"] - v["ci"][0]], [v["ci"][1] - v["r"]]], fmt="o", color=color, capsize=3)
ax.axvline(np.sqrt(reliability), color="k", ls=":", lw=1)
ax.text(np.sqrt(reliability), len(order) - 0.4, " ceiling from rater agreement", fontsize=7, va="top", rotation=90, ha="right")
ax.axvline(0, color="gray", lw=0.8)
ax.set_yticks(range(len(order)), order, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("cross-validated r with the mean translucency rating (held-out objects)")
ax.set_title("gray: image statistics, blue: photo with background, orange: object on white", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "predict_r.png", dpi=120); plt.close(fig)
