"""視線と読み出しの地図 (2a)、見た所とその人の評定 (2b)。

    python soap/gaze_readout.py        （先に soap/occlusion_maps.py）

soap/gaze_model.py の続き。地図は soap/occlusion_maps.py の遮蔽地図 O（CLIP の画像埋め込みの読み出しを、写真の
一部をぼかして調べたもの）と、前回のパッチ地図 S（CLIP、DINO）。格子は 16 x 16（1 セル = 元の 64 画素 = 1.6°）で、
物体のセル（領域が半分以上）だけを使う。地図は写真ごとに物体の中で標準化する。
視線: 参加者 5 人（P003 は前回と同じ理由で除く。2b では入れた場合も示す）。提示開始から 0.1 秒より前に始まった
      注視は注視点から持ち越したものなので捨てる（前回の解析は捨てていない）。物体のセルの外の注視も使わない。
典型の見方: ほかの石けんの注視（5 人分）を、この写真の物体の外接矩形に写した密度（σ = 30 画素）。中心に寄る・
      上に寄るといった、写真によらない見方を表す。

2a  人は地図の高い所を見るか
    (i)  注視点での地図の値から典型の見方での期待値を引いたもの（前回と同じ指標。標準偏差の単位）を、注視全体、
         最初の 2 回（持ち越しを除いて数える）、3 回目以降に分けて求める。
    (ii) 写真の中でどのセルが見られるかのモデル（条件つきロジット: P(セル) ∝ exp(β・特徴)）。特徴は
         log 典型の密度 + 輝度 に、地図を 1 つずつ足す。石けんごとに評価側へ回して、注視 1 回あたりの情報量
         （物体の中で一様に見る場合からの bit）を求める。上限の目安: 同じ写真のほかの 4 人の注視の密度を足した
         モデル（4 人分しかないので、本当の上限より低めに出る）。
2b  見た所とその人の評定は結びつくか（試行ごと、5 人 x 122 枚）
    評定は参加者ごとに標準化した値。写真と参加者の 2 方向の固定効果を除いた残りどうしの偏相関を見る。つまり
    「同じ写真で、ほかの人よりここを見た人は、ほかの人より高く（低く）評定したか」。
      V            その人の注視点のまわり（半径 64 画素）だけくっきり残し、ほかをぼかした写真の読み出し
                   （soap/occlusion_maps.py）。その人が見た情報だけからモデルが出す判断。
      O, S, 輝度   その人の注視点での地図の値の平均（物体の中で標準化した値）。
    区間は石けんを単位にした再標本化、p 値は写真ごとに参加者の間で視線の値を入れ替えて求める。
    残りの自由度は約 490 なので、95% 区間の半幅は r で ±0.09 ほど。

実行前に決めたこと（soap/occlusion_maps.py を 3 枚で試しに動かした後、本番の結果を見る前に書いた。その後、
O と V にでたらめな値を入れてこのスクリプトを試しに動かしたとき、輝度・S・上限の行は本当の値が出た。予測は
そのあと変えていない）:
  2a 予測 1  輝度の効果（典型との差）は前回と同じく +0.2 前後。最初の 2 回の注視で強く、3 回目以降で弱まる。
     予測 2  |O| は輝度と相関するので単独では正に出うるが、log 典型 + 輝度 に足したときの情報量の上積みは、上限の
             モデルの上積み（ほかの 4 人の密度）の 10% 未満。
     予測 3  符号つきの地図（O、S）の典型との差は 0 に近い（|差| < 0.1 SD）。3 回目以降で増えることはあっても小さい。
     判断    地図を足した上積みの 95% 区間が 0 を含めば、その地図は輝度と典型の見方を超えて視線を説明しない。
  2b 判断    偏相関の 95% 区間が 0 を含まなければ、見た所とその人の評定が結びついている証拠とする。
     予測    V とその人の評定の偏相関は小さい（|r| < 0.15）。注視点の輝度との偏相関も小さい。
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
from scipy.optimize import minimize
from scipy.special import logsumexp

from ratings import ROOT, load

FIXATIONS = Path("D:/project/vision-experiments/Tobii_pro/data/master_fixations.csv")
MASKS = ROOT / "data" / "soap_masks" / "final"
OUT = ROOT / "soap" / "figures"
SIZE, BASE, SIGMA, G = 1024, 128, 30.0, 16
MIN_ONSET = 0.1
EXCLUDE = {"P003"}
N_BOOT, N_PERM = 2000, 5000
rng = np.random.default_rng(0)

data, names, groups, Z, y = load()
occ = np.load(ROOT / "data" / "soap_occlusion.npz")
assert list(occ["names"]) == names
people_all = list(occ["people"])
assert Z.shape[0] == len(people_all)          # ratings.load も参加者を名前の順に並べる
main =[p for p in people_all if p not in EXCLUDE]
ids = sorted(set(groups))
index = {g: np.where(groups == g)[0] for g in ids}
n = len(names)
objcell = occ["cellfrac"] >= 0.5


def down(a, g):
    k = a.shape[0] // g
    return a.reshape(g, k, g, k).mean((1, 3))


def density(xy):
    """注視点 (k, 2)（元の画像の画素）-> ぼかした密度 (16, 16)。"""
    h = np.histogram2d(xy[:, 1], xy[:, 0], bins=BASE, range=[[0, SIZE], [0, SIZE]])[0] if len(xy) else np.zeros((BASE, BASE))
    return down(ndimage.gaussian_filter(h, SIGMA / (SIZE / BASE)), G)


def zmap(v, obj):
    out = np.zeros_like(v, dtype=np.float64)
    out[obj] = (v[obj] - v[obj].mean()) / v[obj].std()
    return out


def boot_soaps(fn, n_boot=N_BOOT):
    out = [fn(np.concatenate([index[g] for g in rng.choice(ids, len(ids))])) for _ in range(n_boot)]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


# ---------------------------------------------------------------- 注視（持ち越しを除き、残りに順番をつける）
rows = list(csv.DictReader(open(FIXATIONS, encoding="utf-8-sig")))
raw = {(p, nm): [] for p in people_all for nm in names}
for r in rows:
    if r["x_img_px"] and r["image_name"] in names and float(r["onset_sec"]) >= MIN_ONSET:
        x, yy = float(r["x_img_px"]), float(r["y_img_px"])
        if 0 <= x < SIZE and 0 <= yy < SIZE:
            raw[(r["participant_id"], r["image_name"])].append((float(r["onset_sec"]), x, yy))
fix = {}          # (参加者, 写真番号) -> (k, 3) = x, y, 順番（0 から）
for (p, nm), v in raw.items():
    v = sorted(v)
    fix[(p, names.index(nm))] = np.array([(x, yy, k) for k, (_, x, yy) in enumerate(v)]).reshape(-1, 3)


def cells_of(f, i):
    """注視 (k, 3) -> 物体のセルに落ちたものの (行, 列, 順番)。"""
    c = np.clip((f[:, :2] * G / SIZE).astype(int), 0, G - 1)
    hit = objcell[i][c[:, 1], c[:, 0]]
    return c[hit, 1], c[hit, 0], f[hit, 2]


n_all = sum(len(fix[(p, i)]) for p in main for i in range(n))
n_obj = sum(len(cells_of(fix[(p, i)], i)[0]) for p in main for i in range(n))
print(f"5 observers {main}: {n_all} fixations after dropping the carried-over ones; on object cells {n_obj} "
      f"({n_obj / n_all:.0%})")

# ---------------------------------------------------------------- 地図と典型の見方
maps = {"luminance": occ["lum16"], "abs_O": np.abs(np.nan_to_num(occ["O"])), "O": np.nan_to_num(occ["O"]),
        "S_clip": occ["S_clip"], "S_dino": np.array([down(s, G) for s in occ["S_dino"]])}
zm = {k: np.array([zmap(v[i], objcell[i]) for i in range(n)]) for k, v in maps.items()}
masks = [np.asarray(Image.open(MASKS / (Path(nm).stem + ".png"))) > 0 for nm in names]
boxes = []
for m in masks:
    ys, xs = np.where(m)
    boxes.append((xs.min(), xs.max() + 1, ys.min(), ys.max() + 1))
uv = [np.concatenate([np.column_stack([(fix[(p, i)][:, 0] - b[0]) / (b[1] - b[0]), (fix[(p, i)][:, 1] - b[2]) / (b[3] - b[2])])
                      for p in main]) for i, b in enumerate(boxes)]


def to_image(u, b):
    xy = np.column_stack([b[0] + u[:, 0] * (b[1] - b[0]), b[2] + u[:, 1] * (b[3] - b[2])])
    return xy[(xy[:, 0] >= 0) & (xy[:, 0] < SIZE) & (xy[:, 1] >= 0) & (xy[:, 1] < SIZE)]


def logp(d, obj):
    """密度 -> 物体のセルでの log 確率（小さい値を足して 0 を避ける）。"""
    p = d[obj] / d[obj].sum()
    return np.log(p + 0.1 / obj.sum())


typ = [density(to_image(np.concatenate([uv[j] for j in range(n) if groups[j] != groups[i]]), boxes[i])) for i in range(n)]
p_typ = [typ[i][objcell[i]] / typ[i][objcell[i]].sum() for i in range(n)]
log_typ = [logp(typ[i], objcell[i]) for i in range(n)]
PERIODS = {"all": lambda k: k >= 0, "early (1st-2nd)": lambda k: k < 2, "late (3rd on)": lambda k: k >= 2}

# ---------------------------------------------------------------- 2a (i) 注視点での地図の値 - 典型の見方での期待値
summary = {"observers": main, "fixations": n_all, "on_object_cells": n_obj}
nss = {}
print("\n2a (i) map value at fixations minus that expected from the typical layout (SD within the object)")
for per, sel in PERIODS.items():
    nss[per] = {}
    for k in zm:
        d = np.full(n, np.nan)
        for i in range(n):
            r_, c_ = [], []
            for p in main:
                rr, cc, kk = cells_of(fix[(p, i)], i)
                r_.append(rr[sel(kk)]); c_.append(cc[sel(kk)])
            r_, c_ = np.concatenate(r_), np.concatenate(c_)
            if len(r_):
                d[i] = zm[k][i][r_, c_].mean() - (p_typ[i] * zm[k][i][objcell[i]]).sum()
        ok = ~np.isnan(d)
        nss[per][k] = {"difference": float(np.nanmean(d)), "ci": boot_soaps(lambda s: np.nanmean(d[s])),
                       "images_above": int((d[ok] > 0).sum()), "images": int(ok.sum())}
    print(f"  {per:16s} " + ", ".join(f"{k} {v['difference']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
                                         for k, v in nss[per].items()))
summary["nss_minus_typical"] = nss


# ---------------------------------------------------------------- 2a (ii) どのセルが見られるかのモデル
def negll(beta, units):
    ll, g = 0.0, np.zeros_like(beta)
    for X, c in units:
        s = X @ beta
        lp = s - logsumexp(s)
        ll += c @ lp
        g += c @ X - c.sum() * (np.exp(lp) @ X)
    return -ll, -g


def fit(units, d):
    return minimize(negll, np.zeros(d), args=(units,), jac=True, method="L-BFGS-B").x


def unit_list(keys, period, others=False):
    """(写真番号, 特徴 (セル数, d), 注視の数 (セル数)) のリスト。others=True ならほかの 4 人の密度を足し、
    参加者ごとに分ける。"""
    out = []
    for i in range(n):
        base = np.column_stack([log_typ[i]] + [zm[k][i][objcell[i]] for k in keys])
        units_i = []
        for p in main:
            rr, cc, kk = cells_of(fix[(p, i)], i)
            sel = PERIODS[period](kk)
            cnt = np.zeros((G, G)); np.add.at(cnt, (rr[sel], cc[sel]), 1)
            X = base
            if others:
                oth = np.concatenate([fix[(q, i)][:, :2] for q in main if q != p])
                X = np.column_stack([base, logp(density(oth), objcell[i])])
            units_i.append((X, cnt[objcell[i]]))
        if not others:
            units_i = [(base, sum(c for _, c in units_i))]
        out.append(units_i)
    return out


def cv_ll(keys, period, others=False):
    """石けんごとに評価側へ回した、写真ごとの (対数尤度 - 一様の対数尤度) [nat] と注視の数。"""
    units = unit_list(keys, period, others)
    d = 1 + len(keys) + others
    gain, cnt = np.zeros(n), np.zeros(n)
    for g in ids:
        beta = fit([u for j in range(n) if groups[j] != g for u in units[j]], d)
        for i in index[g]:
            for X, c in units[i]:
                s = X @ beta
                gain[i] += c @ (s - logsumexp(s)) + c.sum() * np.log(len(c))
                cnt[i] += c.sum()
    return gain, cnt


MODELS = {"typical": [], "+ luminance": ["luminance"], "+ luminance + |O|": ["luminance", "abs_O"],
          "+ luminance + O": ["luminance", "O"], "+ luminance + S (CLIP)": ["luminance", "S_clip"],
          "+ luminance + S (DINO)": ["luminance", "S_dino"],
          "+ luminance + all four maps": ["luminance", "abs_O", "O", "S_clip", "S_dino"]}
ig = {}
print("\n2a (ii) information gain per fixation over a uniform layout on the object, held-out soaps [bits]")
for per in PERIODS:
    res = {m: cv_ll(k, per) for m, k in MODELS.items()}
    res["ceiling: + luminance + other 4 observers"] = cv_ll(["luminance"], per, others=True)
    base_g, cnt = res["+ luminance"]
    bits = lambda g, s=slice(None): float(g[s].sum() / cnt[s].sum() / np.log(2))
    ceil_g = res["ceiling: + luminance + other 4 observers"][0]
    ig[per] = {"fixations": int(cnt.sum())}
    for m, (g, _) in res.items():
        e = {"bits": bits(g)}
        if m not in ("typical", "+ luminance"):
            e["gain_over_luminance"] = bits(g - base_g)
            e["ci"] = boot_soaps(lambda s: bits(g - base_g, s))
            e["share_of_ceiling_gain"] = float((g - base_g).sum() / (ceil_g - base_g).sum())
        ig[per][m] = e
    print(f"  {per} ({int(cnt.sum())} fixations)")
    for m, e in ig[per].items():
        if m == "fixations":
            continue
        extra = (f"; over luminance {e['gain_over_luminance']:+.4f} [{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}], "
                 f"{e['share_of_ceiling_gain']:.0%} of the ceiling's") if "ci" in e else ""
        print(f"    {m:42s} {e['bits']:.4f}{extra}")
coefs = {}
for m in ("+ luminance", "+ luminance + |O|", "+ luminance + O", "+ luminance + all four maps"):
    units = [u for us in unit_list(MODELS[m], "all") for u in us]
    coefs[m] = dict(zip(["log_typical"] + MODELS[m], map(float, fit(units, 1 + len(MODELS[m])))))
    print(f"  coefficients, all data, {m}: " + ", ".join(f"{k} {v:+.3f}" for k, v in coefs[m].items()))
summary["information_gain"], summary["coefficients"] = ig, coefs


# ---------------------------------------------------------------- 2b 見た所とその人の評定
def fe_matrix(img, obs):
    """写真と参加者の固定効果を除く射影 I - D D^+。"""
    D = np.column_stack([np.eye(img.max() + 1)[img], np.eye(obs.max() + 1)[obs]])
    return np.eye(len(img)) - D @ np.linalg.pinv(D)


def trial_table(observers):
    t = {k: [] for k in ("img", "obs", "rating", "V", "coverage", "lum", "O", "S_clip", "abs_O", "nfix_obj")}
    for o, p in enumerate(observers):
        pa = people_all.index(p)
        for i in range(n):
            rr, cc, _ = cells_of(fix[(p, i)], i)
            t["img"].append(i); t["obs"].append(o); t["rating"].append(Z[pa, i])
            t["V"].append(occ["V"][pa, i]); t["coverage"].append(occ["coverage"][pa, i]); t["nfix_obj"].append(len(rr))
            for k in ("lum", "O", "S_clip", "abs_O"):
                key = "luminance" if k == "lum" else k
                t[k].append(zm[key][i][rr, cc].mean() if len(rr) else np.nan)
    return {k: np.array(v) for k, v in t.items()}


def partial_r(t, x, rows):
    M = fe_matrix(t["img"][rows], t["obs"][rows])
    a, b = M @ t["rating"][rows], M @ t[x][rows]
    return float(a @ b / np.sqrt(a @ a * (b @ b)))


def analyse(observers, label):
    t = trial_table(observers)
    ok = np.isfinite(t["lum"]) & np.isfinite(t["rating"])
    out = {"trials": int(ok.sum()), "trials_without_object_fixations": int((t["nfix_obj"] == 0).sum())}
    Vm = np.array([t["V"][t["img"] == i].mean() for i in range(n)])
    out["r_mean_V_with_rating_across_photos"] = float(np.corrcoef(Vm, y)[0, 1])
    out["r_mean_V_with_full_readout"] = float(np.corrcoef(Vm, occ["R0"])[0, 1])
    out["coverage_median"] = float(np.median(t["coverage"]))
    print(f"\n2b {label}: {out['trials']} trials ({out['trials_without_object_fixations']} without fixations on "
          f"object cells, left out); median share of the object left sharp {out['coverage_median']:.2f}")
    print(f"   across photos, the mean of V over observers: r = {out['r_mean_V_with_rating_across_photos']:.3f} with "
          f"the rating, {out['r_mean_V_with_full_readout']:.3f} with the readout of the whole photo")
    rows = np.where(ok)[0]
    Mfull = fe_matrix(t["img"][rows], t["obs"][rows])
    a = Mfull @ t["rating"][rows]
    for x in ("V", "lum", "O", "S_clip", "abs_O", "coverage"):
        r = partial_r(t, x, rows)

        def boot(s):
            # 同じ写真が 2 回引かれたら別の写真として扱う
            parts = [rows[t["img"][rows] == i] for i in s]
            rr = np.concatenate(parts)
            new_img = np.concatenate([np.full(len(q), k) for k, q in enumerate(parts)])
            obs = t["obs"][rr]
            D = np.column_stack([np.eye(len(parts))[new_img], np.eye(obs.max() + 1)[obs]])
            uv_ = np.column_stack([t["rating"][rr], t[x][rr]])
            res = uv_ - D @ np.linalg.lstsq(D, uv_, rcond=None)[0]
            u, v = res[:, 0], res[:, 1]
            return u @ v / np.sqrt(u @ u * (v @ v))

        ci = boot_soaps(boot)
        perm = np.zeros(N_PERM)
        xv = t[x][rows].copy()
        groups_img = [np.where(t["img"][rows] == i)[0] for i in range(n)]
        for k in range(N_PERM):
            xp = xv.copy()
            for gi in groups_img:
                xp[gi] = xv[rng.permutation(gi)]
            b = Mfull @ xp
            perm[k] = a @ b / np.sqrt(a @ a * (b @ b))
        p = float((np.sum(np.abs(perm) >= abs(r)) + 1) / (N_PERM + 1))
        out[x] = {"partial_r": r, "ci": ci, "p_perm": p}
        print(f"   partial r with the rating, {x:9s} {r:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}], permutation p = {p:.3f}")
    # 同時に入れた回帰（標準化した残りで）
    X = np.column_stack([Mfull @ t[k][rows] for k in ("V", "lum", "coverage")])
    X = X / X.std(0)
    beta = np.linalg.lstsq(X, a / a.std(), rcond=None)[0]
    out["joint_beta_V_lum_coverage"] = [float(b) for b in beta]
    print(f"   joint (standardised): V {beta[0]:+.3f}, luminance at fixations {beta[1]:+.3f}, coverage {beta[2]:+.3f}")
    return out, t, rows, Mfull


summary["trial_level"], t5, rows5, M5 = analyse(main, "5 observers")
summary["trial_level_with_P003"] = analyse(people_all, "6 observers (P003 included)")[0]
with open(OUT / "gaze_readout_summary.json", "w") as f:
    json.dump(summary, f, indent=1)

# ---------------------------------------------------------------- 図 1: 情報量の上積み（輝度のモデルから）
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ms = [m for m in ig["all"] if m not in ("fixations", "typical", "+ luminance")]
for j, (per, col) in enumerate(zip(PERIODS, ("k", "C0", "C3"))):
    for k, m in enumerate(ms):
        e = ig[per][m]
        ax.errorbar(e["gain_over_luminance"], k + (j - 1) * 0.22, xerr=[[e["gain_over_luminance"] - e["ci"][0]],
                    [e["ci"][1] - e["gain_over_luminance"]]], fmt="o", color=col, ms=4, capsize=2,
                    label=per if k == 0 else None)
ax.axvline(0, color="gray", lw=0.8)
ax.set_yticks(range(len(ms)), [m.replace("+ luminance ", "") or m for m in ms], fontsize=8); ax.invert_yaxis()
ax.set_xlabel("bits per fixation over the typical layout + luminance (held-out soaps, 95% CI)")
ax.legend(fontsize=7, title="fixations", title_fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "gaze_readout_gain.png", dpi=120); plt.close(fig)

# ---------------------------------------------------------------- 図 2: 試行ごと（固定効果を除いた残り）
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
a = M5 @ t5["rating"][rows5]
for ax, (x, lab) in zip(axes, (("V", "readout of the photo, sharp only where this observer looked"),
                               ("lum", "luminance at this observer's fixations (SD)"))):
    b = M5 @ t5[x][rows5]
    ax.scatter(b, a, s=6, alpha=0.5)
    v = summary["trial_level"][x]
    ax.set_title(f"partial r = {v['partial_r']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]", fontsize=9)
    ax.set_xlabel(lab + "\nminus photo and observer means", fontsize=8)
axes[0].set_ylabel("rating minus photo and observer means", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "gaze_readout_trials.png", dpi=120); plt.close(fig)
