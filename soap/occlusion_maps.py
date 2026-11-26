"""遮蔽による読み出しの地図: 写真のどこをぼかすと、CLIP の埋め込みからの半透明感の読み出しが変わるか。

    HF_HOME=data/hf python soap/occlusion_maps.py      （先に soap/patch_features.py）

前回（soap/gaze_model.py）の地図は、CLIP の最終層のパッチトークンに、パッチを平均した特徴で学習した読み出しを
当てたもの。r = 0.92 を出した画像埋め込みの読み出しそのものの地図ではない。ここでは画像埋め込みの読み出しに
直接たずねる: 写真の一部をぼかしたとき、読み出しはどれだけ変わるか。

写真: 被験者が見たままの写真（背景あり）。Part 3 の bg と同じ手順（256 画素に縮め、224 画素で CLIP へ）。
読み出し: CLIP の画像埋め込み（768 次元、ここで fp32 で取り直す）からのリッジ回帰。地図を作る写真の石けんを
          除いて学習する（ratings.ridge。正則化の強さは学習側の中で選ぶ）。単位は評定（Part 3 と同じ）。
ぼかし: 256 画素の画像で σ = 8（元の画像で 32 画素）。ハイライト、縁、きめが消え、平均の色は残る。
視角: 1 画素 = 0.025°（画面の幅 51.2 cm / 1920 画素、距離 61 cm）。元の画像 1024 画素 ≈ 26°、ぼかし σ ≈ 0.8°。

出すもの（data/soap_occlusion.npz）:
  遮蔽地図 O      16 x 16 の格子（1 セル = 元の 64 画素 = CLIP のパッチ 1 個）。物体のセル（領域が半分以上）を
                 中心に半径 64 画素（1.6°）の円をぼかし、読み出しの変化 R(元) - R(ぼかし) を記録する。
                 正 = そこが読み出しを半透明の側へ押している。
  削除曲線        物体のセルを 10% ずつ順にぼかしていき、読み出しの変化 R(ぼかし) - R(元) を記録する。順番は
                 O の大きい順 / 前回のパッチ地図 S（CLIP）の大きい順 / 輝度の高い順 / 無作為（2 回）。
  見た所だけの読み出し V   参加者ごと・試行ごとに、注視点のまわり（半径 64 画素）だけくっきり残し、ほかを
                 ぼかした画像の読み出し。その人が見た情報だけからモデルが出す判断（soap/gaze_readout.py で使う）。
  前回のパッチ地図 S（CLIP 16 x 16、DINO 32 x 32）も soap/gaze_model.py と同じ手順で作り直して保存する。
注視: 提示開始から 0.1 秒より前に始まった注視は、注視点（+）から持ち越したもの（画像の中心から中央値 34 画素）
      なので捨てる。

実行前に決めたこと:
  妥当性  元の写真の読み出し（石けんを除いて学習）は Part 3 の r = 0.92 を再現するはず。
  問い 1  前回のパッチ地図は、画像埋め込みの読み出しが頼っている場所を指しているか。削除曲線の面積（10 段の
          変化の平均。低いほど、読み出しを半透明の側へ押している所を先に消している）を順番どうしで比べる。
          O の順が無作為より低いのは、O が 1 か所ずつぼかして作ったものなのでほぼ当然（まとめてぼかしたときにも
          効くかの確認にはなる）。
          判断: S の順と無作為の差の 95% 区間（石けんを単位に再標本化）が 0 を含めば、前回の地図は読み出しの
          根拠の場所を表していなかったとみなす。
          予測: S は無作為よりわずかに低いが、O と無作為の差の半分にも届かない。
  問い 2  遮蔽地図はどんな形か。
          予測 (a) 写真全体をぼかすと、多くの写真で読み出しが上がる（石けんどうしでは、コントラストの低いものほど
                   半透明と評定される。Part 3 で輝度の SD と評定の r = -0.59）。したがって O は負に寄る。
               (b) |O| は前回の地図より一部に集まる（上位 10% のセルに |O| の合計の 25% 以上。前回の地図は
                   物体の中の平均からのずれの絶対値で同じ指標を求めて比べる）。
               (c) |O| は輝度と正に相関する（ハイライトは明るく、ぼかすと消える。物体の中での相関の平均 > +0.2）。
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import LeaveOneGroupOut
from transformers import CLIPVisionModelWithProjection

from ratings import ROOT, load, ridge

IMAGES = Path("D:/project/vision-experiments/material_aoi_pilot/images")
FIXATIONS = Path("D:/project/vision-experiments/Tobii_pro/data/master_fixations.csv")
MASKS = ROOT / "data" / "soap_masks" / "final"
OUT = ROOT / "soap" / "figures"
SIZE, S256, G = 1024, 256, 16      # 元の画像、CLIP に渡す前の画像、地図の格子
CELL = S256 // G
RADIUS, FEATHER = 16, 4            # 円の半径と縁のぼかし（256 画素で。元の 64 画素と 16 画素）
BLUR = 8.0                         # ぼかしの σ（256 画素で）
MIN_ONSET = 0.1                    # これより前に始まった注視は持ち越しとして捨てる [秒]
FRACS = np.arange(1, 11) / 10
ORDERS = ["occlusion", "patch_clip", "luminance", "random1", "random2"]
BATCH = 32
device = "cuda"
rng = np.random.default_rng(0)

data, names, groups, Z, y = load()
ids = sorted(set(groups))
index = {g: np.where(groups == g)[0] for g in ids}
n = len(names)


def down(a, g):
    k = a.shape[0] // g
    return a.reshape(g, k, g, k).mean((1, 3))


def luminance(rgb):
    c = rgb / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722])


# ---------------------------------------------------------------- 画像、物体の領域、注視
masks = [np.asarray(Image.open(MASKS / (Path(nm).stem + ".png"))) > 0 for nm in names]
cellfrac = np.array([down(m.astype(np.float64), G) for m in masks])          # セルのうち物体の割合
objcell = cellfrac >= 0.5
mask256 = [down(m.astype(np.float64), S256) >= 0.5 for m in masks]
lum16 = np.array([down(luminance(np.asarray(Image.open(IMAGES / nm).convert("RGB"), np.float64)), G) for nm in names])
sharp = np.stack([np.asarray(Image.open(IMAGES / nm).convert("RGB").resize((S256, S256), Image.LANCZOS)) for nm in names])
sharp = sharp.astype(np.float32).transpose(0, 3, 1, 2) / 255.0                # (n, 3, 256, 256)
blur = np.stack([[ndimage.gaussian_filter(ch, BLUR, mode="reflect") for ch in im] for im in sharp]).astype(np.float32)

rows = list(csv.DictReader(open(FIXATIONS, encoding="utf-8-sig")))
people = sorted({r["participant_id"] for r in rows})
fix = {(p, nm): [] for p in people for nm in names}
n_early = 0
for r in rows:
    if not r["x_img_px"] or r["image_name"] not in names:
        continue
    if float(r["onset_sec"]) < MIN_ONSET:
        n_early += 1
        continue
    x, yy = float(r["x_img_px"]), float(r["y_img_px"])
    if 0 <= x < SIZE and 0 <= yy < SIZE:
        fix[(r["participant_id"], r["image_name"])].append((x, yy))
print(f"fixations: {len(rows)}; carried over from the fixation cross (onset < {MIN_ONSET} s), dropped: {n_early}")

yy_, xx_ = np.mgrid[0:S256, 0:S256] + 0.5


def disks(centers):
    """centers: (k, 2)（256 画素の座標） -> 円の和の重み (256, 256)。円の中は 1、縁で 0 へ落ちる。"""
    if len(centers) == 0:
        return np.zeros((S256, S256), np.float32)
    d = np.min([np.hypot(xx_ - cx, yy_ - cy) for cx, cy in centers], axis=0)
    return np.clip((RADIUS + FEATHER - d) / (2 * FEATHER), 0, 1).astype(np.float32)


def blocks(cells):
    """cells: 16 x 16 の真偽 -> そのセルの重み 1 (256, 256)。境目を少しぼかす。"""
    a = np.kron(cells.astype(np.float32), np.ones((CELL, CELL), np.float32))
    return np.clip(ndimage.gaussian_filter(a, 2.0), 0, 1)


# ---------------------------------------------------------------- CLIP と読み出し
repo = snapshot_download("bennyguo/zero123-xl-diffusers", local_files_only=True)
clip = CLIPVisionModelWithProjection.from_pretrained(repo, subfolder="image_encoder", variant="fp16",
                                                     dtype=torch.float32).to(device).eval()
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)


@torch.no_grad()
def embed(x):
    """x: (B, 3, 256, 256) の [0, 1] の配列 -> (B, 768)。"""
    out = []
    for k in range(0, len(x), BATCH):
        t = torch.from_numpy(np.ascontiguousarray(x[k:k + BATCH])).to(device)
        c = F.interpolate(t, size=224, mode="bicubic", antialias=True)
        out.append(clip((c - MEAN) / STD).image_embeds.float().cpu().numpy())
    return np.concatenate(out)


def composite(i, alphas):
    """alphas: (B, 256, 256)、1 = ぼかし版、0 = くっきり版。"""
    a = np.asarray(alphas, np.float32)[:, None]
    return sharp[i][None] * (1 - a) + blur[i][None] * a


E0 = embed(sharp)
saved = data["zero123_clip_bg"]
cos = np.sum(E0 * saved, 1) / np.linalg.norm(E0, axis=1) / np.linalg.norm(saved, axis=1)
w_of, b_of, R0 = {}, {}, np.zeros(n)
for tr, te in LeaveOneGroupOut().split(E0, y, groups):
    m = ridge().fit(E0[tr], y[tr])
    w = m[1].coef_ / m[0].scale_
    w_of[groups[te[0]]], b_of[groups[te[0]]] = w, m[1].intercept_ - w @ m[0].mean_
    R0[te] = E0[te] @ w + b_of[groups[te[0]]]
    assert np.allclose(R0[te], m.predict(E0[te]), atol=1e-4)
r0 = float(np.corrcoef(R0, y)[0, 1])
print(f"CLIP embeddings: cosine to the saved Part 3 features median {np.median(cos):.4f} (min {cos.min():.4f}); "
      f"readout of the original photos, held-out soaps: r = {r0:.3f}")


def readout(i, emb):
    return emb @ w_of[groups[i]] + b_of[groups[i]]


# ---------------------------------------------------------------- 前回のパッチ地図 S（soap/gaze_model.py と同じ）
pf = np.load(ROOT / "data" / "soap_patch_features.npz")
assert list(pf["names"]) == names
S = {}
for feat, g in (("clip", 16), ("dino", 32)):
    Fp = pf[feat].astype(np.float32)
    M = np.array([down(m.astype(np.float64), g) for m in masks])
    Xu = (np.einsum("nhwc,nhw->nc", Fp, M) / M.sum((1, 2))[:, None]).astype(np.float64)
    Sm = np.zeros((n, g, g))
    for tr, te in LeaveOneGroupOut().split(Xu, y, groups):
        m = ridge().fit(Xu[tr], y[tr])
        coef = m[1].coef_ / m[0].scale_
        Sm[te] = Fp[te] @ coef + (m[1].intercept_ - coef @ m[0].mean_)
    S[feat] = Sm
    del Fp

# ---------------------------------------------------------------- 遮蔽地図、削除曲線、見た所だけの読み出し
O = np.full((n, G, G), np.nan)
deletion = np.zeros((n, len(ORDERS), len(FRACS)))
R_blur = np.zeros(n)
V = np.full((len(people), n), np.nan)
coverage = np.zeros((len(people), n))
nfix = np.zeros((len(people), n), int)
centers = (np.arange(G) + 0.5) * CELL
for i in range(n):
    cells = np.argwhere(objcell[i])                                            # (k, 2) = (行, 列)
    emb = embed(composite(i, [disks([(centers[c], centers[r])]) for r, c in cells]))
    O[i][objcell[i]] = R0[i] - readout(i, emb)

    vals = {"occlusion": O[i][objcell[i]], "patch_clip": S["clip"][i][objcell[i]],
            "luminance": lum16[i][objcell[i]], "random1": rng.random(len(cells)), "random2": rng.random(len(cells))}
    alphas = []
    for o in ORDERS:
        rank = np.argsort(-vals[o])
        for f in FRACS:
            sel = np.zeros((G, G), bool)
            k = int(round(f * len(cells)))
            sel[cells[rank[:k], 0], cells[rank[:k], 1]] = True
            alphas.append(blocks(sel))
    deletion[i] = (readout(i, embed(composite(i, alphas))) - R0[i]).reshape(len(ORDERS), len(FRACS))

    alphas = [np.ones((S256, S256), np.float32)]                                # 全体をぼかしたもの
    for p_i, p in enumerate(people):
        f = np.array(fix[(p, names[i])]).reshape(-1, 2) * S256 / SIZE
        nfix[p_i, i] = len(f)
        a = 1 - disks(f)                                                       # 見た所だけくっきり
        coverage[p_i, i] = float((a[mask256[i]] < 0.5).mean())
        alphas.append(a)
    r = readout(i, embed(composite(i, alphas)))
    R_blur[i], V[:, i] = r[0], r[1:]
    if i % 10 == 0 or i == n - 1:
        print(f"{i + 1}/{n} {names[i]}: {len(cells)} object cells, R = {R0[i]:+.2f}, blurred {R_blur[i]:+.2f}, "
              f"O range [{np.nanmin(O[i]):+.3f}, {np.nanmax(O[i]):+.3f}]", flush=True)

np.savez(ROOT / "data" / "soap_occlusion.npz", names=np.array(names), people=np.array(people), O=O,
         S_clip=S["clip"], S_dino=S["dino"], cellfrac=cellfrac, lum16=lum16, R0=R0, R_blur=R_blur,
         deletion=deletion, orders=np.array(ORDERS), fracs=FRACS, V=V, coverage=coverage, nfix=nfix)

# ---------------------------------------------------------------- 問い 1: 削除曲線
auc = deletion.mean(2)                                                         # (n, 順番)
auc_rand = auc[:, 3:].mean(1)


def boot(fn, n_boot=2000):
    out = [fn(np.concatenate([index[g] for g in rng.choice(ids, len(ids))])) for _ in range(n_boot)]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


summary = {"readout_r": r0, "cosine_to_saved_median": float(np.median(cos)), "fixations_dropped_early": n_early,
           "auc": {o: float(auc[:, k].mean()) for k, o in enumerate(ORDERS)}}
for k, o in enumerate(ORDERS[:3]):
    d = auc[:, k] - auc_rand
    summary[f"{o}_minus_random"] = {"mean": float(d.mean()), "ci": boot(lambda s: d[s].mean()),
                                    "images_below_random": int((d < 0).sum())}
d_s, d_o = auc[:, 1] - auc_rand, auc[:, 0] - auc_rand
summary["patch_share_of_occlusion_gap"] = {"value": float(d_s.mean() / d_o.mean()),
                                           "ci": boot(lambda s: d_s[s].mean() / d_o[s].mean())}
print("\n(1) deletion curves: mean readout change over the 10 steps (lower = removes translucent-pushing parts first)")
for k, o in enumerate(ORDERS):
    print(f"    {o:11s} {auc[:, k].mean():+.3f}")
for o in ORDERS[:3]:
    v = summary[f"{o}_minus_random"]
    print(f"    {o} - random {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}], below random in "
          f"{v['images_below_random']}/{n}")
v = summary["patch_share_of_occlusion_gap"]
print(f"    patch map's gap to random as a share of the occlusion map's: {v['value']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]")
print(f"    all object cells blurred: readout change {deletion[:, 0, -1].mean():+.3f} "
      f"(same in every order: {np.allclose(deletion[:, :, -1], deletion[:, :1, -1], atol=1e-4)})")

# ---------------------------------------------------------------- 問い 2: 地図の形
up = R_blur - R0


def top_share(v):
    v = np.sort(np.abs(v))[::-1]
    return v[:max(1, int(round(0.1 * len(v))))].sum() / v.sum()


shape = {"whole_blur_change": float(up.mean()), "whole_blur_up_images": int((up > 0).sum()),
         "O_mean": float(np.nanmean(O)), "O_negative_share": float(np.mean(O[objcell] < 0)),
         "top10_share_absO": float(np.mean([top_share(O[i][objcell[i]]) for i in range(n)])),
         "top10_share_absS_clip": float(np.mean([top_share(S["clip"][i][objcell[i]] - S["clip"][i][objcell[i]].mean())
                                                 for i in range(n)])),
         "r_absO_luminance": float(np.mean([np.corrcoef(np.abs(O[i][objcell[i]]), lum16[i][objcell[i]])[0, 1] for i in range(n)])),
         "r_O_luminance": float(np.mean([np.corrcoef(O[i][objcell[i]], lum16[i][objcell[i]])[0, 1] for i in range(n)])),
         "r_O_S_clip": float(np.mean([np.corrcoef(O[i][objcell[i]], S["clip"][i][objcell[i]])[0, 1] for i in range(n)]))}
summary["shape"] = shape
print(f"\n(2) whole photo blurred: readout {shape['whole_blur_change']:+.3f} on average, up in {shape['whole_blur_up_images']}/{n}")
print(f"    O: mean {shape['O_mean']:+.4f}, negative in {shape['O_negative_share']:.0%} of object cells")
print(f"    share of the total in the top 10% of cells: |O| {shape['top10_share_absO']:.2f}, "
      f"|S - mean| (CLIP patches) {shape['top10_share_absS_clip']:.2f}")
print(f"    within-object correlation, mean over photos: |O| with luminance {shape['r_absO_luminance']:+.2f}, "
      f"O with luminance {shape['r_O_luminance']:+.2f}, O with the patch map S {shape['r_O_S_clip']:+.2f}")
with open(OUT / "occlusion_summary.json", "w") as f:
    json.dump(summary, f, indent=1)

# ---------------------------------------------------------------- 図: 削除曲線と地図の例
fig, ax = plt.subplots(figsize=(5, 3.4))
x = np.concatenate([[0], FRACS])
for k, (o, lab, col) in enumerate((("occlusion", "occlusion map O", "C3"), ("patch_clip", "patch map S (CLIP)", "C0"),
                                   ("luminance", "luminance", "C1"))):
    ax.plot(x, np.concatenate([[0], deletion[:, k].mean(0)]), "o-", color=col, ms=3, label=lab)
ax.plot(x, np.concatenate([[0], deletion[:, 3:].mean((0, 1))]), "o-", color="gray", ms=3, label="random")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("share of object cells blurred, in the order of each map"); ax.set_ylabel("readout change (rating units)")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "occlusion_deletion.png", dpi=120); plt.close(fig)

order = np.argsort(y)
pick = [order[-1], order[-2], order[1], order[0]]
lim = np.nanpercentile(np.abs(np.concatenate([O[i][objcell[i]] for i in pick])), 98)
fig, axes = plt.subplots(3, len(pick), figsize=(3 * len(pick), 9))
for c, i in enumerate(pick):
    im = np.asarray(Image.open(IMAGES / names[i]).convert("RGB"))
    f = np.concatenate([np.array(fix[(p, names[i])]).reshape(-1, 2) for p in people if p != "P003"])
    axes[0, c].imshow(im); axes[0, c].scatter(f[:, 0], f[:, 1], s=8, c="red", edgecolors="white", linewidths=0.3)
    axes[0, c].set_title(f"{names[i]}\nrating {y[i]:+.2f}, readout {R0[i]:+.2f}", fontsize=8)
    for r_, (mp, l_) in enumerate(((O[i], lim), (S["clip"][i] - S["clip"][i][objcell[i]].mean(), None)), start=1):
        l_ = l_ or np.percentile(np.abs(mp[objcell[i]]), 98)
        axes[r_, c].imshow(im, alpha=0.35)
        axes[r_, c].imshow(np.ma.masked_where(~objcell[i], mp), extent=(0, SIZE, SIZE, 0), cmap="coolwarm",
                           vmin=-l_, vmax=l_, alpha=0.85, interpolation="nearest")
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
axes[0, 0].set_ylabel("photo, fixations of 5 observers", fontsize=8)
axes[1, 0].set_ylabel(f"occlusion map O (one scale, +-{lim:.2f})\nred: blurring here lowers the readout", fontsize=8)
axes[2, 0].set_ylabel("patch map S (CLIP), minus its mean\n(scale per photo)", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "occlusion_examples.png", dpi=90); plt.close(fig)
