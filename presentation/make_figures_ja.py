"""説明用の図（日本語）。解析の結果と画像から作るだけで、新しい解析はしない。

    python presentation/make_figures_ja.py        # presentation/figures_ja/ に PNG を書く

石けんの写真と評定（研究室のデータ）を使うので、リポジトリの外の data/ と評定ファイルが必要。
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "image_to_3d"))
sys.path.insert(0, str(ROOT / "soap"))
from common import AZ, COEFS, CONDS, GEN, GT, SEEDS, UNSEEN, load_object, object_mask, stats, stim_path, tag  # noqa: E402
from ratings import cv_predict, load  # noqa: E402

OUT = ROOT / "presentation" / "figures_ja"
OUT.mkdir(parents=True, exist_ok=True)
FIG3, FIG_SOAP = ROOT / "image_to_3d" / "figures", ROOT / "soap" / "figures"
PHOTOS = Path("D:/project/vision-experiments/material_aoi_pilot/images")
MASKS = ROOT / "data" / "soap_masks" / "final"

for f in ("BIZ-UDGothicR.ttc", "BIZ-UDGothicB.ttc"):
    font_manager.fontManager.addfont(f"C:/Windows/Fonts/{f}")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
TRUTH, MODEL, REF = "#2a78d6", "#eb6834", "#8a8984"     # 正解（半透明）、生成モデル、対照・画像統計量
plt.rcParams.update({
    "font.family": "BIZ UDGothic", "font.size": 11, "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlesize": 12,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})

SHAPE = {"stim01": "形状A", "stim02": "形状B"}
LIGHT = {"studio": "スタジオ", "rustig_koppie": "屋外"}
DIR = {("studio", 0): "順光", ("studio", 180): "逆光", ("rustig_koppie", 315): "順光",
       ("rustig_koppie", 225): "側光", ("rustig_koppie", 135): "逆光"}
DENS = {"016": "濁り 少", "128": "濁り 中", "512": "濁り 多"}
AZ_JA = {0: "0度（入力と同じ視点）", 90: "90度", 180: "180度（真裏）", 270: "270度"}


def cond_ja(cond):
    return f"{SHAPE[cond[0]]}・{LIGHT[cond[1]]}・{DIR[(cond[1], cond[2])]}"


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


def blank(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


SHOW_BG = 0.5     # 見せる図の背景（灰色）。白背景だと半透明の物体がくすんで見えるため。モデルへの入力は白背景のまま


def gt_image(cond, coef, az, opaque=False, bg=1.0):
    name = f"gt_opaque_coef{coef}_az{az:03d}.png" if opaque else f"gt_coef{coef}_az{az:03d}.png"
    return load_object(GT / tag(cond) / name, 256, bg=bg, with_mask=True)


def on_gray(im, bg=SHOW_BG):
    """白背景の生成画像を灰色の背景に置き直す（表示用）。画像の縁につながる白い領域を背景とみなす。"""
    from scipy import ndimage
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    white = (a >= 250).all(-1)          # 生成画像の背景は 250 以上。明るい物体の縁を背景と取り違えないよう厳しめに
    lab, _ = ndimage.label(white)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    obj = ndimage.binary_erosion(~np.isin(lab, border[border > 0]), iterations=2)   # 縁に残る白い輪を落とす
    m = ndimage.gaussian_filter(obj.astype(np.float32), 0.8)[..., None]
    return Image.fromarray((a * m + bg * 255 * (1 - m)).round().astype(np.uint8))


def gen_image(cond, name):
    im = Image.open(GEN / tag(cond) / name).convert("RGB")
    return im, object_mask(im)


def fill_frame(im, ratio=0.85):
    """表示用。物体が枠の ratio を占めるように置き直す（TripoSR の描画は物体が小さいので）。"""
    from scipy import ndimage
    a = np.asarray(im.convert("RGB")).copy()
    m = object_mask(im)
    a[~ndimage.binary_dilation(m, iterations=5)] = 255      # 真っ白でない背景を白にそろえる
    ys, xs = np.where(m)
    y0, y1, x0, x1 = max(ys.min() - 3, 0), ys.max() + 4, max(xs.min() - 3, 0), xs.max() + 4
    side = int(max(y1 - y0, x1 - x0) / ratio)
    canvas = np.full((side, side, 3), 255, np.uint8)
    oy, ox = (side - (y1 - y0)) // 2, (side - (x1 - x0)) // 2
    canvas[oy:oy + y1 - y0, ox:ox + x1 - x0] = a[y0:y1, x0:x1]
    return Image.fromarray(canvas).resize((256, 256), Image.LANCZOS)


# ------------------------------------------------------------------ 図 1: Part 2 の刺激
def fig_stimuli():
    fig, axes = plt.subplots(len(CONDS), len(COEFS), figsize=(6.2, 12))
    for r, cond in enumerate(CONDS):
        for c, coef in enumerate(COEFS):
            ax = axes[r, c]
            ax.imshow(load_object(stim_path(cond, coef), 256, bg=SHOW_BG)); blank(ax)
            if r == 0:
                ax.set_title(f"{DENS[coef]}（{int(coef)}）", fontsize=11)
            if c == 0:
                ax.set_ylabel(cond_ja(cond), fontsize=10)
    fig.suptitle("刺激：半透明物体の入力画像（6条件 × 濁り3段階 = 18条件）", fontsize=13, y=1.0)
    fig.tight_layout()
    save(fig, "01_刺激_入力画像.png")


# ------------------------------------------------------------------ 図 2: 石けんの写真と評定
def soap_crop(name, size=320):
    im = Image.open(PHOTOS / name).convert("RGB")
    m = np.asarray(Image.open(MASKS / (Path(name).stem + ".png")).resize(im.size)) > 0
    ys, xs = np.where(m)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = min(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.65, im.size[0] / 2, im.size[1] / 2)
    cx, cy = np.clip(cx, half, im.size[0] - half), np.clip(cy, half, im.size[1] - half)   # 写真の外に出ない
    box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    return im.crop(box).resize((size, size), Image.LANCZOS)


def fig_soaps():
    data, names, groups, Z, y = load()
    order = np.argsort(y)
    low, high = [], []
    for i in order:                      # 石けんが重ならないように選ぶ
        if groups[i] not in {groups[j] for j in low} and len(low) < 4:
            low.append(i)
    for i in order[::-1]:
        if groups[i] not in {groups[j] for j in high} and len(high) < 4:
            high.append(i)
    ids = sorted(set(groups))
    pairs = sorted(ids, key=lambda g: -abs(np.subtract(*y[groups == g])))[:2]
    pair_idx = [i for g in pairs for i in sorted(np.where(groups == g)[0], key=lambda k: y[k])]
    fig, axes = plt.subplots(3, 4, figsize=(10, 8.4))
    rows = (("半透明感が低いと評定された写真", low), ("半透明感が高いと評定された写真", high),
            ("同じ石けん・照明だけが違う2枚（評定の差が大きい2組）", pair_idx))
    for r, (title, idx) in enumerate(rows):
        for c, i in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(soap_crop(names[i])); blank(ax)
            ax.set_xlabel((f"組{c // 2 + 1}：" if r == 2 else "") + f"評定 {y[i]:+.2f}", fontsize=11)
        axes[r, 0].set_title(title, loc="left", fontsize=12, pad=8)
    fig.suptitle("石けんの写真（61個 × 照明2条件 = 122枚）と半透明感の評定（6人、参加者ごとに標準化した平均）",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    save(fig, "02_刺激_石けんの写真と評定.png")


# ------------------------------------------------------------------ 図 3: 正解と生成の見た目
def fig_views(cond, coef, name):
    rows = (("正解（半透明）", lambda az: gt_image(cond, coef, az, bg=SHOW_BG)[0]),
            ("不透明の対照", lambda az: gt_image(cond, coef, az, opaque=True, bg=SHOW_BG)[0]),
            ("Zero-1-to-3 の生成", lambda az: on_gray(gen_image(cond, f"zero123_coef{coef}_az{az:03d}_seed0.png")[0])),
            ("TripoSR の生成", lambda az: on_gray(fill_frame(gen_image(cond, f"triposr_coef{coef}_az{az:03d}.png")[0]))))
    fig, axes = plt.subplots(len(rows), len(AZ), figsize=(9.6, 10.2))
    for r, (label, get) in enumerate(rows):
        for c, az in enumerate(AZ):
            ax = axes[r, c]
            ax.imshow(get(az)); blank(ax)
            if r == 0:
                ax.set_title(AZ_JA[az], fontsize=11)
            if c == 0:
                ax.set_ylabel(label, fontsize=12)
    fig.suptitle(f"{cond_ja(cond)}・{DENS[coef]}：入力は左上の1枚だけ。見えていない側をモデルが描く", fontsize=12, y=1.0)
    fig.tight_layout()
    save(fig, name)


# ------------------------------------------------------------------ 図 4: 裏側の輝度とコントラスト
def far_stats():
    """条件ごとの、見えていない3視点の (平均輝度, コントラスト)。Zero-1-to-3 は乱数3回の平均と標準偏差。"""
    keys = [(cond, coef) for cond in CONDS for coef in COEFS]
    truth = np.array([np.mean([stats(*gt_image(c, k, az)) for az in UNSEEN], 0) for c, k in keys])
    opaque = np.array([np.mean([stats(*gt_image(c, k, az, opaque=True)) for az in UNSEEN], 0) for c, k in keys])
    per_seed = np.array([[np.mean([stats(*gen_image(c, f"zero123_coef{k}_az{az:03d}_seed{s}.png")) for az in UNSEEN], 0)
                          for s in SEEDS] for c, k in keys])
    return truth, opaque, per_seed.mean(1), per_seed.std(1, ddof=1)


def fig_far_stats(truth, gen, sd):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for j, (ax, m) in enumerate(zip(axes, ("平均輝度", "コントラスト（標準偏差 / 平均）"))):
        lim = max(truth[:, j].max(), (gen[:, j] + sd[:, j]).max()) * 1.1
        ax.plot([0, lim], [0, lim], ls=":", color=INK2, lw=1)
        ax.text(lim * 0.97, lim * 0.93, "正解と同じ", ha="right", va="top", fontsize=9, color=INK2, rotation=45,
                rotation_mode="anchor")
        ax.errorbar(truth[:, j], gen[:, j], yerr=sd[:, j], fmt="o", ms=7, color=MODEL, ecolor=MODEL, elinewidth=1,
                    capsize=0, mec="white", mew=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        ax.set_xlabel(f"正解（半透明）の裏側の{m.split('（')[0]}")
        ax.set_ylabel(f"Zero-1-to-3 の裏側の{m.split('（')[0]}")
        ax.set_title(m, loc="left")
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    n_dark = int((gen[:, 0] < truth[:, 0]).sum()); n_con = int((gen[:, 1] > truth[:, 1]).sum())
    axes[0].text(0.03, 0.97, f"18条件中 {n_dark} 条件で\n正解より暗い", transform=axes[0].transAxes, va="top", fontsize=11)
    axes[1].text(0.03, 0.97, f"18条件中 {n_con} 条件で\n正解よりコントラストが高い", transform=axes[1].transAxes, va="top",
                 fontsize=11)
    fig.suptitle("見えていない側（90・180・270度の平均）：生成は正解より暗く、陰影が強い\n"
                 "点 = 1条件（乱数3回の平均）、縦線 = 乱数による標準偏差", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "04_裏側の輝度とコントラスト.png")


# ------------------------------------------------------------------ 図 5: 評定の予測
def dot_rows(ax, rows, ceiling=None, chance=None, xlim=(0, 1)):
    """rows: (ラベル, r, (下限, 上限), 色) を上から順に。"""
    n = len(rows)
    if chance is not None:
        ax.axvspan(xlim[0], chance, color=GRID, lw=0)
        ax.text(chance, n - 0.35, " 偶然でも\n 出る範囲", fontsize=9, color=INK2, va="top")
    for k, (label, r, ci, color) in enumerate(rows):
        yk = n - 1 - k
        ax.plot(ci, [yk, yk], color=color, lw=2, solid_capstyle="round")
        ax.plot(r, yk, "o", ms=9, color=color, mec="white", mew=1.5)
        ax.text(ci[0] - 0.015, yk, f"{r:.2f}", va="center", ha="right", fontsize=11)
    if ceiling is not None:
        ax.axvline(ceiling, color=INK2, ls=":", lw=1)
        ax.text(ceiling, n - 0.35, "評定者どうしの\n一致による上限 ", fontsize=9, color=INK2, ha="right", va="top")
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_ylim(-0.6, n - 0.2); ax.set_xlim(*xlim)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)


def fig_predict():
    s = json.load(open(FIG_SOAP / "predict_summary.json"))
    R = s["results"]
    unet = max((k for k in R if k.startswith("zero123_unet")), key=lambda k: R[k]["r"])
    rows = [("平均色", "color", REF), ("画像統計量 14種", "stats14", REF),
            ("Zero-1-to-3 の画像埋め込み（CLIP）", "zero123_clip_nobg", TRUTH),
            ("Zero-1-to-3 の U-Net 内部", unet, TRUTH), ("TripoSR の画像特徴（DINO）", "triposr_dino_bg", TRUTH)]
    fig, ax = plt.subplots(figsize=(9, 4))
    dot_rows(ax, [(lab, R[k]["r"], R[k]["ci"], c) for lab, k, c in rows], ceiling=float(np.sqrt(s["reliability"])),
             chance=max(R[k]["null_95"] for _, k, _ in rows))
    ax.set_xlabel("予測と評定の相関 r（石けんごとに学習から外して評価。線は95%信頼区間）")
    ax.set_title("半透明感の評定を、モデルの特徴から線形に読み出せる（122枚、61個の石けん）", loc="left")
    fig.tight_layout()
    save(fig, "05_評定の予測精度.png")


# ------------------------------------------------------------------ 図 6: 同じ石けんの2枚
def fig_within():
    data, names, groups, Z, y = load()
    ids = sorted(set(groups))
    P = np.array([[i for i in range(len(names)) if groups[i] == g] for g in ids])
    d = y[P[:, 0]] - y[P[:, 1]]
    feats = {"stats14+color": np.concatenate([data["stats14"], data["color"]], 1).astype(np.float64),
             "zero123_clip_nobg": data["zero123_clip_nobg"].astype(np.float64)}
    s = json.load(open(FIG_SOAP / "within_pair_summary.json"))
    R = s["results"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    sgn = np.sign(d)
    for k, color, lab in (("stats14+color", REF, "画像統計量 + 色"), ("zero123_clip_nobg", TRUTH, "CLIP の読み出し")):
        pred = cv_predict(feats[k], y, groups)
        p = (pred[P[:, 0]] - pred[P[:, 1]]) * sgn
        ax.scatter(p, d * sgn, s=28, color=color, edgecolor="white", lw=0.6,
                   label=f"{lab}（r = {R[k]['method1']['r']:.2f}）")
    lim = 1.45
    ax.axvline(0, color=INK2, lw=0.8)
    ax.text(lim * 0.97, 0.03, "向きが合う →", ha="right", fontsize=9, color=INK2)
    ax.text(-lim * 0.97, 0.03, "← 逆向き", ha="left", fontsize=9, color=INK2)
    ax.set_xlim(-lim, lim); ax.set_ylim(0, 1.4)
    ax.set_xlabel("予測された差（評定が高いほうの写真 − もう一方）"); ax.set_ylabel("評定の差")
    ax.legend(loc="upper left", fontsize=10, frameon=True, framealpha=1, edgecolor="white")
    ax.set_title("1点 = 1個の石けん（照明だけが違う2枚）", loc="left", fontsize=11)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    unet = "zero123_unet_t900_nobg"
    rows = [("画像統計量 + 色", "stats14+color", REF), ("Zero-1-to-3 の画像埋め込み（CLIP）", "zero123_clip_nobg", TRUTH),
            ("Zero-1-to-3 の U-Net 内部", unet, TRUTH), ("TripoSR の画像特徴（DINO）", "triposr_dino_bg", TRUTH)]
    dot_rows(axes[1], [(lab, R[k]["method1"]["r"], R[k]["method1"]["ci"], c) for lab, k, c in rows],
             ceiling=s["ceiling_r"], xlim=(-0.3, 1))
    axes[1].axvline(0, color=INK2, lw=0.8)
    axes[1].set_xlabel("評定の差との相関 r（石けんの種類の違いは使えない。線は95%信頼区間）")
    axes[1].set_title("石けんの間で学習した読み出しを、同じ石けんの2枚に当てる", loc="left", fontsize=11)
    fig.suptitle("「石けんの種類を見分けているだけ」ではない：同じ石けんの照明による違いも追える", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "06_同じ石けんの2枚.png")


# ------------------------------------------------------------------ 図 7: 読み出しで測った裏側
def fig_readout():
    sc = {k: np.array(v) for k, v in json.load(open(FIG3 / "readout_summary.json"))["clip"]["scores"].items()}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    items = [("入力画像\n（正面・正解）", "input_translucent", TRUTH), ("正解の裏側\n（半透明）", "far_translucent", TRUTH),
             ("不透明の対照\nの裏側", "far_opaque", REF), ("Zero-1-to-3\n0度（正面）", "zero123_az0", MODEL),
             ("Zero-1-to-3\nの裏側", "zero123_far", MODEL)]
    for k, (lab, key, color) in enumerate(items):
        v = sc[key]
        ax.scatter(np.full(len(v), k) + np.linspace(-0.18, 0.18, len(v)), v, s=12, color=color, alpha=0.35, lw=0)
        se = v.std(ddof=1) / np.sqrt(len(v))
        ax.errorbar(k, v.mean(), yerr=1.96 * se, fmt="o", ms=10, color=color, mec="white", mew=1.5, elinewidth=2,
                    capsize=0)
        ax.text(k + 0.25, v.mean(), f"{v.mean():+.2f}", va="center", fontsize=10)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([i[0] for i in items], fontsize=10)
    ax.set_ylabel("読み出し値（半透明感の推定。石けんの評定の単位）")
    ax.set_title("平均（大きい点、95%信頼区間）と各条件（小さい点）", loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax = axes[1]
    x = sc["input_translucent"]
    lo, hi = -0.75, 0.6
    ax.plot([lo, hi], [lo, hi], ls=":", color=INK2, lw=1)
    ax.text(hi * 0.97, hi * 0.9, "正面と同じ", ha="right", fontsize=9, color=INK2)
    for key, color, lab, mk in (("far_translucent", TRUTH, "正解の裏側（半透明）", "o"),
                                ("far_opaque", REF, "不透明の対照の裏側", "s"),
                                ("zero123_far", MODEL, "Zero-1-to-3 の裏側", "^")):
        ax.scatter(x, sc[key], s=46, color=color, marker=mk, edgecolor="white", lw=0.8, label=lab)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("入力画像（正面）の読み出し値"); ax.set_ylabel("裏側の読み出し値")
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.set_title("1点 = 1条件", loc="left", fontsize=11)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.suptitle("石けんで学習した読み出しを「半透明感の物差し」として当てる：\n"
                 "本物は裏側のほうが半透明寄りの値になるが、Zero-1-to-3 は正面の値を裏側にそのまま写している", fontsize=12, y=1.04)
    fig.tight_layout()
    save(fig, "07_読み出しで測った裏側.png")


# ------------------------------------------------------------------ 図 8: 埋め込みを押す実験
def fig_steer_curves(truth_far):
    s = json.load(open(FIG3 / "steer_summary.json"))
    b = s["by_alpha"]
    sc = {k: np.array(v) for k, v in json.load(open(FIG3 / "readout_summary.json"))["clip"]["scores"].items()}
    sd = {k: np.array(v) for k, v in json.load(open(FIG3 / "readout_summary.json"))["dino"]["scores"].items()}
    refs = {"luminance": truth_far[:, 0].mean(), "contrast": truth_far[:, 1].mean(),
            "clip_readout": sc["far_translucent"].mean(), "dino_readout": sd["far_translucent"].mean()}
    M = (("luminance", "平均輝度"), ("contrast", "コントラスト"), ("clip_readout", "読み出し（CLIP）"),
         ("dino_readout", "読み出し（DINO）"))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
    steps = (-2, -1, 0, 1, 2)
    for ax, (m, lab) in zip(axes, M):
        for k, c in enumerate(("control1", "control2")):
            ax.plot([-2, 2], [b[f"{c}_{a:+d}"]["far"][m] for a in (-2, 2)], "-s", color=REF, lw=1.5, ms=6,
                    label="対照の方向（同じ長さ、ランダム）" if k == 0 else None)
        ax.plot(steps, [b[f"translucent_{a:+d}"]["far"][m] for a in steps], "-o", color=MODEL, lw=2.2, ms=8,
                mec="white", mew=1.2, label="「半透明」の方向")
        ax.axhline(refs[m], color=TRUTH, ls="--", lw=1.5, label="正解（半透明）の裏側")
        ax.set_title(lab, loc="left")
        ax.set_xticks(steps); ax.set_xlabel("押した量（評定の単位）")
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].legend(loc="center left", fontsize=9, frameon=False, bbox_to_anchor=(0, 0.62))
    tr = s["transmission"]
    fig.text(0.5, -0.06, f"押した量のうち出力に届いたのは約 {tr['translucent_far']['mean'] * 100:.0f}%"
             f"（対照の方向でも {tr['control1_far']['mean'] * 100:.0f}%・{tr['control2_far']['mean'] * 100:.0f}%）。"
             "向きは半透明側にそろう（18条件中 16〜17）が、変化は小さい", ha="center", fontsize=12)
    fig.suptitle("Zero-1-to-3 の画像埋め込みを「半透明」の方向に押したときの、生成された裏側（18条件の平均）\n"
                 "押す幅 −2〜+2 は、石けん122枚が埋め込み上で広がる範囲の端から端に相当", fontsize=12, y=1.06)
    fig.tight_layout()
    save(fig, "08_埋め込みを押す実験.png")


def fig_steer_images(cond, coef):
    from steer_embedding import fname
    rows = [("正解（半透明）", lambda az: gt_image(cond, coef, az, bg=SHOW_BG)[0])]
    for a in (-2, -1, 0, 1, 2):
        lab = {-2: "押す量 −2（不透明側）", 0: "押す量 0（元の生成）", 2: "押す量 +2（半透明側）"}.get(a, f"押す量 {a:+d}")
        rows.append((lab, lambda az, a=a: on_gray(Image.open(GEN / tag(cond) / fname("translucent", a, coef, az)))))
    fig, axes = plt.subplots(len(rows), len(AZ), figsize=(9.2, 14))
    for r, (label, get) in enumerate(rows):
        for c, az in enumerate(AZ):
            ax = axes[r, c]
            ax.imshow(get(az)); blank(ax)
            if r == 0:
                ax.set_title(AZ_JA[az], fontsize=11)
            if c == 0:
                ax.set_ylabel(label, fontsize=11)
    fig.suptitle(f"{cond_ja(cond)}・{DENS[coef]}：埋め込みを押しても、裏側は不透明な物体のような濃い陰影のまま", fontsize=12, y=1.0)
    fig.tight_layout()
    save(fig, "09_埋め込みを押したときの見た目.png")


if __name__ == "__main__":
    fig_stimuli()
    fig_soaps()
    fig_views(("stim01", "studio", 0), "128", "03a_正解と生成_順光.png")
    fig_views(("stim01", "studio", 180), "128", "03b_正解と生成_逆光.png")
    truth, opaque, gen, sd = far_stats()
    fig_far_stats(truth, gen, sd)
    fig_predict()
    fig_within()
    fig_readout()
    fig_steer_curves(truth)
    fig_steer_images(("stim01", "studio", 0), "128")
