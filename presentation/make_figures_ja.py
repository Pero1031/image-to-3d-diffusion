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

for f in ("meiryo.ttc", "meiryob.ttc"):       # メイリオ: 画面・投影で読みやすく、数字の幅が詰まる（"0.4" が "0. 4" にならない）
    font_manager.fontManager.addfont(f"C:/Windows/Fonts/{f}")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
TRUTH, MODEL, REF = "#2a78d6", "#eb6834", "#8a8984"     # 正解（半透明）、生成モデル、対照・画像統計量
plt.rcParams.update({
    "font.family": "Meiryo", "font.size": 13, "axes.unicode_minus": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "axes.linewidth": 1.0, "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "text.color": INK, "axes.titlesize": 14, "axes.labelsize": 13,
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
            ax.set_xlabel((f"組{c // 2 + 1}：" if r == 2 else "") + f"評定 {num(y[i])}", fontsize=11)
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


# ------------------------------------------------------------------ グラフの共通部品
# 発表用のグラフの決まり（Rougier et al. 2014 "Ten simple rules for better figures"、Wilke "Fundamentals of Data
# Visualization" の 16・20・22 章による）:
#   - 1枚の図で言うことは1つ。タイトルはその結論を書く。読み方の説明や数値の詳細は図の外の説明文に回す
#   - 信頼区間などの誤差の表示は、主張がそれに頼るときだけ描く。描かない場合も説明文に数値を書く
#   - 凡例より直接ラベル。縦書きの軸ラベルは避け、軸の左上に横書きで置く
#   - 投影で読めるよう、文字・点・線を大きく太く。強調したい系列だけに色を使い、残りは灰色
CAPTIONS = {}


def caption(name, text):
    CAPTIONS[name] = text


def ytitle(ax, text):
    """縦書きの軸ラベルの代わりに、軸の左上に横書きで置く。"""
    ax.text(0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=12, color=INK2)


def grid(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def title(fig, text, sub=None):
    """図の結論を1行で左上に。sub は読み方の補足（灰色の小さい字）。"""
    h = fig.get_size_inches()[1]
    fig.suptitle(text, x=0.01, y=1.0 if sub is None else 0.985 + 0.7 / h, ha="left", fontsize=16)
    if sub:
        fig.text(0.01, 0.985, sub, ha="left", va="bottom", fontsize=12, color=INK2)


def num(v, digits=2):
    """符号つきの数。マイナスは軸の目盛りと同じ記号（−）にそろえる。"""
    return f"{v:+.{digits}f}".replace("-", "\u2212")


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
    name = "04_裏側の輝度とコントラスト.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    n_dark = int((gen[:, 0] < truth[:, 0]).sum())
    n_con = int((gen[:, 1] > truth[:, 1]).sum())
    panels = ((0, "平均輝度", (0.15, 0.65), "below", f"18条件中 {n_dark} 条件で\n生成のほうが暗い"),
              (1, "コントラスト（標準偏差 ÷ 平均）", (0.2, 1.2), "above", f"18条件中 {n_con} 条件で\n生成のほうが陰影が強い"))
    for ax, (j, head, (lo, hi), side, note) in zip(axes, panels):
        # 主張の側（生成が暗い／陰影が強い）を薄く塗り、そこに数を書く
        ax.fill_between([lo, hi], [lo, hi], lo if side == "below" else hi, color="#f3f2ef", lw=0)
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1.2)
        ax.scatter(truth[:, j], gen[:, j], s=80, color=MODEL, edgecolor="white", lw=1.5, zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        w = hi - lo
        if side == "below":
            ax.text(hi - 0.03 * w, lo + 0.03 * w, note, ha="right", va="bottom", fontsize=13)
        else:
            ax.text(lo + 0.03 * w, hi - 0.03 * w, note, ha="left", va="top", fontsize=13)
        ax.text(lo + 0.86 * w, lo + 0.86 * w, "正解と同じ", ha="center", va="top", fontsize=11, color=INK2,
                rotation=45, rotation_mode="anchor", transform_rotates_text=True)
        ax.set_title(head, loc="left", pad=28)
        ax.set_xlabel("正解（半透明）の裏側")
        ytitle(ax, "Zero-1-to-3 が描いた裏側")
        grid(ax)
    title(fig, "生成された裏側は、正解より暗く、陰影が強い")
    fig.tight_layout()
    save(fig, name)
    gap = np.abs(gen - truth).mean(0)
    caption(name, "点 = 1条件（形状2 × 照明と向き3 × 濁り3段階 = 18条件）。値は見えていない3視点（90・180・270度）の平均で、"
                  "Zero-1-to-3 は乱数3回の平均。斜めの線より下は生成のほうが暗い、上は生成のほうがコントラストが高い。"
                  f"乱数による標準偏差は平均で 輝度 {sd[:, 0].mean():.3f}・コントラスト {sd[:, 1].mean():.3f} で、"
                  f"正解とのずれ（平均 {gap[0]:.3f}・{gap[1]:.3f}）より小さいので、図では省いた。")


# ------------------------------------------------------------------ 図 5: 評定の予測
def fig_predict():
    name = "05_評定の予測精度.png"
    s = json.load(open(FIG_SOAP / "predict_summary.json"))
    R = s["results"]
    unet = max((k for k in R if k.startswith("zero123_unet")), key=lambda k: R[k]["r"])
    rows = [("平均色", "color", REF), ("画像統計量 14種", "stats14", REF),
            ("Zero-1-to-3 の画像埋め込み（CLIP）", "zero123_clip_nobg", TRUTH),
            ("Zero-1-to-3 の U-Net 内部", unet, TRUTH), ("TripoSR の画像特徴（DINO）", "triposr_dino_bg", TRUTH)]
    ceiling = float(np.sqrt(s["reliability"]))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ys = [5.4, 4.4, 2.6, 1.6, 0.6]                       # 画像統計量と生成モデルの間を少しあける
    for yk, (lab, key, color) in zip(ys, rows):
        r = R[key]["r"]
        ax.barh(yk, r, height=0.6, color=color, lw=0)
        ax.text(r + 0.012, yk, f"{r:.2f}", va="center", fontsize=14,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=4)
    ax.axvline(ceiling, color=INK2, lw=1.2, ls=(0, (2, 2)), zorder=3)
    ax.text(ceiling, 6.05, f"評定者どうしの一致から見た上限 {ceiling:.2f}", ha="right", va="bottom", fontsize=11, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=13)
    ax.set_ylim(0, 6.4); ax.set_xlim(0, 1.06); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("予測と評定の相関 r")
    grid(ax, "x")
    title(fig, "半透明感の評定は、生成モデルの特徴から高い精度で読み出せる")
    fig.tight_layout()
    save(fig, name)
    ci = "、".join(f"{lab} [{R[k]['ci'][0]:.2f}, {R[k]['ci'][1]:.2f}]" for lab, k, _ in rows)
    lo, hi = s["best_minus_stats_ci"]
    caption(name, "石けん61個 × 照明2条件 = 122枚。評定は6人の平均（参加者ごとに標準化）。特徴から評定をリッジ回帰で予測し、"
                  "石けん1個ずつ（同じ石けんの2枚は一緒に）学習から外して評価した。"
                  f"95%信頼区間：{ci}。CLIP と画像統計量の差は {R['zero123_clip_nobg']['r'] - R['stats14']['r']:.2f}"
                  f"（95%信頼区間 [{lo:.2f}, {hi:.2f}]）。評定をシャッフルしたときの r の95%点は "
                  f"{min(R[k]['null_95'] for _, k, _ in rows):.2f}〜{max(R[k]['null_95'] for _, k, _ in rows):.2f}。")


# ------------------------------------------------------------------ 図 6: 同じ石けんの2枚
def fig_within():
    name = "06_同じ石けんの2枚.png"
    data, names, groups, Z, y = load()
    ids = sorted(set(groups))
    P = np.array([[i for i in range(len(names)) if groups[i] == g] for g in ids])
    d = y[P[:, 0]] - y[P[:, 1]]
    large = np.abs(d) > np.median(np.abs(d))             # 評定の差が大きい組（within_pair.py と同じ分け方）
    s = json.load(open(FIG_SOAP / "within_pair_summary.json"))
    R = s["results"]
    feats = {"stats14+color": np.concatenate([data["stats14"], data["color"]], 1).astype(np.float64),
             "zero123_clip_nobg": data["zero123_clip_nobg"].astype(np.float64)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    sgn = np.sign(d)                                    # 評定が高いほうの写真を先にそろえる（右側 = 向きが合う）
    lim, top = 1.45, 1.42
    for ax, (k, color, lab) in zip(axes, (("stats14+color", REF, "画像統計量 + 色"), ("zero123_clip_nobg", TRUTH, "CLIP の読み出し"))):
        pred = cv_predict(feats[k], y, groups)
        p = (pred[P[:, 0]] - pred[P[:, 1]]) * sgn
        a = np.abs(d)
        ax.axhspan(0, np.median(a), color="#f3f2ef", lw=0)
        ax.axvline(0, color=INK2, lw=1.2)
        ax.scatter(p[large], a[large], s=70, color=color, edgecolor="white", lw=1.2, zorder=3)
        ax.scatter(p[~large], a[~large], s=40, color=color, alpha=0.35, lw=0, zorder=3)
        n_ok = int((p[large] > 0).sum())
        ax.set_title(f"{lab}：{large.sum()}組中 {n_ok}組で向きが合う", loc="left", pad=28)
        ax.text(lim * 0.96, top * 0.97, "向きが合う →", ha="right", va="top", fontsize=12, color=INK2)
        ax.text(-lim * 0.96, top * 0.97, "← 逆向き", ha="left", va="top", fontsize=12, color=INK2)
        ax.set_xlim(-lim, lim); ax.set_ylim(0, top)
        ax.set_xlabel("予測された差（評定が高いほうの写真 − もう一方）")
        grid(ax)
    ytitle(axes[0], "評定の差")
    title(fig, "照明だけが違う同じ石けんの2枚の差も、CLIP の読み出しは追える",
          f"濃い点：評定の差が大きい組（{large.sum()}組）　薄い点と灰色の帯：差が小さく、評定者の間でも向きがそろわない組（{(~large).sum()}組）")
    fig.tight_layout()
    save(fig, name)
    m = {k: R[k]["method1"] for k in ("stats14+color", "zero123_clip_nobg", "zero123_unet_t900_nobg", "triposr_dino_bg")}
    caption(name, f"点 = 1個の石けん（照明だけが違う2枚、{len(P)}組）。読み出しは石けんの間で学習したもの（その石けんは学習から外す）を"
                  "そのまま2枚に当てた。濃い点は評定の差が中央値より大きい組（評定者を3人ずつに分けると、9割の組で向きがそろう）、"
                  f"薄い点と灰色の帯は差が小さい組。原点まわりの相関 r：画像統計量 + 色 {m['stats14+color']['r']:.2f}"
                  f" [{m['stats14+color']['ci'][0]:.2f}, {m['stats14+color']['ci'][1]:.2f}]、"
                  f"CLIP {m['zero123_clip_nobg']['r']:.2f} [{m['zero123_clip_nobg']['ci'][0]:.2f}, {m['zero123_clip_nobg']['ci'][1]:.2f}]"
                  f"（95%信頼区間）。ほかのモデルでも同様：U-Net 内部 r = {m['zero123_unet_t900_nobg']['r']:.2f}"
                  f"（{m['zero123_unet_t900_nobg']['same_sign_large']}組）、DINO r = {m['triposr_dino_bg']['r']:.2f}"
                  f"（{m['triposr_dino_bg']['same_sign_large']}組）。評定者どうしの一致から見た上限 r = {s['ceiling_r']:.2f}。")


# ------------------------------------------------------------------ 図 7: 読み出しで測った裏側
def fig_readout():
    name = "07_読み出しで測った裏側.png"
    sc = {k: np.array(v) for k, v in json.load(open(FIG3 / "readout_summary.json"))["clip"]["scores"].items()}
    x = sc["input_translucent"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.8), sharey=True)
    for ax, key, color, lab in ((axes[0], "far_translucent", TRUTH, "正解（本物の半透明物体）"),
                                (axes[1], "zero123_far", MODEL, "Zero-1-to-3 の生成")):
        v = sc[key]
        for a, b in zip(x, v):                           # 細い線 = 1条件、太い線 = 18条件の平均
            ax.plot([0, 1], [a, b], color=color, alpha=0.3, lw=1.2, solid_capstyle="round")
        ax.plot([0, 1], [x.mean(), v.mean()], "-o", color=color, lw=4, ms=11, mec="white", mew=1.8, zorder=3)
        ax.text(-0.08, x.mean(), num(x.mean()), ha="right", va="center", fontsize=14)
        ax.text(1.08, v.mean(), num(v.mean()), ha="left", va="center", fontsize=14)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["正面\n（入力画像）", "裏側\n（見えていない側）"], fontsize=13)
        ax.set_xlim(-0.45, 1.45)
        ax.set_title(f"{lab}\n18条件中 {int((v > x).sum())} で裏側が上がる（平均 {num(v.mean() - x.mean())}）", loc="left",
                     pad=28, fontsize=13)
        ax.tick_params(axis="x", length=0)
        grid(ax, "y")
    ytitle(axes[0], "読み出し値（上ほど半透明寄り）")
    title(fig, "半透明感の物差しで測ると、本物は裏側で半透明寄りになるが、生成はほぼ正面のまま")
    fig.tight_layout()
    save(fig, name)
    r = json.load(open(FIG3 / "readout_summary.json"))["clip"]["far_vs_input"]
    caption(name, "物差し = 石けんの写真で学習した、CLIP 埋め込みから評定を出す線形の読み出し（値は石けんの評定の単位）。"
                  "正面は入力画像（両方のパネルで同じ）、裏側は見えていない3視点（90・180・270度）の平均。"
                  f"不透明の対照の裏側は平均 {sc['far_opaque'].mean():+.2f} で、物差しは不透明な物体を低く測る。"
                  f"Zero-1-to-3 の裏側は正解の裏側より入力画像と強く相関する（偏相関 {r['partial_zero123_input_given_truth']:.2f} 対 "
                  f"{r['partial_zero123_truth_given_input']:.2f}）。")


# ------------------------------------------------------------------ 図 8: 埋め込みを押す実験
def fig_steer_curves(truth_far):
    name = "08_埋め込みを押す実験.png"
    s = json.load(open(FIG3 / "steer_summary.json"))
    b = s["by_alpha"]
    rs = json.load(open(FIG3 / "readout_summary.json"))
    refs = {"luminance": truth_far[:, 0].mean(), "contrast": truth_far[:, 1].mean(),
            "clip_readout": np.mean(rs["clip"]["scores"]["far_translucent"]),
            "dino_readout": np.mean(rs["dino"]["scores"]["far_translucent"])}
    M = (("luminance", "平均輝度"), ("contrast", "コントラスト"), ("clip_readout", "読み出し（CLIP）"),
         ("dino_readout", "読み出し（DINO）"))
    steps = (-2, -1, 0, 1, 2)

    def pct(m, v):          # 元の生成（押す量 0）から正解の裏側までの差を 100% とした位置。指標ごとの単位の違いを消す
        base = b["translucent_+0"]["far"][m]
        return 100 * (v - base) / (refs[m] - base)

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.9), sharey=True)
    for ax, (m, lab) in zip(axes, M):
        ax.axhline(100, color=TRUTH, lw=2.2)
        ax.axhline(0, color=INK2, lw=1)
        for c in ("control1", "control2"):
            ax.plot([-2, 0, 2], [pct(m, b[f"{c}_-2"]["far"][m]), 0, pct(m, b[f"{c}_+2"]["far"][m])], "-s",
                    color=REF, lw=2, ms=7, mec="white", mew=1)
        v = [pct(m, b[f"translucent_{a:+d}"]["far"][m]) for a in steps]
        ax.plot(steps, v, "-o", color=MODEL, lw=3, ms=9, mec="white", mew=1.5, zorder=3)
        ax.text(2, v[-1] + 7, f"{num(v[-1], 0)}%", ha="center", va="bottom", fontsize=14, zorder=4)
        ax.set_title(lab, loc="left", pad=28)
        ax.set_xticks(steps); ax.set_xticklabels(["−2", "−1", "0", "+1", "+2"])
        ax.set_xlim(-2.4, 2.4); ax.set_ylim(-60, 112)
        ax.set_yticks([0, 50, 100]); ax.set_yticklabels(["元の生成 0%", "50%", "正解の裏側 100%"])
        ax.set_xlabel("押した量")
        grid(ax)
    ytitle(axes[0], "正解までの差のうち、埋まった割合")
    handles = [plt.Line2D([], [], color=MODEL, lw=3, marker="o", ms=9, mec="white", label="「半透明」の方向に押す"),
               plt.Line2D([], [], color=REF, lw=2, marker="s", ms=7, mec="white", label="対照：同じ長さのランダムな方向（2本）")]
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.0, 0.93))
    plus2 = [pct(m, b["translucent_+2"]["far"][m]) for m, _ in M]
    title(fig, f"埋め込みを「半透明」の方向に押しても、生成された裏側は正解まで {min(plus2):.0f}〜{max(plus2):.0f}% しか近づかない")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, name)
    tr = s["transmission"]
    caption(name, "Zero-1-to-3 の画像埋め込み（CLIP）に「半透明」の方向（石けんの評定に沿って埋め込みが変わる向き）を足して生成し直した。"
                  "押す量 ±2 は、石けん122枚が埋め込み上で広がる範囲の端から端に当たる。値は18条件の平均（乱数は固定）で、"
                  "元の生成から正解の裏側までの差を100%として表した。向きは18条件中16〜17で半透明側にそろうが、"
                  f"押した分のうち出力の埋め込みに届いたのは約 {tr['translucent_far']['mean'] * 100:.0f}%"
                  f"（ランダムな方向でも {tr['control1_far']['mean'] * 100:.0f}%・{tr['control2_far']['mean'] * 100:.0f}%）。")


def fig_steer_images(cond, coef):
    from steer_embedding import fname
    rows = [("正解（半透明）", lambda az: gt_image(cond, coef, az, bg=SHOW_BG)[0])]
    for a in (-2, -1, 0, 1, 2):
        lab = {-2: "押す量 −2（不透明側）", 0: "押す量 0（元の生成）", 2: "押す量 +2（半透明側）"}.get(a, f"押す量 {num(a, 0)}")
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
    with open(OUT / "00_図の説明.txt", "w", encoding="utf-8") as f:     # 図の外に置く説明文（発表のメモやスライドの注に使う）
        for k, v in CAPTIONS.items():
            f.write(f"{k}\n{v}\n\n")
    print("wrote 00_図の説明.txt")
