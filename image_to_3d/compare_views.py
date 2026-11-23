"""正解の別視点（Blender）と、モデルが生成した別視点を並べ、物体内の明るさの統計を比べる。

    python image_to_3d/compare_views.py

視点が完全には揃わないので、画素ごとの差ではなく、物体の内側の統計（平均輝度とコントラスト）で比べる。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

from common import ROOT, load_object

GT, GEN, OUT = ROOT / "data/gt_views", ROOT / "data/gen_views", ROOT / "image_to_3d/figures"
COEFS, AZ = ("016", "128", "512"), (0, 30, 90, 180, 270)


def luminance(rgb8):
    c = np.asarray(rgb8, np.float32)[..., :3] / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)   # sRGB -> 線形
    return lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def object_mask(im):
    """白背景の画像から物体の領域を取る。画像の縁につながった白い領域を背景とみなす。"""
    white = (np.asarray(im)[..., :3] > 240).all(-1)
    lab, _ = ndimage.label(white)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return ndimage.binary_erosion(~np.isin(lab, border[border > 0]), iterations=3)


def stats(im, mask):
    y = luminance(im)[mask]
    return float(y.mean()), float(y.std() / y.mean())   # 平均輝度、コントラスト（標準偏差 / 平均）


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def gt_image(coef, az):
    return load_object(GT / f"gt_coef{int(coef):03d}_az{az:03d}.png", 256, bg=1.0)


def gen_image(model, coef, az, sign):
    return Image.open(GEN / f"{model}_coef{coef}_az{(sign * az) % 360:03d}.png").convert("RGB")


# 方位角の正の向きがモデルと Blender で同じかを、30 度の輪郭の重なりで決める
signs = {}
for model in ("zero123", "triposr"):
    g = object_mask(gt_image("512", 30))
    score = {s: iou(g, object_mask(gen_image(model, "512", 30, s))) for s in (1, -1)}
    signs[model] = max(score, key=score.get)
    print(f"{model}: azimuth sign {signs[model]:+d}  (IoU +30: {score[1]:.3f}, -30: {score[-1]:.3f})")

rows = [("Blender (ground truth)", lambda c, a: gt_image(c, a)),
        ("Zero-1-to-3", lambda c, a: gen_image("zero123", c, a, signs["zero123"])),
        ("TripoSR", lambda c, a: gen_image("triposr", c, a, signs["triposr"]))]
table = {}
for coef in COEFS:
    fig, axes = plt.subplots(3, len(AZ), figsize=(2.2 * len(AZ), 7))
    for r, (name, get) in enumerate(rows):
        for c, az in enumerate(AZ):
            im = get(coef, az)
            table[name, coef, az] = stats(im, object_mask(im))
            ax = axes[r, c]
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"azimuth {az} deg")
            if c == 0:
                ax.set_ylabel(name)
            ax.set_xlabel("mean {:.2f}  contrast {:.2f}".format(*table[name, coef, az]), fontsize=8)
    fig.suptitle(f"density coef {coef}")
    fig.tight_layout()
    fig.savefig(OUT / f"compare_coef{coef}.png", dpi=110)
    plt.close(fig)

# 濁りの違い（coef 016 と 512）が、別視点でも統計の差として残っているか
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for k, label in enumerate(("mean luminance", "contrast (std / mean)")):
    for name, _ in rows:
        for coef, ls in zip(COEFS, (":", "--", "-")):
            axes[k].plot(range(len(AZ)), [table[name, coef, a][k] for a in AZ], ls, marker="o", ms=3,
                         color={"B": "k", "Z": "C0", "T": "C1"}[name[0]], label=f"{name}, coef {coef}")
    axes[k].set_xticks(range(len(AZ)), AZ); axes[k].set_xlabel("azimuth (deg)"); axes[k].set_title(label)
axes[1].legend(fontsize=6, ncol=1)
fig.tight_layout(); fig.savefig(OUT / "compare_stats.png", dpi=120); plt.close(fig)

print(f"{'':24s}{'coef':>5s}" + "".join(f"{a:>14d}" for a in AZ))
for name, _ in rows:
    for coef in COEFS:
        print(f"{name:24s}{coef:>5s}" + "".join("   {:.3f}/{:.3f}".format(*table[name, coef, a]) for a in AZ))
