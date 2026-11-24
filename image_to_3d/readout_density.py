"""半透明感の物差しが、CG の刺激でも材質の違いを捉えているかを確かめる。

    python image_to_3d/readout_density.py

石けんの評定で学習した読み出し（translucency_readout.py と同じもの）を、濁り（散乱の密度 sigma_t）だけを
変えた入力画像 41 枚に当てる。6 条件（形 x 照明と向き）それぞれで、形・照明・視点は同じで濁りだけが違う。
  stim01 の 5 条件: 濁り 16, 32, 64, 128, 256, 512（1 オクターブ刻み）
  stim02 の 1 条件: 濁り 16〜512 を半オクターブ刻みで 11 段階

実行前に決めたこと:
  予想    濁りが増えるほど読み出しは下がる（石けんでは、透き通ったものほど半透明感が高く評定されている）。
          各条件で、濁りとの順位相関が負になり、隣どうしの段階の大半で下がるはず。
  比較 1  画像統計量（物体内の平均輝度とコントラスト）でも同じように並ぶか。並ぶなら、読み出しが並ぶこと
          だけでは「物差しが画像統計量以上のものを見ている」とは言えない。
  比較 2  濁りの違いと照明の違いのどちらで読み出しが大きく動くか（stim01 の 5 条件 x 6 段階で、
          分散を濁り・条件に分ける）。人も逆光の物体をより半透明に見るので、照明で動くこと自体は失敗ではない。
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common import CONDS, ROOT, STIM, load_object, stats, tag
from translucency_readout import Encoders, fit_readout, load

OUT = ROOT / "image_to_3d" / "figures"
LIGHT = {("studio", 0): "front", ("studio", 180): "back", ("rustig_koppie", 315): "front",
         ("rustig_koppie", 225): "side", ("rustig_koppie", 135): "back"}


def inputs(cond):
    """条件 cond の入力画像を濁りの順に [(濁り, パス)]。"""
    shape, light, rot = cond
    paths = sorted((STIM / f"{shape}_{light}").glob(f"{shape}_rot{rot:03d}_coef*.png"))
    return sorted((int(p.stem.split("coef")[1]), p) for p in paths)


def main():
    data, names, groups, Z, y = load()
    readouts = {}
    for space, key in (("clip", "zero123_clip_nobg"), ("dino", "triposr_dino_nobg")):
        readouts[space] = fit_readout(data[key].astype(np.float64), y, groups)[0]
    enc = Encoders()

    rows = []
    for cond in CONDS:
        for dens, path in inputs(cond):
            clip, dino = enc(load_object(path, 512, bg=1.0))
            lum, con = stats(*load_object(path, 256, bg=1.0, with_mask=True))
            rows.append({"cond": tag(cond), "lighting": LIGHT[(cond[1], cond[2])], "density": dens,
                         "clip": float(readouts["clip"].predict(clip[None].astype(np.float64))[0]),
                         "dino": float(readouts["dino"].predict(dino[None].astype(np.float64))[0]),
                         "luminance": lum, "contrast": con})
        print(tag(cond), [r["density"] for r in rows if r["cond"] == tag(cond)], flush=True)

    measures = ("clip", "dino", "luminance", "contrast")
    summary = {"n_images": len(rows), "rows": rows, "per_condition": {}, "variance_stim01": {}}
    print(f"\n{len(rows)} images. Spearman rho with density, and adjacent steps that go down:")
    print(f"{'condition':30s}" + "".join(f"{m:>18s}" for m in measures))
    for cond in CONDS:
        sel = [r for r in rows if r["cond"] == tag(cond)]
        d = np.array([r["density"] for r in sel])
        res = {}
        for m in measures:
            v = np.array([r[m] for r in sel])
            res[m] = {"rho": float(spearmanr(d, v)[0]), "steps_down": int((np.diff(v) < 0).sum()),
                      "steps": len(v) - 1, "range": float(v.max() - v.min())}
        summary["per_condition"][tag(cond)] = res
        print(f"{tag(cond):30s}" + "".join(f"{res[m]['rho']:>+10.2f} ({res[m]['steps_down']:>2d}/{res[m]['steps']:>2d})"
                                           for m in measures))

    # 比較 2: stim01 の 5 条件 x 6 段階を、濁りと条件（照明と向き）に分ける（交互作用は残りに入る）
    dens6 = (16, 32, 64, 128, 256, 512)
    for m in measures:
        M = np.array([[next(r[m] for r in rows if r["cond"] == tag(c) and r["density"] == k) for k in dens6]
                      for c in CONDS if c[0] == "stim01"])                       # (5, 6)
        tot = ((M - M.mean()) ** 2).sum()
        ss_d = M.shape[0] * ((M.mean(0) - M.mean()) ** 2).sum()
        ss_c = M.shape[1] * ((M.mean(1) - M.mean()) ** 2).sum()
        summary["variance_stim01"][m] = {"density": float(ss_d / tot), "condition": float(ss_c / tot),
                                         "residual": float(1 - (ss_d + ss_c) / tot),
                                         "density_means": M.mean(0).tolist()}
    print("\nshare of variance (stim01, 5 conditions x 6 densities):")
    for m, v in summary["variance_stim01"].items():
        print(f"  {m:10s} density {v['density']:.2f}   condition {v['condition']:.2f}   residual {v['residual']:.2f}")
    print("soap ratings: SD over the 122 photos = 0.94 (readout units)")
    (OUT / "readout_density.json").write_text(json.dumps(summary, indent=1))

    # 図: 濁りに対する読み出しと画像統計量。線 = 1 条件、色 = 照明の向き
    color = {"front": "#2a78d6", "side": "#8a8984", "back": "#eb6834"}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, m, lab in zip(axes, measures, ("readout (CLIP)", "readout (DINO)", "mean luminance", "contrast (SD / mean)")):
        for cond in CONDS:
            sel = [r for r in rows if r["cond"] == tag(cond)]
            ax.plot([r["density"] for r in sel], [r[m] for r in sel], marker="o", ms=4, lw=1.5,
                    color=color[sel[0]["lighting"]], ls="--" if cond[0] == "stim02" else "-")
        ax.set_xscale("log", base=2); ax.set_xticks(dens6); ax.set_xticklabels(dens6)
        ax.set_xlabel("density (sigma_t)"); ax.set_title(lab, loc="left")
        ax.grid(color="#e5e4e0", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], color=c, lw=2, label=f"{k}-lit") for k, c in color.items()]
    handles.append(plt.Line2D([], [], color="#52514e", lw=1.5, ls="--", label="shape B (stim02)"))
    axes[0].legend(handles=handles, frameon=False, fontsize=9)
    fig.suptitle("Input images that differ only in density: does the soap-trained readout order them?", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(OUT / "readout_density.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
