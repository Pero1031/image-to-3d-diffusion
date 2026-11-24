"""Zero-1-to-3 の CLIP 埋め込みを「半透明」の方向へ動かすと、生成される裏側は変わるか。

    HF_HOME=data/hf python image_to_3d/steer_embedding.py generate   # 画像を作る（数時間。作ったものは飛ばす）
    HF_HOME=data/hf python image_to_3d/steer_embedding.py analyze    # 測る（生成とは別のプロセスで）

動機: CLIP 埋め込みからは人の半透明感の評定を読み出せる（Part 3）のに、生成される裏側は半透明に見えない。
「情報はあるが生成器が使っていない」のか、「生成器は使えるが入力から十分に引き出されていない」のかを、
埋め込みを直接動かして確かめる。U-Net に連結する入力画像の潜在表現はそのままにする。

方向（いずれも CLIP の image_embeds の空間、768 次元）:
    半透明      石けんの写真 122 枚（背景なし）で、埋め込みの各成分を評定に回帰した傾き。評定が 1 上がるときに
                埋め込みが平均してどう動くか。読み出しの重みそのものより、実際の画像の散らばりに沿った方向になる
    対照 1, 2   評定を石けんごとに入れ替えて同じように求めた方向。半透明の方向と直交させ、同じ長さにそろえる
動かす量 α（評定の単位）: 半透明 -2, -1, 0, +1, +2。対照 -2, +2。石けんの評定の標準偏差は 0.94。
入力: Part 2 の 18 条件の半透明の刺激。乱数は 0 番で、α によらず同じ初期ノイズを使う（差は条件づけだけから生じる）。
α = 0 は Part 2 で作った画像（zero123_*_seed0）と同じになるはずなので、最初の条件で一致を確かめてから使い回す。

実行前に決めたこと:
  測るもの  裏側（90・180・270 度の平均）と入力と同じ視点（0 度）の、平均輝度・コントラスト（Part 2）と、
            石けんで学習した CLIP と DINO の読み出し（translucency_readout.py と同じ）。
  主な比較  α = +2 と -2 の差。半透明の方向と、同じ長さの対照の方向とで比べる。
  予想      半透明の方向に動かすと、裏側は対照より大きく、半透明の向き（明るく、コントラストが低く、
            読み出しが高く）に変わる。裏側の見えを決める経路は CLIP 埋め込みしかないので。
  判断      半透明の方向の変化が対照と区別できなければ、生成器は埋め込みの半透明の情報を使っていない。
            区別できれば、生成器は使えるのに、入力の埋め込みがそこまで動いていない（引き出されていない）。
"""
import json
import sys
import time

import numpy as np

from common import AZ, COEFS, CONDS, GEN, ROOT, UNSEEN, load_object, object_mask, stats, stim_path, tag

sys.path.insert(0, str(ROOT / "soap"))
from ratings import load  # noqa: E402

OUT = ROOT / "image_to_3d" / "figures"
STEPS = {"translucent": (-2, -1, 0, 1, 2), "control1": (-2, 2), "control2": (-2, 2)}


def directions():
    """{名前: (768,) の方向}。半透明の方向は評定 1 あたりの埋め込みの変化。"""
    data, names, groups, Z, y = load()
    X = data["zero123_clip_nobg"].astype(np.float64)
    Xc = X - X.mean(0)

    def slope(t):
        t = t - t.mean()
        return Xc.T @ t / (t @ t)

    d = slope(y)
    out = {"translucent": d}
    rng = np.random.default_rng(0)
    ids = sorted(set(groups))
    for k in (1, 2):
        perm = dict(zip(ids, rng.permutation(ids)))
        idx = {g: np.where(groups == g)[0] for g in ids}
        yp = np.empty_like(y)
        for g in ids:
            yp[idx[g]] = y[idx[perm[g]]]
        c = slope(yp)
        c = c - (c @ d) / (d @ d) * d                 # 半透明の方向と直交させ、同じ長さに
        out[f"control{k}"] = c / np.linalg.norm(c) * np.linalg.norm(d)
    return out


def fname(direction, alpha, coef, az):
    if alpha == 0:
        return f"zero123_coef{coef}_az{az:03d}_seed0.png"
    return f"steer_{direction}_a{alpha:+d}_coef{coef}_az{az:03d}_seed0.png"


def generate():
    from PIL import Image

    import run_zero123 as z
    dirs = directions()
    print("norm of the direction:", round(float(np.linalg.norm(dirs["translucent"])), 4), flush=True)
    checked = False
    for cond in CONDS:
        out = GEN / tag(cond)
        for coef in COEFS:
            image = load_object(stim_path(cond, coef), 256, bg=1.0)
            if not checked:     # α = 0 で Part 2 の画像が再現されるか
                new = z.novel_views(image, AZ, seed=0, emb_offset=np.zeros(768))
                old = [np.asarray(Image.open(out / fname("", 0, coef, az))) for az in AZ]
                diff = max(np.abs(np.asarray(n, np.int16) - o.astype(np.int16)).max() for n, o in zip(new, old))
                print(f"alpha = 0 reproduces the Part 2 images: max pixel difference {diff}", flush=True)
                checked = True
            for direction, alphas in STEPS.items():
                for a in alphas:
                    if a == 0 or (out / fname(direction, a, coef, AZ[-1])).exists():
                        continue
                    t0 = time.time()
                    for az, v in zip(AZ, z.novel_views(image, AZ, seed=0, emb_offset=a * dirs[direction])):
                        v.save(out / fname(direction, a, coef, az))
                    print(f"{tag(cond)} {coef} {direction} {a:+d}: {time.time() - t0:.0f}s", flush=True)


def analyze():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from translucency_readout import Encoders, fit_readout, recrop

    data, names, groups, Z, y = load()
    dirs = directions()
    enc = Encoders()
    readouts = {}
    for space, key in (("clip", "zero123_clip_nobg"), ("dino", "triposr_dino_nobg")):
        readouts[space] = fit_readout(data[key].astype(np.float64), y, groups)[0]
    # 埋め込みを方向 1 単位動かしたときの読み出しの変化（読み出しは線形なので、どこから動かしても同じ）
    gain_emb = {k: float(readouts["clip"].predict(v[None])[0] - readouts["clip"].predict(np.zeros((1, 768)))[0])
                for k, v in dirs.items()}
    print("embedding readout change per unit step:", {k: round(v, 3) for k, v in gain_emb.items()})

    def measure(im):
        im = im.convert("RGB")
        lum, con = stats(im, object_mask(im))
        c, d = enc(recrop(im))
        return np.concatenate([[lum, con, readouts["clip"].predict(c[None].astype(np.float64))[0],
                                readouts["dino"].predict(d[None].astype(np.float64))[0]], c])   # 4 + 768

    M = ("luminance", "contrast", "clip_readout", "dino_readout")
    rows = []           # (条件, 方向, α) -> 裏側と 0 度の測定値
    for cond in CONDS:
        for coef in COEFS:
            for direction, alphas in STEPS.items():
                for a in alphas:
                    v = {az: measure(Image.open(GEN / tag(cond) / fname(direction, a, coef, az))) for az in AZ}
                    rows.append({"cell": f"{tag(cond)}_coef{coef}", "direction": direction, "alpha": a,
                                 "far": np.mean([v[az] for az in UNSEEN], 0), "front": v[0]})
            print(tag(cond), coef, flush=True)
    cells = sorted({r["cell"] for r in rows})

    def get(direction, a, part, emb=False):
        v = np.array([next(r[part] for r in rows if r["cell"] == c and r["direction"] == direction and r["alpha"] == a)
                      for c in cells])
        return v[:, 4:] if emb else v[:, :4]                                  # (18, 768) か (18, 4)

    # 乱数による違いの目安: Part 2 の 3 つの乱数での裏側の標準偏差（輝度とコントラストだけ）
    seed_sd = []
    for cond in CONDS:
        for coef in COEFS:
            s = [np.mean([stats(im, object_mask(im)) for im in
                          [Image.open(GEN / tag(cond) / f"zero123_coef{coef}_az{az:03d}_seed{k}.png").convert("RGB") for az in UNSEEN]], 0)
                 for k in range(3)]
            seed_sd.append(np.std(s, 0, ddof=1))
    seed_sd = np.mean(seed_sd, 0)

    summary = {"embedding_readout_per_unit": gain_emb, "seed_sd_far": {"luminance": float(seed_sd[0]), "contrast": float(seed_sd[1])},
               "norm_direction": float(np.linalg.norm(dirs["translucent"])), "by_alpha": {}, "plus2_minus2": {}}
    for direction, alphas in STEPS.items():
        for a in alphas:
            summary["by_alpha"][f"{direction}_{a:+d}"] = {part: dict(zip(M, get(direction, a, part).mean(0).tolist()))
                                                         for part in ("far", "front")}
    sign = np.array([1, -1, 1, 1])        # 半透明の向き: 明るく、コントラストが低く、読み出しが高く
    print(f"\nseed SD of the far side (Part 2): luminance {seed_sd[0]:.3f}, contrast {seed_sd[1]:.3f}")
    print(f"{'change from alpha -2 to +2':34s}" + "".join(f"{m:>15s}" for m in M))
    for direction in STEPS:
        for part in ("far", "front"):
            dlt = get(direction, 2, part) - get(direction, -2, part)
            summary["plus2_minus2"][f"{direction}_{part}"] = {
                m: {"mean": float(dlt[:, j].mean()), "sd": float(dlt[:, j].std(ddof=1)),
                    "cells_translucent_way": int((dlt[:, j] * sign[j] > 0).sum())} for j, m in enumerate(M)}
            print(f"{direction + ', ' + part:34s}" + "".join(
                f"{dlt[:, j].mean():>+9.3f} ({(dlt[:, j] * sign[j] > 0).sum():>2d})" for j in range(4)))
    # 半透明の方向の変化は、対照より大きいか（条件ごとの変化の大きさを比べる）
    comp = {}
    for part in ("far", "front"):
        t = (get("translucent", 2, part) - get("translucent", -2, part)) * sign
        cs = [(get(c, 2, part) - get(c, -2, part)) * sign for c in ("control1", "control2")]
        comp[part] = {m: {"translucent_minus_control_mean": float((t[:, j] - np.mean([c[:, j] for c in cs], 0)).mean()),
                          "cells_translucent_larger_than_both_controls": int(np.sum(t[:, j] > np.maximum(np.abs(cs[0][:, j]), np.abs(cs[1][:, j]))))}
                      for j, m in enumerate(M)}
    summary["translucent_vs_controls"] = comp
    print("\ncells where the change along the translucent direction (translucent way) exceeds both controls' |change|:")
    for part in ("far", "front"):
        print(f"  {part:6s}" + "".join(f"{m}: {comp[part][m]['cells_translucent_larger_than_both_controls']}/18   " for m in M))
    # 出力の読み出しは、埋め込みの読み出しの変化のどれだけ動いたか
    for part in ("far", "front"):
        dl = (get("translucent", 2, part) - get("translucent", -2, part))[:, 2].mean()
        summary[f"gain_clip_readout_{part}"] = float(dl / (4 * gain_emb["translucent"]))
        print(f"gain ({part}): output CLIP readout change / embedding readout change = {summary[f'gain_clip_readout_{part}']:.2f}")
    # 追加（実行後に決めた）: 出力画像の CLIP 埋め込みが、動かした方向にどれだけ動いたか。
    # (出力の埋め込みの +2 と -2 の差を方向に射影) / (入力の埋め込みを動かした量 4|d|)。1 なら動かした分がそのまま出力に出る。
    # 対照の方向でも同じ量が測れるので、「半透明の情報だけ届かない」のか「埋め込みの変化が全般に届かない」のかを分けられる
    summary["transmission"] = {}
    print("\ntransmission along each direction (output CLIP embedding change projected on the direction / input change):")
    for direction in STEPS:
        u = dirs[direction] / np.linalg.norm(dirs[direction])
        for part in ("far", "front"):
            tr = (get(direction, 2, part, emb=True) - get(direction, -2, part, emb=True)) @ u / (4 * np.linalg.norm(dirs[direction]))
            summary["transmission"][f"{direction}_{part}"] = {"mean": float(tr.mean()), "sd": float(tr.std(ddof=1)),
                                                              "cells_positive": int((tr > 0).sum())}
            print(f"  {direction + ', ' + part:20s} {tr.mean():+.3f} (SD {tr.std(ddof=1):.3f}, positive in {(tr > 0).sum()}/18)")
    with open(OUT / "steer_summary.json", "w") as f:
        json.dump(summary, f, indent=1)

    # 図 1: α ごとの平均（18 条件）。実線 = 裏側、点線 = 0 度。四角 = 対照の方向
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    for j, (ax, m) in enumerate(zip(axes, M)):
        for part, ls in (("far", "-"), ("front", ":")):
            ax.plot(STEPS["translucent"], [get("translucent", a, part)[:, j].mean() for a in STEPS["translucent"]], ls,
                    marker="o", color="C3", label=f"translucent direction, {'far side' if part == 'far' else '0 deg'}")
            for k, c in enumerate(("control1", "control2")):
                ax.plot([-2, 2], [get(c, a, part)[:, j].mean() for a in (-2, 2)], ls, marker="s", color="0.6",
                        label=f"control directions, {'far side' if part == 'far' else '0 deg'}" if k == 0 else None)
        ax.set_title(m.replace("_", " "), fontsize=9); ax.set_xlabel("step along the direction (soap-rating units)")
    axes[0].legend(fontsize=6)
    fig.suptitle("Zero-1-to-3 with its CLIP embedding moved along the translucency direction (mean of 18 cells)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "steer_curves.png", dpi=110); plt.close(fig)

    # 図 2: 例。行 = α、列 = 視点
    cond, coef = CONDS[0], COEFS[0]
    fig, axes = plt.subplots(len(STEPS["translucent"]), len(AZ), figsize=(2.1 * len(AZ), 2.1 * len(STEPS["translucent"])))
    for r, a in enumerate(STEPS["translucent"]):
        for c, az in enumerate(AZ):
            axes[r, c].imshow(Image.open(GEN / tag(cond) / fname("translucent", a, coef, az)))
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(f"azimuth {az} deg", fontsize=9)
            if c == 0:
                axes[r, c].set_ylabel(f"step {a:+d}", fontsize=9)
    fig.suptitle(f"{tag(cond)}, density {coef}: embedding moved along the translucency direction", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "steer_example.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    {"generate": generate, "analyze": analyze}[sys.argv[1]]()
