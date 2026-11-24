"""石けんの評定で学習した読み出しを「半透明感の物差し」にして、Part 2 の画像に当てる。

    python image_to_3d/translucency_readout.py

Part 3 では、Zero-1-to-3 が入力画像を読み取る部分（CLIP）の特徴から、人の半透明感の評定を線形に
読み出せた。ここではその読み出し（石けんの写真 122 枚すべてで学習したリッジ回帰）を Part 2 の
画像に当てる。モデル自身の特徴空間で、生成された裏側が半透明に見えるかを測る。
比較のため、TripoSR が画像を読み取る部分（DINO）で学習した読み出しも当てる。

読み出しの値の単位は、石けんの評定（参加者ごとに標準化した値の平均。石けん 122 枚での標準偏差は 0.94）。
画像はすべて、物体が画像の 85% を占めるように白い背景に置く（石けんの「背景なし」の特徴と同じ）。

実行前に決めたこと:
  妥当性  本物の半透明の物体は、不透明の対照より高く出るはず（正面・裏側の各 18 条件）。
          ここで大半の条件が逆なら、物差しが石けんから刺激画像へ持ち越せていないので以下は解釈しない。
  問い 1  Zero-1-to-3 の裏側は、本物の半透明の裏側より低く出るか（Part 2 の明るさの解析では、暗く、
          コントラストが強かった）。
  問い 2  入力と同じ視点（0 度）の出力は入力と同じ値を保つか。保つなら、下がるのは新しい視点を作る
          ところで起きている。
  問い 3  正面の明るさを揃えた 10 条件で、裏側の値の差（半透明を入力 - 不透明を入力）は、本物の裏側の
          差と入力の正面の差のどちらに従うか（Part 2 の組の解析と同じ形）。
"""
import json
import pickle
import sys
import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import CLIPVisionModelWithProjection

from common import AZ, COEFS, CONDS, GEN, GT, ROOT, SEEDS, UNSEEN, load_object, object_mask, stim_path, tag

sys.path.insert(0, str(ROOT / "soap"))
from ratings import ALPHAS, load  # noqa: E402

OUT = ROOT / "image_to_3d" / "figures"
device = "cuda"
LIGHT = {"studio_rot000": "front", "studio_rot180": "back", "rustig_koppie_rot315": "front",
         "rustig_koppie_rot225": "side", "rustig_koppie_rot135": "back"}


def lighting(cond):
    return LIGHT["{}_rot{:03d}".format(cond[1], cond[2])]


# ---------------------------------------------------------------- 読み出し（石けんで学習）
def fit_readout(X, y, groups):
    """正則化の強さを個体ごとの交差検証で選び、122 枚すべてで学習し直す。(モデル, 交差検証の r)。"""
    best = None
    for a in ALPHAS:
        pred = np.zeros(len(y))
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            pred[te] = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X[tr], y[tr]).predict(X[te])
        r = np.corrcoef(pred, y)[0, 1]
        if best is None or r > best[1]:
            best = (a, r)
    return make_pipeline(StandardScaler(), Ridge(alpha=best[0])).fit(X, y), float(best[1]), float(best[0])


# ---------------------------------------------------------------- 画像の準備と特徴
def recrop(im, ratio=0.85):
    """白い背景の生成画像を、物体が ratio を占めるように置き直す（正解画像の load_object と揃える）。"""
    a = np.asarray(im.convert("RGB"))
    m = object_mask(im)
    ys, xs = np.where(m)
    y0, y1, x0, x1 = max(ys.min() - 3, 0), min(ys.max() + 4, a.shape[0]), max(xs.min() - 3, 0), min(xs.max() + 4, a.shape[1])
    side = int(max(y1 - y0, x1 - x0) / ratio)
    canvas = np.full((side, side, 3), 255, np.uint8)
    oy, ox = (side - (y1 - y0)) // 2, (side - (x1 - x0)) // 2
    canvas[oy:oy + y1 - y0, ox:ox + x1 - x0] = a[y0:y1, x0:x1]
    return Image.fromarray(canvas)


def gt(cond, kind, coef, az):
    name = f"gt_coef{coef}_az{az:03d}.png" if kind == "translucent" else f"gt_opaque_coef{coef}_az{az:03d}.png"
    return load_object(GT / tag(cond) / name, 512, bg=1.0)


def generated(cond, name):
    return recrop(Image.open(GEN / tag(cond) / name))


class Encoders:
    def __init__(self):
        repo = snapshot_download("bennyguo/zero123-xl-diffusers", local_files_only=True)
        self.clip = CLIPVisionModelWithProjection.from_pretrained(repo, subfolder="image_encoder", variant="fp16",
                                                                  torch_dtype=torch.float32).to(device).eval()
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        for mod in ("torchmcubes", "rembg"):           # TripoSR の取り込みに要るが、ここでは使わない
            sys.modules.setdefault(mod, types.ModuleType(mod))
        sys.modules["torchmcubes"].marching_cubes = None
        sys.path.insert(0, str(ROOT / "data/third_party/TripoSR"))
        from tsr.system import TSR
        self.tsr = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml",
                                       weight_name="model.ckpt").to(device).eval()

    @torch.no_grad()
    def __call__(self, im):
        """im: 物体が 85% を占める白背景の PIL。soap/features.py と同じ手順で (CLIP, DINO) を返す。"""
        x = torch.from_numpy(np.asarray(im.convert("RGB").resize((256, 256), Image.LANCZOS))).float()
        x = x.div(127.5).sub(1).permute(2, 0, 1)[None].to(device)
        c = F.interpolate(x * 0.5 + 0.5, size=224, mode="bicubic", antialias=True)
        clip = self.clip((c - self.mean) / self.std).image_embeds[0].float().cpu().numpy()
        x = torch.from_numpy(np.asarray(im.convert("RGB").resize((512, 512), Image.LANCZOS))).float().div(255)
        dino = self.tsr.image_tokenizer(x.permute(2, 0, 1)[None, None].to(device))[0, 0].mean(-1).float().cpu().numpy()
        return clip, dino


# ---------------------------------------------------------------- 計測
def collect(enc):
    """条件ごとに、各種の画像の特徴を集める。{key: (n, 768)} の配列を条件ごとのリストで返す。"""
    cells = []
    for cond in CONDS:
        with open(GT / tag(cond) / "opaque_albedo.json") as f:
            matched = json.load(f)["matched"]
        for coef in COEFS:
            ims = {
                "input_translucent": [load_object(stim_path(cond, coef), 512, bg=1.0)],
                "input_opaque": [gt(cond, "opaque", coef, 0)],
                "far_translucent": [gt(cond, "translucent", coef, az) for az in UNSEEN],
                "far_opaque": [gt(cond, "opaque", coef, az) for az in UNSEEN],
                "triposr_az0": [generated(cond, f"triposr_coef{coef}_az000.png")],
                "triposr_far": [generated(cond, f"triposr_coef{coef}_az{az:03d}.png") for az in UNSEEN],
            }
            for m in ("zero123", "zero123op"):        # (乱数, 視点) の順に並べる
                ims[f"{m}_az0"] = [generated(cond, f"{m}_coef{coef}_az000_seed{s}.png") for s in SEEDS]
                ims[f"{m}_far"] = [generated(cond, f"{m}_coef{coef}_az{az:03d}_seed{s}.png") for s in SEEDS for az in UNSEEN]
            feats = {k: [enc(im) for im in v] for k, v in ims.items()}
            cells.append({"cond": cond, "coef": coef, "matched": bool(matched[coef]),
                          "clip": {k: np.array([f[0] for f in v]) for k, v in feats.items()},
                          "dino": {k: np.array([f[1] for f in v]) for k, v in feats.items()}})
            print(tag(cond), coef, flush=True)
    return cells


def main():
    data, names, groups, Z, y = load()
    enc = Encoders()
    cache = ROOT / "data" / "readout_features.pkl"        # 特徴の取り直しには数分かかるので保存しておく
    if cache.exists():
        cells = pickle.loads(cache.read_bytes())
    else:
        cells = collect(enc)
        cache.write_bytes(pickle.dumps(cells))

    # 特徴の取り方が石けんのときと揃っているか: 背景なしの石けんの写真から取り直し、保存済みの特徴と比べる
    from features import prepare
    check = [enc(prepare(n, "nobg", 512)[0]) for n in names[:4]]
    for i, key in enumerate(("zero123_clip_nobg", "triposr_dino_nobg")):
        cos = [float(c[i] @ s / np.linalg.norm(c[i]) / np.linalg.norm(s)) for c, s in zip(check, data[key][:4])]
        print(f"re-extracted vs saved {key} (cosine):", np.round(cos, 4))

    summary, plots = {}, {}
    for space, key in (("clip", "zero123_clip_nobg"), ("dino", "triposr_dino_nobg")):
        model, r_cv, alpha = fit_readout(data[key].astype(np.float64), y, groups)
        sc = [{k: model.predict(v.astype(np.float64)) for k, v in c[space].items()} for c in cells]
        S = {k: np.array([s[k].mean() for s in sc]) for k in sc[0]}             # (18,)
        seeds = {m: np.array([s[f"{m}_far"].reshape(len(SEEDS), len(UNSEEN)).mean(1) for s in sc])
                 for m in ("zero123", "zero123op")}                                # (18, 3)
        light = np.array([lighting(c["cond"]) for c in cells])
        matched = np.array([c["matched"] for c in cells])
        res = {"readout_cv_r_on_soaps": r_cv, "alpha": alpha,
               "scores": {k: v.tolist() for k, v in S.items()},
               "cells": [f"{tag(c['cond'])}_coef{c['coef']}" for c in cells], "lighting": light.tolist(),
               "matched": matched.tolist(),
               "seed_sd_far": {m: float(v.std(1, ddof=1).mean()) for m, v in seeds.items()}}
        # 妥当性
        res["valid_front"] = int((S["input_translucent"] > S["input_opaque"]).sum())
        res["valid_far"] = int((S["far_translucent"] > S["far_opaque"]).sum())
        # 問い 1, 2
        res["zero123_far_below_truth"] = int((S["zero123_far"] < S["far_translucent"]).sum())
        res["triposr_far_below_truth"] = int((S["triposr_far"] < S["far_translucent"]).sum())
        groups_light = {g: light == g for g in ("front", "side", "back")} | {"all": np.ones(len(light), bool)}
        res["means"] = {g: {k: float(v[sel].mean()) for k, v in S.items()} for g, sel in groups_light.items()}
        # 問い 3
        d_true = S["far_translucent"] - S["far_opaque"]
        d_front = S["input_translucent"] - S["input_opaque"]
        d_model = S["zero123_far"] - S["zero123op_far"]
        d_az0 = S["zero123_az0"] - S["zero123op_az0"]
        noise = np.sqrt(seeds["zero123"].var(1, ddof=1) / 3 + seeds["zero123op"].var(1, ddof=1) / 3)
        x, t, f, a = d_model[matched], d_true[matched], d_front[matched], d_az0[matched]
        res["pairs"] = {"n": int(matched.sum()), "slope_on_truth": float((x * t).sum() / (t * t).sum()),
                        "mean_true_diff": float(t.mean()), "mean_front_diff": float(f.mean()),
                        "mean_model_far_diff": float(x.mean()), "mean_model_az0_diff": float(a.mean()),
                        "mean_noise": float(noise[matched].mean()),
                        "r_model_truth": float(np.corrcoef(x, t)[0, 1]), "r_model_front": float(np.corrcoef(x, f)[0, 1]),
                        "r_front_truth": float(np.corrcoef(f, t)[0, 1]),
                        "model_same_sign_as_truth": int((np.sign(x) == np.sign(t)).sum())}
        # 追加（結果を見てから加えたもの）: 裏側の値は、入力の正面の値をそのまま写したものか
        inp, far, zf, tf = S["input_translucent"], S["far_translucent"], S["zero123_far"], S["triposr_far"]

        def partial(a, b, c):
            """c を除いた a と b の偏相関。"""
            rab, rac, rbc = np.corrcoef(a, b)[0, 1], np.corrcoef(a, c)[0, 1], np.corrcoef(b, c)[0, 1]
            return float((rab - rac * rbc) / np.sqrt((1 - rac ** 2) * (1 - rbc ** 2)))
        res["far_vs_input"] = {
            "mean_change_truth": float((far - inp).mean()), "mean_change_zero123": float((zf - inp).mean()),
            "mean_change_zero123_from_az0": float((zf - S["zero123_az0"]).mean()),
            "mean_change_triposr": float((tf - inp).mean()),
            "mean_change_triposr_from_az0": float((tf - S["triposr_az0"]).mean()),
            "truth_above_input": int((far > inp).sum()), "zero123_above_input": int((zf > inp).sum()),
            "r_zero123_input": float(np.corrcoef(zf, inp)[0, 1]), "r_zero123_truth": float(np.corrcoef(zf, far)[0, 1]),
            "r_input_truth": float(np.corrcoef(inp, far)[0, 1]),
            "partial_zero123_truth_given_input": partial(zf, far, inp),
            "partial_zero123_input_given_truth": partial(zf, inp, far)}
        summary[space] = res
        plots[space] = (S, seeds)

        print(f"\n=== readout from {key} (cross-validated r on soaps {r_cv:.3f}, alpha {alpha:g}) ===")
        print(f"validity: translucent > opaque in {res['valid_front']}/18 front views, {res['valid_far']}/18 far sides")
        print(f"far side below the true translucent far side: Zero-1-to-3 {res['zero123_far_below_truth']}/18, "
              f"TripoSR {res['triposr_far_below_truth']}/18;  seed SD of Zero-1-to-3 far side {res['seed_sd_far']['zero123']:.3f}")
        cols = ["input_translucent", "input_opaque", "far_translucent", "far_opaque", "zero123_az0", "zero123_far",
                "zero123op_az0", "zero123op_far", "triposr_az0", "triposr_far"]
        print(f"{'lighting':8s}" + "".join(f"{c:>18s}" for c in cols))
        for g in ("front", "side", "back", "all"):
            print(f"{g:8s}" + "".join(f"{res['means'][g][c]:>18.2f}" for c in cols))
        pr = res["pairs"]
        print(f"matched pairs (n={pr['n']}): true far diff {pr['mean_true_diff']:+.2f}, front diff {pr['mean_front_diff']:+.2f}, "
              f"model az0 diff {pr['mean_model_az0_diff']:+.2f}, model far diff {pr['mean_model_far_diff']:+.2f} "
              f"(seed noise {pr['mean_noise']:.2f}); slope on truth {pr['slope_on_truth']:+.2f}; "
              f"r(model, truth) {pr['r_model_truth']:+.2f}, r(model, front) {pr['r_model_front']:+.2f}, "
              f"r(front, truth) {pr['r_front_truth']:+.2f}; same sign as truth {pr['model_same_sign_as_truth']}/{pr['n']}")
        fv = res["far_vs_input"]
        print(f"far side minus input view: truth {fv['mean_change_truth']:+.2f} (higher in {fv['truth_above_input']}/18), "
              f"Zero-1-to-3 {fv['mean_change_zero123']:+.2f} (higher in {fv['zero123_above_input']}/18; "
              f"{fv['mean_change_zero123_from_az0']:+.2f} from its own 0-degree view), TripoSR {fv['mean_change_triposr']:+.2f} "
              f"({fv['mean_change_triposr_from_az0']:+.2f} from its own 0-degree view)")
        print(f"across 18 cells: r(Zero-1-to-3 far, input) {fv['r_zero123_input']:+.2f}, r(Zero-1-to-3 far, true far) "
              f"{fv['r_zero123_truth']:+.2f}, r(input, true far) {fv['r_input_truth']:+.2f}; partial r with truth given input "
              f"{fv['partial_zero123_truth_given_input']:+.2f}, with input given truth {fv['partial_zero123_input_given_truth']:+.2f}")

        # 図: 条件ごとに、各画像の読み出しの値を 1 行に並べる
        fig, ax = plt.subplots(figsize=(8.5, 7))
        marks = [("input_translucent", "o", "C0", "input: translucent (true front)"),
                 ("far_translucent", "o", "k", "truth far side: translucent"),
                 ("far_opaque", "s", "0.6", "truth far side: opaque control"),
                 ("zero123_az0", "^", "C3", "Zero-1-to-3, same view as input"),
                 ("zero123_far", "v", "C3", "Zero-1-to-3 far side"),
                 ("triposr_far", "D", "C2", "TripoSR far side")]
        for k, mk, col, lab in marks:
            ax.scatter(S[k], np.arange(len(cells)), marker=mk, s=30,
                       facecolors=col if mk in "osD" else "none", edgecolors=col, label=lab, zorder=3)
        for i in range(len(cells)):
            ax.plot([S["far_translucent"][i], S["zero123_far"][i]], [i, i], color="C3", lw=0.8, alpha=0.5)
        ax.set_yticks(range(len(cells)), [f"{tag(c['cond'])} {c['coef']} ({lighting(c['cond'])})" for c in cells], fontsize=7)
        ax.invert_yaxis(); ax.axvline(0, color="gray", lw=0.6)
        ax.set_xlabel("translucency readout (units of the soap ratings; higher = more translucent)")
        ax.set_title(f"readout trained on soap photos ({'Zero-1-to-3 CLIP' if space == 'clip' else 'TripoSR DINO'} features)",
                     fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout(); fig.savefig(OUT / f"readout_{space}.png", dpi=110); plt.close(fig)

    with open(OUT / "readout_summary.json", "w") as f:
        json.dump(summary, f, indent=1)

    # 図: 横軸 = 入力（正面）の値、縦軸 = 裏側の値。対角線上なら「裏側も正面と同じだけ半透明に見える」
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, space in zip(axes, ("clip", "dino")):
        S, seeds = plots[space]
        x = S["input_translucent"]
        ax.scatter(x, S["far_opaque"], marker="s", s=22, color="0.65", label="truth: opaque control")
        ax.scatter(x, S["far_translucent"], marker="o", s=26, color="k", label="truth: translucent")
        ax.errorbar(x, S["zero123_far"], yerr=seeds["zero123"].std(1, ddof=1), fmt="v", ms=5, color="C3",
                    mfc="white", capsize=2, label="Zero-1-to-3 (SD over seeds)")
        ax.scatter(x, S["triposr_far"], marker="D", s=20, color="C2", label="TripoSR")
        lo = min(x.min(), *(S[k].min() for k in ("far_opaque", "far_translucent", "zero123_far", "triposr_far"))) - 0.05
        hi = max(x.max(), *(S[k].max() for k in ("far_opaque", "far_translucent", "zero123_far", "triposr_far"))) + 0.05
        ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel("readout of the input image (true front view)")
        ax.set_ylabel("readout of the far side (mean of 90, 180, 270 deg)")
        ax.set_title(f"readout from {'Zero-1-to-3 CLIP' if space == 'clip' else 'TripoSR DINO'} features", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "readout_scatter.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
