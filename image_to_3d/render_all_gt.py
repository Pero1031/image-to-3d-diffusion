"""全条件について、正解（半透明）と対照（不透明）の別視点を描く。描画済みのものは飛ばす。

    python image_to_3d/render_all_gt.py

不透明の対照の反射率は、正面（0 度）の平均輝度が半透明と同じになるように条件ごとに決める。
"""
import json
import subprocess
import time

import numpy as np

from common import AZ, BLEND_DIR, BLENDER, COEFS, CONDS, GT, LIGHTS, ROOT, load_object, stats, stim_path, tag

PILOT = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)   # 0.9 を、現実的な不透明の物体の上限とする


def blender(cond, out, *extra):
    shape, light, rot = cond
    hdri, k = LIGHTS[light]
    cmd = [BLENDER, "--background", str(BLEND_DIR / f"blob_{shape}_fine.blend"), "--python-exit-code", "1",
           "--python", str(ROOT / "image_to_3d/render_gt_views.py"), "--", "--out", str(out),
           "--hdri", str(BLEND_DIR / hdri), "--rotation", str(rot), "--exposure-k", str(k), *extra]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def mean_lum(path):
    return stats(*load_object(path, 256, bg=1.0, with_mask=True))[0]


for cond in CONDS:
    out, t0 = GT / tag(cond), time.time()
    if not (out / f"gt_coef{COEFS[-1]}_az{AZ[-1]:03d}.png").exists():
        blender(cond, out)
    # 確認: 0 度の描画は既存の刺激画像と同じ明るさになるはず
    check = [(mean_lum(out / f"gt_coef{c}_az000.png"), mean_lum(stim_path(cond, c))) for c in COEFS]
    if not (out / "opaque_albedo.json").exists():
        blender(cond, out / "pilot", "--azimuths", "0", "--samples", "256",
                "--opaque", *[f"a{round(a * 100):03d}={a}" for a in PILOT])
        lum = [mean_lum(out / "pilot" / f"gt_opaque_a{round(a * 100):03d}_az000.png") for a in PILOT]
        albedo = {c: float(np.interp(target, lum, PILOT)) for c, (target, _) in zip(COEFS, check)}
        blender(cond, out, "--opaque", *[f"coef{c}={a:.4f}" for c, a in albedo.items()])
        with open(out / "opaque_albedo.json", "w") as f:
            # matched が False の条件は、反射率を上限にしても半透明の正面の明るさに届かなかった
            json.dump({"albedo": albedo, "matched": {c: bool(t <= max(lum)) for c, (t, _) in zip(COEFS, check)},
                       "pilot_luminance": dict(zip(map(str, PILOT), lum))}, f, indent=1)
    with open(out / "opaque_albedo.json") as f:
        info = json.load(f)
    albedo = info["albedo"]
    print(f"{tag(cond)}: {time.time() - t0:.0f}s  render vs stimulus at 0 deg: "
          + ", ".join(f"{a:.3f}/{b:.3f}" for a, b in check)
          + "  albedo: " + ", ".join(f"{albedo[c]:.2f}{'' if info['matched'][c] else '(max)'}" for c in COEFS), flush=True)
