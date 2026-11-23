"""正解の別視点を Blender (Cycles) で描く。カメラを物体のまわりに水平に回すだけで、物体と照明は動かさない。

    blender --background blob_stim01_fine.blend --python image_to_3d/render_gt_views.py -- --out data/gt_views

設定は本番の刺激と同じ（256 バウンス、デノイズなし、Standard）。違いは解像度 512x512 とサンプル数 512 固定。
.blend は保存しない。
"""
import argparse
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np

PARAM = {16: (15.84, 0.16), 128: (126.72, 1.28), 512: (506.88, 5.12)}  # 散乱・吸収の密度
TEMPLATE = dict(max_bounces=256, volume_bounces=256, transmission_bounces=64, sample_clamp_direct=0.0,
                sample_clamp_indirect=0.0, use_denoising=False, time_limit=0.0)

argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--hdri", type=Path, required=True)
ap.add_argument("--coefs", type=int, nargs="+", default=[16, 128, 512])
ap.add_argument("--azimuths", type=float, nargs="+", default=[0, 30, 90, 180, 270])
ap.add_argument("--samples", type=int, default=512)
ap.add_argument("--size", type=int, default=512)
args = ap.parse_args(argv)
args.out = args.out.resolve()  # 相対パスだと Blender が別の場所に書くため
args.out.mkdir(parents=True, exist_ok=True)

scene = bpy.context.scene
c = scene.cycles
scene.render.engine = "CYCLES"
vs = scene.view_settings
vs.view_transform, vs.look, vs.exposure, vs.gamma = "Standard", "None", 0.0, 1.0
for k, v in TEMPLATE.items():
    setattr(c, k, v)
c.use_adaptive_sampling, c.samples, c.seed = False, args.samples, 0
scene.render.resolution_x = scene.render.resolution_y = args.size
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
for o in scene.objects:
    if o.type == "LIGHT":
        o.hide_render = True
env = next(n for n in scene.world.node_tree.nodes if n.type == "TEX_ENVIRONMENT")
env.image = bpy.data.images.load(str(args.hdri), check_existing=True)
for n in scene.world.node_tree.nodes:
    if n.type == "MAPPING":
        n.inputs["Rotation"].default_value[2] = 0.0   # スタジオ照明、回転 0 度
c.device = "GPU"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "CUDA"
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == "CUDA"

mat = bpy.data.materials["Translucent"]
mat.cycles.homogeneous_volume = True
nodes = {n.type: n for n in mat.node_tree.nodes}
cam = scene.camera                      # Track To で常に物体（原点）を向く
x0, y0, z0 = cam.location
radius, phi0 = math.hypot(x0, y0), math.atan2(y0, x0)
print(f"[gt] camera: elevation {math.degrees(math.atan2(z0, radius)):.2f} deg, distance {math.hypot(radius, z0):.3f}, "
      f"fov {math.degrees(cam.data.angle):.2f} deg", flush=True)

scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
for coef in args.coefs:
    nodes["VOLUME_SCATTER"].inputs["Density"].default_value, nodes["VOLUME_ABSORPTION"].inputs["Density"].default_value = PARAM[coef]
    for az in args.azimuths:
        phi = phi0 + math.radians(az)   # 上から見て反時計回りが正
        cam.location = (radius * math.cos(phi), radius * math.sin(phi), z0)
        scene.render.filepath = str(args.out / f"gt_coef{coef:03d}_az{int(az):03d}.png")
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        print(f"[gt] coef {coef} az {az:.0f}: {time.perf_counter() - t0:.0f}s", flush=True)
