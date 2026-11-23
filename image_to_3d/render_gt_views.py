"""正解の別視点を Blender (Cycles) で描く。カメラを物体のまわりに水平に回すだけで、物体と照明は動かさない。

    blender --background blob_stim01_fine.blend --python image_to_3d/render_gt_views.py -- --out DIR --hdri FILE ...

設定は本番の刺激と同じ（256 バウンス、デノイズなし、Standard）。違いは解像度 512x512 とサンプル数 512 固定。
--opaque を付けると、対照条件（形・照明・表面の粗さと屈折率は同じで、光が中に入らない材質）を描く。
.blend は保存しない。
"""
import argparse
import math
import sys
import time
from pathlib import Path

import bpy

PARAM = {16: (15.84, 0.16), 128: (126.72, 1.28), 512: (506.88, 5.12)}  # 散乱・吸収の密度
TEMPLATE = dict(max_bounces=256, volume_bounces=256, transmission_bounces=64, sample_clamp_direct=0.0,
                sample_clamp_indirect=0.0, use_denoising=False, time_limit=0.0)

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--hdri", type=Path, required=True)
ap.add_argument("--rotation", type=float, default=0.0, help="照明の回転角（度）")
ap.add_argument("--exposure-k", type=float, default=1.0, help="露出倍率（本番の PNG 変換と同じ値）")
ap.add_argument("--coefs", type=int, nargs="+", default=[16, 128, 512])
ap.add_argument("--opaque", nargs="+", default=None, help="名前=反射率 の並び。例: coef016=0.34")
ap.add_argument("--azimuths", type=float, nargs="+", default=[0, 90, 180, 270])
ap.add_argument("--samples", type=int, default=512)
ap.add_argument("--size", type=int, default=512)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
args.out = args.out.resolve()  # 相対パスだと Blender が別の場所に書くため
args.out.mkdir(parents=True, exist_ok=True)

scene = bpy.context.scene
c = scene.cycles
scene.render.engine = "CYCLES"
vs = scene.view_settings
vs.view_transform, vs.look, vs.gamma = "Standard", "None", 1.0
vs.exposure = math.log2(args.exposure_k)      # 線形値に k を掛けるのと同じ
for k, v in TEMPLATE.items():
    setattr(c, k, v)
c.use_adaptive_sampling, c.samples, c.seed = False, args.samples, 0
scene.render.resolution_x = scene.render.resolution_y = args.size
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
for o in scene.objects:
    if o.type == "LIGHT":
        o.hide_render = True

# 照明: 本番の render_stimulus.py と同じつなぎ方で回転させる
tree = scene.world.node_tree
env = next(n for n in tree.nodes if n.type == "TEX_ENVIRONMENT")
env.image = bpy.data.images.load(str(args.hdri), check_existing=True)
mapping = next((n for n in tree.nodes if n.type == "MAPPING"), None)
if mapping is None or not env.inputs["Vector"].is_linked:
    tc = tree.nodes.new("ShaderNodeTexCoord")
    mapping = mapping or tree.nodes.new("ShaderNodeMapping")
    tree.links.new(tc.outputs["Generated"], mapping.inputs["Vector"])
    tree.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
mapping.inputs["Rotation"].default_value[2] = math.radians(args.rotation)

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


def render_views(stem):
    for az in args.azimuths:
        phi = phi0 + math.radians(az)   # 上から見て反時計回りが正
        cam.location = (radius * math.cos(phi), radius * math.sin(phi), z0)
        scene.render.filepath = str(args.out / f"{stem}_az{int(az):03d}.png")
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        print(f"[gt] {stem} az {az:.0f}: {time.perf_counter() - t0:.0f}s", flush=True)


if args.opaque:
    mtree = mat.node_tree
    for link in list(mtree.links):
        mtree.links.remove(link)
    bsdf = mtree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = nodes["BSDF_GLASS"].inputs["Roughness"].default_value
    bsdf.inputs["IOR"].default_value = nodes["BSDF_GLASS"].inputs["IOR"].default_value
    mtree.links.new(bsdf.outputs["BSDF"], nodes["OUTPUT_MATERIAL"].inputs["Surface"])
    for item in args.opaque:
        name, albedo = item.split("=")
        bsdf.inputs["Base Color"].default_value = (float(albedo),) * 3 + (1.0,)
        render_views(f"gt_opaque_{name}")
else:
    for coef in args.coefs:
        scatter, absorb = PARAM[coef]
        nodes["VOLUME_SCATTER"].inputs["Density"].default_value = scatter
        nodes["VOLUME_ABSORPTION"].inputs["Density"].default_value = absorb
        render_views(f"gt_coef{coef:03d}")
