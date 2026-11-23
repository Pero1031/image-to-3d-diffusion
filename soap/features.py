"""石けんの写真 122 枚から、画像統計とモデルの内部の特徴を取り出す。

    python soap/features.py

入力は 2 通り: bg = 写真そのまま（背景あり）、nobg = 物体だけを白い背景に置いたもの（背景なし）。
どちらも、空間のある特徴は物体の領域の中だけで平均して 1 本のベクトルにする。
出力: data/soap_features.npz（特徴の名前 -> (122, 次元) の配列）
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "image_to_3d"))
IMAGES = Path("D:/project/vision-experiments/material_aoi_pilot/images")
MASKS = ROOT / "data" / "soap_masks" / "final"
LAB_STATS = Path("D:/project/vision-experiments/image_stats_pilot/image_stats.py")
device = "cuda"
T_LEVELS = (100, 500, 900)        # U-Net の特徴を取るノイズの段階（小さいほどノイズが少ない）


def prepare(name, variant, size):
    """(RGB 画像, 物体の領域) を size x size で返す。"""
    im = Image.open(IMAGES / name).convert("RGB")
    mask = Image.open(MASKS / (Path(name).stem + ".png"))
    if variant == "nobg":       # 物体を切り出し、画像の 85% を占めるように白い背景へ置く
        a = np.asarray(mask) > 0
        ys, xs = np.where(a)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        side = int(max(y1 - y0, x1 - x0) / 0.85)
        oy, ox = (side - (y1 - y0)) // 2, (side - (x1 - x0)) // 2
        rgb = np.full((side, side, 3), 255, np.uint8)
        m = np.zeros((side, side), np.uint8)
        rgb[oy:oy + y1 - y0, ox:ox + x1 - x0][a[y0:y1, x0:x1]] = np.asarray(im)[y0:y1, x0:x1][a[y0:y1, x0:x1]]
        m[oy:oy + y1 - y0, ox:ox + x1 - x0] = a[y0:y1, x0:x1] * 255
        im, mask = Image.fromarray(rgb), Image.fromarray(m)
    return im.resize((size, size), Image.LANCZOS), np.asarray(mask.resize((size, size), Image.BILINEAR)) > 127


def masked_mean(feat, mask):
    """feat: (C, h, w)。mask を特徴の解像度に縮め、領域内で平均する。"""
    w = F.interpolate(torch.from_numpy(mask).float()[None, None].to(feat.device), size=feat.shape[-2:], mode="area")[0, 0]
    return ((feat * w).sum((1, 2)) / w.sum().clamp_min(1e-6)).float().cpu().numpy()


def main():
    names = sorted(p.name for p in IMAGES.glob("*.jpg"))
    feats = {}

    # --- 画像統計（研究室の 14 種）と色。物体の領域内だけで計算するので背景の有無によらない ---
    spec = importlib.util.spec_from_file_location("lab_image_stats", LAB_STATS)
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)
    stat_rows, color_rows, stat_names = [], [], None
    for name in names:
        mask = np.asarray(Image.open(MASKS / (Path(name).stem + ".png"))) > 0
        st = lab.compute_image_stats(IMAGES / name, mask)
        st = {k: v for k, v in st.items() if k not in ("n_pixels", "subband_levels_used")}
        stat_names = stat_names or list(st)
        stat_rows.append([st[k] for k in stat_names])
        px = np.asarray(Image.open(IMAGES / name).convert("RGB"), np.float32)[mask] / 255.0
        color_rows.append(np.concatenate([px.mean(0), px.std(0)]))
    feats["stats14"], feats["color"] = np.array(stat_rows, np.float32), np.array(color_rows, np.float32)
    print("image statistics:", stat_names, flush=True)

    # --- Zero-1-to-3: 画像の読み取り部分（CLIP）と U-Net の中間層 ---
    import run_zero123 as z   # モデルを読み込む。生成のループは実行されない
    captured = {}
    hooks = [z.unet.mid_block.register_forward_hook(lambda m, i, o: captured.__setitem__("mid", o)),
             z.unet.up_blocks[1].register_forward_hook(lambda m, i, o: captured.__setitem__("up1", o)),
             z.unet.up_blocks[2].register_forward_hook(lambda m, i, o: captured.__setitem__("up2", o))]
    z.sched.set_timesteps(1000)
    for variant in ("bg", "nobg"):
        clip_rows, unet_rows = [], {t: [] for t in T_LEVELS}
        for name in names:
            im, mask = prepare(name, variant, 256)
            x = torch.from_numpy(np.asarray(im)).float().div(127.5).sub(1).permute(2, 0, 1)[None].to(device)
            with torch.no_grad():
                c = F.interpolate(x * 0.5 + 0.5, size=224, mode="bicubic", antialias=True)
                emb = z.clip((c - z.CLIP_MEAN) / z.CLIP_STD).image_embeds
                clip_rows.append(emb[0].float().cpu().numpy())
                # 視点を変えない条件（回転 0 度）で、入力画像そのものにノイズを足して U-Net に通す
                cam = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)
                ctx = z.proj(torch.cat([emb, cam], dim=1))[:, None]
                dist = z.vae.encode(x).latent_dist
                cat, z0 = dist.mode(), dist.mode() * z.vae.config.scaling_factor
                noise = torch.randn(z0.shape, device=device, generator=torch.Generator(device).manual_seed(0))
                for t in T_LEVELS:
                    tt = torch.tensor([t], device=device)
                    zt = z.sched.add_noise(z0, noise, tt)
                    z.unet(torch.cat([zt, cat], 1), tt, encoder_hidden_states=ctx)
                    unet_rows[t].append(np.concatenate([masked_mean(captured[k][0], mask) for k in ("mid", "up1", "up2")]))
        feats[f"zero123_clip_{variant}"] = np.array(clip_rows)
        for t in T_LEVELS:
            feats[f"zero123_unet_t{t}_{variant}"] = np.array(unet_rows[t])
        print("zero123", variant, {k: v.shape for k, v in feats.items() if k.endswith(variant)}, flush=True)
    for h in hooks:
        h.remove()
    del z.unet, z.vae, z.clip
    torch.cuda.empty_cache()

    # --- TripoSR: 画像の読み取り部分（DINO）。パッチごとの特徴を平均する ---
    for mod in ("torchmcubes", "rembg"):
        sys.modules.setdefault(mod, types.ModuleType(mod))
    sys.modules["torchmcubes"].marching_cubes = None
    sys.path.insert(0, str(ROOT / "data/third_party/TripoSR"))
    from tsr.system import TSR
    tsr = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt").to(device).eval()
    for variant in ("bg", "nobg"):
        rows = []
        for name in names:
            im, _ = prepare(name, variant, 512)
            x = torch.from_numpy(np.asarray(im)).float().div(255).permute(2, 0, 1)[None, None].to(device)
            with torch.no_grad():
                tok = tsr.image_tokenizer(x)          # (1, 1, C, トークン数)
            rows.append(tok[0, 0].mean(-1).float().cpu().numpy())
        feats[f"triposr_dino_{variant}"] = np.array(rows)
        print("triposr", variant, feats[f"triposr_dino_{variant}"].shape, flush=True)

    np.savez(ROOT / "data" / "soap_features.npz", names=np.array(names), stat_names=np.array(stat_names), **feats)
    print("saved", len(feats), "feature sets")


if __name__ == "__main__":
    main()
