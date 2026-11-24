"""視線と比べるために、石けんの写真（被験者が見たままの、背景ありの写真）からパッチごとの特徴を取り出す。

    HF_HOME=data/hf python soap/patch_features.py

出力: data/soap_patch_features.npz
    dino  (122, 32, 32, 768)   TripoSR の DINO のパッチ（512 x 512 に縮めた写真、1 パッチ = 元の 32 画素）
    clip  (122, 16, 16, 1024)  Zero-1-to-3 の CLIP の最終層のパッチ（224 x 224、1 パッチ = 元の 64 画素）
いずれも半精度で保存する。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "image_to_3d"))
from features import prepare  # noqa: E402  （soap/features.py。image_to_3d へのパスもここで通る）
from translucency_readout import Encoders  # noqa: E402


def main():
    data = np.load(ROOT / "data" / "soap_features.npz")
    names = list(data["names"])
    enc = Encoders()
    dino, clip = [], []
    for i, name in enumerate(names):
        im, _ = prepare(name, "bg", 512)
        emb, ct, dt = enc.tokens(im)
        if i < 3:   # 取り方が保存済みの特徴（Part 3）と同じか
            cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
            print(name, "cosine to saved: CLIP", round(cos(emb, data["zero123_clip_bg"][i]), 4),
                  "DINO", round(cos(dt.mean(0), data["triposr_dino_bg"][i]), 4), " tokens", ct.shape, dt.shape, flush=True)
        clip.append(ct[1:].reshape(16, 16, -1).astype(np.float16))
        dino.append(dt[1:].reshape(32, 32, -1).astype(np.float16))
    np.savez(ROOT / "data" / "soap_patch_features.npz", names=np.array(names), dino=np.array(dino), clip=np.array(clip))
    print("saved", np.array(dino).shape, np.array(clip).shape)


if __name__ == "__main__":
    main()
