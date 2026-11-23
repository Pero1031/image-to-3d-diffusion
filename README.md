# 単一画像からの3D生成と拡散モデル

学習用フォルダ。

## 流れの目安

1. 拡散モデルの基礎: DDPM（ノイズ付加と除去、ε予測）、DDIM（決定的サンプリング）
2. 条件付き生成: classifier-free guidance、潜在拡散（Stable Diffusion）
3. 2D 拡散を 3D に使う: Score Distillation Sampling（DreamFusion）
4. 視点を条件にした拡散: Zero-1-to-3、多視点同時生成（MVDream など）
5. 拡散を使わない feed-forward 再構成: LRM 系、3DGS を直接出す手法

nerf-toy / 3dgs-toy の表現（NeRF・3DGS）を 3 と 4 の出力側として使える。
