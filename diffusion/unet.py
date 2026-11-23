"""画像用の eps_theta(x_t, t)。小さな U-Net。

U-Net を使う理由: 出力（ノイズ）は入力と同じ解像度が必要で、
かつ粗い構造（低解像度）と細部（高解像度）の両方を見る必要があるため。
時刻 t は各 ResBlock に足し込み、「今どのくらいノイズが乗っているか」を伝える。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("freq", torch.exp(-math.log(10000) * torch.arange(dim // 2) / (dim // 2)))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        a = t.float()[:, None] * self.freq[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], dim=1))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(temb)[:, :, None, None]  # 時刻をチャンネルごとの加算で注入
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """32x32 入力。解像度 32 -> 16 -> 8 と下げ、スキップ接続つきで 8 -> 16 -> 32 と戻す。"""

    def __init__(self, ch=1, base=32, tdim=128, n_classes=0):
        super().__init__()
        # クラス条件: ラベルの埋め込みを時刻の埋め込みに足す。番号 n_classes は「条件なし」を表す
        self.n_classes = n_classes
        if n_classes:
            self.label = nn.Embedding(n_classes + 1, tdim)
        c1, c2, c3 = base, base * 2, base * 4
        self.temb = TimeEmbedding(tdim)
        self.inp = nn.Conv2d(ch, c1, 3, padding=1)
        self.d1 = ResBlock(c1, c1, tdim)
        self.down1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.d2 = ResBlock(c1, c2, tdim)
        self.down2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)
        self.mid1 = ResBlock(c2, c3, tdim)
        self.mid2 = ResBlock(c3, c3, tdim)
        self.u2 = ResBlock(c3 + c2, c2, tdim)
        self.u1 = ResBlock(c2 + c1, c1, tdim)
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, ch, 3, padding=1))

    def forward(self, x, t, y=None):
        temb = self.temb(t)
        if self.n_classes:
            if y is None:
                y = torch.full((x.shape[0],), self.n_classes, device=x.device)
            temb = temb + self.label(y)
        h1 = self.d1(self.inp(x), temb)                 # 32x32
        h2 = self.d2(self.down1(h1), temb)              # 16x16
        h = self.mid2(self.mid1(self.down2(h2), temb), temb)  # 8x8
        h = self.u2(torch.cat([F.interpolate(h, scale_factor=2.0), h2], dim=1), temb)
        h = self.u1(torch.cat([F.interpolate(h, scale_factor=2.0), h1], dim=1), temb)
        return self.out(h)
