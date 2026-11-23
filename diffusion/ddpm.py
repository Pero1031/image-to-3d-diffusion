"""DDPM (Ho et al., 2020) の中核。データの形によらず使える。

前向き過程  q(x_t | x_0) = N( sqrt(abar_t) x_0, (1 - abar_t) I )
学習        eps を当てる:  L = E || eps - eps_theta(x_t, t) ||^2
逆向き過程  x_{t-1} = (x_t - beta_t / sqrt(1 - abar_t) * eps_theta) / sqrt(alpha_t) + sigma_t z
"""
import torch


class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        # beta_t: 各ステップで足すノイズの分散。論文と同じ線形スケジュール。
        self.beta = torch.linspace(beta_start, beta_end, T, device=device)
        self.alpha = 1.0 - self.beta
        # abar_t = alpha_1 * ... * alpha_t: x_0 の信号が t 時点でどれだけ残っているか（2乗の比率）
        self.abar = torch.cumprod(self.alpha, dim=0)

    def _at(self, v, t, x):
        """長さ T の係数 v から t 番目を取り、x に掛けられる形にする。"""
        return v[t].view(-1, *([1] * (x.dim() - 1)))

    def q_sample(self, x0, t, eps):
        """前向き過程を 1 回で計算する。x_0 から x_t へ、t ステップ分のノイズを一度に足す。"""
        return self._at(self.abar, t, x0).sqrt() * x0 + (1 - self._at(self.abar, t, x0)).sqrt() * eps

    def loss(self, model, x0):
        """ランダムな時刻 t でノイズを足し、足したノイズ eps をモデルに当てさせる。"""
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        eps = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps)
        return ((eps - model(xt, t)) ** 2).mean()

    @torch.no_grad()
    def p_step(self, model, xt, t):
        """逆向き過程の 1 ステップ。x_t から x_{t-1} を作る。t は整数。"""
        tt = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
        eps = model(xt, tt)
        mean = (xt - self.beta[t] / (1 - self.abar[t]).sqrt() * eps) / self.alpha[t].sqrt()
        if t == 0:
            return mean  # 最後のステップではノイズを足さない
        return mean + self.beta[t].sqrt() * torch.randn_like(xt)  # sigma_t^2 = beta_t

    @torch.no_grad()
    def sample(self, model, shape, device, keep=()):
        """純粋なノイズ x_T から始めて T ステップ戻す。keep に入れた時刻の途中経過も返す。"""
        x = torch.randn(shape, device=device)
        traj = {self.T: x.cpu()} if self.T in keep else {}
        for t in reversed(range(self.T)):
            x = self.p_step(model, x, t)
            if t in keep:
                traj[t] = x.cpu()
        return x, traj
