"""
CSC で使う基本演算(FFT による畳み込み D とその随伴 Dᵀ、ソフトしきい値)

テンソルの形の約束:
    filters : (K, h, w)        実数。辞書フィルタ
    Dhat    : (K, H, W)        複素数。フィルタの FFT
    x       : (B, K, H, W)     実数。係数マップ
    s, y    : (B, 1, H, W)     実数。画像
"""

import torch


def filters_to_fft(filters:torch.Tensor, H: int, W: int) -> torch.Tensor:
    """フィルタを画像サイズまでゼロ埋めし、中心を原点に移してから FFT する。

    中心を原点 (0, 0) に移さないと、再構成画像が半フィルタ分ずれる。
    """
    K, h, w = filters.shape
    if h > H or w > W:
        raise ValueError("フィルタが画像より大きい")

    padded = filters.new_zeros(K, H, W)
    padded[:, :h, :w] = filters
    # フィルタの中心(h//2, w//2) を原点に移す
    padded = torch.roll(padded, shift=(-(h // 2), -(w // 2)), dims=(1, 2))
    return torch.fft.rfft2(padded)


def apply_D(x: torch.Tensor, Dhat:torch.Tensor) -> torch.Tensor:
    """D x = Σ_k d_k * x_k を計算する。 (B, K, H, W) -> (B, 1, H, W)"""
    xhat = torch.fft.fft2(x)
# (B, K, H, W)
    shat = (Dhat.unsqueeze(0) * xhat).sum(dim=1, keepdim=True) # K 方向に緩和
    return torch.fft.ifft2(shat).real



def apply_Dt(y: torch.Tensor, Dhat: torch.Tensor) -> torch.Tensor:
    """随伴 Dᵀ y を計算する。 (B, 1, H, W) -> (B, K, H, W)"""
    yhat = torch.fft.fft2(y)
# (B, 1, H, W)
    xhat = Dhat.conj().unsqueeze(0) * yhat
# 共役を掛ける(和はとらない)
    return torch.fft.ifft2(xhat).real


def soft_threshold(v: torch.Tensor, tau) -> torch.Tensor:
    """ソフトしきい値 S_τ(v) = sign(v) · max(|v| - τ, 0)"""
    return torch.sign(v) * torch.clamp(v.abs() - tau, min=0.0)