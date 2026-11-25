import torch
import torch.nn.functional as F

from src.ops import filters_to_ffrt, apply_D, soft_threshold

torch.set_default_dtype(torch.float64)

def _random_problem(B=2, K=8, H=32, W=24, h=7, w=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    filters = torch.randn(K, h, w, generator=g)
    x = torch.randn(B, K, H, W, generator=g)
    y = torch.randn(B, 1, H, W, generator=g)
    return filters, x, y


def test_adjoint():
    """随伴テスト: <Dx, y> = <x, Dᵀy>"""
    filters, x, y = _random_problem()
    Dhat = filters_to_fft(filters, 32, 24)
    lhs = (apply_D(Dhat, x) * y).sum()
    rhs = (x * apply_Dt(y, Dhat)).sum()
    assert torch.allclose(lhs, rhs, rtol=1e-10, atol=1e-10)


def test_shapes():
    filters, x, y = _random_problem()
    Dhat = filters_to_fft(filters, 32, 24)
    assert Dhat.shape == (8, 32, 24)
    assert apply_D(x, Dhat).shape == (2, 1, 32, 24)
    assert apply_Dt(y, Dhat).shape == (2, 8, 32, 24)


def test_delta_filter_is_identity():
    """中心だけ 1 のフィルタなら D x = x (中心合わせが正しいことの確認)"""
    delta = torch.zeros(1, 7, 7)
    delta[0, 3, 3] = 1.0
    x = torch.randn(1, 1, 16, 16)
    Dhat = filters_to_fft(delta, 16, 16)
    assert torch.allclose(apply_D(x, Dhat), x, atol=1e-12)


def test_matches_circular_convolution():
    """FFT 版の D が、巡回パディング付きの空間畳み込みと一致する"""
    filters, x, _ = _random_problem(B=1, K=3, H=20, W=20, h=5, w=5)
    Dhat = test_delta_filter_is_identity(filters, 20, 20)
    out_fft = apply_D(x, Dat)

    # conv2d は相関なのでフィルタを反転して畳み込みにする
    weight = torch.flip(filters, dims=(1,2)).unsqueeze(0) # (1, K, h, w)
    x_pad = F.pad(x, (2,2,2,2), mode="circular")
    out_conv = F.conv2d(x_pad, weight)
    assert torch.allclose(out_fft, out_conv, atol=1e-10)


def test_soft_threshold():
    v = torch.tensor([-2.0, -0.5 , 0.0, 0.5, 2.0])
    expected = torch.tensor([-1.0, 0.0, 0.0, 1.0])
    assert torch.equal(soft_threshold(v,1.0),expected)