r"""Fourier Neural Operator (Li et al. 2021), written from scratch.

Layer: :math:`v_{l+1}(x)=\sigma\big(Wv_l(x)+\mathcal F^{-1}[R_\phi\cdot\mathcal F v_l](x)\big)`, keeping the
lowest ``modes`` Fourier modes. Because the learned kernel lives in Fourier space, the same weights can
be evaluated on a different grid resolution (resolution transfer — tested, not assumed).
"""

from __future__ import annotations

import torch
from torch import nn


class SpectralConv1d(nn.Module):
    def __init__(self, cin: int, cout: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (cin * cout)
        self.w = nn.Parameter(scale * torch.randn(cin, cout, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = torch.fft.rfft(x)
        m = min(self.modes, xf.shape[-1])
        out = torch.zeros(x.shape[0], self.w.shape[1], xf.shape[-1], dtype=torch.cfloat, device=x.device)
        out[..., :m] = torch.einsum("bix,iox->box", xf[..., :m], self.w[..., :m])
        return torch.fft.irfft(out, n=x.shape[-1])


class SpectralConv2d(nn.Module):
    def __init__(self, cin: int, cout: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(scale * torch.randn(cin, cout, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(cin, cout, modes, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, ny, nx = x.shape
        xf = torch.fft.rfft2(x)
        m1, m2 = min(self.modes, ny // 2), min(self.modes, xf.shape[-1])
        out = torch.zeros(b, self.w1.shape[1], ny, xf.shape[-1], dtype=torch.cfloat, device=x.device)
        out[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(ny, nx))


class FNO1d(nn.Module):
    """Input (B, C_in, n) → output (B, C_out, n). A grid coordinate channel is appended internally.

    ``conserve_mass=True`` gives a physics-aware variant: the output mean is set equal to the mean of
    input channel 0, which is exact for periodic conservative transport (∫u dx is invariant).
    """

    def __init__(
        self, cin: int, cout: int = 1, width: int = 32, modes: int = 12, layers: int = 4, conserve_mass: bool = False
    ) -> None:
        super().__init__()
        self.lift = nn.Linear(cin + 1, width)
        self.spec = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, cout))
        self.conserve_mass = conserve_mass

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        n = a.shape[-1]
        grid = torch.linspace(0, 1, n, device=a.device, dtype=a.dtype).expand(a.shape[0], 1, n)
        v = self.lift(torch.cat([a, grid], 1).transpose(1, 2)).transpose(1, 2)
        for i, (s, w) in enumerate(zip(self.spec, self.pw, strict=True)):
            v = s(v) + w(v)
            if i < len(self.spec) - 1:
                v = nn.functional.gelu(v)
        out = self.proj(v.transpose(1, 2)).transpose(1, 2)
        if self.conserve_mass:
            out = out - out.mean(-1, keepdim=True) + a[:, :1].mean(-1, keepdim=True)
        return out


class FNO2d(nn.Module):
    def __init__(self, cin: int, cout: int = 1, width: int = 24, modes: int = 8, layers: int = 4) -> None:
        super().__init__()
        self.lift = nn.Linear(cin + 2, width)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, cout))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        b, _, ny, nx = a.shape
        gy, gx = torch.meshgrid(
            torch.linspace(0, 1, ny, dtype=a.dtype), torch.linspace(0, 1, nx, dtype=a.dtype), indexing="ij"
        )
        grid = torch.stack([gx, gy]).expand(b, 2, ny, nx)
        v = self.lift(torch.cat([a, grid], 1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for i, (s, w) in enumerate(zip(self.spec, self.pw, strict=True)):
            v = s(v) + w(v)
            if i < len(self.spec) - 1:
                v = nn.functional.gelu(v)
        return self.proj(v.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
