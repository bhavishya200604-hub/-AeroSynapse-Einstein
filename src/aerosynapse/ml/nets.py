r"""Network building blocks: MLP, random Fourier features (Tancik et al. 2020), periodic embedding.

Inputs are affinely normalised to [-1, 1] using the problem bounds before the first layer.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

ACTIVATIONS = {"tanh": nn.Tanh, "sin": None, "gelu": nn.GELU, "silu": nn.SiLU}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch; enable deterministic algorithms where available."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds legacy global RNG used by third-party code
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class FourierFeatures(nn.Module):
    r""":math:`\gamma(x) = [\cos(2\pi Bx), \sin(2\pi Bx)]`, :math:`B_{ij}\sim N(0,\sigma^2)` (fixed)."""

    B: torch.Tensor

    def __init__(self, in_dim: int, n_features: int, scale: float, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(in_dim, n_features, generator=g) * scale)
        self.out_dim = 2 * n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = 2 * math.pi * x @ self.B
        return torch.cat([torch.cos(z), torch.sin(z)], dim=-1)


class MLP(nn.Module):
    """Fully connected network with optional Fourier features and periodic embedding of input dims.

    Parameters
    ----------
    lo, hi
        Input bounds used for normalisation to [-1, 1].
    periodic_dims
        Input indices embedded as (cos 2πx/L, sin 2πx/L) — enforces periodicity of u and all its
        derivatives exactly (hard periodic BC).
    """

    lo: torch.Tensor
    hi: torch.Tensor

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        width: int = 32,
        depth: int = 3,
        activation: str = "tanh",
        lo: Sequence[float] | None = None,
        hi: Sequence[float] | None = None,
        fourier_features: int = 0,
        fourier_scale: float = 1.0,
        periodic_dims: Sequence[int] = (),
        seed: int = 0,
    ) -> None:
        super().__init__()
        lo_t = torch.tensor(lo if lo is not None else [-1.0] * in_dim, dtype=torch.get_default_dtype())
        hi_t = torch.tensor(hi if hi is not None else [1.0] * in_dim, dtype=torch.get_default_dtype())
        self.register_buffer("lo", lo_t)
        self.register_buffer("hi", hi_t)
        self.periodic_dims = list(periodic_dims)
        d = in_dim + len(self.periodic_dims)
        self.ff = FourierFeatures(d, fourier_features, fourier_scale, seed) if fourier_features else None
        d = self.ff.out_dim if self.ff is not None else d
        act = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.Linear(d, width), (act() if act is not None else Sin())]
            d = width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        z = 2 * (x - self.lo) / (self.hi - self.lo) - 1
        if self.periodic_dims:
            extra = []
            for i in self.periodic_dims:
                ang = math.pi * (z[:, i : i + 1] + 1)  # one period over [lo, hi]
                extra += [torch.cos(ang), torch.sin(ang)]
            keep = [z[:, j : j + 1] for j in range(z.shape[1]) if j not in self.periodic_dims]
            z = torch.cat(extra + keep, dim=1)
        return self.ff(z) if self.ff is not None else z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.embed(x))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
