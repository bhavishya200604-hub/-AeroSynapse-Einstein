"""Baseline network: fully connected MLP with fixed input scaling to [-1, 1].

Inputs are named columns (e.g. x, t, logp); the bounds of each column are fixed by the PROBLEM definition (not learned,
not data-dependent), so the network sees inputs in [-1, 1].
"""

from __future__ import annotations

import os

import torch
from torch import nn

ACT = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}


class Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def resolve_device(name: str) -> torch.device:
    # PINN_DEVICE overrides "auto" (D150): on a ZeroGPU Space `import spaces` reports CUDA as available everywhere,
    # but a GPU is attached only inside @spaces.GPU calls, so THERMAL LAB keeps every network on the CPU there
    if name == "auto" and os.environ.get("PINN_DEVICE"):
        return torch.device(os.environ["PINN_DEVICE"])
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class MCDropout(nn.Dropout):
    """nn.Dropout (one mask per point while training) that can hold ONE fixed mask for every input: with ``mask`` set,
    the network is a single sampled function (MC-dropout prediction, Gal & Ghahramani 2016); mask None = nn.Dropout."""

    def __init__(self, p: float):
        super().__init__(p)
        self.mask: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask if self.mask is not None else super().forward(x)


def sample_dropout_masks(model: nn.Module, gen: torch.Generator | None = None) -> int:
    """Draw one fixed mask per MCDropout layer (the k-th dropout follows the k-th Linear layer; mask length = its
    out_features, kept units scaled 1/(1-p)). Returns the number of layers masked."""
    lin = [x for x in model.modules() if isinstance(x, nn.Linear)]
    drops = [x for x in model.modules() if isinstance(x, MCDropout)]
    for m, ref in zip(drops, lin, strict=False):
        keep = torch.rand(ref.out_features, generator=gen, dtype=ref.weight.dtype) >= m.p
        m.mask = keep.to(ref.weight.dtype) / (1.0 - m.p)
    return len(drops)


def clear_dropout_masks(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, MCDropout):
            m.mask = None


class MLP(nn.Module):
    """in_dim -> [width]*depth -> out_dim, smooth activation (needed for second derivatives)."""

    def __init__(
        self,
        lo: list[float],
        hi: list[float],
        width: int = 32,
        depth: int = 3,
        activation: str = "tanh",
        init: str = "xavier",
        out_dim: int = 1,
        dropout: float = 0.0,
    ):
        """dropout > 0: nn.Dropout after every hidden activation (Step 6.2 MC-dropout); 0 = the plain MLP."""
        super().__init__()
        if len(lo) != len(hi):
            raise ValueError("lo/hi length mismatch")
        self.register_buffer("lo", torch.tensor(lo))
        self.register_buffer("hi", torch.tensor(hi))
        act = Sin if activation == "sin" else ACT[activation]
        layers: list[nn.Module] = []
        d = len(lo)
        for _ in range(depth):
            layers += [nn.Linear(d, width), act()] + ([MCDropout(dropout)] if dropout > 0 else [])
            d = width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                if init == "xavier":
                    nn.init.xavier_normal_(
                        m.weight, gain=nn.init.calculate_gain("tanh") if activation == "tanh" else 1.0
                    )
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z = 2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0
        return self.net(z)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
