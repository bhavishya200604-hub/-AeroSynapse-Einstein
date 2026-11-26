"""Network architectures for PINNs (Step 4.1) behind one factory :func:`build_network`.

All networks map named input columns X (N, d) -> outputs (N, out_dim) (1 in 1D; 3-4 for the 2D formulations)
with the same FIXED input scaling to [-1, 1] as the baseline
(bounds from the problem definition), so Trainer, losses and metrics are architecture-agnostic.

    mlp           baseline fully connected network (aerosynapse.thermal.pinn.networks.MLP)
    fourier       Fourier-feature embedding gamma(z) = [sin(2 pi B z), cos(2 pi B z)], B ~ N(0, sigma^2) (fixed or
                  trainable) followed by an MLP; sigma is a number or one number per input column
    modified_mlp  two encoders U, V and gated hidden layers H_{k+1} = (1 - Z_k) * U + Z_k * V
                  (Wang, Teng, Perdikaris 2021; from memory, UNVERIFIED in docs/LITERATURE.md)
    resnet        input embedding + residual blocks h <- h + act(L2 act(L1 h))  (block_layers = 1 or 2)
    rwf           random weight factorisation W = diag(exp(s)) V with s ~ N(mu, sigma), V = W0 / exp(s)  (Wang, Wang,
                  Perdikaris 2022; from memory, UNVERIFIED)
    siren         sine-activated network with frequency omega0 and the Sitzmann et al. initialisation (UNVERIFIED)
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from aerosynapse.thermal.pinn.networks import ACT, MLP, Sin

ARCHS = ["mlp", "fourier", "modified_mlp", "resnet", "rwf", "siren"]


def _act(name: str) -> nn.Module:
    return Sin() if name == "sin" else ACT[name]()


def _init_linear(m: nn.Linear, init: str, activation: str) -> None:
    if init == "xavier":
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("tanh") if activation == "tanh" else 1.0)
    else:
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    nn.init.zeros_(m.bias)


class _Scaled(nn.Module):
    """Base: fixed input scaling to [-1, 1] by problem-defined bounds."""

    def __init__(self, lo: list[float], hi: list[float]):
        super().__init__()
        if len(lo) != len(hi):
            raise ValueError("lo/hi length mismatch")
        self.register_buffer("lo", torch.tensor(lo))
        self.register_buffer("hi", torch.tensor(hi))

    def scale(self, X: torch.Tensor) -> torch.Tensor:
        return 2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _body(in_dim: int, width: int, depth: int, activation: str, init: str, out_dim: int = 1) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, width), _act(activation)]
        d = width
    layers.append(nn.Linear(d, out_dim))
    seq = nn.Sequential(*layers)
    for m in seq:
        if isinstance(m, nn.Linear):
            _init_linear(m, init, activation)
    return seq


class FourierMLP(_Scaled):
    def __init__(
        self,
        lo: list[float],
        hi: list[float],
        width: int = 32,
        depth: int = 3,
        activation: str = "tanh",
        init: str = "xavier",
        sigma: float | list[float] = 1.0,
        n_features: int = 32,
        trainable: bool = False,
        include_input: bool = False,
        out_dim: int = 1,
    ):
        super().__init__(lo, hi)
        d = len(lo)
        sig = torch.as_tensor(sigma, dtype=torch.get_default_dtype()).reshape(-1)
        if sig.numel() == 1:
            sig = sig.expand(d)
        if sig.numel() != d or bool((sig <= 0).any()):
            raise ValueError(f"sigma must be a positive number or a list of {d} positive numbers")
        B = torch.randn(n_features, d) * sig
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)
        self.include_input = include_input
        self.body = _body(2 * n_features + (d if include_input else 0), width, depth, activation, init, out_dim)

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        z = self.scale(X)
        proj = 2.0 * math.pi * z @ self.B.T
        feats = [torch.sin(proj), torch.cos(proj)]
        if self.include_input:
            feats.append(z)
        return torch.cat(feats, dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.body(self.embed(X))


class ModifiedMLP(_Scaled):
    """H1 = act(W z); U = act(Wu z), V = act(Wv z); Z_k = act(W_k H_k); H_{k+1} = (1 - Z_k) U + Z_k V; out = W_out H."""

    def __init__(
        self,
        lo: list[float],
        hi: list[float],
        width: int = 32,
        depth: int = 3,
        activation: str = "tanh",
        init: str = "xavier",
        out_dim: int = 1,
    ):
        super().__init__(lo, hi)
        d = len(lo)
        self.enc_u, self.enc_v, self.first = nn.Linear(d, width), nn.Linear(d, width), nn.Linear(d, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, out_dim)
        self.act = _act(activation)
        for m in [self.enc_u, self.enc_v, self.first, *self.hidden, self.out]:
            _init_linear(m, init, activation)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z = self.scale(X)
        U, V = self.act(self.enc_u(z)), self.act(self.enc_v(z))
        H = self.act(self.first(z))
        for lin in self.hidden:
            Zk = self.act(lin(H))
            H = (1.0 - Zk) * U + Zk * V
        return self.out(H)


class ResNet(_Scaled):
    def __init__(
        self,
        lo: list[float],
        hi: list[float],
        width: int = 32,
        depth: int = 3,
        activation: str = "tanh",
        init: str = "xavier",
        block_layers: int = 2,
        out_dim: int = 1,
    ):
        super().__init__(lo, hi)
        if block_layers not in (1, 2):
            raise ValueError("block_layers must be 1 or 2")
        d = len(lo)
        self.embed = nn.Linear(d, width)
        self.blocks = nn.ModuleList(
            [nn.ModuleList([nn.Linear(width, width) for _ in range(block_layers)]) for _ in range(depth)]
        )
        self.out = nn.Linear(width, out_dim)
        self.act = _act(activation)
        for m in [self.embed, *[layer for blk in self.blocks for layer in blk], self.out]:
            _init_linear(m, init, activation)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = self.act(self.embed(self.scale(X)))
        for blk in self.blocks:
            r = h
            for i, lin in enumerate(blk):
                r = lin(r)
                r = self.act(r) if i < len(blk) - 1 else r
            h = h + self.act(r)
        return self.out(h)


class RWFLinear(nn.Module):
    """Linear layer with weight W = exp(s)[:, None] * V (per-output-neuron scale s)."""

    def __init__(self, in_dim: int, out_dim: int, init: str, activation: str, mu: float, sigma: float):
        super().__init__()
        W0 = torch.empty(out_dim, in_dim)
        if init == "xavier":
            nn.init.xavier_normal_(W0, gain=nn.init.calculate_gain("tanh") if activation == "tanh" else 1.0)
        else:
            nn.init.kaiming_normal_(W0, nonlinearity="relu")
        s = mu + sigma * torch.randn(out_dim)
        self.s = nn.Parameter(s)
        self.V = nn.Parameter(W0 / torch.exp(s)[:, None])
        self.b = nn.Parameter(torch.zeros(out_dim))

    def weight(self) -> torch.Tensor:
        return torch.exp(self.s)[:, None] * self.V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight().T + self.b


class RWFMLP(_Scaled):
    def __init__(
        self,
        lo: list[float],
        hi: list[float],
        width: int = 32,
        depth: int = 3,
        activation: str = "tanh",
        init: str = "xavier",
        mu: float = 1.0,
        sigma: float = 0.1,
        out_dim: int = 1,
    ):
        super().__init__(lo, hi)
        dims = [len(lo)] + [width] * depth + [out_dim]
        self.layers = nn.ModuleList(
            [RWFLinear(a, b, init, activation, mu, sigma) for a, b in zip(dims[:-1], dims[1:], strict=True)]
        )
        self.act = _act(activation)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = self.scale(X)
        for lin in self.layers[:-1]:
            h = self.act(lin(h))
        return self.layers[-1](h)


class Siren(_Scaled):
    """sin(omega0 (W z + b)) layers; first layer W ~ U(-1/in, 1/in), others U(-sqrt(6/in)/omega0, +...)."""

    def __init__(
        self, lo: list[float], hi: list[float], width: int = 32, depth: int = 3, omega0: float = 10.0, out_dim: int = 1
    ):
        super().__init__(lo, hi)
        if omega0 <= 0:
            raise ValueError("omega0 must be > 0")
        self.omega0 = float(omega0)
        d = len(lo)
        dims = [d] + [width] * depth
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:], strict=True)])
        self.out = nn.Linear(width, out_dim)
        with torch.no_grad():
            for i, lin in enumerate(self.layers):
                bound = 1.0 / lin.in_features if i == 0 else math.sqrt(6.0 / lin.in_features) / self.omega0
                lin.weight.uniform_(-bound, bound)
            self.out.weight.uniform_(-math.sqrt(6.0 / width) / self.omega0, math.sqrt(6.0 / width) / self.omega0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = self.scale(X)
        for lin in self.layers:
            h = torch.sin(self.omega0 * lin(h))
        return self.out(h)


def build_network(cfg: dict[str, Any], lo: list[float], hi: list[float], out_dim: int = 1) -> nn.Module:
    """Create the network described by a ``network`` spec block (arch, width, depth, activation, init, sub-blocks)."""
    arch = cfg.get("arch", "mlp")
    w, d, act, init = cfg["width"], cfg["depth"], cfg["activation"], cfg["init"]
    if arch == "mlp":
        return MLP(lo, hi, w, d, act, init, out_dim, float(cfg.get("dropout", 0.0)))
    if float(cfg.get("dropout", 0.0)) > 0:
        raise ValueError(f"network.dropout > 0 is implemented for arch mlp only (got {arch!r})")
    if arch == "fourier":
        f = cfg["fourier"]
        return FourierMLP(
            lo, hi, w, d, act, init, f["sigma"], f["n_features"], f["trainable"], f["include_input"], out_dim
        )
    if arch == "modified_mlp":
        return ModifiedMLP(lo, hi, w, d, act, init, out_dim)
    if arch == "resnet":
        return ResNet(lo, hi, w, d, act, init, cfg["resnet"]["block_layers"], out_dim)
    if arch == "rwf":
        return RWFMLP(lo, hi, w, d, act, init, cfg["rwf"]["mu"], cfg["rwf"]["sigma"], out_dim)
    if arch == "siren":
        return Siren(lo, hi, w, d, cfg["siren"]["omega0"], out_dim)
    raise ValueError(f"unknown architecture {arch!r}; available: {ARCHS}")
