"""Baselines: a plain supervised network with the same architecture as the PINN (spec rule 4)."""

from __future__ import annotations

import time

import numpy as np
import torch

from aerosynapse.ml.nets import MLP


def train_supervised(
    X: np.ndarray,
    y: np.ndarray,
    lo: list[float],
    hi: list[float],
    width: int = 32,
    depth: int = 3,
    iterations: int = 2000,
    lr: float = 1e-3,
    seed: int = 0,
    periodic_dims: tuple[int, ...] = (),
) -> tuple[MLP, float]:
    """MSE regression on (X, y) — the 'data-only' baseline that sees exact labels but no physics."""
    torch.manual_seed(seed)
    m = MLP(X.shape[1], 1, width, depth, lo=lo, hi=hi, periodic_dims=periodic_dims, seed=seed)
    dt = torch.get_default_dtype()
    Xt, yt = torch.tensor(X, dtype=dt), torch.tensor(y.reshape(-1, 1), dtype=dt)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    t0 = time.perf_counter()
    for _ in range(iterations):
        loss = torch.mean((m(Xt) - yt) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m, time.perf_counter() - t0
