"""Generic supervised training loop for field-to-field operators (FNO, CNN, MLP) and DeepONet."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
from torch import nn


def rel_l2_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, y.ndim))
    return torch.mean(torch.sqrt(((pred - y) ** 2).sum(dims)) / torch.sqrt((y**2).sum(dims)).clamp_min(1e-12))


def train_operator(
    model: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-3,
    batch: int = 32,
    seed: int = 0,
    forward: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[nn.Module, list[float], float]:
    """Adam + cosine decay on the relative-L2 loss. Returns (model, loss history, wall seconds)."""
    torch.manual_seed(seed)
    dt = torch.get_default_dtype()
    Xt, Yt = torch.tensor(X, dtype=dt), torch.tensor(Y, dtype=dt)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    fwd = forward or (lambda m, x: m(x))
    g = torch.Generator().manual_seed(seed)
    hist = []
    t0 = time.perf_counter()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for i in range(0, len(Xt), batch):
            idx = perm[i : i + batch]
            loss = rel_l2_loss(fwd(model, Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        hist.append(tot / len(Xt))
    return model, hist, time.perf_counter() - t0


def predict_operator(
    model: nn.Module, X: np.ndarray, forward: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None
) -> np.ndarray:
    fwd = forward or (lambda m, x: m(x))
    with torch.no_grad():
        return fwd(model, torch.tensor(X, dtype=torch.get_default_dtype())).numpy()


def per_sample_rel_l2(pred: np.ndarray, Y: np.ndarray) -> np.ndarray:
    ax = tuple(range(1, Y.ndim))
    return np.asarray(np.sqrt(((pred - Y) ** 2).sum(ax)) / np.sqrt((Y**2).sum(ax)))
