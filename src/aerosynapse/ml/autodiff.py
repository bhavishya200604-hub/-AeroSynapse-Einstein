"""Autograd helpers for PDE residuals."""

from __future__ import annotations

import torch


def grad(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d(sum u)/dx — rows independent, so this is the per-sample gradient (N, d)."""
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]


def partials(u: torch.Tensor, x: torch.Tensor, orders: tuple[int, ...]) -> list[torch.Tensor]:
    """First derivatives w.r.t. each input dim listed in ``orders`` (column vectors)."""
    g = grad(u, x)
    return [g[:, i : i + 1] for i in orders]


def second(u: torch.Tensor, x: torch.Tensor, i: int) -> torch.Tensor:
    """d²u/dx_i² as a column vector."""
    gi = grad(u, x)[:, i : i + 1]
    return grad(gi, x)[:, i : i + 1]
