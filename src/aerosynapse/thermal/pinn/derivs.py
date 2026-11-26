"""Autograd derivative utilities for u(x, [t], [p]) networks.

The model is evaluated on named input COLUMNS (leaf tensors); derivatives are taken with respect to the columns, so
du/dx, du/dt, d2u/dx2 are exact autograd derivatives (graph retained for training).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

Tensor = torch.Tensor


def grad(u: Tensor, x: Tensor, create_graph: bool = True) -> Tensor:
    """d(sum u)/dx for a column x (same shape as u); allow_unused -> zeros."""
    g = torch.autograd.grad(u.sum(), x, create_graph=create_graph, allow_unused=True)[0]
    return torch.zeros_like(x) if g is None else g


def evaluate(
    model: Callable[[Tensor], Tensor],
    cols: dict[str, Tensor],
    order: Sequence[str],
    first: Sequence[str] = ("x", "t"),
    second: Sequence[str] = ("x",),
    create_graph: bool = True,
) -> dict[str, Tensor]:
    """u and requested derivatives. Keys: 'u', 'd_<n>' (first), 'dd_<n>' (second), n a column name.

    cols: name -> (N,) tensor; the columns in ``first``/``second`` are made leaf tensors with requires_grad.
    order: column order fed to the model (model input = stack(order, dim=1)).
    """
    c = {k: v for k, v in cols.items()}
    need = set(first) | set(second)
    for k in need:
        if k not in c:
            raise KeyError(f"column {k!r} not provided")
        c[k] = c[k].detach().clone().requires_grad_(True)
    X = torch.stack([c[k] for k in order], dim=1)
    u = model(X).squeeze(-1)
    out = {"u": u}
    firsts = {}
    for k in need:
        firsts[k] = grad(u, c[k], create_graph=True)
    for k in first:
        out[f"d_{k}"] = firsts[k]
    for k in second:
        out[f"dd_{k}"] = grad(firsts[k], c[k], create_graph=create_graph)
    if not create_graph:
        out = {k: v.detach() for k, v in out.items()}
    return out
