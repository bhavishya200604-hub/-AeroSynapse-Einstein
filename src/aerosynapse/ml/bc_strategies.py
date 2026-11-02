r"""Comparison of boundary-condition enforcement strategies for PINNs (spec Section 6).

1. ``penalty``  — soft constraint added to the loss (``problem.hard = False``).
2. ``hard``     — trial function satisfying the BC exactly (``problem.hard = True``).
3. ``domain decomposition`` — two-subdomain PINN (XPINN/cPINN-style) with interface continuity of u
   and the flux u_x, implemented below for the steady boundary layer.
4. ``complex geometry`` — approximate distance function (φ = 1 - r² on the unit disk) in
   :class:`aerosynapse.ml.problems.DiskPoisson`.

Applicability: hard trial functions need a closed-form distance-like function and (for Neumann /
Robin) a more elaborate construction; penalties work everywhere but introduce loss imbalance;
decomposition helps localise sharp layers at the cost of interface terms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from aerosynapse.ml.autodiff import grad, second
from aerosynapse.ml.nets import MLP


@dataclass
class DDResult:
    rel_l2: float
    wall_s: float
    interface_jump_u: float
    interface_jump_flux: float


def train_dd_boundary_layer(
    pe: float,
    iterations: int = 2000,
    n: int = 400,
    x_split: float = 0.8,
    width: int = 32,
    depth: int = 3,
    seed: int = 0,
    lr: float = 1e-3,
) -> DDResult:
    """Steady layer u_x = u_xx/Pe on [0, x_split] ∪ [x_split, 1], penalty BCs + interface conditions."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nets = [
        MLP(1, 1, width, depth, lo=[0.0], hi=[x_split], seed=seed),
        MLP(1, 1, width, depth, lo=[x_split], hi=[1.0], seed=seed + 1),
    ]
    opt = torch.optim.Adam([p for m in nets for p in m.parameters()], lr=lr)
    dt = torch.get_default_dtype()
    xs = [
        torch.tensor(rng.uniform(0, x_split, (n, 1)), dtype=dt, requires_grad=True),
        torch.tensor(rng.uniform(x_split, 1, (n, 1)), dtype=dt, requires_grad=True),
    ]
    xi = torch.tensor([[x_split]], dtype=dt, requires_grad=True)
    x0, x1 = torch.zeros(1, 1, dtype=dt), torch.ones(1, 1, dtype=dt)
    t0 = time.perf_counter()
    for _ in range(iterations):
        loss = torch.zeros((), dtype=dt)
        for m, x in zip(nets, xs, strict=True):
            u = m(x)
            loss = loss + torch.mean((grad(u, x) - second(u, x, 0) / pe) ** 2)
        ua, ub = nets[0](xi), nets[1](xi)
        loss = loss + (ua - ub).pow(2).sum() + (grad(ua, xi) - grad(ub, xi)).pow(2).sum() / pe
        loss = loss + nets[0](x0).pow(2).sum() + (nets[1](x1) - 1).pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    xe = np.linspace(0, 1, 1001)
    ex = (np.exp(pe * (xe - 1.0)) - np.exp(-pe)) / (1.0 - np.exp(-pe))
    with torch.no_grad():
        X = torch.tensor(xe[:, None], dtype=dt)
        pred = torch.where(x_split >= X, nets[0](X), nets[1](X)).numpy().ravel()
    ua, ub = nets[0](xi), nets[1](xi)
    jf = float((grad(ua, xi) - grad(ub, xi)).abs())
    return DDResult(
        float(np.linalg.norm(pred - ex) / np.linalg.norm(ex)), time.perf_counter() - t0, float((ua - ub).abs()), jf
    )
