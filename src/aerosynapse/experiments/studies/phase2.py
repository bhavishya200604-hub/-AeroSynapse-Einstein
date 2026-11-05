"""Phase 2 studies: PINN accuracy vs exact solutions and baselines; BC strategies; loss imbalance."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from aerosynapse.ml.baselines import train_supervised
from aerosynapse.ml.bc_strategies import train_dd_boundary_layer
from aerosynapse.ml.nets import MLP, set_seed
from aerosynapse.ml.pinn import PINNConfig, predict, train_pinn
from aerosynapse.ml.problems import (
    Burgers1D,
    ConvDiff1D,
    DiskPoisson,
    Heat1D,
    PINNProblem,
    Poisson2D,
    SteadyBoundaryLayer,
)
from aerosynapse.ml.weighting import gradient_imbalance
from aerosynapse.validation.metrics import error_distribution, field_metrics

PROFILES = {
    "quick": {"iters": 3000, "n": 1000, "seeds": 3},
    "standard": {"iters": 10000, "n": 2000, "seeds": 5},
    "full": {"iters": 30000, "n": 4000, "seeds": 10},
}


def make_problem(name: str, hard: bool) -> PINNProblem:
    return {
        "heat1d": lambda: Heat1D(hard=hard),
        "poisson2d": lambda: Poisson2D(hard=hard),
        "burgers1d": lambda: Burgers1D(nu=0.1),
        "convdiff_pe2": lambda: ConvDiff1D(pe=2.0, hard=hard),
        "convdiff_pe10": lambda: ConvDiff1D(pe=10.0, hard=hard),
        "bl_pe20": lambda: SteadyBoundaryLayer(pe=20.0, hard=hard),
        "disk": lambda: DiskPoisson(hard=hard),
    }[name]()


def pinn_job(job: dict[str, Any]) -> dict[str, Any]:
    set_seed(job["seed"])
    p = make_problem(job["problem"], job["hard"])
    m = MLP(len(p.lo), 1, 32, 3, lo=p.lo, hi=p.hi, periodic_dims=p.periodic_dims, seed=job["seed"])
    cfg = PINNConfig(
        iterations=job["iters"],
        n_collocation=job["n"],
        log_every=max(1, job["iters"] // 20),
        adaptive_weights=job.get("weights"),
        lr_decay=0.9,
    )
    r = train_pinn(p, m, cfg, seed=job["seed"])
    G = p.eval_grid(101 if len(p.lo) == 2 else 1001)
    pred, ref = predict(p, m, G), p.exact(G)
    out = {
        **job,
        **field_metrics(pred, ref),
        **error_distribution(pred, ref),
        "wall_s": r.wall_s,
        "final_loss": r.history[-1]["loss"],
        "final_weights": r.final_weights,
    }
    if job.get("imbalance"):
        X = p.tensor(G[:: max(1, len(G) // 512)]).requires_grad_(True)
        set_seed(job["seed"])
        m0 = MLP(len(p.lo), 1, 32, 3, lo=p.lo, hi=p.hi, periodic_dims=p.periodic_dims, seed=job["seed"])
        out["imbalance_init"] = gradient_imbalance(
            p.residual(m0, X).pow(2).mean(), p.constraint_losses(m0), list(m0.parameters())
        )
        out["imbalance_final"] = gradient_imbalance(
            p.residual(m, X).pow(2).mean(), p.constraint_losses(m), list(m.parameters())
        )
    return out


def supervised_job(job: dict[str, Any]) -> dict[str, Any]:
    set_seed(job["seed"])
    p = make_problem(job["problem"], False)
    rng = np.random.default_rng(job["seed"])
    lo, hi = np.array(p.lo), np.array(p.hi)
    X = lo + rng.random((job["n"], len(lo))) * (hi - lo)
    if isinstance(p, DiskPoisson):
        X = X[p.inside(X)]
    m, wall = train_supervised(
        X, p.exact(X), p.lo, p.hi, iterations=job["iters"], seed=job["seed"], periodic_dims=p.periodic_dims
    )
    G = p.eval_grid(101 if len(p.lo) == 2 else 1001)
    with torch.no_grad():
        pred = m(p.tensor(G)).numpy().ravel()
    return {**job, **field_metrics(pred, p.exact(G)), "wall_s": wall}


def dd_job(job: dict[str, Any]) -> dict[str, Any]:
    r = train_dd_boundary_layer(20.0, iterations=job["iters"], n=job["n"] // 2, seed=job["seed"])
    return {
        **job,
        "rel_l2": r.rel_l2,
        "wall_s": r.wall_s,
        "interface_jump_u": r.interface_jump_u,
        "interface_jump_flux": r.interface_jump_flux,
    }


def classical_reference_timing() -> dict[str, Any]:
    """Classical solver error and time at moderate resolution, for context in the comparison table."""
    from aerosynapse.numerics import heat, poisson
    from aerosynapse.physics import library as L

    out: dict[str, Any] = {}
    p = L.heat_1d()
    t0 = time.perf_counter()
    x, s = heat.solve_heat_1d(p, 128, nt=128)
    out["heat1d_fd_n128"] = {
        "time_s": time.perf_counter() - t0,
        "rel_l2": float(np.linalg.norm(s.u - p.exact(x, 0.5)) / np.linalg.norm(p.exact(x, 0.5))),
    }
    q = L.poisson_2d()
    t0 = time.perf_counter()
    X, Y, U = poisson.solve_poisson_2d(q, 64)
    out["poisson2d_fd_n64"] = {
        "time_s": time.perf_counter() - t0,
        "rel_l2": float(np.linalg.norm(U - q.exact(X, Y)) / np.linalg.norm(q.exact(X, Y))),
    }
    return out
