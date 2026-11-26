r"""PINN training loop.

Total loss :math:`\mathcal L=\lambda_r\mathcal L_r+\sum_k\lambda_k\mathcal L_k` with
:math:`\mathcal L_r=\overline{r^2}` over the collocation set and constraint terms (IC, BC, data,
conservation, interface, regularisation). Options: residual-based samplers, learning-rate annealing
weights, causal residual weighting (Wang, Sankaran & Perdikaris 2022), Adam then optional L-BFGS.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from aerosynapse.exceptions import SolverDivergenceError
from aerosynapse.ml.problems import PINNProblem
from aerosynapse.ml.sampling import Sampler, make_sampler
from aerosynapse.ml.weighting import AnnealingWeights


@dataclass
class PINNConfig:
    iterations: int = 2000
    lr: float = 1e-3
    n_collocation: int = 1000
    sampler: str = "uniform"
    resample_every: int = 250
    lbfgs_iterations: int = 0
    weights: dict[str, float] = field(default_factory=dict)
    adaptive_weights: str | None = None  # "annealing"
    anneal_every: int = 100
    causal_eps: float | None = None
    causal_bins: int = 16
    conservation: float = 0.0
    weight_decay: float = 0.0
    log_every: int = 100
    lr_decay: float = 1.0  # multiplicative per 1000 iterations
    time_budget_s: float | None = None
    sampler_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class PINNResult:
    model: torch.nn.Module
    history: list[dict[str, float]]
    wall_s: float
    residual_evals: int
    points: np.ndarray
    final_weights: dict[str, float]


def residual_numpy(problem: PINNProblem, model: torch.nn.Module, X: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """|r| at NumPy points (gradients w.r.t. inputs only; no parameter graph kept)."""
    out = []
    for i in range(0, len(X), chunk):
        Xt = problem.tensor(X[i : i + chunk]).requires_grad_(True)
        r = problem.residual(model, Xt)
        out.append(r.detach().cpu().numpy().ravel())
    return np.concatenate(out)


def predict(problem: PINNProblem, model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return problem.u(model, problem.tensor(X)).cpu().numpy().ravel()


def _causal_loss(problem: PINNProblem, r2: torch.Tensor, X: torch.Tensor, eps: float, bins: int) -> torch.Tensor:
    t = X[:, -1]
    lo, hi = problem.lo[-1], problem.hi[-1]
    idx = torch.clamp(((t - lo) / (hi - lo) * bins).long(), 0, bins - 1)
    Lb = torch.zeros(bins, dtype=r2.dtype).index_add_(0, idx, r2.ravel())
    cnt = torch.zeros(bins, dtype=r2.dtype).index_add_(0, idx, torch.ones_like(r2.ravel())).clamp_min(1)
    Lb = Lb / cnt
    with torch.no_grad():
        w = torch.exp(-eps * torch.cumsum(torch.cat([Lb.new_zeros(1), Lb[:-1]]), 0))
    return torch.mean(w * Lb)


def train_pinn(
    problem: PINNProblem,
    model: torch.nn.Module,
    cfg: PINNConfig,
    seed: int = 0,
    data: tuple[np.ndarray, np.ndarray] | None = None,
    callback: Callable[[int, dict[str, float]], None] | None = None,
    sampler: Sampler | None = None,
) -> PINNResult:
    """Train ``model`` on ``problem``. Raises :class:`SolverDivergenceError` on a non-finite loss."""
    torch.manual_seed(seed)
    inside = getattr(problem, "inside", None)
    smp = sampler or make_sampler(
        cfg.sampler, problem.lo, problem.hi, cfg.n_collocation, seed, inside=inside, **cfg.sampler_kwargs
    )
    X = smp.initial()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, cfg.lr_decay ** (1 / 1000))
    anneal = None
    hist: list[dict[str, float]] = []
    res_evals = 0
    Xd = Ud = None
    if data is not None:
        Xd, Ud = problem.tensor(data[0]), problem.tensor(data[1].reshape(-1, 1))
    t0 = time.perf_counter()

    def losses(Xt: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        r = problem.residual(model, Xt)
        r2 = r**2
        Lr = _causal_loss(problem, r2, Xt, cfg.causal_eps, cfg.causal_bins) if cfg.causal_eps else r2.mean()
        cons = problem.constraint_losses(model)
        if Xd is not None and Ud is not None:
            cons["data"] = torch.mean((problem.u(model, Xd) - Ud) ** 2)
        if cfg.conservation > 0 and hasattr(problem, "conservation_loss"):
            cons["conservation"] = problem.conservation_loss(model)
        return Lr, cons

    def total(Lr: torch.Tensor, cons: dict[str, torch.Tensor]) -> torch.Tensor:
        w: dict[str, float] = anneal.w if anneal is not None else {}
        out = cfg.weights.get("residual", 1.0) * Lr
        for k, L in cons.items():
            base = cfg.conservation if k == "conservation" else cfg.weights.get(k, 1.0)
            out = out + base * w.get(k, 1.0) * L
        return out

    Xt = problem.tensor(X).requires_grad_(True)
    for it in range(cfg.iterations):
        if it > 0 and it % cfg.resample_every == 0:
            X = smp.update(X, lambda P: residual_numpy(problem, model, P), it)
            Xt = problem.tensor(X).requires_grad_(True)
        Lr, cons = losses(Xt)
        if cfg.adaptive_weights == "annealing" and cons and it % cfg.anneal_every == 0:
            anneal = anneal or AnnealingWeights(list(cons))
            anneal.update(Lr, cons, params)
        loss = total(Lr, cons)
        if not torch.isfinite(loss):
            raise SolverDivergenceError(f"PINN loss became non-finite at iteration {it}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        res_evals += len(X)
        if it % cfg.log_every == 0 or it == cfg.iterations - 1:
            row = {
                "it": float(it),
                "loss": float(loss),
                "L_r": float(Lr),
                **{f"L_{k}": float(v) for k, v in cons.items()},
                "n_points": float(len(X)),
                "time_s": time.perf_counter() - t0,
            }
            hist.append(row)
            if callback:
                callback(it, row)
        if cfg.time_budget_s is not None and time.perf_counter() - t0 > cfg.time_budget_s:
            break
    if cfg.lbfgs_iterations > 0:
        lb = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=cfg.lbfgs_iterations,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
        )

        def closure() -> torch.Tensor:
            lb.zero_grad()
            Lr_, c_ = losses(Xt)
            L_ = total(Lr_, c_)
            L_.backward()
            return L_

        lb.step(closure)
        Lr, cons = losses(Xt)
        hist.append(
            {
                "it": float(cfg.iterations + cfg.lbfgs_iterations),
                "loss": float(total(Lr, cons)),
                "L_r": float(Lr),
                "n_points": float(len(X)),
                "time_s": time.perf_counter() - t0,
            }
        )
    return PINNResult(
        model, hist, time.perf_counter() - t0, res_evals + smp.extra_evals, X, dict(anneal.w) if anneal else {}
    )


def train_pinn_checkpointed(
    problem: PINNProblem,
    model: torch.nn.Module,
    iterations: int,
    n_collocation: int,
    ckpt_path: str,
    ckpt_every: int,
    resume: bool = False,
    seed: int = 0,
    lr: float = 1e-3,
    stop_at: int | None = None,
) -> int:
    """Fixed-collocation Adam training with checkpoint/resume (used to test interrupt-and-resume).

    ``stop_at`` simulates an interruption: training stops after that step (after checkpointing).
    Returns the last completed step.
    """
    from pathlib import Path

    from aerosynapse.ml.io import load_checkpoint, save_checkpoint

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lo, hi = np.array(problem.lo), np.array(problem.hi)
    X = problem.tensor(lo + rng.random((n_collocation, len(lo))) * (hi - lo)).requires_grad_(True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    start = 0
    if resume and Path(ckpt_path).exists():
        start = load_checkpoint(ckpt_path, model, opt)
    for it in range(start, iterations):
        loss = problem.residual(model, X).pow(2).mean() + sum(problem.constraint_losses(model).values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % ckpt_every == 0:
            save_checkpoint(ckpt_path, model, opt, it + 1)
        if stop_at is not None and it + 1 >= stop_at:
            return it + 1
    return iterations
