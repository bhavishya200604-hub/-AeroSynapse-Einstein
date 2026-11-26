"""Sampling and training strategies (Step 4.4). Formulas checked against the papers (docs/LITERATURE.md).

Residual-based adaptive sampling of the PDE points (Wu, Zhu, Tan, Kartha & Lu, arXiv:2207.10289, eq. 2 and Algorithms
2-3): density p(x) ~ eps^k(x) / E[eps^k] + c, eps = |PDE residual|, evaluated on a dense candidate set.
    rad    every `period` Adam epochs the whole PDE point set is redrawn from p (paper default k = 1, c = 1)
    rar_d  start with `start_fraction` of the points; every `period` epochs add points drawn from p until n_pde is
           reached (paper default k = 2, c = 0)
Causal weighting (Wang, Sankaran & Perdikaris, arXiv:2203.07404, eqs. 3.2-3.3): time split into N_t chunks,
    L_r = (1/N_t) sum_i w_i L_r(t_i), w_i = exp(-eps sum_{k<i} L_r(t_k)) (no gradient through w); eps raised via the
    schedule [1e-2, 1e-1, 1, 10, 100] whenever min_i w_i > delta = 0.99. Implemented by rescaling each PDE residual
    so that mean(r_scaled^2) equals the weighted loss exactly. Weights frozen for L-BFGS (as all adaptive weights).
Time-marching / sequence-to-sequence (Krishnapriyan et al., arXiv:2109.01050): [0, T] split into windows trained one
    after another with separate networks; the initial condition of window k is the prediction of window k-1 at its end;
    the total collocation budget is split between windows.
Curriculum (same paper): the PDE coefficient (Pe or Re) is raised from `start_fraction` x target to the target in
    `stages` geometric steps, warm-starting the same network; L-BFGS only in the last stage.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn.derivs import evaluate

PAPER_KC = {"rad": (1.0, 1.0), "rar_d": (2.0, 0.0)}


# --- PDE residual magnitude at arbitrary points -----------------------------------------------------------------------
def pde_residual_abs(model: nn.Module, problem: Any, pde: Any, chunk: int | None = None) -> np.ndarray:
    """|PDE residual| at the points of a PDE batch (1D: dict of columns; 2D: tensor (N, 2)). 2D: Euclidean norm over
    the TRAINED (normalised) residuals, i.e. the same terms the loss uses. `chunk`: evaluate at most this many points
    at a time (bounds the memory of large candidate sets; pointwise values unchanged; None or n <= chunk: one pass)."""
    n = _size(pde)
    if chunk and n > chunk:
        return np.concatenate(
            [pde_residual_abs(model, problem, _take(pde, np.arange(i, min(i + chunk, n)))) for i in range(0, n, chunk)]
        )
    if hasattr(problem, "fields"):
        r = problem.residuals(problem.fields(model, pde, order=2))
        keys = [k for k in r if not (k == "cont" and problem.kind == "streamfunction")]
        sq = sum((r[k].detach() / problem.scales[k]) ** 2 for k in keys)
        return torch.sqrt(sq).cpu().numpy()
    first, second = problem.derivative_spec()
    d = evaluate(model, pde, problem.input_names, first=first, second=second)
    return problem.residual(d, problem.p_of(pde)).detach().abs().cpu().numpy()


def _take(pde: Any, idx: np.ndarray) -> Any:
    ix = torch.as_tensor(idx, dtype=torch.long)
    if isinstance(pde, dict):
        return {k: v[ix.to(v.device)] for k, v in pde.items()}
    return pde[ix.to(pde.device)]


def _cat(a: Any, b: Any) -> Any:
    if isinstance(a, dict):
        return {k: torch.cat([a[k], b[k]]) for k in a}
    return torch.cat([a, b])


def _size(pde: Any) -> int:
    return len(next(iter(pde.values()))) if isinstance(pde, dict) else pde.shape[0]


class AdaptiveSampler:
    """RAD / RAR-D for the PDE points; state = the current point set (checkpointed by the Trainer)."""

    def __init__(self, cfg: dict, n_pde: int, epochs: int, seed: int):
        self.method = cfg["method"]
        k0, c0 = PAPER_KC[self.method]
        self.k = float(cfg["k"]) if cfg.get("k") is not None else k0
        self.c = float(cfg["c"]) if cfg.get("c") is not None else c0
        self.period = int(cfg["period"])
        self.n_cand = int(cfg["n_candidates"])
        self.chunk = int(cfg["chunk"]) if cfg.get("chunk") else 4096  # candidates evaluated per pass (memory bound)
        self.n_final = n_pde
        self.seed = seed
        self.points: Any = None
        self.n_updates = 0
        if self.method == "rar_d":
            self.n_start = max(1, int(round(float(cfg["start_fraction"]) * n_pde)))
            n_upd = max((epochs - 1) // self.period, 1)
            self.n_add = max(1, math.ceil((n_pde - self.n_start) / n_upd))
        else:
            self.n_start, self.n_add = n_pde, 0

    def initial(self, pde: Any) -> Any:
        """First point set: the normal (Sobol/random/LHS) set; rar_d keeps only its first n_start points."""
        if self.points is None:
            self.points = pde if self.method == "rad" else _take(pde, np.arange(self.n_start))
        return self.points

    def density(self, eps: np.ndarray) -> np.ndarray:
        ek = eps**self.k
        p = ek / max(float(ek.mean()), 1e-300) + self.c
        if not np.isfinite(p).all() or p.sum() <= 0:
            p = np.ones_like(eps)
        return p / p.sum()

    def update(self, model: nn.Module, problem: Any, step: int, device: Any, dtype: Any, method: str) -> None:
        if step == 0 or step % self.period:
            return
        if self.method == "rar_d" and _size(self.points) >= self.n_final:
            return
        cand = problem.sample(self.n_cand, 1, 1, method, self.seed * 7919 + 13 * step, device, dtype)["pde"]
        p = self.density(pde_residual_abs(model, problem, cand, self.chunk))
        rng = np.random.default_rng(self.seed * 1_000_003 + step)
        if self.method == "rad":
            self.points = _take(cand, rng.choice(len(p), size=self.n_final, replace=False, p=p))
        else:
            m = min(self.n_add, self.n_final - _size(self.points))
            self.points = _cat(self.points, _take(cand, rng.choice(len(p), size=m, replace=False, p=p)))
        self.n_updates += 1

    def state_dict(self) -> dict:
        return {"points": self.points, "n_updates": self.n_updates}

    def load_state_dict(self, s: dict) -> None:
        self.points, self.n_updates = s["points"], s["n_updates"]


class CausalWeights:
    """Temporal causal weighting of the PDE residual (transient 1D problems)."""

    def __init__(self, cfg: dict, t0: float, t1: float):
        self.n = int(cfg["n_chunks"])
        self.eps_list = [float(e) for e in cfg["eps"]]
        self.delta = float(cfg["delta"])
        self.t0, self.t1 = t0, t1
        self.i_eps = 0
        self.frozen = False
        self.w_frozen: Tensor | None = None
        self.last_min_w = float("nan")
        self.eps_history: list[tuple[int, float]] = []

    @property
    def eps(self) -> float:
        return self.eps_list[self.i_eps]

    def transform(self, r: Tensor, t: Tensor, step: int | None) -> Tensor:
        c = ((t.detach() - self.t0) / (self.t1 - self.t0) * self.n).floor().clamp(0, self.n - 1).long()
        cnt = torch.bincount(c, minlength=self.n).to(r.dtype)
        L = torch.zeros(self.n, dtype=r.dtype, device=r.device).index_add_(0, c, r.detach() ** 2)
        Lc = torch.where(cnt > 0, L / cnt.clamp(min=1), torch.zeros_like(L))
        if self.frozen and self.w_frozen is not None:
            w = self.w_frozen
        else:
            cum = torch.cumsum(Lc, 0) - Lc  # sum over k < i
            w = torch.exp(-self.eps * cum)
            self.last_min_w = float(w[cnt > 0].min())
            if step is not None and self.last_min_w > self.delta and self.i_eps < len(self.eps_list) - 1:
                self.i_eps += 1  # next epsilon (the paper's loop; the budget stays fixed here)
                self.eps_history.append((step, self.eps))
            self.w_frozen = w
        nz = int((cnt > 0).sum())
        scale = w[c] * r.shape[0] / (nz * cnt[c])  # mean(r^2 * scale) = (1/N_t) sum_i w_i L_i
        return r * torch.sqrt(scale)

    def freeze(self) -> None:
        self.frozen = True

    def state_dict(self) -> dict:
        return {"i_eps": self.i_eps, "frozen": self.frozen, "w": self.w_frozen, "hist": self.eps_history}

    def load_state_dict(self, s: dict) -> None:
        self.i_eps, self.frozen, self.w_frozen, self.eps_history = s["i_eps"], s["frozen"], s["w"], s["hist"]


# --- time-marching ---------------------------------------------------------------------------------------------------
class WindowProblem(P.Problem):
    """A transient 1D problem restricted to t in [t0, t1]; IC = previous window's network at t0 (or the true IC)."""

    def __init__(self, base: P.Problem, t0: float, t1: float, prev: nn.Module | None):
        P.Problem.__init__(
            self, base.name, base.cond, T=t1, param=base.param, prange=base.prange, parametric=base.parametric
        )
        i = self.input_names.index("t")
        self.lo[i], self.hi[i] = float(t0), float(t1)
        self.t0, self.prev = float(t0), prev

    def sample(self, n_pde, n_ic, n_bc, method, seed, device, dtype):
        out = super().sample(n_pde, n_ic, n_bc, method, seed, device, dtype)
        out["ic"]["t"] = torch.full_like(out["ic"]["t"], self.t0)
        return out

    def ic_target(self, cols: dict[str, Tensor]) -> Tensor:
        if self.prev is None:
            return super().ic_target(cols)
        with torch.no_grad():
            return self.prev(torch.stack([cols[k] for k in self.input_names], 1)).squeeze(-1)


class CompositeModel(nn.Module):
    """Routes each input row to the window network that owns its time (last window includes its end)."""

    def __init__(self, models: list[nn.Module], edges: list[float], t_index: int):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.edges = list(edges)
        self.ti = t_index

    def forward(self, X: Tensor) -> Tensor:
        t = X[:, self.ti]
        out = torch.zeros(X.shape[0], 1, dtype=X.dtype, device=X.device)
        for k, m in enumerate(self.models):
            lo, hi = self.edges[k], self.edges[k + 1]
            mask = (t >= lo) & ((t < hi) if k < len(self.models) - 1 else (t <= hi))
            if mask.any():
                out = out.index_put((mask.nonzero().squeeze(-1),), m(X[mask]))
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def curriculum_values(target: float, start_fraction: float, stages: int) -> list[float]:
    """Geometric sequence from start_fraction * target to target (stages values, the last = target exactly)."""
    if stages <= 1:
        return [float(target)]
    v = np.geomspace(start_fraction * target, target, stages)
    v[-1] = target
    return [float(x) for x in v]


def split_budget(total: int, parts: int) -> list[int]:
    base, rem = divmod(int(total), parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]
