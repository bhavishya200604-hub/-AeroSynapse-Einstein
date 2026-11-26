"""1D problem definitions for the baseline PINN: residuals, initial/boundary conditions, sampling, references.

Problems (all on x in [0, 1]):
  A_steady : c u_x - u_xx / Pe = 0,           u(0) = theta_left, u(1) = theta_right            (Dirichlet)
  A_pulse  : u_t + c u_x - u_xx / Pe = 0,     u(x,0) = pulse(x,0); u(0,t), u(1,t) = analytic pulse (infinite line)
  B_slab   : u_t - u_xx + m^2 u - Q = 0,      u(x,0) = theta0; u_x(0,t) = 0; u_x(1,t) + Bi u(1,t) = 0 (Fourier time)
The parameter (Pe or Bi) is a fixed number (one network per value) or, if ``parametric``, a network input
logp = ln(parameter) sampled log-uniformly over the parameter range.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from aerosynapse.thermal.data.design import sobol_unit
from aerosynapse.thermal.physics import analytic as an
from aerosynapse.thermal.pinn.derivs import evaluate
from aerosynapse.thermal.pinn.spec import param_range

Tensor = torch.Tensor


def pulse_torch(x: Tensor, t: Tensor, Pe: Tensor | float, pu: dict[str, float]) -> Tensor:
    """Analytic Gaussian pulse in torch: baseline + A (s0/s) exp(-(x - x0 - c t)^2 / (2 s^2)), s^2 = s0^2 + 2t/Pe."""
    s2 = pu["sigma0"] ** 2 + 2.0 * t / Pe
    return pu["baseline"] + pu["amplitude"] * torch.sqrt(pu["sigma0"] ** 2 / s2) * torch.exp(
        -((x - pu["x0"] - pu["c"] * t) ** 2) / (2.0 * s2)
    )


@dataclass
class Problem:
    name: str
    cond: dict[str, Any]
    T: float = 0.0
    param: float | None = None  # fixed parameter value (non-parametric)
    prange: tuple[float, float] = (1.0, 1.0)  # parameter range (parametric)
    parametric: bool = False
    input_names: list[str] = field(init=False)
    lo: list[float] = field(init=False)
    hi: list[float] = field(init=False)

    def __post_init__(self) -> None:
        names, lo, hi = ["x"], [0.0], [1.0]
        if self.has_time:
            names.append("t")
            lo.append(0.0)
            hi.append(self.T)
        if self.parametric:
            names.append("logp")
            lo.append(float(np.log(self.prange[0])))
            hi.append(
                float(np.log(self.prange[1]))
                if self.prange[1] > self.prange[0]
                else float(np.log(self.prange[0])) + 1e-9
            )
        self.input_names, self.lo, self.hi = names, lo, hi

    @property
    def has_time(self) -> bool:
        return self.name != "A_steady"

    @property
    def param_name(self) -> str:
        return "Bi" if self.name == "B_slab" else "Pe"

    # --- parameter access -----------------------------------------------------------------------
    def p_of(self, cols: dict[str, Tensor]) -> Tensor | float:
        return torch.exp(cols["logp"]) if self.parametric else float(self.param)  # type: ignore[arg-type]

    # --- PDE residual and conditions -------------------------------------------------------------
    def derivative_spec(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return (("x",), ("x",)) if not self.has_time else (("x", "t"), ("x",))

    def residual(self, d: dict[str, Tensor], p: Tensor | float) -> Tensor:
        c = self.cond
        if self.name == "A_steady":
            return c["c"] * d["d_x"] - d["dd_x"] / p
        if self.name == "A_pulse":
            return d["d_t"] + c["pulse"]["c"] * d["d_x"] - d["dd_x"] / p
        return d["d_t"] - d["dd_x"] + c["m"] ** 2 * d["u"] - c["Q"]

    def ic_target(self, cols: dict[str, Tensor]) -> Tensor:
        if self.name == "A_pulse":
            return pulse_torch(cols["x"], cols["t"], self.p_of(cols), self.cond["pulse"])
        return torch.full_like(cols["x"], float(self.cond["theta0"]))

    def bc_residual(self, side: str, d: dict[str, Tensor], cols: dict[str, Tensor]) -> Tensor:
        """Boundary residual on side 'left' (x = 0) or 'right' (x = 1); d holds u and d_x at the boundary points."""
        c, p = self.cond, self.p_of(cols)
        if self.name == "A_steady":
            return d["u"] - float(c["theta_left"] if side == "left" else c["theta_right"])
        if self.name == "A_pulse":
            return d["u"] - pulse_torch(cols["x"], cols["t"], p, c["pulse"])
        if side == "left":
            return d["d_x"]
        return d["d_x"] + p * d["u"]

    # --- sampling --------------------------------------------------------------------------------
    def _unit(self, n: int, dim: int, method: str, seed: int) -> np.ndarray:
        if method == "sobol":
            return sobol_unit(n, dim, seed)
        if method == "lhs":  # Latin hypercube (Step 4.4)
            from scipy.stats import qmc

            return qmc.LatinHypercube(d=dim, seed=seed).random(n)
        return np.random.default_rng(seed).uniform(size=(n, dim))

    def _col(self, name: str, u: np.ndarray, k: int) -> np.ndarray:
        i = self.input_names.index(name)
        return self.lo[i] + (self.hi[i] - self.lo[i]) * u[:, k]

    def sample(
        self, n_pde: int, n_ic: int, n_bc: int, method: str, seed: int, device: Any, dtype: Any
    ) -> dict[str, dict[str, Tensor]]:
        """Collocation batches: 'pde', 'ic' (transient), 'bc_left', 'bc_right' (dicts of named columns)."""
        mk = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)  # noqa: E731
        par = self.parametric
        out: dict[str, dict[str, Tensor]] = {}
        names = [n for n in self.input_names]
        u = self._unit(n_pde, len(names), method, seed)
        out["pde"] = {n: mk(self._col(n, u, k)) for k, n in enumerate(names)}
        if self.has_time:
            k_dim = 1 + par
            u = self._unit(n_ic, k_dim, method, seed + 1)
            ic = {"x": mk(self._col("x", u, 0)), "t": mk(np.zeros(n_ic))}
            if par:
                ic["logp"] = mk(self._col("logp", u, 1))
            out["ic"] = ic
        for j, (side, xb) in enumerate((("bc_left", 0.0), ("bc_right", 1.0))):
            if self.has_time:
                u = self._unit(n_bc, 1 + par, method, seed + 2 + j)
                bc = {"x": mk(np.full(n_bc, xb)), "t": mk(self._col("t", u, 0))}
                if par:
                    bc["logp"] = mk(self._col("logp", u, 1))
            else:
                nb = n_bc if par else 1
                u = self._unit(nb, 1, method, seed + 2 + j)
                bc = {"x": mk(np.full(nb, xb))}
                if par:
                    bc["logp"] = mk(self._col("logp", u, 0))
            out[side] = bc
        return out

    # --- references (numpy, analytic) -------------------------------------------------------------
    def reference(self, x: np.ndarray, t: np.ndarray | None, p: float) -> np.ndarray:
        """Analytic solution on the tensor grid: steady -> (nx,), transient -> (nt, nx)."""
        c = self.cond
        if self.name == "A_steady":
            Pe_eff = p * c["c"]
            return c["theta_left"] + (c["theta_right"] - c["theta_left"]) * an.steady_convdiff(x, Pe_eff)
        assert t is not None
        if self.name == "A_pulse":
            pu = c["pulse"]
            return pu["baseline"] + pu["amplitude"] * an.gaussian_pulse(
                x[None, :], t[:, None], p, pu["sigma0"], pu["x0"], pu["c"]
            )
        return np.stack([an.slab_transient(x, float(tt), p, c["Q"], c["m"], c["theta0"], n_terms=400) for tt in t])

    def reference_dx(self, x: np.ndarray, t: np.ndarray | None, p: float, h: float = 1e-6) -> np.ndarray:
        """d/dx of the analytic reference by central differences of the (exact) reference (error ~ h^2 f''')."""
        if self.name == "A_steady":
            c = self.cond
            return (c["theta_right"] - c["theta_left"]) * an.steady_convdiff_dx(x, p * c["c"])
        return (self.reference(x + h, t, p) - self.reference(x - h, t, p)) / (2 * h)


def build_problem(name: str, pcfg: dict, value: float | None = None) -> Problem:
    """Problem from a spec block. value=None + parametric -> one network over the parameter range."""
    T = float(pcfg.get("T", 0.0))
    prange = param_range(pcfg["parameter"])
    if pcfg.get("parametric", False):
        return Problem(name, pcfg["conditions"], T=T, prange=prange, parametric=True)
    return Problem(name, pcfg["conditions"], T=T, param=float(value), prange=prange)


def residual_vectors(model, problem: Problem, batch: dict[str, dict[str, Tensor]]) -> dict[str, Tensor]:
    """Per-point residuals of each loss component (PDE, IC, each boundary); loss_k = mean(r_k^2)."""
    order = problem.input_names
    first, second = problem.derivative_spec()
    out: dict[str, Tensor] = {}
    cols = batch["pde"]
    d = evaluate(model, cols, order, first=first, second=second)
    out["pde"] = problem.residual(d, problem.p_of(cols))
    if problem.has_time:
        cols = batch["ic"]
        u = model(torch.stack([cols[k] for k in order], dim=1)).squeeze(-1)
        out["ic"] = u - problem.ic_target(cols)
    for side in ("bc_left", "bc_right"):
        cols = batch[side]
        d = evaluate(model, cols, order, first=("x",), second=())
        out[side] = problem.bc_residual(side.split("_")[1], d, cols)
    return out


def component_losses(model, problem: Problem, batch: dict[str, dict[str, Tensor]]) -> dict[str, Tensor]:
    """Mean-squared PDE residual, IC error and boundary residuals (autograd derivatives)."""
    return {k: (v**2).mean() for k, v in residual_vectors(model, problem, batch).items()}


def weighted_total(losses: dict[str, Tensor], weights: dict[str, float]) -> Tensor:
    tot = weights["pde"] * losses["pde"] + weights["bc"] * (losses["bc_left"] + losses["bc_right"])
    if "ic" in losses:
        tot = tot + weights["ic"] * losses["ic"]
    return tot
