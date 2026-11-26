"""Parametric 2D PINN (Step 4.5): one network for a RANGE of Re / Pr (or Pe) / Bi [/ Ri].

Inputs (x, y, log Re, log Pr, log Bi [, Ri]) - log for log-scaled design dimensions, linear otherwise - scaled to
[-1, 1] by the FULL design range, while collocation points sample the parameters only inside the TRAINING box of the
Phase 2 parameter design (unit coordinates u <= u_lim), so the extrapolation test points are genuinely outside it.
The residuals use per-point Re, Pe = Re Pr (or a swept Pe) and Ri; Robin walls whose Biot number is a swept parameter
use the per-point Bi. Everything else (formulations, hard constraints, normalisation, validation loss) is inherited
from aerosynapse.thermal.pinn.problems2d.Problem2D, whose non-parametric behaviour is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from aerosynapse.thermal.data.design import Dim, sobol_unit
from aerosynapse.thermal.pinn import problems2d as P2

WALLS = P2.WALLS


def col_name(d: Dim) -> str:
    return f"log{d.name}" if d.is_log else d.name


@dataclass
class ParamProblem2D(P2.Problem2D):
    sweep: tuple[Dim, ...] = ()
    u_lim: float = 1.0  # collocation parameters drawn from the unit-cube box [0, u_lim]^d
    robin_param: dict[str, tuple[str, float]] = field(default_factory=dict)  # wall -> (parameter, theta_inf)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.input_names = ("x", "y", *[col_name(d) for d in self.sweep])
        enc = [(np.log(d.lo), np.log(d.hi)) if d.is_log else (d.lo, d.hi) for d in self.sweep]
        self.lo = (0.0, 0.0, *[float(a) for a, _ in enc])
        self.hi = (1.0, 1.0, *[float(b) for _, b in enc])
        self.names = [d.name for d in self.sweep]

    # --- parameter columns <-> physical values ---------------------------------------------------------------
    def encode(self, values: dict[str, float]) -> list[float]:
        return [float(np.log(values[d.name])) if d.is_log else float(values[d.name]) for d in self.sweep]

    def par(self, X: Tensor) -> dict[str, Tensor | float]:
        """Per-point swept parameters (no gradient needed) + fixed ones."""
        out: dict[str, Tensor | float] = {k: float(v) for k, v in self.params.items()}
        for i, d in enumerate(self.sweep):
            c = X[:, 2 + i].detach()
            out[d.name] = torch.exp(c) if d.is_log else c
        return out

    def _phys(self, par: dict[str, Tensor | float]) -> tuple[Any, Any, Any]:
        Re = par["Re"]
        Pe = par["Pe"] if "Pe" in self.names else Re * par["Pr"]
        return Re, Pe, par.get("Ri", 0.0)

    # --- overrides -------------------------------------------------------------------------------------------
    def transform(self, model: torch.nn.Module, X: Tensor) -> dict[str, Tensor]:
        g = self.gauge
        if not (g["kind"] == "point" and g["mode"] == "hard"):
            return super().transform(model, X)
        saved, self.gauge = self.gauge, {**g, "mode": "soft"}
        try:
            F = super().transform(model, X)
        finally:
            self.gauge = saved
        X0 = torch.cat([self.X0.to(X).expand(X.shape[0], 2), X[:, 2:]], 1)  # gauge point at each row's parameters
        F["p"] = F["p"] - model(X0)[:, self.p_index] + float(g["value"])
        return F

    def fields(self, model: torch.nn.Module, X: Tensor, order: int = 2) -> dict[str, Tensor]:
        d = super().fields(model, X, order)
        d["par"] = self.par(X)
        return d

    def residuals(self, d: dict[str, Tensor]) -> dict[str, Tensor]:
        Re, Pe, Ri = self._phys(d["par"])
        u, v, th = d["u"], d["v"], d["theta"]
        z = torch.zeros_like(u)
        return {
            "mom_x": u * d["u_x"] + v * d["u_y"] + d["p_x"] - (d["u_xx"] + d["u_yy"]) / Re - z,
            "mom_y": u * d["v_x"] + v * d["v_y"] + d["p_y"] - (d["v_xx"] + d["v_yy"]) / Re - Ri * th - z,
            "cont": d["u_x"] + d["v_y"],
            "energy": u * d["theta_x"] + v * d["theta_y"] - (d["theta_xx"] + d["theta_yy"]) / Pe - z,
        }

    def wall_residuals(self, model: torch.nn.Module, walls: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        out = {}
        for w, Xw in walls.items():
            d = self.fields(model, Xw, order=1)
            bc = self.walls[w]
            x, y = d["x"], d["y"]
            nx, ny = P2.NORMAL[w]
            dn = nx * d["theta_x"] + ny * d["theta_y"]
            if w in self.robin_param:
                pname, tinf = self.robin_param[w]
                a = d["par"][pname]
                th_res = a * d["theta"] + dn - a * tinf
            else:
                th_res = bc.alpha * d["theta"] + bc.beta * dn - bc.g(x, y)
            out[w] = {"u": d["u"] - bc.u(x, y), "v": d["v"] - bc.v(x, y), "theta": th_res}
        return out

    def sample(self, n_pde, n_ic, n_bc, method, seed, device, dtype):
        """Interior points with parameters in the TRAINING box; n_bc points per wall, each with its own parameters."""
        k = len(self.sweep)

        def unit(n: int, dim: int, s: int) -> np.ndarray:
            if method == "sobol":
                return sobol_unit(n, dim, s)
            if method == "lhs":
                from scipy.stats import qmc

                return qmc.LatinHypercube(d=dim, seed=s).random(n)
            return np.random.default_rng(s).uniform(size=(n, dim))

        def params_cols(u: np.ndarray) -> np.ndarray:
            uu = u * self.u_lim
            cols = [d.to_phys(uu[:, i]) for i, d in enumerate(self.sweep)]
            return np.stack([np.log(c) if d.is_log else c for c, d in zip(cols, self.sweep, strict=True)], 1)

        mk = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)  # noqa: E731
        u = unit(n_pde, 2 + k, seed)
        pde = np.concatenate([u[:, :2], params_cols(u[:, 2:])], 1)
        walls = {}
        for j, w in enumerate(WALLS):
            uw = unit(n_bc, 1 + k, seed + 2 + j)
            walls[w] = mk(np.concatenate([P2.wall_points(w, uw[:, 0]), params_cols(uw[:, 1:])], 1))
        return {"pde": mk(pde), "walls": walls}

    # --- evaluation at one parameter point ---------------------------------------------------------------------
    def grid_inputs(self, x: np.ndarray, y: np.ndarray, values: dict[str, float], like: Tensor) -> Tensor:
        X, Y = np.meshgrid(x, y)
        P = np.tile(self.encode(values), (X.size, 1))
        return torch.as_tensor(
            np.concatenate([np.stack([X.ravel(), Y.ravel()], 1), P], 1), dtype=like.dtype, device=like.device
        )


def evaluate_point(
    problem: P2.Problem2D,
    model: torch.nn.Module,
    ref: dict[str, Any],
    values: dict[str, float] | None,
    n_wall: int = 200,
) -> dict[str, Any]:
    """Errors vs a reference (keys x, y, u, v, p, theta, nu_top) at one parameter point; ``values`` = None for a
    non-parametric problem. Pressure compared mean-free. Includes max |div u| and Nu_top."""
    p0 = next(model.parameters())
    param = isinstance(problem, ParamProblem2D)
    if param:
        pts = problem.grid_inputs(np.asarray(ref["x"]), np.asarray(ref["y"]), values, p0)
    else:
        X, Y = np.meshgrid(ref["x"], ref["y"])
        pts = torch.as_tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=p0.dtype, device=p0.device)
    pred = P2._batched(problem, model, pts, 1, (*P2.FIELDS, "u_x", "v_y"))
    shape = np.asarray(ref["u"]).shape
    res: dict[str, Any] = {}
    for k in P2.FIELDS:
        r = np.asarray(ref[k])
        pk = pred[k].reshape(shape)
        if k == "p":
            r, pk = r - r.mean(), pk - pk.mean()
        res[f"rel_l2_{k}"] = float(np.linalg.norm(pk - r) / max(np.linalg.norm(r), 1e-300))
        res[f"linf_{k}"] = float(np.abs(pk - r).max())
    res["rel_l2_max"] = max(res["rel_l2_u"], res["rel_l2_v"], res["rel_l2_theta"])
    res["max_div"] = float(np.abs(pred["u_x"] + pred["v_y"]).max())
    xs = np.asarray(ref["x"])
    top = np.stack([xs, np.ones_like(xs)], 1)
    if param:
        top = np.concatenate([top, np.tile(problem.encode(values), (len(xs), 1))], 1)
    tt = torch.as_tensor(top, dtype=p0.dtype, device=p0.device)
    nu = float(problem.fields(model, tt, order=1)["theta_y"].detach().mean())
    res["nu_top"], res["nu_top_ref"] = nu, float(ref["nu_top"])
    res["nu_top_rel_err"] = abs(nu - ref["nu_top"]) / max(abs(ref["nu_top"]), 1e-300)
    return res


def build_param_cavity(
    cavity_cfg: dict, form: dict, sweep: list[Dim], fixed: dict[str, float], u_lim: float
) -> ParamProblem2D:
    """Parametric C_cavity with the Case C wall conditions; Robin walls naming a swept Biot number use it per point."""
    names = [d.name for d in sweep]
    nominal = {**fixed, **{d.name: float(d.to_phys(0.5 * u_lim)) for d in sweep}}
    base_cfg = {**cavity_cfg, "parameters": {**cavity_cfg["parameters"], **nominal}}
    fake = {
        "x": np.array([0.5]),
        "y": np.array([0.5]),
        "nu_top": float("nan"),
        **{k: np.zeros((1, 1)) for k in P2.FIELDS},
    }
    base = P2.build_cavity(base_cfg, form, reference=fake)
    cond = P2.cavity_conditions(cavity_cfg)
    robin = {}
    for w in WALLS:
        t = cond["thermal"][w]
        if t["type"] == "robin" and isinstance(t["Bi"], str) and t["Bi"] in names:
            robin[w] = (t["Bi"], float(t.get("theta_inf", 0.0)))
    pp = ParamProblem2D(
        "C_cavity_param",
        base.params,
        base.walls,
        form,
        None,
        {},
        sweep=tuple(sweep),
        u_lim=float(u_lim),
        robin_param=robin,
    )
    return pp
