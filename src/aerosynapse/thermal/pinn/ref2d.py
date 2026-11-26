"""Reference solutions for the 2D PINN problems: steady heated lid-driven cavity by the Phase 1 Newton solver.

The cavity uses the Case C wall conditions (Phase 2, ``configs/data``) at one parameter point (Re, Pr, Bi, Ri). The
steady state is solved by Newton's method with the exact Jacobian (aerosynapse.thermal.numerics.ns_steady); this requires a residual
that is at most quadratic, hence the energy convection scheme must be ``central`` (TVD/upwind limiters are not
quadratic and the coloured-probing Jacobian would be inexact). The reference is solved on ``n`` and ``refine * n``
cells; the fine solution is the truth, and the coarse-fine difference is reported as the reference uncertainty.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from aerosynapse.thermal.data.generators import _thermal_bc, _vel_bc, restrict_cell_average
from aerosynapse.thermal.numerics import fvm2d as fv
from aerosynapse.thermal.numerics import ns_projection as ns
from aerosynapse.thermal.numerics.ns_steady import SteadyNewtonNS

Array = np.ndarray
WALLS = ("left", "right", "bottom", "top")


def cavity_solver(params: dict[str, float], cond: dict[str, Any], n: int) -> ns.MACNavierStokes2D:
    Re, Pr, Ri = float(params["Re"]), float(params["Pr"]), float(params.get("Ri", 0.0))
    en = ns.EnergySpec(
        Pe=Re * Pr,
        scheme="central",
        bc_order=2,
        bcs={w: _thermal_bc(cond["thermal"][w], params) for w in WALLS},
    )
    vbcs = {w: _vel_bc(w, cond["velocity"].get(w, {"profile": "zero"})) for w in WALLS}
    return ns.MACNavierStokes2D(fv.Grid2D(n, n), Re, vbcs, energy=en, Ri=Ri)


def solve_steady(params: dict[str, float], cond: dict[str, Any], n: int, tol: float = 1e-10) -> dict[str, Any]:
    """Steady Newton solve from rest; if Newton fails, continuation in Re (Re/8, Re/4, Re/2, Re)."""
    t0 = time.perf_counter()
    Re = float(params["Re"])
    s = cavity_solver(params, cond, n)
    N = SteadyNewtonNS(s)
    x = N.pack(s.u, s.v, None, np.zeros((n, n)))
    info = N.solve(x, tol=tol, max_iter=40)
    ramp: list[float] = []
    if not info["converged"]:
        x = N.pack(np.zeros_like(s.u), np.zeros_like(s.v), None, np.zeros((n, n)))
        for r in (Re / 8, Re / 4, Re / 2, Re):
            s.Re = r
            info = N.solve(x, tol=tol, max_iter=40)
            x = N.x
            ramp.append(r)
    if not info["converged"]:
        raise RuntimeError(
            f"steady Newton reference did not converge at n={n}: |R| = {info['residual_history'][-1]:.3e}"
        )
    p = N.store()
    uc, vc = s.cell_velocity()
    flat = s.theta.ravel()
    return {
        "u": uc,
        "v": vc,
        "p": p - p.mean(),
        "theta": s.theta.copy(),
        "nu_top": -float(np.mean(s.energy.wall_gradient(flat, s.t, "top"))),
        "nu_bottom": -float(np.mean(s.energy.wall_gradient(flat, s.t, "bottom"))),
        "x": s.g.xc.copy(),
        "y": s.g.yc.copy(),
        "newton_iterations": info["iterations"],
        "newton_residual": info["residual_history"][-1],
        "continuation": ramp,
        "max_div": s.max_div(),
        "wall_s": time.perf_counter() - t0,
    }


def solve_steady_decoupled(
    params: dict[str, float], cond: dict[str, Any], n: int, tol: float = 1e-10, scheme: str = "quick"
) -> dict[str, Any]:
    """Ri = 0 only (passive scalar): the flow by steady Newton (momentum only, exact Jacobian; continuation in Re if
    needed), then the energy equation on the frozen flow with a LINEAR scheme (default QUICK, upwind-biased: bounded
    at higher cell Peclet numbers than central) solved directly; second-order walls (bc_order 2). Step 4.5 (D65)."""
    if float(params.get("Ri", 0.0)) != 0.0:
        raise ValueError("solve_steady_decoupled needs Ri = 0 (temperature must not act on the flow)")
    t0 = time.perf_counter()
    Re, Pr = float(params["Re"]), float(params["Pr"])
    vbcs = {w: _vel_bc(w, cond["velocity"].get(w, {"profile": "zero"})) for w in WALLS}
    s = ns.MACNavierStokes2D(fv.Grid2D(n, n), Re, vbcs)
    N = SteadyNewtonNS(s)
    info = N.solve(N.pack(s.u, s.v, None, None), tol=tol, max_iter=40)
    ramp: list[float] = []
    if not info["converged"]:
        x = N.pack(np.zeros_like(s.u), np.zeros_like(s.v), None, None)
        for r in (Re / 8, Re / 4, Re / 2, Re):
            s.Re = r
            info = N.solve(x, tol=tol, max_iter=40)
            x = N.x
            ramp.append(r)
    if not info["converged"]:
        raise RuntimeError(f"steady Newton (flow) did not converge at n={n}: |R| = {info['residual_history'][-1]:.3e}")
    p = N.store()
    E = fv.FVEnergy2D(
        s.g,
        1.0 / (Re * Pr),
        fv.FaceVelocity.from_arrays(s.g, s.u, s.v),
        bcs={w: _thermal_bc(cond["thermal"][w], params) for w in WALLS},
        scheme=scheme,
        bc_order=2,
    )
    th = E.steady()
    uc, vc = s.cell_velocity()
    bal = E.energy_balance(th, 0.0)
    return {
        "u": uc,
        "v": vc,
        "p": p - p.mean(),
        "theta": th.reshape(n, n),
        "nu_top": -float(np.mean(E.wall_gradient(th, 0.0, "top"))),
        "nu_bottom": -float(np.mean(E.wall_gradient(th, 0.0, "bottom"))),
        "x": s.g.xc.copy(),
        "y": s.g.yc.copy(),
        "newton_iterations": info["iterations"],
        "newton_residual": info["residual_history"][-1],
        "continuation": ramp,
        "max_div": s.max_div(),
        "energy_balance": bal,
        "wall_s": time.perf_counter() - t0,
    }


def _rel(a: Array, b: Array) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def cavity_reference(
    params: dict[str, float],
    cond: dict[str, Any],
    n: int,
    refine: int = 2,
    tol: float = 1e-10,
    solver: Any = None,
) -> dict[str, Any]:
    """Fine (refine*n) solution as truth + coarse-fine differences (cell-average restriction of the fine field).

    The coarse-fine difference is used as a CONSERVATIVE reference-uncertainty indicator. A Richardson estimate
    (difference / (refine^2 - 1)) is deliberately not reported: on the default cosh50-lid cavity the observed
    convergence (32/64/128) is below second order, so the asymptotic estimate would be optimistic (D46)."""
    solve = solver or solve_steady  # solver(params, cond, n, tol); default: coupled Newton (central energy)
    coarse = solve(params, cond, n, tol)
    fine = solve(params, cond, refine * n, tol)
    diff = {k: _rel(coarse[k], restrict_cell_average(fine[k], refine)) for k in ("u", "v", "theta")}
    pr = restrict_cell_average(fine["p"], refine)
    diff["p"] = _rel(coarse["p"], pr - pr.mean())
    diff["nu_top"] = abs(coarse["nu_top"] - fine["nu_top"]) / max(abs(fine["nu_top"]), 1e-300)
    return {
        **{k: fine[k] for k in ("u", "v", "p", "theta", "x", "y", "nu_top", "nu_bottom")},
        "n_coarse": n,
        "n_fine": refine * n,
        "coarse_fine_diff": diff,
        "newton": {
            "coarse": {
                k: coarse[k] for k in ("newton_iterations", "newton_residual", "continuation", "max_div", "wall_s")
            },
            "fine": {k: fine[k] for k in ("newton_iterations", "newton_residual", "continuation", "max_div", "wall_s")},
        },
    }
