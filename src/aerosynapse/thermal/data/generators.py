"""Snapshot generators (ground truth) for Cases A, B (analytic) and C (2D cavity + heat transfer, numerical).

All conditions come from the resolved spec (``aerosynapse.thermal.data.spec``): parameter values per point, wall velocities, wall
thermal conditions, initial state, grid, scheme, snapshot times. Layout: Case A/B theta[p, t, x]; Case C
theta/u/v/p[p, t, y, x] on cell centres (velocities averaged from MAC faces, pressure zero-mean).
nu_top / nu_bottom = minus the mean inward wall gradient (positive when the wall is hotter than the adjacent fluid;
only a meaningful Nusselt number for walls held at a fixed temperature).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerosynapse.thermal.data.spec import eval_profile
from aerosynapse.thermal.numerics import fvm2d as fv
from aerosynapse.thermal.numerics import ns_projection as ns
from aerosynapse.thermal.physics import analytic as an

Array = NDArray[np.float64]


def snapshot_times(spec: dict[str, Any]) -> Array:
    if spec["kind"] == "linear":
        return np.linspace(spec["t0"], spec["t1"], int(spec["n"]))
    if spec["kind"] == "geom":
        return np.geomspace(spec["t0"], spec["t1"], int(spec["n"]))
    return np.asarray(spec["values"], float)


def gen_case_a(Pe: Array, cfg: dict[str, Any]) -> dict[str, Array]:
    """Gaussian pulse (analytic, infinite line) and steady boundary-layer profile for each Pe."""
    x = np.linspace(0.0, 1.0, int(cfg["nx"]))
    t = snapshot_times(cfg["times"])
    pu, st = cfg["conditions"]["pulse"], cfg["conditions"]["steady"]
    pulse = np.stack(
        [
            pu.get("baseline", 0.0)
            + pu.get("amplitude", 1.0)
            * an.gaussian_pulse(x[None, :], t[:, None], float(p), pu["sigma0"], pu["x0"], pu["c"])
            for p in Pe
        ]
    )
    lo, hi = st["theta_left"], st["theta_right"]
    steady = np.stack([lo + (hi - lo) * an.steady_convdiff(x, float(p)) for p in Pe])
    return {"x": x, "t": t, "theta": pulse, "theta_steady": steady}


def gen_case_b(Bi: Array, cfg: dict[str, Any]) -> dict[str, Array]:
    """Slab with Robin BC, source and fin term: eigen-series ground truth in Fourier time."""
    x = np.linspace(0.0, 1.0, int(cfg["nx"]))
    t = snapshot_times(cfg["times"])
    s = cfg["conditions"]["slab"]
    n_terms = int(s.get("n_terms", 400))
    theta = np.stack(
        [
            [an.slab_transient(x, float(tt), float(b), s["Q"], s["m"], s["theta0"], n_terms=n_terms) for tt in t]
            for b in Bi
        ]
    )
    steady = np.stack([an.slab_steady(x, float(b), s["Q"], s["m"]) for b in Bi])
    return {"x": x, "t": t, "theta": theta, "theta_steady": steady}


def _thermal_bc(spec: dict[str, Any], params: dict[str, float]) -> fv.FaceBC:
    kind = spec["type"]
    if kind == "dirichlet":
        return fv.dirichlet2(float(spec.get("value", 0.0)))
    if kind == "neumann":
        return fv.neumann2(float(spec.get("value", 0.0)))
    b = spec["Bi"]
    return fv.robin2(float(params[b]) if isinstance(b, str) else float(b), float(spec.get("theta_inf", 0.0)))


def _vel_bc(wall: str, vel: dict[str, Any]) -> ns.VelBC:
    f = lambda s, t, v=vel: eval_profile(v, s, t)  # noqa: E731
    return ns.wall(u=f) if wall in ("top", "bottom") else ns.wall(v=f)


def gen_case_c_point(
    params: dict[str, float], times: Array, cfg: dict[str, Any], grid: int | None = None
) -> dict[str, Any]:
    """One cavity run with the conditions of ``cfg`` (spec case dict) and the point's parameters.

    params: values for Re, Pr, Bi, Ri (all required; fixed ones included). Energy: Pe = Re Pr.
    """
    n = int(grid or cfg["grid"])
    cond = cfg["conditions"]
    Re, Pr, Ri = float(params["Re"]), float(params["Pr"]), float(params.get("Ri", 0.0))
    en = ns.EnergySpec(
        Pe=Re * Pr,
        scheme=cfg["scheme"],
        bc_order=int(cfg["bc_order"]),
        bcs={w: _thermal_bc(cond["thermal"][w], params) for w in ("top", "bottom", "left", "right")},
    )
    vbcs = {w: _vel_bc(w, cond["velocity"].get(w, {"profile": "zero"})) for w in ("top", "bottom", "left", "right")}
    s = ns.MACNavierStokes2D(fv.Grid2D(n, n), Re, vbcs, energy=en, Ri=Ri)
    s.set_state(np.zeros_like(s.u), np.zeros_like(s.v), theta=np.full((n, n), float(cond.get("initial_theta", 0.0))))
    th, uu, vv, pp, nu_top, nu_bot, ke = [], [], [], [], [], [], []
    nu_left, nu_right = [], []
    t0 = time.perf_counter()
    steps = 0
    max_div = 0.0
    for tk in times:
        info = s.run(float(tk))
        steps += info["steps"]
        max_div = max(max_div, info["max_div"])
        uc, vc = s.cell_velocity()
        p = s.pressure()
        th.append(s.theta.copy())
        uu.append(uc)
        vv.append(vc)
        pp.append(p - p.mean())
        flat = s.theta.ravel()
        # Nu_wall = -mean(inward wall gradient); sign convention: positive when heat flows from the wall into the fluid
        nu_top.append(-float(np.mean(s.energy.wall_gradient(flat, s.t, "top"))))
        nu_bot.append(-float(np.mean(s.energy.wall_gradient(flat, s.t, "bottom"))))
        nu_left.append(-float(np.mean(s.energy.wall_gradient(flat, s.t, "left"))))
        nu_right.append(-float(np.mean(s.energy.wall_gradient(flat, s.t, "right"))))
        ke.append(s.kinetic_energy())
    g = s.g
    return {
        "theta": np.array(th),
        "u": np.array(uu),
        "v": np.array(vv),
        "p": np.array(pp),
        "nu_top": np.array(nu_top),
        "nu_bottom": np.array(nu_bot),
        "nu_left": np.array(nu_left),  # side walls (Step 10.4: differentially heated cavity); not in gen_case_c
        "nu_right": np.array(nu_right),
        "ke": np.array(ke),
        "xc": g.xc,
        "yc": g.yc,
        "steps": steps,
        "max_div": max_div,
        "wall_s": time.perf_counter() - t0,
    }


def gen_case_c(table: dict[str, Array], cfg: dict[str, Any]) -> dict[str, Any]:
    """table: parameter name -> array (N,) (swept values per point, fixed ones broadcast)."""
    times = snapshot_times(cfg["times"])
    n_pts = len(next(iter(table.values())))
    runs = [gen_case_c_point({k: float(v[i]) for k, v in table.items()}, times, cfg) for i in range(n_pts)]
    out: dict[str, Any] = {
        k: np.stack([r[k] for r in runs])
        for k in ("theta", "u", "v", "p", "nu_top", "nu_bottom", "ke", "steps", "max_div", "wall_s")
    }
    out.update({"x": runs[0]["xc"], "y": runs[0]["yc"], "t": times})
    return out


def restrict_cell_average(f: Array, factor: int) -> Array:
    """Average factor x factor blocks of the last two axes (fine -> coarse cell-centred restriction)."""
    *lead, ny, nx = f.shape
    return f.reshape(*lead, ny // factor, factor, nx // factor, factor).mean(axis=(-3, -1))
