"""Phase 6 study: NS validation (Ghia cavity, DFG cylinder), analytic flows, Blasius, panel method,
aerodynamic optimisation under the 8G protocol, panel-data Cp surrogate."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from aerosynapse.numerics import verification as V
from aerosynapse.numerics.boundary_layer import blasius, march_boundary_layer
from aerosynapse.numerics.external import preflight
from aerosynapse.numerics.ns2d import FluidBC, NSSolver, cavity_centerlines, lid_driven_cavity
from aerosynapse.numerics.panel import solve_panel
from aerosynapse.optim.problem import EvalResult
from aerosynapse.optim.protocol import OptimizationStudy
from aerosynapse.optim.variables import DesignSpace, Variable
from aerosynapse.physics.airfoil import cst, max_thickness, naca4
from aerosynapse.physics.regime import guard

DATA = Path(__file__).resolve().parents[2] / "data"


def cavity(res: list[int], ns: list[int]) -> dict[str, Any]:
    g = json.loads((DATA / "ghia1982.json").read_text())
    out: dict[str, Any] = {}
    for re in res:
        for n in ns:
            t0 = time.perf_counter()
            r = lid_driven_cavity(re, n=n, t_end=200.0, tol=1e-5)
            y, u, x, v = cavity_centerlines(r)
            gu, gv = g["u_vertical_centerline"], g["v_horizontal_centerline"]
            mask_v = np.ones(len(gv["coord"]), bool)
            for xc in g["excluded_points"]["v_horizontal_centerline"].get(str(re), []):
                mask_v &= ~np.isclose(np.array(gv["coord"]), xc)
            du = np.interp(gu["coord"], y, u) - np.array(gu[str(re)])
            dv = (np.interp(gv["coord"], x, v) - np.array(gv[str(re)]))[mask_v]
            out[f"re{re}_n{n}"] = {
                "rms_u": float(np.sqrt(np.mean(du**2))),
                "max_u": float(np.abs(du).max()),
                "rms_v": float(np.sqrt(np.mean(dv**2))),
                "max_v": float(np.abs(dv).max()),
                "u_min": float(u.min()),
                "converged": r.converged,
                "steps": r.steps,
                "wall_s": time.perf_counter() - t0,
                "div_max": r.divergence_max,
                "profile": {"y": y.tolist(), "u": u.tolist(), "x": x.tolist(), "v": v.tolist()},
            }
    return out


def analytic_flows() -> dict[str, Any]:
    nu = 0.1
    W = {"bottom": FluidBC("wall"), "top": FluidBC("wall")}
    P = {"left": FluidBC("periodic"), "right": FluidBC("periodic")}
    out: dict[str, Any] = {}
    s = NSSolver(16, 8, 2.0, 1.0, nu, {**P, "bottom": FluidBC("wall"), "top": FluidBC("wall", 1.0)})
    r = s.run(50, 1e-9)
    out["couette"] = float(np.abs(r.centre_velocity()[0] - s.yc).max())
    for n in (8, 16, 32):
        s = NSSolver(n, n, 1.0, 1.0, nu, {**P, **W}, body_force=(1.0, 0.0))
        r = s.run(60, 1e-10)
        ex = s.yc * (1 - s.yc) / (2 * nu)
        out[f"poiseuille_rel_err_n{n}"] = float(np.abs(r.centre_velocity()[0] - ex).max() / ex.max())
    out["poiseuille_order"] = V.observed_order(
        [1 / 8, 1 / 16, 1 / 32], [out[f"poiseuille_rel_err_n{n}"] for n in (8, 16, 32)]
    ).pairwise_orders
    return out


def cylinder(h_list: list[float]) -> dict[str, Any]:
    ref = json.loads((DATA / "cylinder_dfg2d1.json").read_text())
    out: dict[str, Any] = {"reference_CD": ref["C_D"], "regime": guard("cylinder", 20.0)}
    H, Lx, D, Um, nu = 0.41, 2.2, 0.1, 0.3, 1e-3
    for h in h_list:
        ny = int(round(H / h))
        nx = int(round(Lx / h))
        hh = H / ny
        xc = (np.arange(nx) + 0.5) * hh
        yc = (np.arange(ny) + 0.5) * hh
        X, Y = np.meshgrid(xc, yc, indexing="ij")
        solid = (X - 0.2) ** 2 + (Y - 0.2) ** 2 < (D / 2) ** 2
        prof = lambda y: 4 * Um * y * (H - y) / H**2  # noqa: E731
        s = NSSolver(
            nx,
            ny,
            nx * hh,
            H,
            nu,
            {
                "left": FluidBC("inlet", prof),
                "right": FluidBC("outlet"),
                "bottom": FluidBC("wall"),
                "top": FluidBC("wall"),
            },
            solid=solid,
        )
        t0 = time.perf_counter()
        r = s.run(30.0, tol=1e-5)
        U = 2 * Um / 3
        cd = 2 * r.body_force[0] / (U**2 * D)
        out[f"h{h}"] = {
            "nx": nx,
            "ny": ny,
            "C_D": cd,
            "rel_err": (cd - ref["C_D"]) / ref["C_D"],
            "converged": r.converged,
            "t": r.t,
            "wall_s": time.perf_counter() - t0,
            "note": "staircase immersed boundary, first-order geometry representation",
        }
        if h == h_list[-1]:
            uc, vc = r.centre_velocity()
            out["field"] = {"u": uc[::2, ::2].tolist(), "v": vc[::2, ::2].tolist(), "h": 2 * hh}
    return out


def boundary_layer() -> dict[str, Any]:
    eta, fp, fpp0 = blasius()
    out: dict[str, Any] = {"blasius_fpp0_computed": fpp0, "regime": guard("flat_plate", 1e4)}
    for ny in (50, 100, 200):
        xs, y, u = march_boundary_layer(1.0, 1e-4, 1.0, ny=ny, nx=200)
        prof = np.interp(y * np.sqrt(1.0 / (1e-4 * xs[-1])), eta, fp)
        out[f"ny{ny}_max_err"] = float(np.abs(u[-1] - prof).max())
    return out


def panel() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for n in (41, 81, 161):
        X, Y = naca4("0012", n)
        out[f"naca0012_n{n}_cl_alpha4"] = solve_panel(X, Y, 4.0).cl
    X, Y = naca4("0012", 101)
    alphas = np.array([-4, -2, 0, 2, 4, 6.0])
    cls = [solve_panel(X, Y, a).cl for a in alphas]
    out["naca0012_lift_slope_per_rad"] = float(np.polyfit(np.deg2rad(alphas), cls, 1)[0])
    out["thin_airfoil_2pi"] = float(2 * np.pi)
    r = solve_panel(*naca4("2412", 101), 0.0)
    out["naca2412_alpha0"] = {"cl": r.cl, "cm_c4": r.cm_c4}
    out["naca2412_cp"] = {"x": r.xc.tolist(), "y": r.yc.tolist(), "cp": r.cp.tolist(), "alpha": 0.0}
    r4 = solve_panel(*naca4("2412", 101), 4.0)
    out["naca2412_cp_alpha4"] = {"x": r4.xc.tolist(), "y": r4.yc.tolist(), "cp": r4.cp.tolist(), "cl": r4.cl}
    out["limitations"] = "inviscid: zero profile drag, no stall; PG valid M < 0.7"
    return out


def airfoil_space() -> DesignSpace:
    return DesignSpace(
        [
            Variable("au0", 0.10, 0.30),
            Variable("au1", 0.05, 0.35),
            Variable("au2", 0.05, 0.35),
            Variable("al0", -0.30, -0.05),
            Variable("al1", -0.30, 0.10),
            Variable("al2", -0.30, 0.10),
        ]
    )


def airfoil_model(
    d: dict[str, Any], n: int = 81, alpha: float = 4.0, cp_min_lim: float = -2.0, t_min: float = 0.12
) -> EvalResult:
    X, Y = cst(np.array([d["au0"], d["au1"], d["au2"]]), np.array([d["al0"], d["al1"], d["al2"]]), n)
    m = (len(X) + 1) // 2
    gap = np.interp(np.linspace(0.02, 0.98, 50), X[m - 1 :], Y[m - 1 :]) - np.interp(
        np.linspace(0.02, 0.98, 50), X[:m][::-1], Y[:m][::-1]
    )
    r = solve_panel(X, Y, alpha)
    t = max_thickness(X, Y)
    return EvalResult(
        f=[-r.cl],
        g=[1 - t / t_min, float(cp_min_lim - r.cp.min()) / abs(cp_min_lim), float(0.005 - gap.min()) / 0.005],
        info={"cl": r.cl, "t": t, "cp_min": float(r.cp.min())},
    )


def aero_optimisation() -> dict[str, Any]:
    st = OptimizationStudy(
        "airfoil_cl",
        baseline_design={"au0": 0.17, "au1": 0.16, "au2": 0.15, "al0": -0.17, "al1": -0.16, "al2": -0.15},
        space=airfoil_space(),
        objective="max C_L at alpha = 4 deg (inviscid panel model)",
        constraints=["t/c >= 0.12", "Cp_min >= -2.0 (suction-peak proxy, heuristic)", "no surface crossing"],
        constraints_declared=True,
        budget=60,
        stopping="budget (60 panel evaluations)",
        model=airfoil_model,
        trusted=lambda d: airfoil_model(d, n=201),
    )
    rep = st.run("surrogate_tr", "de", seed=0, n_cand=500, pop=12)
    rep["model_note"] = (
        "C_L from an inviscid panel method; drag is not available, so C_L/C_D is NOT optimised. "
        "Cp_min is a heuristic proxy for adverse-pressure-gradient severity, not a stall model."
    )
    return rep


def cp_surrogate(n_train: int = 300, n_test: int = 100) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    sp = airfoil_space()
    t0 = time.perf_counter()
    rows, Zs, rejected = [], [], 0
    while len(rows) < n_train + n_test:
        z = rng.random(sp.dim)
        d = sp.decode(z)
        X, Y = cst(np.array([d["au0"], d["au1"], d["au2"]]), np.array([d["al0"], d["al1"], d["al2"]]), 81)
        m = (len(X) + 1) // 2
        xs = np.linspace(0.02, 0.98, 50)
        if np.min(np.interp(xs, X[m - 1 :], Y[m - 1 :]) - np.interp(xs, X[:m][::-1], Y[:m][::-1])) <= 0.0:
            rejected += 1  # self-intersecting geometry: not a valid airfoil, excluded from the dataset
            continue
        rows.append(solve_panel(X, Y, 4.0).cp)
        Zs.append(z)
    Z = np.array(Zs)
    t_data = time.perf_counter() - t0
    C = np.array(rows)
    from aerosynapse.uq.gp import GaussianProcess

    # POD of Cp fields + GP regression of the coefficients on the 6 CST parameters
    cm = C[:n_train].mean(0)
    _, sv, Vt = np.linalg.svd(C[:n_train] - cm, full_matrices=False)
    k = 8
    coef = (C[:n_train] - cm) @ Vt[:k].T
    gps = [GaussianProcess(noise=1e-6, restarts=1).fit(Z[:n_train], coef[:, i]) for i in range(k)]
    pred = cm + np.column_stack([g.predict(Z[n_train:])[0] for g in gps]) @ Vt[:k]
    err = np.linalg.norm(pred - C[n_train:], axis=1) / np.linalg.norm(C[n_train:], axis=1)
    return {
        "n_train": n_train,
        "n_test": n_test,
        "pod_rank": k,
        "pod_energy": float((sv[:k] ** 2).sum() / (sv**2).sum()),
        "data_gen_s": t_data,
        "rel_l2_median": float(np.median(err)),
        "rel_l2_q90": float(np.quantile(err, 0.9)),
        "rel_l2_max": float(err.max()),
    }


def misc() -> dict[str, Any]:
    pf = preflight()
    return {"openfoam_preflight": {"available": pf.available, "message": pf.message}}
