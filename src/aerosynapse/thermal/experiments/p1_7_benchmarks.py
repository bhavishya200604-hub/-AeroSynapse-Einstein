"""Step 1.7: validation benchmarks + three-grid GCI (Celik procedure, Fs = 1.25).

(A) Lid-driven cavity (uniform lid) vs Ghia et al. 1982 Tables I/II (VERIFIED-SECONDARY), steady Newton
    with Re continuation; GCI of psi_min, u_min(x=0.5), v_max/v_min(y=0.5); Richardson-extrapolated profiles.
(B) Schäfer–Turek 2D-1 (Re=20, steady Newton) C_D, C_L, dp vs FEATFLOW reference (VERIFIED); staircase cylinder.
(C) Schäfer–Turek 2D-2 (Re=100, unsteady) C_D,max, C_L,max, St vs FEATFLOW reference (VERIFIED).
(D) Graetz (parallel plates, FV energy with prescribed Poiseuille flow): CWT and CHF fully developed Nu vs
    the values DERIVED in src/physics/graetz.py (spectral eigenvalue incl. axial conduction; exact 140/17).
(E) de Vahl Davis natural convection (Pr=0.71) mean Nu vs de Vahl Davis / Hortmann (VERIFIED-SECONDARY).

Run:  python experiments/p1_7_benchmarks.py --profile smoke|full
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from aerosynapse.thermal.analysis import benchmarks as bm
from aerosynapse.thermal.numerics import fvm2d as fv
from aerosynapse.thermal.numerics import ns_projection as ns
from aerosynapse.thermal.numerics.convergence import gci, richardson
from aerosynapse.thermal.numerics.ns_steady import SteadyNewtonNS
from aerosynapse.thermal.physics import graetz as gz
from aerosynapse.thermal.utils.experiment import Experiment, cli_profile
from aerosynapse.thermal.viz.style import SEQ_CMAP, apply_style, save_fig

PROFILES = {
    "smoke": {
        "cavity_N": [32, 64, 128],
        "cavity_Re": [100, 400, 1000],
        "st_k": [1, 2],
        "st2_k": 1,
        "st2_T": 60.0,
        "graetz_ny": [16, 32, 64],
        "dvd_N": [32, 64, 128],
        "dvd_Ra": [1e4],
    },
    "full": {
        "cavity_N": [64, 128, 256],
        "cavity_Re": [100, 400, 1000, 3200],
        "st_k": [1, 2, 4],
        "st2_k": 2,
        "st2_T": 60.0,
        "graetz_ny": [16, 32, 64, 128],
        "dvd_N": [64, 128, 256],
        "dvd_Ra": [1e4, 1e5, 1e6],
    },
}


# --------------------------------------------------------------------------- helpers
def continuation(
    N: SteadyNewtonNS, x: np.ndarray, setter, targets: list[float], start: float, log: list
) -> tuple[np.ndarray, dict]:
    """Solve at each target parameter, inserting geometric midpoints when Newton fails."""
    results = {}
    cur = start
    for tgt in targets:
        todo = [tgt]
        while todo:
            val = todo[0]
            setter(val)
            info = N.solve(x, tol=1e-9, max_iter=25)
            log.append(
                {
                    "param": val,
                    "iterations": info["iterations"],
                    "converged": info["converged"],
                    "residual": info["residual_history"][-1],
                    "jac_dropped": info["jac_max_dropped"],
                }
            )
            if info["converged"]:
                x = N.x.copy()
                cur = val
                todo.pop(0)
                if val == tgt:
                    results[tgt] = info
            else:
                mid = float(np.sqrt(cur * val))
                if abs(mid - cur) < 1e-3 * cur:
                    raise RuntimeError(f"continuation stalled at {val}")
                todo.insert(0, mid)
    return x, results


def gci_record(Ns: list[int], vals: list[float]) -> dict:
    """Three finest grids, h = 1/N; f1 = finest."""
    h = [1.0 / n for n in Ns[-3:]]
    f = vals[-3:]
    return gci(f[2], f[1], f[0], h[2], h[1], h[0])


# --------------------------------------------------------------------------- (A) Ghia cavity
def cavity_quantities(s: ns.MACNavierStokes2D) -> dict:
    n = s.g.nx
    yu = np.concatenate([[0.0], s.g.yc, [1.0]])
    uu = np.concatenate([[0.0], s.u[:, n // 2], [1.0]])
    xv = np.concatenate([[0.0], s.g.xc, [1.0]])
    vv = np.concatenate([[0.0], s.v[n // 2, :], [0.0]])
    su, sv = CubicSpline(yu, uu), CubicSpline(xv, vv)
    dense = np.linspace(0, 1, 4001)
    psi = s.streamfunction()
    j, i = np.unravel_index(np.argmin(psi), psi.shape)
    return {
        "psi_min": float(psi[j, i]),
        "psi_min_xy": [float(s.g.xf[i]), float(s.g.yf[j])],
        "u_min": float(su(dense).min()),
        "v_max": float(sv(dense).max()),
        "v_min": float(sv(dense).min()),
        "max_div": s.max_div(),
        "_su": su,
        "_sv": sv,
    }


def ghia_compare(su, sv, Re: int) -> dict:
    out = {}
    for which, spl in (("u", su), ("v", sv)):
        c, ref, m = bm.ghia_points(Re, which)
        d = spl(c) - ref
        out[which] = {
            "max_abs_dev": float(np.max(np.abs(d[m]))),
            "rms_dev": float(np.sqrt(np.mean(d[m] ** 2))),
            "n_points": int(m.sum()),
            "excluded": int((~m).sum()),
            "dev_at_points": d.tolist(),
        }
    return out


def cavity_study(prof: dict, ex: Experiment) -> tuple[dict, dict]:
    Ns, Res = prof["cavity_N"], prof["cavity_Re"]
    sols: dict = {}
    out: dict = {"Newton_log": {}}
    for n in Ns:
        s = ns.MACNavierStokes2D(fv.Grid2D(n, n), float(Res[0]), {"top": ns.wall(u=1.0)})
        N = SteadyNewtonNS(s)
        x = N.pack(s.u, s.v, None, None)
        log: list = []
        t0 = time.perf_counter()

        def setter(Re: float, s: ns.MACNavierStokes2D = s) -> None:
            s.Re = Re

        prev = 10.0  # Newton from rest; midpoints inserted automatically if a jump fails
        for Re in Res:
            x, res = continuation(N, x, setter, [float(Re)], prev, log)
            prev = float(Re)
            N.x = x
            N.store()
            q = cavity_quantities(s)
            sols[(Re, n)] = q
            ex.log(
                study="cavity",
                Re=Re,
                N=n,
                **{k: v for k, v in q.items() if not k.startswith("_") and not isinstance(v, list)},
            )
        out["Newton_log"][f"N{n}"] = {"steps": log, "wall_s": time.perf_counter() - t0}
    for Re in Res:
        rec: dict = {}
        for key in ("psi_min", "u_min", "v_max", "v_min"):
            vals = [sols[(Re, n)][key] for n in Ns]
            rec[key] = {"values": vals, "N": Ns, "gci": gci_record(Ns, vals)}
        fine, mid = sols[(Re, Ns[-1])], sols[(Re, Ns[-2])]
        rec["ghia_finest"] = ghia_compare(fine["_su"], fine["_sv"], Re)
        r = Ns[-1] / Ns[-2]
        su_ext = lambda y, f=fine, m=mid, r=r: richardson(f["_su"](y), m["_su"](y), r, 2.0)  # noqa: E731
        sv_ext = lambda x_, f=fine, m=mid, r=r: richardson(f["_sv"](x_), m["_sv"](x_), r, 2.0)  # noqa: E731
        rec["ghia_richardson"] = ghia_compare(su_ext, sv_ext, Re)
        rec["psi_min_xy_finest"] = fine["psi_min_xy"]
        rec["max_div_finest"] = fine["max_div"]
        out[f"Re{Re}"] = rec
    return out, sols


# --------------------------------------------------------------------------- (B,C) Schäfer–Turek
def st_solver(k: int, Re: float, convective_outflow: bool = False) -> ns.MACNavierStokes2D:
    """Nondimensional with D = 0.1 m and U_mean = 0.2 m/s: domain 22 x 4.1, cylinder r = 0.5 at (2, 2).
    convective_outflow=True for the unsteady 2D-2 case (the zero-gradient outflow is unstable with backflow)."""
    ny, nx = 41 * k, 220 * k
    g = fv.Grid2D(nx, ny, 22.0, 4.1)
    solid = (g.Xc - 2.0) ** 2 + (g.Yc - 2.0) ** 2 < 0.25
    H = 4.1
    prof = lambda s_, t: 4 * 1.5 * s_ * (H - s_) / H**2  # noqa: E731  (U_max = 1.5 U_mean)
    return ns.MACNavierStokes2D(g, Re, {"left": ns.wall(u=prof), "right": ns.outflow(convective_outflow)}, solid=solid)


def st_pressure_difference(s: ns.MACNavierStokes2D, p: np.ndarray) -> float:
    """p(1.5, 2) - p(2.5, 2): average the two rows adjacent to y = 2 and extrapolate linearly from the
    two nearest fluid cells to the (staircase) surface points."""
    g = s.g
    jrow = int(round(2.0 / g.dy))  # y = 2 is the face between rows jrow-1 and jrow
    prow = 0.5 * (p[jrow - 1, :] + p[jrow, :])
    solid_row = s.solid[jrow - 1, :] | s.solid[jrow, :]
    xs = g.xc
    front = np.flatnonzero(~solid_row & (xs < 2.0))[-2:]
    back = np.flatnonzero(~solid_row & (xs > 2.0))[:2]

    def extrap(idx: np.ndarray, x0: float) -> float:
        x1, x2 = xs[idx]
        p1, p2 = prow[idx]
        return float(p1 + (p2 - p1) * (x0 - x1) / (x2 - x1))

    return extrap(front, 1.5) - extrap(back, 2.5)


def st1_study(ks: list[int], ex: Experiment) -> dict:
    ref = bm.SCHAFER_TUREK_2D1
    out: dict = {"reference": {k: ref[k] for k in ("C_D", "C_L", "dp", "source")}}
    vals: dict = {"C_D": [], "C_L": [], "dp": []}
    for k in ks:
        s = st_solver(k, 20.0)
        N = SteadyNewtonNS(s)
        t0 = time.perf_counter()
        info = N.solve(N.pack(s.u, s.v, None, None), tol=1e-9)
        p = N.store()
        Fx, Fy = s.body_force(p)
        CD, CL = 2 * Fx, 2 * Fy  # U_mean = 1, D = 1, rho = 1
        dp = st_pressure_difference(s, p) * ref["rho"] * ref["U_mean"] ** 2  # back to Pa
        rec = {
            "k": k,
            "cells_per_D": 10 * k,
            "C_D": CD,
            "C_L": CL,
            "dp": dp,
            "newton_iterations": info["iterations"],
            "residual": info["residual_history"][-1],
            "wall_s": time.perf_counter() - t0,
            "max_div": s.max_div(),
            "rel_err_C_D": (CD - ref["C_D"]) / ref["C_D"],
            "rel_err_C_L": (CL - ref["C_L"]) / ref["C_L"],
            "rel_err_dp": (dp - ref["dp"]) / ref["dp"],
            "solid_cells": int(s.solid.sum()),
            "staircase_area_over_pi_r2": float(s.solid.sum() * s.g.vol / (np.pi * 0.25)),
        }
        out[f"k{k}"] = rec
        for q in vals:
            vals[q].append(rec[q])
        ex.log(study="schafer_turek_2d1", **{a: b for a, b in rec.items() if not isinstance(b, dict)})
    if len(ks) >= 3:
        hs = [1.0 / (10 * k) for k in ks]
        out["gci"] = {q: gci(v[-1], v[-2], v[-3], hs[-1], hs[-2], hs[-3]) for q, v in vals.items()}
    return out


def st2_study(k: int, T: float, ex: Experiment) -> tuple[dict, dict]:
    ref = bm.SCHAFER_TUREK_2D2
    s = st_solver(k, 100.0, convective_outflow=True)
    s.set_state(np.zeros_like(s.u), np.zeros_like(s.v))
    hist: dict = {"t": [], "CD": [], "CL": []}

    def cb(so: ns.MACNavierStokes2D, h: float) -> None:
        if so.t > T / 3 and so.nsteps % 2 == 0:
            Fx, Fy = so.body_force()
            hist["t"].append(so.t)
            hist["CD"].append(2 * Fx)
            hist["CL"].append(2 * Fy)

    t0 = time.perf_counter()
    info = s.run(T, callback=cb)
    t, CD, CL = (np.array(hist[q]) for q in ("t", "CD", "CL"))
    # last 5 periods from upward zero crossings of C_L
    zc = np.flatnonzero((CL[:-1] < 0) & (CL[1:] >= 0))
    tz = t[zc] - CL[zc] * (t[zc + 1] - t[zc]) / (CL[zc + 1] - CL[zc])
    rec: dict = {
        "k": k,
        "steps": info["steps"],
        "wall_s": time.perf_counter() - t0,
        "max_div": info["max_div"],
        "reference": {q: ref[q] for q in ("C_D_max", "C_L_max", "St", "source")},
    }
    if tz.size >= 3:
        periods = np.diff(tz)[-5:]
        f = 1.0 / periods.mean()
        last = t >= tz[-min(6, tz.size)]
        rec.update(
            St=float(f * 1.0 / 1.0),
            period_std_rel=float(periods.std() / periods.mean()),
            C_D_max=float(CD[last].max()),
            C_L_max=float(CL[last].max()),
            n_periods=int(periods.size),
        )
        rec.update(
            rel_err_St=(rec["St"] - ref["St"]) / ref["St"],
            rel_err_C_D_max=(rec["C_D_max"] - ref["C_D_max"]) / ref["C_D_max"],
            rel_err_C_L_max=(rec["C_L_max"] - ref["C_L_max"]) / ref["C_L_max"],
        )
    else:
        rec["note"] = "no periodic shedding detected within T"
    ex.log(study="schafer_turek_2d2", **{a: b for a, b in rec.items() if not isinstance(b, dict)})
    return rec, {"t": t, "CD": CD, "CL": CL, "solver": s}


# --------------------------------------------------------------------------- (D) Graetz
def graetz_fv(ny: int, Pe: float, kind: str, bc_order: int = 2) -> dict:
    Lx = 8.0
    nx = int(Lx * ny)
    g = fv.Grid2D(nx, ny, Lx, 1.0)
    vel = fv.FaceVelocity.from_point_function(g, lambda X, Y, t: (6 * Y * (1 - Y), 0 * X))
    if kind == "cwt":
        bcs = {
            "left": fv.dirichlet2(1.0),
            "right": fv.neumann2(0.0),
            "bottom": fv.dirichlet2(0.0),
            "top": fv.dirichlet2(0.0),
        }
    else:
        bcs = {
            "left": fv.dirichlet2(0.0),
            "right": fv.neumann2(0.0),
            "bottom": fv.neumann2(1.0),
            "top": fv.neumann2(1.0),
        }
    p = fv.FVEnergy2D(g, 1.0 / Pe, vel, bcs=bcs, scheme="central", bc_order=bc_order)
    th = p.steady()
    T = th.reshape(ny, nx)
    uc = 6 * g.yc * (1 - g.yc)
    tb = (uc @ T) / uc.sum()
    grad = 0.5 * (p.wall_gradient(th, 0.0, "bottom") + p.wall_gradient(th, 0.0, "top"))
    half = 0.5 * (p.wall_diffusive_flux(th, 0.0, "bottom") + p.wall_diffusive_flux(th, 0.0, "top"))
    bv = p.boundary_values(T, 0.0)
    tw = 0.5 * (bv["bottom"] + bv["top"])
    Nu2 = 2 * np.abs(grad) / np.abs(tb - tw)
    Nu1 = 2 * np.abs(half) / np.abs(tb - tw)
    station = 0.6 * Lx
    i = int(np.argmin(np.abs(g.xc - station)))
    return {
        "Nu_2nd_order_grad": float(Nu2[i]),
        "Nu_half_cell_flux": float(Nu1[i]),
        "x_station": float(g.xc[i]),
        "theta_b_station": float(tb[i]),
        "cell_Pe_x": float(1.5 * g.dx * Pe),
        "_x": g.xc,
        "_Nu": Nu2,
    }


def graetz_study(nys: list[int], ex: Experiment) -> tuple[dict, dict]:
    Pe = 20.0
    ref_cwt = gz.nu_cwt_spectral(64, Pe)
    out: dict = {
        "Pe_H": Pe,
        "Nu_cwt_spectral_axial": ref_cwt["Nu"],
        "Nu_cwt_spectral_no_axial": gz.nu_cwt_spectral(64)["Nu"],
        "Nu_chf_exact": float(gz.nu_chf_symbolic()),
        "Nu_chf_exact_str": str(gz.nu_chf_symbolic()),
    }
    curves = {}
    for (kind, ref), bco in [(kr, b) for b in (1, 2) for kr in (("cwt", ref_cwt["Nu"]), ("chf", out["Nu_chf_exact"]))]:
        recs = [graetz_fv(ny, Pe, kind, bco) for ny in nys]
        for ny, r in zip(nys, recs, strict=True):
            ex.log(
                study="graetz",
                kind=kind,
                bc_order=bco,
                ny=ny,
                **{a: b for a, b in r.items() if not a.startswith("_")},
            )
        vals = [r["Nu_2nd_order_grad"] for r in recs]
        key = kind if bco == 2 else f"{kind}_bc1"  # bc_order=2 is the primary (D30)
        out[key] = {
            "bc_order": bco,
            "err_vs_reference": [abs(v - ref) for v in vals],
            "pairwise_order_vs_reference": np.log2(
                np.array([abs(v - ref) for v in vals[:-1]]) / np.array([abs(v - ref) for v in vals[1:]])
            ).tolist(),
            "ny": nys,
            "Nu": vals,
            "Nu_half_cell": [r["Nu_half_cell_flux"] for r in recs],
            "reference": ref,
            "rel_err_finest": (vals[-1] - ref) / ref,
            "gci": gci_record(nys, vals),
            "x_station": recs[-1]["x_station"],
            "cell_Pe_x": [r["cell_Pe_x"] for r in recs],
        }
        if bco == 2:
            curves[kind] = (recs[-1]["_x"], recs[-1]["_Nu"], ref)
    return out, curves


# --------------------------------------------------------------------------- (E) de Vahl Davis
def dvd_study(Ns: list[int], Ras: list[float], ex: Experiment, bc_order: int = 2) -> tuple[dict, dict]:
    Pr = 0.71
    out: dict = {
        "bc_order": bc_order,
        "Pr": Pr,
        "reference": {str(k): v for k, v in bm.DE_VAHL_DAVIS_NU.items()},
        "hortmann": {str(k): v for k, v in bm.HORTMANN_NU.items()},
    }
    fields: dict = {}
    nus: dict = {Ra: [] for Ra in Ras}
    for n in Ns:
        en = ns.EnergySpec(
            Pe=np.sqrt(1e3 * Pr),
            bcs={
                "left": fv.dirichlet2(1.0),
                "right": fv.dirichlet2(0.0),
                "bottom": fv.neumann2(0.0),
                "top": fv.neumann2(0.0),
            },
            bc_order=bc_order,
        )
        s = ns.MACNavierStokes2D(fv.Grid2D(n, n), np.sqrt(1e3 / Pr), {}, energy=en, Ri=1.0)
        N = SteadyNewtonNS(s)
        x = N.pack(s.u, s.v, None, 1.0 - s.g.Xc)

        def setter(Ra: float, s: ns.MACNavierStokes2D = s) -> None:
            s.Re = float(np.sqrt(Ra / Pr))
            s.energy.D = 1.0 / float(np.sqrt(Ra * Pr))
            s.energy_spec.Pe = float(np.sqrt(Ra * Pr))

        log: list = []
        cur = 1e3
        for Ra in Ras:
            x, _ = continuation(N, x, setter, [Ra], cur, log)
            cur = Ra
            N.x = x
            N.store()
            th = s.theta.ravel()
            nu_hot = -float(np.mean(s.energy.wall_gradient(th, 0.0, "left")))
            nu_cold = -float(np.mean(s.energy.wall_gradient(th, 0.0, "right")))
            q = s.energy.boundary_heat_flow(th, 0.0)
            Pe = np.sqrt(Ra * Pr)
            rec = {
                "Ra": Ra,
                "N": n,
                "Nu_hot_2nd": nu_hot,
                "Nu_cold_2nd": nu_cold,
                "Nu_conservative_flux": q["left"] * Pe,
                "heat_balance_rel": (q["left"] + q["right"]) / q["left"],
                "max_div": s.max_div(),
                "u_max_alpha_units": float(np.max(s.u[:, n // 2]) * Pe),
                "v_max_alpha_units": float(np.max(s.v[n // 2, :]) * Pe),
            }
            nus[Ra].append(nu_hot)
            out[f"Ra{Ra:g}_N{n}"] = rec
            ex.log(study="de_vahl_davis", bc_order=bc_order, **rec)
            fields[(Ra, n)] = (s.g, s.theta.copy(), s.streamfunction())
        out[f"newton_log_N{n}"] = log
    for Ra in Ras:
        ref = bm.DE_VAHL_DAVIS_NU[Ra]
        out[f"Ra{Ra:g}"] = {
            "N": Ns,
            "Nu": nus[Ra],
            "gci": gci_record(Ns, nus[Ra]),
            "rel_err_finest_vs_dVD": (nus[Ra][-1] - ref) / ref,
            "rel_err_finest_vs_Hortmann": (nus[Ra][-1] - bm.HORTMANN_NU[Ra]) / bm.HORTMANN_NU[Ra],
        }
    return out, fields


# --------------------------------------------------------------------------- figures
def figures(
    cav: dict,
    sols: dict,
    st2: dict,
    graetz_curves: dict,
    dvd_fields: dict,
    prof: dict,
    ex: Experiment,
    tag: str,
) -> list[str]:
    paths = []
    Ns, Res = prof["cavity_N"], prof["cavity_Re"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    yy = np.linspace(0, 1, 400)
    for Re in Res:
        q = sols[(Re, Ns[-1])]
        c, ref, m = bm.ghia_points(Re, "u")
        (ln,) = axs[0].plot(q["_su"](yy), yy, label=f"Re={Re}")
        axs[0].plot(ref[m], c[m], "o", color=ln.get_color(), mfc="none", ms=4)
        axs[0].plot(ref[~m], c[~m], "x", color="r", ms=6)
        c, ref, m = bm.ghia_points(Re, "v")
        axs[1].plot(yy, q["_sv"](yy), color=ln.get_color(), label=f"Re={Re}")
        axs[1].plot(c[m], ref[m], "o", color=ln.get_color(), mfc="none", ms=4)
        axs[1].plot(c[~m], ref[~m], "x", color="r", ms=6)
    axs[0].set(xlabel="$u(0.5, y)$", ylabel="$y$", title=f"Cavity, {Ns[-1]}$^2$ (o: Ghia 1982, x: excluded)")
    axs[1].set(xlabel="$x$", ylabel="$v(x, 0.5)$", title="Horizontal centreline")
    axs[0].legend(fontsize=7)
    paths += save_fig(fig, f"p1_7_ghia_{tag}", ex.dir)

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.3))
    if "solver" in st2:
        s = st2["solver"]
        w = s.vorticity()
        sp_ = np.pad(s.solid, 1, constant_values=False)
        inside = sp_[:-1, :-1] & sp_[1:, :-1] & sp_[:-1, 1:] & sp_[1:, 1:]
        w = np.where(inside, np.nan, w)
        axs[0].pcolormesh(s.g.xf, s.g.yf, w, cmap="RdBu_r", vmin=-5, vmax=5)
        axs[0].set(aspect="equal", xlim=(0, 12), title=f"2D-2 vorticity, t*={s.t:.0f}", xlabel="$x/D$")
        axs[1].plot(st2["t"], st2["CL"], label="$C_L$")
        axs[1].plot(st2["t"], st2["CD"] - 3, label="$C_D - 3$")
        axs[1].set(xlabel="$t U/D$", title="2D-2 force coefficients")
        axs[1].legend(fontsize=7)
    for kind, (xg, nu, ref) in graetz_curves.items():
        (ln,) = axs[2].plot(xg / (2 * 40.0), nu, label=f"FV {kind.upper()}")
        axs[2].axhline(ref, color=ln.get_color(), ls="--", lw=0.8)
    axs[2].set(
        xlabel=r"$x^+ = x/(D_h Pe_{D_h})$",
        ylabel="$Nu_{D_h}$",
        ylim=(0, 20),
        title="Graetz, Pe$_H$=20 (dashed: derived limits)",
    )
    axs[2].legend(fontsize=7)
    paths += save_fig(fig, f"p1_7_st_graetz_{tag}", ex.dir)

    keys = sorted(dvd_fields)
    fig, axs = plt.subplots(1, len({k[0] for k in keys}), figsize=(4 * len({k[0] for k in keys}), 3.6), squeeze=False)
    for ax, Ra in zip(axs[0], sorted({k[0] for k in keys}), strict=True):
        n = max(k[1] for k in keys if k[0] == Ra)
        g, th, psi = dvd_fields[(Ra, n)]
        im = ax.contourf(g.xc, g.yc, th, 20, cmap=SEQ_CMAP)
        ax.contour(g.xf, g.yf, psi, 12, colors="w", linewidths=0.5)
        ax.set(aspect="equal", title=f"de Vahl Davis Ra={Ra:g}, {n}$^2$", xlabel="$x$", ylabel="$y$")
        fig.colorbar(im, ax=ax, label=r"$\theta$")
    paths += save_fig(fig, f"p1_7_dvd_{tag}", ex.dir)
    return [str(p) for p in paths]


def strip(d):
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if not (isinstance(k, str) and k.startswith("_"))}
    return d


def main() -> None:
    profile = cli_profile()
    prof = PROFILES[profile]
    apply_style()
    with Experiment(f"p1_7_benchmarks_{profile}", {"seed": 0, "profile": profile, **prof}) as ex:
        t0 = time.perf_counter()
        cav, sols = cavity_study(prof, ex)
        print(f"[cavity done {time.perf_counter() - t0:.0f}s]", flush=True)
        st1 = st1_study(prof["st_k"], ex)
        print(f"[2D-1 done {time.perf_counter() - t0:.0f}s]", flush=True)
        st2, st2_hist = st2_study(prof["st2_k"], prof["st2_T"], ex)
        print(f"[2D-2 done {time.perf_counter() - t0:.0f}s]", flush=True)
        gr, gr_curves = graetz_study(prof["graetz_ny"], ex)
        dvd, dvd_fields = dvd_study(prof["dvd_N"], prof["dvd_Ra"], ex, bc_order=2)
        dvd1, _ = dvd_study(prof["dvd_N"], prof["dvd_Ra"], ex, bc_order=1)
        print(f"[dVD done {time.perf_counter() - t0:.0f}s]", flush=True)
        ex.metrics.update(
            strip(
                {
                    "cavity": cav,
                    "schafer_turek_2d1": st1,
                    "schafer_turek_2d2": st2,
                    "graetz": gr,
                    "de_vahl_davis": dvd,
                    "de_vahl_davis_bc1": {k: v for k, v in dvd1.items() if k.startswith("Ra") and "_N" not in k},
                }
            )
        )
        ex.metrics["figures"] = figures(cav, sols, st2_hist, gr_curves, dvd_fields, prof, ex, profile)
    m = ex.metrics
    for Re in prof["cavity_Re"]:
        r = m["cavity"][f"Re{Re}"]
        print(
            f"Re={Re}: psi_min={[f'{v:.6f}' for v in r['psi_min']['values']]} p={r['psi_min']['gci'].get('p')}"
            f" GCI={r['psi_min']['gci'].get('GCI_fine21')} | Ghia u max|dev|={r['ghia_finest']['u']['max_abs_dev']:.4f}"
            f" v={r['ghia_finest']['v']['max_abs_dev']:.4f} | Richardson u={r['ghia_richardson']['u']['max_abs_dev']:.4f}"
            f" v={r['ghia_richardson']['v']['max_abs_dev']:.4f}"
        )
    for k in prof["st_k"]:
        r = m["schafer_turek_2d1"][f"k{k}"]
        print(
            f"2D-1 k={k}: CD={r['C_D']:.4f} ({100 * r['rel_err_C_D']:+.2f}%) CL={r['C_L']:.5f} "
            f"({100 * r['rel_err_C_L']:+.1f}%) dp={r['dp']:.5f} ({100 * r['rel_err_dp']:+.2f}%) its={r['newton_iterations']}"
            f" area_ratio={r['staircase_area_over_pi_r2']:.4f}"
        )
    print(
        "2D-2:",
        {
            a: (round(b, 5) if isinstance(b, float) else b)
            for a, b in m["schafer_turek_2d2"].items()
            if not isinstance(b, dict)
        },
    )
    for kind in ("cwt", "chf", "cwt_bc1", "chf_bc1"):
        r = m["graetz"][kind]
        print(
            f"Graetz {kind}: Nu={[f'{v:.5f}' for v in r['Nu']]} ref={r['reference']:.5f} "
            f"err={100 * r['rel_err_finest']:+.3f}% order_vs_ref={np.round(r['pairwise_order_vs_reference'], 3)} "
            f"GCI p={r['gci'].get('p')}"
        )
    for tag in ("de_vahl_davis", "de_vahl_davis_bc1"):
        for Ra in prof["dvd_Ra"]:
            r = m[tag][f"Ra{Ra:g}"]
            print(
                f"{tag} Ra={Ra:g}: Nu={[f'{v:.4f}' for v in r['Nu']]} vs dVD {100 * r['rel_err_finest_vs_dVD']:+.2f}% "
                f"Hortmann {100 * r['rel_err_finest_vs_Hortmann']:+.2f}% p={r['gci'].get('p')} GCI={r['gci'].get('GCI_fine21')}"
            )


if __name__ == "__main__":
    main()
