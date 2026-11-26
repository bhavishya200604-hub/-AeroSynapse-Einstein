"""CFD-driven design optimisation of the main wing: Bayesian optimisation with a Gaussian-process surrogate.

Every evaluation is a full 3D flow run (``flow3d.run_setup``, stopped once its mean forces settle), so the
budget is counted in CFD runs. The loop:

1. the baseline design plus a space-filling set (Latin hypercube) of wings;
2. a Gaussian process (Matern 5/2 kernel with one length scale per variable, plus a noise term for the solver's
   scatter) is fitted to the objective;
3. the next wing is the one with the largest Expected Improvement over the best so far (searched over random
   candidates, then polished with L-BFGS-B);
4. repeat until the budget is spent.

Variables (each optional, with its own range): sweep, taper ratio, tip twist, dihedral, and span (at constant wing
area: the root chord follows). Objectives: largest L/D, least drag, most lift; an optional lowest CL acts as a
penalty. The surrogate's length scales say which variables matter most.

Outputs (folder): ``history.csv`` (every run), ``best_design.json``, ``summary.txt``, ``result.json`` and figures:
convergence, the objective against each variable, the surrogate map over the two most important variables, the
planforms, and the flow round the baseline and the best wing (two-panel pictures).
"""

from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from aerosynapse.sim.common import NEON, finish, plt, save

VARS: dict[str, tuple[str, str]] = {  # key: (label, unit)
    "sweep": ("leading-edge sweep", "deg"),
    "taper": ("taper ratio", ""),
    "twist": ("tip twist", "deg"),
    "dihedral": ("dihedral", "deg"),
    "span": ("span (constant area)", "x baseline"),
}


def wing_of(design: dict[str, Any]) -> dict[str, Any]:
    from aerosynapse.structures.wing import main_wing

    return main_wing(design)


def apply(design: dict[str, Any], x: dict[str, float], keep_area: bool = True) -> dict[str, Any]:
    """A copy of the design with the main wing changed to the variables ``x`` (span as a factor on the baseline)."""
    d = copy.deepcopy(design)
    w = wing_of(d)
    b0, c0, l0 = float(w["span"]), float(w["root_chord"]), float(w.get("taper", 0.5))
    area = 0.5 * b0 * c0 * (1 + l0)
    for k in ("sweep", "twist", "dihedral"):
        if k in x:
            w[k] = float(x[k])
    if "taper" in x:
        w["taper"] = float(x["taper"])
    if "span" in x:
        w["span"] = b0 * float(x["span"])
    if keep_area:
        w["root_chord"] = 2 * area / (float(w["span"]) * (1 + float(w.get("taper", 0.5))))
    return d


def planform(w: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Outline of one half wing seen from above (x along the chord, y along the span)."""
    t = w.get("translate", [0, 0, 0])
    b, cr, lam = float(w["span"]), float(w["root_chord"]), float(w.get("taper", 0.5))
    sw = np.radians(float(w.get("sweep", 30)))
    x0, y0 = float(t[0]), float(t[1])
    xt = x0 + b * np.tan(sw)
    return np.array([x0, xt, xt + lam * cr, x0 + cr, x0]), np.array([y0, y0 + b, y0 + b, y0, y0])


def lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    u = (rng.random((n, d)) + np.arange(n)[:, None]) / n
    for j in range(d):
        u[:, j] = u[rng.permutation(n), j]
    return u


class Surrogate:
    """Gaussian process on the unit cube, minimising."""

    def __init__(self, dim: int, seed: int = 0):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

        k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.4] * dim, length_scale_bounds=(0.04, 5.0), nu=2.5) + WhiteKernel(1e-3, (1e-7, 0.2))
        self.gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=6, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Surrogate:
        import warnings

        with warnings.catch_warnings():  # hyper-parameters at a bound are normal with few runs
            warnings.simplefilter("ignore")
            self.gp.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m, s = self.gp.predict(np.atleast_2d(X), return_std=True)
        return m, np.maximum(s, 1e-9)

    def length_scales(self) -> np.ndarray:
        return np.atleast_1d(self.gp.kernel_.k1.k2.length_scale)

    def expected_improvement(self, X: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
        from scipy.stats import norm

        m, s = self.predict(X)
        imp = best - m - xi * np.abs(best)
        z = imp / s
        return imp * norm.cdf(z) + s * norm.pdf(z)

    def next_point(self, best: float, rng: np.random.Generator, dim: int, taken: np.ndarray) -> np.ndarray:
        from scipy.optimize import minimize

        cand = rng.random((6000, dim))
        ei = self.expected_improvement(cand, best)
        starts = cand[np.argsort(ei)[-6:]]
        top, top_ei = starts[-1], float(ei.max())
        for x0 in starts:
            r = minimize(lambda x: -float(self.expected_improvement(x[None, :], best)[0]), x0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * dim)
            if -r.fun > top_ei:
                top, top_ei = np.clip(r.x, 0, 1), -float(r.fun)
        if len(taken) and np.min(np.linalg.norm(taken - top, axis=1)) < 0.02:  # never repeat a run: explore instead
            m, s = self.predict(cand)
            top = cand[int(np.argmax(s))]
        return np.asarray(top)


def _score(objective: str, lift: float, drag: float, cl_min: float) -> float:
    """The quantity minimised."""
    if objective == "min drag":
        f = drag
    elif objective == "max lift":
        f = -lift
    else:
        f = -lift / drag if drag > 1e-6 else 0.0
    if cl_min > 0 and lift < cl_min:  # a soft constraint: lose 30 % of the objective's scale per 10 % below the target
        f += 3.0 * (cl_min - lift) / cl_min * max(abs(f), 1e-3)
    return float(f)


def run(design: dict[str, Any], setup: dict[str, Any] | None, out: str | Path) -> dict[str, Any]:
    from aerosynapse.sim import flow3d, flowviz
    from aerosynapse.sim import setup as SU

    S = SU.merged("optimise", setup)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(S["seed"]))
    w0 = wing_of(design)
    base_vals = {"sweep": float(w0.get("sweep", 30)), "taper": float(w0.get("taper", 0.5)), "twist": float(w0.get("twist", 0)),
                 "dihedral": float(w0.get("dihedral", 0)), "span": 1.0}
    names = [k for k in VARS if bool(S[f"vary_{k}"])]
    if not names:
        raise ValueError("choose at least one variable to vary")
    lo = np.array([float(S[f"{k}_min"]) for k in names])
    hi = np.array([float(S[f"{k}_max"]) for k in names])
    if np.any(hi <= lo):
        raise ValueError("each variable needs its largest value above its smallest")
    dim = len(names)
    objective, cl_min, keep_area = str(S["objective"]), float(S["cl_min"]), bool(S["keep_area"])
    budget, n0 = int(S["budget"]), max(2, min(int(S["initial_runs"]), int(S["budget"]) - 1))
    cfd = {"preset": "custom", "mach": float(S["mach"]), "alpha": float(S["alpha"]), "beta": 0.0, "altitude": float(S["altitude"]),
           "gust_amplitude": 0.0, "turbulence_intensity": 0.0, "quality": str(S["fidelity"]), "unsteady": "off",
           "start": "free stream", "force_band": float(S["force_band"]), "save_volume": True, "flow_pictures": "none", "full_model": True}
    # full model: at zero sideslip a half model should agree, but on the draft Falcon grid it gave C_L 0.068 / C_D 0.0296
    # against 0.0435 / 0.0107 for the full aircraft (KI-68), so the optimiser does not rely on the symmetry plane

    def to_x(u: np.ndarray) -> dict[str, float]:
        return {k: float(lo[i] + u[i] * (hi[i] - lo[i])) for i, k in enumerate(names)}

    def to_u(x: dict[str, float]) -> np.ndarray:
        return np.array([np.clip((x[k] - lo[i]) / (hi[i] - lo[i]), 0, 1) for i, k in enumerate(names)])

    rows: list[dict[str, Any]] = []
    U: list[np.ndarray] = []
    Y: list[float] = []
    log = out / "history.csv"

    def evaluate(u: np.ndarray, why: str) -> None:
        n = len(rows)
        x = to_x(u) if why != "baseline" else {k: base_vals[k] for k in names}
        d = out / "runs" / f"{n:02d}"
        t0 = time.time()
        row: dict[str, Any] = {"run": n, "why": why, **x}
        try:
            r = flow3d.run_setup(apply(design, x, keep_area), cfd, d)
            c = r.get("coefficients") or {}
            CL, CD = float(c.get("CL", np.nan)), float(c.get("CD", np.nan))
            ok = bool(np.isfinite(CL) and np.isfinite(CD))
            walls = int(r.get("wall_cells", 0) or 0) + int(r.get("thin_surface_cells", 0) or 0)
            row.update(CL=CL, CD=CD, LD=CL / CD if ok and CD > 1e-6 else np.nan, steps=r.get("steps"), stop=r.get("stop_reason"), ok=ok,
                       wall_cells=walls, cells=r.get("cells"))
        except Exception as e:  # a failed run counts as a poor design, not as the end of the study
            row.update(CL=np.nan, CD=np.nan, LD=np.nan, ok=False, error=str(e)[-200:])
        row["seconds"] = time.time() - t0
        row["score"] = _score(objective, row["CL"], row["CD"], cl_min) if row["ok"] else np.nan
        rows.append(row)
        U.append(to_u(x) if why == "baseline" else u)
        Y.append(row["score"])
        with open(log, "w", newline="", encoding="utf-8") as fh:
            keys = ["run", "why", *names, "CL", "CD", "LD", "score", "steps", "stop", "cells", "wall_cells", "seconds", "ok"]
            wr = csv.DictWriter(fh, keys, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        print(f"run {n:2d} ({why}): " + ", ".join(f"{k} {x[k]:.3g}" for k in names) + f" -> CL {row['CL']:.4f} CD {row['CD']:.5f} L/D {row['LD']:.2f}  ({row['seconds']:.0f} s)", flush=True)

    t_all = time.time()
    evaluate(np.zeros(dim), "baseline")
    for u in lhs(n0 - 1, dim, rng):
        evaluate(u, "space-filling")
    gp = None
    while len(rows) < budget:
        y = np.array(Y, float)
        good = np.isfinite(y)
        if good.sum() < 2:
            evaluate(rng.random(dim), "random")
            continue
        worst = float(np.nanmax(y))
        yf = np.where(good, y, worst + abs(worst) * 0.5 + 1e-3)
        gp = Surrogate(dim, int(S["seed"])).fit(np.array(U), yf)
        evaluate(gp.next_point(float(np.min(yf)), rng, dim, np.array(U)), "expected improvement")
    y = np.array(Y, float)
    if not np.isfinite(y).any():
        raise RuntimeError("no CFD run of the study finished")
    ib = int(np.nanargmin(y))
    best, base = rows[ib], rows[0]
    best_x = {k: best[k] for k in names}
    best_design = apply(design, best_x, keep_area)
    (out / "best_design.json").write_text(json.dumps(best_design, indent=1), encoding="utf-8")
    good = np.isfinite(y)
    gp = Surrogate(dim, int(S["seed"])).fit(np.array(U)[good], y[good]) if good.sum() >= 3 else gp
    ls = gp.length_scales() if gp is not None else np.ones(dim)
    importance = (1 / ls) / np.sum(1 / ls)

    # ---------------------------------------------------------------- figures
    figs: list[str] = []
    pl = plt()
    label_obj = {"max L/D": "L/D", "min drag": "$C_D$", "max lift": "$C_L$"}[objective] if objective in ("max L/D", "min drag", "max lift") else objective
    val = np.array([r["LD"] if objective == "max L/D" else r["CD"] if objective == "min drag" else r["CL"] for r in rows], float)
    fig, ax = pl.subplots(figsize=(6.4, 3.4))
    n = np.arange(len(rows))
    best_val = np.array([val[int(np.nanargmin(np.where(np.isfinite(y[: i + 1]), y[: i + 1], np.inf)))] for i in range(len(rows))])
    cols = [NEON[1] if r["why"] == "baseline" else NEON[2] if r["why"] == "space-filling" else NEON[0] for r in rows]
    ax.scatter(n, val, c=cols, s=22, zorder=3, lw=0)
    ax.step(n, best_val, where="post", color="#111111", lw=1.2, label="best so far")
    ax.axhline(val[0], color=NEON[1], lw=0.8, ls="--", label="baseline")
    for c, lab in ((NEON[2], "space-filling runs"), (NEON[0], "expected-improvement runs")):
        ax.scatter([], [], c=c, s=22, label=lab)
    ax.set(xlabel="CFD run", ylabel=label_obj, title=f"Optimisation history: {objective}")
    ax.legend(frameon=False, fontsize=8)
    figs.append(save(fig, out, "1_optimisation_history"))

    fig, axs = pl.subplots(1, dim, figsize=(2.6 * dim + 0.6, 2.9), squeeze=False)
    for i, k in enumerate(names):
        a = axs[0, i]
        xs = np.array([r[k] for r in rows], float)
        a.scatter(xs, val, c=cols, s=18, lw=0, zorder=3)
        a.scatter([best[k]], [val[ib]], s=70, facecolors="none", edgecolors="#111111", lw=1.2, zorder=4)
        if gp is not None:  # the surrogate along this variable, others at the best
            g = np.linspace(0, 1, 120)
            P = np.repeat(to_u(best_x)[None, :], len(g), 0)
            P[:, i] = g
            m, s = gp.predict(P)
            gx = lo[i] + g * (hi[i] - lo[i])
            mm = m if objective == "min drag" else -m  # the surrogate models the minimised score
            a.plot(gx, mm, color="#555555", lw=1.0)
            a.fill_between(gx, mm - 2 * s, mm + 2 * s, color="#999999", alpha=0.18, lw=0)
        lab, unit = VARS[k]
        a.set(xlabel=f"{lab}" + (f" [{unit}]" if unit else ""), title=f"weight {100 * importance[i]:.0f} %")
        if i == 0:
            a.set_ylabel(label_obj)
    fig.suptitle("Objective against each variable (line: surrogate through the best wing, band: 2 sigma)")
    figs.append(save(fig, out, "2_variables"))

    if gp is not None and dim >= 2:
        i1, i2 = (int(j) for j in np.argsort(-importance)[:2])
        g = np.linspace(0, 1, 90)
        G1, G2 = np.meshgrid(g, g)
        P = np.repeat(to_u(best_x)[None, :], G1.size, 0)
        P[:, i1], P[:, i2] = G1.ravel(), G2.ravel()
        m, _s = gp.predict(P)
        Mv = (-m if objective != "min drag" else m).reshape(G1.shape)
        fig, ax = pl.subplots(figsize=(5.6, 4.2))
        X1, X2 = lo[i1] + G1 * (hi[i1] - lo[i1]), lo[i2] + G2 * (hi[i2] - lo[i2])
        from aerosynapse.thermal.viz import style as TS

        cf = ax.contourf(X1, X2, Mv, 18, cmap=TS.SEQ_CMAP if objective != "min drag" else TS.SEQ_CMAP + "_r")
        ax.contour(X1, X2, Mv, 9, colors="white", linewidths=0.4, alpha=0.6)
        TS.colorbar(fig, cf, ax, label_obj + " (surrogate)")
        ax.scatter([r[names[i1]] for r in rows], [r[names[i2]] for r in rows], c="white", edgecolors="#111111", s=20, lw=0.6, zorder=3)
        ax.scatter([best[names[i1]]], [best[names[i2]]], marker="*", s=160, c=NEON[1], edgecolors="#111111", lw=0.6, zorder=4, label="best")
        ax.scatter([base[names[i1]]], [base[names[i2]]], marker="s", s=40, c="white", edgecolors=NEON[1], lw=1.2, zorder=4, label="baseline")
        l1, u1 = VARS[names[i1]]
        l2, u2 = VARS[names[i2]]
        ax.set(xlabel=l1 + (f" [{u1}]" if u1 else ""), ylabel=l2 + (f" [{u2}]" if u2 else ""), title="Surrogate map over the two most important variables")
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.grid(False)
        figs.append(save(fig, out, "3_surrogate_map"))

    fig, ax = pl.subplots(figsize=(5.8, 3.6))
    for wdict, c, lab in ((wing_of(design), NEON[1], "baseline"), (wing_of(best_design), NEON[0], "best")):
        px, py = planform(wdict)
        ax.plot(py, -px, color=c, lw=1.4, label=lab)
    ax.set(aspect="equal", xlabel="y [m]", ylabel="-x [m]", title="Half-wing planforms (wing area " + ("kept" if keep_area else "free") + ")")
    ax.legend(frameon=False)
    figs.append(save(fig, out, "4_planforms"))
    for nm, rr in (("5_flow_baseline", base), ("6_flow_best", best)):
        rd = out / "runs" / f"{rr['run']:02d}"
        try:
            dz = flowviz._design(rd)
            if dz is not None:
                yst, xle, c = flowviz.wing_station(dz, 0.55)
                f = flowviz.figure(rd, ("y", yst), nm, f"{'Baseline' if rr is base else 'Best'} wing at 55 % span", (xle - 1.0 * c, xle + 3.0 * c, -0.8 * c, 0.8 * c))
                (rd / f).replace(out / f)
                figs.append(f)
        except Exception as e:
            print(f"flow picture {nm}: {e}")

    secs = time.time() - t_all
    gain = (val[ib] - val[0]) / abs(val[0]) * 100 if np.isfinite(val[0]) and val[0] != 0 else float("nan")
    walls = [int(r.get("wall_cells") or 0) for r in rows if r.get("ok")]
    coarse = bool(walls) and min(walls) < 120
    lines = ([f"WARNING: as few as {min(walls)} cells touch the aircraft (below about 120 the forces can be wrong by a factor of 2 to 6); the ranking of wings "
              "may come from grid error, not aerodynamics. Use a finer grid of each run before acting on this result."] if coarse else []) + [f"Objective: {objective}" + (f", with CL at least {cl_min:g}" if cl_min > 0 else "") + f"; Mach {S['mach']}, alpha {S['alpha']} deg, altitude {S['altitude']} m",
             f"{len(rows)} CFD runs ({S['fidelity']} grid, each stopped when its mean forces settled within {S['force_band']} %), {secs / 60:.1f} min in all",
             f"baseline: CL {base['CL']:.4f}, CD {base['CD']:.5f}, L/D {base['LD']:.2f}",
             f"best (run {ib}): CL {best['CL']:.4f}, CD {best['CD']:.5f}, L/D {best['LD']:.2f}  ({gain:+.1f} % {label_obj.strip('$')})",
             "best wing: " + ", ".join(f"{VARS[k][0]} {best[k]:.3g}" + (f" {VARS[k][1]}" if VARS[k][1] else "") for k in names),
             "variable weights from the surrogate (1 / length scale): " + ", ".join(f"{k} {100 * importance[i]:.0f} %" for i, k in enumerate(names)),
             "Read with care: forces from the chosen grid carry its discretisation error. On the Falcon (mesh study 2026-11-11) "
             "drag converges from 48 cells per length (GCI 4.4 % on 96) but lift does not: within 2.6 % on 34-68, then 18 % "
             "lower on 96 (thin-surface steps, KI-84); compare wings on one grid only "
             "and confirm the best wing on a finer one (Studio: open best_design.json and run CFD)."]
    res = {"objective": objective, "variables": names, "bounds": {k: [float(lo[i]), float(hi[i])] for i, k in enumerate(names)}, "runs": rows,
           "best_run": ib, "best": best, "baseline": base, "gain_percent": gain, "importance": dict(zip(names, importance.tolist(), strict=True)),
           "seconds": secs, "figures": figs, "min_wall_cells": min(walls) if walls else None, "grid_warning": coarse}
    return finish(out, "CFD design optimisation (Bayesian, Gaussian process)", lines, res)
