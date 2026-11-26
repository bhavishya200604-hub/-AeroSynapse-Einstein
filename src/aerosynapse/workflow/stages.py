"""The eight stages of the design workflow. Each takes the project and a profile and returns its summary."""

from __future__ import annotations

import copy
import json
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from aerosynapse.cad import analyse as A
from aerosynapse.cad import kernel
from aerosynapse.workflow.project import KEYS, ROOT, STAGES, Project

RESULTS = ROOT / "results"

PROFILES: dict[str, dict[str, Any]] = {
    "quick": {"design_budget": 60, "generic_budget": 40, "wing_budget": 1200, "body_budget": 48, "pinn_levels": [250, 500, 1000, 2000], "pinn_lbfgs": 40,
              "cfd_h": [0.45, 0.32, 0.225], "cfd_steps": 700, "mesh_h": 0.05, "step_h": 0.12, "body_h": 0.12, "threads": 4},
    "standard": {"design_budget": 300, "generic_budget": 120, "wing_budget": 5000, "body_budget": 160, "pinn_levels": [400, 800, 1600, 3200, 6400], "pinn_lbfgs": 100,
                 "cfd_h": [0.32, 0.225, 0.16], "cfd_steps": 1500, "mesh_h": 0.03, "step_h": 0.08, "body_h": 0.09, "threads": 4},
}

NAVY, CYAN, STEEL, ORANGE = "#16325c", "#0096d6", "#7d8ea8", "#e07a1f"


# ------------------------------------------------------------------ helpers
def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as p

    p.rcParams.update({"font.size": 9, "axes.edgecolor": "#55657d", "axes.grid": True, "grid.color": "#dfe6ef", "axes.titleweight": "bold", "axes.titlesize": 10,
                       "figure.dpi": 110, "savefig.bbox": "tight", "figure.constrained_layout.use": True})
    return p


def save_fig(fig: Any, path: Path) -> str:
    fig.savefig(path)
    plt().close(fig)
    return path.name


def calib() -> float:
    f = RESULTS / "fighter" / "calibration.json"
    return float(json.loads(f.read_text())["calib"]) if f.exists() else 1.0


def family_of(p: Project, design: dict[str, Any]) -> str | None:
    return p.meta["requirements"]["family"] if "variables" in design else None


def fighter_perf(fam: str, variables: dict[str, float]) -> Any:
    from aerosynapse.fighter import design as D
    from aerosynapse.fighter import performance as P

    return P.evaluate(fam, D.complete(fam, {k: float(v) for k, v in variables.items() if isinstance(v, (int, float))}), P.Mission(calib=calib()))


def perf_summary(r: Any) -> dict[str, Any]:
    keys = ("cruise_LD", "cruise_drag_N", "dash_drag_N", "turn_rate_sustained_dps", "Ps_climb", "takeoff_m", "supercruise_margin", "specific_range_km_per_kg")
    return {"W0_kg": r.W0, "empty_kg": r.empty, "fuel_kg": r.fuel, "fuel_capacity_kg": r.fuel_capacity, "feasible": r.feasible, "CD0": r.CD0, "static_margin": r.static_margin,
            "metrics": {k: r.metrics[k] for k in keys if k in r.metrics}, "violated": {k: v for k, v in r.constraints.items() if v > 0}}


def scorecard(design: dict[str, Any], h: float = 0.08) -> dict[str, Any]:
    """The same quick figures for every stage's design, so progress is comparable."""
    a = A.analyse(design, h=h)
    g = a["geometry"]
    lat = a.get("lattice", {}).get("M0.3", {})
    return {"volume_m3": g["volume"], "wetted_area_m2": g["wetted_area"], "frontal_area_m2": g["frontal_area"], "length_m": g["length"],
            "ref_area_m2": a["reference"]["S"], "span_m": a["reference"]["b"], "friction_CD0": a["friction_CD0_estimate"],
            "wave_CD_M1.2": a["wave_drag_CD"].get("M1.2"), "wave_CD_M1.6": a["wave_drag_CD"].get("M1.6"), "wave_D_over_q_M1.6_m2": a["wave_drag_D_over_q_m2"].get("M1.6"),
            "span_efficiency": lat.get("span_efficiency"), "CL_alpha_per_rad": lat.get("CL_alpha_per_rad"), "static_margin_centroid": lat.get("static_margin_about_centroid")}


def batch(fun: Callable[[np.ndarray], float], Z: np.ndarray, threads: int) -> np.ndarray:
    with ThreadPoolExecutor(threads) as ex:
        return np.array(list(ex.map(fun, list(Z))))


def main_wing(design: dict[str, Any]) -> dict[str, Any]:
    from aerosynapse.structures.wing import main_wing as mw

    return mw(design)


def reference(design: dict[str, Any], length: float) -> dict[str, Any]:
    f = main_wing(design)
    span, cr, lam = float(f["span"]), float(f["root_chord"]), float(f.get("taper", 0.5))
    t = f.get("translate", [0, 0, 0])
    y0 = float(t[1])
    S = 2 * 0.5 * (1 + lam) * cr * span + 2 * y0 * cr
    cmac = 2 / 3 * cr * (1 + lam + lam**2) / (1 + lam)
    ymac = span * (1 + 2 * lam) / (3 * (1 + lam))
    xle = float(t[0]) + ymac * np.tan(np.radians(float(f.get("sweep", 30))))
    return {"length": length, "area": S, "chord": cmac, "span": 2 * (span + y0), "moment_point": [xle + 0.25 * cmac, 0.0, 0.0]}


# ------------------------------------------------------------------ 1 concept
def concept(p: Project, prof: dict[str, Any]) -> dict[str, Any]:
    from aerosynapse.fighter import design as D

    inp = p.dir / "input" / "design.json"
    req = p.meta["requirements"]
    if inp.exists():
        design = json.loads(inp.read_text(encoding="utf-8"))
        origin = p.meta.get("origin", "input body")
    else:
        fam = req["family"]
        base = json.loads((RESULTS / "fighter" / "baseline_design.json").read_text())["variables"]
        fixed = D.FAMILIES[fam]["fixed"]
        v = D.complete(fam, {k: float(x) for k, x in base.items() if k not in fixed})
        design = D.build(v, name=p.name)
        origin = f"{fam} family baseline (reference fighter variables)"
    design["name"] = p.name
    chk = kernel.check(design)
    if not chk.get("ok"):
        raise RuntimeError(f"the design does not build: {chk.get('errors')}")
    p.log(f"concept: {origin}, {len(design['features'])} features")
    res: dict[str, Any] = {"origin": origin, "features": [{"id": f.get("id"), "type": f.get("type"), "op": f.get("op", "add")} for f in design["features"]]}
    fam = family_of(p, design)
    if fam:
        r = fighter_perf(fam, design["variables"])
        res["performance"] = perf_summary(r)
        p.log(f"concept: take-off mass {r.W0:.0f} kg, feasible {r.feasible}")
    res["scorecard"] = scorecard(design)
    info = kernel.info(design, h=prof["step_h"])
    res["mass_properties"] = {k: info[k] for k in ("volume", "wetted_area", "frontal_area", "planform_area", "side_area", "length", "centroid", "bbox_lo", "bbox_hi")}
    p.save_design("concept", design, origin)
    return res


# ------------------------------------------------------------------ 2 design optimisation
def design_opt(p: Project, prof: dict[str, Any]) -> dict[str, Any]:
    from aerosynapse.fighter import design as D
    from aerosynapse.fighter import optimise as O

    design, _ = p.design_before("design")
    fam = family_of(p, design)
    out = p.stage_dir("design")
    res: dict[str, Any] = {"scorecard_before": scorecard(design)}
    if fam:
        r0 = fighter_perf(fam, design["variables"])
        ref = O.reference_terms(r0)
        job = {"family": fam, "calib": calib(), "reference": ref, "seed": 0, "budget": prof["design_budget"], "start": design["variables"], "sigma": 0.15, "restarts": 0}
        p.log(f"design: CMA-ES over the {fam} design space, {prof['design_budget']} evaluations of the full performance model")
        best = O.run(job)
        r1 = fighter_perf(fam, best["variables"])
        J0, _ = O.objective(r0, ref)
        new = D.build(D.complete(fam, best["variables"]), name=p.name)
        res.update({"method": "fighter design space, CMA-ES, full performance model (weights, drag, turn, climb, gust, extreme states)", "family": fam,
                    "J_before": J0, "J_after": best["J"], "terms_after": best["terms"], "evaluations": best["evaluations"], "history": best["history"],
                    "performance_before": perf_summary(r0), "performance_after": perf_summary(r1),
                    "changed_variables": {k: [design["variables"].get(k), v] for k, v in best["variables"].items() if abs(float(design["variables"].get(k, v)) - float(v)) > 1e-6}})
        p.log(f"design: J {J0:.3f} -> {best['J']:.3f}, take-off mass {r0.W0:.0f} -> {r1.W0:.0f} kg")
        hist = np.array(best["history"], float) if best["history"] else np.zeros((0, 2))
    else:
        # an input body without design-space variables: optimise the main wing's planform with the quick analysis
        new, hist, info = generic_planform(p, design, prof)
        res.update(info)
    res["scorecard_after"] = scorecard(new)
    if len(hist):
        pl = plt()
        fig, ax = pl.subplots(figsize=(5.2, 3.0))
        ax.plot(hist[:, 0], hist[:, 1], color=NAVY)
        ax.set(xlabel="evaluations", ylabel="best objective", title="Design optimisation convergence")
        res["figure"] = save_fig(fig, out / "convergence.png")
    p.save_design("design", new, "design optimisation")
    return res


def generic_planform(p: Project, design: dict[str, Any], prof: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    from aerosynapse.shapeopt.cmaes import cmaes

    f0 = main_wing(design)
    idx = next(i for i, f in enumerate(design["features"]) if f is f0)
    base = {k: float(f0.get(k, d)) for k, d in (("span", 4.0), ("root_chord", 4.0), ("taper", 0.3), ("sweep", 40.0))}
    x0 = float(f0.get("translate", [0, 0, 0])[0])
    L = kernel.info(design, h=prof["step_h"])["length"]
    S0 = 0.5 * (1 + base["taper"]) * base["root_chord"] * base["span"]

    def make(z: np.ndarray) -> dict[str, Any]:
        d = copy.deepcopy(design)
        f = d["features"][idx]
        f["span"] = base["span"] * (0.85 + 0.35 * z[0])
        f["root_chord"] = base["root_chord"] * (0.85 + 0.35 * z[1])
        f["taper"] = 0.1 + 0.5 * z[2]
        f["sweep"] = base["sweep"] - 12 + 24 * z[3]
        t = list(f.get("translate", [0, 0, 0]))
        t[0] = x0 + L * (-0.08 + 0.16 * z[4])
        f["translate"] = t
        return d

    def terms(d: dict[str, Any]) -> dict[str, float]:
        s = scorecard(d, h=prof["body_h"])
        a = A.analyse(d, h=prof["body_h"])
        K = a["lattice"]["M0.85"]["induced_K"]
        return {"cruise": s["friction_CD0"] + K * 0.3**2, "turn": K * 0.8**2, "wave": s["wave_CD_M1.6"] or 0.0, "sm": s["static_margin_centroid"] or 0.0}

    t0 = terms(design)

    def J(z: np.ndarray) -> float:
        try:
            d = make(z)
            f = d["features"][idx]
            S = 0.5 * (1 + f["taper"]) * f["root_chord"] * f["span"]
            t = terms(d)
            pen = 20 * max(0.0, 0.98 - S / S0) + 5 * max(0.0, -0.05 - t["sm"]) + 5 * max(0.0, t["sm"] - 0.2)
            return 0.4 * t["cruise"] / t0["cruise"] + 0.3 * t["turn"] / t0["turn"] + 0.3 * t["wave"] / max(t0["wave"], 1e-6) + pen
        except Exception:
            return 50.0

    p.log(f"design: input body has no design-space variables; optimising the main wing planform ({prof['generic_budget']} evaluations)")
    zs = np.array([(1 / 0.35) * 0.15, (1 / 0.35) * 0.15, (base["taper"] - 0.1) / 0.5, 0.5, 0.5]).clip(0, 1)
    r = cmaes(lambda Z: batch(J, Z, prof["threads"]), zs, sigma=0.2, seed=0, budget=prof["generic_budget"], pop=8, restarts=0, stall_gens=20)
    new = make(r.x)
    info = {"method": "main wing planform (span, chord, taper, sweep, position), CMA-ES on the quick analysis (lattice, friction, area rule)",
            "J_before": J(zs), "J_after": float(r.f), "evaluations": r.evals, "terms_before": t0, "terms_after": terms(new),
            "wing_before": {k: f0.get(k) for k in ("span", "root_chord", "taper", "sweep", "translate")},
            "wing_after": {k: new["features"][idx].get(k) for k in ("span", "root_chord", "taper", "sweep", "translate")}}
    p.log(f"design: J {info['J_before']:.3f} -> {info['J_after']:.3f}")
    return new, np.array(r.history, float), info


# ------------------------------------------------------------------ 3 wing optimisation
# (state, Mach, wing CL, Reynolds number, weight). High-lift states (take-off, landing, instantaneous turn) are left to
# flaps and vortex lift: a thin fighter section is outside the 2D model's validity there (NeuralFoil confidence and
# Lock's wave-drag law both fail above about c_l 0.8 at these Mach numbers).
WING_STATES = [("cruise", 0.85, 0.20, 3.0e7, 0.35), ("sustained turn", 0.60, 0.45, 3.5e7, 0.35), ("loiter", 0.50, 0.30, 2.5e7, 0.30)]
SUPERSONIC_WEIGHT = 0.40
CONF_MIN = 0.80


def wing_opt(p: Project, prof: dict[str, Any]) -> dict[str, Any]:
    from aerosynapse.shapeopt import aero2d as A2
    from aerosynapse.shapeopt import geometry as G
    from aerosynapse.shapeopt.cmaes import cmaes
    from aerosynapse.structures.wing import Airfoil

    design, _ = p.design_before("wing")
    design = copy.deepcopy(design)
    out = p.stage_dir("wing")
    f = main_wing(design)
    af = Airfoil.of(f)
    u = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 121)))
    yu, yl = af.camber_line(u) + af.half(u), af.camber_line(u) - af.half(u)
    xy0 = np.c_[np.r_[u[::-1], u[1:]], np.r_[yu[::-1], yl[1:]]]
    W0, fit_err = G.fit_cst(xy0)
    lim = G.SectionLimits.from_baseline(W0)
    sweep = np.radians(float(f.get("sweep", 35.0)))
    cs = np.cos(sweep)
    # simple sweep theory: the section works at the normal Mach number and the normal lift coefficient
    cond = A2.Conditions(Re=np.array([s[3] for s in WING_STATES]), mach=np.array([s[1] * cs for s in WING_STATES]),
                         cl=np.minimum(np.array([s[2] / cs**2 for s in WING_STATES]), 1.0), ncrit=np.full(len(WING_STATES), 9.0), xtr=np.ones(len(WING_STATES)))
    wts = np.array([s[4] for s in WING_STATES])
    xs = np.linspace(0.002, 0.998, 400)

    def wave(W: np.ndarray) -> np.ndarray:
        su, sl = G.surfaces(np.atleast_2d(W), xs)
        return np.trapezoid(np.gradient(su, xs, axis=1) ** 2 + np.gradient(sl, xs, axis=1) ** 2, xs, axis=1)

    r0 = A2.solve_at_cl(W0[None], cond)
    cd0, cm0, w0 = r0["cd"][0], r0["cm"][0], wave(W0)[0]
    span = np.r_[np.full(16, 0.06), 0.06, 0.003]

    def decode(Z: np.ndarray) -> np.ndarray:
        W = W0[None] + (np.atleast_2d(Z) - 0.5) * 2 * span
        W[:, 17] = np.clip(W[:, 17], 0.0, 0.004)
        return W

    def J(Z: np.ndarray) -> np.ndarray:
        W = decode(Z)
        r = A2.solve_at_cl(W, cond)
        sub = (r["cd"] / cd0[None]) @ wts
        sup = wave(W) / w0
        g = G.section_violations(W, lim)
        pen = 10 * np.maximum(g, 0).sum(1) + 5 * np.maximum(CONF_MIN - r["conf"], 0).sum(1) + 20 * r["cl_short"].sum(1) + 5 * np.maximum(0, cm0[None] - r["cm"] - 0.02).sum(1)
        return sub + SUPERSONIC_WEIGHT * sup + pen

    p.log(f"wing: airfoil of '{f.get('id')}' fitted with CST (error {fit_err:.1e}); CMA-ES, {prof['wing_budget']} section evaluations over {len(WING_STATES)} states + supersonic wave drag")
    t0 = time.time()
    r = cmaes(J, np.full(18, 0.5), sigma=0.12, seed=0, budget=prof["wing_budget"], pop=16, restarts=0, stall_gens=40)
    W1 = decode(r.x)[0]
    r1 = A2.solve_at_cl(W1[None], cond)
    J0 = float(J(np.full((1, 18), 0.5))[0])
    drag0 = float(wts.sum() + SUPERSONIC_WEIGHT)
    drag1 = float((r1["cd"][0] / cd0) @ wts + SUPERSONIC_WEIGHT * wave(W1)[0] / w0)
    if drag0 + 1e-6 < J0:
        p.log(f"wing: note: the starting section is penalised ({J0 - drag0:.3f}): it misses a lift, confidence, moment or geometry limit")
    states = []
    for k, s in enumerate(WING_STATES):
        states.append({"state": s[0], "mach": s[1], "mach_normal": float(cond.mach[k]), "cl_normal": float(cond.cl[k]), "Re": s[3],
                       "cd_before": float(cd0[k]), "cd_after": float(r1["cd"][0, k]), "cm_before": float(cm0[k]), "cm_after": float(r1["cm"][0, k]),
                       "confidence_after": float(r1["conf"][0, k])})
    m0, m1 = G.section_metrics(W0[None]), G.section_metrics(W1[None])
    res: dict[str, Any] = {"method": "CST section (18 numbers), NeuralFoil with Prandtl-Glauert, critical Mach and Lock wave drag at the normal Mach number; "
                                     "linear supersonic (Ackeret) wave drag; thickness, area, nose radius, rear-spar depth and waviness kept at least the baseline's",
                           "feature": f.get("id"), "sweep_deg": float(np.degrees(sweep)), "cst_fit_error": fit_err, "J_before": J0, "J_after": float(r.f), "drag_index_before": drag0, "drag_index_after": drag1, "evaluations": r.evals,
                           "seconds": time.time() - t0, "states": states, "supersonic_wave_ratio": float(wave(W1)[0] / w0),
                           "thickness_before": float(m0["t"][0]), "thickness_after": float(m1["t"][0]), "area_before": float(m0["area"][0]), "area_after": float(m1["area"][0]),
                           "history": r.history}
    xy1 = G.coords(W1, n=121)
    G.write_dat(out / "baseline_section.dat", G.coords(W0, n=121), f"{p.name} baseline")
    G.write_dat(out / "optimised_section.dat", xy1, f"{p.name} optimised")
    # independent check with XFOIL at the cruise and loiter states (a thin section converges in XFOIL only up to
    # about 3 degrees, so the turn state's lift is outside its converged range; Mach is capped at XFOIL's limit here)
    try:
        from aerosynapse.shapeopt import xfoil as XF

        chk_all = {}
        for k in (0, 2):
            M = min(float(cond.mach[k]), XF.MACH_LIMIT)
            chk = {}
            for tag, W in (("before", W0), ("after", W1)):
                pol = XF.polar(G.coords(W, n=161), float(cond.Re[k]), M, a_lo=-2, a_hi=8, step=0.5)
                chk[tag] = XF.at_cl(pol, float(cond.cl[k]))
            chk_all[WING_STATES[k][0]] = {"mach": M, "cl": float(cond.cl[k]), **chk}
            p.log(f"wing: XFOIL at {WING_STATES[k][0]} (M {M:.2f}, cl {cond.cl[k]:.2f}): cd {chk['before']['cd']:.5f} -> {chk['after']['cd']:.5f}")
        res["xfoil"] = chk_all
    except Exception as e:  # XFOIL missing or not converged: reported, never filled in
        res["xfoil"] = {"error": str(e)[:300]}
    # back into the 3D wing: a section table, kept at the feature's own t/c distribution
    uu = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 61)))
    su, sl = G.surfaces(W1[None], uu)
    t_tab, z_tab = (su - sl)[0], 0.5 * (su + sl)[0]
    tc_old = float(f.get("tc", m0["t"][0]))
    f["profile"] = "table"
    f["section"] = {"u": [round(float(v), 6) for v in uu], "t": [round(float(v), 6) for v in t_tab], "zc": [round(float(v), 6) for v in z_tab]}
    if float(f.get("tip_tc", -1)) > 0:
        f["tip_tc"] = round(float(f["tip_tc"]) * float(t_tab.max()) / tc_old, 5)
    f["tc"] = round(float(t_tab.max()), 5)
    f["section_source"] = "workflow wing stage (CST + NeuralFoil + Ackeret, XFOIL check)"
    pl = plt()
    fig, ax = pl.subplots(1, 2, figsize=(10, 3.0))
    x0c = G.coords(W0, n=121)
    ax[0].plot(x0c[:, 0], x0c[:, 1], color=STEEL, label="before")
    ax[0].plot(xy1[:, 0], xy1[:, 1], color=CYAN, lw=1.8, label="optimised")
    ax[0].set(aspect="equal", title="Wing section", xlabel="x/c")
    ax[0].legend()
    lab = [s["state"] for s in states]
    xx = np.arange(len(lab))
    ax[1].bar(xx - 0.18, [1e4 * s["cd_before"] for s in states], 0.36, color=STEEL, label="before")
    ax[1].bar(xx + 0.18, [1e4 * s["cd_after"] for s in states], 0.36, color=CYAN, label="optimised")
    ax[1].set_xticks(xx, lab)
    ax[1].set(ylabel="section drag, counts", title="Drag at each state (normal Mach and lift)")
    ax[1].legend()
    res["figure"] = save_fig(fig, out / "section.png")
    res["scorecard_after"] = scorecard(design)
    p.log(f"wing: objective {J0:.3f} -> {r.f:.3f} (drag index {drag0:.3f} -> {drag1:.3f}); supersonic wave drag x{res['supersonic_wave_ratio']:.3f}")
    p.save_design("wing", design, "optimised airfoil in the main wing")
    return res


# ------------------------------------------------------------------ 4 body optimisation
def area_distribution(design: dict[str, Any], n_x: int = 48, n_yz: int = 70) -> tuple[np.ndarray, np.ndarray]:
    info = kernel.info(design, h=0.1, nx=60)
    lo, hi = np.array(info["bbox_lo"]), np.array(info["bbox_hi"])
    xs = np.linspace(lo[0] + 1e-3, hi[0] - 1e-3, n_x)
    yy = np.linspace(lo[1] - 0.05, hi[1] + 0.05, n_yz)
    zz = np.linspace(lo[2] - 0.05, hi[2] + 0.05, n_yz)
    Y, Z = np.meshgrid(yy, zz, indexing="ij")
    pts = np.vstack([np.c_[np.full(Y.size, x), Y.ravel(), Z.ravel()] for x in xs])
    d = kernel.distance(design, pts).reshape(n_x, -1)
    dA = (yy[1] - yy[0]) * (zz[1] - zz[0])
    return xs, (d < 0).sum(1) * dA


def body_opt(p: Project, prof: dict[str, Any]) -> dict[str, Any]:
    from aerosynapse.fighter import design as D
    from aerosynapse.shapeopt.cmaes import cmaes

    design, _ = p.design_before("body")
    out = p.stage_dir("body")
    bi = next((i for i, f in enumerate(design["features"]) if f.get("type") == "fuselage"), None)
    if bi is None:
        p.log("body: no fuselage feature; nothing to optimise")
        p.save_design("body", design, "no fuselage")
        return {"note": "the design has no fuselage feature"}
    st0 = np.array(design["features"][bi]["stations"], float)
    inner = np.arange(1, len(st0) - 1)
    k = len(inner)
    env = None
    if "variables" in design:
        e = D.envelope_points(D.complete(p.meta["requirements"]["family"], design["variables"]))
        env = np.vstack([e["engine"], e["cockpit"]])
    h = prof["body_h"]

    def make(z: np.ndarray) -> dict[str, Any]:
        d = copy.deepcopy(design)
        st = st0.copy()
        st[inner, 1] *= 0.8 + 0.4 * z[:k]
        st[inner, 2] *= 0.8 + 0.4 * z[k : 2 * k]
        d["features"][bi]["stations"] = [[round(float(v), 5) for v in row] for row in st]
        return d

    def measure(d: dict[str, Any]) -> dict[str, float]:
        info = kernel.info(d, h=h, nx=60)
        w12 = A._mach_plane(d, info, 1.2, n_x=24, n_roll=3)
        w16 = A._mach_plane(d, info, 1.6, n_x=24, n_roll=3)
        viol = 0.0
        if env is not None:
            viol = float(np.maximum(kernel.distance(D.outer(d), env) + 0.01, 0).max())
        return {"volume": info["volume"], "wetted": info["wetted_area"], "w12": w12, "w16": w16, "envelope_violation_m": viol}

    m0 = measure(design)

    def J(z: np.ndarray) -> float:
        try:
            m = measure(make(z))
            pen = 20 * max(0.0, 1 - m["volume"] / (0.99 * m0["volume"])) + 50 * m["envelope_violation_m"]
            return 0.35 * m["w12"] / max(m0["w12"], 1e-6) + 0.35 * m["w16"] / max(m0["w16"], 1e-6) + 0.30 * m["wetted"] / m0["wetted"] + pen
        except Exception:
            return 50.0

    p.log(f"body: {2 * k} cross-section variables, area rule at Mach 1.2 and 1.6, volume and envelopes held; {prof['body_budget']} evaluations")
    z0 = np.full(2 * k, 0.5)
    r = cmaes(lambda Z: batch(J, Z, prof["threads"]), z0, sigma=0.18, seed=0, budget=prof["body_budget"], pop=8, restarts=0, stall_gens=15)
    new = make(r.x) if r.f < J(z0) else copy.deepcopy(design)
    m1 = measure(new)
    xs0, a0 = area_distribution(design)
    xs1, a1 = area_distribution(new)
    L = float(xs1[-1] - xs1[0])
    V = float(np.trapezoid(a1, xs1))
    xi = (xs1 - xs1[0]) / L
    sh = 16 * V / (3 * np.pi * L) * np.clip(4 * xi * (1 - xi), 0, None) ** 1.5
    pl = plt()
    fig, ax = pl.subplots(figsize=(6.4, 3.0))
    ax.plot(xs0, a0, color=STEEL, label="before")
    ax.plot(xs1, a1, color=CYAN, lw=1.8, label="optimised")
    ax.plot(xs1, sh, "--", color=ORANGE, lw=1, label="Sears-Haack, same volume and length")
    ax.set(xlabel="x, m", ylabel="cross-section area, m$^2$", title="Area distribution (normal sections)")
    ax.legend()
    res = {"method": "half-width and half-height of each interior fuselage station (x0.8 to x1.2), CMA-ES; objective: Mach-plane wave drag at M1.2 and M1.6 "
                     "and wetted area; volume >= 99 % of the start, engine and cockpit envelopes inside the body",
           "J_before": float(J(z0)), "J_after": float(min(r.f, J(z0))), "evaluations": r.evals, "before": m0, "after": m1,
           "stations_before": st0.tolist(), "stations_after": new["features"][bi]["stations"], "figure": save_fig(fig, out / "area_rule.png"),
           "history": r.history, "scorecard_after": scorecard(new)}
    p.log(f"body: wave D/q at M1.2 {m0['w12']:.3f} -> {m1['w12']:.3f} m2, at M1.6 {m0['w16']:.3f} -> {m1['w16']:.3f} m2, volume {m0['volume']:.2f} -> {m1['volume']:.2f} m3")
    p.save_design("body", new, "area-ruled fuselage")
    return res


def structure_figure(res: dict[str, Any], out: Path) -> str:
    wg = res["wing"]
    s = wg["stations"]
    pl = plt()
    fig, ax = pl.subplots(2, 2, figsize=(10, 6))
    vn = res["vn"]
    ax[0, 0].plot(vn["V_eas"], vn["n_manoeuvre_pos"], color=NAVY)
    ax[0, 0].plot(vn["V_eas"], vn["n_manoeuvre_neg"], color=NAVY)
    for _kk, g in vn["gust_points"].items():
        ax[0, 0].plot([0, g["V"]], [1, g["n_up"]], ":", color=ORANGE)
        ax[0, 0].plot([0, g["V"]], [1, g["n_down"]], ":", color=ORANGE)
    ax[0, 0].axvline(vn["V_A"], color=STEEL, lw=0.8)
    ax[0, 0].set(xlabel="equivalent airspeed, m/s", ylabel="load factor n", title=f"V-n diagram ({vn['code']['name']})")
    ax[0, 1].plot(s["eta"], s["moment_kNm"], color=NAVY, label="bending moment, kNm")
    ax[0, 1].plot(s["eta"], s["shear_kN"], color=CYAN, label="shear, kN")
    ax[0, 1].plot(s["eta"], np.array(s["torque_kNm"]) * 5, color=ORANGE, label="torque x5, kNm")
    ax[0, 1].set(xlabel="spanwise station", title="Limit loads per side")
    ax[0, 1].legend()
    ax[1, 0].plot(s["eta"], s["cover_mm"], color=NAVY, label="cover")
    ax[1, 0].plot(s["eta"], s["web_mm"], color=CYAN, label="spar web")
    ax[1, 0].set(xlabel="spanwise station", ylabel="thickness, mm", title=f"Sized gauges ({wg['material']})")
    ax[1, 0].legend()
    fus = res["fuselage"]
    if "x" in fus:
        ax[1, 1].plot(fus["x"], fus["moment_kNm"], color=NAVY)
        ax[1, 1].set(xlabel="x, m", ylabel="kNm", title=f"Fuselage bending at n = {fus['n_ultimate']:.1f}")
    return save_fig(fig, out / "structure.png")


# ------------------------------------------------------------------ 5 structure
def structure(p: Project, prof: dict[str, Any]) -> dict[str, Any]:
    from aerosynapse.aircraft import materials as mt
    from aerosynapse.structures import wing as SW
    from aerosynapse.thermal import skin_design as SD

    design, _ = p.design_before("structure")
    design = copy.deepcopy(design)
    out = p.stage_dir("structure")
    req = p.meta["requirements"]
    fam = family_of(p, design)
    if fam:
        r = fighter_perf(fam, design["variables"])
        W0, fuel = r.W0, r.fuel
    else:
        W0, fuel = float(req["W0_kg"]), 0.3 * float(req["W0_kg"])
    zT = SD.zone_temperatures()
    screen = {c.zone: c for c in mt.select(zT)}["wing_box"].material  # Ashby screening
    T_box = zT["wing_box"]
    code = SW.C.CODES[req["code"]]
    p.log(f"structure: {req['code']}, W0 {W0:.0f} kg, wing at {T_box:.0f} K; sizing the box in every qualified material")
    trade = []
    for key in ("al7075", "al2024", "al_li", "ti64", "ti6242", "cfrp_im7", "cfrp_bmi"):
        m = mt.LIBRARY[key]
        if m.T_max - 20 < T_box:
            trade.append({"material": m.name, "key": key, "note": f"service limit {m.T_max:.0f} K is below the {T_box:.0f} K wing temperature + 20 K"})
            continue
        w = SW.size_wing(design, W0, code, key, T_box, v_d_eas=float(req["v_d_eas_ms"]), fuel_kg=fuel)
        trade.append({"material": m.name, "key": key, "box_mass_kg": w["box_mass_both_sides_kg"], "strength_sized_kg": w["strength_sized_box_mass_kg"],
                      "stiffness_scale": w["stiffness_scale"], "tip_deflection_pct": w["tip_deflection_pct_semispan"], "flutter_margin": w["margins_min"]["flutter"]})
        p.log(f"  {m.name}: box {w['box_mass_both_sides_kg']:.0f} kg, tip deflection {w['tip_deflection_pct_semispan']:.1f} %")
    sized = [t for t in trade if "box_mass_kg" in t]
    mat = min(sized, key=lambda t: t["box_mass_kg"])["key"] if sized else screen
    p.log(f"structure: lightest box in {mt.LIBRARY[mat].name} (Ashby screening suggested {mt.LIBRARY[screen].name})")
    res = SW.analyse(design, W0, req["code"], mat, T_box, v_d_eas=float(req["v_d_eas_ms"]), fuel_kg=fuel)
    res["screening_choice"] = mt.LIBRARY[screen].name
    res["material_trade"] = trade
    res["W0_kg"], res["fuel_kg"] = W0, fuel
    wg = res["wing"]
    s = wg["stations"]
    fus = res["fuselage"]
    res["figure"] = structure_figure(res, out)
    design["structure"] = {"code": res["code"], "wing_box_material": wg["material"], "cover_root_mm": s["cover_mm"][0], "web_root_mm": s["web_mm"][0],
                           "box_mass_kg": wg["box_mass_both_sides_kg"], "fuselage_skin_mm": fus.get("skin_required_mm"), "margins": wg["margins_min"]}
    p.log(f"structure: box {wg['box_mass_both_sides_kg']:.0f} kg (strength-sized {wg['strength_sized_box_mass_kg']:.0f}), flutter V {wg['flutter']['V_flutter_eas_ms']:.0f} m/s EAS, "
          f"tip deflection {wg['tip_deflection_pct_semispan']:.1f} %")
    p.save_design("structure", design, "structural sizing recorded")
    return res


# ------------------------------------------------------------------ 6 thermal
def thermal(p: Project, prof: dict[str, Any]) -> dict[str, Any]:
    from aerosynapse.thermal import bridge as B
    from aerosynapse.thermal import skin_design as SD

    design, _ = p.design_before("thermal")
    design = copy.deepcopy(design)
    out = p.stage_dir("thermal")
    p.log(f"thermal: PINN convergence study, Adam budgets {prof['pinn_levels']}")
    conv = SD.convergence(out / "pinn_convergence", prof["pinn_levels"], prof["pinn_lbfgs"], log=p.log)
    cands = [(conv["levels"][-1]["max_abs"], conv["best_checkpoint"])]
    shared = RESULTS / "thermal_skin" / "skin_pinn.pt"
    if shared.exists():
        n_, pr_ = B.load(shared)
        cands.append((SD.error_vs_exact(n_, pr_)["max_abs"], str(shared)))
    err, ck = min(cands)
    net, prob = B.load(ck)
    p.log(f"thermal: designing the skin with {Path(ck).parent.name}/skin_pinn.pt (max error {err:.2e})")
    stc = p.result("structure")
    cover = float(np.mean(stc["wing"]["stations"]["cover_mm"])) * 1e-3 if stc else 0.0
    fus = (stc.get("fuselage", {}).get("skin_required_mm") or 0.0) * 1e-3 if stc else 0.0
    t_min = {"wing_box": cover, "centre_fuselage": fus, "fwd_fuselage": 0.6 * fus, "aft_fuselage": 0.8 * fus, "control": 0.4 * cover}
    ref_mat = stc["wing"]["material_key"] if stc else "al7075"  # the structural gauges are in the material the structure stage chose
    opt = SD.optimise(net, prob, t_min, log=p.log, ref=ref_mat)
    design["skin"] = {z: ({"material": v["optimum"]["material"], "thickness_mm": round(v["optimum"]["thickness_mm"], 3)} if v["optimum"] else None) for z, v in opt["zones"].items()}
    pl = plt()
    fig, ax = pl.subplots(1, 2, figsize=(10, 3.2))
    lv = conv["levels"]
    x = [r["adam_epochs"] for r in lv]
    ax[0].loglog(x, [r["max_abs"] for r in lv], "o-", color=NAVY, label="max error")
    ax[0].loglog(x, [r["rel_l2"] for r in lv], "s--", color=CYAN, label="relative L2")
    ax[0].set(xlabel="Adam epochs", ylabel="error vs exact series", title="PINN convergence (skin slab, Bi 0.3-10)")
    ax[0].legend()
    zn = [z for z, v in opt["zones"].items() if v["optimum"]]
    ax[1].barh(zn, [opt["zones"][z]["optimum"]["areal_kg_m2"] for z in zn], color=CYAN, label="PINN-optimised")
    ax[1].barh(zn, [opt["zones"][z]["nominal"]["areal_kg_m2"] if opt["zones"][z]["nominal"] else 0 for z in zn], color="none", edgecolor=NAVY, label="nominal")
    ax[1].set(xlabel="areal mass, kg/m$^2$", title="Skin per zone")
    ax[1].legend()
    res = {"convergence": conv, "network_used": ck, "network_max_error": err, "structural_minimum_m": t_min, "design": opt, "figure": save_fig(fig, out / "thermal.png")}
    p.save_design("thermal", design, "skin materials and gauges from the PINN")
    return res


# ------------------------------------------------------------------ 7 simulation
def _coeffs(d: Path) -> dict[str, float]:
    from aerosynapse.cfd import runner

    try:
        h = runner.history(d)
        n = len(h["CL"])
        s = slice(int(0.75 * n), n)
        return {"CL": float(np.mean(h["CL"][s])), "CD": float(np.mean(h["CD"][s])), "Cm": float(np.mean(h["Cm"][s])) if "Cm" in h else float("nan"),
                "CL_unsteadiness": float(np.std(h["CL"][s]))}
    except Exception:
        r = json.loads((d / "result.json").read_text())
        return {k: float(r["last"][k]) for k in ("CL", "CD", "Cm")}


def richardson(f: list[float], h: list[float]) -> dict[str, Any]:
    """f, h ordered coarse to fine. Monotone convergence: observed order, Richardson extrapolation and the fine-grid
    GCI (Roache, safety factor 1.25; the formal order 2 is used when the observed order is outside 0.5 to 4).
    Oscillatory convergence: no extrapolation is valid, so the finest value is kept and the uncertainty is the
    oscillation range (half the spread of the three grids, times 1.25), stated as a fraction of the finest value."""
    f3, f2, f1 = f
    r = h[1] / h[2]
    e32, e21 = f3 - f2, f2 - f1
    monotone = e32 * e21 > 0 and abs(e21) > 0
    if not monotone:
        band = 1.25 * 0.5 * (max(f) - min(f))
        return {"observed_order": float("nan"), "order_used": float("nan"), "extrapolated": float(f1), "gci_fine": float(band / abs(f1)) if f1 else float("nan"),
                "monotone": False, "ratio": float(r), "kind": "oscillatory: finest value, uncertainty from the oscillation range"}
    p_obs = float(np.log(abs(e32 / e21)) / np.log(r))
    p_use = p_obs if 0.5 < p_obs < 4 else 2.0
    ext = f1 + (f1 - f2) / (r**p_use - 1)
    gci = 1.25 * abs((f1 - f2) / f1) / (r**p_use - 1) if f1 else float("nan")
    return {"observed_order": p_obs, "order_used": p_use, "extrapolated": float(ext), "gci_fine": float(gci), "monotone": True, "ratio": float(r),
            "kind": "monotone: Richardson extrapolation"}


def grid_figure(res: dict[str, Any], out: Path) -> str:
    gs = res["grid_study"]
    hs = [g["h"] for g in gs]
    pl = plt()
    fig, ax = pl.subplots(1, 2, figsize=(10, 3.0))
    for i, k in enumerate(("CL", "CD")):
        vals = [g[k] for g in gs]
        rx = res["richardson"][k]
        ax[i].plot(hs, vals, "o-", color=NAVY, label="3D flow solver")
        if rx["monotone"]:
            ax[i].axhline(rx["extrapolated"], ls="--", color=ORANGE, label=f"Richardson (p = {rx['order_used']:.2f})")
            ttl = f"{k}: GCI {100 * rx['gci_fine']:.1f} %"
        else:
            band = rx["gci_fine"] * abs(vals[-1])
            ax[i].axhspan(vals[-1] - band, vals[-1] + band, color=ORANGE, alpha=0.15, label="uncertainty (oscillation range)")
            ttl = f"{k}: oscillatory, +/-{100 * rx['gci_fine']:.0f} %"
        ax[i].set(xlabel="cell size h, m", ylabel=k, title=f"Grid convergence of {ttl}")
        ax[i].invert_xaxis()
        ax[i].legend()
    return save_fig(fig, out / "grid_convergence.png")


def refresh_simulation(p: Project) -> dict[str, Any]:
    """Recompute the grid-convergence summary and figure of a finished simulation stage from its saved runs."""
    res = p.result("simulation")
    gs = res["grid_study"]
    res["richardson"] = {k: richardson([g[k] for g in gs], [g["h"] for g in gs]) for k in ("CL", "CD")}
    res["figure"] = grid_figure(res, p.stage_dir("simulation"))
    p.write_result("simulation", res)
    return res


def simulation(p: Project, prof: dict[str, Any]) -> dict[str, Any]:
    from aerosynapse.cfd import runner

    design, _ = p.design_before("simulation")
    out = p.stage_dir("simulation")
    req = p.meta["requirements"]
    L = kernel.info(design, h=prof["step_h"])["length"]
    ref = reference(design, L)
    res: dict[str, Any] = {"reference": ref, "grid_study": [], "states": {}}
    cr = req["cruise"]
    fl = runner.Flight(mach=cr["mach"], alpha=cr["alpha"], altitude=cr["altitude"])
    for h in prof["cfd_h"]:
        d = out / f"cruise_h{h:g}"
        p.log(f"simulation: cruise M{cr['mach']} on grid h = {h} m ({prof['cfd_steps']} steps)")
        rr = runner.run(design, fl, ref, d, runner.Numerics(cells_per_length=L / h, steps=prof["cfd_steps"], max_cells=2.4e6, residual_drop=2e-4))
        c = _coeffs(d)
        res["grid_study"].append({"h": h, "cells": rr["cells"], "seconds": rr["seconds"], **c, "dir": str(d.relative_to(p.dir))})
        p.log(f"  cells {rr['cells']}, CL {c['CL']:.4f}, CD {c['CD']:.5f}, {rr['seconds']:.0f} s")
    gs = res["grid_study"]
    hs = [g["h"] for g in gs]
    res["richardson"] = {k: richardson([g[k] for g in gs], hs) for k in ("CL", "CD")}
    mid = prof["cfd_h"][1]
    for key in ("manoeuvre", "supersonic"):
        s = req[key]
        d = out / key
        p.log(f"simulation: {key} M{s['mach']} alpha {s['alpha']} on the medium grid")
        rr = runner.run(design, runner.Flight(mach=s["mach"], alpha=s["alpha"], altitude=s["altitude"]), ref, d,
                        runner.Numerics(cells_per_length=L / mid, steps=prof["cfd_steps"], max_cells=2.4e6, residual_drop=2e-4))
        res["states"][key] = {**s, **_coeffs(d), "cells": rr["cells"], "seconds": rr["seconds"], "separated_fraction": rr.get("separated_fraction"),
                              "T_wall_max_K": rr.get("T_wall_max_K"), "dir": str(d.relative_to(p.dir))}
    try:
        a = A.analyse(design)
        cla = a["lattice"]["M0.85"]["CL_alpha_per_rad"]
        res["fast_model"] = {"CL_cruise_lattice": float(cla * np.radians(cr["alpha"])), "friction_CD0": a["friction_CD0_estimate"], "wave_CD_M1.6": a["wave_drag_CD"]["M1.6"]}
    except Exception as e:
        res["fast_model"] = {"error": str(e)[:200]}
    res["studio_results_dir"] = gs[-1]["dir"]
    res["figure"] = grid_figure(res, out)
    try:
        pln = runner.plane(p.dir / gs[-1]["dir"], "plane_symmetry")
        var = "mach" if "mach" in pln else next(k for k in pln if k not in ("coord_u", "coord_v", "axis", "at", "u_axis", "v_axis"))
        fig, ax = plt().subplots(figsize=(8, 3))
        im = ax.pcolormesh(pln["coord_u"], pln["coord_v"], pln[var], cmap="viridis", shading="auto")
        fig.colorbar(im, ax=ax, label=var)
        ax.set(aspect="equal", title=f"Symmetry plane, cruise, finest grid ({var})")
        res["figure_field"] = save_fig(fig, out / "field.png")
    except Exception as e:
        res["figure_field_error"] = str(e)[:200]
    rc = res["richardson"]
    p.log(f"simulation: CD {rc['CD']['extrapolated']:.5f} +/- {100 * rc['CD']['gci_fine']:.1f} % ({rc['CD']['kind']}), CL {rc['CL']['extrapolated']:.4f} +/- {100 * rc['CL']['gci_fine']:.1f} %")
    p.save_design("simulation", design, "simulated")
    return res


# ------------------------------------------------------------------ 8 export
def export(p: Project, prof: dict[str, Any]) -> dict[str, Any]:
    from aerosynapse.workflow import export as X

    design, _ = p.design_before("export")
    return X.run(p, design, prof)


FUNCS: dict[str, Callable[[Project, dict[str, Any]], dict[str, Any]]] = {
    "concept": concept, "design": design_opt, "wing": wing_opt, "body": body_opt, "structure": structure, "thermal": thermal, "simulation": simulation, "export": export,
}


GEOMETRY_ONLY = {"simulation"}  # stages whose results depend on the shape alone


def geometry_hash(p: Project, key: str) -> str | None:
    """Checksum of the feature tree a stage last wrote (None if it has not written one)."""
    import hashlib

    f = p.dir / f"{KEYS.index(key) + 1:02d}_{key}" / "design.json"
    if not f.exists():
        return None
    feats = json.loads(f.read_text(encoding="utf-8")).get("features", [])
    return hashlib.sha256(json.dumps(feats, sort_keys=True).encode()).hexdigest()


def run_stage(p: Project, key: str, profile: str = "quick") -> dict[str, Any]:
    if key not in FUNCS:
        raise ValueError(f"unknown stage {key}; stages: {', '.join(KEYS)}")
    prof = PROFILES[profile]
    title = dict(STAGES)[key]
    p.log(f"=== stage {KEYS.index(key) + 1}: {title} ({profile})")
    old_geo = geometry_hash(p, key)
    p.set_stage(key, status="running", started=time.strftime("%Y-%m-%dT%H:%M:%S"), profile=profile)
    t0 = time.time()
    try:
        res = FUNCS[key](p, prof)
    except Exception as e:
        p.set_stage(key, status="failed", error=f"{type(e).__name__}: {e}", seconds=time.time() - t0)
        p.log(traceback.format_exc())
        raise
    res["seconds"] = time.time() - t0
    res["profile"] = profile
    p.write_result(key, res)
    # later stages depend on this one: mark them stale. The 3D flow depends only on the shape, so a stage that leaves
    # the geometry unchanged (structure, thermal, or a rerun with the same result) does not invalidate it.
    same_shape = old_geo is not None and old_geo == geometry_hash(p, key)
    for k in KEYS[KEYS.index(key) + 1 :]:
        if p.status(k) == "done" and not (k in GEOMETRY_ONLY and same_shape):
            p.set_stage(k, status="stale")
    p.set_stage(key, status="done", seconds=res["seconds"], finished=time.strftime("%Y-%m-%dT%H:%M:%S"), error=None)
    p.log(f"=== stage {key} done in {res['seconds']:.0f} s")
    return res


def run_all(p: Project, profile: str = "quick", start: str | None = None, stop: str | None = None) -> None:
    keys = KEYS[KEYS.index(start) if start else 0 : (KEYS.index(stop) + 1) if stop else None]
    for k in keys:
        if k != start and p.status(k) == "done":  # the start stage always reruns; later ones only if it invalidated them
            continue
        run_stage(p, k, profile)
