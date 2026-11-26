"""Stages 2 and 4: high-lift topology, then independent verification of the optimised design.

Everything here is outside the optimisation loop, so nothing was tuned to these checks:

* hold-out flow states the optimiser never saw, on a larger NeuralFoil network;
* the independent panel + boundary-layer solver, on NASA-validated ground;
* unsteady Navier-Stokes at low Re, including stall and vortex shedding;
* flutter, divergence, gust and dynamic stall.
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from _common import RESULTS  # noqa: E402
from _jobs import run_jobs  # noqa: E402

from aerosynapse.optim.robust import cvar  # noqa: E402
from aerosynapse.physics.airfoil import cosine_spacing  # noqa: E402
from aerosynapse.shapeopt import aero2d, dynamics, multielement, scenarios, vii  # noqa: E402
from aerosynapse.shapeopt import geometry as G  # noqa: E402
from aerosynapse.shapeopt import wing as wg  # noqa: E402
from aerosynapse.shapeopt.objective import G_NAMES, PLAN_NAMES, Problem  # noqa: E402
from aerosynapse.shapeopt.search import holdout_problem, pick_best  # noqa: E402


def robust_table(W0, Wb, plan0, planb, H):
    """Scenario-by-scenario comparison on hold-out states, largest network."""
    c = H.cond
    a = aero2d.solve_at_cl(W0[None], c, "xxxlarge")
    b = aero2d.solve_at_cl(Wb[None], c, "xxxlarge")
    named = []
    for i, (name, Re, M, cl, nc, xt, regime) in enumerate(scenarios.NAMED):
        named.append(
            {
                "name": name, "regime": regime, "Re": Re, "mach": M, "cl": cl, "ncrit": nc, "xtr": xt,
                "cd_base": float(a["cd"][0, i]), "cd_opt": float(b["cd"][0, i]),
                "alpha_base": float(a["alpha"][0, i]), "alpha_opt": float(b["alpha"][0, i]),
                "conf_base": float(a["conf"][0, i]), "conf_opt": float(b["conf"][0, i]),
                "short_base": float(a["cl_short"][0, i]), "short_opt": float(b["cl_short"][0, i]),
            }
        )  # fmt: skip
    ratio = b["cd"][0] / a["cd"][0]
    return named, {
        "n_states": len(c),
        "ratio": ratio.tolist(),
        "Re": c.Re.tolist(),
        "mach": c.mach.tolist(),
        "mean": float(ratio.mean()),
        "median": float(np.median(ratio)),
        "cvar90": float(cvar(ratio, 0.9)),
        "worst": float(ratio.max()),
        "best": float(ratio.min()),
        "frac_better": float((ratio < 1).mean()),
        "conf_min_opt": float(b["conf"].min()),
        "conf_min_base": float(a["conf"].min()),
    }


def wing_table(P: Problem, z0, zb):
    out = {}
    fs = P.fs
    for name, z in (("base", z0), ("opt", zb)):
        W, pl = P.decode(z)
        t = P._tables(W, pl)
        met = G.section_metrics(W)
        w = P.wing_eval(pl[0], {k: t[k][0] for k in t}, float(met["t"][0]))
        s = w.structure
        out[name] = {
            "drag": w.drag.tolist(), "CL": w.CL.tolist(), "CDi": w.CDi.tolist(), "CD": w.CD.tolist(),
            "cl_max_wing": w.cl_max_wing, "eta_stall": w.eta_stall, "v_stall": w.v_stall,
            "wing_mass": s.wing_mass, "takeoff_mass": s.takeoff_mass, "root_bending": s.root_bending,
            "n_gust": s.n_gust, "v_flutter": s.v_flutter, "v_divergence": s.v_divergence,
            "k_flutter": s.k_flutter, "extras": s.extras, "y": w.y.tolist(), "chord": w.chord.tolist(),
            "cl_section": w.cl_section[0].tolist(), "gamma": w.gamma[0].tolist(),
            "span": pl[0].span, "mac": pl[0].mac, "plan": dict(zip(PLAN_NAMES, pl[0].as_array().tolist(), strict=False)),
            "t_c": float(met["t"][0]),
        }  # fmt: skip
    out["names"] = fs.names
    out["v_dive"] = P.v.v_dive
    return out


def vii_compare(xy0, xyb):
    """Independent solver: polars and drag at matched lift."""
    a = np.arange(-4.0, 20.5, 1.0)
    conds = {
        "turbulent_Re6e6": (6e6, 9.0, 1.0),
        "transitional_Re6e5": (6e5, 9.0, 1.0),
        "laminar_Re2e5": (2e5, 11.0, 1.0),
        "high_Tu_Re2e6": (2e6, 1.5, 1.0),
        "contaminated_Re3e6": (3e6, 9.0, 0.03),
    }
    res = {}
    for k, (Re, nc, xt) in conds.items():
        p0 = vii.polar(*G.to_panel_order(xy0), a, Re, nc, xt)
        p1 = vii.polar(*G.to_panel_order(xyb), a, Re, nc, xt)
        row = {"alpha": a.tolist(), "base": {q: np.asarray(v, float).tolist() for q, v in p0.items()}, "opt": {q: np.asarray(v, float).tolist() for q, v in p1.items()}}
        cmp_ = []
        for cl in (0.3, 0.5, 0.8, 1.1):

            def at(p, cl=cl):
                k_ = int(np.argmax(p["CL"]))
                return float(np.interp(cl, p["CL"][: k_ + 1], p["CD"][: k_ + 1])) if p["CL"][: k_ + 1].max() >= cl and p["CL"][0] <= cl else float("nan")

            cmp_.append({"cl": cl, "cd_base": at(p0), "cd_opt": at(p1)})
        row["matched"] = cmp_
        row["clmax_base"], row["clmax_opt"] = float(p0["CL"].max()), float(p1["CL"].max())
        row["astall_base"], row["astall_opt"] = float(a[np.argmax(p0["CL"])]), float(a[np.argmax(p1["CL"])])
        res[k] = row
    return res


def dyn_stall(xy0, xyb):
    a = np.arange(-6.0, 26.5, 0.5)
    out = {}
    for name, xy in (("base", xy0), ("opt", xyb)):
        p = vii.polar(*G.to_panel_order(xy), a, 3e6, 9.0, 1.0)
        d = dynamics.dynamic_stall(a, p["CL"], 12.0, 8.0, k=0.1)
        out[name] = {"alpha": np.asarray(d["alpha"]).tolist()[::6], "cl": np.asarray(d["cl"]).tolist()[::6], "cl_peak": d["cl_peak"], "loop_area": d["loop_area"], "cl_drop": d["cl_drop"], "alpha_st": a.tolist(), "cl_st": p["CL"].tolist()}
    return out


def high_lift(W, cl_max):
    x = cosine_spacing(61)
    yu, yl = G.surfaces(W, x)
    d0 = multielement.calibrate(x, yu, yl, cl_max)
    out = {"dcp_crit": d0, "clean_cl_max": cl_max, "options": []}
    for name, sl, fl in (("slat", True, False), ("slotted flap", False, True), ("slat + slotted flap", True, True)):
        best = None
        for seed in range(3):
            lay, cl, al = multielement.optimise_layout(x, yu, yl, d0, sl, fl, seed=seed, budget=600)
            if best is None or cl > best[1]:
                best = (lay, cl, al)
        lay, cl, al = best
        els = multielement.build(x, yu, yl, lay)
        lay_d = {k: float(v) if not isinstance(v, bool) else v for k, v in lay.__dict__.items()}
        out["options"].append({"name": name, "cl_max": cl, "alpha": al, "gain": cl - cl_max, "layout": lay_d, "clearance": multielement.min_clearance(els), "elements": [e.tolist() for e in els]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="naca2412")
    ap.add_argument("--ns", default="standard", choices=["skip", "quick", "standard"])
    ap.add_argument("--profile", default="", help="verify search_<profile>.json instead of search.json (then only verify_<profile>.json is written)")
    n = ap.parse_args()
    out = RESULTS / "shapeopt" / n.shape.replace(":", "_")
    d = json.loads((out / (f"search_{n.profile}.json" if n.profile else "search.json")).read_text())
    promoted = not n.profile or json.loads((out / "search.json").read_text()).get("profile") == n.profile
    W0, plan0 = np.array(d["W0"]), wg.Planform(*d["plan0"])
    runs = d["runs"]
    best = pick_best(runs)
    t0 = time.time()
    H = holdout_problem(W0, plan0, 300, 60, model="xxxlarge")
    zb = np.array(best["z"])
    z0 = H.encode(W0, plan0)
    e0, eb = H.evaluate(z0), H.evaluate(zb)
    Wb, plb = H.decode(zb)
    Wb = Wb[0]
    viol = {k: float(q) for k, q in zip(G_NAMES, eb["g"][0], strict=False) if q > 1e-6}
    print("hold-out J", float(eb["J"][0]), "J0", float(eb["J0"][0]), "J2", float(eb["J2"][0]), "J3", float(eb["J3"][0]),
          "| baseline J", float(e0["J"][0]), "| feasible", not viol, "violated", viol, flush=True)
    xy0, xyb = G.coords(W0, 121), G.coords(Wb, 121)

    res = {
        "shape": n.shape,
        "profile": d.get("profile"),
        "best_seed": best["seed"],
        "feasible": not viol,
        "violated": viol,
        "baseline_J": float(e0["J"][0]),
        "seeds": [{"method": r["method"], "seed": r["seed"], "holdout_J": r["holdout_J"], "evals": r["evals"]} for r in runs],
        "holdout": {
            "J": float(eb["J"][0]), "J0": float(eb["J0"][0]), "J2": float(eb["J2"][0]), "J3": float(eb["J3"][0]),
            "constraints": dict(zip(G_NAMES, [float(q) for q in eb["g"][0]], strict=False)), "constraints_base": dict(zip(G_NAMES, [float(q) for q in e0["g"][0]], strict=False)),
        },
        "section": {"W0": W0.tolist(), "Wb": Wb.tolist(), "xy0": xy0.tolist(), "xyb": xyb.tolist(),
                    "m0": {k: float(v[0]) for k, v in G.section_metrics(W0[None]).items()},
                    "mb": {k: float(v[0]) for k, v in G.section_metrics(Wb[None]).items()}},
    }  # fmt: skip
    named, stats = robust_table(W0, Wb, plan0, plb[0], H)
    res["named"], res["robust"] = named, stats
    print("robust", {k: v for k, v in stats.items() if not isinstance(v, list)}, flush=True)
    res["wing"] = wing_table(H, z0, zb)
    print("wing done", flush=True)
    res["polars"] = {}
    for key, Re, nc, xt in (("Re2e5", 2e5, 11.0, 1.0), ("Re3e6", 3e6, 9.0, 1.0), ("Re3e6_dirty", 3e6, 9.0, 0.03)):
        res["polars"][key] = {"base": {k: np.asarray(v).tolist() for k, v in aero2d.polar(W0, Re, nc, xt).items()}, "opt": {k: np.asarray(v).tolist() for k, v in aero2d.polar(Wb, Re, nc, xt).items()}}
    res["vii"] = vii_compare(xy0, xyb)
    print("vii done", {k: (round(v["clmax_base"], 3), round(v["clmax_opt"], 3)) for k, v in res["vii"].items()}, flush=True)
    res["dynamic_stall"] = dyn_stall(xy0, xyb)
    # stall limit: the independent solver reads conservative near stall, NeuralFoil optimistic
    cl_max_b = float(res["vii"]["turbulent_Re6e6"]["clmax_opt"])
    res["high_lift"] = high_lift(Wb, min(cl_max_b, 1.6))
    print("high lift", [(o["name"], round(o["cl_max"], 3)) for o in res["high_lift"]["options"]], flush=True)
    (out / f"verify_{d.get('profile', 'unknown')}.json").write_text(json.dumps(res))
    if not promoted:
        print(f"verified search_{n.profile}.json; search.json, verify.json and the exports are left as they were")
        return
    (out / "verify.json").write_text(json.dumps(res))

    # exports: the optimised shape itself
    G.write_dat(out / "optimised_section.dat", xyb, f"AeroSynapse optimised from {n.shape}")
    G.write_dat(out / "baseline_section.dat", xy0, f"{n.shape} (CST fit)")
    v, f = wg.loft(plb[0], xyb[::3])
    G.write_stl(out / "optimised_wing.stl", v, f, "optimised_wing")
    (out / "optimised_design.json").write_text(
        json.dumps({"section_cst_weights": Wb.tolist(), "planform": dict(zip(PLAN_NAMES, plb[0].as_array().tolist(), strict=False)), "span_m": plb[0].span, "root_chord_m": plb[0].c_root}, indent=1)
    )

    if n.ns == "skip":
        return
    # unsteady Navier-Stokes: airfoils at low Re, and the cylinder as a solver check
    hq = 1 / 32 if n.ns == "standard" else 1 / 20
    te = 40.0 if n.ns == "standard" else 12.0
    jobs = []
    for a in (0.0, 8.0, 18.0):
        for nm, xy in (("base", xy0), ("opt", xyb)):
            jobs.append(dict(name=f"airfoil_{nm}_a{a:g}", polys=[xy], Re=500.0, alpha=a, h=hq, t_end=te, t_avg=te / 2.5))
    circ = G.superellipse(0.5, 0.5, 2.0)
    jobs.append(dict(name="cylinder_Re100", polys=[circ], Re=100.0, alpha=0.0, h=1 / 16, lx=24.0, ly=12.0, x_body=6.0, t_end=160.0 if n.ns == "standard" else 30.0, t_avg=60.0 if n.ns == "standard" else 10.0, pivot=(0.0, 0.0)))
    jobs.append(dict(name="cylinder_Re40", polys=[circ], Re=40.0, alpha=0.0, h=1 / 16, lx=24.0, ly=12.0, x_body=6.0, t_end=80.0 if n.ns == "standard" else 20.0, t_avg=15.0, pivot=(0.0, 0.0)))
    for tag, pth in (("square_shape", "square_shape"), ("square_grey", "square_grey")):
        tj = RESULTS / "shapeopt" / "topology" / f"{pth}.json"
        if tj.exists():
            t = json.loads(tj.read_text())
            polys = [np.array(q) for q in t["polygons"]]
            c = np.vstack(polys).mean(0)
            polys = [q - c for q in polys]
            for Re in (40.0, 100.0):
                jobs.append(dict(name=f"topo_{tag}_Re{Re:g}", polys=polys, Re=Re, alpha=0.0, h=1 / 24, lx=14.0, ly=7.0, x_body=4.0, t_end=70.0 if n.ns == "standard" else 15.0, t_avg=25.0 if n.ns == "standard" else 6.0, pivot=(0.0, 0.0)))
            if tag == "square_shape":
                base = np.array(t["base_polygon"])
                bq = base - base.mean(0)
                for Re in (40.0, 100.0):
                    jobs.append(dict(name=f"topo_baseline_Re{Re:g}", polys=[bq], Re=Re, alpha=0.0, h=1 / 24, lx=14.0, ly=7.0, x_body=4.0, t_end=70.0 if n.ns == "standard" else 15.0, t_avg=25.0 if n.ns == "standard" else 6.0, pivot=(0.0, 0.0)))
    print(f"{len(jobs)} Navier-Stokes cases", flush=True)
    for j in jobs:
        j["polys"] = [np.asarray(q).tolist() for q in j["polys"]]
    results = run_jobs([(j["name"], "ns", j) for j in jobs], out / "jobs" / f"ns_{n.ns}")
    for r in results:
        print(f"{r['name']}: Cd {r['cd_mean']:.4f} Cl {r['cl_mean']:.4f} Cl' {r['cl_rms']:.4f} St {r['strouhal']:.3f} ({r['wall_s']:.0f}s)", flush=True)
    (out / "ns.json").write_text(json.dumps(results), encoding="utf-8")
    print("total", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
