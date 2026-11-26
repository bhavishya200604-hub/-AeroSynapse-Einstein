"""Step 6.6: physical validation in 2D — Nusselt numbers vs Re / Pe / Ra against correlations verified from sources.

(A) Graetz channel (parallel plates, fully developed parabolic inflow, ISOTHERMAL walls theta_w = 1, inflow 0):
    verified FV solver at a ladder of (Re_H, Pr) -> local Nu_Dh(x*) and mean Nu_m(x*) (energy balance), x* = x/(4 Pe_H);
    references: the Graetz eigen-series without axial conduction (src/physics/graetz.py; reproduces the fully developed
    7.541 and the Leveque asymptote 1.2325 x*^-1/3 — both quoted from sources, LITERATURE.md), so the solver-series
    difference at low Pe is the AXIAL CONDUCTION the classical result neglects. Grid check at two pairs.
(B) The 2D PINN surrogate (Graetz ensemble of Step 4.7, Robin walls) vs the solver at the SAME conditions: local Nu
    along the wall, mean Nu; and the Phase 4 gate PINNs at isothermal walls (Re 20, Pr 1) against the series.
(C) Natural convection in a differentially heated square cavity: hot-wall Nu vs Ra (and Pr) on several grids with
    Richardson extrapolation, against Berkovsky-Polevikov (Nu = 0.18 [Pr / (0.2 + Pr) Ra]^0.29 for 1 < H/L < 2,
    verified from a source quoting the original; H/L = 1 is the edge of its range) and de Vahl Davis' benchmark values;
    Phase 4 gate PINNs at Ra 1e4.
Solver solutions are cached on disk (validation_study.cache_dir) and reused by later runs.

    python experiments/p6_6_validation.py [--config my.yaml] [--set key=value] [--tag name] [--parts graetz,pinn,natural]
                                          [--figures-only]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from aerosynapse.thermal.physics import graetz as G
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.utils.cache import cached
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz import validation_figs as F
from aerosynapse.thermal.viz.style import save_fig

MOD = "aerosynapse.thermal.experiments.p6_6_validation"
CODE_TAG = "p6_6-v1"  # part of every cache key: bump when the solver or the post-processing changes


def bp_nu(Ra, Pr) -> np.ndarray:
    """Berkovsky-Polevikov, 1 < H/L < 2 (square: (L/H)^-0.13 = 1): Nu = 0.18 [Pr / (0.2 + Pr) Ra]^0.29."""
    return 0.18 * (np.asarray(Pr, float) / (0.2 + np.asarray(Pr, float)) * np.asarray(Ra, float)) ** 0.29


def wall_nusselt(x, y, u, theta, th_w, dthdn, Pe):
    """Local Nu_Dh(x) (mean of both walls) = -2 dtheta/dn_in / (theta_w - theta_b), mean Nu_m(x*) by energy balance
    ln((theta_w - theta_in) / (theta_w - theta_b)) / (4 x*), x* = x / (4 Pe_H) (theta_in = 0, theta_w = 1 for Nu_m)."""
    u, theta = np.asarray(u), np.asarray(theta)
    tb = np.sum(u * theta, axis=0) / np.sum(u, axis=0)
    nus = [-2.0 * np.asarray(dthdn[w]) / (np.asarray(th_w[w]) - tb) for w in ("bottom", "top")]
    xs = np.asarray(x) / (4.0 * Pe)
    return {
        "xstar": xs,
        "Nu_local": 0.5 * (nus[0] + nus[1]),
        "theta_b": tb,
        "Nu_mean": np.log(1.0 / np.clip(1.0 - tb, 1e-300, None)) / (4.0 * xs),
    }


# --------------------------------------------------------------------------- (A) solver, isothermal walls
def graetz_task(pcfg: dict, Re: float, Pr: float, cells: int, cache_dir: str) -> dict:
    from aerosynapse.thermal.pinn import channel2d as CH

    params = {"Re": float(Re), "Pr": float(Pr), "Bi": 1.0, "Ri": 0.0}
    key = {"tag": CODE_TAG, "case": "graetz_cwt", "params": params, "cells": cells, "pcfg": pcfg}
    r, hit = cached(cache_dir, "graetz", key, lambda: CH.solve_channel(params, pcfg, cells, 1e-10, "quick"))
    w = {k: r[f"theta_wall_{k}"] for k in ("bottom", "top")}
    d = {k: r[f"dtheta_dn_{k}"] for k in ("bottom", "top")}
    nu = wall_nusselt(r["x"], r["y"], r["u"], r["theta"], w, d, Re * Pr)
    return (
        {"Re": Re, "Pr": Pr, "Pe": Re * Pr, "cells": cells, "cache_hit": hit, "wall_s": r["wall_s"]}
        | {k: np.asarray(v).tolist() for k, v in nu.items()}
        | ({"x": r["x"].tolist(), "y": r["y"].tolist(), "theta": r["theta"].tolist()})
    )


# --------------------------------------------------------------------------- (B) PINN surrogate vs solver, Robin walls
def pinn_task(sur: dict, point: dict, cells: int, cache_dir: str, exclude_outlet: float = 1.0) -> dict:
    import torch

    from aerosynapse.thermal.experiments.p6_4_sensitivity import Graetz
    from aerosynapse.thermal.pinn import channel2d as CH
    from aerosynapse.thermal.pinn import problems2d as P2

    ad = Graetz({"surrogate": sur, "solver": {"cells_per_unit": cells, "n": 2}, "pinn": {"n": 2, "n_wall": 2}})
    vals = {**ad.fixed, **point}
    key = {"tag": CODE_TAG, "case": "graetz_robin", "params": vals, "cells": cells, "pcfg": ad.pcfg}
    r, hit = cached(cache_dir, "graetz", key, lambda: CH.solve_channel(vals, ad.pcfg, cells, 1e-10, "quick"))
    Pe = vals["Re"] * vals["Pr"]
    sol = wall_nusselt(
        r["x"],
        r["y"],
        r["u"],
        r["theta"],
        {k: r[f"theta_wall_{k}"] for k in ("bottom", "top")},
        {k: r[f"dtheta_dn_{k}"] for k in ("bottom", "top")},
        Pe,
    )
    x, y = np.asarray(r["x"]), np.asarray(r["y"])
    X, Y = np.meshgrid(x, y)
    mem, ths, us, tws_, dns = [], [], [], [], []
    for _, prob, net in ad.members:
        p0 = next(net.parameters())
        pr = P2._batched(prob, net, prob.inputs(np.stack([X.ravel(), Y.ravel()], 1), vals, p0), 0, P2.FIELDS)
        th, u = pr["theta"].reshape(X.shape), pr["u"].reshape(X.shape)
        ths.append(th)
        tw, dn = {}, {}
        for wname, yw, sgn in (("bottom", 0.0, 1.0), ("top", 1.0, -1.0)):
            Xw = torch.as_tensor(
                prob.inputs(np.stack([x, np.full_like(x, yw)], 1), vals, p0).cpu().numpy(), dtype=p0.dtype
            )
            dd = prob.fields(net, Xw, order=1)
            tw[wname] = dd["theta"].detach().numpy()
            dn[wname] = sgn * dd["theta_y"].detach().numpy()  # inward normal: +y at the bottom, -y at the top
        mem.append(wall_nusselt(x, y, u, th, tw, dn, Pe))
        us.append(u)
        tws_.append(tw)
        dns.append(dn)
    # the ensemble prediction is the MEAN FIELD; its Nu (not the mean of member Nu, which a member whose own
    # theta_w - theta_b crosses zero would dominate)
    avg = lambda L: {w: np.mean([d[w] for d in L], axis=0) for w in ("bottom", "top")}  # noqa: E731
    ens = wall_nusselt(x, y, np.mean(us, axis=0), np.mean(ths, axis=0), avg(tws_), avg(dns), Pe)
    rel = np.abs(ens["Nu_local"] - sol["Nu_local"]) / np.abs(sol["Nu_local"])
    away = x >= 0.5  # the inlet singularity (Nu -> infinity at x = 0) excluded
    k = int(np.flatnonzero(x <= ad.Lx - exclude_outlet)[-1])  # evaluation station (as in the Graetz part)
    tws = 0.5 * (np.asarray(r["theta_wall_bottom"]) + np.asarray(r["theta_wall_top"]))
    th_e = np.mean(ths, axis=0)
    th_s = np.asarray(r["theta"])
    return {
        "point": point,
        "Pe": Pe,
        "cache_hit": hit,
        "xstar": sol["xstar"].tolist(),
        "solver": {k: np.asarray(sol[k]).tolist() for k in ("Nu_local", "Nu_mean", "theta_b")},
        "pinn": {k: np.asarray(v).tolist() for k, v in ens.items()},
        "members": [np.asarray(m["Nu_local"]).tolist() for m in mem],
        "rel_err_local_median": float(np.median(rel)),
        "rel_err_local_max": float(rel.max()),
        "rel_err_local_median_x05": float(np.median(rel[away])),
        "rel_err_local_max_x05": float(rel[away].max()),
        "rel_err_mean_end": float(abs(ens["Nu_mean"][-1] - sol["Nu_mean"][-1]) / abs(sol["Nu_mean"][-1])),
        "x_station": float(x[k]),
        "rel_err_mean_station": float(abs(ens["Nu_mean"][k] - sol["Nu_mean"][k]) / abs(sol["Nu_mean"][k])),
        # diagnosis: the bulk temperature itself, the driving difference theta_w - theta_b (small -> Nu ill-conditioned)
        "theta_b_abs_err_max": float(np.max(np.abs(ens["theta_b"] - sol["theta_b"]))),
        "theta_rel_l2": float(np.linalg.norm(th_e - th_s) / np.linalg.norm(th_s)),
        "theta_abs_err_max": float(np.max(np.abs(th_e - th_s))),
        "dT_wall_bulk_station": float(tws[k] - sol["theta_b"][k]),
    }


# --------------------------------------------------------------------------- (C) natural convection
def natural_task(Pr: float, n: int, Ras: list[float], bc_order: int, cache_dir: str) -> dict:
    def solve():
        from aerosynapse.thermal.experiments.p1_7_benchmarks import continuation
        from aerosynapse.thermal.numerics import fvm2d as fv
        from aerosynapse.thermal.numerics import ns_projection as ns
        from aerosynapse.thermal.numerics.ns_steady import SteadyNewtonNS

        en = ns.EnergySpec(
            Pe=np.sqrt(Ras[0] * Pr),
            bcs={
                "left": fv.dirichlet2(1.0),
                "right": fv.dirichlet2(0.0),
                "bottom": fv.neumann2(0.0),
                "top": fv.neumann2(0.0),
            },
            bc_order=bc_order,
        )
        s = ns.MACNavierStokes2D(fv.Grid2D(n, n), np.sqrt(Ras[0] / Pr), {}, energy=en, Ri=1.0)
        N = SteadyNewtonNS(s)
        x = N.pack(s.u, s.v, None, 1.0 - s.g.Xc)

        def setter(Ra):
            s.Re = float(np.sqrt(Ra / Pr))
            s.energy.D = 1.0 / float(np.sqrt(Ra * Pr))
            s.energy_spec.Pe = float(np.sqrt(Ra * Pr))

        out, log, cur = [], [], float(Ras[0]) * 0.999
        for Ra in Ras:
            t0 = time.perf_counter()
            x, _ = continuation(N, x, setter, [float(Ra)], cur, log)
            cur = float(Ra)
            N.x = x
            N.store()
            th = s.theta.ravel()
            q = s.energy.boundary_heat_flow(th, 0.0)
            out.append(
                {
                    "Ra": float(Ra),
                    "Nu_hot": -float(np.mean(s.energy.wall_gradient(th, 0.0, "left"))),
                    "Nu_cold": -float(np.mean(s.energy.wall_gradient(th, 0.0, "right"))),
                    "heat_balance_rel": float((q["left"] + q["right"]) / q["left"]),
                    "theta": s.theta.copy(),
                    "psi": s.streamfunction().copy(),
                    "xc": s.g.xc.copy(),
                    "yc": s.g.yc.copy(),
                    "wall_s": time.perf_counter() - t0,
                }
            )
        return out

    key = {"tag": CODE_TAG, "case": "natural", "Pr": Pr, "n": n, "Ras": [float(r) for r in Ras], "bc_order": bc_order}
    res, hit = cached(cache_dir, "natural", key, solve)
    return {
        "Pr": Pr,
        "N": n,
        "cache_hit": hit,
        "records": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in res],
    }


# --------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--parts", default="graetz,pinn,natural")
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    spec = SP.load_spec(a.config or None, a.profile, SP.parse_set(a.sets))
    vs = spec["validation_study"]
    custom = bool(a.config or a.sets or a.tag != "default" or a.profile != "smoke" or a.parts != "graetz,pinn,natural")
    odir = ROOT / "results" / f"p6_6_validation_{a.tag}"
    table = odir / "validation.md" if custom else ROOT / "docs" / "tables" / "phase6_6_validation.md"
    if a.figures_only:
        met = json.loads((odir / "metrics.json").read_text(encoding="utf-8"))
        met["figures"] = figures(met, a.tag, odir, docs=not custom)
        report(met, table, a.tag)
        print(f"wrote {table}")
        return
    parts = {p.strip() for p in a.parts.split(",")}
    cache_dir = str(ROOT / vs["cache_dir"])
    g = vs["graetz"]
    pcfg = SP.deep_merge(
        spec["problems2d"]["D_graetz"],
        {
            "geometry": {"Lx": float(g["Lx"])},
            "thermal": {
                "bottom": {"_replace": True, "type": "dirichlet", "value": 1.0},
                "top": {"_replace": True, "type": "dirichlet", "value": 1.0},
            },
        },
    )
    with Experiment(f"p6_6_validation_{a.tag}", spec) as ex:
        jobs, labels = [], []
        if "graetz" in parts:
            for Re, Pr in g["pairs"]:
                jobs.append(
                    (
                        MOD,
                        "graetz_task",
                        dict(
                            pcfg=pcfg, Re=float(Re), Pr=float(Pr), cells=int(g["cells_per_unit"]), cache_dir=cache_dir
                        ),
                    )
                )
                labels.append(("graetz", (Re, Pr), int(g["cells_per_unit"])))
            for Re, Pr in (g["pairs"][0], g["pairs"][-1]):
                jobs.append(
                    (
                        MOD,
                        "graetz_task",
                        dict(pcfg=pcfg, Re=float(Re), Pr=float(Pr), cells=int(g["check_cells"]), cache_dir=cache_dir),
                    )
                )
                labels.append(("graetz_check", (Re, Pr), int(g["check_cells"])))
        if "pinn" in parts:
            from aerosynapse.thermal.experiments.p6_4_extended import point_values
            from aerosynapse.thermal.experiments.p6_4_sensitivity import Graetz

            pc = vs["pinn"]
            box = Graetz(
                {"surrogate": pc["surrogate"], "solver": {"cells_per_unit": 8, "n": 2}, "pinn": {"n": 2, "n_wall": 2}}
            ).box
            for p in pc["points"]:
                jobs.append(
                    (
                        MOD,
                        "pinn_task",
                        dict(
                            sur=pc["surrogate"],
                            point=point_values(p, box),
                            cells=int(pc["cells_per_unit"]),
                            cache_dir=cache_dir,
                            exclude_outlet=float(g.get("exclude_outlet", 1.0)),
                        ),
                    )
                )
                labels.append(("pinn", str(p), None))
        if "natural" in parts:
            nat = vs["natural"]
            for Pr in nat["Pr"]:
                for n in nat["grids"]:
                    jobs.append(
                        (
                            MOD,
                            "natural_task",
                            dict(
                                Pr=float(Pr),
                                n=int(n),
                                Ras=[float(r) for r in nat["Ra"]],
                                bc_order=int(nat["bc_order"]),
                                cache_dir=cache_dir,
                            ),
                        )
                    )
                    labels.append(("natural", Pr, n))
        print(f"{len(jobs)} tasks", flush=True)
        outs = run_tasks(
            jobs,
            {"isolate": True, **vs.get("execution", {})},
            lambda i, r, sec: print(
                f"[{i + 1}/{len(jobs)}] {labels[i]}: {sec:.0f} s (cache hit: {r.get('cache_hit')})", flush=True
            ),
        )
        met = ex.metrics
        met["graetz"] = [o for (k, *_), o in zip(labels, outs, strict=True) if k == "graetz"]
        met["graetz_check"] = [o for (k, *_), o in zip(labels, outs, strict=True) if k == "graetz_check"]
        met["pinn"] = [o for (k, *_), o in zip(labels, outs, strict=True) if k == "pinn"]
        met["natural"] = [o for (k, *_), o in zip(labels, outs, strict=True) if k == "natural"]
        save_fields(met, ex.dir)  # fields -> fields.npz (git-ignored); metrics keep the numbers
        sr = vs["graetz"]["series"]
        xs = np.geomspace(*map(float, vs["graetz"]["xstar_plot"]), 200)
        ser = G.entry_series(xs, int(sr["n_modes"]), int(sr["N"]))
        met["series"] = {k: np.asarray(ser[k]).tolist() for k in ("xstar", "Nu_local", "Nu_mean")} | {
            "mu1": float(ser["mu"][0])
        }
        met["analysis"] = analyse(met, vs)
        met["plot_pair"] = vs["plots"]["graetz_3d_pair"]
        met["plot_Ra"] = [float(r) for r in vs["plots"]["cavity_3d_Ra"]]
        met["gate_pinns"] = gate_pinns()
        met["git"] = git_hash()
        met["figures"] = figures(met, a.tag, ex.dir, docs=not custom)
        report(met, table, a.tag)
        met["table"] = str(table.relative_to(ROOT))
    print(f"wrote {table}")


def save_fields(met: dict, odir: Path) -> None:
    """Move the solver fields out of the metrics into <odir>/fields.npz (Graetz pairs; the finest cavity grid)."""
    arrs = {}
    for r in met["graetz"]:
        k = f"graetz_{r['Re']:g}_{r['Pr']:g}"
        for f in ("x", "y", "theta"):
            arrs[f"{k}_{f}"] = np.asarray(r.pop(f))
    for r in met["graetz_check"]:
        for f in ("x", "y", "theta"):
            r.pop(f, None)
    finest = {}
    for r in met["natural"]:
        finest[r["Pr"]] = max(finest.get(r["Pr"], 0), r["N"])
    for r in met["natural"]:
        for rec in r["records"]:
            keep = r["N"] == finest[r["Pr"]]
            for f in ("theta", "psi", "xc", "yc"):
                v = rec.pop(f, None)
                if keep and v is not None:
                    arrs[f"nat_{r['Pr']:g}_{rec['Ra']:g}_{f}"] = np.asarray(v)
    np.savez_compressed(odir / "fields.npz", **arrs)


def analyse(met: dict, vs: dict) -> dict:
    out = {"graetz": [], "natural": {}}
    sr = vs["graetz"]["series"]
    Lx = float(vs["graetz"]["Lx"])
    xe = Lx - float(
        vs["graetz"].get("exclude_outlet", 1.0)
    )  # evaluation station: the zero-gradient outflow distorts the last cells
    for r in met["graetz"]:
        xs = np.asarray(r["xstar"])
        xx = xs * 4.0 * r["Pe"]  # back to x / H
        ok = (xs >= 1e-4) & (xx <= xe)
        k = int(np.flatnonzero(xx <= xe)[-1])
        ser = G.entry_series(xs[ok], int(sr["n_modes"]), int(sr["N"]))
        dl = (np.asarray(r["Nu_local"])[ok] - ser["Nu_local"]) / ser["Nu_local"]
        dm = (np.asarray(r["Nu_mean"])[ok] - ser["Nu_mean"]) / ser["Nu_mean"]
        s_k = G.entry_series(xs[k : k + 1], int(sr["n_modes"]), int(sr["N"]))
        out["graetz"].append(
            {
                "Re": r["Re"],
                "Pr": r["Pr"],
                "Pe": r["Pe"],
                "x_station": float(xx[k]),
                "xstar_station": float(xs[k]),
                "Nu_local_station": float(r["Nu_local"][k]),
                "Nu_mean_station": float(r["Nu_mean"][k]),
                "series_local_station": float(s_k["Nu_local"][0]),
                "series_mean_station": float(s_k["Nu_mean"][0]),
                "Nu_fd_axial": float(
                    G.nu_cwt_spectral(64, Pe=float(r["Pe"]))["Nu"]
                ),  # fully developed WITH axial conduction
                "dev_local_median": float(np.median(np.abs(dl))),
                "dev_mean_median": float(np.median(np.abs(dm))),
                "dev_mean_station": float(dm[-1]),
            }
        )
    for c in met.get("graetz_check", []):
        f = next(r for r in met["graetz"] if r["Re"] == c["Re"] and r["Pr"] == c["Pr"])
        a_ = float(np.interp(xe, np.asarray(f["xstar"]) * 4 * f["Pe"], f["Nu_mean"]))
        b_ = float(np.interp(xe, np.asarray(c["xstar"]) * 4 * c["Pe"], c["Nu_mean"]))
        out.setdefault("grid_check", []).append(
            {
                "Re": c["Re"],
                "Pr": c["Pr"],
                "x_station": xe,
                "Nu_mean_coarse": a_,
                "Nu_mean_fine": b_,
                "rel_diff": float(abs(a_ - b_) / abs(b_)),
            }
        )
    from aerosynapse.thermal.analysis.benchmarks import DE_VAHL_DAVIS_NU
    from aerosynapse.thermal.numerics.convergence import gci

    nat = {}
    for r in met["natural"]:
        for rec in r["records"]:
            nat.setdefault((r["Pr"], rec["Ra"]), {})[r["N"]] = rec["Nu_hot"]
    for (Pr, Ra), byN in sorted(nat.items()):
        Ns = sorted(byN)
        e = {
            "Pr": Pr,
            "Ra": Ra,
            "Nu": [byN[n] for n in Ns],
            "N": Ns,
            "BP": float(bp_nu(Ra, Pr)),
            "BP_validity": float(Pr / (0.2 + Pr) * Ra),
        }
        if len(Ns) >= 3:
            q = gci(byN[Ns[-1]], byN[Ns[-2]], byN[Ns[-3]], 1 / Ns[-1], 1 / Ns[-2], 1 / Ns[-3])
            e["richardson"], e["gci"], e["order"] = q.get("f_ext"), q.get("GCI_fine21"), q.get("p")
        best = (
            e.get("richardson") if e.get("richardson") is not None and np.isfinite(e.get("richardson")) else e["Nu"][-1]
        )
        e["best"] = float(best)
        e["dev_BP"] = float((best - e["BP"]) / e["BP"])
        if abs(Pr - 0.71) < 1e-9 and Ra in DE_VAHL_DAVIS_NU:
            e["dVD"] = DE_VAHL_DAVIS_NU[Ra]
            e["dev_dVD"] = float((best - e["dVD"]) / e["dVD"])
        out["natural"][f"Pr{Pr:g}_Ra{Ra:g}"] = e
    return out


def gate_pinns() -> dict:
    f = ROOT / "results" / "p4_gate_smoke" / "metrics.json"
    if not f.exists():
        return {}
    b = json.loads(f.read_text(encoding="utf-8"))["benchmarks"]
    out = {}
    if "dvd_ra1e4" in b:
        d = b["dvd_ra1e4"]
        out["dvd"] = {"Ra": d["cfg"]["Ra"], "Pr": d["cfg"]["Pr"], "Nu_hot": [r["Nu_hot"] for r in d["runs"]]}
    if "graetz_cwt" in b:
        d = b["graetz_cwt"]
        Pe = d["cfg"]["Re"] * d["cfg"]["Pr"]
        out["graetz"] = {
            "Pe": Pe,
            "Nu_axial": d["literature"]["Nu_axial"],
            "runs": [
                {"xstar": (np.asarray(r["x"]) / (4 * Pe)).tolist(), "Nu_x": r["Nu_x"], "Nu_mean": r["Nu_mean"]}
                for r in d["runs"]
            ],
        }
    return out


def _f(v, f=".4g") -> str:
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, f)


def report(met: dict, path: Path, tag: str) -> None:
    an = met["analysis"]
    L = [
        f"# Step 6.6 physical validation (2D) — `{tag}` (auto-generated by experiments/p6_6_validation.py)",
        "",
        "Correlations and constants verified from sources (LITERATURE.md): fully developed Nu_Dh = 7.541 (isothermal parallel "
        "plates), Leveque Nu_Dh = 0.4273 (f Re / x*)^(1/3), f Re = 24 (local; mean 3/2 of it), Berkovsky–Polevikov "
        "Nu = 0.18 [Pr/(0.2+Pr) Ra]^0.29 (1 < H/L < 2; the square is its edge). The Graetz eigen-series (no axial "
        "conduction) reproduces 7.541 and the Leveque asymptote; solver − series = axial conduction (low Pe). Solver data "
        "are cached on disk.",
        "",
    ]
    if an.get("graetz"):
        L += [
            f"## Graetz channel, isothermal walls (solver vs eigen-series), at the station x = {an['graetz'][0]['x_station']:g} H",
            "",
            "| Re_H | Pr | Pe_H | x* at the station | Nu_Dh local / mean | series local / mean (no axial conduction) | fully developed with axial conduction | median |dev| from the series, local / mean (x* ≥ 1e-4) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for e in an["graetz"]:
            L.append(
                f"| {e['Re']:g} | {e['Pr']:g} | {e['Pe']:g} | {e['xstar_station']:.3g} | {e['Nu_local_station']:.4g} / {e['Nu_mean_station']:.4g} | "
                f"{e['series_local_station']:.4g} / {e['series_mean_station']:.4g} | {e['Nu_fd_axial']:.4g} | "
                f"{100 * e['dev_local_median']:.2g} % / {100 * e['dev_mean_median']:.2g} % |"
            )
        for c in an.get("grid_check", []):
            L.append(
                f"\nGrid check (Re {c['Re']:g}, Pr {c['Pr']:g}): Nu_m at x = {c['x_station']:g} H {c['Nu_mean_coarse']:.5g} vs {c['Nu_mean_fine']:.5g} on the finer grid ({100 * c['rel_diff']:.2g} %)."
            )
        L.append(f"\nSeries: first eigenvalue (fully developed Nu) {met['series']['mu1']:.4f} (source value 7.541).")
        L.append("")
    if met.get("pinn"):
        L += [
            "## 2D PINN surrogate (Graetz ensemble, Robin walls) vs the solver at the same conditions",
            "",
            "| point (Re, Pr, Bi) | Pe | local Nu: median / max rel. error (all x) | same for x ≥ 0.5 H (inlet singularity excluded) | mean Nu rel. error at the station | theta: rel. L2 / max abs. error | bulk theta: max abs. error | theta_w − theta_b at the station (solver) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in met["pinn"]:
            p = r["point"]
            L.append(
                f"| ({p['Re']:.3g}, {p['Pr']:.3g}, {p['Bi']:.3g}) | {r['Pe']:.3g} | {100 * r['rel_err_local_median']:.2g} % / {100 * r['rel_err_local_max']:.2g} % | "
                f"{100 * r['rel_err_local_median_x05']:.2g} % / {100 * r['rel_err_local_max_x05']:.2g} % | {100 * r['rel_err_mean_station']:.2g} % | "
                f"{r['theta_rel_l2']:.3g} / {r['theta_abs_err_max']:.3g} | {r['theta_b_abs_err_max']:.3g} | {r['dT_wall_bulk_station']:.3g} |"
            )
        L.append("")
    gp = met.get("gate_pinns") or {}
    if gp.get("graetz"):
        L.append(
            f"Phase 4 gate PINNs, isothermal walls (Pe {gp['graetz']['Pe']:g}): mean Nu over the window "
            + ", ".join(f"{r['Nu_mean']:.4g}" for r in gp["graetz"]["runs"])
            + f" vs {gp['graetz']['Nu_axial']:.4g} (with axial conduction) / 7.541 (without)."
        )
        L.append("")
    if an.get("natural"):
        L += [
            "## Natural convection, square cavity (hot-wall Nu)",
            "",
            "| Pr | Ra | Nu per grid | Richardson (order, GCI) | Berkovsky–Polevikov (dev.) | de Vahl Davis (dev.) |",
            "|---|---|---|---|---|---|",
        ]
        for e in an["natural"].values():
            gtxt = f"{_f(e.get('richardson'))} (p {_f(e.get('order'), '.2f')}, {_f(100 * e['gci'] if e.get('gci') is not None else None, '.2g')} %)"
            dv = f"{e['dVD']:.4g} ({100 * e['dev_dVD']:+.2g} %)" if "dVD" in e else "—"
            L.append(
                f"| {e['Pr']:g} | {e['Ra']:.3g} | "
                + " / ".join(f"{v:.4g}" for v in e["Nu"])
                + f" | {gtxt} | {e['BP']:.4g} ({100 * e['dev_BP']:+.2g} %{'' if e['BP_validity'] > 1e3 else ', outside its range'}) | {dv} |"
            )
        if gp.get("dvd"):
            L.append(
                f"\nPhase 4 gate PINNs (Ra {gp['dvd']['Ra']:g}, Pr {gp['dvd']['Pr']:g}): Nu_hot "
                + ", ".join(f"{v:.4g}" for v in gp["dvd"]["Nu_hot"])
                + "."
            )
        L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(met: dict, tag: str, odir, docs: bool) -> list[str]:
    F.style()
    paths = []
    fz = Path(odir) / "fields.npz"
    fields = dict(np.load(fz)) if fz.exists() else {}
    if met.get("graetz"):
        paths += save_fig(F.graetz_nu(met), f"p6_6_graetz_nu_{tag}", odir, docs=docs)
        paths += save_fig(F.graetz_surface3d(met), f"p6_6_graetz_surface3d_{tag}", odir, docs=docs)
        paths += save_fig(F.graetz_field3d(met, fields), f"p6_6_graetz_field3d_{tag}", odir, docs=docs)
    if met.get("pinn"):
        paths += save_fig(F.pinn_nu(met["pinn"]), f"p6_6_pinn_nu_{tag}", odir, docs=docs)
    if met.get("natural"):
        paths += save_fig(F.natural_nu(met), f"p6_6_natural_nu_{tag}", odir, docs=docs)
        paths += save_fig(F.natural_fields(met, fields), f"p6_6_natural_fields_{tag}", odir, docs=docs)
        paths += save_fig(F.natural_field3d(met, fields), f"p6_6_natural_field3d_{tag}", odir, docs=docs)
    return [str(p) for p in paths]


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"done in {time.perf_counter() - t0:.0f} s")
