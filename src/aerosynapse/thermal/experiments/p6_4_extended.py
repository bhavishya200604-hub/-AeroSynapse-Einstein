"""Step 6.4b: solver cost and grid convergence (with a prediction for a large TARGET grid, e.g. 1024 x 1024), and the
2D / 3D figures of the Step 6.4 sensitivity study.

--part grid   for every case: a ladder of grids at a few parameter points, ONE solve per fresh process (wall time,
              memory above the pre-solve level, QoI); grids are run in increasing order and a grid is started
              only if the cost model fitted to the grids already done predicts it within max_memory_fraction x RAM
              and max_solve_s; power-law fits of time and memory vs number of cells with 95 % prediction intervals;
              prediction for the target grid (one solve, and the whole Sobol study of Step 6.4 at that grid);
              grid convergence of the QoI (observed order, Richardson value, GCI); the Sobol study at the target grid is
              run only when predicted to fit (run_target auto) or forced
--part plots  from the saved Step 6.4 run (solver and surrogate outputs on the Saltelli design): PCE of the solver and
              of the PINN (ensemble and members) -> response maps, 3D surfaces, main effects, interaction indices and
              variance decomposition, QoI distribution; solver vs PINN fields at a parameter point (2D and 3D)

    python experiments/p6_4_extended.py [--part grid|plots|all] [--config my.yaml] [--set key=value] [--tag name]
                                        [--cases D_graetz,C_cavity] [--figures-only]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from aerosynapse.thermal.experiments import p6_4_sensitivity as P64
from aerosynapse.thermal.numerics.convergence import gci
from aerosynapse.thermal.pinn import sensitivity as SA
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.utils import costmodel as CM
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz import sensitivity_plots as F
from aerosynapse.thermal.viz.style import save_fig

MOD = "aerosynapse.thermal.experiments.p6_4_extended"
NAMES = P64.NAMES


def cells(case: str, grid: int, Lx: float = 8.0) -> int:
    return int(round(Lx * grid * grid)) if case == "D_graetz" else int(grid * grid)


def point_values(spec_pt, box: dict) -> dict:
    if isinstance(spec_pt, dict):
        return {k: float(spec_pt[k]) for k in NAMES}
    u = {"centre": 0.5, "low": 0.0, "high": 1.0}[spec_pt]
    return {k: float(np.exp(np.log(box[k][0]) + u * (np.log(box[k][1]) - np.log(box[k][0])))) for k in NAMES}


# --------------------------------------------------------------------------- one measured solve (own process)
def measure_task(case: str, kw: dict, warm: dict) -> dict:
    """One grid in a fresh process. (1) Untimed warm-up solve of the SAME problem with scipy's SuperLU wrapped to count
    the largest factorisation: unknowns n, nnz of the Jacobian and of L + U -> direct-solver memory estimate
    12 (nnz_LU + 2 nnz_J) + 80 n bytes (8-byte values + 4-byte indices; CSR and CSC copies of J; ~10 work vectors);
    (2) the timed solve (unpatched); (3) the process's absolute peak memory (imports included)."""
    import scipy.sparse.linalg as spla

    meth = kw.pop("_method")
    orig, st = spla.splu, {"n": 0, "nnz_J": 0, "nnz_LU": 0, "factorisations": 0}

    def counted(A, *a, **k):
        lu = orig(A, *a, **k)
        st["factorisations"] += 1
        nl = int(lu.L.nnz + lu.U.nnz)
        if nl > st["nnz_LU"]:
            st.update(n=int(A.shape[0]), nnz_J=int(A.nnz), nnz_LU=nl)
        return lu

    spla.splu = counted
    try:
        P64.solve_chunk(case, meth, dict(kw))
    finally:
        spla.splu = orig
    t0 = time.perf_counter()
    q = P64.solve_chunk(case, meth, kw)[0]
    wall = time.perf_counter() - t0
    mem = 12 * (st["nnz_LU"] + 2 * st["nnz_J"]) + 80 * st["n"]
    peak = CM.peak_memory_bytes()
    return {
        "qoi": float(q),
        "wall_s": wall,
        "mem_bytes": int(max(mem, 1)),
        "peak_process_bytes": peak,
        "baseline_bytes": int(max(peak - mem, 0)),  # process memory not explained by the solver (imports, Python)
    } | st


def _fit(rows: list[dict], key: str, last: int) -> dict:
    """Power law on the `last` largest grids measured (the asymptotic regime of the cost)."""
    g = sorted({r["cells"] for r in rows})[-last:]
    sel = [r for r in rows if r["cells"] in g]
    return CM.fit_power([r["cells"] for r in sel], [r[key] for r in sel])


def grid_part(st: dict, cases: list[str], ex) -> dict:
    gs = st["grid_study"]
    mach = CM.machine()
    out = {"machine": mach}
    last = int(gs.get("fit_last", 4))
    # memory guard: the solver's EXTRA memory must stay below the memory free at the start (paging distorts timings) and
    # the whole process (baseline + solver) below max_memory_fraction x RAM
    mem_total_limit = float(gs["max_memory_fraction"]) * (mach.get("total_memory") or np.inf)
    mem_free = float(mach.get("available_memory") or np.inf)
    out["memory_limit_bytes"] = min(mem_total_limit, mem_free)
    for name in cases:
        ad = P64.ADAPTERS[name](st["cases"][name])
        cg = gs["cases"][name]
        pts = [point_values(p, ad.box) for p in gs["points"]]
        key = "cells_per_unit" if name == "D_graetz" else "grid"
        Lx = getattr(ad, "Lx", 1.0)
        rows, skipped = [], []
        for g in cg["grids"]:
            nc = cells(name, g, Lx)
            if len({r["cells"] for r in rows}) >= 3:  # point prediction from the largest grids before starting
                ft, fm = _fit(rows, "wall_s", last), _fit(rows, "mem_bytes", last)
                pt_ = float(CM.predict(ft, nc)["value"])
                extra = float(CM.predict(fm, nc)["value"])
                pm = extra + min(r["baseline_bytes"] for r in rows)
                if extra > mem_free or pm > mem_total_limit or pt_ > float(gs["max_solve_s"]):
                    skipped.append({"grid": g, "cells": nc, "pred_time_s": pt_, "pred_mem_bytes": pm})
                    print(
                        f"{name}: grid {g} skipped (predicted {CM.human_time(pt_)}, {pm / 2**30:.2f} GiB)", flush=True
                    )
                    continue
            jobs = []
            for p in pts:
                meth, kw = ad.solver_job([[p[k] for k in NAMES]], {key: g})
                jobs.append((MOD, "measure_task", dict(case=name, kw={**kw, "_method": meth}, warm={})))
            res = run_tasks(jobs, {"isolate": True, "workers": int(gs.get("workers", 1)), "threads_per_worker": 0})
            for p, r in zip(pts, res, strict=True):
                rows.append({"grid": g, "cells": nc, "point": p} | r)
            print(
                f"{name}: grid {g} ({nc} cells): "
                + ", ".join(f"{r['wall_s']:.2f} s / {r['mem_bytes'] / 2**20:.1f} MiB" for r in res),
                flush=True,
            )
        ft, fm = _fit(rows, "wall_s", last), _fit(rows, "mem_bytes", last)
        nt = cells(name, int(cg["target"]), Lx)
        n_runs = int(st["cases"][name]["solver"]["n"]) * 5
        pt_, pm = CM.predict(ft, nt), CM.predict(fm, nt)
        base = min(r["baseline_bytes"] for r in rows)
        workers = int(st.get("execution", {}).get("workers", 2))
        pred = {
            "target": int(cg["target"]),
            "cells": nt,
            "time_per_solve_s": [float(pt_["value"]), float(pt_["lo"]), float(pt_["hi"])],
            "memory_bytes": [float(pm["value"]) + base, float(pm["lo"]) + base, float(pm["hi"]) + base],
            "sobol_runs": n_runs,
            "sobol_time_s": [float(v) * n_runs / workers for v in (pt_["value"], pt_["lo"], pt_["hi"])],
            "fits_in_memory": bool(
                float(pm["value"]) + base <= float(gs["max_memory_fraction"]) * (mach.get("total_memory") or np.inf)
            ),
        }
        # grid convergence of the QoI at every point (three finest grids done)
        conv = []
        for p in pts:
            rs = sorted([r for r in rows if r["point"] == p], key=lambda r: r["grid"])
            e = {"point": p, "grid": [r["grid"] for r in rs], "qoi": [r["qoi"] for r in rs]}
            if len(rs) >= 3:
                g3 = rs[-3:]
                h = [1.0 / r["grid"] for r in g3]
                e["gci"] = gci(g3[2]["qoi"], g3[1]["qoi"], g3[0]["qoi"], h[2], h[1], h[0])
            conv.append(e)
        out[name] = {
            "rows": rows,
            "skipped": skipped,
            "fit_time": ft,
            "fit_memory": fm,
            "prediction": pred,
            "convergence": conv,
            "key": key,
        }
        print(
            f"{name}: target {cg['target']} ({nt} cells): {CM.human_time(pred['time_per_solve_s'][0])} per solve "
            f"[{CM.human_time(pred['time_per_solve_s'][1])}, {CM.human_time(pred['time_per_solve_s'][2])}], "
            f"{pred['memory_bytes'][0] / 2**30:.3g} GiB; Sobol study {CM.human_time(pred['sobol_time_s'][0])}; "
            f"fits: {pred['fits_in_memory']}",
            flush=True,
        )
        mode = gs.get("run_target", "auto")
        ok = mode == "force" or (mode == "auto" and pred["fits_in_memory"] and pred["sobol_time_s"][2] < 7 * 86400)
        out[name]["target_run"] = "started" if ok else f"not started ({mode}; predicted not to fit this machine)"
        if ok:
            s = SP.deep_merge(st["cases"][name], {"solver": {key: int(cg["target"])}})
            out[name]["target_sobol"] = P64.run_case(name, s, st, ex)
    return out


# --------------------------------------------------------------------------- plot data from the saved 6.4 run
def plot_part(st: dict, cases: list[str], run: Path) -> dict:
    met = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    pl = st["plots"]
    res = {}
    for name in cases:
        c = met["cases"][name]
        z = np.load(run / f"{name}_outputs.npz")
        deg = int(st["cases"][name]["pce_degree"])
        box = c["box"]
        ranges = [(box[k][0], box[k][1], "log") for k in NAMES]
        pce_s = SA.pce_fit(z["U_s"], z["f_solver"], deg)
        members = sorted(k for k in z.files if k.startswith("pinn_"))
        F_p = np.stack([z[k] for k in members])
        pce_p = SA.pce_fit(z["U_pinn"], F_p.mean(axis=0), deg)
        pce_m = [SA.pce_fit(z["U_pinn"], f, deg) for f in F_p]
        fixed_u = 0.5 if pl["fixed"] == "centre" else None
        if fixed_u is None:
            fixed_u = {
                k: (np.log(pl["fixed"][k]) - np.log(box[k][0])) / (np.log(box[k][1]) - np.log(box[k][0])) for k in NAMES
            }
        n = int(pl["resolution"])
        g = np.linspace(0, 1, n)
        maps = {}
        for a, b in ((0, 1), (0, 2), (1, 2)):
            U = np.full((n * n, 3), 0.5)
            if isinstance(fixed_u, dict):
                for i, k in enumerate(NAMES):
                    U[:, i] = fixed_u[k]
            A_, B_ = np.meshgrid(g, g)
            U[:, a], U[:, b] = A_.ravel(), B_.ravel()
            maps[f"{NAMES[a]}-{NAMES[b]}"] = {
                "x": SA.to_phys(np.column_stack([g, g, g]), ranges)[:, a].tolist(),
                "y": SA.to_phys(np.column_stack([g, g, g]), ranges)[:, b].tolist(),
                "solver": SA.pce_eval(pce_s, U).reshape(n, n).tolist(),
                "pinn": SA.pce_eval(pce_p, U).reshape(n, n).tolist(),
            }

        # main effects E[Q | x_i] - E[Q] from the univariate PCE terms
        def effect(m, i, u):
            al = [tuple(x) for x in m["alphas"]]
            keep = [j for j, x in enumerate(al) if x[i] > 0 and sum(x) == x[i]]
            U = np.full((len(u), 3), 0.5)
            U[:, i] = u
            return (SA._basis(U, [al[j] for j in keep]) @ np.asarray(m["coef"])[keep]).tolist()

        main = {
            k: {
                "x": SA.to_phys(np.column_stack([g, g, g]), ranges)[:, i].tolist(),
                "solver": effect(pce_s, i, g),
                "pinn": effect(pce_p, i, g),
                "members": [effect(m, i, g) for m in pce_m],
            }
            for i, k in enumerate(NAMES)
        }
        res[name] = {
            "qoi": c["qoi"],
            "box": box,
            "maps": maps,
            "main_effects": main,
            "decomposition": {"solver": F.decompose(pce_s), "pinn": F.decompose(pce_p)},
            "pce_loo": {"solver": pce_s["loo_rel_error"], "pinn": pce_p["loo_rel_error"]},
            "distribution": {"solver": z["f_solver"].tolist(), "pinn": F_p.mean(axis=0)[: len(z["f_solver"])].tolist()},
            "fields": fields(name, st, point_values(pl["field_point"], box)),
        }
    return res


def fields(name: str, st: dict, p: dict) -> dict:
    """Solver (finer check grid) and PINN ensemble-mean fields u, v, theta at one parameter point."""
    from aerosynapse.thermal.pinn import problems2d as P2

    ad = P64.ADAPTERS[name](st["cases"][name])
    vals = {**ad.fixed, **p}
    if name == "D_graetz":
        from aerosynapse.thermal.pinn import channel2d as CH

        r = CH.solve_channel(vals, ad.pcfg, int(st["cases"][name]["check_grid"]["cells_per_unit"]), 1e-10, "quick")
    else:
        from aerosynapse.thermal.pinn.ref2d import solve_steady_decoupled

        r = solve_steady_decoupled(vals, ad.cond, int(st["cases"][name]["check_grid"]["grid"]), 1e-10, ad.scheme)
    x, y = np.asarray(r["x"]), np.asarray(r["y"])
    X, Y = np.meshgrid(x, y)
    preds = []
    for _, prob, net in ad.members:
        p0 = next(net.parameters())
        Xin = (
            prob.inputs(np.stack([X.ravel(), Y.ravel()], 1), vals, p0)
            if name == "D_graetz"
            else prob.grid_inputs(x, y, vals, p0)
        )
        pr = P2._batched(prob, net, Xin, 0, P2.FIELDS)
        preds.append({k: pr[k].reshape(X.shape) for k in ("u", "v", "theta")})
    mean = {k: np.mean([q[k] for q in preds], axis=0) for k in ("u", "v", "theta")}
    solid = np.asarray(r.get("solid", np.zeros(X.shape, bool)))
    return {
        "point": p,
        "x": x.tolist(),
        "y": y.tolist(),
        "solver": {k: np.where(solid, np.nan, np.asarray(r[k])).tolist() for k in ("u", "v", "theta")},
        "pinn": {k: np.where(solid, np.nan, mean[k]).tolist() for k in ("u", "v", "theta")},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="all", choices=["all", "grid", "plots"])
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    spec = SP.load_spec(a.config or None, a.profile, SP.parse_set(a.sets))
    st = spec["sensitivity_study"]
    custom = bool(a.config or a.sets or a.tag != "default" or a.profile != "smoke" or a.cases)
    odir = ROOT / "results" / f"p6_4_extended_{a.tag}"
    table = odir / "extended.md" if custom else ROOT / "docs" / "tables" / "phase6_4_extended.md"
    cases = [c.strip() for c in a.cases.split(",")] if a.cases else list(st["cases"])
    if a.figures_only:
        met = json.loads((odir / "metrics.json").read_text(encoding="utf-8"))
        met["figures"] = figures(met, a.tag, odir, docs=not custom)
        report(met, table, a.tag)
        (odir / "metrics.json").write_text(json.dumps(met, indent=2), encoding="utf-8")
        print(f"wrote {table}")
        return
    prev = odir / "metrics.json"
    keep = json.loads(prev.read_text(encoding="utf-8")) if prev.exists() else {}
    with Experiment(f"p6_4_extended_{a.tag}", spec) as ex:
        for k in ("grid", "plots"):
            if k in keep and a.part not in ("all", k):
                ex.metrics[k] = keep[k]  # the other part of an earlier run is kept
        if a.part in ("all", "grid"):
            ex.metrics["grid"] = grid_part(st, cases, ex)
        if a.part in ("all", "plots"):
            run = (
                ROOT / "results" / f"p6_4_sensitivity_{a.tag}"
                if st["plots"]["run"] == "auto"
                else ROOT / st["plots"]["run"]
            )
            ex.metrics["plots"] = plot_part(st, cases, run)
            ex.metrics["plots_from"] = str(run)
        ex.metrics["git"] = git_hash()
        ex.metrics["figures"] = figures(ex.metrics, a.tag, ex.dir, docs=not custom)
        report(ex.metrics, table, a.tag)
        ex.metrics["table"] = str(table.relative_to(ROOT))
    print(f"wrote {table}")


def report(met: dict, path: Path, tag: str) -> None:
    L = [
        f"# Step 6.4b solver cost, grid convergence and figures — `{tag}` (auto-generated by experiments/p6_4_extended.py)",
        "",
    ]
    g = met.get("grid")
    if g:
        m = g["machine"]
        L += [
            f"Machine: {m.get('cpus')} logical CPUs, {m.get('total_memory', 0) / 2**30:.1f} GiB RAM "
            f"({m.get('available_memory', 0) / 2**30:.1f} GiB free at the start; a grid is started only if its predicted solver "
            "memory fits in the free memory and the whole process in max_memory_fraction x RAM). One grid per fresh process: an untimed warm-up "
            "solve counts the largest SuperLU factorisation (memory = 12 bytes per stored non-zero of L + U and of two "
            "Jacobian copies + 80 bytes per unknown — a lower bound of the direct solver's need), then the timed solve; "
            "predicted memory = that model + the process baseline (measured peak minus the model). Predictions = "
            "power-law fit on the largest grids measured (fit_last) vs number of cells, with its 95 % prediction interval.",
            "",
        ]
        for name, c in g.items():
            if name in ("machine", "memory_limit_bytes"):
                continue
            ft, fm, p = c["fit_time"], c["fit_memory"], c["prediction"]
            L += [
                f"## {name}",
                "",
                "| grid | cells | wall time per solve (points) | direct-solver memory, LU model [MiB] | QoI (points) |",
                "|---|---|---|---|---|",
            ]
            for gr in sorted({r["grid"] for r in c["rows"]}):
                rs = [r for r in c["rows"] if r["grid"] == gr]
                L.append(
                    f"| {gr} | {rs[0]['cells']} | "
                    + " / ".join(f"{r['wall_s']:.2f} s" for r in rs)
                    + " | "
                    + " / ".join(f"{r['mem_bytes'] / 2**20:.0f}" for r in rs)
                    + " | "
                    + " / ".join(f"{r['qoi']:.5g}" for r in rs)
                    + " |"
                )
            for s in c["skipped"]:
                L.append(
                    f"| {s['grid']} | {s['cells']} | skipped: predicted {CM.human_time(s['pred_time_s'])} | {s['pred_mem_bytes'] / 2**30:.2f} GiB (over a limit) | — |"
                )
            L += [
                "",
                f"Fits: time ∝ cells^{ft['b']:.2f} (± {ft['b_se']:.2f}); memory ∝ cells^{fm['b']:.2f} (± {fm['b_se']:.2f}).",
                f"**Target grid {p['target']} ({p['cells']:,} cells): {CM.human_time(p['time_per_solve_s'][0])} per solve "
                f"(95 % {CM.human_time(p['time_per_solve_s'][1])} – {CM.human_time(p['time_per_solve_s'][2])}), "
                f"{p['memory_bytes'][0] / 2**30:.3g} GiB (95 % {p['memory_bytes'][1] / 2**30:.3g} – {p['memory_bytes'][2] / 2**30:.3g}); "
                f"the Sobol study ({p['sobol_runs']} solves) {CM.human_time(p['sobol_time_s'][0])} "
                f"(95 % {CM.human_time(p['sobol_time_s'][1])} – {CM.human_time(p['sobol_time_s'][2])}). "
                f"Fits in memory: {p['fits_in_memory']}. Target run: {c['target_run']}.**",
                "",
                "| point (Re, Pr, Bi) | QoI per grid | observed order | Richardson value | GCI fine |",
                "|---|---|---|---|---|",
            ]
            for e in c["convergence"]:
                q = e.get("gci") or {}
                pv = e["point"]
                L.append(
                    f"| ({pv['Re']:.3g}, {pv['Pr']:.3g}, {pv['Bi']:.3g}) | "
                    + " / ".join(f"{v:.5g}" for v in e["qoi"])
                    + f" | {q.get('p', float('nan')):.2f} ({q.get('type', '—')}) | {q.get('f_ext', float('nan')):.5g} | "
                    + (f"{100 * q['GCI_fine21']:.2g} %" if "GCI_fine21" in q else "—")
                    + " |"
                )
            L.append("")
    pl = met.get("plots")
    if pl:
        L += ["## Figures from the Step 6.4 run (PCE of solver and PINN)", ""]
        for name, c in pl.items():
            d = c["decomposition"]
            L.append(
                f"- {name}: PCE LOO error solver {c['pce_loo']['solver']:.2g}, PINN {c['pce_loo']['pinn']:.2g}; variance shares "
                f"(first order / pairs / triple) solver {d['solver']['first']:.3f} / {d['solver']['pairs']:.3f} / {d['solver']['triple']:.3f}, "
                f"PINN {d['pinn']['first']:.3f} / {d['pinn']['pairs']:.3f} / {d['pinn']['triple']:.3f}; largest pair: solver "
                f"{max(d['solver']['S_ij'], key=d['solver']['S_ij'].get)} {max(d['solver']['S_ij'].values()):.3f}, PINN "
                f"{max(d['pinn']['S_ij'], key=d['pinn']['S_ij'].get)} {max(d['pinn']['S_ij'].values()):.3f}"
            )
        L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(met: dict, tag: str, odir, docs: bool) -> list[str]:
    F.style()
    paths = []
    if met.get("grid"):
        paths += save_fig(F.cost(met["grid"]), f"p6_4_cost_{tag}", odir, docs=docs)
        paths += save_fig(F.grid_convergence(met["grid"]), f"p6_4_gridconv_{tag}", odir, docs=docs)
    pl = met.get("plots")
    if pl:
        paths += save_fig(F.surfaces3d(pl), f"p6_4_surfaces3d_{tag}", odir, docs=docs)
        for name in pl:
            paths += save_fig(F.response_maps(pl[name], name), f"p6_4_maps_{name}_{tag}", odir, docs=docs)
        paths += save_fig(F.main_effects(pl), f"p6_4_main_effects_{tag}", odir, docs=docs)
        paths += save_fig(F.interactions(pl), f"p6_4_interactions_{tag}", odir, docs=docs)
        paths += save_fig(F.distribution(pl), f"p6_4_distribution_{tag}", odir, docs=docs)
        for name in pl:
            paths += save_fig(F.fields2d(pl[name], name), f"p6_4_fields_{name}_{tag}", odir, docs=docs)
        paths += save_fig(F.field3d(pl), f"p6_4_field3d_{tag}", odir, docs=docs)
    return [str(p) for p in paths]


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"done in {time.perf_counter() - t0:.0f} s")
