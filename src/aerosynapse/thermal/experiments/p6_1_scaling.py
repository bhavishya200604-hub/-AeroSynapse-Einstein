"""Step 6.1: convergence / scaling of the PINN — error vs width, depth, collocation points and training epochs; fitted
power laws; PINN vs mesh solvers (FD, Chebyshev) at equal error.

For every problem of scaling_study.problems (Phase 3 problems with analytic truth) and every level of every sweep
(one factor at a time around the profile's base settings), scaling_study.seeds networks are trained (one fresh process
each) and evaluated against the analytic solution on the evaluation grid (relative L2). The base setting is trained
once per seed and shared by the four sweeps. Power laws: e = C x^p and e = A x^-q + floor, AIC choice, bootstrap
interval over the seeds (src/pinn/scaling.py). Mesh solvers: FD (central, Crank-Nicolson) and Chebyshev on grids of
increasing size, same truth and evaluation grid, best-of-repeats wall time; equal error = the cheapest run reaching each
target. Timings are single-thread on the stated machine (2 trainings in parallel): comparable within this study.

    python experiments/p6_1_scaling.py [--profile smoke] [--config my.yaml] [--set key=value] [--tag name] [--figures-only]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from aerosynapse.thermal.experiments.p3_pinn1d import reference_solvers, train_one
from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import scaling as S
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz import scaling_figs as F
from aerosynapse.thermal.viz.style import save_fig

MOD = "aerosynapse.thermal.experiments.p6_1_scaling"
OVERRIDE = {
    "width": lambda v: {"network": {"width": int(v)}},
    "depth": lambda v: {"network": {"depth": int(v)}},
    "n_pde": lambda v: {"sampling": {"n_pde": int(v)}},
    "adam_epochs": lambda v: {"training": {"adam": {"epochs": int(v)}}},
    "lbfgs_iterations": lambda v: {"training": {"lbfgs": {"iterations": int(v)}}},
}
LABEL = {
    "width": "width (neurons per layer)",
    "depth": "depth (hidden layers)",
    "n_pde": "collocation points",
    "adam_epochs": "Adam epochs",
    "lbfgs_iterations": "L-BFGS iterations",
}


def base_levels(spec: dict) -> dict:
    return {
        "width": spec["network"]["width"],
        "depth": spec["network"]["depth"],
        "n_pde": spec["sampling"]["n_pde"],
        "adam_epochs": spec["training"]["adam"]["epochs"],
        "lbfgs_iterations": spec["training"]["lbfgs"]["iterations"],
    }


def train_task(files, profile: str, cli: dict, problem: str, value: float, over: dict, seed: int, out_dir: str) -> dict:
    """One training run (own process): the setting's overrides on top of the profile, evaluated vs the analytic truth."""
    spec = SP.load_spec(files or None, profile, SP.deep_merge(cli, SP.deep_merge(over, {"seed": int(seed)})))
    prob = P.build_problem(problem, spec["problems"][problem], value)
    out = train_one(spec, prob, [value], Path(out_dir), f"{problem}_s{seed}")
    rec, _ = out["records"][0]
    keep = (
        "rel_l2",
        "linf",
        "rel_h1",
        "wall_s",
        "n_params",
        "status",
        "final_losses",
        "val_loss",
        "adam_epochs",
        "lbfgs_iterations",
    )
    return {k: rec[k] for k in keep if k in rec}


def solver_curves(spec: dict, problem: str, value: float, so: dict) -> dict:
    """FD and Chebyshev on grids of increasing size: relative L2 vs the analytic truth and best-of-repeats wall time."""
    prob = P.build_problem(problem, spec["problems"][problem], value)
    nx, nt = spec["evaluation"]["nx"], spec["evaluation"]["nt"]
    out = {"fd": [], "spectral": []}
    for kind, Ns in (("fd", so["fd_N"]), ("spectral", so["spectral_N"])):
        for N in Ns:
            rs = {
                "fd_N": int(N) if kind == "fd" else 10,
                "spectral_N": int(N) if kind == "spectral" else 6,
                "fd_dt": float(so["fd_dt_per_h"]) / (int(N) if kind == "fd" else 10),
                "fd_solver": so.get("fd_solver", "splu"),  # compiled SuperLU (the Phase 1 Thomas is a Python loop)
            }
            best, err = np.inf, None
            for _ in range(int(so.get("repeats", 1))):
                r = reference_solvers(prob, value, nx, nt, rs)[kind]
                best, err = min(best, r["wall_s"]), r["rel_l2"]
            out[kind].append({"N": int(N), "rel_l2": float(err), "wall_s": float(best)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--diagnose", action="store_true", help="error location of the saved base networks (no training)")
    ap.add_argument(
        "--resummarise",
        action="store_true",
        help="re-analyse the saved trainings (fits, settings, equal error) and re-time the mesh solvers; no training",
    )
    a = ap.parse_args()
    cli = SP.parse_set(a.sets)
    spec = SP.load_spec(a.config or None, a.profile, cli)
    st = spec["scaling_study"]
    custom = bool(a.config or a.sets or a.tag != "default" or a.profile != "smoke")
    odir = ROOT / "results" / f"p6_1_scaling_{a.tag}"
    table = odir / "scaling.md" if custom else ROOT / "docs" / "tables" / "phase6_1_scaling.md"
    if a.figures_only or a.resummarise or a.diagnose:
        met = json.loads((odir / "metrics.json").read_text(encoding="utf-8"))
        if a.diagnose:
            met["diagnosis"] = diagnose(spec, odir, met) | {"git": git_hash()}
            (odir / "metrics.json").write_text(json.dumps(met, indent=2), encoding="utf-8")
        if a.resummarise:
            met["fits"], met["solvers"], met["equal_error"], met["settings"] = summarise(spec, met["runs"], met["base"])
            met["resummarised"] = {
                "git": git_hash(),
                "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "solvers": st["solvers"],
            }
            (odir / "metrics.json").write_text(json.dumps(met, indent=2), encoding="utf-8")
        figures(met, a.tag, odir, docs=not custom)
        report(met, table, a.tag)
        print(f"wrote {table}")
        return
    with Experiment(f"p6_1_scaling_{a.tag}", spec) as ex:
        base = base_levels(spec)
        jobs, keys = [], []
        seen = {}
        for prob, pc in st["problems"].items():
            for factor, levels in st["sweeps"].items():
                for lv in levels:
                    is_base = int(lv) == int(base[factor])
                    for sd in st["seeds"]:
                        key = (prob, "base" if is_base else factor, "base" if is_base else int(lv), int(sd))
                        if key in seen:
                            continue
                        seen[key] = len(jobs)
                        over = {} if is_base else OVERRIDE[factor](lv)
                        jobs.append(
                            (
                                MOD,
                                "train_task",
                                dict(
                                    files=a.config,
                                    profile=a.profile,
                                    cli=cli,
                                    problem=prob,
                                    value=float(pc["value"]),
                                    over=over,
                                    seed=int(sd),
                                    out_dir=str(ex.dir / "runs" / f"{prob}_{key[1]}_{key[2]}_s{sd}"),
                                ),
                            )
                        )
                        keys.append(key)
        print(f"{len(jobs)} trainings ({len(st['problems'])} problems)", flush=True)
        exe = {"isolate": True, **st.get("execution", {})}
        outs = run_tasks(
            jobs,
            exe,
            lambda i, r, sec: print(
                f"[{i + 1}/{len(jobs)}] {keys[i]}: rel L2 {r['rel_l2']:.3g}, {r['wall_s']:.0f} s", flush=True
            ),
        )
        runs = [
            dict(zip(("problem", "factor", "level", "seed"), k, strict=True)) | r
            for k, r in zip(keys, outs, strict=True)
        ]
        for r in runs:
            ex.log(**{k: v for k, v in r.items() if not isinstance(v, dict)})
        fits, solvers, equal, settings = summarise(spec, runs, base)
        ex.metrics.update(
            {
                "runs": runs,
                "fits": fits,
                "solvers": solvers,
                "equal_error": equal,
                "settings": settings,
                "base": base,
                "problems": st["problems"],
                "sweeps": st["sweeps"],
                "seeds": st["seeds"],
                "git": git_hash(),
            }
        )
        ex.metrics["figures"] = figures(ex.metrics, a.tag, ex.dir, docs=not custom)  # custom runs: results folder only
        report(ex.metrics, table, a.tag)
        ex.metrics["table"] = str(table.relative_to(ROOT))
    print(f"wrote {table}")


def diagnose(spec: dict, run_dir: Path, met: dict) -> dict:
    """Where is the error of the BASE networks? Reloads each saved base checkpoint and evaluates it on the evaluation
    grid: transient problems -> relative L2 per time slice and the share of the squared error in the first slices;
    every problem -> share of the squared error within 0.1 of each end of the domain. No training."""
    import torch

    from aerosynapse.thermal.pinn.architectures import build_network
    from aerosynapse.thermal.pinn.metrics import predict_grid

    st, ev = spec["scaling_study"], spec["evaluation"]
    out = {}
    for prob, pc in st["problems"].items():
        pb = P.build_problem(prob, spec["problems"][prob], float(pc["value"]))
        x = np.linspace(0.0, 1.0, ev["nx"])
        t = (pb.T * np.arange(1, ev["nt"] + 1) / ev["nt"]) if pb.has_time else None
        ref = pb.reference(x, t, float(pc["value"]))
        rows = []
        for sd in met["seeds"]:
            ck = run_dir / "runs" / f"{prob}_base_base_s{sd}" / "checkpoint.pt"
            if not ck.exists():
                continue
            net = build_network(spec["network"], pb.lo, pb.hi).to(dtype=torch.float64)
            net.load_state_dict(torch.load(ck, weights_only=False)["model"])
            u, _ = predict_grid(net, pb, x, t, float(pc["value"]))
            e2 = (u - ref) ** 2
            r = {"seed": int(sd), "rel_l2": float(np.sqrt(e2.sum() / (ref**2).sum()))}
            xa = e2 if not pb.has_time else e2.sum(axis=0)
            r["share_x_lt_0.1"] = float(xa[x < 0.1].sum() / xa.sum())
            r["share_x_gt_0.9"] = float(xa[x > 0.9].sum() / xa.sum())
            if pb.has_time:
                r["t"] = t.tolist()
                r["rel_l2_per_t"] = (np.sqrt(e2.sum(axis=1) / (ref**2).sum(axis=1))).tolist()
                r["share_first_3_slices"] = float(e2[:3].sum() / e2.sum())
                r["share_last_half"] = float(e2[len(t) // 2 :].sum() / e2.sum())
            rows.append(r)
        out[prob] = rows
    return out


def summarise(spec: dict, runs: list[dict], base: dict) -> tuple[dict, dict, dict, dict]:
    """Fits per factor (median over seeds), mesh-solver curves, per-setting medians and the equal-error table.
    Equal error uses SETTINGS (median error and median time over the seeds), not the luckiest single seed."""
    st = spec["scaling_study"]
    fits, solvers, equal, settings = {}, {}, {}, {}
    for prob, pc in st["problems"].items():
        solvers[prob] = solver_curves(spec, prob, float(pc["value"]), st["solvers"])
        pr = [r for r in runs if r["problem"] == prob]
        fits[prob] = {}
        for factor, levels in st["sweeps"].items():
            lv_sorted = sorted({int(v) for v in levels})
            vals = []
            for lv in lv_sorted:
                is_base = lv == int(base[factor])
                if is_base:
                    sel = [r for r in pr if r["factor"] == "base"]
                else:
                    sel = [r for r in pr if r["factor"] == factor and r["level"] == lv]
                vals.append(np.array([r["rel_l2"] for r in sel]))
            fits[prob][factor] = S.fit(np.array(lv_sorted), vals, int(st.get("bootstrap", 1000)))
        settings[prob] = S.by_setting(pr)
        e_s = np.array([s["rel_l2"] for s in settings[prob]])
        t_s = np.array([s["wall_s"] for s in settings[prob]])
        equal[prob] = []
        for tgt in st["targets"]:
            row = {"target": float(tgt), "pinn": S.cheapest(e_s, t_s, tgt)}
            if row["pinn"]:
                w = settings[prob][row["pinn"]["index"]]
                row["pinn"]["setting"] = "base" if w["factor"] == "base" else f"{w['factor']} {w['level']}"
                row["pinn"]["error_max"] = w["rel_l2_max"]
            for kind in ("fd", "spectral"):
                c = solvers[prob][kind]
                ch = S.cheapest(np.array([x["rel_l2"] for x in c]), np.array([x["wall_s"] for x in c]), tgt)
                if ch:
                    ch["N"] = c[ch["index"]]["N"]
                row[kind] = ch
            equal[prob].append(row)
    return fits, solvers, equal, settings


def _exponent(f: dict) -> str:
    """Exponent with its bootstrap interval; at the fit bound: unresolved (a step to the floor, not a rate)."""
    if not f.get("resolved", True):
        return "unresolved (step to the floor)"
    ci = f["exponent_ci95"]
    if not ci:
        return f"{f['exponent']:+.2f}"
    lo = f"≤ {ci[0]:+.0f} (bound)" if f.get("ci_at_bound") else f"{ci[0]:+.2f}"
    return f"{f['exponent']:+.2f} [{lo}, {ci[1]:+.2f}]"


def _f(v, fmt=".3g"):
    return "—" if v is None else format(v, fmt)


def report(met: dict, path, tag: str) -> None:
    b = met["base"]
    L = [
        f"# Step 6.1 convergence and scaling — `{tag}` (auto-generated by experiments/p6_1_scaling.py)",
        "",
        f"One factor at a time around the base setting (width {b['width']}, depth {b['depth']}, {b['n_pde']} collocation points, Adam "
        f"{b['adam_epochs']} + L-BFGS {b['lbfgs_iterations']}); seeds {met['seeds']}; error = relative L2 vs the analytic solution on the "
        "evaluation grid. Fits on the median over seeds: power law e = C·x^p and saturating e = A·x^−q + floor (AIC choice); 95 % "
        "interval of the exponent by bootstrap over the seeds; an exponent at the fit bound is reported as unresolved. Equal error: "
        "the cheapest SETTING whose median error over the seeds meets the target. FD solves use the compiled sparse LU. Wall times: single-thread trainings, two in parallel (comparable "
        "within this study).",
        "",
    ]
    for prob, fits in met["fits"].items():
        L += [
            f"## {prob} ({SP.PROBLEMS.get(prob, 'parameter')} = {met['problems'][prob]['value']:g})",
            "",
            "| factor | levels | median rel L2 per level | best model | exponent [95 %] | floor | R² (power law) |",
            "|---|---|---|---|---|---|---|",
        ]
        for factor, f in fits.items():
            fl = f["saturating"]["floor"] if f["best"] == "saturating" else None
            L.append(
                f"| {factor} | {', '.join(str(int(x)) for x in f['levels'])} | {', '.join(f'{v:.2g}' for v in f['median'])} | {f['best']} | "
                f"{_exponent(f)} | {_f(fl, '.2g')} | {f['loglog']['r2']:.2f} |"
            )
        sv = met["solvers"][prob]
        L += [
            "",
            "Mesh solvers (same truth and grid): FD "
            + ", ".join(f"N {x['N']}: {x['rel_l2']:.2g} ({1e3 * x['wall_s']:.3g} ms)" for x in sv["fd"])
            + "; Chebyshev "
            + ", ".join(f"N {x['N']}: {x['rel_l2']:.2g} ({1e3 * x['wall_s']:.3g} ms)" for x in sv["spectral"])
            + ".",
            "",
            "| target rel L2 | cheapest PINN setting (median error over seeds [max], median training time) | FD (N, time) | Chebyshev (N, time) | PINN time / FD time |",
            "|---|---|---|---|---|",
        ]
        for row in met["equal_error"][prob]:
            p, fd, sp = row["pinn"], row["fd"], row["spectral"]
            ratio = f"{p['wall_s'] / fd['wall_s']:.3g}" if (p and fd) else "—"
            L.append(
                f"| {row['target']:g} | "
                + (
                    f"{p['setting']}, {p['error']:.2g} [{p.get('error_max', p['error']):.2g}], {p['wall_s']:.0f} s"
                    if p
                    else "not reached"
                )
                + " | "
                + (f"N {fd['N']}, {1e3 * fd['wall_s']:.3g} ms" if fd else "not reached")
                + " | "
                + (f"N {sp['N']}, {1e3 * sp['wall_s']:.3g} ms" if sp else "not reached")
                + f" | {ratio} |"
            )
        L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(met: dict, tag: str, odir, docs: bool) -> list[str]:
    F.style()
    paths = []
    paths += save_fig(
        F.scaling_grid(met["runs"], met["fits"], met["base"], LABEL), f"p6_1_scaling_{tag}", odir, docs=docs
    )
    paths += save_fig(
        F.cost_vs_error(
            met["runs"],
            met["solvers"],
            [t["target"] for t in next(iter(met["equal_error"].values()))],
            met.get("settings"),
        ),
        f"p6_1_cost_{tag}",
        odir,
        docs=docs,
    )
    return [str(p) for p in paths]


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
