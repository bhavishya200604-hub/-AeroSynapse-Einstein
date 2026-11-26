"""Phase 2: PINN benchmarks vs exact solutions, supervised baseline, BC strategies, loss imbalance."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from _common import RESULTS, parse_profile

from aerosynapse.experiments import ExperimentRecord, ExperimentRun
from aerosynapse.experiments.parallel import run_parallel
from aerosynapse.experiments.studies import phase2 as S


def agg(rows: list[dict], key: tuple[str, ...]) -> dict[str, dict[str, float]]:
    g: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        g["/".join(str(r.get(k)) for k in key)].append(r["rel_l2"])
    return {
        k: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "n": len(v),
        }
        for k, v in g.items()
    }


def main() -> None:
    a = parse_profile(__doc__ or "")
    P = S.PROFILES[a.profile]
    seeds = list(range(P["seeds"]))
    base = {"iters": P["iters"], "n": P["n"]}
    rec = ExperimentRecord(
        experiment_id="phase2_pinn_benchmarks",
        research_question="How accurate are PINNs on heat, Poisson, Burgers, low-Pe conv-diff and a boundary layer, "
        "compared with exact solutions, a supervised network and classical solvers?",
        hypothesis="H2.1 hard BCs reduce rel. L2 vs penalty for heat and Poisson. H2.2 the supervised net with exact "
        "labels beats the PINN in rel. L2 at equal iterations. H2.3 the classical solvers are more accurate and orders "
        "of magnitude faster on these 1D/2D problems.",
        mathematical_formulation="see aerosynapse.ml.problems; loss = mean r^2 + sum constraint MSE",
        seed=0,
        model_configuration={"arch": "MLP 3x32 tanh", "dtype": "float32"},
        training_configuration={"profile": a.profile, **base, "optimizer": "Adam lr=1e-3, decay 0.9/1000 it"},
        evaluation_metrics=["rel_l2", "linf", "error quantiles", "wall_s"],
        dataset_description="collocation points sampled uniformly; supervised baseline uses exact labels at the same "
        "number of points",
        reproducibility=f"python scripts/run_phase2.py --profile {a.profile}",
        limitations=["small networks / budgets (CPU)", "float32 training"],
    )
    with ExperimentRun(rec, RESULTS) as run:
        jobs = [
            dict(problem=p, hard=h, seed=s, **base)
            for s in seeds
            for p, h in [
                ("heat1d", False),
                ("heat1d", True),
                ("poisson2d", False),
                ("poisson2d", True),
                ("burgers1d", False),
                ("convdiff_pe2", False),
                ("convdiff_pe10", False),
                ("bl_pe20", False),
                ("bl_pe20", True),
                ("disk", False),
                ("disk", True),
            ]
        ]
        jobs += [
            dict(problem="poisson2d", hard=False, seed=s, weights="annealing", imbalance=True, **base) for s in seeds
        ]
        jobs[2]["imbalance"] = True
        rows = run_parallel(S.pinn_job, jobs)
        sup = run_parallel(
            S.supervised_job,
            [
                dict(problem=p, seed=s, **base)
                for s in seeds
                for p in ("heat1d", "poisson2d", "burgers1d", "convdiff_pe10")
            ],
        )
        dd = run_parallel(S.dd_job, [dict(seed=s, **base) for s in seeds])
        res = {
            "pinn": agg(rows, ("problem", "hard", "weights")),
            "pinn_annealing_poisson2d": agg([r for r in rows if r.get("weights")], ("problem",)),
            "pinn_penalty_poisson2d_noanneal": agg(
                [r for r in rows if r["problem"] == "poisson2d" and not r["hard"] and not r.get("weights")],
                ("problem",),
            ),
            "supervised": agg(sup, ("problem",)),
            "domain_decomposition_bl_pe20": agg([{**d, "x": "dd"} for d in dd], ("x",)),
            "dd_interface_jumps": [{"u": d["interface_jump_u"], "flux": d["interface_jump_flux"]} for d in dd],
            "imbalance": [
                {
                    "seed": r["seed"],
                    "weights": r.get("weights"),
                    "init": r["imbalance_init"],
                    "final": r["imbalance_final"],
                }
                for r in rows
                if "imbalance_init" in r
            ],
            "worst_case_q100": {
                f"{r['problem']}/{r['hard']}/{r['seed']}": r["q100"] for r in rows if not r.get("weights")
            },
            "pinn_wall_s_mean": float(np.mean([r["wall_s"] for r in rows])),
            "classical": S.classical_reference_timing(),
            "rows": rows,
            "supervised_rows": sup,
        }
        run.record.results.update(res)
    for k in ("pinn", "pinn_annealing_poisson2d", "supervised", "domain_decomposition_bl_pe20"):
        for kk, v in res[k].items():
            print(f"{k:28s} {kk:24s} rel_l2 = {v['mean']:.3e} ± {v['std']:.1e} (n={v['n']})")
    print("imbalance", res["imbalance"][:2])
    print("classical", res["classical"], "pinn wall mean", res["pinn_wall_s_mean"])


if __name__ == "__main__":
    main()
