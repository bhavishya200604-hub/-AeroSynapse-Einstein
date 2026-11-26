"""Phase 3: equal-budget adaptive-collocation study for transient conv-diff across Pe (multi-seed)."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from _common import RESULTS, parse_profile

from aerosynapse.experiments import ExperimentRecord, ExperimentRun
from aerosynapse.experiments import stats as ST
from aerosynapse.experiments.parallel import run_parallel
from aerosynapse.experiments.studies import phase3 as S

HYPOTHESES = (
    "Pre-registered before execution (2026-11-02). Primary metric: relative L2 error on a 201x201 space-time grid vs "
    "the exact solution. "
    "H3.1: PINN error depends non-monotonically on Pe (U-shape): larger at Pe=1 (stiff initial diffusion layer) and at "
    "Pe=1000 (sharp transported pulse) than at Pe=10. "
    "H3.2: at equal collocation budget and iterations, residual-based adaptive samplers (RAD, R3) reduce rel. L2 vs "
    "fixed uniform sampling at Pe>=100; tested with paired two-sided Wilcoxon signed-rank tests over seeds, Holm-"
    "corrected within each Pe (6 comparisons vs uniform), alpha=0.05. "
    "H3.3: at Pe<=10 no sampler differs significantly from uniform. "
    "H3.4 (ablation): Fourier features reduce error at Pe=1000; causal weighting reduces early-time error at Pe=1."
)


def main() -> None:
    a = parse_profile(__doc__ or "")
    P = S.PROFILES[a.profile]
    base = {"iters": P["iters"], "n": P["n"], "resample_every": P["resample_every"]}
    rec = ExperimentRecord(
        experiment_id="phase3_adaptive_collocation",
        research_question="Does residual-based adaptive collocation improve PINN accuracy for transient convection-"
        "diffusion across Péclet numbers under an equal compute budget?",
        hypothesis=HYPOTHESES,
        mathematical_formulation="u_t + u_x = Pe^-1 u_xx on periodic [0,1) x [0,1], u0 = Gaussian(x0=0.25, sigma0=0.1); "
        "exact solution by the method of images (physics.library.convdiff_exact)",
        seed=0,
        model_configuration={
            "arch": "MLP 3x32 tanh, periodic embedding of x (hard periodic BC), soft IC",
            "dtype": "float32",
        },
        training_configuration={
            "profile": a.profile,
            **base,
            "optimizer": "Adam lr 1e-3, decay 0.9 per 1000 it",
            "methods": S.METHODS,
            "pes": P["pes"],
            "seeds": P["seeds"],
        },
        evaluation_metrics=["rel_l2", "linf", "error quantiles", "early/late-time rel_l2", "wall_s", "residual_evals"],
        dataset_description="no data; exact reference solution (zero discretisation error up to image truncation "
        "< 1e-12)",
        reproducibility=f"python scripts/run_phase3.py --profile {a.profile}",
        limitations=[
            "single architecture and optimiser",
            "budget equalised by points and iterations, not wall "
            "time; adaptive samplers spend extra residual evaluations (recorded)",
            "float32",
        ],
    )
    with ExperimentRun(rec, RESULTS) as run:
        jobs = [dict(pe=pe, method=m, seed=s, **base) for pe in P["pes"] for m in S.METHODS for s in range(P["seeds"])]
        abl = [
            dict(pe=1000.0, method=m, seed=s, ff=16, ff_scale=1.0, ablation="fourier", **base)
            for m in ("uniform", "rad")
            for s in range(P["seeds"])
        ]
        abl += [
            dict(pe=pe, method="uniform", seed=s, causal_eps=1.0, ablation="causal", **base)
            for pe in (1.0, 1000.0)
            for s in range(P["seeds"])
        ]
        abl += [
            dict(pe=pe, method="uniform", seed=s, hard=True, ablation="hard_ic", **base)
            for pe in (1.0, 1000.0)
            for s in range(P["seeds"])
        ]
        rows = run_parallel(S.job, jobs + abl)
        main_rows = [r for r in rows if "ablation" not in r]
        abl_rows = [r for r in rows if "ablation" in r]
        E: dict[float, dict[str, list[float]]] = defaultdict(dict)
        for pe in P["pes"]:
            for m in S.METHODS:
                rs = sorted([r for r in main_rows if r["pe"] == pe and r["method"] == m], key=lambda r: r["seed"])
                E[pe][m] = [r["rel_l2"] for r in rs]
        table: dict[str, dict[str, object]] = {}
        tests: dict[str, object] = {}
        for pe in P["pes"]:
            u = E[pe]["uniform"]
            pv = {m: ST.paired_wilcoxon(E[pe][m], u) for m in S.METHODS if m != "uniform"}
            tests[str(pe)] = {
                "holm": ST.holm(pv),
                "friedman_p": ST.friedman(E[pe]),
                "log10_ratio_vs_uniform": {m: ST.effect_size_log_ratio(E[pe][m], u) for m in pv},
            }
            for m in S.METHODS:
                med, lo, hi = ST.bootstrap_ci(E[pe][m])
                ev = [r["residual_evals"] for r in main_rows if r["pe"] == pe and r["method"] == m]
                wt = [r["wall_s"] for r in main_rows if r["pe"] == pe and r["method"] == m]
                table[f"{pe}/{m}"] = {
                    "median": med,
                    "ci95": [lo, hi],
                    "mean": float(np.mean(E[pe][m])),
                    "std": float(np.std(E[pe][m], ddof=1)),
                    "min": float(np.min(E[pe][m])),
                    "max": float(np.max(E[pe][m])),
                    "residual_evals_mean": float(np.mean(ev)),
                    "wall_s_mean": float(np.mean(wt)),
                }
        ablation: dict[str, object] = {}
        for key in sorted({(r["ablation"], r["pe"], r["method"]) for r in abl_rows}):
            rs = sorted([r for r in abl_rows if (r["ablation"], r["pe"], r["method"]) == key], key=lambda r: r["seed"])
            vals = [r["rel_l2"] for r in rs]
            ref = E[key[1]][key[2]]
            early = [r["rel_l2_early_t"] for r in rs]
            ref_early = [
                r["rel_l2_early_t"]
                for r in sorted(
                    [r for r in main_rows if r["pe"] == key[1] and r["method"] == key[2]], key=lambda r: r["seed"]
                )
            ]
            ablation["/".join(map(str, key))] = {
                "median": float(np.median(vals)),
                "ci95": list(ST.bootstrap_ci(vals)[1:]),
                "baseline_median": float(np.median(ref)),
                "p_wilcoxon_vs_baseline": ST.paired_wilcoxon(vals, ref),
                "log10_ratio": ST.effect_size_log_ratio(vals, ref),
                "early_t_median": float(np.median(early)),
                "baseline_early_t_median": float(np.median(ref_early)),
                "p_early_vs_baseline": ST.paired_wilcoxon(early, ref_early),
            }
        run.record.results.update(
            {
                "table": table,
                "tests": tests,
                "ablation": ablation,
                "n_runs": len(rows),
                "seeds": P["seeds"],
                "rows": [{k: v for k, v in r.items() if k != "history"} for r in rows],
            }
        )
        (run.dir / "histories.json").write_text(
            json.dumps(
                [
                    {
                        "pe": r["pe"],
                        "method": r["method"],
                        "seed": r["seed"],
                        "ablation": r.get("ablation"),
                        "history": r["history"],
                    }
                    for r in rows
                ]
            )
        )
    for k, v in table.items():
        print(f"{k:24s} median {v['median']:.3e} CI [{v['ci95'][0]:.2e}, {v['ci95'][1]:.2e}]")
    print(json.dumps(tests, indent=1, default=str)[:3000])
    print(json.dumps(ablation, indent=1)[:2000])


if __name__ == "__main__":
    main()
