"""Phase 5: optimisation engine and inverse problems (see aerosynapse.experiments.studies.phase5)."""

from __future__ import annotations

import json

from _common import RESULTS, parse_profile

from aerosynapse.experiments import ExperimentRecord, ExperimentRun
from aerosynapse.experiments.studies import phase5 as S


def main() -> None:
    a = parse_profile(__doc__ or "")
    seeds = {"quick": 3, "standard": 5, "full": 10}[a.profile]
    rec = ExperimentRecord(
        experiment_id="phase5_optimization",
        research_question="Does the optimisation engine find known optima, enforce constraints and budgets, and keep "
        "final candidates feasible under the trusted solver?",
        hypothesis="H5.1 L-BFGS/SQP/DE reach Rosenbrock f < 1e-4 within budget. H5.2 SQP recovers the QP KKT point "
        "(mu = 1). H5.3 BO beats DE/GA on Branin at 40 evaluations. H5.4 every final candidate is feasible under the "
        "trusted solver. H5.5 the robust optimum satisfies the 95% chance constraint where the deterministic optimum may "
        "not. H5.6 adjoint LSQ recovers alpha within 2 standard errors; the PINN estimate is less accurate.",
        mathematical_formulation="see aerosynapse.optim.demos and optim.inverse docstrings",
        seed=0,
        optimization_configuration={"profile": a.profile, "seeds": seeds},
        evaluation_metrics=[
            "best feasible objective",
            "gap to known optimum",
            "KKT residual",
            "hypervolume",
            "trusted re-evaluation",
            "chance-constraint probability",
            "alpha error",
        ],
        dataset_description="synthetic observations for the inverse problem (noise sigma = 0.01)",
        reproducibility=f"python scripts/run_phase5.py --profile {a.profile}",
        limitations=["illustrative material values", "1D thermal / conv-diff models", "cost model P ∝ m² assumed"],
    )
    with ExperimentRun(rec, RESULTS) as run:
        res = {
            "benchmarks": S.benchmarks(seeds),
            "protocol": S.protocol_runs(),
            "multiobjective": S.multiobjective(),
            "surrogate_vs_direct": S.surrogate_vs_direct(seeds),
            "robust": S.robust(),
            "inverse": S.inverse({"quick": 3000, "standard": 10000, "full": 20000}[a.profile]),
        }
        run.record.results.update(res)
    short = {k: v for k, v in res.items() if k != "protocol" and k != "multiobjective"}
    print(json.dumps(short, indent=1, default=str)[:5000])
    for k, v in res["protocol"].items():
        print(k, v["verdict"], {t: v[t].get("final") for t in ("method", "baseline_method")})
    mo = res["multiobjective"]
    print({k: mo[k] for k in mo if k not in ("beam_points", "beam_pareto")})


if __name__ == "__main__":
    main()
