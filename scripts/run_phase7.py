"""Phase 7: validation engine — trust/fallback, trust reports, resource estimator, notebooks."""

from __future__ import annotations

import json

from _common import RESULTS, ROOT, parse_profile

from aerosynapse.experiments import ExperimentRecord, ExperimentRun
from aerosynapse.experiments.notebooks import write_notebooks
from aerosynapse.experiments.studies import phase7 as S


def main() -> None:
    a = parse_profile(__doc__ or "")
    rec = ExperimentRecord(
        experiment_id="phase7_validation_engine",
        research_question="Can the platform decide automatically when to trust a surrogate and when to fall back to the "
        "classical solver?",
        hypothesis="H7.1 fallback rate <= 5% in-distribution and >= 80% on OOD-Pe queries. H7.2 the guarded system's "
        "worst-case error on OOD queries is lower than the raw surrogate's. H7.3 the resource estimator is within a "
        "factor of 2 of measured run times.",
        mathematical_formulation="thresholds at the 99th percentile of a calibration split (Mahalanobis, ensemble "
        "spread, mass residual) plus a hard parameter-range check",
        seed=0,
        evaluation_metrics=["fallback rate", "guarded vs raw error", "estimator relative error"],
        dataset_description="synthetic conv-diff operator data (FV solver)",
        reproducibility=f"python scripts/run_phase7.py --profile {a.profile}",
        limitations=["one operator family for the fallback study", "thresholds depend on calibration split size"],
    )
    with ExperimentRun(rec, RESULTS) as run:
        fb = S.trust_fallback(run.dir, {"quick": 100, "standard": 300, "full": 600}[a.profile])
        res = {
            "fallback": fb,
            "resources": S.resource_estimator(),
            "notebooks": [str(p.relative_to(ROOT)) for p in write_notebooks(ROOT / "notebooks")],
        }
        res["trust_reports"] = sorted(S.trust_reports(run.dir, RESULTS, fb))
        run.record.results.update(res)
    print(json.dumps(res, indent=1, default=str)[:5000])


if __name__ == "__main__":
    main()
