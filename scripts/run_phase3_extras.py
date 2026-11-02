"""Phase 3 extras for figures: (a) PINN space-time fields + final collocation points at several Pe (Fig. C);
(b) collocation-budget sweep (Pe × budget × sampler, Fig. D)."""

from __future__ import annotations

import json

import numpy as np
from _common import RESULTS, parse_profile

from aerosynapse.experiments import ExperimentRecord, ExperimentRun
from aerosynapse.experiments.parallel import run_parallel
from aerosynapse.experiments.studies import phase3 as S


def field_job(j: dict) -> dict:
    from aerosynapse.ml.nets import MLP, set_seed
    from aerosynapse.ml.pinn import PINNConfig, predict, train_pinn
    from aerosynapse.ml.problems import ConvDiff1D

    set_seed(0)
    p = ConvDiff1D(pe=j["pe"], sigma0=S.SIGMA0, t_final=S.T_FINAL)
    m = MLP(2, 1, 32, 3, lo=p.lo, hi=p.hi, periodic_dims=(0,))
    r = train_pinn(
        p,
        m,
        PINNConfig(
            iterations=j["iters"],
            n_collocation=j["n"],
            sampler="rad",
            lr_decay=0.9,
            resample_every=250,
            log_every=j["iters"],
        ),
    )
    G = p.eval_grid(81)
    pred, ref = predict(p, m, G), p.exact(G)
    return {
        "pe": j["pe"],
        "x": np.linspace(0, 1, 81).tolist(),
        "t": np.linspace(0, S.T_FINAL, 81).tolist(),
        "pred": pred.reshape(81, 81).tolist(),
        "ref": ref.reshape(81, 81).tolist(),
        "points": r.points[:: max(1, len(r.points) // 600)].tolist(),
        "rel_l2": float(np.linalg.norm(pred - ref) / np.linalg.norm(ref)),
    }


def main() -> None:
    a = parse_profile(__doc__ or "")
    iters = {"quick": 4000, "standard": 10000, "full": 20000}[a.profile]
    seeds = {"quick": 3, "standard": 5, "full": 10}[a.profile]
    rec = ExperimentRecord(
        experiment_id="phase3_extras",
        research_question="How does PINN error depend jointly on Pe, collocation "
        "budget and sampler; what do the learned fields look like?",
        hypothesis="H3.5 error decreases with budget for both samplers; the RAD advantage (if any) is largest at the "
        "smallest budget and highest Pe.",
        mathematical_formulation="as phase3_adaptive_collocation",
        seed=0,
        training_configuration={"iters": iters, "budgets": [500, 1000, 2000], "seeds": seeds},
        reproducibility=f"python scripts/run_phase3_extras.py --profile {a.profile}",
        evaluation_metrics=["rel_l2"],
        limitations=["fewer seeds than the main study"],
    )
    with ExperimentRun(rec, RESULTS) as run:
        fields = run_parallel(field_job, [{"pe": pe, "iters": iters, "n": 2000} for pe in (1.0, 100.0, 1000.0)])
        jobs = [
            dict(pe=pe, method=m, seed=s, iters=iters, n=n, resample_every=250)
            for pe in (1.0, 10.0, 100.0, 1000.0)
            for n in (500, 1000, 2000)
            for m in ("uniform", "rad")
            for s in range(seeds)
        ]
        rows = run_parallel(S.job, jobs)
        sweep = {}
        for pe in (1.0, 10.0, 100.0, 1000.0):
            for n in (500, 1000, 2000):
                for m in ("uniform", "rad"):
                    v = [r["rel_l2"] for r in rows if r["pe"] == pe and r["n"] == n and r["method"] == m]
                    sweep[f"{pe}/{n}/{m}"] = {"median": float(np.median(v)), "values": v}
        run.record.results.update({"budget_sweep": sweep, "field_rel_l2": {str(f["pe"]): f["rel_l2"] for f in fields}})
        (run.dir / "fields.json").write_text(json.dumps(fields))
    print(json.dumps(sweep, indent=0)[:2500])


if __name__ == "__main__":
    main()
