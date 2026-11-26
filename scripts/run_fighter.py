"""Fighter-class study: optimise every topology family against every flight situation, then compare.

    python scripts/run_fighter.py --profile standard

Steps: calibrate the empty mass on the reference design (public F-16-class data), evaluate it, run CMA-ES on
each family (separate processes; finished jobs are reused), re-evaluate the winners on a hold-out set of
random extreme states, write results/fighter/study.json plus a design file and STL per family. Every
evaluation also goes to the data store for surrogate training.
"""

from __future__ import annotations

import argparse
import json
import time

from _common import RESULTS
from _jobs import run_jobs

from aerosynapse.cad import kernel
from aerosynapse.fighter import design as D
from aerosynapse.fighter import optimise as O
from aerosynapse.fighter import performance as P
from aerosynapse.store.shards import store_root

PROFILES = {"quick": dict(budget=120, pop=10, restarts=0, seeds=[0]), "standard": dict(budget=420, pop=14, restarts=1, seeds=[0]), "full": dict(budget=1500, pop=16, restarts=2, seeds=[0, 1])}
FAMILIES = ["conventional", "canard_delta", "twin_tail_lerx", "tailless_blended", "freeform"]
# every family starts from the reference design with its own options switched on
STARTS = {
    "canard_delta": {"c_size": 0.10, "w_sweep": 52.0, "w_taper": 0.15, "w_area": 40.0, "w_ar": 2.4},
    "twin_tail_lerx": {"v_cant": 20.0, "s_size": 0.07},
    "tailless_blended": {"v_cant": 25.0, "s_size": 0.06, "b_count": 1.5, "n_depth": 0.03, "w_sweep": 50.0, "w_area": 42.0, "width": 1.1},
    "freeform": {"c_size": 0.05, "v_cant": 15.0, "b_count": 1.0, "n_depth": 0.02},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="standard", choices=list(PROFILES))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    prof = PROFILES[a.profile]
    out = RESULTS / "fighter"
    out.mkdir(parents=True, exist_ok=True)
    base_design = json.loads((out / "baseline_design.json").read_text(encoding="utf-8"))
    base = base_design["variables"]
    t0 = time.time()
    mission = P.calibrate("conventional", base, 8570.0)
    ref_r = P.evaluate("conventional", base, mission)
    ref = O.reference_terms(ref_r)
    print(f"reference: W0 {ref_r.W0:.0f} kg, empty {ref_r.empty:.0f} kg, CD0 {ref_r.CD0:.4f}, feasible {ref_r.feasible}", flush=True)
    work = out / f"jobs_{a.profile}"
    jobs = []
    for fam in FAMILIES:
        for seed in prof["seeds"]:
            start = {**base, **STARTS.get(fam, {})}
            jobs.append((f"{fam}_s{seed}", "fighter", {"family": fam, "seed": seed, "budget": prof["budget"], "pop": prof["pop"], "restarts": prof["restarts"], "calib": mission.calib,
                                                      "reference": ref, "store": str(store_root() / "fighter_evals"), "start": start}))
    res = run_jobs(jobs, work, a.workers)
    best: dict[str, dict] = {}
    for r in res:
        if r["family"] not in best or (r["feasible"], -r["J"]) > (best[r["family"]]["feasible"], -best[r["family"]]["J"]):
            best[r["family"]] = r
    J_ref, terms_ref = O.objective(ref_r, ref)
    hold = {}
    for fam, r in best.items():
        rr = P.evaluate(fam, r["variables"], mission, extreme_seed=2)
        hold[fam] = {"extreme_cvar": rr.metrics["extreme_cvar"], "extreme_flyable": rr.metrics["extreme_flyable"]}
        des = D.build(D.complete(fam, r["variables"]), f"optimised {fam}")
        (out / f"{fam}.design.json").write_text(json.dumps(des, indent=1), encoding="utf-8")
        kernel.mesh(des, out / f"{fam}.stl", h=0.05)
    rb = P.evaluate("conventional", base, mission, extreme_seed=2)
    hold["reference"] = {"extreme_cvar": rb.metrics["extreme_cvar"], "extreme_flyable": rb.metrics["extreme_flyable"]}
    kernel.mesh(base_design, out / "reference.stl", h=0.05)
    study = {"profile": a.profile, "seconds": time.time() - t0, "calibration": {"calib": mission.calib, "empty_target_kg": 8570.0}, "objective_weights": O.WEIGHTS,
             "reference": {"J": J_ref, "terms": terms_ref, "summary": P.summary(ref_r)}, "families": best, "holdout_extreme": hold,
             "situations": [s.__dict__ for s in P.SITUATIONS]}
    (out / "study.json").write_text(json.dumps(study, indent=1, default=float), encoding="utf-8")
    for fam, r in sorted(best.items(), key=lambda kv: kv[1]["J"]):
        s = r["summary"]
        print(f"{fam:18s} J {r['J']:.3f} feasible {r['feasible']} W0 {s['W0']:.0f} cruise D {s['metrics']['cruise_drag_N']:.0f} dash D {s['metrics']['dash_drag_N']:.0f} turn {s['metrics']['turn_rate_sustained_dps']:.2f}")
    print(f"reference          J {J_ref:.3f}")


if __name__ == "__main__":
    main()
