"""Stage 1: whole-design search (see aerosynapse.shapeopt.search). Usually run via run_shapeopt.py."""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from _common import RESULTS  # noqa: E402
from _jobs import run_jobs  # noqa: E402

from aerosynapse.shapeopt import geometry as G  # noqa: E402
from aerosynapse.shapeopt.search import PROFILE_RANK, pick_best, run_feasible  # noqa: E402

PROFILES = {
    "quick": dict(n2=16, n3=4, pop=12, sigma=0.08, budget_round=240, max_rounds=2, patience=1, tol=2e-3, hold_n2=40, hold_n3=10, seeds=2),
    "standard": dict(n2=36, n3=8, pop=16, sigma=0.10, budget_round=2400, max_rounds=6, patience=2, tol=1e-3, hold_n2=150, hold_n3=40, seeds=3),
    "full": dict(n2=48, n3=10, pop=20, sigma=0.12, budget_round=6000, max_rounds=10, patience=3, tol=5e-4, hold_n2=300, hold_n3=60, seeds=3),
}


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--shape", default="naca2412")
    a.add_argument("--profile", default="standard", choices=list(PROFILES))
    a.add_argument("--plan", default="16.2,7.5,1.0,0.0,1.0,0.0", help="area, AR, taper, tip twist, twist exponent, winglet")
    a.add_argument("--promote", action="store_true", help="make this search the one verified and reported even if a larger search exists")
    n = a.parse_args()
    cfg = dict(PROFILES[n.profile])
    xy = G.load_shape(n.shape)
    W0, rms = G.fit_cst(xy)
    if rms > 3e-3:
        raise SystemExit(f"{n.shape}: CST fit error {rms:.4f} chord. Not airfoil-like: use the topology stage first.")
    out = RESULTS / "shapeopt" / n.shape.replace(":", "_").replace("/", "_").replace("\\", "_")
    out.mkdir(parents=True, exist_ok=True)
    cfg.update(W0=W0.tolist(), plan0=[float(q) for q in n.plan.split(",")])
    t0 = time.time()
    work = out / "jobs" / n.profile
    jobs = [(f"cma_{s}", "cma", {**cfg, "seed": s, "checkpoint": str(work / f"cma_{s}.checkpoint.json")}) for s in range(cfg["seeds"])]
    jobs.append(("de_7", "de", {**cfg, "seed": 7}))
    res = run_jobs(jobs, work)
    best = pick_best(res)
    rec = json.dumps({"shape": n.shape, "profile": n.profile, "fit_rms": rms, "W0": cfg["W0"], "plan0": cfg["plan0"], "runs": res, "best_seed": best["seed"],
                      "best_feasible": run_feasible(best), "wall_s": time.time() - t0}, indent=1)
    (out / f"search_{n.profile}.json").write_text(rec)  # every profile keeps its own record
    # search.json is what verification and the report read: a smaller search never replaces a larger one
    old = json.loads((out / "search.json").read_text()) if (out / "search.json").exists() else None
    if old is None or PROFILE_RANK.get(old.get("profile", "quick"), 0) <= PROFILE_RANK[n.profile] or n.promote:
        (out / "search.json").write_text(rec)
    else:
        print(f"kept the {old['profile']} search as search.json; this {n.profile} search is in search_{n.profile}.json (--promote to replace it)")
    print("best holdout", best["holdout_J"], "feasible", run_feasible(best), "seed", best["seed"], "wall", round(time.time() - t0))
    print({r["method"] + str(r["seed"]): round(r["holdout_J"], 4) for r in res})


if __name__ == "__main__":
    main()
