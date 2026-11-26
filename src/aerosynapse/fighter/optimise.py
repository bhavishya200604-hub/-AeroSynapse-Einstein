"""Optimise a fighter family: CMA-ES over the family's variables, every situation in the objective.

Objective (lower is better), each term relative to the reference design:
  0.30 take-off weight + 0.15 cruise drag + 0.15 dash drag + 0.15 / sustained turn rate
  + 0.10 / (climb specific excess power + 100 m/s) + 0.05 gust load + 0.05 turbulence load
  + 0.05 CVaR of drag/thrust over random extreme states
plus a quadratic penalty on every violated constraint (engine and cockpit fit, fuel volume, balance, field
length on normal and hot days, approach speed, dash thrust, sustained turn, directional stability at low and
high incidence, span, length, material temperatures). Every evaluation is appended to a data store on disk
(:mod:`aerosynapse.store`) for surrogate training.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerosynapse.fighter import design as D
from aerosynapse.fighter import performance as P
from aerosynapse.shapeopt.cmaes import cmaes

WEIGHTS = {"W0": 0.30, "cruise": 0.15, "dash": 0.15, "turn": 0.15, "climb": 0.10, "gust": 0.05, "turb": 0.05, "extreme": 0.05}


def objective(r: P.Result, ref: dict[str, float]) -> tuple[float, dict[str, float]]:
    m = r.metrics
    t = {
        "W0": r.W0 / ref["W0"],
        "cruise": m["cruise_drag_N"] / ref["cruise_drag_N"],
        "dash": m["dash_drag_N"] / ref["dash_drag_N"],
        "turn": ref["turn_rate_sustained_dps"] / max(m["turn_rate_sustained_dps"], 1.0),
        "climb": (ref["Ps_climb"] + 100) / max(m["Ps_climb"] + 100, 10.0),
        "gust": m["gust_dn"] / ref["gust_dn"],
        "turb": m["turbulence_rms_g"] / ref["turbulence_rms_g"],
        "extreme": m["extreme_cvar"] / ref["extreme_cvar"],
    }
    J = sum(WEIGHTS[k] * v for k, v in t.items())
    pen = sum(max(0.0, c) ** 2 + 0.5 * max(0.0, c) for c in r.constraints.values())
    return float(J + 5.0 * pen), t


def reference_terms(r: P.Result) -> dict[str, float]:
    return {"W0": r.W0, **{k: r.metrics[k] for k in ("cruise_drag_N", "dash_drag_N", "turn_rate_sustained_dps", "Ps_climb", "gust_dn", "turbulence_rms_g", "extreme_cvar")}}


def run(job: dict[str, Any]) -> dict[str, Any]:
    fam = job["family"]
    mission = P.Mission(calib=job["calib"])
    ref = job["reference"]
    store = None
    if job.get("store"):
        from aerosynapse.store.shards import ShardWriter

        store = ShardWriter(job["store"], f"{fam}_s{job['seed']}", ["z", "J", "feasible", "W0", "metrics", "constraints"])
    log: list[dict[str, Any]] = []
    t0 = time.time()

    def one(z: NDArray[np.float64]) -> float:
        try:
            r = P.evaluate(fam, D.decode(fam, z), mission)
            J, _ = objective(r, ref)
            if not np.isfinite(J) or 5 * ref["W0"] < r.W0:  # the weight iteration diverged: the design does not close
                J = 50.0
            if store is not None:
                store.add({"z": z.tolist(), "J": J, "feasible": r.feasible, "W0": r.W0, "metrics": r.metrics, "constraints": r.constraints, "variables": r.variables})
            log.append({"J": J, "feasible": r.feasible})
            return J
        except Exception as e:  # a variable set the kernel cannot build: a large value, never a crash
            log.append({"J": 50.0, "error": str(e)[:200]})
            return 50.0

    x0 = np.full(len(D.variables(fam)), 0.5)
    if job.get("start"):
        x0 = D.encode(fam, D.complete(fam, job["start"]))
    res = cmaes(lambda Z: np.array([one(z) for z in Z]), x0, sigma=job.get("sigma", 0.2), seed=job["seed"], budget=job["budget"], pop=job.get("pop"), restarts=job.get("restarts", 1), stall_gens=25)
    if store is not None:
        store.close()
    best = D.decode(fam, res.x)
    r = P.evaluate(fam, best, mission)
    J, terms = objective(r, ref)
    return {"family": fam, "seed": job["seed"], "J": J, "terms": terms, "feasible": r.feasible, "variables": best, "evaluations": res.evals, "seconds": time.time() - t0,
            "history": res.history, "failed_evaluations": sum(1 for e in log if "error" in e), "summary": P.summary(r)}
