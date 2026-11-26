"""Iterated global search over the whole design.

Each round re-draws the random flow states and restarts CMA-ES from the best design so far, so a
design only survives if it keeps working on states it has not been tuned to. Progress is measured
on a fixed hold-out set that the optimiser never sees, and the search stops when the hold-out
score stops improving. Several seeds run in parallel, and differential evolution runs on the same
budget as a control on the choice of algorithm.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution

from aerosynapse.shapeopt import wing as wg
from aerosynapse.shapeopt.cmaes import cmaes
from aerosynapse.shapeopt.objective import Problem

FloatArr = NDArray[np.float64]


def holdout_problem(W0: FloatArr, plan0: wg.Planform, n2: int = 150, n3: int = 40, model: str = "xlarge") -> Problem:
    return Problem(W0, plan0, n_random_2d=n2, n_random_3d=n3, seed=20240, model=model)


def _feasible(e: dict[str, FloatArr]) -> bool:
    return bool(np.all(e["g"] <= 1e-6))


def run_cma(cfg: dict[str, Any]) -> dict[str, Any]:
    """One seed: rounds of CMA-ES with re-drawn states. Runs in its own process."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    W0, plan0 = np.asarray(cfg["W0"]), wg.Planform(*cfg["plan0"])
    seed, t0 = int(cfg["seed"]), time.time()
    H = holdout_problem(W0, plan0, cfg["hold_n2"], cfg["hold_n3"])
    best = None
    z = None
    best_h = float("inf")
    best_feasible = False  # a feasible design beats any infeasible one, whatever their penalised scores
    trace: list[dict[str, Any]] = []
    used = 0
    stale = 0
    for it in range(cfg["max_rounds"]):
        P = Problem(W0, plan0, cfg["n2"], cfg["n3"], seed=1000 * (seed + 1) + it)
        if z is None:
            z = P.encode(W0, plan0)
        sigma = cfg["sigma"] * (0.6**it)
        r = cmaes(P, z, sigma=sigma, seed=seed * 100 + it, budget=cfg["budget_round"], pop=cfg["pop"], restarts=1, stall_gens=40)
        z = r.x
        used += r.evals
        e = H.evaluate(z)
        h = float(e["J"][0])
        trace.append(
            {
                "round": it,
                "evals": used,
                "train_J": float(r.f),
                "holdout_J": h,
                "holdout_J2": float(e["J2"][0]),
                "holdout_J3": float(e["J3"][0]),
                "feasible": _feasible(e),
                "history": r.history[:: max(1, len(r.history) // 60)],
                "time_s": time.time() - t0,
            }
        )
        print(f"[seed {seed}] round {it}: train {r.f:.4f} holdout {h:.4f} evals {used} t {time.time() - t0:.0f}s", flush=True)
        feas = _feasible(e)
        if h < best_h - cfg["tol"]:
            stale = 0
        else:
            stale += 1
        if (feas and not best_feasible) or (feas == best_feasible and h < best_h):
            best_h, best, best_feasible = h, z.copy(), feas
        if cfg.get("checkpoint") and best is not None:
            Path(cfg["checkpoint"]).write_text(json.dumps({"round": it, "z": best.tolist(), "holdout_J": best_h, "holdout_feasible": best_feasible, "evals": used}), encoding="utf-8")
        if stale >= cfg["patience"]:
            break
    return {"method": "cma-es", "seed": seed, "z": None if best is None else best.tolist(), "holdout_J": best_h, "holdout_feasible": best_feasible,
            "trace": trace, "evals": used}


def run_de(cfg: dict[str, Any]) -> dict[str, Any]:
    """Control: differential evolution on the same budget, from a random population."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    W0, plan0 = np.asarray(cfg["W0"]), wg.Planform(*cfg["plan0"])
    P = Problem(W0, plan0, cfg["n2"], cfg["n3"], seed=999)
    H = holdout_problem(W0, plan0, cfg["hold_n2"], cfg["hold_n3"])
    z0 = P.encode(W0, plan0)
    n = len(z0)
    rng = np.random.default_rng(int(cfg["seed"]))
    pop = 2 * n
    init = np.clip(z0 + 0.08 * rng.standard_normal((pop, n)), 0, 1)
    init[0] = z0
    count = [0]

    def f(Zt: FloatArr) -> FloatArr:
        Z = np.atleast_2d(Zt.T)
        count[0] += len(Z)
        return P(Z)

    total = cfg["budget_round"] * cfg["max_rounds"]
    res = differential_evolution(
        f,
        [(0.0, 1.0)] * n,
        init=init,
        maxiter=max(1, total // pop - 1),
        vectorized=True,
        updating="deferred",
        polish=False,
        tol=0,
        seed=int(cfg["seed"]),
    )
    e = H.evaluate(res.x)
    print(f"[DE] train {res.fun:.4f} holdout {float(e['J'][0]):.4f} evals {count[0]}", flush=True)
    return {"method": "differential-evolution", "seed": int(cfg["seed"]), "z": res.x.tolist(), "holdout_J": float(e["J"][0]), "holdout_feasible": _feasible(e),
            "trace": [], "evals": count[0]}


PROFILE_RANK = {"quick": 0, "standard": 1, "full": 2}


def run_feasible(r: dict[str, Any]) -> bool:
    """Whether a run's reported design was feasible on the search hold-out. Records written before the flag
    existed take it from the trace round that produced the reported score."""
    if "holdout_feasible" in r:
        return bool(r["holdout_feasible"])
    rounds = [t for t in r.get("trace", []) if abs(t["holdout_J"] - r["holdout_J"]) < 1e-12]
    return bool(rounds and rounds[-1].get("feasible"))


def pick_best(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """The CMA-ES run to report: feasible on the hold-out first, then the lowest hold-out score."""
    return min((r for r in runs if r["method"] == "cma-es"), key=lambda r: (not run_feasible(r), r["holdout_J"]))
