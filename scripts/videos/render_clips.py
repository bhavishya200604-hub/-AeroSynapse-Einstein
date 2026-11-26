"""Render every clip in parallel worker processes."""

from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))


def run(spec: str) -> str:
    os.environ["OMP_NUM_THREADS"] = "1"
    mod, fn = spec.split(":")
    t0 = time.perf_counter()
    try:
        m = __import__(mod)
        out = getattr(m, fn)()
        return f"OK   {spec:28s} {time.perf_counter() - t0:6.1f}s {out}"
    except Exception:
        return f"FAIL {spec}\n{traceback.format_exc()}"


if __name__ == "__main__":
    specs = sys.argv[1:] or [
        "clips_sim:cylinder",
        "clips_ml:pinn_training",
        "clips_sim:cavity",
        "clips_sim:heat2d",
        "clips_sim:convdiff",
        "clips_ml:fno",
        "clips_ml:airfoil_morph",
        "clips_graphs:convergence",
        "clips_graphs:samplers",
        "clips_graphs:generalisation",
        "clips_graphs:performance",
        "clips_graphs:pareto",
        "clips_graphs:spacetime",
        "clips_graphs:trust",
        "clips_graphs:ghia",
    ]
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(run, specs):
            print(r, flush=True)
