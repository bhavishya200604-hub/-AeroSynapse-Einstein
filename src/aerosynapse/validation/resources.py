r"""Resource estimator calibrated by micro-benchmarks on the running machine (spec Section 12).

Models (basis stated, not invented):

* PINN training time ≈ iterations · (c0 + c_pinn · N_collocation · n_params), with the per-iteration overhead
  c0 and the slope c_pinn fitted from two measured calibration runs of different size (a single-point
  proportional model over-predicted by 6.6x in Phase 7 because it ignored the fixed overhead);
* FV explicit solver time ≈ c_fv · n_cells · n_steps (c_fv measured);
* memory of a float array = 8 bytes × elements (float64) — exact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Calibration:
    c_pinn: float
    c_fv: float
    basis: str
    c0_pinn: float = 0.0


def calibrate(quick: bool = True) -> Calibration:
    import torch

    from aerosynapse.ml.nets import MLP, n_params
    from aerosynapse.ml.pinn import PINNConfig, train_pinn
    from aerosynapse.ml.problems import ConvDiff1D
    from aerosynapse.numerics.convdiff import solve_convdiff_fv

    p = ConvDiff1D(pe=10.0)
    it = 60 if quick else 150
    pts: list[tuple[float, float]] = []
    for width, n in ((16, 256), (32, 1024)):
        torch.manual_seed(0)
        m = MLP(2, 1, width, 2 if width == 16 else 3, lo=p.lo, hi=p.hi, periodic_dims=(0,))
        r = train_pinn(p, m, PINNConfig(iterations=it, n_collocation=n, log_every=10**9))
        pts.append((n * n_params(m), r.wall_s / it))
    (w1, t1), (w2, t2) = pts
    c_pinn = (t2 - t1) / (w2 - w1)
    c0 = max(t1 - c_pinn * w1, 0.0)
    t0 = time.perf_counter()
    res = solve_convdiff_fv(0, 1, 256, 1.0, 0.01, lambda x: np.sin(2 * np.pi * x), 0.2, scheme="vanleer")
    c_fv = (time.perf_counter() - t0) / (256 * res.steps)
    basis = (
        f"PINN: affine fit from 2 runs ({it} it; work {w1:.3g} and {w2:.3g} pts*params); "
        f"FV: 256 cells x {res.steps} steps"
    )
    return Calibration(c_pinn, c_fv, basis, c0)


def estimate_pinn_seconds(cal: Calibration, iterations: int, n_collocation: int, n_params: int) -> float:
    return iterations * (cal.c0_pinn + cal.c_pinn * n_collocation * n_params)


def estimate_fv_seconds(cal: Calibration, n_cells: int, n_steps: int) -> float:
    return cal.c_fv * n_cells * n_steps


def array_memory_mb(*shape: int, bytes_per: int = 8) -> float:
    return float(np.prod(shape)) * bytes_per / 2**20
