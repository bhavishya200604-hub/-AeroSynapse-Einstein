r"""Resource estimator calibrated by micro-benchmarks on the running machine (spec Section 12).

Models (basis stated, not invented):

* PINN training time ≈ c_pinn · iterations · N_collocation · n_params  (c_pinn measured);
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


def calibrate(quick: bool = True) -> Calibration:
    import torch

    from aerosynapse.ml.nets import MLP, n_params
    from aerosynapse.ml.pinn import PINNConfig, train_pinn
    from aerosynapse.ml.problems import ConvDiff1D
    from aerosynapse.numerics.convdiff import solve_convdiff_fv

    torch.manual_seed(0)
    p = ConvDiff1D(pe=10.0)
    m = MLP(2, 1, 16, 2, lo=p.lo, hi=p.hi, periodic_dims=(0,))
    it, n = (60, 256) if quick else (200, 512)
    r = train_pinn(p, m, PINNConfig(iterations=it, n_collocation=n, log_every=10**9))
    c_pinn = r.wall_s / (it * n * n_params(m))
    t0 = time.perf_counter()
    res = solve_convdiff_fv(0, 1, 256, 1.0, 0.01, lambda x: np.sin(2 * np.pi * x), 0.2, scheme="vanleer")
    c_fv = (time.perf_counter() - t0) / (256 * res.steps)
    return Calibration(
        c_pinn, c_fv, f"PINN: {it} it x {n} pts x {n_params(m)} params; FV: 256 cells x {res.steps} steps"
    )


def estimate_pinn_seconds(cal: Calibration, iterations: int, n_collocation: int, n_params: int) -> float:
    return cal.c_pinn * iterations * n_collocation * n_params


def estimate_fv_seconds(cal: Calibration, n_cells: int, n_steps: int) -> float:
    return cal.c_fv * n_cells * n_steps


def array_memory_mb(*shape: int, bytes_per: int = 8) -> float:
    return float(np.prod(shape)) * bytes_per / 2**20
