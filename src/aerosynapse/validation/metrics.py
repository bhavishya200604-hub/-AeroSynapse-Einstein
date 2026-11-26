r"""Validation metrics (spec Section 10). All are computed from predictions and an independent
reference — never from a model's training loss."""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]


def rel_l2(pred: FloatArr, ref: FloatArr) -> float:
    return float(np.linalg.norm(pred - ref) / max(float(np.linalg.norm(ref)), 1e-300))


def abs_l2(pred: FloatArr, ref: FloatArr, weight: float = 1.0) -> float:
    """Discrete L2 norm sqrt(weight·Σ e²) (weight = cell volume for a quadrature-consistent norm)."""
    return float(np.sqrt(weight * np.sum((pred - ref) ** 2)))


def linf(pred: FloatArr, ref: FloatArr) -> float:
    return float(np.max(np.abs(pred - ref)))


def mae(pred: FloatArr, ref: FloatArr) -> float:
    return float(np.mean(np.abs(pred - ref)))


def rmse(pred: FloatArr, ref: FloatArr) -> float:
    return float(np.sqrt(np.mean((pred - ref) ** 2)))


def field_metrics(pred: FloatArr, ref: FloatArr) -> dict[str, float]:
    p, r = np.asarray(pred, float).ravel(), np.asarray(ref, float).ravel()
    return {"rel_l2": rel_l2(p, r), "abs_l2_rms": rmse(p, r), "linf": linf(p, r), "mae": mae(p, r), "rmse": rmse(p, r)}


def error_distribution(
    pred: FloatArr, ref: FloatArr, qs: tuple[float, ...] = (0.5, 0.9, 0.99, 1.0)
) -> dict[str, float]:
    """Quantiles of the pointwise absolute error (q = 1.0 is the worst case)."""
    e = np.abs(np.asarray(pred, float).ravel() - np.asarray(ref, float).ravel())
    return {f"q{int(q * 100)}": float(np.quantile(e, q)) for q in qs}


def conservation_error(masses: list[float] | FloatArr) -> float:
    """max |M(t) - M(0)| / |M(0)| for a conserved integral M."""
    m = np.asarray(masses, float)
    return float(np.max(np.abs(m - m[0])) / max(abs(float(m[0])), 1e-300))


def bc_error(values: FloatArr, target: FloatArr) -> float:
    return linf(np.asarray(values, float), np.asarray(target, float))


def energy_drift(E: list[float] | FloatArr) -> float:
    e = np.asarray(E, float)
    return float((e.max() - e.min()) / max(abs(float(e[0])), 1e-300))


def speedup(t_reference: float, t_model: float) -> float:
    return float(t_reference / max(t_model, 1e-12))


def break_even_queries(train_cost_s: float, data_cost_s: float, t_solver: float, t_model: float) -> float:
    r"""Number of queries after which a surrogate is cheaper overall:
    :math:`N^* = (C_{train} + C_{data}) / (t_{solver} - t_{model})` (inf if the model is not faster)."""
    gain = t_solver - t_model
    return float("inf") if gain <= 0 else float((train_cost_s + data_cost_s) / gain)


def measure(fn: Callable[[], Any], repeats: int = 3) -> dict[str, float]:
    """Median wall time and peak Python-heap memory (tracemalloc; excludes native torch buffers)."""
    times = []
    tracemalloc.start()
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"time_s_median": float(np.median(times)), "peak_mem_mb": peak / 2**20}
