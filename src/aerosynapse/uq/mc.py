"""Monte Carlo uncertainty propagation with standard errors (parameter uncertainty → output)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

FloatArr = NDArray[np.float64]


def propagate(
    f: Callable[[FloatArr], float], sampler: Callable[[np.random.Generator, int], FloatArr], n: int, seed: int = 0
) -> dict[str, float | FloatArr]:
    rng = np.random.default_rng(seed)
    xs = sampler(rng, n)
    ys = np.array([f(x) for x in xs])
    return {
        "mean": float(ys.mean()),
        "std": float(ys.std(ddof=1)),
        "se_mean": float(ys.std(ddof=1) / np.sqrt(n)),
        "q05": float(np.quantile(ys, 0.05)),
        "q95": float(np.quantile(ys, 0.95)),
        "samples": ys,
    }


def lhs_uniform(lo: FloatArr, hi: FloatArr, n: int, seed: int = 0) -> FloatArr:
    return np.asarray(lo + qmc.LatinHypercube(d=len(lo), seed=seed).random(n) * (np.asarray(hi) - np.asarray(lo)))
