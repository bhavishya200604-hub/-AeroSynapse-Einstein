r"""Robust and uncertainty-aware objectives by Monte Carlo (spec Section 8F).

* expectation :math:`E_\xi f(x,\xi)` with standard error;
* CVaR_β (mean of the worst (1-β) tail; Rockafellar–Uryasev sample estimator);
* chance constraint :math:`P(g(x,\xi)\le0)\ge1-\epsilon` with a Wilson lower bound;
* worst case over the sample set (a sampled, not certified, worst case).
Common random numbers (fixed ξ samples) are used so that designs are compared on the same draws.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]


def expectation(samples: FloatArr) -> tuple[float, float]:
    s = np.asarray(samples, float)
    return float(s.mean()), float(s.std(ddof=1) / np.sqrt(s.size))


def cvar(samples: FloatArr, beta: float = 0.9) -> float:
    s = np.sort(np.asarray(samples, float))
    var = np.quantile(s, beta)
    return float(var + np.mean(np.maximum(s - var, 0.0)) / (1 - beta))


def chance(g_samples: FloatArr, eps: float = 0.05, z: float = 1.96) -> dict[str, float | bool]:
    s = np.asarray(g_samples, float)
    n = s.size
    p = float(np.mean(s <= 0))
    den = 1 + z**2 / n
    lo = (p + z**2 / (2 * n) - z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / den
    return {"p_feasible": p, "p_lower95": float(lo), "satisfied": bool(lo >= 1 - eps)}


def worst_case(samples: FloatArr) -> float:
    return float(np.max(samples))
