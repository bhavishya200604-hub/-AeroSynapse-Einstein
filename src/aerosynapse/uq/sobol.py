r"""Variance-based global sensitivity: Saltelli (2010) first-order and Jansen total-effect estimators.

With A, B independent sample matrices and :math:`A_B^{(i)}` (column i from B):
:math:`V_i\approx\frac1N\sum f(B)(f(A_B^{(i)})-f(A))`,
:math:`V_{T_i}\approx\frac1{2N}\sum(f(A)-f(A_B^{(i)}))^2`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

FloatArr = NDArray[np.float64]


def sobol_indices(
    f: Callable[[FloatArr], FloatArr], lo: FloatArr, hi: FloatArr, n: int = 4096, seed: int = 0
) -> dict[str, FloatArr]:
    d = len(lo)
    S = qmc.Sobol(2 * d, scramble=True, seed=seed).random(n)
    A = lo + S[:, :d] * (hi - lo)
    B = lo + S[:, d:] * (hi - lo)
    fA, fB = f(A), f(B)
    var = np.var(np.concatenate([fA, fB]))
    S1, ST = np.zeros(d), np.zeros(d)
    for i in range(d):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        fABi = f(ABi)
        S1[i] = np.mean(fB * (fABi - fA)) / var
        ST[i] = 0.5 * np.mean((fA - fABi) ** 2) / var
    return {"S1": S1, "ST": ST, "n_evals": np.array(n * (d + 2))}


def ishigami(X: FloatArr, a: float = 7.0, b: float = 0.1) -> FloatArr:
    return np.asarray(np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0]))


def ishigami_analytic(a: float = 7.0, b: float = 0.1) -> dict[str, FloatArr]:
    """Closed-form indices for X_i ~ U(-π, π) (derived from the ANOVA decomposition)."""
    pi = np.pi
    V1 = 0.5 * (1 + b * pi**4 / 5) ** 2
    V2 = a**2 / 8
    V13 = b**2 * pi**8 * (1 / 18 - 1 / 50)
    V = V1 + V2 + V13
    return {"S1": np.array([V1, V2, 0.0]) / V, "ST": np.array([V1 + V13, V2, V13]) / V}
