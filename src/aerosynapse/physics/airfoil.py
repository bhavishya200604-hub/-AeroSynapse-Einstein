r"""Airfoil geometry: NACA 4-digit series and the CST (class/shape transformation) parameterisation.

NACA 4-digit thickness :math:`y_t=5t(0.2969\sqrt x-0.1260x-0.3516x^2+0.2843x^3-a_4x^4)` with
a_4 = 0.1015 (open trailing edge) or 0.1036 (closed); mean camber line with maximum camber m at p.
CST (Kulfan): :math:`y=x^{0.5}(1-x)\sum_{i=0}^{n}A_iK_ix^i(1-x)^{n-i}+x\,\Delta z_{te}`,
:math:`K_i=\binom ni` (Bernstein basis). Coordinates are returned in the panel-method ordering:
trailing edge → lower surface → leading edge → upper surface → trailing edge (clockwise).
"""

from __future__ import annotations

from math import comb

import numpy as np
from numpy.typing import NDArray

from aerosynapse.exceptions import ConfigurationError

FloatArr = NDArray[np.float64]


def cosine_spacing(n: int) -> FloatArr:
    return np.asarray(0.5 * (1 - np.cos(np.linspace(0, np.pi, n))))


def naca4(code: str, n: int = 81, closed_te: bool = True) -> tuple[FloatArr, FloatArr]:
    if len(code) != 4 or not code.isdigit():
        raise ConfigurationError("NACA 4-digit code must have 4 digits, e.g. '2412'")
    return naca4_params(int(code[0]), int(code[1]), int(code[2:]), n, closed_te)


def naca4_params(
    m_pct: float, p_tenths: float, t_pct: float, n: int = 81, closed_te: bool = True
) -> tuple[FloatArr, FloatArr]:
    """NACA 4-digit family with real-valued parameters (as used in AirfRANS file names)."""
    m, p, t = m_pct / 100, p_tenths / 10, t_pct / 100
    x = cosine_spacing(n)
    a4 = 0.1036 if closed_te else 0.1015
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - a4 * x**4)
    if m > 0 and p > 0:
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    return _close(xl, yl, xu, yu)


def cst(a_upper: FloatArr, a_lower: FloatArr, n: int = 81, dz_te: float = 0.0) -> tuple[FloatArr, FloatArr]:
    x = cosine_spacing(n)

    def surf(A: FloatArr) -> FloatArr:
        k = len(A) - 1
        S = sum(A[i] * comb(k, i) * x**i * (1 - x) ** (k - i) for i in range(k + 1))
        return np.asarray(np.sqrt(x) * (1 - x) * S)

    yu = surf(np.asarray(a_upper)) + x * dz_te / 2
    yl = surf(np.asarray(a_lower)) - x * dz_te / 2
    return _close(x, yl, x, yu)


def _close(xl: FloatArr, yl: FloatArr, xu: FloatArr, yu: FloatArr) -> tuple[FloatArr, FloatArr]:
    X = np.concatenate([xl[::-1], xu[1:]])
    Y = np.concatenate([yl[::-1], yu[1:]])
    return X, Y


def max_thickness(X: FloatArr, Y: FloatArr) -> float:
    n = (len(X) + 1) // 2
    xl, yl = X[:n][::-1], Y[:n][::-1]
    xu, yu = X[n - 1 :], Y[n - 1 :]
    xs = np.linspace(0, 1, 501)
    return float(np.max(np.interp(xs, xu, yu) - np.interp(xs, xl, yl)))
