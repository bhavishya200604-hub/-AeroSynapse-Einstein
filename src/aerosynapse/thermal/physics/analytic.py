"""Closed-form solutions used as ground truth.

Case A (convective scaling, x in [0,1]):  theta_t + c theta_x = (1/Pe) theta_xx
    * steady, theta(0)=0, theta(1)=1:  theta = (e^{Pe x} - 1)/(e^{Pe} - 1)  (overflow-safe form)
    * Gaussian pulse on the infinite line: theta = (s0/s) exp(-(x-x0-ct)^2/(2 s^2)), s^2 = s0^2+2t/Pe

Case B (diffusive/Fourier scaling, slab x in [0,1], symmetry at x=0, Robin at x=1):
    theta_t = theta_xx - m^2 theta + Q,   theta_x(0)=0,   -theta_x(1) = Bi theta(1),  theta(x,0)=theta0
    Bi = h L / k_solid (conduction in a solid slab).  Eigenvalues: lambda tan(lambda) = Bi.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

Array = NDArray[np.float64]


# --- Case A -----------------------------------------------------------------------------------


def steady_convdiff(x: ArrayLike, Pe: float) -> Array:
    """theta = (e^{Pe x}-1)/(e^{Pe}-1), evaluated without overflow for |Pe| up to ~1e300."""
    x = np.asarray(x, dtype=float)
    if Pe == 0.0:
        return x.copy()
    if Pe > 0:
        return np.exp(Pe * (x - 1.0)) * (-np.expm1(-Pe * x)) / (-np.expm1(-Pe))
    return np.expm1(Pe * x) / np.expm1(Pe)


def steady_convdiff_dx(x: ArrayLike, Pe: float) -> Array:
    x = np.asarray(x, dtype=float)
    if Pe == 0.0:
        return np.ones_like(x)
    if Pe > 0:
        return Pe * np.exp(Pe * (x - 1.0)) / (-np.expm1(-Pe))
    return Pe * np.exp(Pe * x) / np.expm1(Pe)


def gaussian_pulse(
    x: ArrayLike,
    t: float | ArrayLike,
    Pe: float,
    sigma0: float = 0.05,
    x0: float = 0.25,
    c: float = 1.0,
) -> Array:
    """Advected/diffused Gaussian on the infinite line (exact for theta_t + c theta_x = theta_xx/Pe)."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    s2 = sigma0**2 + 2.0 * t / Pe
    return np.sqrt(sigma0**2 / s2) * np.exp(-((x - x0 - c * t) ** 2) / (2.0 * s2))


# --- Case B -----------------------------------------------------------------------------------


def robin_eigenvalues(Bi: float, n: int) -> Array:
    """First n positive roots of lambda sin(lambda) - Bi cos(lambda) = 0 (i.e. lambda tan lambda = Bi)."""
    if Bi <= 0:
        raise ValueError("Bi must be > 0")
    f = lambda lam: lam * np.sin(lam) - Bi * np.cos(lam)  # noqa: E731
    roots = np.empty(n)
    for k in range(n):
        a, b = k * np.pi, k * np.pi + 0.5 * np.pi
        roots[k] = brentq(f, a, b, xtol=1e-15, rtol=1e-15, maxiter=200)
    return roots


def slab_steady(x: ArrayLike, Bi: float, Q: float = 0.0, m: float = 0.0) -> Array:
    """Steady theta'' - m^2 theta + Q = 0, theta'(0)=0, -theta'(1)=Bi theta(1)."""
    x = np.asarray(x, dtype=float)
    if Q == 0.0:
        return np.zeros_like(x)
    if m == 0.0:
        return 0.5 * Q * (1.0 - x**2) + Q / Bi
    A = -Bi * Q / m**2 / (m * np.sinh(m) + Bi * np.cosh(m))
    return Q / m**2 + A * np.cosh(m * x)


def _cos_projection_steady(lam: Array, Bi: float, Q: float, m: float) -> Array:
    """int_0^1 theta_ss(x) cos(lam x) dx in closed form."""
    I0 = np.sin(lam) / lam
    if Q == 0.0:
        return np.zeros_like(lam)
    if m == 0.0:
        Ix2 = np.sin(lam) / lam + 2 * np.cos(lam) / lam**2 - 2 * np.sin(lam) / lam**3
        return (0.5 * Q + Q / Bi) * I0 - 0.5 * Q * Ix2
    A = -Bi * Q / m**2 / (m * np.sinh(m) + Bi * np.cosh(m))
    Icosh = (m * np.sinh(m) * np.cos(lam) + lam * np.cosh(m) * np.sin(lam)) / (m**2 + lam**2)
    return Q / m**2 * I0 + A * Icosh


def slab_transient(
    x: ArrayLike,
    t: float,
    Bi: float,
    Q: float = 0.0,
    m: float = 0.0,
    theta0: float = 1.0,
    n_terms: int = 200,
) -> Array:
    """Eigen-series solution of Case B: theta = theta_ss + sum C_n e^{-(lam_n^2+m^2) t} cos(lam_n x).

    Truncation error ~ exp(-(n_terms*pi)^2 t); use t >= ~1e-3 with the default n_terms.
    """
    x = np.asarray(x, dtype=float)
    lam = robin_eigenvalues(Bi, n_terms)
    norm = 0.5 + np.sin(2 * lam) / (4 * lam)
    proj = theta0 * np.sin(lam) / lam - _cos_projection_steady(lam, Bi, Q, m)
    C = proj / norm
    decay = np.exp(-(lam**2 + m**2) * t)
    series = np.cos(np.outer(x.ravel(), lam)) @ (C * decay)
    return slab_steady(x, Bi, Q, m) + series.reshape(x.shape)
