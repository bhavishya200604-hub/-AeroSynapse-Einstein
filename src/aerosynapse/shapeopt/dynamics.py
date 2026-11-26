r"""Unsteady and aeroelastic instability models.

Flutter: two-degree-of-freedom typical section with Theodorsen's exact unsteady aerodynamics,
solved by the k method. With :math:`Z=(\omega_\alpha/\omega)^2(1+ig)`, the determinant

.. math:: \det\left(\begin{bmatrix}\mu&\mu x_\alpha\\\mu x_\alpha&\mu r^2\end{bmatrix}+A(k)
          -Z\begin{bmatrix}\mu\sigma^2&0\\0&\mu r^2\end{bmatrix}\right)=0

gives g(U); flutter is the first crossing of g = 0 from below. A state-space model with Jones'
approximation of the Wagner function gives an independent check (:func:`flutter_statespace`).

Divergence: :math:`q_D=K_\alpha/(e\,c\,c_{l\alpha})`.

Dynamic stall: first-order lag of the separation point (Goman-Khrabrov type) on a Kirchhoff lift
model, :math:`\tau\dot f+f=f_{st}(\alpha-\tau_2\dot\alpha)`. The net work done per cycle tells
whether the airfoil feeds energy into the pitch motion (stall flutter). The time constants are
typical literature magnitudes rather than fitted values, so compare designs with it and do not
read its absolute numbers as predictions.

Gust: Kussner step response and the Pratt-Truitt load factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import hankel2

FloatArr = NDArray[np.float64]


def theodorsen(k: FloatArr) -> NDArray[np.complex128]:
    """C(k) = H1(k) / (H1(k) + i H0(k)) with Hankel functions of the second kind."""
    h1, h0 = hankel2(1, k), hankel2(0, k)
    return np.asarray(h1 / (h1 + 1j * h0))


@dataclass
class Section2DOF:
    mu: float  # mass ratio m / (pi rho b^2)
    a: float  # elastic axis position, semi-chords aft of mid-chord
    x_alpha: float  # centre of gravity aft of the elastic axis, semi-chords
    r2: float  # radius of gyration squared, semi-chords squared
    sigma: float  # omega_h / omega_alpha


def flutter_k(p: Section2DOF, ks: FloatArr | None = None) -> tuple[float, float]:
    """Flutter speed U_f / (b w_alpha) and reduced frequency. Returns (inf, 0) if none is found."""
    ks = np.geomspace(2.0, 0.02, 160) if ks is None else ks
    C = theodorsen(ks)
    Lh = 1 - 2j * C / ks
    La = 0.5 - 1j * (1 + 2 * C) / ks - 2 * C / ks**2
    Mh = 0.5 + 0 * ks
    Ma = 3.0 / 8.0 - 1j / ks
    h = 0.5 + p.a
    A11, A12 = p.mu + Lh, p.mu * p.x_alpha + La - Lh * h
    A21 = p.mu * p.x_alpha + Mh - Lh * h
    A22 = p.mu * p.r2 + Ma - (La + Mh) * h + Lh * h**2
    d1, d2 = p.mu * p.sigma**2, p.mu * p.r2
    # (A11 - d1 Z)(A22 - d2 Z) - A12 A21 = 0
    qa, qb, qc = d1 * d2, -(A11 * d2 + A22 * d1), A11 * A22 - A12 * A21
    disc = np.sqrt(qb**2 - 4 * qa * qc)
    Z = np.stack([(-qb + disc) / (2 * qa), (-qb - disc) / (2 * qa)], axis=1)
    # keep the two roots on continuous branches as k decreases
    for i in range(1, len(ks)):
        if abs(Z[i, 0] - Z[i - 1, 0]) + abs(Z[i, 1] - Z[i - 1, 1]) > abs(Z[i, 1] - Z[i - 1, 0]) + abs(Z[i, 0] - Z[i - 1, 1]):
            Z[i] = Z[i, ::-1]
    best, kf = np.inf, 0.0
    for j in range(2):
        with np.errstate(invalid="ignore", divide="ignore"):
            g = Z[:, j].imag / Z[:, j].real
            U = 1.0 / (ks * np.sqrt(np.maximum(Z[:, j].real, 1e-12)))
        ok = Z[:, j].real > 0
        for i in range(len(ks) - 1):
            if ok[i] and ok[i + 1] and g[i] < 0 <= g[i + 1]:
                t = -g[i] / (g[i + 1] - g[i])
                u = U[i] + t * (U[i + 1] - U[i])
                if u < best:
                    best, kf = float(u), float(ks[i] + t * (ks[i + 1] - ks[i]))
    return best, kf


def flutter_statespace(p: Section2DOF, u_grid: FloatArr | None = None) -> float:
    """Flutter speed from eigenvalues of the time-domain model. Independent of the k method.

    Time is in units of 1/w_alpha. Lift deficiency uses the Jones two-lag approximation
    C(s) ~ 1 - 0.165 s/(s + 0.0455) - 0.335 s/(s + 0.3), with s the reduced Laplace variable.
    """
    u_grid = np.linspace(0.3, 8.0, 500) if u_grid is None else u_grid
    0.5 + p.a
    mu, xa, r2, sg, a = p.mu, p.x_alpha, p.r2, p.sigma, p.a

    def growth(U: float) -> float:
        # states: [xi, th, xi', th', l1, l2]; w = 3/4-chord downwash per unit U
        Ms = np.array([[mu + 1.0, mu * xa - a], [mu * xa - a, mu * r2 + 1.0 / 8.0 + a**2]])
        Cs = np.array([[0.0, U], [0.0, U * (0.5 - a)]])
        Ks = np.array([[mu * sg**2, 0.0], [0.0, mu * r2]])
        # circulatory lift L = 2 U (C w): w = xi' + U th + (1/2 - a) th'
        wv = np.array([1.0, 0.5 - a])  # multiplies (xi', th')
        wd = np.array([0.0, U])  # multiplies (xi, th)
        cl_vec = np.array([-1.0, 1.0 / 2.0 + a])  # lift enters +(-1) in xi eq, +(1/2+a) in th eq
        b1, b2 = 0.0455 * U, 0.3 * U
        A = np.zeros((6, 6))
        Mi = np.linalg.inv(Ms)
        A[0:2, 2:4] = np.eye(2)
        lift_d = 2 * U * np.outer(cl_vec, wd) * 0.5
        lift_v = 2 * U * np.outer(cl_vec, wv) * 0.5
        lag = 2 * U * cl_vec
        # C w = 0.5 w + 0.165 b1 l1 + 0.335 b2 l2 ; l_i' = -b_i l_i + w
        A[2:4, 0:2] = Mi @ (-Ks + lift_d)
        A[2:4, 2:4] = Mi @ (-Cs + lift_v)
        A[2:4, 4] = Mi @ (lag * 0.165 * b1)
        A[2:4, 5] = Mi @ (lag * 0.335 * b2)
        A[4, 0:2], A[4, 2:4], A[4, 4] = wd, wv, -b1
        A[5, 0:2], A[5, 2:4], A[5, 5] = wd, wv, -b2
        return float(np.linalg.eigvals(A).real.max())

    prev = growth(float(u_grid[0]))
    for i in range(1, len(u_grid)):
        cur = growth(float(u_grid[i]))
        if prev < 0 <= cur:
            t = -prev / (cur - prev)
            return float(u_grid[i - 1] + t * (u_grid[i] - u_grid[i - 1]))
        prev = cur
    return float("inf")


def kussner(s: FloatArr) -> FloatArr:
    """Lift build-up on entering a sharp-edged gust; s in semi-chords travelled (Sears-Sparks fit)."""
    return 1 - 0.5 * np.exp(-0.13 * s) - 0.5 * np.exp(-s)


def gust_load_factor(rho: float, V: float, cl_alpha_per_rad: float, wing_loading: float, chord: float, u_de: float = 15.24) -> float:
    """Pratt-Truitt gust load factor; wing_loading in N/m^2."""
    mu = 2 * wing_loading / (rho * chord * 9.81 * cl_alpha_per_rad)
    kg = 0.88 * mu / (5.3 + mu)
    return float(1 + kg * 1.225 * u_de * V * cl_alpha_per_rad / (2 * wing_loading))


def dynamic_stall(
    alpha_st: FloatArr,
    cl_st: FloatArr,
    a_mean: float,
    a_amp: float,
    k: float = 0.1,
    tau1: float = 6.0,
    tau2: float = 3.0,
    cycles: int = 4,
    steps: int = 720,
) -> dict[str, FloatArr | float]:
    """Pitch oscillation alpha = a_mean + a_amp sin(2 k s/c...) with s in semi-chords.

    ``alpha_st``/``cl_st`` is a static curve (ascending alpha). Returns the final loop, the peak
    lift, the loop area (hysteresis) and the net pitch-damping work; negative work per cycle means
    that the flow feeds energy into the motion.
    """
    sl = np.polyfit(alpha_st[(alpha_st > -2) & (alpha_st < 6)], cl_st[(alpha_st > -2) & (alpha_st < 6)], 1)
    a0 = -sl[1] / sl[0]
    pot = sl[0] * (alpha_st - a0)
    ratio = np.clip(cl_st / np.where(np.abs(pot) < 1e-3, 1e-3, pot), 0.0, 1.0)
    f_st = np.clip(2 * np.sqrt(ratio) - 1, 0.0, 1.0) ** 2
    ds = 2 * np.pi / k / steps
    s = np.arange(cycles * steps) * ds
    a = a_mean + a_amp * np.sin(k * s)
    ad = a_amp * k * np.cos(k * s)
    f = float(np.interp(a[0], alpha_st, f_st))
    F = np.empty_like(s)
    for i in range(len(s)):
        target = float(np.interp(a[i] - tau2 * ad[i], alpha_st, f_st))
        f += ds * (target - f) / tau1
        F[i] = f
    cl = sl[0] * (a - a0) * ((1 + np.sqrt(np.clip(F, 0, 1))) / 2) ** 2
    last = slice((cycles - 1) * steps, cycles * steps)
    A, L = a[last], cl[last]
    area = float(np.trapezoid(np.r_[L, L[0]], np.r_[A, A[0]]))
    return {"alpha": A, "cl": L, "cl_peak": float(L.max()), "loop_area": area, "cl_drop": float(L.max() - L.min())}
