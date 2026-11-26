"""Thermally developing / fully developed laminar flow between parallel plates (Graetz problem).

Channel y in [0, 1] (H = 1), fully developed velocity u = 6 y (1 - y) (mean 1), D = 1/Pe_H, D_h = 2H = 2.
Nu_Dh = h D_h / k = 2 (dtheta/dn_in at the wall) / (theta_b - theta_w),
theta_b = int u theta / int u (mixing-cup bulk temperature).

* CHF (uniform flux on both walls): theta = gamma x + f(y) is exact for any Pe (theta_xx = 0); Nu = 140/17,
  derived symbolically in :func:`nu_chf_symbolic`.
* CWT (walls at theta_w = 0): fully developed theta = exp(-lambda x) f(y) with
      D f'' + lambda u f + D lambda^2 f = 0,  f(0) = f(1) = 0
  (the lambda^2 term is axial conduction). Solved with Chebyshev collocation as a (quadratic) eigenvalue
  problem in :func:`nu_cwt_spectral`; without axial conduction Nu = lambda / D is Pe-independent.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import sympy as sp

from aerosynapse.thermal.numerics.spectral import cheb_mapped, clencurt


def nu_chf_symbolic() -> sp.Rational:
    """Exact fully developed CHF Nusselt number (both walls, D_h = 2H)."""
    y, g = sp.symbols("y gamma", positive=True)
    u = 6 * y * (1 - y)
    # D f'' = gamma u  (D scales out); symmetric about y = 1/2 -> f'(1/2) = 0
    fp = sp.integrate(g * u, (y, sp.Rational(1, 2), y))  # f'(y) with f'(1/2) = 0
    f = sp.integrate(fp, (y, 0, y))  # f(0) = 0 (wall temperature reference)
    fb = sp.integrate(u * f, (y, 0, 1)) / sp.integrate(u, (y, 0, 1))
    dfdn_in = fp.subs(y, 0)  # inward normal at y = 0 is +y
    return sp.Abs(sp.simplify(2 * dfdn_in / (fb - 0)))  # Nu = 2 |dtheta/dn| / |theta_b - theta_w|


def _operators(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    D1, y = cheb_mapped(N, 0.0, 1.0)
    D2 = D1 @ D1
    u = 6 * y * (1 - y)
    return D1, D2, y, u


def nu_cwt_spectral(N: int = 64, Pe: float | None = None) -> dict[str, float]:
    """Fully developed CWT Nusselt number by Chebyshev collocation.

    Pe = None: no axial conduction (classical Graetz limit). Otherwise Pe = Pe_H = u_mean H / alpha.
    Returns lambda (decay rate in x/H), Nu and the eigenfunction residual.
    """
    D1, D2, y, u = _operators(N)
    w = clencurt(N, 0.0, 1.0)
    I = np.arange(1, N)  # interior nodes (Dirichlet f = 0 at both walls)
    A2 = D2[np.ix_(I, I)]
    U = np.diag(u[I])
    if Pe is None:
        # f'' + mu u f = 0  ->  -A2 f = mu U f ; Nu = mu (see module docstring)
        vals, vecs = sla.eig(-A2, U)
        vals = np.real_if_close(vals)
        real = np.isreal(vals) & (np.real(vals) > 0)
        k = np.argmin(np.where(real, np.real(vals), np.inf))
        mu = float(np.real(vals[k]))
        lam = mu  # with D = 1 scaling; Nu independent of Pe
        f_int = np.real(vecs[:, k])
        D = 1.0
    else:
        D = 1.0 / Pe
        n = I.size
        # D A2 f + lam U f + D lam^2 f = 0 ; linearise with g = lam f:
        # [0 I; -D A2 -U] [f; g] = lam [I 0; 0 D I] [f; g]
        Z, Id = np.zeros((n, n)), np.eye(n)
        L = np.block([[Z, Id], [-D * A2, -U]])
        M = np.block([[Id, Z], [Z, D * Id]])
        vals, vecs = sla.eig(L, M)
        ok = np.isfinite(vals) & (np.abs(np.imag(vals)) < 1e-8) & (np.real(vals) > 0)
        k = np.argmin(np.where(ok, np.real(vals), np.inf))
        lam = float(np.real(vals[k]))
        f_int = np.real(vecs[:n, k])
    f = np.zeros(N + 1)
    f[I] = f_int
    if w @ (u * f) < 0:
        f = -f
    fb = (w @ (u * f)) / (w @ u)
    dfdn_in = (D1 @ f)[0]  # y = 0 is node 0 (ascending mapping), inward normal +y
    nu = 2 * dfdn_in / fb
    res = D * (D2 @ f) + lam * u * f + (D * lam**2 * f if Pe is not None else 0 * f)
    if Pe is None:
        res = D2 @ f + lam * u * f
    return {
        "lambda": lam,
        "Nu": float(nu),
        "eig_residual": float(np.max(np.abs(res[I]))),
        "Nu_energy_balance": float(
            lam / D * (w @ (u * f) + (D * lam * (w @ f) if Pe is not None else 0.0)) / (w @ (u * f))
        )
        if Pe is not None
        else float(lam),
    }


def entry_series(xstar: np.ndarray, n_modes: int = 60, N: int = 200) -> dict[str, np.ndarray]:
    """Thermal entry (Graetz) problem between parallel plates, constant wall temperature, NO axial conduction.

    theta~ = (theta - theta_w) / (theta_in - theta_w) = sum_n c_n f_n(y) exp(-4 mu_n x*), x* = x / (D_h Re_Dh Pr)
    = x / (4 Pe_H) (H = 1), with f_n'' + mu_n u f_n = 0, f_n(0) = f_n(1) = 0 (Chebyshev collocation, N nodes) and
    c_n = int u f_n / int u f_n^2 (u-weighted orthogonality). Returns local Nu_Dh(x*) = 2 f'(0)-sum / theta~_b, the mean
    Nu_m(x*) = ln(1 / theta~_b) / (4 x*) (energy balance) and theta~_b. Accurate where n_modes resolve the entry
    region (x* >~ 1e-4 for 60 modes); the short-entry (Leveque) asymptote takes over below.
    """
    D1, D2, y, u = _operators(N)
    w = clencurt(N, 0.0, 1.0)
    I = np.arange(1, N)
    vals, vecs = sla.eig(-D2[np.ix_(I, I)], np.diag(u[I]))
    ok = np.isfinite(vals) & (np.abs(np.imag(vals)) < 1e-6 * np.abs(vals)) & (np.real(vals) > 0)
    order = np.argsort(np.real(vals[ok]))[:n_modes]
    mu = np.real(vals[ok])[order]
    F = np.zeros((N + 1, mu.size))
    F[I, :] = np.real(vecs[:, ok][:, order])
    c = (w @ (u[:, None] * F)) / (w @ (u[:, None] * F**2))
    fb = (w @ (u[:, None] * F)) / (w @ u)  # bulk of each mode
    df0 = (D1 @ F)[0]  # inward derivative at y = 0
    xs = np.atleast_1d(np.asarray(xstar, float))
    E = np.exp(-4.0 * np.outer(xs, mu))
    tb = E @ (c * fb)
    q = E @ (c * df0)
    return {"xstar": xs, "Nu_local": 2.0 * q / tb, "Nu_mean": np.log(1.0 / tb) / (4.0 * xs), "theta_b": tb, "mu": mu}


NU_FD_CWT = 7.541  # fully developed, isothermal walls, D_h = 2H (Lienhard 2025; Shah & London 1978) — LITERATURE.md


def leveque_local(xstar) -> np.ndarray:
    """Short-entry asymptote, local Nu_Dh = 0.4273 (f Re / x*)^(1/3) with f Re = 24 (parallel plates)
    (Muzychka & Yovanovich 2004, Eq. for UWT; f Re from Shah & London) = 1.2325 x*^(-1/3)."""
    return 0.4273 * (24.0 / np.asarray(xstar, float)) ** (1.0 / 3.0)


def leveque_mean(xstar) -> np.ndarray:
    """Mean over [0, x*]: 3/2 of the local value (Muzychka & Yovanovich 2004)."""
    return 1.5 * leveque_local(xstar)
