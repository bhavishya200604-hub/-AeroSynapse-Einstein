r"""Hess–Smith panel method: constant-strength sources per panel + one uniform vortex strength,
flow tangency at panel mid-points, Kutta condition (equal and opposite tangential velocity on the
two trailing-edge panels). 2D, inviscid, incompressible potential flow.

Limitations (stated, not hidden): no viscosity → zero profile drag (d'Alembert) and no stall /
separation; compressibility only via the Prandtl–Glauert factor 1/√(1-M²) for M < 0.7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aerosynapse.physics.regime import mach_guard
from aerosynapse.registry import SOLVERS

FloatArr = NDArray[np.float64]


@dataclass
class PanelResult:
    xc: FloatArr
    yc: FloatArr
    cp: FloatArr
    vt: FloatArr
    cl: float
    cl_pressure: float
    cm_c4: float
    gamma: float
    length: FloatArr


def _geometry(X: FloatArr, Y: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr, FloatArr]:
    xc, yc = 0.5 * (X[:-1] + X[1:]), 0.5 * (Y[:-1] + Y[1:])
    th = np.arctan2(np.diff(Y), np.diff(X))
    return xc, yc, th, np.hypot(np.diff(X), np.diff(Y))


def _influence(X: FloatArr, Y: FloatArr, xc: FloatArr, yc: FloatArr, th: FloatArr) -> tuple[FloatArr, ...]:
    dx0 = X[None, :-1] - xc[:, None]
    dy0 = Y[None, :-1] - yc[:, None]
    dx1 = X[None, 1:] - xc[:, None]
    dy1 = Y[None, 1:] - yc[:, None]
    r0, r1 = np.hypot(dx0, dy0), np.hypot(dx1, dy1)
    beta = np.arctan2(dx0 * dy1 - dy0 * dx1, dx0 * dx1 + dy0 * dy1)
    np.fill_diagonal(beta, np.pi)
    lr = np.log(r1 / r0)
    np.fill_diagonal(lr, 0.0)
    d = th[:, None] - th[None, :]
    return np.sin(d), np.cos(d), lr, beta


@SOLVERS.register("panel_hess_smith")
def solve_panel(X: FloatArr, Y: FloatArr, alpha_deg: float, mach: float = 0.0, lifting: bool = True) -> PanelResult:
    """Solve for unit free stream at angle ``alpha_deg``; chord assumed 1 (x from 0 to 1)."""
    mach_guard(mach)
    a = np.deg2rad(alpha_deg)
    xc, yc, th, L = _geometry(X, Y)
    n = len(xc)
    s, c, lr, beta = _influence(X, Y, xc, yc, th)
    An = (s * lr + c * beta) / (2 * np.pi)  # normal velocity from unit sources
    Bn = (c * lr - s * beta) / (2 * np.pi)  # normal velocity from unit vortices
    At = (s * beta - c * lr) / (2 * np.pi)  # tangential from sources
    Bt = (s * lr + c * beta) / (2 * np.pi)  # tangential from vortices
    if lifting:
        M = np.zeros((n + 1, n + 1))
        rhs = np.zeros(n + 1)
        M[:n, :n] = An
        M[:n, n] = Bn.sum(1)
        rhs[:n] = -np.sin(a - th)  # free-stream normal component with outward normal (-sin th, cos th)
        M[n, :n] = At[0] + At[-1]
        M[n, n] = Bt[0].sum() + Bt[-1].sum()
        rhs[n] = -np.cos(th[0] - a) - np.cos(th[-1] - a)
        sol = np.linalg.solve(M, rhs)
        q, gam = sol[:n], float(sol[n])
    else:
        q, gam = np.linalg.solve(An, -np.sin(a - th)), 0.0
    vt = At @ q + gam * Bt.sum(1) + np.cos(th - a)
    beta_pg = np.sqrt(1 - mach**2)
    cp = (1 - vt**2) / beta_pg
    cl = 2 * gam * L.sum() / beta_pg
    # pressure integration (force per unit span, chord = 1); outward normal for clockwise panels
    nx, ny = -np.sin(th), np.cos(th)
    fx, fy = -np.sum(cp * L * nx), -np.sum(cp * L * ny)
    cl_p = float(fy * np.cos(a) - fx * np.sin(a))
    # moment about c/4, positive nose-up: M = sum cp L [(x - 1/4) n_y - y n_x]
    cm = float(np.sum(cp * L * ((xc - 0.25) * ny - yc * nx)))
    return PanelResult(xc, yc, cp, vt, float(cl), cl_p, cm, gam, L)
