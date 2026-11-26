"""Time integrators for semi-discrete systems  y' = f(y, t)  or  y' = A y + b(t).

Explicit (any rhs):     euler (1), ssprk3 (3, SSP/TVD), rk4 (4)
Implicit (linear A,b):  implicit_euler (1), crank_nicolson (2), bdf2 (2; CN start-up step)
The constant matrices (I - c dt A) are factorised once (Thomas if tridiagonal, else SuperLU).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from aerosynapse.thermal.numerics.linalg import factorize

Array = NDArray[np.float64]

EXPLICIT = ("euler", "ssprk3", "rk4")
IMPLICIT = ("implicit_euler", "crank_nicolson", "bdf2")
FORMAL_ORDER = {"euler": 1, "ssprk3": 3, "rk4": 4, "implicit_euler": 1, "crank_nicolson": 2, "bdf2": 2}


class SemiDiscrete(Protocol):
    def rhs(self, y: Array, t: float) -> Array: ...
    def b(self, t: float) -> Array: ...
    @property
    def A(self) -> sps.spmatrix: ...


Callback = Callable[[int, float, Array], None]


def _explicit_step(f: Callable[[Array, float], Array], y: Array, t: float, dt: float, m: str) -> Array:
    if m == "euler":
        return y + dt * f(y, t)
    if m == "ssprk3":
        y1 = y + dt * f(y, t)
        y2 = 0.75 * y + 0.25 * (y1 + dt * f(y1, t + dt))
        return y / 3.0 + 2.0 / 3.0 * (y2 + dt * f(y2, t + 0.5 * dt))
    if m == "rk4":
        k1 = f(y, t)
        k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(y + dt * k3, t + dt)
        return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    raise ValueError(m)


def integrate(
    prob: SemiDiscrete,
    y0: Array,
    t0: float,
    t1: float,
    dt: float,
    method: str = "crank_nicolson",
    callback: Callback | None = None,
    solver: str = "auto",
) -> Array:
    """Advance from t0 to t1 with a uniform step (dt adjusted so that t1 is hit exactly)."""
    nsteps = max(1, int(np.ceil((t1 - t0) / dt - 1e-12)))
    dt = (t1 - t0) / nsteps
    y = np.array(y0, dtype=float)
    if method in EXPLICIT:
        for n in range(nsteps):
            y = _explicit_step(prob.rhs, y, t0 + n * dt, dt, method)
            if callback:
                callback(n + 1, t0 + (n + 1) * dt, y)
        return y
    if method not in IMPLICIT:
        raise ValueError(f"unknown method {method!r}")

    A = sps.csr_matrix(prob.A)
    I = sps.identity(A.shape[0], format="csr")
    if method == "implicit_euler":
        solve = factorize(I - dt * A, solver)
        for n in range(nsteps):
            t = t0 + (n + 1) * dt
            y = solve(y + dt * prob.b(t))
            if callback:
                callback(n + 1, t, y)
        return y

    half = factorize(I - 0.5 * dt * A, solver)
    Bexp = I + 0.5 * dt * A

    def cn_step(y: Array, t: float) -> Array:
        return half(Bexp @ y + 0.5 * dt * (prob.b(t) + prob.b(t + dt)))

    if method == "crank_nicolson":
        for n in range(nsteps):
            y = cn_step(y, t0 + n * dt)
            if callback:
                callback(n + 1, t0 + (n + 1) * dt, y)
        return y

    # BDF2: (I - 2/3 dt A) y^{n+2} = 4/3 y^{n+1} - 1/3 y^n + 2/3 dt b^{n+2}; CN start-up
    solve = factorize(I - (2.0 / 3.0) * dt * A, solver)
    y_prev, y = y, cn_step(y, t0)
    if callback:
        callback(1, t0 + dt, y)
    for n in range(1, nsteps):
        t = t0 + (n + 1) * dt
        y_prev, y = y, solve(4.0 / 3.0 * y - 1.0 / 3.0 * y_prev + 2.0 / 3.0 * dt * prob.b(t))
        if callback:
            callback(n + 1, t, y)
    return y
