r"""Viscous Burgers :math:`u_t + (u^2/2)_x = \nu u_{xx}`: conservative FV, MUSCL-minmod +
local Lax–Friedrichs (Rusanov) flux, central viscous flux, SSP-RK3 with CFL-adaptive Δt."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from aerosynapse.exceptions import ConfigurationError
from aerosynapse.numerics.convdiff import FVResult
from aerosynapse.numerics.guards import check_field
from aerosynapse.numerics.timestep import ssp_rk3
from aerosynapse.registry import SOLVERS

FloatArr = NDArray[np.float64]


def _minmod(a: FloatArr, b: FloatArr) -> FloatArr:
    return np.asarray(np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0))


def burgers_rhs(u: FloatArr, h: float, nu: float, limiter: bool = True) -> FloatArr:
    """Semi-discrete conservative RHS on a periodic grid."""
    du = np.roll(u, -1) - u
    slope = _minmod(np.roll(du, 1), du) if limiter else 0.5 * (np.roll(du, 1) + du)
    uL = u + 0.5 * slope  # left state at face i+1/2
    uR = np.roll(u - 0.5 * slope, -1)  # right state at face i+1/2
    a = np.maximum(np.abs(uL), np.abs(uR))
    F = 0.25 * (uL**2 + uR**2) - 0.5 * a * (uR - uL) - nu * du / h
    return np.asarray(-(F - np.roll(F, 1)) / h)


@SOLVERS.register("burgers_fv")
def solve_burgers(
    x_lo: float,
    x_hi: float,
    n: int,
    nu: float,
    u0: Callable[[FloatArr], FloatArr],
    t_final: float,
    cfl: float = 0.4,
    limiter: bool = True,
    max_steps: int = 2_000_000,
) -> FVResult:
    """Periodic solver with adaptive time step ``Δt = cfl·min(h/max|u|, h²/(2ν))``."""
    if nu < 0:
        raise ConfigurationError("viscosity must be non-negative")
    h = (x_hi - x_lo) / n
    x = np.asarray(x_lo + h * (np.arange(n) + 0.5), dtype=np.float64)
    gp = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]) * h / 2
    gw = np.array([5, 8, 5]) / 18
    u = np.asarray(sum(w * u0(x + p) for p, w in zip(gp, gw, strict=True)), dtype=float)
    res = FVResult(x=x, u=u, t=0.0, steps=0)
    res.mass.append(float(h * u.sum()))
    t, k = 0.0, 0
    while t < t_final - 1e-14:
        umax = max(float(np.abs(u).max()), 1e-12)
        dt = cfl * min(h / umax, h**2 / (2 * nu) if nu > 0 else np.inf)
        dt = min(dt, t_final - t)
        u = ssp_rk3(lambda w, _t: burgers_rhs(w, h, nu, limiter), u, t, dt)
        t += dt
        k += 1
        check_field(u, step=k)
        res.mass.append(float(h * u.sum()))
        if k > max_steps:
            raise ConfigurationError("max_steps exceeded")
    res.u, res.t, res.steps = u, t, k
    return res
