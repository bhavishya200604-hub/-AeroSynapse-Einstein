r"""Wave equation :math:`u_{tt}=c^2u_{xx}` by the second-order leapfrog (central-in-time) scheme.

Stable iff the Courant number :math:`C=c\Delta t/\Delta x\le 1` (von Neumann). The discrete energy
:math:`E^{n+1/2}=\tfrac12\sum[(u^{n+1}-u^n)/\Delta t]^2 h+\tfrac{c^2}2\sum(u^{n+1}_{i+1}-u^{n+1}_i)(u^n_{i+1}-u^n_i)/h`
is exactly conserved; we monitor its drift.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from aerosynapse.exceptions import ConfigurationError, StabilityError
from aerosynapse.numerics.guards import check_field
from aerosynapse.registry import SOLVERS

FloatArr = NDArray[np.float64]


@SOLVERS.register("wave_fd")
def solve_wave(
    L: float,
    n: int,
    c: float,
    u0: Callable[[FloatArr], FloatArr],
    t_final: float,
    courant: float = 0.8,
    v0: Callable[[FloatArr], FloatArr] | None = None,
) -> tuple[FloatArr, FloatArr, list[float]]:
    """Fixed ends u(0)=u(L)=0. Returns ``(x, u(t_final), energy_history)``."""
    if courant > 1.0:
        raise StabilityError(f"Courant number {courant} > 1: leapfrog is unstable")
    if c <= 0:
        raise ConfigurationError("wave speed must be positive")
    h = L / n
    x = np.linspace(0.0, L, n + 1)
    nt = int(np.ceil(t_final * c / (courant * h)))
    dt = t_final / nt
    C2 = (c * dt / h) ** 2
    u_prev = u0(x).astype(float)
    vel = v0(x) if v0 is not None else np.zeros_like(x)
    lap = np.zeros_like(x)
    lap[1:-1] = u_prev[2:] - 2 * u_prev[1:-1] + u_prev[:-2]
    u = u_prev + dt * vel + 0.5 * C2 * lap  # second-order Taylor start
    u[0] = u[-1] = 0.0

    def energy(a: FloatArr, b: FloatArr) -> float:
        kin = 0.5 * np.sum(((b - a) / dt) ** 2) * h
        pot = 0.5 * c**2 * np.sum(np.diff(b) * np.diff(a)) / h
        return float(kin + pot)

    E = [energy(u_prev, u)]
    for k in range(1, nt):
        un = np.empty_like(u)
        un[1:-1] = 2 * u[1:-1] - u_prev[1:-1] + C2 * (u[2:] - 2 * u[1:-1] + u[:-2])
        un[0] = un[-1] = 0.0
        u_prev, u = u, un
        check_field(u, step=k)
        E.append(energy(u_prev, u))
    return x, u, E
