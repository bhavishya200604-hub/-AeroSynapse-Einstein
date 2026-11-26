r"""Time integration for linear semi-discrete systems ``u_t = A u + b(t) + s(t)``.

* :func:`theta_integrate`: θ-method (θ=0 explicit Euler, ½ Crank–Nicolson, 1 backward Euler),
  fixed or adaptive (step-doubling local error control) step size.
* :func:`ssp_rk3`: strong-stability-preserving RK3 (Shu–Osher) for explicit nonlinear / FV schemes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from aerosynapse.exceptions import ConfigurationError, SolverDivergenceError
from aerosynapse.numerics.fd1d import Operator1D
from aerosynapse.numerics.guards import check_field

FloatArr = NDArray[np.float64]
SourceFn = Callable[[FloatArr, float], FloatArr]


@dataclass
class TimeSolution:
    t: float
    u: FloatArr
    steps: int
    rejected: int = 0
    dt_history: list[float] = field(default_factory=list)
    mass_history: list[float] = field(default_factory=list)


class _ThetaStepper:
    def __init__(self, op: Operator1D, theta: float, source: SourceFn | None) -> None:
        if not 0.0 <= theta <= 1.0:
            raise ConfigurationError("theta must be in [0, 1]")
        self.op, self.theta, self.source = op, theta, source
        self.dmask = np.zeros(op.x.size, dtype=bool)
        self.dmask[list(op.dirichlet)] = True
        self._cache: dict[float, tuple[Callable[[FloatArr], FloatArr], sp.csr_matrix]] = {}

    def _factor(self, dt: float) -> tuple[Callable[[FloatArr], FloatArr], sp.csr_matrix]:
        key = round(dt, 15)
        if key not in self._cache:
            n = self.op.x.size
            I = sp.identity(n, format="csr")
            A = self.op.A
            lhs = (I - self.theta * dt * A).tolil()
            rhs_m = (I + (1 - self.theta) * dt * A).tolil()
            for i in self.op.dirichlet:
                lhs.rows[i], lhs.data[i] = [i], [1.0]
                rhs_m.rows[i], rhs_m.data[i] = [], []
            solve = spla.factorized(lhs.tocsc()) if self.theta > 0 else (lambda r: r)
            if len(self._cache) > 32:
                self._cache.clear()
            self._cache[key] = (solve, rhs_m.tocsr())
        return self._cache[key]

    def step(self, u: FloatArr, t: float, dt: float) -> FloatArr:
        solve, R = self._factor(dt)
        th = self.theta
        rhs = R @ u + dt * (th * self.op.b(t + dt) + (1 - th) * self.op.b(t))
        if self.source is not None:
            rhs = rhs + dt * (th * self.source(self.op.x, t + dt) + (1 - th) * self.source(self.op.x, t))
        for i, g in self.op.dirichlet_values(t + dt).items():
            rhs[i] = g
        return np.asarray(solve(rhs))


def theta_integrate(
    op: Operator1D,
    u0: FloatArr,
    t_final: float,
    dt: float,
    theta: float = 0.5,
    source: SourceFn | None = None,
    adaptive: bool = False,
    tol: float = 1e-6,
    dt_min: float = 1e-10,
    max_steps: int = 10_000_000,
    weights: FloatArr | None = None,
) -> TimeSolution:
    """Integrate to ``t_final``. With ``adaptive=True`` a step-doubling error estimate
    ``||u_dt - u_{dt/2,dt/2}||_inf / (2^p - 1)`` (p = 2 for CN, 1 otherwise) controls ``dt``."""
    if t_final <= 0 or dt <= 0:
        raise ConfigurationError("t_final and dt must be positive")
    stepper = _ThetaStepper(op, theta, source)
    p = 2 if abs(theta - 0.5) < 1e-12 else 1
    w = weights if weights is not None else np.full(op.x.size, op.h)
    u = np.asarray(u0, dtype=float).copy()
    for i, g in op.dirichlet_values(0.0).items():
        u[i] = g
    t, steps, rejected = 0.0, 0, 0
    sol = TimeSolution(t=0.0, u=u, steps=0)
    sol.mass_history.append(float(w @ u))
    while t < t_final - 1e-14 * t_final:
        h = min(dt, t_final - t)
        if adaptive:
            full = stepper.step(u, t, h)
            half = stepper.step(stepper.step(u, t, h / 2), t + h / 2, h / 2)
            err = float(np.max(np.abs(full - half))) / (2**p - 1)
            if err > tol and h > dt_min:
                rejected += 1
                dt = max(dt_min, h * max(0.2, 0.9 * (tol / err) ** (1.0 / (p + 1))))
                continue
            u = half
            dt = h * min(2.0, max(0.2, 0.9 * (tol / max(err, 1e-300)) ** (1.0 / (p + 1))))
        else:
            u = stepper.step(u, t, h)
        t += h
        steps += 1
        sol.dt_history.append(h)
        sol.mass_history.append(float(w @ u))
        check_field(u, step=steps)
        if steps > max_steps:
            raise SolverDivergenceError("maximum number of time steps exceeded")
    sol.t, sol.u, sol.steps, sol.rejected = t, u, steps, rejected
    return sol


def ssp_rk3(rhs: Callable[[FloatArr, float], FloatArr], u: FloatArr, t: float, dt: float) -> FloatArr:
    """One Shu–Osher SSP-RK3 step (third order, TVD under CFL ≤ 1 for TVD spatial operators)."""
    u1 = u + dt * rhs(u, t)
    u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1, t + dt))
    return np.asarray(u / 3.0 + 2.0 / 3.0 * (u2 + dt * rhs(u2, t + 0.5 * dt)))
