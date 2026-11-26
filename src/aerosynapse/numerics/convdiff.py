r"""Conservative finite-volume solver for 1D transient convection–diffusion.

.. math:: \frac{d\bar u_i}{dt} = -\frac{F_{i+1/2}-F_{i-1/2}}{h} + s_i,\qquad
          F_{i+1/2} = v\,u_{i+1/2} - D\frac{\bar u_{i+1}-\bar u_i}{h}

Face reconstructions (for v > 0; mirrored for v < 0):

* ``upwind``  :math:`u_{i+1/2}=\bar u_i` (1st order, monotone)
* ``central`` :math:`(\bar u_i+\bar u_{i+1})/2` (2nd order, oscillates for :math:`Pe_h>2`)
* ``quick``   :math:`\tfrac68\bar u_i+\tfrac38\bar u_{i+1}-\tfrac18\bar u_{i-1}` (Leonard; 3rd-order
  face interpolation, 2nd-order overall FV with cell averages treated as point values)
* ``vanleer`` MUSCL with the van Leer limiter (TVD, 2nd order away from extrema)

Time integration: SSP-RK3 (explicit, with CFL / diffusion-number checks) or θ-method for the linear
schemes (matrix built from the stencil). Periodic, Dirichlet (two mirrored ghost cells) and Neumann
ends are supported. Mass ``h Σ ū`` is conserved to round-off with periodic BCs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from aerosynapse.exceptions import ConfigurationError, StabilityError
from aerosynapse.numerics.guards import check_field
from aerosynapse.numerics.timestep import ssp_rk3
from aerosynapse.physics.bc import BoundaryCondition, Periodic
from aerosynapse.registry import SOLVERS

FloatArr = NDArray[np.float64]
SCHEMES = ("upwind", "central", "quick", "vanleer")
NG = 2  # ghost cells


@dataclass
class FVResult:
    x: FloatArr
    u: FloatArr
    t: float
    steps: int
    mass: list[float] = field(default_factory=list)
    dt: float = 0.0


def _ghosts(u: FloatArr, h: float, t: float, left: BoundaryCondition, right: BoundaryCondition) -> FloatArr:
    if left.kind == "periodic":
        return np.concatenate([u[-NG:], u, u[:NG]])
    ug = np.empty(u.size + 2 * NG)
    ug[NG:-NG] = u
    for side, bc in (("l", left), ("r", right)):
        g = float(bc.value(t))
        a, b = (u[0], u[1]) if side == "l" else (u[-1], u[-2])
        if bc.kind == "dirichlet":
            g1, g2 = 2 * g - a, 2 * g - b
        elif bc.kind == "neumann":  # outward derivative g across the face
            g1, g2 = a + h * g, b + 3 * h * g
        else:
            raise ConfigurationError(f"FV solver supports periodic/dirichlet/neumann, got {bc.kind}")
        if side == "l":
            ug[NG - 1], ug[NG - 2] = g1, g2
        else:
            ug[-NG], ug[-NG + 1] = g1, g2
    return ug


def _face_values(ug: FloatArr, v: float, scheme: str) -> FloatArr:
    """Face values at the n+1 faces of the n physical cells (ug includes NG ghosts each side)."""
    if v < 0:
        return _face_values(ug[::-1], -v, scheme)[::-1]
    um, u0, up = (
        ug[NG - 2 : -NG - 1],
        ug[NG - 1 : -NG],
        ug[NG : len(ug) - NG + 1],
    )  # upwind-1, upwind, downwind
    if scheme == "upwind":
        return u0
    if scheme == "central":
        return 0.5 * (u0 + up)
    if scheme == "quick":
        return 0.75 * u0 + 0.375 * up - 0.125 * um
    if scheme == "vanleer":
        d1, d0 = up - u0, u0 - um
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(np.abs(d1) > 1e-300, d0 / d1, 0.0)
        psi = (r + np.abs(r)) / (1 + np.abs(r))
        return np.asarray(u0 + 0.5 * psi * d1)
    raise ConfigurationError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")


def fv_rhs(
    u: FloatArr,
    t: float,
    h: float,
    v: float,
    D: float,
    scheme: str,
    left: BoundaryCondition,
    right: BoundaryCondition,
    source: Callable[[FloatArr, float], FloatArr] | None = None,
    x: FloatArr | None = None,
) -> FloatArr:
    ug = _ghosts(u, h, t, left, right)
    uf = _face_values(ug, v, scheme)
    grad = (ug[NG : len(ug) - NG + 1] - ug[NG - 1 : -NG]) / h
    F = v * uf - D * grad
    out = -(F[1:] - F[:-1]) / h
    if source is not None and x is not None:
        out = out + source(x, t)
    return np.asarray(out)


def stable_dt(h: float, v: float, D: float, cfl: float = 0.5) -> float:
    """Explicit SSP-RK3 step: advective |v|Δt/h ≤ cfl and diffusive DΔt/h² ≤ cfl/2.

    With cfl = 1 the diffusive eigenvalues span [-2, 0], inside the RK3 real-axis stability
    interval (≈ [-2.51, 0])."""
    lims = []
    if v != 0:
        lims.append(cfl * h / abs(v))
    if D > 0:
        lims.append(cfl * h**2 / (2 * D))
    if not lims:
        raise ConfigurationError("v and D are both zero")
    return min(lims)


@SOLVERS.register("convdiff_fv")
def solve_convdiff_fv(
    x_lo: float,
    x_hi: float,
    n: int,
    v: float,
    D: float,
    u0: Callable[[FloatArr], FloatArr],
    t_final: float,
    scheme: str = "vanleer",
    left: BoundaryCondition | None = None,
    right: BoundaryCondition | None = None,
    cfl: float = 0.5,
    dt: float | None = None,
    integrator: str = "rk3",
    theta: float = 0.5,
    source: Callable[[FloatArr, float], FloatArr] | None = None,
) -> FVResult:
    """Solve on ``n`` cells; returns cell centres and cell values at ``t_final``."""
    left = left if left is not None else Periodic()
    right = right if right is not None else Periodic()
    if scheme not in SCHEMES:
        raise ConfigurationError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")
    h = (x_hi - x_lo) / n
    x = np.asarray(x_lo + h * (np.arange(n) + 0.5), dtype=np.float64)
    # cell averages of u0 by 3-point Gauss quadrature (4th-order accurate initial data)
    gp = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]) * h / 2
    gw = np.array([5, 8, 5]) / 18
    u = sum(w * u0(x + p) for p, w in zip(gp, gw, strict=True))
    u = np.asarray(u, dtype=float)
    limit = stable_dt(h, v, D, 1.0)
    if dt is None:
        dt = stable_dt(h, v, D, cfl) if integrator == "rk3" else min(h / max(abs(v), 1e-300), t_final / 10)
    if integrator == "rk3" and dt > limit * 1.0000001:
        raise StabilityError(f"dt={dt:.3e} exceeds explicit stability limit {limit:.3e}")
    nt = int(np.ceil(t_final / dt - 1e-12))
    dt = t_final / nt
    res = FVResult(x=x, u=u, t=0.0, steps=0, dt=dt)
    res.mass.append(float(h * u.sum()))

    def rhs(w: FloatArr, t: float) -> FloatArr:
        return fv_rhs(w, t, h, v, D, scheme, left, right, source, x)

    if integrator == "rk3":
        t = 0.0
        for k in range(nt):
            u = ssp_rk3(rhs, u, t, dt)
            t += dt
            check_field(u, step=k)
            res.mass.append(float(h * u.sum()))
    elif integrator == "theta":
        if scheme == "vanleer":
            raise ConfigurationError("theta integrator requires a linear scheme (not vanleer)")
        if left.kind != "periodic" and any(float(b.value(0.0)) != 0.0 for b in (left, right)):
            raise ConfigurationError("theta integrator implemented for periodic or homogeneous BCs")
        J = sp.csr_matrix(np.column_stack([rhs(e, 0.0) - rhs(np.zeros(n), 0.0) for e in np.eye(n)]))
        I = sp.identity(n, format="csc")
        solve = spla.factorized((I - theta * dt * J).tocsc())
        R = (I + (1 - theta) * dt * J).tocsr()
        t = 0.0
        for k in range(nt):
            b = np.asarray(R @ u, dtype=np.float64)
            if source is not None:
                b = b + dt * (theta * source(x, t + dt) + (1 - theta) * source(x, t))
            u = solve(b)
            t += dt
            check_field(u, step=k)
            res.mass.append(float(h * u.sum()))
    else:
        raise ConfigurationError(f"unknown integrator {integrator!r}")
    res.u, res.t, res.steps = u, t_final, nt
    return res
