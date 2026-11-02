r"""Second-order node-centred finite-difference assembly for 1D linear operators.

Builds the semi-discrete operator for

.. math:: u_t = D u_{xx} - v u_x - \kappa u + s(x,t) = A u + b(t) + s

with Dirichlet, Neumann, Robin or periodic ends. Neumann/Robin use a ghost node eliminated with
the central difference of the outward normal derivative (second order). Convection is either
``"central"`` (2nd order, oscillates for cell Péclet > 2) or ``"upwind"`` (1st order, monotone).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from aerosynapse.exceptions import ConfigurationError
from aerosynapse.physics.bc import BoundaryCondition, Dirichlet, Neumann, Robin

FloatArr = NDArray[np.float64]


@dataclass
class Operator1D:
    """Semi-discrete affine operator ``A u + b(t)``; Dirichlet nodes are handled by the solver."""

    x: FloatArr
    h: float
    A: sp.csr_matrix
    periodic: bool
    dirichlet: dict[int, BoundaryCondition] = field(default_factory=dict)
    affine: list[tuple[int, float, BoundaryCondition]] = field(default_factory=list)

    def b(self, t: float) -> FloatArr:
        out = np.zeros(self.x.size)
        for row, coef, bc in self.affine:
            out[row] += coef * float(bc.value(t))
        return out

    def dirichlet_values(self, t: float) -> dict[int, float]:
        return {i: float(bc.value(t)) for i, bc in self.dirichlet.items()}


def _robin_ab(bc: BoundaryCondition) -> tuple[float, float]:
    if isinstance(bc, Neumann):
        return 0.0, 1.0
    if isinstance(bc, Robin):
        return bc.a, bc.b
    raise ConfigurationError(f"unsupported BC {bc.kind}")


def assemble_1d(
    x_lo: float,
    x_hi: float,
    n: int,
    D: float,
    v: float = 0.0,
    left: BoundaryCondition | None = None,
    right: BoundaryCondition | None = None,
    scheme: str = "central",
    kappa: float = 0.0,
) -> Operator1D:
    """Assemble ``A`` for ``n`` intervals. Periodic BCs use ``n`` unknowns, others ``n+1``."""
    if n < 2:
        raise ConfigurationError("need at least 2 intervals")
    if D < 0:
        raise ConfigurationError("diffusivity must be non-negative")
    if scheme not in ("central", "upwind"):
        raise ConfigurationError(f"unknown convection scheme {scheme!r}")
    left = left if left is not None else Dirichlet(0.0)
    right = right if right is not None else Dirichlet(0.0)
    periodic = left.kind == "periodic"
    if periodic != (right.kind == "periodic"):
        raise ConfigurationError("periodic BC must be set on both ends")
    h = (x_hi - x_lo) / n
    m = n if periodic else n + 1
    x = np.asarray(x_lo + h * np.arange(m), dtype=np.float64)
    dif = D / h**2
    if scheme == "central":
        cm, cc, cp = dif + v / (2 * h), -2 * dif, dif - v / (2 * h)
    elif v >= 0:
        cm, cc, cp = dif + v / h, -2 * dif - v / h, dif
    else:
        cm, cc, cp = dif, -2 * dif + v / h, dif - v / h
    cc -= kappa
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def add(r: int, c: int, val: float) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(val)

    op = Operator1D(x=x, h=h, A=sp.csr_matrix((m, m)), periodic=periodic)
    for i in range(m):
        if periodic:
            add(i, (i - 1) % m, cm)
            add(i, i, cc)
            add(i, (i + 1) % m, cp)
            continue
        if i == 0 or i == m - 1:
            bc = left if i == 0 else right
            if bc.kind == "dirichlet":
                op.dirichlet[i] = bc
                continue
            a, bb = _robin_ab(bc)
            # ghost: u_g = u_nb + 2h (g - a u_b)/b, nb = interior neighbour
            ghost_coef, nb_coef = (cm, cp) if i == 0 else (cp, cm)
            nb = 1 if i == 0 else m - 2
            add(i, i, cc - ghost_coef * 2 * h * a / bb)
            add(i, nb, nb_coef + ghost_coef)
            op.affine.append((i, ghost_coef * 2 * h / bb, bc))
            continue
        add(i, i - 1, cm)
        add(i, i, cc)
        add(i, i + 1, cp)
    op.A = sp.csr_matrix((vals, (rows, cols)), shape=(m, m))
    return op


def trapezoid_weights(op: Operator1D) -> FloatArr:
    """Quadrature weights consistent with the node-centred grid (exact mass for periodic)."""
    w = np.full(op.x.size, op.h)
    if not op.periodic:
        w[0] = w[-1] = op.h / 2
    return w


def steady_solve(
    op: Operator1D, source: Callable[[FloatArr], FloatArr] | None = None, gauge_tol: float = 1e-8
) -> FloatArr:
    r"""Solve ``A u + b + s = 0``. Without Dirichlet rows and with ``A`` singular (pure Neumann /
    periodic diffusion) a Lagrange multiplier fixes the zero-mean gauge; an incompatible
    right-hand side (discrete solvability condition violated) raises ``ConfigurationError``."""
    import scipy.sparse.linalg as spla

    m = op.x.size
    rhs = -(op.b(0.0) + (source(op.x) if source is not None else 0.0))
    A = op.A.tolil()
    for i, g in op.dirichlet_values(0.0).items():
        A.rows[i] = [i]
        A.data[i] = [1.0]
        rhs[i] = g
    A = A.tocsr()
    if not op.dirichlet and np.abs(A @ np.ones(m)).max() < 1e-9 * abs(A).max():
        w = trapezoid_weights(op)
        K = sp.bmat([[A, w[:, None]], [w[None, :], None]], format="csc")
        sol = spla.spsolve(K, np.append(rhs, 0.0))
        lam = sol[-1]
        if abs(lam) > gauge_tol * max(1.0, float(np.abs(rhs).max())):
            raise ConfigurationError(f"incompatible pure-Neumann/periodic data: solvability defect {lam:.3e}")
        return np.asarray(sol[:m])
    return np.asarray(spla.spsolve(A.tocsc(), rhs))
