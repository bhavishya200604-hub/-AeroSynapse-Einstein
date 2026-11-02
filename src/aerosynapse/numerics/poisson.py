r"""Poisson :math:`-\nabla^2u=f`: 1D (all BC types via :mod:`fd1d`) and 2D five-point FD with
Dirichlet or Neumann sides (ghost-node elimination, second order)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from aerosynapse.exceptions import ConfigurationError
from aerosynapse.numerics.fd1d import assemble_1d, steady_solve
from aerosynapse.physics.bc import BoundaryCondition
from aerosynapse.physics.problem import PDEProblem
from aerosynapse.registry import SOLVERS

FloatArr = NDArray[np.float64]


@SOLVERS.register("poisson1d_fd")
def solve_poisson_1d(
    x_lo: float,
    x_hi: float,
    n: int,
    f: Callable[[FloatArr], FloatArr],
    left: BoundaryCondition,
    right: BoundaryCondition,
) -> tuple[FloatArr, FloatArr]:
    """``-u'' = f`` ⇔ ``u'' + f = 0``; pure-Neumann/periodic cases are gauge-fixed (zero mean)."""
    op = assemble_1d(x_lo, x_hi, n, 1.0, 0.0, left, right)
    return op.x, steady_solve(op, source=f)


@SOLVERS.register("poisson2d_fd")
def solve_poisson_2d(problem: PDEProblem, n: int) -> tuple[FloatArr, FloatArr, FloatArr]:
    """Node-centred grid with ``n`` intervals per direction. Each side may be Dirichlet(g(s)) or
    Neumann(g(s)) with ``s`` the coordinate along the side. Returns ``(X, Y, U)``, row = y."""
    (x0, x1), (y0, y1) = problem.domain
    if problem.source is None:
        raise ConfigurationError("Poisson problem needs a source f")
    h, k = (x1 - x0) / n, (y1 - y0) / n
    xs, ys = np.linspace(x0, x1, n + 1), np.linspace(y0, y1, n + 1)
    X, Y = np.meshgrid(xs, ys)
    N = (n + 1) ** 2
    idx = np.arange(N).reshape(n + 1, n + 1)
    F = problem.source(X, Y).astype(float)
    sides = {s: problem.bcs[s] for s in ("left", "right", "bottom", "top")}
    if all(b.kind == "neumann" for b in sides.values()):
        raise ConfigurationError("pure-Neumann 2D Poisson not supported here; fix one side as Dirichlet")
    A = sp.lil_matrix((N, N))
    rhs = F.ravel().copy()
    cx, cy = 1.0 / h**2, 1.0 / k**2
    for j in range(n + 1):
        for i in range(n + 1):
            r = idx[j, i]
            dside = None
            for name, cond in (("left", i == 0), ("right", i == n), ("bottom", j == 0), ("top", j == n)):
                if cond and sides[name].kind == "dirichlet":
                    dside = name
            if dside is not None:
                s = Y[j, i] if dside in ("left", "right") else X[j, i]
                A[r, r] = 1.0
                rhs[r] = float(sides[dside].value(s))
                continue
            A[r, r] = 2 * cx + 2 * cy
            for di, dj, c, name, cond in (
                (-1, 0, cx, "left", i == 0),
                (1, 0, cx, "right", i == n),
                (0, -1, cy, "bottom", j == 0),
                (0, 1, cy, "top", j == n),
            ):
                if cond:  # Neumann ghost: u_g = u_mirror + 2 h g (outward derivative g)
                    s = Y[j, i] if name in ("left", "right") else X[j, i]
                    hh = h if di else k
                    A[r, idx[j - dj, i - di]] -= c
                    rhs[r] += c * 2 * hh * float(sides[name].value(s))
                else:
                    A[r, idx[j + dj, i + di]] -= c
    U = spla.spsolve(A.tocsc(), rhs).reshape(n + 1, n + 1)
    return X, Y, U
