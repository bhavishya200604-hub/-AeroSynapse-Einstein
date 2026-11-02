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
    cx, cy = 1.0 / h**2, 1.0 / k**2
    J, I = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")  # J = row (y), I = column (x)
    rhs = F.ravel().copy()
    dmask = np.zeros((n + 1, n + 1), bool)
    dval = np.zeros((n + 1, n + 1))
    for name, sel, coord in (
        ("left", (slice(None), 0), Y[:, 0]),
        ("right", (slice(None), n), Y[:, n]),
        ("bottom", (0, slice(None)), X[0, :]),
        ("top", (n, slice(None)), X[n, :]),
    ):
        if sides[name].kind == "dirichlet":
            dmask[sel] = True
            dval[sel] = np.asarray(sides[name].value(coord), dtype=float) * np.ones(n + 1)
    free = ~dmask
    rows: list[np.ndarray] = [idx[free]]
    cols: list[np.ndarray] = [idx[free]]
    vals: list[np.ndarray] = [np.full(int(free.sum()), 2 * cx + 2 * cy)]
    for di, dj, c, name in ((-1, 0, cx, "left"), (1, 0, cx, "right"), (0, -1, cy, "bottom"), (0, 1, cy, "top")):
        Ii, Jj = I + di, J + dj
        out = (Ii < 0) | (Ii > n) | (Jj < 0) | (Jj > n)
        inside = free & ~out
        rows.append(idx[inside])
        cols.append(idx[Jj[inside], Ii[inside]])
        vals.append(np.full(int(inside.sum()), -c))
        ghost = free & out  # Neumann ghost: u_g = u_mirror + 2 h g (outward derivative g)
        if ghost.any():
            hh = h if di else k
            Im, Jm = I - di, J - dj
            rows.append(idx[ghost])
            cols.append(idx[Jm[ghost], Im[ghost]])
            vals.append(np.full(int(ghost.sum()), -c))
            s_coord = Y[ghost] if name in ("left", "right") else X[ghost]
            rhs[idx[ghost]] += (
                c * 2 * hh * np.asarray(sides[name].value(s_coord), dtype=float) * np.ones(int(ghost.sum()))
            )
    rows.append(idx[dmask])
    cols.append(idx[dmask])
    vals.append(np.ones(int(dmask.sum())))
    rhs[idx[dmask]] = dval[dmask]
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    U = spla.spsolve(A.tocsc(), rhs).reshape(n + 1, n + 1)
    return X, Y, U
