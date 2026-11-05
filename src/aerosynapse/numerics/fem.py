r"""Minimal P1 (linear triangle) finite-element solver for :math:`-\nabla^2u=f` on a rectangle.

Independent cross-check of the FD Poisson solver. Weak form: find :math:`u\in V_h` with
:math:`\int\nabla u\cdot\nabla v=\int fv` for all :math:`v\in V_{h,0}`; Dirichlet data imposed
nodally. Expected rates: :math:`O(h^2)` in :math:`L^2`, :math:`O(h)` in :math:`H^1`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from aerosynapse.registry import SOLVERS

FloatArr = NDArray[np.float64]
IntArr = NDArray[np.int64]


def structured_mesh(
    n: int, x0: float = 0.0, x1: float = 1.0, y0: float = 0.0, y1: float = 1.0
) -> tuple[FloatArr, IntArr]:
    """Right-triangle mesh of a rectangle: nodes (N, 2), triangles (M, 3) counter-clockwise."""
    xs, ys = np.linspace(x0, x1, n + 1), np.linspace(y0, y1, n + 1)
    X, Y = np.meshgrid(xs, ys)
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    tris = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    return nodes, tris.astype(np.int64)


def _grads(nodes: FloatArr, tris: IntArr) -> tuple[FloatArr, FloatArr]:
    P = nodes[tris]  # (M, 3, 2)
    d1, d2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    area = 0.5 * np.abs(det)
    # gradients of barycentric basis functions
    G = np.empty((len(tris), 3, 2))
    G[:, 1] = np.column_stack([d2[:, 1], -d2[:, 0]]) / det[:, None]
    G[:, 2] = np.column_stack([-d1[:, 1], d1[:, 0]]) / det[:, None]
    G[:, 0] = -G[:, 1] - G[:, 2]
    return G, area


_QP = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])  # edge-midpoint rule (degree 2)


@SOLVERS.register("poisson2d_fem_p1")
def solve_poisson_fem(
    n: int, f: Callable[[FloatArr, FloatArr], FloatArr], g: Callable[[FloatArr, FloatArr], FloatArr]
) -> tuple[FloatArr, IntArr, FloatArr]:
    """Solve on the unit square with Dirichlet data ``g``; returns ``(nodes, tris, u)``."""
    nodes, tris = structured_mesh(n)
    G, area = _grads(nodes, tris)
    Ke = np.einsum("mik,mjk->mij", G, G) * area[:, None, None]
    rows = np.repeat(tris, 3, axis=1).ravel()
    cols = np.tile(tris, (1, 3)).ravel()
    K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(len(nodes),) * 2)
    P = nodes[tris]
    b = np.zeros(len(nodes))
    for q in _QP:
        xq = np.einsum("i,mik->mk", q, P)
        fq = f(xq[:, 0], xq[:, 1])
        np.add.at(b, tris, (fq * area / 3)[:, None] * q[None, :])
    x, y = nodes[:, 0], nodes[:, 1]
    bnd = np.isclose(x, 0) | np.isclose(x, 1) | np.isclose(y, 0) | np.isclose(y, 1)
    u = np.zeros(len(nodes))
    u[bnd] = g(x[bnd], y[bnd])
    free = ~bnd
    rhs = b[free] - K[free][:, bnd] @ u[bnd]
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    return nodes, tris, u


def fem_errors(
    nodes: FloatArr,
    tris: IntArr,
    u: FloatArr,
    exact: Callable[[FloatArr, FloatArr], FloatArr],
    grad_exact: Callable[[FloatArr, FloatArr], tuple[FloatArr, FloatArr]],
) -> tuple[float, float]:
    """``(L2 error, H1-seminorm error)`` by the degree-2 edge-midpoint rule."""
    G, area = _grads(nodes, tris)
    P = nodes[tris]
    ue = u[tris]
    e2 = 0.0
    gu = np.einsum("mi,mik->mk", ue, G)
    h1 = 0.0
    for q in _QP:
        xq = np.einsum("i,mik->mk", q, P)
        uh = ue @ q
        e2 += float(np.sum(area / 3 * (uh - exact(xq[:, 0], xq[:, 1])) ** 2))
        gx, gy = grad_exact(xq[:, 0], xq[:, 1])
        h1 += float(np.sum(area / 3 * ((gu[:, 0] - gx) ** 2 + (gu[:, 1] - gy) ** 2)))
    return float(np.sqrt(e2)), float(np.sqrt(h1))
