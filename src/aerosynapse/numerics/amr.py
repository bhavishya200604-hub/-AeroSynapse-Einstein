r"""1D adaptive mesh refinement for the steady convection–diffusion boundary layer
:math:`v u_x = D u_{xx}` with Dirichlet ends, on non-uniform grids.

Discretisation (non-uniform three-point, second order for smoothly varying grids):
:math:`u_{xx}\approx \frac{2}{h_l+h_r}\big(\frac{u_{i+1}-u_i}{h_r}-\frac{u_i-u_{i-1}}{h_l}\big)`,
upwind or central convection. The refinement indicator is the interpolation-error estimate
:math:`\eta_K = h_K^2|u_{xx}|` per cell; cells above ``frac·max η`` are bisected.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]


def solve_nonuniform(x: FloatArr, v: float, D: float, uL: float, uR: float, scheme: str = "central") -> FloatArr:
    n = x.size
    hl, hr = np.diff(x)[:-1], np.diff(x)[1:]
    A = sp.lil_matrix((n, n))
    b = np.zeros(n)
    A[0, 0] = A[n - 1, n - 1] = 1.0
    b[0], b[-1] = uL, uR
    for i in range(1, n - 1):
        a, c = hl[i - 1], hr[i - 1]
        dl, dr = 2 * D / (a * (a + c)), 2 * D / (c * (a + c))
        if scheme == "central":
            cl, cc, cr = -c / (a * (a + c)), (c - a) / (a * c), a / (c * (a + c))
        else:  # upwind, v > 0
            cl, cc, cr = -1 / a, 1 / a, 0.0
        A[i, i - 1] = dl - v * cl
        A[i, i] = -(dl + dr) - v * cc
        A[i, i + 1] = dr - v * cr
    return np.asarray(spla.spsolve(A.tocsr(), b))


def refine(x: FloatArr, u: FloatArr, frac: float = 0.3) -> FloatArr:
    h = np.diff(x)
    du = np.diff(u) / h
    curv = np.zeros_like(h)
    curv[1:-1] = np.abs(np.diff(du)[1:] + np.diff(du)[:-1]) / (h[1:-1] + 0.5 * (h[:-2] + h[2:]))
    curv[0], curv[-1] = curv[1], curv[-2]
    eta = h**2 * curv
    mark = eta > frac * float(np.max(eta))
    mids = 0.5 * (x[:-1] + x[1:])[mark]
    return np.sort(np.concatenate([x, mids]))


def adaptive_solve(
    v: float, D: float, n0: int = 16, max_nodes: int = 200, frac: float = 0.3, scheme: str = "central"
) -> tuple[FloatArr, FloatArr, list[int]]:
    """Refine until ``max_nodes`` is reached. Returns final grid, solution, node-count history."""
    x = np.linspace(0, 1, n0 + 1)
    hist = [x.size]
    u = solve_nonuniform(x, v, D, 0.0, 1.0, scheme)
    while True:
        xn = refine(x, u, frac)
        if xn.size > max_nodes or xn.size == x.size:
            break
        x = xn
        u = solve_nonuniform(x, v, D, 0.0, 1.0, scheme)
        hist.append(x.size)
    return x, u, hist
