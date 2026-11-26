r"""Laminar flat-plate boundary layer (spec Level 2: flow over a flat plate).

1. Blasius similarity solution :math:`f'''+\tfrac12ff''=0`, f(0)=f'(0)=0, f'(∞)=1, solved by shooting
   (RK4 + bisection on f''(0)); the wall shear parameter is *computed*, not recalled.
2. A marching solver for the Prandtl boundary-layer equations
   :math:`uu_x+vu_y=\nu u_{yy}`, :math:`u_x+v_y=0` (implicit in y, Picard-linearised), whose profiles
   are verified against Blasius in similarity variables :math:`\eta=y\sqrt{U/(\nu x)}`.
Valid only for laminar flow: :mod:`aerosynapse.physics.regime` guards Re_x.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]


def _rk4(fpp0: float, eta_max: float, n: int) -> tuple[FloatArr, FloatArr]:
    h = eta_max / n
    Y = np.zeros((n + 1, 3))
    Y[0] = [0.0, 0.0, fpp0]

    def F(y: FloatArr) -> FloatArr:
        return np.array([y[1], y[2], -0.5 * y[0] * y[2]])

    for i in range(n):
        y = Y[i]
        k1 = F(y)
        k2 = F(y + 0.5 * h * k1)
        k3 = F(y + 0.5 * h * k2)
        k4 = F(y + h * k3)
        Y[i + 1] = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.linspace(0, eta_max, n + 1), Y


def blasius(eta_max: float = 10.0, n: int = 4000) -> tuple[FloatArr, FloatArr, float]:
    """Returns (eta, f'(eta), f''(0)) with f'(eta_max) = 1 to 1e-10."""
    lo, hi = 0.1, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        _, Y = _rk4(mid, eta_max, n)
        if Y[-1, 1] > 1.0:
            hi = mid
        else:
            lo = mid
    eta, Y = _rk4(0.5 * (lo + hi), eta_max, n)
    return eta, Y[:, 1], float(0.5 * (lo + hi))


def march_boundary_layer(
    U: float = 1.0, nu: float = 1e-4, L: float = 1.0, ny: int = 200, nx: int = 400, y_max: float | None = None
) -> tuple[FloatArr, FloatArr, FloatArr]:
    """Marches from x = L/nx to L. Returns (x, y, u[x, y])."""
    y_max = y_max or 8.0 * np.sqrt(nu * L / U)
    y = np.linspace(0, y_max, ny + 1)
    dy = y[1] - y[0]
    xs = np.linspace(0, L, nx + 1)[1:]
    x0 = xs[0]
    eta, fp, _ = blasius()
    u = np.interp(y * np.sqrt(U / (nu * x0)), eta, fp) * U  # start from a similarity profile at x0
    out = [u.copy()]
    for k in range(1, len(xs)):
        dx = xs[k] - xs[k - 1]
        un = u.copy()
        for _ in range(4):  # Picard iterations
            v = np.zeros_like(u)
            dudx = (un - u) / dx
            v[1:] = -np.cumsum(0.5 * (dudx[1:] + dudx[:-1])) * dy
            n_in = ny - 1
            main = un[1:-1] / dx + 2 * nu / dy**2
            lower = -v[2:-1] / (2 * dy) - nu / dy**2
            upper = v[1:-2] / (2 * dy) - nu / dy**2
            A = sp.diags([lower, main, upper], [-1, 0, 1], shape=(n_in, n_in), format="csc")
            rhs = un[1:-1] * u[1:-1] / dx
            rhs[-1] -= (v[-2] / (2 * dy) - nu / dy**2) * U
            un[1:-1] = spla.spsolve(A, rhs)
            un[0], un[-1] = 0.0, U
        u = un
        out.append(u.copy())
    return xs, y, np.array(out)
