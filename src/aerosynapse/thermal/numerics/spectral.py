"""Chebyshev spectral collocation for  theta_t + a theta_x = D theta_xx - s theta + Q  on [x0, x1].

Nodes: Chebyshev–Gauss–Lobatto xi_j = cos(pi j / N), mapped to x = x0 + (x1 - x0)(1 - xi)/2 (ascending).
Differentiation matrix: Trefethen (2000) construction with the negative-sum trick for the diagonal.
Boundary conditions  alpha theta + beta dtheta/dn = g  are imposed by row replacement (steady) or,
equivalently for time-dependent problems, by eliminating the two boundary values from the interior
equations, so that the interior unknowns obey the ODE  u' = A u + b(t)  (usable by time_integrators).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray
from scipy.interpolate import BarycentricInterpolator

from aerosynapse.thermal.numerics.fd1d import BC, dirichlet

Array = NDArray[np.float64]


def cheb(N: int) -> tuple[Array, Array]:
    """Chebyshev differentiation matrix D (N+1 x N+1) and nodes xi_j = cos(pi j/N) on [-1, 1]."""
    if N == 0:
        return np.zeros((1, 1)), np.ones(1)
    xi = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = c[-1] = 2.0
    c *= (-1.0) ** np.arange(N + 1)
    X = np.tile(xi, (N + 1, 1)).T
    dX = X - X.T
    Dm = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    Dm -= np.diag(Dm.sum(axis=1))
    return Dm, xi


def cheb_mapped(N: int, x0: float = 0.0, x1: float = 1.0) -> tuple[Array, Array]:
    """Differentiation matrix and ascending nodes on [x0, x1]."""
    Dm, xi = cheb(N)
    x = x0 + (x1 - x0) * (1.0 - xi) / 2.0
    return Dm * (-2.0 / (x1 - x0)), x


def clencurt(N: int, x0: float = 0.0, x1: float = 1.0) -> Array:
    """Clenshaw–Curtis quadrature weights on the mapped nodes (Trefethen's clencurt)."""
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    ii = np.arange(1, N)
    v = np.ones(N - 1)
    if N % 2 == 0:
        w[0] = w[N] = 1.0 / (N**2 - 1)
        for k in range(1, N // 2):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
        v -= np.cos(N * theta[ii]) / (N**2 - 1)
    else:
        w[0] = w[N] = 1.0 / N**2
        for k in range(1, (N - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
    w[ii] = 2 * v / N
    return w * (x1 - x0) / 2.0


def interpolate(x_nodes: Array, values: Array, x_eval: Array) -> Array:
    """Barycentric (spectrally accurate) interpolation from Chebyshev nodes."""
    return np.asarray(BarycentricInterpolator(x_nodes, values)(x_eval))


class ChebConvDiff1D:
    """Chebyshev collocation with the same interface as :class:`fd1d.ConvDiff1D` (rhs, A, b, nodal...)."""

    def __init__(
        self,
        N: int,
        a: float = 1.0,
        D: float = 0.01,
        s: float = 0.0,
        source: Callable[[Array, float], Array] | None = None,
        left: BC | None = None,
        right: BC | None = None,
        x0: float = 0.0,
        x1: float = 1.0,
    ):
        self.N, self.a, self.D, self.s, self.source = N, float(a), float(D), float(s), source
        self.left, self.right = left or dirichlet(0.0), right or dirichlet(0.0)
        self.D1, self.x = cheb_mapped(N, x0, x1)
        self.D2 = self.D1 @ self.D1
        n1 = N + 1
        self.L = self.D * self.D2 - self.a * self.D1 - self.s * np.eye(n1)
        # boundary rows: alpha e_j + beta (n . D1)_j ; outward normal -x at x0, +x at x1
        self.B = np.zeros((2, n1))
        self.B[0] = self.left.beta * (-self.D1[0])
        self.B[0, 0] += self.left.alpha
        self.B[1] = self.right.beta * self.D1[N]
        self.B[1, N] += self.right.alpha
        self.idx = np.arange(1, N)  # interior unknowns
        self.x_active = self.x[self.idx]
        bnd = [0, N]
        Bbb, Bbi = self.B[:, bnd], self.B[:, self.idx]
        self._Bbb_inv = np.linalg.inv(Bbb)
        # theta_full = P u + E g
        self.P = np.zeros((n1, N - 1))
        self.P[self.idx, np.arange(N - 1)] = 1.0
        self.P[bnd] = -self._Bbb_inv @ Bbi
        self.E = np.zeros((n1, 2))
        self.E[bnd] = self._Bbb_inv
        self._A = (self.L @ self.P)[self.idx]
        self._Ab = (self.L @ self.E)[self.idx]

    def g(self, t: float) -> Array:
        return np.array([self.left.value(t), self.right.value(t)])

    @property
    def A(self) -> sps.csr_matrix:
        return sps.csr_matrix(self._A)

    @property
    def A_dense(self) -> Array:
        return self._A

    def b(self, t: float) -> Array:
        out = self._Ab @ self.g(t)
        if self.source is not None:
            out = out + self.source(self.x_active, t)
        return out

    def rhs(self, y: Array, t: float) -> Array:
        return self._A @ y + self.b(t)

    def nodal(self, y: Array, t: float) -> Array:
        return self.P @ y + self.E @ self.g(t)

    def steady(self, t: float = 0.0) -> Array:
        """Row-replacement solve of L theta = -Q with BC rows in place of rows 0 and N (returns interior)."""
        M = self.L.copy()
        rhs = -(self.source(self.x, t) if self.source is not None else np.zeros(self.N + 1))
        M[0], M[self.N] = self.B[0], self.B[1]
        rhs[0], rhs[self.N] = self.g(t)
        return np.linalg.solve(M, rhs)[self.idx]
