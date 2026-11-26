"""1D finite-difference convection-diffusion-reaction solver on a uniform vertex grid.

    theta_t + a theta_x = D theta_xx - s theta + Q(x, t),   x in [x0, x1]
    BC at each end:  alpha*theta + beta*d(theta)/dn = g(t)   (n = outward normal, D9)

Convection is written in flux form  a (theta_{i+1/2} - theta_{i-1/2}) / h  with face values:
    upwind1  theta_U                                   (1st order)
    central  (theta_U + theta_D)/2                     (2nd order; oscillates if Pe_h = |a|h/D > 2)
    upwind2  3/2 theta_U - 1/2 theta_UU                (2nd order, linear upwind)
    quick    3/4 theta_U + 3/8 theta_D - 1/8 theta_UU  (Leonard QUICK; 2nd order as an FD derivative)
    tvd_<l>  theta_U + 1/2 psi(r)(theta_D - theta_U),  r = (theta_U - theta_UU)/(theta_D - theta_U)
             limiters l in {minmod, vanleer, superbee, vanalbada}  (nonlinear -> explicit integrators)
Boundary treatment: Dirichlet nodes are eliminated from the unknown vector; Neumann/Robin use a
second-order central ghost node; extra ghosts needed by wide stencils use quadratic extrapolation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

Array = NDArray[np.float64]
SourceFn = Callable[[Array, float], Array]

LINEAR_SCHEMES = ("upwind1", "central", "upwind2", "quick")
LIMITERS: dict[str, Callable[[Array], Array]] = {
    "minmod": lambda r: np.maximum(0.0, np.minimum(1.0, r)),
    "vanleer": lambda r: (r + np.abs(r)) / (1.0 + np.abs(r)),
    "superbee": lambda r: np.maximum.reduce([np.zeros_like(r), np.minimum(2 * r, 1.0), np.minimum(r, 2.0)]),
    "vanalbada": lambda r: np.where(r > 0, (r * r + r) / (r * r + 1.0), 0.0),
}
SCHEMES = LINEAR_SCHEMES + tuple(f"tvd_{k}" for k in LIMITERS)


@dataclass(frozen=True)
class BC:
    """alpha*theta + beta*dtheta/dn = g(t) on one boundary (n outward)."""

    alpha: float
    beta: float
    g: float | Callable[[float], float] = 0.0

    def value(self, t: float) -> float:
        return float(self.g(t)) if callable(self.g) else float(self.g)

    @property
    def is_dirichlet(self) -> bool:
        return self.beta == 0.0


def dirichlet(g: float | Callable[[float], float]) -> BC:
    return BC(1.0, 0.0, g)


def neumann(g: float | Callable[[float], float]) -> BC:
    """dtheta/dn = g (outward normal)."""
    return BC(0.0, 1.0, g)


def robin(Bi: float, theta_inf: float = 0.0) -> BC:
    """Newton cooling  -dtheta/dn = Bi (theta - theta_inf)  <=>  Bi theta + dtheta/dn = Bi theta_inf."""
    return BC(Bi, 1.0, Bi * theta_inf)


def face_values(U: Array, UU: Array, Dn: Array, scheme: str) -> Array:
    """Face reconstruction from upwind (U), far-upwind (UU) and downwind (Dn) nodal values."""
    if scheme == "upwind1":
        return U.copy()
    if scheme == "central":
        return 0.5 * (U + Dn)
    if scheme == "upwind2":
        return 1.5 * U - 0.5 * UU
    if scheme == "quick":
        return 0.75 * U + 0.375 * Dn - 0.125 * UU
    if scheme.startswith("tvd_"):
        psi = LIMITERS[scheme[4:]]
        num, den = U - UU, Dn - U
        flat = den == 0.0
        # psi(r) * den = 0 whenever den = 0 for any bounded limiter, so r is irrelevant there;
        # set r = 0 (avoids r ~ 1e300 overflow -> inf/inf = NaN in e.g. van Albada)
        r = np.where(flat, 0.0, num / np.where(flat, 1.0, den))
        return U + 0.5 * psi(r) * den
    raise ValueError(f"unknown scheme {scheme!r}")


class ConvDiff1D:
    """Semi-discretisation d(theta)/dt = A theta + b(t) (or nonlinear rhs for TVD) on active nodes."""

    def __init__(
        self,
        N: int,
        a: float = 1.0,
        D: float = 0.01,
        s: float = 0.0,
        source: SourceFn | None = None,
        left: BC | None = None,
        right: BC | None = None,
        scheme: str = "central",
        x0: float = 0.0,
        x1: float = 1.0,
    ):
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")
        self.N, self.a, self.D, self.s, self.scheme = N, float(a), float(D), float(s), scheme
        self.source = source
        self.left = left or dirichlet(0.0)
        self.right = right or dirichlet(0.0)
        self.x = np.linspace(x0, x1, N + 1)
        self.h = (x1 - x0) / N
        act = np.ones(N + 1, dtype=bool)
        act[0] = not self.left.is_dirichlet
        act[-1] = not self.right.is_dirichlet
        self.active = act
        self.idx = np.flatnonzero(act)
        self.x_active = self.x[self.idx]
        self._A: sps.csr_matrix | None = None

    # --- properties -------------------------------------------------------------------------
    @property
    def is_linear(self) -> bool:
        return self.scheme in LINEAR_SCHEMES

    @property
    def cell_peclet(self) -> float:
        return abs(self.a) * self.h / self.D if self.D > 0 else np.inf

    # --- assembly ---------------------------------------------------------------------------
    def expand(self, y: Array, t: float) -> Array:
        """Active-node vector -> nodal array with two ghost nodes on each side (len N+5)."""
        N, h = self.N, self.h
        f = np.zeros(N + 5)
        f[2 + self.idx] = y
        L, R = self.left, self.right
        if L.is_dirichlet:
            f[2] = L.value(t) / L.alpha
            f[1] = 3 * f[2] - 3 * f[3] + f[4]
        else:  # -dtheta/dx at x0:  alpha f0 - beta (f1 - f_-1)/(2h) = g
            f[1] = f[3] + 2 * h * (L.value(t) - L.alpha * f[2]) / L.beta
        f[0] = 3 * f[1] - 3 * f[2] + f[3]
        if R.is_dirichlet:
            f[N + 2] = R.value(t) / R.alpha
            f[N + 3] = 3 * f[N + 2] - 3 * f[N + 1] + f[N]
        else:  # +dtheta/dx at x1
            f[N + 3] = f[N + 1] + 2 * h * (R.value(t) - R.alpha * f[N + 2]) / R.beta
        f[N + 4] = 3 * f[N + 3] - 3 * f[N + 2] + f[N + 1]
        return f

    def nodal(self, y: Array, t: float) -> Array:
        """Active vector -> nodal solution including Dirichlet boundary values (len N+1)."""
        return self.expand(y, t)[2 : self.N + 3]

    def rhs_nodal(self, f: Array, t: float) -> Array:
        N, h = self.N, self.h
        if self.a >= 0:
            U, UU, Dn = f[1 : N + 3], f[0 : N + 2], f[2 : N + 4]
        else:
            U, UU, Dn = f[2 : N + 4], f[3 : N + 5], f[1 : N + 3]
        F = face_values(U, UU, Dn, self.scheme)  # faces -1/2 ... N+1/2
        conv = self.a * (F[1:] - F[:-1]) / h
        th = f[2 : N + 3]
        diff = self.D * (f[3 : N + 4] - 2 * th + f[1 : N + 2]) / h**2
        r = -conv + diff - self.s * th
        if self.source is not None:
            r = r + self.source(self.x, t)
        return r

    def rhs(self, y: Array, t: float) -> Array:
        return self.rhs_nodal(self.expand(y, t), t)[self.idx]

    def b(self, t: float) -> Array:
        """Affine part of the (linear) semi-discrete operator: rhs(0, t)."""
        return self.rhs(np.zeros(self.idx.size), t)

    @property
    def A(self) -> sps.csr_matrix:
        """Constant sparse Jacobian of the linear semi-discretisation, built by coloured probing."""
        if not self.is_linear:
            raise ValueError(f"scheme {self.scheme} is nonlinear; use an explicit integrator")
        if self._A is None:
            n, K = self.idx.size, 7  # stencil half-width <= 3
            b0 = self.b(0.0)
            rows, cols, vals = [], [], []
            for c in range(K):
                v = np.zeros(n)
                v[c::K] = 1.0
                Av = self.rhs(v, 0.0) - b0
                for j in range(c, n, K):
                    r = np.arange(max(0, j - 3), min(n, j + 4))
                    rows.append(r)
                    cols.append(np.full(r.size, j))
                    vals.append(Av[r])
            A = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
            A.eliminate_zeros()
            # safety check: A y + b == rhs(y) for a random y
            y = np.random.default_rng(0).standard_normal(n)
            err = np.max(np.abs(A @ y + b0 - self.rhs(y, 0.0)))
            if err > 1e-9 * max(1.0, np.max(np.abs(A.data))):
                raise RuntimeError(f"probing failed (err {err:.2e}); stencil wider than 3?")
            self._A = A
        return self._A

    def steady(self, t: float = 0.0) -> Array:
        """Solve A y = -b(t) (for time-independent data)."""
        return np.asarray(spla.spsolve(sps.csc_matrix(self.A), -self.b(t)))
