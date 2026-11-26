"""2D cell-centred finite-volume solver for the energy equation on a uniform Cartesian grid:

    theta_t + div(u theta) = D lap(theta) - s theta + Q(x, y, t)        (div u = 0)

Grid: nx x ny cells on [0, Lx] x [0, Ly]; unknowns theta[j, i] (row j = y index), flattened row-major
k = j*nx + i. Face velocities live on MAC-staggered faces (same layout as Step 1.6):
    U: (ny, nx+1) normal velocity on vertical faces x_{i-1/2};  V: (ny+1, nx) on horizontal faces y_{j-1/2}.
BCs per side ('left','right','bottom','top'):  alpha theta + beta dtheta/dn = g(s, t), n outward, s = the
coordinate along the side. The boundary value theta_b is eliminated with the half-cell gradient
    dtheta/dn ~ (theta_b - theta_P)/(h/2)  =>  theta_b = (g + 2 beta/h theta_P) / (alpha + 2 beta/h),
and 2 ghost layers theta_g = 2 theta_b - theta_(interior mirror) serve the wide/limited stencils.
Convective face values reuse :func:`fd1d.face_values` (upwind1, central, upwind2, quick, tvd_*).
Every face flux is shared by the two adjacent cells, so the scheme is conservative to round-off.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from aerosynapse.thermal.numerics.fd1d import LINEAR_SCHEMES, SCHEMES, face_values

Array = NDArray[np.float64]
SIDES = ("left", "right", "bottom", "top")


@dataclass(frozen=True)
class FaceBC:
    """alpha theta + beta dtheta/dn = g(s, t) on one side (s: coordinate along the side)."""

    alpha: float
    beta: float
    g: float | Callable[[Array, float], Array] = 0.0

    def value(self, s: Array, t: float) -> Array:
        if callable(self.g):
            return np.broadcast_to(np.asarray(self.g(s, t), dtype=float), s.shape)
        return np.full(s.shape, float(self.g))

    @property
    def is_dirichlet(self) -> bool:
        return self.beta == 0.0

    def homogeneous(self) -> FaceBC:
        return FaceBC(self.alpha, self.beta, 0.0)


def dirichlet2(g: float | Callable[[Array, float], Array]) -> FaceBC:
    return FaceBC(1.0, 0.0, g)


def neumann2(g: float | Callable[[Array, float], Array]) -> FaceBC:
    return FaceBC(0.0, 1.0, g)


def robin2(Bi: float, theta_inf: float = 0.0) -> FaceBC:
    """Newton cooling -dtheta/dn = Bi (theta - theta_inf)."""
    return FaceBC(Bi, 1.0, Bi * theta_inf)


def bcs_from_mms(
    ms: object, spec: dict[str, tuple[float, float]], Lx: float = 1.0, Ly: float = 1.0
) -> dict[str, FaceBC]:
    """FaceBCs whose data g = alpha theta + beta dtheta/dn come from a 2D ManufacturedSolution.

    spec: side -> (alpha, beta).
    """
    nrm = {"left": (-1, 0), "right": (1, 0), "bottom": (0, -1), "top": (0, 1)}
    pos = {
        "left": lambda s: (0 * s, s),
        "right": lambda s: (Lx + 0 * s, s),
        "bottom": lambda s: (s, 0 * s),
        "top": lambda s: (s, Ly + 0 * s),
    }
    out = {}
    for side, (a, b) in spec.items():
        gf = ms.robin_data(nrm[side], a, b)  # type: ignore[attr-defined]
        out[side] = FaceBC(a, b, lambda s, t, gf=gf, p=pos[side]: gf(*p(s), t + 0 * s))
    return out


@dataclass
class Grid2D:
    nx: int
    ny: int
    Lx: float = 1.0
    Ly: float = 1.0

    def __post_init__(self) -> None:
        self.dx, self.dy = self.Lx / self.nx, self.Ly / self.ny
        self.xc = (np.arange(self.nx) + 0.5) * self.dx
        self.yc = (np.arange(self.ny) + 0.5) * self.dy
        self.xf = np.arange(self.nx + 1) * self.dx
        self.yf = np.arange(self.ny + 1) * self.dy
        self.Xc, self.Yc = np.meshgrid(self.xc, self.yc)  # (ny, nx)
        self.vol = self.dx * self.dy
        self.n = self.nx * self.ny


class FaceVelocity:
    """Normal velocities on MAC faces. ``steady`` velocities allow a constant implicit operator."""

    def __init__(self, grid: Grid2D, fn: Callable[[float], tuple[Array, Array]], steady: bool = True):
        self.grid, self._fn, self.steady = grid, fn, steady
        self._cache: tuple[Array, Array] | None = None

    def at(self, t: float) -> tuple[Array, Array]:
        if self.steady:
            if self._cache is None:
                self._cache = self._fn(0.0)
            return self._cache
        return self._fn(t)

    @classmethod
    def zero(cls, grid: Grid2D) -> FaceVelocity:
        return cls(grid, lambda t: (np.zeros((grid.ny, grid.nx + 1)), np.zeros((grid.ny + 1, grid.nx))))

    @classmethod
    def from_arrays(cls, grid: Grid2D, U: Array, V: Array) -> FaceVelocity:
        return cls(grid, lambda t: (np.asarray(U, float), np.asarray(V, float)))

    @classmethod
    def from_streamfunction(
        cls, grid: Grid2D, psi: Callable[[Array, Array, float], Array], steady: bool = True
    ) -> FaceVelocity:
        """U = (psi(top corner) - psi(bottom corner))/dy, V = -(psi(right) - psi(left))/dx.
        Discretely divergence-free to round-off: sum of face fluxes of every cell telescopes to 0."""
        g = grid

        def fn(t: float) -> tuple[Array, Array]:
            Xn, Yn = np.meshgrid(g.xf, g.yf)  # cell corners (ny+1, nx+1)
            P = psi(Xn, Yn, t)
            U = (P[1:, :] - P[:-1, :]) / g.dy
            V = -(P[:, 1:] - P[:, :-1]) / g.dx
            return U, V

        return cls(grid, fn, steady)

    @classmethod
    def from_point_function(
        cls, grid: Grid2D, vel: Callable[[Array, Array, float], tuple[Array, Array]], steady: bool = True
    ) -> FaceVelocity:
        g = grid

        def fn(t: float) -> tuple[Array, Array]:
            Xu, Yu = np.meshgrid(g.xf, g.yc)
            Xv, Yv = np.meshgrid(g.xc, g.yf)
            return np.asarray(vel(Xu, Yu, t)[0], float), np.asarray(vel(Xv, Yv, t)[1], float)

        return cls(grid, fn, steady)


def discrete_divergence(grid: Grid2D, U: Array, V: Array) -> Array:
    return (U[:, 1:] - U[:, :-1]) / grid.dx + (V[1:, :] - V[:-1, :]) / grid.dy


class FVEnergy2D:
    """Semi-discrete FV operator with the same interface as the 1D solvers (rhs, A, b, x_active...)."""

    def __init__(
        self,
        grid: Grid2D,
        D: float,
        velocity: FaceVelocity | None = None,
        s: float = 0.0,
        source: Callable[[Array, Array, float], Array] | None = None,
        bcs: dict[str, FaceBC] | None = None,
        scheme: str = "central",
        bc_order: int = 1,
    ):
        """bc_order=1: half-cell boundary gradient (theta_1 - theta_b)/(h/2): solution O(h^2) but wall
        gradients / Nu only O(h). bc_order=2: one-sided (-8 theta_b + 9 theta_1 - theta_2)/(3h) for both the
        BC elimination and the boundary diffusive flux: wall gradients O(h^2). Both keep a 3-point stencil per
        grid line (ADI-compatible) and the M-matrix sign pattern of the diffusion operator (checked for
        Dirichlet and Robin). Added in Step 1.7 after the wall-gradient order check; default 1 keeps earlier
        results."""
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")
        if bc_order not in (1, 2):
            raise ValueError("bc_order must be 1 or 2")
        self.g, self.D, self.s, self.source, self.scheme = grid, float(D), float(s), source, scheme
        self.bc_order = bc_order
        self.vel = velocity or FaceVelocity.zero(grid)
        self.bcs = {k: dirichlet2(0.0) for k in SIDES} | (bcs or {})
        self.idx = np.arange(grid.n)
        self._A: sps.csr_matrix | None = None
        self._split: tuple[sps.csr_matrix, sps.csr_matrix] | None = None

    # --- boundary values & ghosts -----------------------------------------------------------
    def _near_wall(self, th: Array) -> dict[str, tuple[Array, Array, Array, float]]:
        g = self.g
        return {
            "left": (th[:, 0], th[:, min(1, g.nx - 1)], g.yc, g.dx),
            "right": (th[:, -1], th[:, max(-2, -g.nx)], g.yc, g.dx),
            "bottom": (th[0, :], th[min(1, g.ny - 1), :], g.xc, g.dy),
            "top": (th[-1, :], th[max(-2, -g.ny), :], g.xc, g.dy),
        }

    def boundary_values(self, th: Array, t: float) -> dict[str, Array]:
        out = {}
        for side, (P1, P2, sc, h) in self._near_wall(th).items():
            bc = self.bcs[side]
            if self.bc_order == 1:  # dtheta/dn_out = (theta_b - theta_1)/(h/2)
                k = 2.0 * bc.beta / h
                out[side] = (bc.value(sc, t) + k * P1) / (bc.alpha + k)
            else:  # dtheta/dn_out = (8 theta_b - 9 theta_1 + theta_2)/(3h)
                k = bc.beta / (3.0 * h)
                out[side] = (bc.value(sc, t) + k * (9 * P1 - P2)) / (bc.alpha + 8 * k)
        return out

    def padded(self, th: Array, tb: dict[str, Array]) -> Array:
        """(ny+4, nx+4) array with two odd-reflection ghost layers about the boundary values."""
        ny, nx = th.shape
        T = np.zeros((ny + 4, nx + 4))
        T[2:-2, 2:-2] = th
        T[2:-2, 1] = 2 * tb["left"] - th[:, 0]
        T[2:-2, 0] = 2 * tb["left"] - th[:, min(1, nx - 1)]
        T[2:-2, -2] = 2 * tb["right"] - th[:, -1]
        T[2:-2, -1] = 2 * tb["right"] - th[:, max(-2, -nx)]
        T[1, 2:-2] = 2 * tb["bottom"] - th[0, :]
        T[0, 2:-2] = 2 * tb["bottom"] - th[min(1, ny - 1), :]
        T[-2, 2:-2] = 2 * tb["top"] - th[-1, :]
        T[-1, 2:-2] = 2 * tb["top"] - th[max(-2, -ny), :]
        return T

    # --- fluxes -----------------------------------------------------------------------------
    def _conv_face(self, L: Array, LL: Array, R: Array, RR: Array, un: Array) -> Array:
        up = face_values(L, LL, R, self.scheme)
        dn = face_values(R, RR, L, self.scheme)
        return np.where(un >= 0, up, dn)

    def flux_parts(self, y: Array, t: float, include_data: bool = True) -> tuple[Array, Array]:
        """Return (Rx, Ry): x- and y-direction flux divergences (conv + diff) per cell, shape (ny, nx)."""
        fx, fy = self.face_fluxes(y, t, include_data)
        return (fx[:, 1:] - fx[:, :-1]) / self.g.dx, (fy[1:, :] - fy[:-1, :]) / self.g.dy

    def face_fluxes(self, y: Array, t: float, include_data: bool = True) -> tuple[Array, Array]:
        """(fx, fy) = MINUS the physical total flux (u theta - D grad theta) through x-faces (ny, nx+1)
        in +x and y-faces (ny+1, nx) in +y, so that d theta/dt = div_h(f) + ..."""
        g = self.g
        th = y.reshape(g.ny, g.nx)
        tb = self.boundary_values(th, t) if include_data else self._homog_bvals(th)
        T = self.padded(th, tb)
        U, V = self.vel.at(t)
        # x-faces (ny, nx+1)
        rows = slice(2, -2)
        Lx, LLx = T[rows, 1:-2], T[rows, 0:-3]
        Rx, RRx = T[rows, 2:-1], T[rows, 3:]
        Fx = self._conv_face(Lx, LLx, Rx, RRx, U)
        Fx[:, 0], Fx[:, -1] = tb["left"], tb["right"]
        Gx = (Rx - Lx) / g.dx  # +x gradient (boundary faces: half-cell via ghost)
        if self.bc_order == 2:  # one-sided second-order boundary gradients (consistent with boundary_values)
            Gx[:, 0] = (-8 * tb["left"] + 9 * th[:, 0] - th[:, min(1, g.nx - 1)]) / (3 * g.dx)
            Gx[:, -1] = (8 * tb["right"] - 9 * th[:, -1] + th[:, max(-2, -g.nx)]) / (3 * g.dx)
        fx = -U * Fx + self.D * Gx
        # y-faces (ny+1, nx)
        cols = slice(2, -2)
        Ly_, LLy = T[1:-2, cols], T[0:-3, cols]
        Ry_, RRy = T[2:-1, cols], T[3:, cols]
        Fy = self._conv_face(Ly_, LLy, Ry_, RRy, V)
        Fy[0, :], Fy[-1, :] = tb["bottom"], tb["top"]
        Gy = (Ry_ - Ly_) / g.dy
        if self.bc_order == 2:
            Gy[0, :] = (-8 * tb["bottom"] + 9 * th[0, :] - th[min(1, g.ny - 1), :]) / (3 * g.dy)
            Gy[-1, :] = (8 * tb["top"] - 9 * th[-1, :] + th[max(-2, -g.ny), :]) / (3 * g.dy)
        fy = -V * Fy + self.D * Gy
        return fx, fy

    def _homog_bvals(self, th: Array) -> dict[str, Array]:
        saved = self.bcs
        self.bcs = {k: v.homogeneous() for k, v in saved.items()}
        try:
            return self.boundary_values(th, 0.0)
        finally:
            self.bcs = saved

    def rhs(self, y: Array, t: float) -> Array:
        Rx, Ry = self.flux_parts(y, t)
        r = (Rx + Ry).ravel() - self.s * y
        if self.source is not None:
            r = r + self.source(self.g.Xc, self.g.Yc, t).ravel()
        return r

    def b(self, t: float) -> Array:
        return self.rhs(np.zeros(self.g.n), t)

    # --- linear operators (probing) ---------------------------------------------------------
    def _probe(self, fn: Callable[[Array], Array]) -> sps.csr_matrix:
        """Recover a sparse linear map whose stencil is a cross of radius <= 2 (13 coloured probes)."""
        g, K = self.g, 13
        J, I = np.meshgrid(np.arange(g.ny), np.arange(g.nx), indexing="ij")
        color = ((I + 3 * J) % K).ravel()
        offs = [(0, 0), (1, 0), (-1, 0), (2, 0), (-2, 0), (0, 1), (0, -1), (0, 2), (0, -2)]
        rows, cols, vals = [], [], []
        for c in range(K):
            v = (color == c).astype(float)
            Av = fn(v)
            for di, dj in offs:
                # column of colour c at offset (di, dj) from each row cell
                ii, jj = I.ravel() + di, J.ravel() + dj
                ok = (ii >= 0) & (ii < g.nx) & (jj >= 0) & (jj < g.ny)
                col = jj[ok] * g.nx + ii[ok]
                sel = color[col] == c
                r = np.flatnonzero(ok)[sel]
                rows.append(r)
                cols.append(col[sel])
                vals.append(Av[r])
        A = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(g.n, g.n))
        A.sum_duplicates()
        A.eliminate_zeros()
        return A

    def _check_linear(self) -> None:
        if self.scheme not in LINEAR_SCHEMES:
            raise ValueError(f"scheme {self.scheme} is nonlinear; use an explicit integrator")
        if not self.vel.steady:
            raise ValueError("implicit operators need a steady velocity field")

    @property
    def A(self) -> sps.csr_matrix:
        if self._A is None:
            self._check_linear()
            A = self._probe(lambda v: self._homog_rhs(v))
            y = np.random.default_rng(0).standard_normal(self.g.n)
            err = np.max(np.abs(A @ y + self.b(0.0) - self.rhs(y, 0.0)))
            if err > 1e-8 * max(1.0, np.abs(A.data).max()):
                raise RuntimeError(f"probing failed (err {err:.2e})")
            self._A = A
        return self._A

    def _homog_rhs(self, v: Array) -> Array:
        Rx, Ry = self.flux_parts(v, 0.0, include_data=False)
        return (Rx + Ry).ravel() - self.s * v

    def split_operators(self) -> tuple[sps.csr_matrix, sps.csr_matrix]:
        """(A_x, A_y) with A = A_x + A_y; reaction split equally. Used by ADI."""
        if self._split is None:
            self._check_linear()
            Ax = self._probe(lambda v: self.flux_parts(v, 0.0, False)[0].ravel() - 0.5 * self.s * v)
            Ay = self._probe(lambda v: self.flux_parts(v, 0.0, False)[1].ravel() - 0.5 * self.s * v)
            self._split = (Ax, Ay)
        return self._split

    def steady(self, t: float = 0.0) -> Array:
        return np.asarray(spla.spsolve(sps.csc_matrix(self.A), -self.b(t)))

    # --- diagnostics ------------------------------------------------------------------------
    def boundary_heat_flow(self, y: Array, t: float) -> dict[str, float]:
        """Net heat INTO the domain through each side (convective + diffusive), from the scheme's own
        boundary face fluxes: inflow = physical flux . n_inward."""
        g = self.g
        fx, fy = self.face_fluxes(y, t)
        return {
            "left": float(-fx[:, 0].sum() * g.dy),
            "right": float(fx[:, -1].sum() * g.dy),
            "bottom": float(-fy[0, :].sum() * g.dx),
            "top": float(fy[-1, :].sum() * g.dx),
        }

    def wall_diffusive_flux(self, y: Array, t: float, side: str) -> Array:
        """Local -dtheta/dn_out on a side using the SCHEME'S OWN boundary gradient (half-cell for bc_order=1,
        one-sided second order for bc_order=2) — i.e. the conservative wall flux / D."""
        if self.bc_order == 2:
            return self.wall_gradient(y, t, side)
        g = self.g
        th = y.reshape(g.ny, g.nx)
        tb = self.boundary_values(th, t)[side]
        P, h = {
            "left": (th[:, 0], g.dx),
            "right": (th[:, -1], g.dx),
            "bottom": (th[0, :], g.dy),
            "top": (th[-1, :], g.dy),
        }[side]
        return -(tb - P) / (0.5 * h)

    def wall_gradient(self, y: Array, t: float, side: str) -> Array:
        """One-sided d(theta)/dn_in (inward normal) from theta_b, theta_1, theta_2 at distances 0, h/2, 3h/2:
        (-8 theta_b + 9 theta_1 - theta_2) / (3h). NOTE (Step 1.7): with bc_order=1 the discrete solution has
        an O(h^2) error at the first cell that does not vanish at the wall, so ANY wall derivative of it is
        only O(h); use bc_order=2 for second-order wall gradients / Nusselt numbers."""
        g = self.g
        th = y.reshape(g.ny, g.nx)
        tb = self.boundary_values(th, t)[side]
        P1, P2, h = {
            "left": (th[:, 0], th[:, 1], g.dx),
            "right": (th[:, -1], th[:, -2], g.dx),
            "bottom": (th[0, :], th[1, :], g.dy),
            "top": (th[-1, :], th[-2, :], g.dy),
        }[side]
        return (-8 * tb + 9 * P1 - P2) / (3 * h)

    def energy_balance(self, y: Array, t: float) -> dict[str, float]:
        """Steady balance: boundary inflow + int Q - int s theta (= int theta_t = 0 at steady state)."""
        g = self.g
        q = self.boundary_heat_flow(y, t)
        src = float(np.sum(self.source(g.Xc, g.Yc, t)) * g.vol) if self.source is not None else 0.0
        sink = float(self.s * np.sum(y) * g.vol)
        storage = float(np.sum(self.rhs(y, t)) * g.vol)  # = d/dt int theta
        total_in = sum(q.values()) + src - sink
        return {
            **{f"q_{k}": v for k, v in q.items()},
            "source": src,
            "sink": sink,
            "storage_rate": storage,
            "residual": total_in - storage,
            "scale": sum(abs(v) for v in q.values()) + abs(src) + abs(sink),
        }
