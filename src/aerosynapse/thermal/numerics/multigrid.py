"""Geometric multigrid (V-cycle) for the cell-centred Helmholtz/Poisson operator

    M x = (c I - D L_h) x = f,     L_h: 5-point FV Laplacian with per-side alpha/beta BCs (homogeneous)

Components: red-black Gauss–Seidel smoothing (symmetric ordering: RB pre, BR post, so the V-cycle is a
symmetric preconditioner for CG), full-weighting-by-averaging restriction (2x2 cell average), bilinear
cell-centred prolongation (9/16, 3/16, 3/16, 1/16) with odd ghosts at Dirichlet sides and even ghosts
elsewhere, rediscretised coarse operators, sparse-LU coarsest solve. The pure-Neumann Poisson case
(c = 0, all beta != 0, alpha = 0) is singular: the RHS is projected onto the range (zero mean) and the
solution is fixed to zero mean (gauge), as needed for the pressure Poisson equation in Step 1.6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from aerosynapse.thermal.numerics.fvm2d import SIDES, FaceBC, FVEnergy2D, Grid2D

Array = NDArray[np.float64]


def laplacian_fv(grid: Grid2D, bcs: dict[str, FaceBC]) -> sps.csr_matrix:
    """5-point FV Laplacian L_h with homogeneous versions of the given BCs."""
    hb = {k: v.homogeneous() for k, v in bcs.items()}
    return FVEnergy2D(grid, D=1.0, bcs=hb, scheme="central").A


@dataclass
class _Level:
    grid: Grid2D
    M: sps.csr_matrix
    diag: Array
    red: Array
    black: Array
    M_rb: sps.csr_matrix  # rows red, cols black
    M_br: sps.csr_matrix


class MultigridHelmholtz:
    def __init__(
        self, grid: Grid2D, c: float, D: float, bcs: dict[str, FaceBC], min_cells: int = 4, nu1: int = 2, nu2: int = 2
    ):
        self.c, self.D, self.nu1, self.nu2 = float(c), float(D), nu1, nu2
        self.bcs = dict(bcs)
        self.dirichlet_side = {k: bcs[k].is_dirichlet for k in SIDES}
        self.singular = c == 0.0 and all(bcs[k].alpha == 0.0 for k in SIDES)
        self.levels: list[_Level] = []
        g = grid
        while True:
            self.levels.append(self._make_level(g))
            if g.nx % 2 or g.ny % 2 or min(g.nx, g.ny) // 2 < min_cells:
                break
            g = Grid2D(g.nx // 2, g.ny // 2, g.Lx, g.Ly)
        Mc = self.levels[-1].M
        if self.singular:
            self._coarse_pinv = np.linalg.pinv(Mc.toarray())
        else:
            self._coarse_lu = spla.splu(sps.csc_matrix(Mc))

    def _make_level(self, g: Grid2D) -> _Level:
        M = sps.csr_matrix(self.c * sps.identity(g.n) - self.D * laplacian_fv(g, self.bcs))
        J, I = np.meshgrid(np.arange(g.ny), np.arange(g.nx), indexing="ij")
        colour = ((I + J) % 2).ravel()
        red, black = np.flatnonzero(colour == 0), np.flatnonzero(colour == 1)
        return _Level(g, M, M.diagonal().copy(), red, black, M[red][:, black], M[black][:, red])

    # --- components -------------------------------------------------------------------------
    def _smooth(self, L: _Level, x: Array, f: Array, order: str) -> None:
        for col in order:
            if col == "r":
                x[L.red] = (f[L.red] - L.M_rb @ x[L.black]) / L.diag[L.red]
            else:
                x[L.black] = (f[L.black] - L.M_br @ x[L.red]) / L.diag[L.black]

    @staticmethod
    def restrict(r: Array, g: Grid2D) -> Array:
        R = r.reshape(g.ny, g.nx)
        return 0.25 * (R[0::2, 0::2] + R[1::2, 0::2] + R[0::2, 1::2] + R[1::2, 1::2]).ravel()

    def prolong(self, e: Array, gc: Grid2D) -> Array:
        E = e.reshape(gc.ny, gc.nx)
        sgn = {k: (-1.0 if self.dirichlet_side[k] else 1.0) for k in SIDES}
        P = np.zeros((gc.ny + 2, gc.nx + 2))
        P[1:-1, 1:-1] = E
        P[1:-1, 0], P[1:-1, -1] = sgn["left"] * E[:, 0], sgn["right"] * E[:, -1]
        P[0, :], P[-1, :] = sgn["bottom"] * P[1, :], sgn["top"] * P[-2, :]
        F = np.empty((2 * gc.ny, 2 * gc.nx))
        ny, nx = gc.ny, gc.nx

        def sh(sx: int, sy: int) -> Array:
            return P[1 + sy : ny + 1 + sy, 1 + sx : nx + 1 + sx]

        for a, sx in ((0, -1), (1, 1)):
            for b, sy in ((0, -1), (1, 1)):
                F[b::2, a::2] = (9 * sh(0, 0) + 3 * sh(sx, 0) + 3 * sh(0, sy) + sh(sx, sy)) / 16.0
        return F.ravel()

    def _coarse_solve(self, f: Array) -> Array:
        if self.singular:
            return self._coarse_pinv @ (f - f.mean())
        return self._coarse_lu.solve(f)

    def vcycle(self, x: Array, f: Array, lev: int = 0) -> Array:
        L = self.levels[lev]
        if lev == len(self.levels) - 1:
            return self._coarse_solve(f)
        for _ in range(self.nu1):
            self._smooth(L, x, f, "rb")
        r = f - L.M @ x
        gc = self.levels[lev + 1].grid
        ec = self.vcycle(np.zeros(gc.n), self.restrict(r, L.grid), lev + 1)
        x += self.prolong(ec, gc)
        for _ in range(self.nu2):
            self._smooth(L, x, f, "br")
        return x

    # --- drivers ----------------------------------------------------------------------------
    @property
    def M(self) -> sps.csr_matrix:
        return self.levels[0].M

    def solve(
        self, f: Array, x0: Array | None = None, rtol: float = 1e-10, maxiter: int = 100
    ) -> tuple[Array, list[float]]:
        """Stand-alone V-cycle iteration. Returns (x, relative residual history ||r_k||/||f||)."""
        f = np.asarray(f, float)
        if self.singular:
            f = f - f.mean()
        x = np.zeros_like(f) if x0 is None else np.array(x0, float)
        nf = max(np.linalg.norm(f), 1e-300)
        hist = [float(np.linalg.norm(f - self.M @ x) / nf)]
        for _ in range(maxiter):
            x = self.vcycle(x, f)
            if self.singular:
                x -= x.mean()
            hist.append(float(np.linalg.norm(f - self.M @ x) / nf))
            if hist[-1] < rtol:
                break
        return x, hist

    def precondition(self, r: Array) -> Array:
        """One V-cycle from zero: symmetric positive definite approximation of M^{-1}."""
        z = self.vcycle(np.zeros_like(r), r - r.mean() if self.singular else r)
        return z - z.mean() if self.singular else z
