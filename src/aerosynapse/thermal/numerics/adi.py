"""Douglas–Gunn ADI (approximate factorisation of Crank–Nicolson) in delta form.

    (I - dt/2 A_x) D*  = dt (A y^n + (b^n + b^{n+1})/2)
    (I - dt/2 A_y) D   = D*
    y^{n+1} = y^n + D
Expanding: (I - dt/2 A) D + dt^2/4 A_x A_y D = dt(A y^n + b^{n+1/2}), i.e. CN plus an O(dt^3) term
(D = O(dt)), so the scheme is second order in time and unconditionally stable for the diffusion
part. Each sweep is a batch of independent tridiagonal systems (one per grid line) solved with the
batched Thomas algorithm. Requires a 3-point stencil per direction (upwind1 / central).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from aerosynapse.thermal.numerics.fvm2d import FVEnergy2D
from aerosynapse.thermal.numerics.linalg import ThomasFactor

Array = NDArray[np.float64]


def _line_factors(M: sps.csr_matrix, nx: int, ny: int, direction: str) -> ThomasFactor:
    n = nx * ny
    off = 1 if direction == "x" else nx
    allowed = {0, off, -off}
    C = sps.coo_matrix(M)
    bad = set(np.unique(C.col - C.row).tolist()) - allowed
    if bad:
        raise ValueError(f"operator is not tridiagonal along {direction}-lines (offsets {sorted(bad)})")
    lo, up = np.zeros(n), np.zeros(n)
    lo[off:] = M.diagonal(-off)
    up[:-off] = M.diagonal(off)
    di = M.diagonal().copy()
    shape = (ny, nx)
    lo, di, up = (v.reshape(shape) for v in (lo, di, up))
    if direction == "y":
        lo, di, up = lo.T.copy(), di.T.copy(), up.T.copy()  # lines along j: (nx, ny)
    return ThomasFactor(lo, di, up)


class DouglasGunnADI:
    def __init__(self, prob: FVEnergy2D, dt: float):
        g = prob.g
        self.prob, self.dt, self.nx, self.ny = prob, dt, g.nx, g.ny
        Ax, Ay = prob.split_operators()
        I = sps.identity(g.n, format="csr")
        self.fx = _line_factors(sps.csr_matrix(I - 0.5 * dt * Ax), g.nx, g.ny, "x")
        self.fy = _line_factors(sps.csr_matrix(I - 0.5 * dt * Ay), g.nx, g.ny, "y")
        self.A = prob.A

    def step(self, y: Array, t: float) -> Array:
        dt, nx, ny = self.dt, self.nx, self.ny
        r = dt * (self.A @ y + 0.5 * (self.prob.b(t) + self.prob.b(t + dt)))
        dstar = self.fx.solve(r.reshape(ny, nx))
        d = self.fy.solve(dstar.T.copy()).T
        return y + d.ravel()


def integrate_adi(
    prob: FVEnergy2D,
    y0: Array,
    t0: float,
    t1: float,
    dt: float,
    callback: Callable[[int, float, Array], None] | None = None,
) -> Array:
    nsteps = max(1, int(np.ceil((t1 - t0) / dt - 1e-12)))
    dt = (t1 - t0) / nsteps
    stepper = DouglasGunnADI(prob, dt)
    y = np.array(y0, dtype=float)
    for n in range(nsteps):
        y = stepper.step(y, t0 + n * dt)
        if callback:
            callback(n + 1, t0 + (n + 1) * dt, y)
    return y
