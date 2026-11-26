r"""Hybrid numerical–ML models, each paired with its conventional baseline.

1. :func:`coarse_solver_features` + a CNN correction: a coarse first-order FV solve is upsampled and a
   network learns :math:`u_{fine}-I(u_{coarse})` (neural correction). Baseline: the coarse solver alone.
2. :func:`cg_iterations` — "residual correction": a neural prediction is used as the initial guess of
   conjugate gradients, which then drives the PDE residual to tolerance (the classical solver remains
   the authority). Baseline: CG from zero.
3. :func:`kalman_convdiff` — linear Kalman-filter data assimilation for periodic conv-diff with sparse
   noisy sensors. Baseline: free run from the same wrong initial condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from aerosynapse.numerics.convdiff import solve_convdiff_fv
from aerosynapse.numerics.fd1d import assemble_1d
from aerosynapse.physics.bc import Periodic

FloatArr = NDArray[np.float64]


def periodic_upsample(u: FloatArr, factor: int) -> FloatArr:
    """Linear interpolation of cell-centred periodic data onto a ``factor``-times finer grid."""
    nc = u.size
    xc = (np.arange(nc) + 0.5) / nc
    xf = (np.arange(nc * factor) + 0.5) / (nc * factor)
    return np.asarray(np.interp(xf, np.concatenate([xc - 1, xc, xc + 1]), np.tile(u, 3)))


def coarse_solver_features(u0_fine: FloatArr, pe: float, T: float, factor: int = 4) -> FloatArr:
    """Coarse first-order upwind solve on n/factor cells, upsampled to n."""
    n = u0_fine.size
    uc0 = u0_fine.reshape(-1, factor).mean(1)
    xc = (np.arange(n // factor) + 0.5) / (n // factor)
    r = solve_convdiff_fv(
        0, 1, n // factor, 1.0, 1.0 / pe, lambda x: np.interp(x, xc, uc0, period=1.0), T, scheme="upwind", cfl=0.4
    )
    return periodic_upsample(r.u, factor)


def cg_iterations(A: sp.spmatrix, b: FloatArr, x0: FloatArr | None, tol: float = 1e-8) -> tuple[FloatArr, int]:
    """Conjugate gradients on SPD ``A``; returns (solution, iterations to relative residual ``tol``)."""
    it = [0]

    def cb(_: FloatArr) -> None:
        it[0] += 1

    x, info = spla.cg(A, b, x0=x0, rtol=tol, maxiter=10_000, callback=cb)
    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info})")
    return np.asarray(x), it[0]


@dataclass
class KFResult:
    rmse_free: list[float]
    rmse_kf: list[float]
    coverage_2sigma: float
    final_rmse_free: float
    final_rmse_kf: float


def kalman_convdiff(
    pe: float = 50.0,
    n: int = 64,
    steps: int = 100,
    dt: float = 0.005,
    n_sensors: int = 8,
    obs_sigma: float = 0.02,
    model_sigma: float = 1e-3,
    seed: int = 0,
) -> KFResult:
    """Twin experiment: truth from u0 = 1 + 0.5 sin(2πx); background starts from u0 = 1 (wrong)."""
    rng = np.random.default_rng(seed)
    op = assemble_1d(0, 1, n, 1.0 / pe, 1.0, Periodic(), Periodic(), scheme="central")
    I = sp.identity(n, format="csc")
    M = spla.spsolve((I - 0.5 * dt * op.A).tocsc(), (I + 0.5 * dt * op.A).toarray())  # dense CN propagator
    x = op.x
    truth = 1 + 0.5 * np.sin(2 * np.pi * x)
    free = np.ones(n)
    xa = np.ones(n)
    P = 0.25 * np.exp(-(((x[:, None] - x[None, :] + 0.5) % 1 - 0.5) ** 2) / (2 * 0.1**2))  # periodic Gaussian prior
    Q = model_sigma**2 * np.eye(n)
    idx = np.linspace(0, n, n_sensors, endpoint=False).astype(int)
    H = np.zeros((n_sensors, n))
    H[np.arange(n_sensors), idx] = 1.0
    R = obs_sigma**2 * np.eye(n_sensors)
    rf, rk, inside = [], [], []
    for _ in range(steps):
        truth, free = M @ truth, M @ free
        xa = M @ xa
        P = M @ P @ M.T + Q
        y = H @ truth + rng.normal(0, obs_sigma, n_sensors)
        S = H @ P @ H.T + R
        K = np.linalg.solve(S, H @ P).T
        xa = xa + K @ (y - H @ xa)
        P = (np.eye(n) - K @ H) @ P
        rf.append(float(np.sqrt(np.mean((free - truth) ** 2))))
        rk.append(float(np.sqrt(np.mean((xa - truth) ** 2))))
        inside.append(float(np.mean(np.abs(xa - truth) <= 2 * np.sqrt(np.clip(np.diag(P), 0, None)))))
    return KFResult(rf, rk, float(np.mean(inside[steps // 2 :])), rf[-1], rk[-1])
