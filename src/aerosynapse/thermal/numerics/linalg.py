"""Direct linear solvers: Thomas algorithm (single + batched), banded/sparse factorisation wrappers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

Array = NDArray[np.float64]


def thomas(lower: Array, diag: Array, upper: Array, rhs: Array) -> Array:
    """Solve a tridiagonal system. lower[0] and upper[-1] are ignored.

    Row i:  lower[i] x[i-1] + diag[i] x[i] + upper[i] x[i+1] = rhs[i].
    No pivoting: requires (e.g.) diagonal dominance, true for all implicit schemes used here.
    Works on batches: all arrays may have shape (..., n); the recursion runs along the last axis.
    """
    a, b, c, d = (np.asarray(v, dtype=float) for v in (lower, diag, upper, rhs))
    n = b.shape[-1]
    cp = np.empty_like(b)
    dp = np.empty(np.broadcast_shapes(b.shape, d.shape))
    cp[..., 0] = c[..., 0] / b[..., 0]
    dp[..., 0] = d[..., 0] / b[..., 0]
    for i in range(1, n):
        den = b[..., i] - a[..., i] * cp[..., i - 1]
        cp[..., i] = c[..., i] / den if i < n - 1 else 0.0
        dp[..., i] = (d[..., i] - a[..., i] * dp[..., i - 1]) / den
    x = np.empty_like(dp)
    x[..., -1] = dp[..., -1]
    for i in range(n - 2, -1, -1):
        x[..., i] = dp[..., i] - cp[..., i] * x[..., i + 1]
    return x


class ThomasFactor:
    """Pre-factorised tridiagonal matrix: each solve is one forward + one backward sweep."""

    def __init__(self, lower: Array, diag: Array, upper: Array):
        a, b, c = (np.asarray(v, dtype=float) for v in (lower, diag, upper))
        n = b.shape[-1]
        self.a = a
        self.cp = np.empty_like(b)
        self.inv = np.empty_like(b)  # 1/den
        self.inv[..., 0] = 1.0 / b[..., 0]
        self.cp[..., 0] = c[..., 0] * self.inv[..., 0]
        for i in range(1, n):
            self.inv[..., i] = 1.0 / (b[..., i] - a[..., i] * self.cp[..., i - 1])
            self.cp[..., i] = c[..., i] * self.inv[..., i] if i < n - 1 else 0.0
        self.n = n

    def solve(self, d: Array) -> Array:
        dp = np.empty(np.broadcast_shapes(self.inv.shape, np.shape(d)))
        dp[..., 0] = d[..., 0] * self.inv[..., 0]
        for i in range(1, self.n):
            dp[..., i] = (d[..., i] - self.a[..., i] * dp[..., i - 1]) * self.inv[..., i]
        for i in range(self.n - 2, -1, -1):
            dp[..., i] -= self.cp[..., i] * dp[..., i + 1]
        return dp


def bandwidth(M: sps.spmatrix) -> tuple[int, int]:
    """(lower, upper) bandwidth of a sparse matrix."""
    C = sps.coo_matrix(M)
    if C.nnz == 0:
        return 0, 0
    off = C.col - C.row
    return int(max(0, -off.min())), int(max(0, off.max()))


def tridiag_diagonals(M: sps.spmatrix) -> tuple[Array, Array, Array]:
    """Extract (lower, diag, upper) arrays in the row convention used by :func:`thomas`."""
    M = sps.csr_matrix(M)
    n = M.shape[0]
    lo = np.zeros(n)
    up = np.zeros(n)
    lo[1:] = M.diagonal(-1)
    up[:-1] = M.diagonal(1)
    return lo, M.diagonal().copy(), up


def factorize(
    M: sps.spmatrix,
    method: str | Callable[[sps.spmatrix], Callable[[Array], Array]] = "auto",
) -> Callable[[Array], Array]:
    """Return solve(rhs) for a fixed matrix. method: auto | thomas | splu | bicgstab_ilu | callable.

    auto -> Thomas if the matrix is tridiagonal, sparse LU (SuperLU) otherwise. A callable is a
    user-supplied factory ``M -> solve`` (e.g. a multigrid or MG-preconditioned CG solver).
    """
    if callable(method):
        return method(M)
    if method == "bicgstab_ilu":
        Mc = sps.csc_matrix(M)
        ilu = spla.spilu(Mc, drop_tol=1e-5, fill_factor=10)
        P = spla.LinearOperator(M.shape, ilu.solve)

        def solve_it(rhs: Array) -> Array:
            x, info = spla.bicgstab(Mc, rhs, M=P, rtol=1e-12, atol=0.0, maxiter=500)
            if info != 0:
                raise RuntimeError(f"bicgstab did not converge (info={info})")
            return x

        return solve_it
    lb, ub = bandwidth(M)
    if method == "thomas" or (method == "auto" and lb <= 1 and ub <= 1):
        if lb > 1 or ub > 1:
            raise ValueError(f"matrix is not tridiagonal (bandwidth {lb},{ub})")
        return ThomasFactor(*tridiag_diagonals(M)).solve
    lu = spla.splu(sps.csc_matrix(M))
    return lu.solve


def cg(
    A: sps.spmatrix | Callable[[Array], Array],
    b: Array,
    x0: Array | None = None,
    M: Callable[[Array], Array] | None = None,
    rtol: float = 1e-10,
    maxiter: int = 10_000,
) -> tuple[Array, list[float]]:
    """(Preconditioned) conjugate gradients for SPD systems.

    Returns x and the relative residual history ||r_k|| / ||b||. ``M`` applies an SPD approximation of
    A^{-1} (e.g. Jacobi or one symmetric multigrid V-cycle).
    """
    Aop = A if callable(A) else (lambda v: A @ v)
    Mop = M if M is not None else (lambda v: v)
    x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, dtype=float)
    r = b - Aop(x)
    z = Mop(r)
    p = z.copy()
    rz = float(r @ z)
    nb = max(float(np.linalg.norm(b)), 1e-300)
    hist = [float(np.linalg.norm(r)) / nb]
    for _ in range(maxiter):
        if hist[-1] < rtol:
            break
        Ap = Aop(p)
        alpha = rz / float(p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        hist.append(float(np.linalg.norm(r)) / nb)
        z = Mop(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, hist


def jacobi_preconditioner(A: sps.spmatrix) -> Callable[[Array], Array]:
    d = 1.0 / sps.csr_matrix(A).diagonal()
    return lambda r: d * r
