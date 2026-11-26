r"""Optimality diagnostics (spec Section 8E).

At a candidate x* with objective f, inequality constraints g_i ≤ 0 and equalities h_j = 0:

* feasibility: max(g_i, |h_j|);
* active set A = {i : g_i ≥ -tol};
* LICQ: the active-constraint gradients are linearly independent (rank test, smallest singular value);
* multipliers by non-negative least squares on ∇f + Σ μ_i ∇g_i + Σ λ_j ∇h_j = 0 (μ ≥ 0);
* first-order KKT residual ‖∇_x L‖, complementarity max|μ_i g_i|;
* second order: eigenvalues of ∇²L projected onto the null space of the active gradients;
* conditioning: condition number of the reduced Hessian.

KKT satisfaction certifies a local stationary point under a constraint qualification. It does
**not** certify a global optimum unless the problem is convex.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import null_space
from scipy.optimize import nnls

FloatArr = NDArray[np.float64]
VecFn = Callable[[FloatArr], FloatArr]

NONCONVEX_WARNING = (
    "KKT conditions are necessary for local optimality under a constraint qualification; "
    "for a non-convex problem they do not guarantee a global optimum."
)


def fd_jac(fun: VecFn, x: FloatArr, h: float = 1e-6) -> FloatArr:
    f0 = np.atleast_1d(fun(x))
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        J[:, i] = (np.atleast_1d(fun(x + e)) - np.atleast_1d(fun(x - e))) / (2 * h)
    return J


@dataclass
class KKTReport:
    feasibility: float
    active: list[int]
    licq: bool
    min_singular_active: float
    mu: list[float]
    lam: list[float]
    stationarity: float
    complementarity: float
    reduced_hessian_eigs: list[float]
    second_order_ok: bool
    condition_number: float
    convex_known: bool
    warning: str

    @property
    def kkt_ok(self) -> bool:
        return self.feasibility < 1e-6 and self.stationarity < 1e-4 and self.complementarity < 1e-6


def kkt_check(
    f: Callable[[FloatArr], float],
    x: FloatArr,
    g: VecFn | None = None,
    h: VecFn | None = None,
    tol: float = 1e-6,
    convex: bool = False,
    step: float = 1e-5,
) -> KKTReport:
    x = np.asarray(x, float)
    gx = np.atleast_1d(g(x)) if g else np.zeros(0)
    hx = np.atleast_1d(h(x)) if h else np.zeros(0)
    feas = float(max([0.0, *np.maximum(gx, 0), *np.abs(hx)]))
    df = fd_jac(lambda z: np.array([f(z)]), x)[0]
    act = [i for i, v in enumerate(gx) if v >= -tol]
    Jg = fd_jac(g, x)[act] if g and act else np.zeros((0, x.size))
    Jh = fd_jac(h, x) if h else np.zeros((0, x.size))
    Ja = np.vstack([Jg, Jh])
    sv = np.linalg.svd(Ja, compute_uv=False) if Ja.size else np.array([np.inf])
    licq = bool(Ja.shape[0] <= x.size and sv.min() > 1e-8) if Ja.size else True
    # multipliers: mu >= 0 for inequalities, lam free (split into +/-) for equalities
    M = np.hstack([Jg.T, Jh.T, -Jh.T]) if Ja.size else np.zeros((x.size, 0))
    if M.shape[1]:
        coef, _ = nnls(M, -df)
        mu = np.zeros(len(gx))
        mu[act] = coef[: len(act)]
        lam = coef[len(act) : len(act) + len(hx)] - coef[len(act) + len(hx) :]
    else:
        mu, lam = np.zeros(len(gx)), np.zeros(len(hx))
    gradL = df + (Jg.T @ mu[act] if act else 0) + (Jh.T @ lam if len(hx) else 0)

    def lag(z: FloatArr) -> float:
        val = f(z)
        if g:
            val += float(np.atleast_1d(g(z)) @ mu)
        if h:
            val += float(np.atleast_1d(h(z)) @ lam)
        return float(val)

    n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej = np.eye(n)[i] * step, np.eye(n)[j] * step
            H[i, j] = (lag(x + ei + ej) - lag(x + ei - ej) - lag(x - ei + ej) + lag(x - ei - ej)) / (4 * step**2)
    Z = null_space(Ja) if Ja.size else np.eye(n)
    Hr = Z.T @ (0.5 * (H + H.T)) @ Z if Z.size else np.zeros((0, 0))
    eig = np.linalg.eigvalsh(Hr) if Hr.size else np.array([])
    cond = float(np.abs(eig).max() / max(np.abs(eig).min(), 1e-300)) if eig.size else 1.0
    return KKTReport(
        feas,
        act,
        licq,
        float(sv.min()),
        mu.tolist(),
        list(map(float, lam)),
        float(np.linalg.norm(gradL)),
        float(np.max(np.abs(mu * gx))) if len(gx) else 0.0,
        eig.tolist(),
        bool(np.all(eig > -1e-6)),
        cond,
        convex,
        "" if convex else NONCONVEX_WARNING,
    )
