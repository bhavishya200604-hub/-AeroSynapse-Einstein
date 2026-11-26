r"""CMA-ES (Hansen) with a batched objective, box bounds and increasing-population restarts.

Sampling :math:`x_k\sim m+\sigma\mathcal N(0,C)`; the mean, the covariance (rank-one and rank-mu
updates) and the step size (cumulative path length control) follow Hansen's tutorial
(arXiv:1604.00772). It needs no gradients, which matters here because stall, transition and the
surrogate make the objective non-smooth. Out-of-box samples are projected, and the squared
projection distance is added to their fitness.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]


@dataclass
class CMAResult:
    x: FloatArr
    f: float
    evals: int
    generations: int
    history: list[tuple[int, float]] = field(default_factory=list)  # (evaluations, best so far)
    restarts: int = 0
    stop: str = ""


def _run(
    fun: Callable[[FloatArr], FloatArr],
    x0: FloatArr,
    sigma: float,
    lam: int,
    rng: np.random.Generator,
    budget: int,
    tol_f: float,
    stall_gens: int,
) -> tuple[FloatArr, float, int, int, list[tuple[int, float]], str]:
    n = len(x0)
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1 / np.sum(w**2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chi = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
    m = x0.copy()
    C = np.eye(n)
    pc, ps = np.zeros(n), np.zeros(n)
    B, Dg = np.eye(n), np.ones(n)
    best_x, best_f = x0.copy(), np.inf
    evals = g = 0
    hist: list[tuple[int, float]] = []
    last_gain = 0
    stop = "budget"
    while evals < budget:
        g += 1
        Z = rng.standard_normal((lam, n))
        Y = (Z * Dg) @ B.T
        X = m + sigma * Y
        Xc = np.clip(X, 0.0, 1.0)
        pen = 10.0 * np.sum((X - Xc) ** 2, axis=1)
        f = np.asarray(fun(Xc), float) + pen
        evals += lam
        order = np.argsort(f)
        if f[order[0]] < best_f - tol_f:
            last_gain = g
        if f[order[0]] < best_f:
            best_f, best_x = float(f[order[0]]), Xc[order[0]].copy()
        hist.append((evals, best_f))
        Ysel = Y[order[:mu]]
        yw = w @ Ysel
        m = m + sigma * yw
        Cinv_y = B @ ((B.T @ yw) / Dg)
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Cinv_y
        hs = float(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * g)) / chi < 1.4 + 2 / (n + 1))
        pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * yw
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C) + cmu * (Ysel.T * w) @ Ysel
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi - 1))
        C = 0.5 * (C + C.T)
        ev, B = np.linalg.eigh(C)
        Dg = np.sqrt(np.maximum(ev, 1e-20))
        if g - last_gain > stall_gens:
            stop = "stagnation"
            break
        if sigma * Dg.max() < 1e-6:
            stop = "converged (step size)"
            break
    return best_x, best_f, evals, g, hist, stop


def cmaes(
    fun: Callable[[FloatArr], FloatArr],
    x0: FloatArr,
    sigma: float = 0.15,
    seed: int = 0,
    budget: int = 5000,
    pop: int | None = None,
    restarts: int = 2,
    tol_f: float = 1e-4,
    stall_gens: int = 60,
) -> CMAResult:
    """Minimise ``fun`` over [0, 1]^n. ``fun`` takes a batch (lam, n) and returns (lam,)."""
    rng = np.random.default_rng(seed)
    lam = pop or 4 + int(3 * np.log(len(x0)))
    best_x, best_f = np.asarray(x0, float).copy(), np.inf
    used = gens = 0
    hist: list[tuple[int, float]] = []
    stop = ""
    run = 0
    for run in range(restarts + 1):
        left = budget - used
        if left < lam:
            break
        start = best_x if run == 0 or best_f == np.inf else best_x
        x, f, e, g, h, stop = _run(fun, start, sigma, lam, rng, left, tol_f, stall_gens)
        hist += [(used + a, min(b, best_f)) for a, b in h]
        used += e
        gens += g
        if f < best_f:
            best_x, best_f = x, f
        lam *= 2  # a larger population searches more globally on the next pass
        sigma *= 0.8
    return CMAResult(best_x, best_f, used, gens, hist, run, stop)
