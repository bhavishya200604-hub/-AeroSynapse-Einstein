"""Step 6.4: variance-based global sensitivity (Sobol indices) by Monte Carlo and by polynomial chaos.

Inputs are independent; each is uniform in u in [0, 1] and mapped to its physical range (log-uniform for scale 'log').
Monte Carlo (Saltelli 2010, LITERATURE.md Saltelli2010): matrices A, B (N x d) from one scrambled Sobol sequence of
dimension 2d, AB_i = A with column i taken from B;
    V      = var([f(A), f(B)])
    S_i    = mean(f(B) (f(AB_i) - f(A))) / V                       (first order, Saltelli 2010)
    ST_i   = mean((f(A) - f(AB_i))^2) / (2 V)                      (total, Jansen 1999)
needing N (d + 2) model runs; intervals by bootstrap over the N rows.
Polynomial chaos: orthonormal Legendre polynomials in x = 2u - 1, total degree <= p, least squares; the indices
follow from the coefficients (S_i: terms in x_i only; ST_i: terms containing x_i); leave-one-out error from the hat
matrix. Verified on the Ishigami function (analytic indices) in tests/unit/test_pinn_sensitivity.py.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.polynomial import legendre
from scipy.stats import qmc


def to_phys(u: np.ndarray, ranges: list[tuple[float, float, str]]) -> np.ndarray:
    """u (n, d) in [0, 1] -> physical values; ranges = [(lo, hi, 'log'|'linear')]."""
    out = np.empty_like(u, dtype=float)
    for j, (lo, hi, sc) in enumerate(ranges):
        out[:, j] = (
            np.exp(np.log(lo) + u[:, j] * (np.log(hi) - np.log(lo))) if sc == "log" else lo + u[:, j] * (hi - lo)
        )
    return out


def saltelli(n: int, d: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """(A, B, [AB_i]) in the unit cube; n a power of two (prefixes of the scrambled Sobol sequence are balanced)."""
    s = qmc.Sobol(2 * d, scramble=True, seed=seed).random(n)
    A, B = s[:, :d], s[:, d:]
    AB = []
    for i in range(d):
        m = A.copy()
        m[:, i] = B[:, i]
        AB.append(m)
    return A, B, AB


def stack(A, B, AB) -> np.ndarray:
    """All evaluation points in the order A, B, AB_1, ..., AB_d (N (d + 2) rows)."""
    return np.concatenate([A, B, *AB])


def indices(f: np.ndarray, n: int, d: int) -> dict:
    """First-order and total indices from model outputs in the order of `stack`."""
    f = np.asarray(f, float)
    fA, fB = f[:n], f[n : 2 * n]
    fAB = [f[(2 + i) * n : (3 + i) * n] for i in range(d)]
    V = float(np.var(np.concatenate([fA, fB])))
    S = [float(np.mean(fB * (fAB[i] - fA)) / V) for i in range(d)]
    ST = [float(np.mean((fA - fAB[i]) ** 2) / (2 * V)) for i in range(d)]
    return {"S": S, "ST": ST, "var": V, "mean": float(np.mean(np.concatenate([fA, fB])))}


def bootstrap(f: np.ndarray, n: int, d: int, n_boot: int = 1000, seed: int = 0) -> dict:
    """Percentile 95 % intervals of S and ST by resampling the N rows (all blocks with the same rows)."""
    f = np.asarray(f, float)
    blocks = f.reshape(d + 2, n)
    g = np.random.default_rng(seed)
    S, ST = [], []
    for _ in range(int(n_boot)):
        r = g.integers(0, n, n)
        ix = indices(blocks[:, r].ravel(), n, d)
        S.append(ix["S"])
        ST.append(ix["ST"])
    S, ST = np.array(S), np.array(ST)
    return {
        "S_ci95": np.percentile(S, [2.5, 97.5], axis=0).T.tolist(),
        "ST_ci95": np.percentile(ST, [2.5, 97.5], axis=0).T.tolist(),
    }


# --------------------------------------------------------------------------- polynomial chaos
def multi_indices(d: int, p: int) -> list[tuple[int, ...]]:
    return [a for a in itertools.product(range(p + 1), repeat=d) if sum(a) <= p]


def _basis(u: np.ndarray, alphas: list[tuple[int, ...]]) -> np.ndarray:
    x = 2.0 * np.asarray(u, float) - 1.0
    p = max(max(a) for a in alphas)
    # orthonormal Legendre on U(-1, 1): sqrt(2k + 1) P_k(x)
    P = [
        [legendre.legval(x[:, j], [0] * k + [1]) * np.sqrt(2 * k + 1) for k in range(p + 1)] for j in range(x.shape[1])
    ]
    return np.column_stack([np.prod([P[j][a[j]] for j in range(len(a))], axis=0) for a in alphas])


def pce_fit(u: np.ndarray, f: np.ndarray, p: int) -> dict:
    """Least-squares PCE of total degree p; coefficients, LOO relative error, Sobol indices."""
    d = u.shape[1]
    alphas = multi_indices(d, p)
    Psi = _basis(u, alphas)
    f = np.asarray(f, float)
    c, *_ = np.linalg.lstsq(Psi, f, rcond=None)
    Q, _ = np.linalg.qr(Psi, mode="reduced")  # leverages h_ii = ||Q_i||^2 without the n x n hat matrix
    h = np.einsum("ij,ij->i", Q, Q)
    res = f - Psi @ c
    loo = res / np.clip(1.0 - h, 1e-12, None)
    V = float(sum(ci**2 for ci, a in zip(c, alphas, strict=True) if any(a)))
    S = [
        float(sum(ci**2 for ci, a in zip(c, alphas, strict=True) if a[i] > 0 and sum(a) == a[i]) / V) for i in range(d)
    ]
    ST = [float(sum(ci**2 for ci, a in zip(c, alphas, strict=True) if a[i] > 0) / V) for i in range(d)]
    return {
        "degree": p,
        "n_terms": len(alphas),
        "n_samples": len(f),
        "mean": float(c[0]),
        "var": V,
        "S": S,
        "ST": ST,
        "loo_rel_error": float(np.sqrt(np.mean(loo**2)) / max(np.std(f), 1e-300)),
        "coef": c.tolist(),
        "alphas": [list(a) for a in alphas],
    }


def pce_eval(model: dict, u: np.ndarray) -> np.ndarray:
    return _basis(u, [tuple(a) for a in model["alphas"]]) @ np.asarray(model["coef"])
