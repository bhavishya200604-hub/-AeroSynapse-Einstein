"""Step 5.1: local (practical) identifiability BEFORE any training or fitting.

For a model y(p) of the measurements (n_obs values) and Gaussian measurement noise with standard deviations s_i:

    J      = dy / d(log p)                    sensitivity Jacobian (central differences in log-parameters);
                                              column j = the change of every reading per relative change of p_j
    F      = (J / s)^T (J / s)                Fisher information of log p
    CRLB   = sqrt(diag(F^-1))                 Cramer-Rao lower bound on the standard deviation of any unbiased
                                              estimate of log p_j  (= relative standard deviation of p_j to first
                                              order; reported in %)
    SVD    of J / s: singular values s_1 >= ... ; condition number s_1 / s_n; a singular value below
           rank_tol * s_1 marks a non-identifiable direction (its right singular vector in log-parameter space)
    corr   = correlation matrix of F^-1 (|corr| -> 1: parameters confounded)
    gamma  = collinearity index 1 / sqrt(lambda_min(Jn^T Jn)) of the unit-length-normalised columns Jn
             (Brun et al. 2001, see docs/LITERATURE.md)

With a structurally non-identifiable direction F is singular: the bound is infinite for every parameter that has a
component along that direction; reported as inf, not as a pseudo-inverse value.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def jacobian(
    fn: Callable[[dict[str, float]], np.ndarray],
    p: dict[str, float],
    names: list[str] | tuple[str, ...],
    rel_step: float = 1e-5,
) -> np.ndarray:
    """dy/d(log p_j) by central differences: (y(p e^{+d}) - y(p e^{-d})) / (2 d); y flattened."""
    cols = []
    for n in names:
        hi, lo = dict(p), dict(p)
        hi[n], lo[n] = p[n] * np.exp(rel_step), p[n] * np.exp(-rel_step)
        cols.append((np.ravel(fn(hi)) - np.ravel(fn(lo))) / (2.0 * rel_step))
    return np.stack(cols, 1)


def analyse(
    J: np.ndarray, sigma: float | np.ndarray, names: list[str] | tuple[str, ...], rank_tol: float = 1e-8
) -> dict:
    """Identifiability measures of the weighted Jacobian J / sigma (sigma: scalar or per-observation)."""
    s = np.broadcast_to(np.ravel(np.asarray(sigma, float)), (J.shape[0],))
    if np.any(s <= 0):
        raise ValueError("noise standard deviations must be > 0 (the bound is 0 for noise-free data)")
    W = J / s[:, None]
    U, sv, Vt = np.linalg.svd(W, full_matrices=False)
    null = sv <= rank_tol * sv[0]  # '<=': an all-zero Jacobian is rank 0, not full rank
    F = W.T @ W
    out: dict = {
        "names": list(names),
        "singular_values": sv.tolist(),
        "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        "rank": int((~null).sum()),
        "null_directions": [Vt[i].tolist() for i in np.flatnonzero(null)],
    }
    if null.any():
        involved = np.any(np.abs(Vt[null]) > 1e-3, axis=0)
        Fr = np.linalg.pinv(F)  # only used for the identifiable parameters (no component along a null direction)
        crlb = np.where(involved, np.inf, np.sqrt(np.clip(np.diag(Fr), 0, None)))
        corr = None
    else:
        C = np.linalg.inv(F)
        crlb = np.sqrt(np.diag(C))
        d = np.sqrt(np.diag(C))
        corr = (C / np.outer(d, d)).tolist()
    norms = np.linalg.norm(J, axis=0, keepdims=True)
    if np.all(norms > 0):
        Jn = J / norms
        lam_min = float(np.linalg.eigvalsh(Jn.T @ Jn)[0])
    else:  # a parameter the readings do not depend on at all (e.g. alpha at steady state)
        lam_min = 0.0
    out.update(
        {
            "crlb_log": [float(c) for c in crlb],
            "crlb_percent": {n: float(100.0 * c) for n, c in zip(names, crlb, strict=True)},
            "correlation": corr,
            # singular (rank test above) -> inf: lam_min is then round-off (e.g. 5.6e-17, giving 1.3e8; D144)
            "collinearity_index": float(1.0 / np.sqrt(lam_min)) if lam_min > 0 and not null.any() else float("inf"),
            "fisher": F.tolist(),
        }
    )
    return out
