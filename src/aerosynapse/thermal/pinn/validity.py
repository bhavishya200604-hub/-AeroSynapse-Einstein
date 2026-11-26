"""Step 6.3: extrapolation distance, truth-free validity indicators and how well they detect large errors.

distance      decades outside the training range of a positive parameter: max(0, log10(p / hi), log10(lo / p))
physics       residuals of a (parametric) 1D network at ONE parameter value on a tensor grid, without the truth:
              PDE residual in the interior, initial-condition and boundary residuals (the training loss terms);
              the residual operators are linear in u, so the residual of an ensemble MEAN is the mean residual
detection     an indicator flags a prediction when it exceeds a threshold calibrated on in-range values
              (a quantile of the indicator there); scored against "bad" = error > tolerance:
              TPR (bad flagged), FPR (good flagged), AUC (rank-based, threshold-free), Spearman(indicator, error)
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from aerosynapse.thermal.pinn.derivs import evaluate


def distance(p, lo: float, hi: float) -> np.ndarray:
    p = np.asarray(p, float)
    return np.maximum(0.0, np.maximum(np.log10(p / hi), np.log10(lo / p)))


def _cols(names, x, t, p, dtype, dev) -> dict:
    n = x.size
    c = {"x": torch.as_tensor(x, dtype=dtype, device=dev)}
    if "t" in names:
        c["t"] = torch.as_tensor(t if t is not None else np.zeros(n), dtype=dtype, device=dev)
    if "logp" in names:
        c["logp"] = torch.full((n,), float(np.log(p)), dtype=dtype, device=dev)
    return c


def residual_vectors(model, problem, p: float, x: np.ndarray, t: np.ndarray | None) -> dict[str, np.ndarray]:
    """PDE residual on the tensor grid (interior x, all t), IC residual at t = 0 (transient), boundary residuals at
    x = 0 and 1 (all t). Parametric problems take p through the logp input; others use their fixed parameter."""
    ref = next(model.parameters())
    dt, dev = ref.dtype, ref.device
    names = problem.input_names
    first, second = problem.derivative_spec()
    model.eval()
    out = {}
    xi = x[1:-1]
    if problem.has_time:
        T_, X_ = np.meshgrid(t, xi, indexing="ij")
        cols = _cols(names, X_.ravel(), T_.ravel(), p, dt, dev)
    else:
        cols = _cols(names, xi, None, p, dt, dev)
    d = evaluate(model, cols, names, first=first, second=second, create_graph=False)
    out["pde"] = problem.residual(d, problem.p_of(cols)).detach().cpu().numpy()
    if problem.has_time:
        c0 = _cols(names, x, np.zeros_like(x), p, dt, dev)
        u0 = model(torch.stack([c0[k] for k in names], dim=1)).squeeze(-1)
        out["ic"] = (u0 - problem.ic_target(c0)).detach().cpu().numpy()
    for side, xb in (("left", 0.0), ("right", 1.0)):
        tb = t if problem.has_time else None
        n = len(tb) if tb is not None else 1
        cb = _cols(names, np.full(n, xb), tb, p, dt, dev)
        db = evaluate(model, cb, names, first=("x",), second=(), create_graph=False)
        out[f"bc_{side}"] = problem.bc_residual(side, db, cb).detach().cpu().numpy()
    return out


def physics_indicator(res: dict[str, np.ndarray]) -> dict[str, float]:
    """rms of every residual group and the total sqrt(sum of mean squares) (the unweighted training loss, rooted)."""
    ms = {k: float(np.mean(v**2)) for k, v in res.items()}
    return {**{f"{k}_rms": float(np.sqrt(m)) for k, m in ms.items()}, "total": float(np.sqrt(sum(ms.values())))}


def mean_residuals(per_member: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.mean([r[k] for r in per_member], axis=0) for k in per_member[0]}


def auc(scores, bad) -> float | None:
    """P(score of a bad case > score of a good case) (ties count half); None if one class is empty."""
    s, b = np.asarray(scores, float), np.asarray(bad, bool)
    n1, n0 = int(b.sum()), int((~b).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return float((r[b].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def detection(scores, errors, tol: float, threshold: float) -> dict:
    s, e = np.asarray(scores, float), np.asarray(errors, float)
    bad, flag = e > tol, s > threshold
    nb, ng = int(bad.sum()), int((~bad).sum())
    return {
        "tol": float(tol),
        "threshold": float(threshold),
        "n_bad": nb,
        "n_good": ng,
        "tpr": float((flag & bad).sum() / nb) if nb else None,
        "fpr": float((flag & ~bad).sum() / ng) if ng else None,
        "missed_bad": int((~flag & bad).sum()),
        "auc": auc(s, bad),
    }


def spearman(a, b) -> float:
    return float(spearmanr(a, b).statistic)
