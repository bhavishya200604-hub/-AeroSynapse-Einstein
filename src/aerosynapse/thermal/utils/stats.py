"""Pre-registered tests of the Phase 6B gap runs (paper/PREREG_G1_G3.md).

alpha = 0.05; Holm correction within a run; bootstrap 95 % CIs with 10 000 resamples (seed 0); effects as
geometric-mean ratios (log scale). With n = 5 paired seeds an exact two-sided Wilcoxon test cannot reach p < 0.05
(minimum 0.0625), so the primary paired test is the t-test on log ratios; the Wilcoxon p is reported alongside.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def holm(p: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1); NaN stays NaN and is not counted."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = len(ok)
    order = ok[np.argsort(p[ok])]
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        out[i] = run
    return out.tolist()


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of x."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def paired_log_ratio(test: np.ndarray, ref: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired (by seed) comparison of positive errors: d = log(test / ref). Two-sided paired t-test on d (primary),
    bootstrap CI of mean d, exact Wilcoxon signed-rank p; effect = geometric-mean ratio exp(mean d)."""
    d = np.log(np.asarray(test, dtype=float)) - np.log(np.asarray(ref, dtype=float))
    n = len(d)
    t = stats.ttest_1samp(d, 0.0)
    lo, hi = bootstrap_mean_ci(d, n_boot, seed)
    w = stats.wilcoxon(d, method="exact").pvalue if n >= 2 and np.any(d != 0) else float("nan")
    return {
        "n": n,
        "mean_log_ratio": float(d.mean()),
        "gm_ratio": float(np.exp(d.mean())),
        "gm_ratio_ci95": [float(np.exp(lo)), float(np.exp(hi))],
        "t": float(t.statistic),
        "p_t": float(t.pvalue),
        "p_wilcoxon": float(w),
        "log_ratios": d.tolist(),
    }


def welch_less(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """One-sided Welch t-test H1: mean(a) < mean(b); bootstrap CI of mean(a) - mean(b) (independent resampling)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    t = stats.ttest_ind(a, b, equal_var=False, alternative="less")
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    diff = da - db
    return {
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": float(a.mean() - b.mean()),
        "diff_ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
        "t": float(t.statistic),
        "p": float(t.pvalue),
    }


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    r = stats.spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": len(x)}
