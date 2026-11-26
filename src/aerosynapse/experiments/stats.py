"""Statistics for multi-seed comparisons: paired Wilcoxon, Holm correction, bootstrap CIs, Friedman."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats


def bootstrap_ci(
    x: Sequence[float], stat: str = "median", n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(estimate, lo, hi) percentile bootstrap CI of the mean or median."""
    a = np.asarray(x, float)
    f = np.median if stat == "median" else np.mean
    rng = np.random.default_rng(seed)
    boots = f(a[rng.integers(0, len(a), (n_boot, len(a)))], axis=1)
    return float(f(a)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided exact Wilcoxon signed-rank p-value for paired samples (a vs b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(a, b, alternative="two-sided", method="exact").pvalue)


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    """Holm–Bonferroni step-down correction; returns adjusted p and rejection flags."""
    keys = sorted(pvals, key=pvals.get)  # type: ignore[arg-type]
    m = len(keys)
    out: dict[str, dict[str, float | bool]] = {}
    running = 0.0
    for i, k in enumerate(keys):
        adj = min(1.0, (m - i) * pvals[k])
        running = max(running, adj)
        out[k] = {"p": pvals[k], "p_holm": running, "reject": running < alpha}
    return out


def friedman(groups: dict[str, Sequence[float]]) -> float:
    """Friedman test p-value across >= 3 related groups (blocks = seeds)."""
    return float(stats.friedmanchisquare(*[np.asarray(v, float) for v in groups.values()]).pvalue)


def effect_size_log_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    """Median of log10(a/b) over pairs (negative: a better than b)."""
    return float(np.median(np.log10(np.asarray(a, float) / np.asarray(b, float))))
