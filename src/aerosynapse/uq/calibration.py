r"""Calibration diagnostics for Gaussian predictive distributions N(μ, σ²).

* coverage curve: empirical fraction of targets in the central interval vs nominal level;
* PIT values Φ((y-μ)/σ) (uniform if calibrated) and a reliability histogram;
* negative log-likelihood, sharpness (mean σ), interval score, and the mean absolute coverage error.
A confidence score is never interpreted as a probability unless these checks pass.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArr = NDArray[np.float64]
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])


def coverage_curve(y: FloatArr, mu: FloatArr, sigma: FloatArr, levels: FloatArr = LEVELS) -> FloatArr:
    z = np.abs((y - mu) / np.clip(sigma, 1e-300, None))
    return np.array([np.mean(z <= norm.ppf(0.5 + lv / 2)) for lv in levels])


def calibration_report(y: FloatArr, mu: FloatArr, sigma: FloatArr) -> dict[str, float | list[float]]:
    y, mu, sigma = (np.asarray(a, float).ravel() for a in (y, mu, sigma))
    cov = coverage_curve(y, mu, sigma)
    pit = norm.cdf((y - mu) / np.clip(sigma, 1e-300, None))
    hist, _ = np.histogram(pit, bins=10, range=(0, 1))
    lo, hi = mu - 1.96 * sigma, mu + 1.96 * sigma
    alpha = 0.05
    iscore = (hi - lo) + 2 / alpha * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    return {
        "levels": LEVELS.tolist(),
        "coverage": cov.tolist(),
        "mean_abs_coverage_error": float(np.mean(np.abs(cov - LEVELS))),
        "coverage_95": float(cov[list(LEVELS).index(0.95)]),
        "nll": float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((y - mu) / sigma) ** 2)),
        "sharpness_mean_sigma": float(np.mean(sigma)),
        "interval_score_95": float(np.mean(iscore)),
        "pit_histogram": (hist / hist.sum()).tolist(),
    }


def binomial_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval: the range of empirical coverages consistent with nominal ``p`` at n samples."""
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return float(c - h), float(c + h)
