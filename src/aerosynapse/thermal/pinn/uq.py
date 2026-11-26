"""Step 6.2: calibration of predictive uncertainty against a known truth (pointwise; arrays of any matching shape).

A method gives a predictive mean mu and standard deviation sd at every evaluation point; y is the truth.
  coverage(level)   fraction of points with |y - mu| <= z_{(1+level)/2} sd   (central Gaussian interval)
  ece               mean |coverage(level) - level| over levels 0.05, 0.10, ..., 0.95
  crps              mean continuous ranked probability score of N(mu, sd^2) at y (closed form; same units as y)
  dispersion        rms(error) / rms(sd): 1 = spread matches error in size, > 1 overconfident
  rank_corr         Spearman correlation of sd and |error| over the points: does the spread RANK the errors?
Split conformal (Lei et al. 2018, LITERATURE.md Lei2018): normalised scores s = |y - mu| / sd on a
calibration set; q = the ceil((n+1)(1-alpha))-th smallest score; interval mu +- q sd on new points.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import norm, spearmanr

SD_FLOOR = 1e-300
ECE_LEVELS = np.round(np.arange(0.05, 0.951, 0.05), 2)


def _prep(mu, sd, y):
    mu, sd, y = (np.asarray(v, float).ravel() for v in (mu, sd, y))
    return mu, np.maximum(sd, SD_FLOOR), y


def z_of(level: float) -> float:
    return float(ndtri(0.5 + 0.5 * level))


def coverage(mu, sd, y, levels) -> dict:
    mu, sd, y = _prep(mu, sd, y)
    r = np.abs(y - mu) / sd
    return {float(lv): float(np.mean(r <= z_of(lv))) for lv in levels}


def ece(mu, sd, y, levels=ECE_LEVELS) -> float:
    c = coverage(mu, sd, y, levels)
    return float(np.mean([abs(c[float(lv)] - float(lv)) for lv in levels]))


def crps_gaussian(mu, sd, y) -> float:
    mu, sd, y = _prep(mu, sd, y)
    z = (y - mu) / sd
    return float(np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def dispersion(mu, sd, y) -> float:
    mu, sd, y = _prep(mu, sd, y)
    return float(np.sqrt(np.mean((y - mu) ** 2)) / np.sqrt(np.mean(sd**2)))


def rank_corr(mu, sd, y) -> float:
    mu, sd, y = _prep(mu, sd, y)
    return float(spearmanr(sd, np.abs(y - mu)).statistic)


def conformal_scale(mu, sd, y, alpha: float) -> float:
    """Split-conformal multiplier q of sd for miscoverage alpha (finite-sample rank; inf if n is too small)."""
    mu, sd, y = _prep(mu, sd, y)
    s = np.sort(np.abs(y - mu) / sd)
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return float(s[k - 1]) if k <= len(s) else float("inf")


def summary(mu, sd, y, levels) -> dict:
    """All metrics of one method on one set of points."""
    mu_, sd_, y_ = _prep(mu, sd, y)
    e = y_ - mu_
    return {
        "rel_l2": float(np.linalg.norm(e) / max(np.linalg.norm(y_), SD_FLOOR)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "rms_sd": float(np.sqrt(np.mean(sd_**2))),
        "dispersion": dispersion(mu, sd, y),
        "coverage": {str(k): v for k, v in coverage(mu, sd, y, levels).items()},
        "ece": ece(mu, sd, y),
        "crps": crps_gaussian(mu, sd, y),
        "mae": float(np.mean(np.abs(e))),
        "rank_corr": rank_corr(mu, sd, y),
    }


def moments(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation over the first axis (members or MC samples; ddof 1)."""
    s = np.asarray(samples, float)
    return s.mean(axis=0), s.std(axis=0, ddof=1)
