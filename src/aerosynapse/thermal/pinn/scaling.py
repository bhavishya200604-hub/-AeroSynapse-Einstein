"""Step 6.1: power-law fits of error vs a size variable, and the cost comparison at equal error.

  loglog      e = C x^p                 straight line in log-log (least squares on log e): slope p, standard error, R^2
  saturating  e = A x^(-q) + e_floor    the usual shape when another error source (optimisation, the other factors)
                                        takes over: fitted in log space (q >= 0, floor >= 0)
Model choice by AIC on the log errors (2 vs 3 parameters). Exponent intervals: bootstrap over the seeds (each level's
seed values resampled with replacement, the median per level refitted).
An exponent at the fit bound (q = Q_MAX) is NOT a measurement: the error falls to its floor within one level step, and
the exponent is reported as unresolved ("resolved": False).
Equal error: the cheapest setting (median wall time over seeds) whose median error is at or below a target.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

Q_MAX = 10.0  # upper bound of the saturating exponent


def loglog(x: np.ndarray, e: np.ndarray) -> dict:
    lx, le = np.log(np.asarray(x, float)), np.log(np.asarray(e, float))
    A = np.column_stack([np.ones_like(lx), lx])
    coef, res, *_ = np.linalg.lstsq(A, le, rcond=None)
    r = le - A @ coef
    n = len(lx)
    s2 = float(r @ r) / max(n - 2, 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    ss = float(((le - le.mean()) ** 2).sum())
    return {
        "model": "loglog",
        "slope": float(coef[1]),
        "slope_se": float(np.sqrt(cov[1, 1])),
        "C": float(np.exp(coef[0])),
        "r2": 1.0 - float(r @ r) / ss if ss > 0 else 1.0,
        "rss_log": float(r @ r),
        "n": n,
    }


def saturating(x: np.ndarray, e: np.ndarray) -> dict | None:
    """e = A x^-q + floor, least squares on log e; None if the fit fails."""
    x, e = np.asarray(x, float), np.asarray(e, float)
    lx = np.log(x)

    def f(lxv, la, q, lf):
        return np.log(np.exp(la - q * lxv) + np.exp(lf))

    p0 = [float(np.log(e.max()) + np.log(x.min())), 1.0, float(np.log(max(e.min() * 0.5, 1e-300)))]
    try:
        with warnings.catch_warnings():  # the parameter covariance is not used (exponent intervals: bootstrap)
            warnings.simplefilter("ignore", OptimizeWarning)
            p, _ = curve_fit(
                f, lx, np.log(e), p0=p0, bounds=([-np.inf, 0.0, -745.0], [np.inf, Q_MAX, 0.0]), maxfev=20000
            )
    except (RuntimeError, ValueError):
        return None
    r = np.log(e) - f(lx, *p)
    return {
        "model": "saturating",
        "q": float(p[1]),
        "A": float(np.exp(p[0])),
        "floor": float(np.exp(p[2])),
        "rss_log": float(r @ r),
        "n": len(x),
    }


def aic(rss: float, n: int, k: int) -> float:
    return n * np.log(max(rss, 1e-300) / n) + 2 * k


def fit(levels: np.ndarray, values: list[np.ndarray], n_boot: int = 1000, seed: int = 0) -> dict:
    """levels (m,), values: one array of seed errors per level. Median per level is fitted by both models; the
    preferred model is the one with the lower AIC; bootstrap percentiles of its exponent over the seeds."""
    x = np.asarray(levels, float)
    med = np.array([np.median(v) for v in values])
    ll, sat = loglog(x, med), saturating(x, med)
    a_ll = aic(ll["rss_log"], len(x), 2)
    a_sat = aic(sat["rss_log"], len(x), 3) if sat else np.inf
    best = "saturating" if (sat and len(x) > 3 and a_sat < a_ll) else "loglog"
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(int(n_boot)):
        m = np.array([np.median(rng.choice(v, len(v), replace=True)) for v in values])
        if best == "loglog":
            boot.append(loglog(x, m)["slope"])
        else:
            s = saturating(x, m)
            if s:
                boot.append(-s["q"])
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if len(boot) > 10 else None
    exponent = ll["slope"] if best == "loglog" else -sat["q"]
    at_bound = best == "saturating" and sat["q"] >= 0.99 * Q_MAX
    return {
        "levels": x.tolist(),
        "median": med.tolist(),
        "loglog": ll,
        "saturating": sat,
        "aic": {"loglog": float(a_ll), "saturating": float(a_sat)},
        "best": best,
        "exponent": float(exponent),
        "exponent_ci95": ci,
        "resolved": not at_bound,
        "ci_at_bound": bool(ci) and best == "saturating" and ci[0] <= -0.99 * Q_MAX,
        "n_boot": len(boot),
    }


def cheapest(errors: np.ndarray, times: np.ndarray, target: float) -> dict | None:
    """The run with the smallest wall time among those with error <= target; None if none reaches it."""
    e, t = np.asarray(errors, float), np.asarray(times, float)
    ok = np.flatnonzero(e <= target)
    if len(ok) == 0:
        return None
    i = int(ok[np.argmin(t[ok])])
    return {"index": i, "error": float(e[i]), "wall_s": float(t[i])}


def by_setting(runs: list[dict]) -> list[dict]:
    """Group runs by (factor, level): median and max error, median wall time, number of seeds."""
    groups: dict = {}
    for r in runs:
        groups.setdefault((r["factor"], r["level"]), []).append(r)
    out = []
    for (fac, lv), rs in groups.items():
        e = np.array([r["rel_l2"] for r in rs])
        out.append(
            {
                "factor": fac,
                "level": lv,
                "rel_l2": float(np.median(e)),
                "rel_l2_max": float(e.max()),
                "wall_s": float(np.median([r["wall_s"] for r in rs])),
                "n": len(rs),
            }
        )
    return out


def pareto(errors: np.ndarray, times: np.ndarray) -> list[int]:
    """Indices of the error-time Pareto front (no other run is both faster and more accurate), by increasing time."""
    e, t = np.asarray(errors, float), np.asarray(times, float)
    out, best = [], np.inf
    for i in np.argsort(t):
        if e[i] < best:
            out.append(int(i))
            best = e[i]
    return out
