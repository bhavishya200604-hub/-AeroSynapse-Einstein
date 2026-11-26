"""Step 5.5: parameter uncertainty of the least-squares estimator (all in x = ln p, the unknown parameters).

  gauss_newton  Laplace / Gauss-Newton interval  x_hat +- z sd  (sd from classical.fit: central-difference Jacobian)
  bootstrap     parametric: refit to T(x_hat) + sigma * N(0, 1);  residual: refit to T(x_hat) + sigma * r*, r* drawn
                with replacement from the centred weighted residuals of the fit (no distributional assumption);
                percentile intervals and the bias-corrected estimate  x_bc = 2 x_hat - mean(x*)
  posterior     Bayesian: Gaussian likelihood with the known sigma, prior uniform in ln p on [guess / f, guess * f]
                (log-uniform in p); adaptive random-walk Metropolis (Haario et al. 2001 during warm-up, proposal
                2.38^2 / d x covariance, then frozen), equal-tailed credible intervals; integrated autocorrelation
                time (Sokal window) and effective sample size per parameter
Refits start from x_hat. The analytic eigen-series is the default model (equal to the FDM estimator to <= 2e-4 in ln p,
R5.3c / R5.4k) with the number of terms set per evaluation from the current alpha.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import norm

from aerosynapse.thermal.inverse import body as B
from aerosynapse.thermal.inverse import classical as C
from aerosynapse.thermal.inverse import data as D


def model_fn(spec: dict, data: dict, unknown: list[str], model: str = "analytic") -> Callable[[np.ndarray], np.ndarray]:
    """x = ln p (unknown) -> model temperatures at the readings (known parameters at their configured values)."""
    known = {k: float(spec["parameters"][k]) for k in B.PARAMS if k not in unknown}
    L = float(spec["body"]["L"])
    tmin = float(np.min(data["t"]))
    fdm = C.forward_settings(spec, {k: float(spec["initial_guess"][k]) for k in unknown}) if model == "fdm" else None

    def f(x: np.ndarray) -> np.ndarray:
        p = {**known, **dict(zip(unknown, np.exp(x), strict=True))}
        nt = max(B.truncation_terms(tmin * p["alpha"] / L**2), 20)
        return D.model_at(p, spec, data["r"], data["t"], nt, model, fdm)

    return f


def gn_interval(fit: dict, unknown: list[str], level: float) -> dict[str, list[float]]:
    z = float(norm.ppf(0.5 + level / 2))
    return {
        k: [
            float(np.log(fit["estimates"][k]) - z * fit["sd_log"][k]),
            float(np.log(fit["estimates"][k]) + z * fit["sd_log"][k]),
        ]
        for k in unknown
    }


def percentile_interval(samples: np.ndarray, unknown: list[str], level: float) -> dict[str, list[float]]:
    q = np.quantile(samples, [0.5 - level / 2, 0.5 + level / 2], axis=0)
    return {k: [float(q[0, j]), float(q[1, j])] for j, k in enumerate(unknown)}


def bootstrap(
    spec: dict, data: dict, unknown: list[str], fit: dict, n: int, kind: str, seed: int, model: str = "analytic"
) -> dict[str, Any]:
    """n refits of resampled data around the fitted model; samples of ln p (rows), failures dropped and counted."""
    t0 = time.perf_counter()
    x_hat = np.log([fit["estimates"][k] for k in unknown])
    T_hat = model_fn(spec, data, unknown, model)(x_hat)
    sig = np.asarray(data["sigma"], float)
    r = (np.asarray(data["T"], float) - T_hat) / sig
    r = r - r.mean()
    rng = np.random.default_rng(seed)
    init = {k: fit["estimates"][k] for k in unknown}
    rows, failed = [], 0
    for _ in range(n):
        e = rng.standard_normal(len(sig)) if kind == "parametric" else rng.choice(r, len(r), replace=True)
        f = C.safe_fit(spec, {**data, "T": T_hat + sig * e}, unknown, init, model, with_cov=False)
        if f["estimates"] is None:
            failed += 1
            continue
        rows.append([np.log(f["estimates"][k]) for k in unknown])
    X = np.asarray(rows)
    return {
        "kind": kind,
        "samples_log": X,
        "failed": failed,
        "mean_log": dict(zip(unknown, X.mean(0).tolist(), strict=True)),
        "sd_log": dict(zip(unknown, X.std(0, ddof=1).tolist(), strict=True)),
        "bias_log": dict(zip(unknown, (X.mean(0) - x_hat).tolist(), strict=True)),
        "bias_corrected": {k: float(np.exp(2 * x_hat[j] - X[:, j].mean())) for j, k in enumerate(unknown)},
        "wall_s": time.perf_counter() - t0,
    }


def log_posterior(
    spec: dict, data: dict, unknown: list[str], bounds: np.ndarray, model: str = "analytic", priors: dict | None = None
):
    """Gaussian log-likelihood + log prior: uniform in ln p inside bounds, times optional Gaussian priors on ln p
    (priors: {param: (mean ln p, sd)} -- prior knowledge such as a handbook value of k, Step 5.6)."""
    f = model_fn(spec, data, unknown, model)
    T, sig = np.asarray(data["T"], float), np.asarray(data["sigma"], float)
    gp = [(unknown.index(k), float(m), float(s)) for k, (m, s) in (priors or {}).items() if k in unknown]

    def lp(x: np.ndarray) -> float:
        if np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1]):
            return -np.inf
        try:
            r = (f(x) - T) / sig
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return -np.inf
        v = -0.5 * float(r @ r) - 0.5 * sum(((x[j] - m) / s) ** 2 for j, m, s in gp)
        return v if np.isfinite(v) else -np.inf

    return lp


def autocorr_time(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time of a 1D chain (FFT autocorrelation, Sokal's automatic window M >= c tau)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 4 or np.allclose(x, 0):
        return float("nan")
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n]
    acf = acf / acf[0]
    taus = 2.0 * np.cumsum(acf) - 1.0
    m = np.arange(n) < c * taus
    w = int(np.argmin(m)) if not np.all(m) else n - 1
    return float(taus[w])


def metropolis(
    lp: Callable[[np.ndarray], float],
    x0: np.ndarray,
    cov0: np.ndarray,
    steps: int,
    burn: int,
    adapt_every: int,
    seed: int,
) -> dict[str, Any]:
    """Adaptive random-walk Metropolis: the proposal covariance (2.38^2 / d) Sigma is re-estimated from the chain every
    adapt_every steps during the burn-in only (the kept chain is a plain Metropolis chain)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    d = len(x0)
    scale = 2.38**2 / d
    cov = np.asarray(cov0, float) * scale + 1e-12 * np.eye(d)
    x, lx = np.asarray(x0, float).copy(), lp(np.asarray(x0, float))
    if not np.isfinite(lx):
        raise ValueError("the chain must start inside the prior support with a finite likelihood")
    chain = np.empty((burn + steps, d))
    lps = np.empty(burn + steps)
    acc = 0
    for i in range(burn + steps):
        if i < burn and i >= adapt_every and i % adapt_every == 0:
            cov = scale * (np.cov(chain[i // 2 : i].T) + 1e-10 * np.eye(d))
        y = rng.multivariate_normal(x, cov)
        ly = lp(y)
        if np.log(rng.uniform()) < ly - lx:
            x, lx = y, ly
            if i >= burn:
                acc += 1
        chain[i], lps[i] = x, lx
    kept = chain[burn:]
    tau = np.array([autocorr_time(kept[:, j]) for j in range(d)])
    return {
        "samples_log": kept,
        "log_post": lps[burn:],
        "acceptance": acc / steps,
        "tau": tau.tolist(),
        "ess": (steps / tau).tolist(),
        "evaluations": burn + steps + 1,
        "wall_s": time.perf_counter() - t0,
    }


def posterior(
    spec: dict,
    data: dict,
    unknown: list[str],
    fit: dict,
    steps: int,
    burn: int,
    adapt_every: int,
    prior_factor: float,
    seed: int,
    model: str = "analytic",
    priors: dict | None = None,
) -> dict[str, Any]:
    """Posterior samples of ln p started at the least-squares estimate; initial proposal = inverse of the Fisher
    information plus the precision of the Gaussian priors (which also makes a confounded direction proper)."""
    g = np.log([float(spec["initial_guess"][k]) for k in unknown])
    bounds = np.column_stack([g - np.log(prior_factor), g + np.log(prior_factor)])
    x0 = np.clip(np.log([fit["estimates"][k] for k in unknown]), bounds[:, 0] + 1e-9, bounds[:, 1] - 1e-9)
    F = np.asarray(fit["fisher"], float).copy()
    for k, (_, s) in (priors or {}).items():
        if k in unknown:
            F[unknown.index(k), unknown.index(k)] += 1.0 / float(s) ** 2
    cov0 = np.linalg.pinv(F) if np.all(np.isfinite(F)) else np.eye(len(unknown)) * 1e-4
    cov0 = cov0 + np.eye(len(unknown)) * 1e-8 * max(np.trace(cov0), 1e-6)
    lp = log_posterior(spec, data, unknown, bounds, model, priors)
    out = metropolis(lp, x0, cov0, steps, burn, adapt_every, seed)
    out["bounds_log"] = bounds.tolist()
    out["at_prior_edge"] = {
        k: float(
            np.mean((out["samples_log"][:, j] < bounds[j, 0] + 0.05) | (out["samples_log"][:, j] > bounds[j, 1] - 0.05))
        )
        for j, k in enumerate(unknown)
    }
    return out
