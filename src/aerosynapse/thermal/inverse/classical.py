"""Step 5.3: classical baseline — weighted nonlinear least squares with a mesh-based forward solver.

    minimise  sum_i ((T_model(r_i, t_i; p) - T_i) / sigma_i)^2   over x = ln p (the unknown parameters)

with scipy.optimize.least_squares (trust-region reflective by default; finite-difference Jacobian in x). Forward model
'fdm' = the Phase 1 finite-difference solver (slab) / radial finite volumes (cylinder, sphere), Crank-Nicolson with
Rannacher start-up and step counts FIXED from a dimensional step derived from the initial guess, so the model output
is a smooth function of the parameters (finite differences need that); 'analytic' = the eigen-series (reference).

Uncertainty: Gauss-Newton covariance C = (J^T J)^-1 of ln p at the estimate, J = CENTRAL-difference Jacobian of the
weighted residuals (step jacobian.rel_step in ln p, as in Step 5.1) - not the solver's forward-difference Jacobian,
whose O(diff_step) error turns an exact degeneracy (no source: only h/k enters) into a condition number of ~5e3 (D90).
A singular value below identifiability.rank_tol x the largest marks a non-identifiable direction: the parameters
along it get an infinite standard deviation (as in Step 5.1); combinations orthogonal to it (e.g. ln Bi = ln h - ln k)
keep a finite one (log_sd). 95 % intervals exp(ln p +- 1.96 sd).
A path of every model evaluation (parameters, cost) is kept for convergence plots. Multi-start: initial guesses
log-uniform in [guess / spread, guess * spread] (Sobol).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from aerosynapse.thermal.data.design import sobol_unit
from aerosynapse.thermal.inverse import body as B
from aerosynapse.thermal.inverse import data as D
from aerosynapse.thermal.inverse import identifiability as ID
from aerosynapse.thermal.inverse import spec as IS


def forward_settings(spec: dict, init: dict) -> dict:
    """FDM settings of the estimator: classical.fdm over forward.fdm, with the step COUNTS fixed from the initial
    guess of alpha (never from the truth, which a real experiment does not know)."""
    f = IS.deep_merge(spec["forward"]["fdm"], (spec.get("classical") or {}).get("fdm") or {})
    alpha = float(init.get("alpha", spec["parameters"]["alpha"]))
    f["dt_seconds"] = float(f["dt_fourier"]) * float(spec["body"]["L"]) ** 2 / alpha
    return f


def fit(
    spec: dict, data: dict, unknown: list[str], init: dict, model: str | None = None, with_cov: bool = True
) -> dict[str, Any]:
    """Weighted least squares from init; with_cov=False skips the covariance (bootstrap refits, Step 5.5)."""
    cl = spec["classical"]
    model = model or cl["model"]
    known = {k: float(spec["parameters"][k]) for k in B.PARAMS if k not in unknown}
    fdm = forward_settings(spec, init) if model == "fdm" else None
    tmin = (
        float(np.min(data["t"])) * float(init.get("alpha", spec["parameters"]["alpha"])) / float(spec["body"]["L"]) ** 2
    )
    nt = max(B.truncation_terms(0.25 * tmin), 20)
    path: list[dict] = []

    def wres(x: np.ndarray) -> np.ndarray:
        p = {**known, **dict(zip(unknown, np.exp(x), strict=True))}
        return (D.model_at(p, spec, data["r"], data["t"], nt, model, fdm) - data["T"]) / data["sigma"]

    def res(x: np.ndarray) -> np.ndarray:
        r = wres(x)
        path.append({**dict(zip(unknown, np.exp(x).tolist(), strict=True)), "cost": float(0.5 * r @ r)})
        return r

    t0 = time.perf_counter()
    x0 = np.log([float(init[k]) for k in unknown])
    sol = least_squares(
        res,
        x0,
        method=cl["method"],
        jac=cl["jac"],
        diff_step=float(cl["diff_step"]),
        xtol=float(cl["tol"]),
        ftol=float(cl["tol"]),
        gtol=float(cl["tol"]),
        max_nfev=int(cl.get("max_nfev", 400)),
    )
    est = {**known, **dict(zip(unknown, np.exp(sol.x).tolist(), strict=True))}
    if not with_cov:
        return {
            "estimates": est,
            "chi2_per_reading": float(2 * sol.cost / len(data["T"])),
            "success": bool(sol.success),
            "message": str(sol.message),
            "n_model_calls": len(path),
            "wall_s": time.perf_counter() - t0,
        }
    hs = float(spec["jacobian"]["rel_step"])
    J = np.stack([(wres(sol.x + hs * e) - wres(sol.x - hs * e)) / (2 * hs) for e in np.eye(len(unknown))], 1)
    wall = time.perf_counter() - t0
    a = ID.analyse(J, 1.0, unknown, float(spec["identifiability"]["rank_tol"]))
    sd = np.asarray(a["crlb_log"], float)
    sj = np.linalg.svd(sol.jac, compute_uv=False)  # the solver's forward-difference Jacobian, for comparison (D90)
    return {
        "estimates": est,
        "sd_log": dict(zip(unknown, sd.tolist(), strict=True)),
        "ci95": {
            k: [float(np.exp(sol.x[i] - 1.96 * sd[i])), float(np.exp(sol.x[i] + 1.96 * sd[i]))]
            for i, k in enumerate(unknown)
        },
        "correlation": a["correlation"],
        "condition_number": a["condition_number"],
        "condition_number_solver_jac": float(sj[0] / sj[-1]) if sj[-1] > 0 else float("inf"),
        "rank": a["rank"],
        "null_directions": a["null_directions"],
        "fisher": a["fisher"],
        "chi2_per_reading": float(2 * sol.cost / len(data["T"])),
        "status": int(sol.status),
        "success": bool(sol.success),
        "message": str(sol.message),
        "n_model_calls": len(path),
        "n_cov_calls": 2 * len(unknown),
        "wall_s": wall,
        "model": model,
        "path": path,
    }


def multistart(
    spec: dict, data: dict, unknown: list[str], guess: dict, n: int, spread: float, seed: int, model: str | None = None
) -> list[dict]:
    """Fits from the guess and n - 1 Sobol starts log-uniform in [guess / spread, guess * spread]."""
    starts = [dict(guess)]
    for u in sobol_unit(max(n - 1, 0), len(unknown), seed) if n > 1 else []:
        starts.append({k: float(guess[k]) * spread ** (2 * u[i] - 1) for i, k in enumerate(unknown)})
    out = []
    for st in starts:
        r = safe_fit(spec, data, unknown, st, model)
        r["start"] = st
        r.pop("path", None)
        out.append(r)
    return out


def covariance_at(spec: dict, data: dict, unknown: list[str], est: dict, model: str = "analytic") -> dict[str, Any]:
    """Gauss-Newton quantities (central-difference Jacobian, Fisher information, sd, rank) at given parameter values,
    without fitting (e.g. a chain start on a confounded ridge, Step 5.6)."""
    known = {k: float(spec["parameters"][k]) for k in B.PARAMS if k not in unknown}
    L = float(spec["body"]["L"])
    tmin = float(np.min(data["t"]))
    fdm = forward_settings(spec, est) if model == "fdm" else None

    def wres(x: np.ndarray) -> np.ndarray:
        p = {**known, **dict(zip(unknown, np.exp(x), strict=True))}
        nt = max(B.truncation_terms(tmin * p["alpha"] / L**2), 20)
        return (D.model_at(p, spec, data["r"], data["t"], nt, model, fdm) - data["T"]) / data["sigma"]

    x = np.log([float(est[k]) for k in unknown])
    hs = float(spec["jacobian"]["rel_step"])
    J = np.stack([(wres(x + hs * e) - wres(x - hs * e)) / (2 * hs) for e in np.eye(len(unknown))], 1)
    a = ID.analyse(J, 1.0, unknown, float(spec["identifiability"]["rank_tol"]))
    r = wres(x)
    return {
        "estimates": {**known, **{k: float(est[k]) for k in unknown}},
        "sd_log": dict(zip(unknown, a["crlb_log"], strict=True)),
        "fisher": a["fisher"],
        "rank": a["rank"],
        "null_directions": a["null_directions"],
        "chi2_per_reading": float(r @ r) / len(r),
    }


def safe_fit(
    spec: dict, data: dict, unknown: list[str], init: dict, model: str | None = None, with_cov: bool = True
) -> dict[str, Any]:
    """fit(), with a numerical failure (non-finite residuals or covariance at an extreme start) recorded as a failed
    fit instead of aborting a whole study: {success False, error, estimates None, chi2_per_reading inf}."""
    try:
        return fit(spec, data, unknown, init, model, with_cov)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError, MemoryError) as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "estimates": None,
            "chi2_per_reading": float("inf"),
            "sd_log": None,
            "n_model_calls": None,
            "status": None,
            "message": str(e),
        }


def log_sd(fit: dict, unknown: list[str], coef: dict[str, float]) -> float:
    """Standard deviation of sum_j coef_j ln p_j at the estimate (Gauss-Newton): finite if the combination is
    orthogonal to every non-identifiable direction (pseudo-inverse of the Fisher information), else inf."""
    c = np.array([float(coef.get(k, 0.0)) for k in unknown])
    for v in fit["null_directions"]:
        if abs(np.dot(c, v)) > 1e-6 * np.linalg.norm(c):
            return float("inf")
    return float(np.sqrt(max(c @ np.linalg.pinv(np.asarray(fit["fisher"], float)) @ c, 0.0)))
