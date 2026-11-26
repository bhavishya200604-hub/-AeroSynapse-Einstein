"""Phase 5 measurement data: synthetic (forward model at the true parameters + Phase 2 noise) or a user file, and the
classical least-squares estimate on the same data (Levenberg-Marquardt on log-parameters; the reference estimator of
Step 5.2, the classical baseline of Step 5.3).

User file (CSV with a header line, '#' comment lines allowed, or NPZ with the same array names):
    x      sensor position [m] (0 = centre plane / axis / centre, L = surface)
    t      time since the start [s] (> 0)
    T      measured temperature (same unit as body.T0 / body.T_inf)
    sigma  optional per-reading noise standard deviation (same unit as T); default: Phase 2 model at data.level
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from aerosynapse.thermal.inverse import body as B
from aerosynapse.thermal.inverse import spec as IS


def synthetic(spec: dict, level: float, seed: int) -> dict[str, Any]:
    """Noisy temperatures on the measurement design (sensors x times) at the TRUE parameters (analytic model)."""
    body, xs, ts = IS.body_of(spec), IS.sensor_positions(spec), IS.measurement_times(spec)
    p = {k: float(spec["parameters"][k]) for k in B.PARAMS}
    Tc = B.temperatures(p, body, xs, ts, "analytic", IS.n_terms(spec))
    sig = IS.noise_sigma(spec, Tc, level)
    T = Tc + sig * np.random.default_rng(seed).standard_normal(Tc.shape)
    R, Tt = np.meshgrid(xs, ts)
    return {
        "r": R.ravel(),
        "t": Tt.ravel(),
        "T": T.ravel(),
        "sigma": np.broadcast_to(sig, Tc.shape).ravel().copy(),
        "T_clean": Tc.ravel(),
        "source": f"synthetic (level {level:g}, seed {seed})",
    }


def read_file(path: str | Path, spec: dict, level: float) -> dict[str, Any]:
    cols = IS.read_columns(path)
    for k in ("x", "t", "T"):
        if k not in cols:
            raise ValueError(f"data.file '{path}': column '{k}' missing (need x [m], t [s], T, optional sigma)")
    L = float(spec["body"]["L"])
    if np.any(cols["x"] < -1e-12) or np.any(cols["x"] > L * (1 + 1e-9)):
        raise ValueError(f"data.file '{path}': x must lie in [0, L = {L:g}] m")
    if np.any(cols["t"] <= 0) or not np.all(np.isfinite(cols["T"])):
        raise ValueError(f"data.file '{path}': need t > 0 and finite T")
    if "sigma" in cols:
        sig = cols["sigma"]
        if np.any(sig <= 0):
            raise ValueError(f"data.file '{path}': sigma must be > 0")
    else:
        sig = IS.noise_sigma(spec, cols["T"], level)
    return {
        "r": cols["x"],
        "t": cols["t"],
        "T": cols["T"],
        "sigma": np.broadcast_to(sig, cols["T"].shape).copy(),
        "T_clean": None,
        "source": f"file {path}",
    }


def load(spec: dict, seed: int = 0) -> dict[str, Any]:
    d = spec["data"]
    if d["source"] == "file":
        return read_file(d["file"], spec, float(d["level"]))
    return synthetic(spec, float(d["level"]), int(d["seed"]) + seed)


def model_at(
    p: dict, spec: dict, r: np.ndarray, t: np.ndarray, n_terms: int, model: str = "analytic", fdm: dict | None = None
) -> np.ndarray:
    """Forward model at scattered (r, t) readings (evaluated on the grid of their unique values)."""
    xs, ix = np.unique(r, return_inverse=True)
    ts, it = np.unique(t, return_inverse=True)
    return B.temperatures(p, IS.body_of(spec), xs, ts, model, n_terms, fdm)[it, ix]


def least_squares_estimate(spec: dict, data: dict, unknown: list[str], init: dict) -> dict[str, Any]:
    """Weighted least squares on log-parameters (Levenberg-Marquardt, analytic model) from the same initial guess."""
    known = {k: float(spec["parameters"][k]) for k in B.PARAMS if k not in unknown}
    tmin = (
        float(np.min(data["t"])) * float(init.get("alpha", spec["parameters"]["alpha"])) / float(spec["body"]["L"]) ** 2
    )
    nt = max(B.truncation_terms(tmin * 0.25), 20)  # margin: the estimate may have a smaller alpha than the guess

    def res(x):
        p = {**known, **dict(zip(unknown, np.exp(x), strict=True))}
        return (model_at(p, spec, data["r"], data["t"], nt) - data["T"]) / data["sigma"]

    sol = least_squares(res, np.log([float(init[k]) for k in unknown]), method="lm", xtol=1e-12, ftol=1e-12)
    est = {**known, **dict(zip(unknown, np.exp(sol.x).tolist(), strict=True))}
    return {
        "estimates": est,
        "success": bool(sol.success),
        "chi2_per_reading": float(2 * sol.cost / len(data["T"])),
        "message": sol.message,
    }
