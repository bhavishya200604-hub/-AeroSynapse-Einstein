"""Step 5.6: profile likelihood — the practical identifiability of each unknown parameter from the data itself.

For a parameter p_j fixed on a grid around its estimate, every other unknown is re-fitted; the profile is the
minimum total chi2 = sum ((T_model - T) / sigma)^2 as a function of ln p_j. The 95 % profile-likelihood interval is
{ln p_j : chi2 - chi2_min <= 3.84} (chi2 with 1 degree of freedom). A side on which the profile never exceeds 3.84
within the grid is UNBOUNDED: the data cannot rule out arbitrarily small (or large) values there - practical
non-identifiability; a profile that is flat everywhere is a structural non-identifiability (Raue et al. 2009; see
docs/LITERATURE.md). Unlike the Cramer-Rao bound this is not a local, linearised statement.
Each side is traced outward from the estimate (starts: the previous point's solution and the best fit), then
re-fitted inward from the outer neighbour's solution (the lower chi2 kept), on a
two-scale grid: `fine` steps out to 4 Gauss-Newton standard deviations (so the 95 % crossing is resolved), then
`coarse` geometric steps out to the span (to see whether the profile ever closes).
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from aerosynapse.thermal.inverse import classical as C

DCHI2_95 = 3.841458820694124  # chi2_1 quantile 0.95


def offsets(sd: float, span: float, fine: int, coarse: int) -> np.ndarray:
    """One side of the two-scale grid in ln p (positive, increasing)."""
    top = float(np.log(span))
    near = min(4.0 * sd, top) if np.isfinite(sd) and sd > 0 else min(0.4, top)
    a = np.linspace(0.0, near, fine + 1)[1:]
    b = np.geomspace(near, top, coarse + 1)[1:] if near < top else np.array([])
    return np.concatenate([a, b])


def profile(
    spec: dict,
    data: dict,
    unknown: list[str],
    fit: dict,
    param: str,
    span: float,
    fine: int,
    coarse: int,
    model: str = "analytic",
) -> dict[str, Any]:
    """Profile of `param` over [p_hat / span, p_hat * span] around the estimate (two-scale grid)."""
    t0 = time.perf_counter()
    n = len(data["T"])
    x_hat = float(np.log(fit["estimates"][param]))
    off = offsets(float((fit.get("sd_log") or {}).get(param, np.inf)), span, fine, coarse)
    half = len(off)
    others = [k for k in unknown if k != param]
    grid = x_hat + np.concatenate([-off[::-1], [0.0], off])
    chi = np.full(len(grid), np.nan)
    chi[half] = fit["chi2_per_reading"] * n
    nuis: list[dict | None] = [None] * len(grid)
    nuis[half] = {k: fit["estimates"][k] for k in others}
    failed = 0
    mle = {k: fit["estimates"][k] for k in others}

    def solve(i: int, starts: list[dict]) -> dict:
        """Conditional fit at grid point i from every start given; the lowest chi2 wins."""
        s = copy.deepcopy(spec)
        s["parameters"][param] = float(np.exp(grid[i]))
        if not others:  # a single unknown: the profile is the likelihood itself
            try:
                return _chi_fixed(s, data, param, model)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError, MemoryError):
                return {"estimates": None}
        best = {"estimates": None}
        for st in starts:
            r = C.safe_fit(s, data, others, st, model, with_cov=False)
            if r["estimates"] is not None and (
                best["estimates"] is None or r["chi2_per_reading"] < best["chi2_per_reading"]
            ):
                best = r
        return best

    for direction in (-1, 1):
        start = dict(mle)
        for i in range(half + direction, half + direction * (half + 1), direction):
            # two starts: continuation from the previous point and the overall best fit (a single continuation start
            # can stick in a poor conditional minimum and leave a spurious spike that closes an open interval)
            res = solve(i, [start] if start == mle else [start, mle])
            if res["estimates"] is None:
                failed += 1
                continue
            chi[i] = res["chi2_per_reading"] * n
            if others:
                start = {k: res["estimates"][k] for k in others}
                nuis[i] = dict(start)
    repaired = 0
    if others:  # inward pass: each point re-fitted from its OUTER neighbour's solution; a better branch found further
        # out (e.g. another nuisance regime) propagates inward, so the profile is the minimum over both sweeps
        for direction in (-1, 1):
            far = half + direction * half
            for i in range(far - direction, half, -direction):
                o = i + direction
                if nuis[o] is None:
                    continue
                res = solve(i, [nuis[o]])
                if res["estimates"] is not None and (
                    not np.isfinite(chi[i]) or res["chi2_per_reading"] * n < chi[i] - 1e-9
                ):
                    chi[i] = res["chi2_per_reading"] * n
                    nuis[i] = {k: res["estimates"][k] for k in others}
                    repaired += 1
    cmin = float(np.nanmin(chi))
    dchi = chi - cmin
    ci = _interval(grid, dchi, half)
    return {
        "param": param,
        "grid_log": grid.tolist(),
        "chi2": chi.tolist(),
        "delta_chi2": dchi.tolist(),
        "chi2_min": cmin,
        "ci95_log": ci,
        "bounded": {"lower": ci[0] is not None, "upper": ci[1] is not None},
        "max_delta_chi2": float(np.nanmax(dchi)),
        "nuisance": nuis,
        "failed": failed,
        "repaired": repaired,
        "wall_s": time.perf_counter() - t0,
    }


def _chi_fixed(spec: dict, data: dict, param: str, model: str) -> dict:
    """chi2 of the model with every parameter fixed (a single unknown: the profile is the likelihood itself)."""
    from aerosynapse.thermal.inverse import uq as U

    f = U.model_fn(spec, data, [param], model)
    r = (f(np.log([float(spec["parameters"][param])])) - data["T"]) / data["sigma"]
    return {"estimates": {param: float(spec["parameters"][param])}, "chi2_per_reading": float(r @ r) / len(r)}


def _interval(grid: np.ndarray, dchi: np.ndarray, centre: int) -> list[float | None]:
    """Crossings of delta chi2 = 3.84 on each side of the minimum (linear interpolation); None = not crossed."""
    i0 = int(np.nanargmin(dchi)) if np.any(np.isfinite(dchi)) else centre
    out: list[float | None] = []
    for direction in (-1, 1):
        val = None
        last = i0  # last finite point (failed fits are skipped)
        j = i0 + direction
        while 0 <= j < len(grid):
            if np.isfinite(dchi[j]):
                if dchi[j] >= DCHI2_95:
                    a, b = dchi[last], dchi[j]
                    w = (DCHI2_95 - a) / (b - a) if b != a else 0.0
                    val = float(grid[last] + w * (grid[j] - grid[last]))
                    break
                last = j
            j += direction
        out.append(val)
    return out
