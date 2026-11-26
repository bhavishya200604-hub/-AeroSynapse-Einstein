"""Convergence-order utilities (observed order, least-squares fits)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def observed_orders(h: ArrayLike, err: ArrayLike) -> np.ndarray:
    """Pairwise orders p_i = log(e_i/e_{i+1}) / log(h_i/h_{i+1})."""
    h, e = np.asarray(h, float), np.asarray(err, float)
    return np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])


def fit_order(h: ArrayLike, err: ArrayLike) -> tuple[float, float]:
    """Least-squares fit log e = log C + p log h. Returns (p, C)."""
    p, logC = np.polyfit(np.log(np.asarray(h, float)), np.log(np.asarray(err, float)), 1)
    return float(p), float(np.exp(logC))


def richardson(f_fine: float, f_coarse: float, r: float, p: float) -> float:
    """Richardson extrapolation f_ext = (r^p f_fine - f_coarse) / (r^p - 1)."""
    return (r**p * f_fine - f_coarse) / (r**p - 1.0)


def gci(
    f1: float,
    f2: float,
    f3: float,
    h1: float,
    h2: float,
    h3: float,
    Fs: float = 1.25,
    p_formal: float | None = None,
    max_iter: int = 200,
) -> dict[str, float | bool | str]:
    """Grid Convergence Index, three-grid procedure (Celik et al. 2008 / Roache), f1 on the FINEST grid.

    r21 = h2/h1, r32 = h3/h2; eps21 = f2 - f1, eps32 = f3 - f2, s = sign(eps32/eps21).
    Apparent order p solves p = |ln|eps32/eps21| + q(p)| / ln r21, q(p) = ln((r21^p - s)/(r32^p - s))
    (fixed-point iteration; q = 0 for constant r). Returns p, extrapolated value, relative errors,
    GCI_fine21 = Fs e_a21/(r21^p - 1), GCI_coarse32, the asymptotic-range ratio GCI32/(r21^p GCI21) and the
    convergence type (monotone / oscillatory / divergent).
    """
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    out: dict[str, float | bool | str] = {"r21": r21, "r32": r32, "f1": f1, "f2": f2, "f3": f3}
    if e21 == 0.0 or e32 == 0.0:
        out.update(type="exact/indeterminate", p=float("nan"))
        return out
    R = e21 / e32
    out["type"] = "monotone" if 0 < R < 1 else ("oscillatory" if -1 < R < 0 else "divergent")
    s = np.sign(e32 / e21)
    p = 2.0 if p_formal is None else p_formal
    for _ in range(max_iter):
        q = np.log((r21**p - s) / (r32**p - s))
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-12:
            p = p_new
            break
        p = p_new
    rp = r21**p
    f_ext = (rp * f1 - f2) / (rp - 1)
    ea21 = abs((f1 - f2) / f1) if f1 != 0 else float("nan")
    ea32 = abs((f2 - f3) / f2) if f2 != 0 else float("nan")
    gci21 = Fs * ea21 / (rp - 1)
    gci32 = Fs * ea32 / (r32**p - 1)
    out.update(
        p=float(p),
        f_ext=float(f_ext),
        e_a21=float(ea21),
        e_ext21=float(abs((f_ext - f1) / f_ext)),
        GCI_fine21=float(gci21),
        GCI_coarse32=float(gci32),
        asymptotic_ratio=float(gci32 / (rp * gci21)) if gci21 > 0 else float("nan"),
    )
    return out


def error_norms(num: ArrayLike, ref: ArrayLike, dx: float | None = None) -> dict[str, float]:
    """Absolute L2 (grid-weighted if dx given, else RMS), Linf and relative L2."""
    d = np.asarray(num, float) - np.asarray(ref, float)
    r = np.asarray(ref, float)
    w = dx if dx is not None else 1.0 / d.size
    l2 = float(np.sqrt(np.sum(d**2) * w))
    return {
        "L2": l2,
        "Linf": float(np.max(np.abs(d))),
        "relL2": float(np.sqrt(np.sum(d**2)) / max(np.sqrt(np.sum(r**2)), 1e-300)),
    }
