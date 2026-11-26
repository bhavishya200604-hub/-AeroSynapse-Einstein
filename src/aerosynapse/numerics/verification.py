r"""Verification toolkit: observed order, Richardson extrapolation, GCI, von Neumann analysis.

References for the formulae (verified in docs/LITERATURE_AND_NOVELTY.md): Roache's grid
convergence index with safety factor 1.25 for three-grid studies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]


@dataclass
class ConvergenceResult:
    h: list[float]
    errors: list[float]
    pairwise_orders: list[float]
    fitted_order: float

    def as_dict(self) -> dict[str, list[float] | float]:
        return {
            "h": self.h,
            "errors": self.errors,
            "pairwise_orders": self.pairwise_orders,
            "fitted_order": self.fitted_order,
        }


def observed_order(h: Sequence[float], errors: Sequence[float]) -> ConvergenceResult:
    """Observed order ``p`` from ``e ≈ C h^p``: pairwise ``log(e1/e2)/log(h1/h2)`` and a LSQ fit."""
    hh, ee = np.asarray(h, float), np.asarray(errors, float)
    if np.any(ee <= 0):
        raise ValueError("errors must be positive")
    pw = list(np.log(ee[:-1] / ee[1:]) / np.log(hh[:-1] / hh[1:]))
    fit = float(np.polyfit(np.log(hh), np.log(ee), 1)[0])
    return ConvergenceResult(list(map(float, hh)), list(map(float, ee)), [float(p) for p in pw], fit)


def convergence_study(error_fn: Callable[[int], float], ns: Sequence[int], length: float = 1.0) -> ConvergenceResult:
    """Run ``error_fn(n)`` for each resolution and compute observed orders (h = length/n)."""
    return observed_order([length / n for n in ns], [error_fn(n) for n in ns])


def richardson(f_fine: float, f_coarse: float, r: float, p: float) -> float:
    """Richardson extrapolate ``f_exact ≈ f_fine + (f_fine - f_coarse)/(r^p - 1)``."""
    return float(f_fine + (f_fine - f_coarse) / (r**p - 1.0))


@dataclass
class GCIResult:
    p: float
    f_extrapolated: float
    gci_fine: float
    gci_coarse: float
    asymptotic_ratio: float


def gci(f1: float, f2: float, f3: float, r: float, Fs: float = 1.25) -> GCIResult:
    """Three-grid GCI (f1 finest, constant ratio r). ``asymptotic_ratio`` ≈ 1 indicates the
    asymptotic range. Raises ``ValueError`` for oscillatory convergence (sign change)."""
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0 or e32 / e21 <= 0:
        raise ValueError("oscillatory or exact convergence: GCI not defined")
    p = float(np.log(e32 / e21) / np.log(r))
    gf = Fs * abs(e21 / f1) / (r**p - 1) if f1 != 0 else Fs * abs(e21) / (r**p - 1)
    gc = Fs * abs(e32 / f2) / (r**p - 1) if f2 != 0 else Fs * abs(e32) / (r**p - 1)
    return GCIResult(p, richardson(f1, f2, r, p), float(gf), float(gc), float(gc / (r**p * gf)))


def amplification_factor(theta: float, r: float, c: float, scheme: str, kh: FloatArr) -> FloatArr:
    r"""von Neumann amplification factor of the θ-method for :math:`u_t+vu_x=Du_{xx}`.

    ``r = DΔt/h²``, ``c = vΔt/h``. Spatial symbol λ(kh) (times Δt):
    central :math:`-4r\sin^2(kh/2) - i c\sin kh`; upwind adds numerical diffusion
    :math:`-c(1-\cos kh) - ic\sin kh`. :math:`G=(1+(1-θ)λ)/(1-θλ)`."""
    lam: NDArray[np.complex128] = (-4 * r * np.sin(kh / 2) ** 2).astype(np.complex128)
    lam = lam - 1j * c * np.sin(kh) if scheme == "central" else lam - c * (1 - np.cos(kh)) - 1j * c * np.sin(kh)
    return np.asarray(np.abs((1 + (1 - theta) * lam) / (1 - theta * lam)))


def is_von_neumann_stable(theta: float, r: float, c: float, scheme: str, n: int = 721) -> bool:
    kh = np.linspace(0, np.pi, n)
    return bool(np.max(amplification_factor(theta, r, c, scheme, kh)) <= 1 + 1e-12)


def rel_l2(u: FloatArr, ref: FloatArr) -> float:
    return float(np.linalg.norm(u - ref) / np.linalg.norm(ref))


def discrete_l2(u: FloatArr, ref: FloatArr, h: float) -> float:
    return float(np.sqrt(h * np.sum((u - ref) ** 2)))
