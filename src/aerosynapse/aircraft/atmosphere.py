r"""International Standard Atmosphere with a temperature offset, plus air properties.

Layers to 47 km from the ISA lapse rates. ``dT`` shifts temperature at fixed pressure, which is how
a hot or cold day is defined, so density and speed of sound change while pressure altitude does not.
Viscosity follows Sutherland's law and conductivity follows from a constant Prandtl number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G0 = 9.80665
R_AIR = 287.053
GAMMA = 1.4
CP_AIR = GAMMA * R_AIR / (GAMMA - 1)
PR = 0.71
SIGMA_SB = 5.670374e-8
_LAYERS = ((0.0, 288.15, 101325.0, -0.0065), (11000.0, 216.65, 22632.06, 0.0), (20000.0, 216.65, 5474.889, 0.001), (32000.0, 228.65, 868.0187, 0.0028))


def sutherland(T: float) -> float:
    """Dynamic viscosity of air, Pa s."""
    return float(1.458e-6 * T**1.5 / (T + 110.4))


@dataclass(frozen=True)
class Air:
    h: float
    T: float
    p: float
    rho: float
    a: float
    mu: float

    @property
    def nu(self) -> float:
        return self.mu / self.rho

    @property
    def k(self) -> float:
        return self.mu * CP_AIR / PR


def isa(h: float, dT: float = 0.0) -> Air:
    """Air properties at geopotential altitude ``h`` (m) with a temperature offset ``dT`` (K)."""
    if not -500.0 <= h <= 47000.0:
        raise ValueError(f"altitude {h} m is outside the model range [-500, 47000]")
    b, T0, p0, L = _LAYERS[0]
    for lay in _LAYERS:
        if h >= lay[0]:
            b, T0, p0, L = lay
    dh = h - b
    if abs(L) > 1e-12:
        T = T0 + L * dh
        p = p0 * (T / T0) ** (-G0 / (L * R_AIR))
    else:
        T = T0
        p = p0 * np.exp(-G0 * dh / (R_AIR * T0))
    Tt = T + dT
    return Air(h, float(Tt), float(p), float(p / (R_AIR * Tt)), float(np.sqrt(GAMMA * R_AIR * Tt)), sutherland(Tt))
