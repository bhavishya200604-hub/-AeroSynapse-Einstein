"""Flow scenarios: named regimes plus random extreme draws.

The named list covers laminar, transitional, turbulent, contaminated, gust and near-transonic
states. The random draws are built from physical quantities (chord, density, Mach) rather than
independent Re and M, so that impossible states such as Re = 5e4 at M = 0.7 do not appear.

The optimiser only ever sees one fixed set of draws. Reported results use a different seed, so
that a design cannot look good by fitting the particular draws it was trained on.

n_crit follows Mack's relation N = -8.43 - 2.4 ln(Tu), so 9 is a low-turbulence tunnel
(Tu ~ 0.07 %) and 1.5 is a very disturbed stream (Tu ~ 1.8 %).
"""

from __future__ import annotations

import numpy as np

from aerosynapse.shapeopt.aero2d import Conditions

#         name                 Re     Mach   cl    n_crit xtr   regime
NAMED = [
    ("laminar_low_Re", 6e4, 0.02, 0.50, 11.0, 1.0, "laminar"),
    ("laminar_separation_bubble", 2e5, 0.05, 0.80, 11.0, 1.0, "laminar/LSB"),
    ("transitional", 6e5, 0.10, 0.50, 9.0, 1.0, "transitional"),
    ("turbulent_cruise", 6e6, 0.25, 0.35, 9.0, 1.0, "turbulent"),
    ("turbulent_high_Re", 2e7, 0.50, 0.40, 9.0, 1.0, "turbulent"),
    ("high_freestream_turbulence", 2e6, 0.20, 0.60, 1.5, 1.0, "bypass transition"),
    ("contaminated_leading_edge", 3e6, 0.20, 0.60, 9.0, 0.03, "tripped"),
    ("dirty_high_lift", 3e6, 0.15, 1.10, 9.0, 0.03, "tripped, near stall"),
    ("gust_upward", 4e6, 0.25, 1.20, 4.0, 1.0, "gust, near stall"),
    ("gust_downward", 4e6, 0.25, -0.30, 4.0, 1.0, "negative load"),
    ("near_transonic", 1e7, 0.68, 0.40, 9.0, 1.0, "compressible"),
]


def named() -> Conditions:
    a = np.array([r[1:6] for r in NAMED], float)
    return Conditions(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], [r[0] for r in NAMED])


def random_extreme(n: int, seed: int) -> Conditions:
    """Random states, weighted towards the edges of the envelope."""
    r = np.random.default_rng(seed)
    chord = np.exp(r.uniform(np.log(0.15), np.log(3.0), n))
    sigma = r.uniform(0.3, 1.0, n)  # density ratio (altitude)
    mach = np.where(r.random(n) < 0.25, r.uniform(0.55, 0.72, n), r.uniform(0.02, 0.55, n))
    Re = np.clip(2.3e7 * mach * chord * sigma, 4e4, 3e7)
    cl = np.where(r.random(n) < 0.3, r.uniform(0.9, 1.3, n), r.uniform(0.1, 0.9, n))
    cl = np.minimum(cl, 0.45 / np.maximum(mach, 0.3) ** 2 * 0.6 + 0.2)  # lift falls with Mach in real flight
    ncrit = np.where(r.random(n) < 0.35, r.uniform(1.0, 4.0, n), r.uniform(6.0, 12.0, n))
    xtr = np.where(r.random(n) < 0.25, r.uniform(0.02, 0.10, n), 1.0)
    return Conditions(Re, mach, cl, ncrit, xtr, [f"random_{i}" for i in range(n)])


def concat(a: Conditions, b: Conditions) -> Conditions:
    return Conditions(
        np.r_[a.Re, b.Re], np.r_[a.mach, b.mach], np.r_[a.cl, b.cl], np.r_[a.ncrit, b.ncrit], np.r_[a.xtr, b.xtr], a.names + b.names
    )
