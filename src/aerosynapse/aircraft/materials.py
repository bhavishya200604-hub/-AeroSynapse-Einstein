r"""Structural materials: temperature-dependent properties, per-zone selection and tolerance effects.

Selection uses Ashby performance indices for a panel of given load and size: strength-limited mass
scales with :math:`\rho/\sigma(T)`, panel-buckling (stiffness) limited mass with :math:`\rho/E(T)^{1/3}`,
fatigue-limited mass with :math:`\rho/\sigma_f`. A material is only a candidate in a zone if its long-term
service limit is above the hottest temperature the zone reaches, with a margin.

Tolerances: a Monte Carlo over manufacturing tolerances (skin thickness, surface roughness, steps and
gaps at joints) and thermal mismatch at joints gives the spread in mass, drag and stress margin.

Data quality: values are typical handbook figures (room-temperature and elevated-temperature fractions)
at the level of public data sheets, rounded. Each entry says where it came from and whether it was checked
in this project. They are for comparing options, not for certifying a design: upload real allowables
(B-basis) through the ``materials`` data slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aerosynapse.aircraft.atmosphere import Air
from aerosynapse.aircraft.layers import friction


@dataclass(frozen=True)
class Mat:
    key: str
    name: str
    rho: float  # kg/m^3
    E: float  # Pa, room temperature
    sigma: float  # Pa, design allowable (yield or laminate allowable after damage knock-down)
    sigma_f: float  # Pa, fatigue strength at about 1e7 cycles (or laminate equivalent)
    T_max: float  # K, long-term service limit
    alpha: float  # thermal expansion, 1/K
    k: float  # conductivity, W/m/K
    cp: float  # J/kg/K
    emissivity: float
    cost: float  # relative cost per kg, aluminium = 1
    tol_mm: float  # typical skin thickness tolerance, mm (1 sigma)
    rough_um: float  # as-made surface roughness Ra, micrometres
    curve: tuple[tuple[float, float, float], ...]  # (T K, strength fraction, modulus fraction)
    source: str
    verified: str = "not checked in this project"

    def strength(self, T: float) -> float:
        t = np.array(self.curve)
        return float(self.sigma * np.interp(T, t[:, 0], t[:, 1]))

    def modulus(self, T: float) -> float:
        t = np.array(self.curve)
        return float(self.E * np.interp(T, t[:, 0], t[:, 2]))

    def diffusivity(self) -> float:
        return self.k / (self.rho * self.cp)


HANDBOOK = "typical handbook / public data-sheet values, rounded"
LIBRARY: dict[str, Mat] = {m.key: m for m in [
    Mat("al7075", "Aluminium 7075-T6", 2810, 71.7e9, 503e6, 160e6, 393, 23.6e-6, 130, 960, 0.2, 1.0, 0.05, 0.8,
        ((293, 1, 1), (373, 0.90, 0.96), (423, 0.70, 0.92), (473, 0.40, 0.85), (523, 0.20, 0.75)), HANDBOOK),
    Mat("al2024", "Aluminium 2024-T3", 2780, 73.1e9, 345e6, 140e6, 423, 23.2e-6, 121, 875, 0.2, 0.9, 0.05, 0.8,
        ((293, 1, 1), (373, 0.92, 0.96), (423, 0.82, 0.92), (473, 0.60, 0.86), (523, 0.35, 0.78)), HANDBOOK),
    Mat("al_li", "Aluminium-lithium 2099", 2630, 78e9, 470e6, 160e6, 393, 22e-6, 85, 950, 0.2, 2.5, 0.05, 0.8,
        ((293, 1, 1), (373, 0.90, 0.96), (423, 0.72, 0.92), (473, 0.45, 0.85)), HANDBOOK),
    Mat("ti64", "Titanium Ti-6Al-4V (annealed)", 4430, 113.8e9, 880e6, 510e6, 673, 8.6e-6, 6.7, 526, 0.4, 6.0, 0.05, 1.6,
        ((293, 1, 1), (473, 0.80, 0.93), (573, 0.72, 0.89), (673, 0.66, 0.85), (773, 0.55, 0.80), (873, 0.40, 0.72)), HANDBOOK),
    Mat("ti6242", "Titanium Ti-6242", 4540, 114e9, 990e6, 550e6, 813, 7.7e-6, 7.0, 460, 0.4, 9.0, 0.05, 1.6,
        ((293, 1, 1), (573, 0.80, 0.90), (673, 0.75, 0.87), (773, 0.70, 0.83), (873, 0.62, 0.78)), HANDBOOK),
    Mat("in718", "Nickel alloy Inconel 718", 8190, 200e9, 1100e6, 600e6, 923, 13e-6, 11.4, 435, 0.5, 8.0, 0.05, 1.6,
        ((293, 1, 1), (366, 0.97, 0.98), (922, 0.87, 0.83), (1023, 0.65, 0.78), (1123, 0.35, 0.72)),
        "Special Metals / Fort Wayne Metals data sheets (typical)",
        "density 8.19 g/cm3 and the ~13 % yield and ~17 % modulus loss from 93 to 649 C checked against Fort Wayne Metals (web search, 2026-11-07)"),
    Mat("cfrp_im7", "Carbon/epoxy IM7/8552, quasi-isotropic", 1570, 60e9, 300e6, 200e6, 394, 2.5e-6, 5.0, 900, 0.85, 4.0, 0.08, 0.4,
        ((293, 1, 1), (353, 0.92, 0.97), (394, 0.80, 0.93), (430, 0.50, 0.80), (460, 0.20, 0.50)),
        "Hexcel HexPly 8552 data sheet (density, 180 C cure, service to 121 C); laminate allowables after damage knock-down are typical",
        "density 1.57 g/cm3 and 121 C service limit checked against Hexcel (web search, 2026-11-07)"),
    Mat("cfrp_bmi", "Carbon/bismaleimide (BMI)", 1600, 62e9, 300e6, 190e6, 477, 3e-6, 5.0, 950, 0.85, 7.0, 0.08, 0.4,
        ((293, 1, 1), (423, 0.92, 0.96), (477, 0.82, 0.92), (520, 0.50, 0.75)), HANDBOOK),
    Mat("cmc_csic", "Ceramic matrix composite C/SiC", 2100, 90e9, 200e6, 120e6, 1650, 3e-6, 15.0, 1200, 0.85, 40.0, 0.10, 3.2,
        ((293, 1, 1), (1273, 1.0, 0.95), (1673, 0.85, 0.85), (1873, 0.60, 0.70)), HANDBOOK),
    Mat("steel_300m", "Steel 300M (gear, fittings)", 7830, 205e9, 1690e6, 700e6, 573, 12e-6, 32, 470, 0.6, 3.0, 0.03, 0.8,
        ((293, 1, 1), (473, 0.92, 0.95), (573, 0.85, 0.92), (673, 0.70, 0.87)), HANDBOOK),
    Mat("glass_epoxy", "Glass/epoxy (radome)", 1900, 25e9, 250e6, 90e6, 393, 12e-6, 0.3, 1000, 0.9, 1.5, 0.10, 0.6,
        ((293, 1, 1), (353, 0.88, 0.95), (393, 0.75, 0.88)), HANDBOOK),
]}

AL_REF = LIBRARY["al7075"]


@dataclass
class Zone:
    key: str
    name: str
    mode: str  # strength | stiffness | fatigue
    area_frac: float  # share of the structural mass in this zone
    T_extra: float = 0.0  # K added to the aerodynamic temperature (engine heat, plume)
    allowed: tuple[str, ...] = ()
    note: str = ""


ZONES = [
    Zone("radome", "Nose radome", "stiffness", 0.02, 0.0, ("glass_epoxy", "cfrp_bmi"), "must pass radar signals: only dielectric materials"),
    Zone("fwd_fuselage", "Forward fuselage skin", "stiffness", 0.14),
    Zone("centre_fuselage", "Centre fuselage frames and skin", "strength", 0.20),
    Zone("wing_box", "Wing box (spars, ribs, covers)", "strength", 0.24),
    Zone("wing_le", "Wing and strake leading edges", "strength", 0.03, 0.0, (), "stagnation temperature and rain/hail erosion"),
    Zone("control", "Control surfaces and tails", "stiffness", 0.12),
    Zone("aft_fuselage", "Aft fuselage around the engine", "strength", 0.15, 180.0, (), "engine bay heat adds about 180 K"),
    Zone("nozzle", "Nozzle and exhaust surround", "strength", 0.04, 650.0, ("in718", "ti6242", "cmc_csic"), "afterburner exhaust"),
    Zone("gear", "Landing gear and main fittings", "fatigue", 0.06, 0.0, ("steel_300m", "ti64"), ""),
]


def index(m: Mat, T: float, mode: str) -> float:
    """Ashby mass index (lower is lighter) at temperature T, relative to 7075 at room temperature."""
    if mode == "stiffness":
        return (m.rho / m.modulus(T) ** (1 / 3)) / (AL_REF.rho / AL_REF.E ** (1 / 3))
    if mode == "fatigue":
        return (m.rho / m.sigma_f) / (AL_REF.rho / AL_REF.sigma_f)
    return (m.rho / max(m.strength(T), 1.0)) / (AL_REF.rho / AL_REF.sigma)


@dataclass
class Choice:
    zone: str
    name: str
    material: str
    T_K: float
    index: float
    margin_K: float
    ranking: list[dict[str, Any]] = field(default_factory=list)


def select(zone_temps: dict[str, float], margin: float = 20.0, cost_weight: float = 0.02, exclude: tuple[str, ...] = ()) -> list[Choice]:
    """Pick the lightest feasible material for every zone; ties broken by a small cost penalty."""
    out = []
    for z in ZONES:
        T = zone_temps.get(z.key, 293.0) + z.T_extra
        pool = [LIBRARY[k] for k in (z.allowed or LIBRARY) if k not in exclude]
        if not z.allowed:
            pool = [m for m in pool if m.key not in ("steel_300m", "glass_epoxy", "cmc_csic") or T > 900]
        rank = []
        for m in pool:
            feasible = m.T_max >= T + margin
            score = index(m, T, z.mode) * (1 + cost_weight * np.log(m.cost))
            rank.append({"material": m.key, "name": m.name, "feasible": feasible, "index": round(index(m, T, z.mode), 4), "score": round(score, 4), "margin_K": round(m.T_max - T, 1)})
        rank.sort(key=lambda r: (not r["feasible"], r["score"]))
        best = rank[0]
        out.append(Choice(z.key, z.name, best["material"], T, best["index"], best["margin_K"], rank))
    return out


def structure_factor(choices: list[Choice], non_optimum: float = 0.4) -> float:
    """Structural mass relative to an all-7075 airframe. A fixed share (fasteners, systems attachment, minimum
    gauge) does not scale with the material."""
    s = sum(next(z.area_frac for z in ZONES if z.key == c.zone) * c.index for c in choices)
    tot = sum(z.area_frac for z in ZONES)
    return float(non_optimum + (1 - non_optimum) * s / tot)


def tolerance_study(
    choices: list[Choice], air: Air, mach: float, length: float, s_wet: float, skin_mm: float = 2.0, struct_mass: float = 4000.0,
    n: int = 4000, seed: int = 0, step_mm: float = 0.25, gap_mm: float = 1.0, dT: float = 100.0,
) -> dict[str, Any]:
    """Monte Carlo over manufacturing tolerances and thermal mismatch.

    Each sample draws skin thickness errors (per zone, normal with the material's tolerance), surface
    roughness (log-normal about the material's Ra, raised by paint and wear), joint steps and gaps (normal),
    and the gap change from expansion mismatch between neighbouring zones heated by dT. Outputs: mass change,
    drag change (roughness through the rough-plate friction law, steps and gaps through excrescence drag of
    about 0.6 q h per metre of forward-facing step, after Hoerner), and the stress margin at the thinnest skin.
    """
    rng = np.random.default_rng(seed)
    mats = [LIBRARY[c.material] for c in choices]
    fr = np.array([next(z.area_frac for z in ZONES if z.key == c.zone) for c in choices])
    fr = fr / fr.sum()
    tol = np.array([m.tol_mm for m in mats])
    rough = np.array([m.rough_um for m in mats])
    alpha = np.array([m.alpha for m in mats])
    cf0 = friction(air, mach, length, 0.05, roughness=0.0)
    q = 0.5 * air.rho * (mach * air.a) ** 2
    seam_len = 0.6 * s_wet  # metres of transverse joints, about one per 1.7 m^2 of skin
    dm, dd, marg = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        t_err = rng.normal(0, tol)  # mm, per zone
        dm[i] = struct_mass * float(np.sum(fr * t_err / skin_mm)) * 0.5  # skins are about half the structure
        ra = float(np.sum(fr * rough * np.exp(rng.normal(0.3, 0.4, len(rough)))))  # paint and wear raise Ra
        ks = 6e-6 * ra  # equivalent sand-grain roughness about 6 Ra (micrometre -> m scaling)
        cf = friction(air, mach, length, 0.05, roughness=ks)
        step = abs(rng.normal(0, step_mm)) * 1e-3
        mis = abs(float(np.sum(np.diff(alpha) * rng.normal(0, 1, len(alpha) - 1)))) * dT * 1.0  # m per m of joint span
        gap = abs(rng.normal(gap_mm * 1e-3, 0.3e-3)) + mis
        D_rough = (cf - cf0) * q * s_wet
        D_steps = 0.6 * q * step * seam_len * 0.2 + 0.02 * q * gap * seam_len
        dd[i] = D_rough + D_steps
        marg[i] = (skin_mm + float(np.min(t_err))) / skin_mm - 1  # stress margin change at the thinnest skin
    D0 = cf0 * q * s_wet
    p = lambda a, k: float(np.percentile(a, k))  # noqa: E731
    return {
        "mass_kg": {"mean": float(dm.mean()), "p05": p(dm, 5), "p95": p(dm, 95)},
        "drag_N": {"mean": float(dd.mean()), "p05": p(dd, 5), "p95": p(dd, 95), "relative_to_friction_mean": float(dd.mean() / D0)},
        "stress_margin": {"mean": float(marg.mean()), "p05": p(marg, 5)},
        "samples": n,
        "friction_drag_N": float(D0),
    }
