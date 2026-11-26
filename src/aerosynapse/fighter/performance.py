r"""Fighter performance in every flight situation, from one design.

Pipeline for one design
  geometry kernel  -> volume, wetted area, normal and Mach-plane area distributions, envelope fits
  vortex lattice   -> lift slope, induced drag, neutral point, directional and roll stability, vortex lift
  drag build-up    -> friction (Eckert reference temperature, transition, roughness from the materials),
                      form factors (Raymer), transonic drag rise (Korn), supersonic wave drag from the
                      Mach-plane area distributions of the real 3D shape (Harris/Eminton-Lord), lift-dependent
                      wave drag, leading-edge suction (Raymer's suction method)
  high incidence   -> Polhamus vortex lift with vortex breakdown, attached-flow stall, dynamic directional
                      stability Cn_beta,dyn with fin blanketing
  propulsion       -> afterburning low-bypass turbofan, Mattingly installed thrust lapse and fuel flow
  weights          -> Raymer fighter group weights, scaled by the selected materials, closed on take-off weight
                      with the mission fuel; calibrated once to a public empty mass
  situations       -> take-off, approach, cruise, climb, sustained and instantaneous turns, supercruise, dash,
                      maximum dynamic pressure, high incidence, crosswind, gusts, turbulence, hot and cold days,
                      and random extreme states

Model sources are named where used; empirical constants quoted from memory are marked so in the code. This
is a conceptual-design model: its job is to rank shapes consistently. The 3D flow solver checks the chosen
designs (:mod:`aerosynapse.cfd`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from aerosynapse.aircraft import materials as mt
from aerosynapse.aircraft.analysis import wave_drag_area
from aerosynapse.aircraft.atmosphere import G0, Air, isa
from aerosynapse.aircraft.layers import friction, t_adiabatic
from aerosynapse.cad import kernel
from aerosynapse.fighter import design as D
from aerosynapse.fighter import vlm

LB, FT = 0.45359237, 0.3048


@dataclass
class Mission:
    radius_m: float = 500e3
    payload_kg: float = 1500.0
    crew_kg: float = 250.0  # pilot, seat, life support
    avionics_kg: float = 1000.0
    dash_m: float = 50e3  # supersonic leg inside the radius, each way
    combat_s: float = 120.0  # at maximum thrust
    reserve_s: float = 1200.0  # loiter
    n_ult: float = 13.5  # 9 g limit times 1.5
    field_m: float = 900.0
    v_approach_max: float = 86.0  # m/s (the reference design flies its approach at 85)
    flap_dCL: float = 0.25  # leading- and trailing-edge flaps at take-off and landing
    cg_reach: float = 0.05  # internal layout can move the centre of gravity by this share of the length
    turbulence_sigma: float = 6.0  # m/s rms, severe low-level turbulence
    span_max: float = 14.0
    length_max: float = 20.0
    fuel_density: float = 800.0
    calib: float = 1.0  # empty-mass calibration factor


@dataclass(frozen=True)
class Situation:
    key: str
    name: str
    mach: float
    altitude: float
    n: float = 1.0
    alpha: float = float("nan")  # set for fixed-incidence states
    beta: float = 0.0
    dT: float = 0.0
    gust: float = 0.0  # derived equivalent gust speed, m/s
    turbulence: float = 0.0  # vertical velocity rms, m/s
    thrust: str = "max"
    fuel_frac: float = 0.5  # share of mission fuel on board


SITUATIONS: tuple[Situation, ...] = (
    Situation("takeoff", "Take-off, sea level", 0.2, 0.0, fuel_frac=1.0),
    Situation("hot_takeoff", "Take-off on a hot day (+35 K, 1500 m field)", 0.2, 1500.0, dT=35.0, fuel_frac=1.0),
    Situation("approach", "Landing approach", 0.2, 0.0, fuel_frac=0.15),
    Situation("crosswind", "Approach with 12 deg sideslip", 0.2, 0.0, beta=12.0, fuel_frac=0.15),
    Situation("cruise", "Subsonic cruise M0.85 at 11 km", 0.85, 11000.0, thrust="mil"),
    Situation("climb", "Climb M0.9 at 6 km (specific excess power)", 0.9, 6000.0),
    Situation("turn_sustained", "Sustained turn M0.9 at 4.6 km", 0.9, 4600.0),
    Situation("turn_instant", "Instantaneous turn M0.5 at 4.6 km", 0.5, 4600.0),
    Situation("supercruise", "Supercruise M1.4 at 11 km, dry thrust", 1.4, 11000.0, thrust="mil"),
    Situation("dash", "Dash M1.8 at 12 km", 1.8, 12000.0),
    Situation("cold_high", "M1.6 at 15 km on a cold day (-20 K)", 1.6, 15000.0, dT=-20.0),
    Situation("max_q", "Maximum dynamic pressure M1.2 at 300 m", 1.2, 300.0, fuel_frac=0.3),
    Situation("high_alpha", "High incidence 35 deg at M0.3, 6 km", 0.3, 6000.0, alpha=35.0, fuel_frac=0.4),
    Situation("gust", "Vertical gust 15 m/s at M0.8, 1 km", 0.8, 1000.0, gust=15.0),
    Situation("turbulence", "Severe low-level turbulence (6 m/s rms) at M0.9, 300 m", 0.9, 300.0, turbulence=6.0),
)


# --------------------------------------------------------------------------- geometry
@dataclass
class GeoProps:
    volume: float
    wetted: float  # body (fuselage, canopy, cuts); lifting surfaces are counted from their planforms
    frontal: float
    planform: float
    length: float
    x: list[float]
    A: list[float]
    mach_plane: dict[str, float]  # D/q from Mach-plane cuts, keyed by Mach
    envelope_violation: float  # m the engine or cockpit stick out of the body (0 = fits)
    duct_volume: float
    bay_volume: float


def _hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def _mach_plane_area(design: dict[str, Any], mach: float, z_lo: float, z_hi: float, n_x: int = 32, n_roll: int = 4, n_y: int = 40, n_z: int = 60) -> float:
    """Average wave-drag area D/q over roll angles from oblique (Mach-plane) cuts of the 3D shape.

    Points on each cutting plane count as inside by a soft fraction of their signed distance,
    clip(0.5 - d / dz, 0, 1), which integrates thin surfaces (a wing a few cm thick) correctly on a coarse
    sampling grid."""
    mu = np.arcsin(1 / mach)
    v = design["variables"]
    L = v["length"]
    half = 0.5 * float(np.sqrt(v["w_area"] * v["w_ar"])) + 0.3
    zc, zh = 0.5 * (z_lo + z_hi), 0.5 * (z_hi - z_lo) + 0.2
    u = np.linspace(-half, half, n_y)
    w = np.linspace(-zh, zh, n_z)
    du, dw = u[1] - u[0], w[1] - w[0]
    U, Wz = np.meshgrid(u, w, indexing="ij")
    pts, keys = [], []
    for k in range(n_roll):
        th = np.pi * k / n_roll  # 0..180 deg is enough: the shape is mirrored
        n = np.array([np.cos(mu), -np.sin(mu) * np.cos(th), -np.sin(mu) * np.sin(th)])  # plane normal
        e1 = np.cross(n, [0.0, 0.0, 1.0])
        e1 = e1 / max(np.linalg.norm(e1), 1e-9)
        e2 = np.cross(n, e1)
        e2 = e2 * np.sign(e2[2] or 1.0)
        ext = max(half, zh) / np.tan(mu)
        for x0 in np.linspace(-ext, L + ext, n_x):
            P = np.array([x0, 0.0, zc]) + U[..., None] * e1 + Wz[..., None] * e2
            pts.append(P.reshape(-1, 3))
            keys.append((k, x0))
    dist = kernel.distance(design, np.vstack(pts)).reshape(len(keys), -1)
    frac = np.clip(0.5 - dist / dw, 0, 1)
    area = frac.sum(1) * du * dw * np.cos(mu)  # projected normal to the axis
    out = []
    for k in range(n_roll):
        idx = [i for i, kk in enumerate(keys) if kk[0] == k]
        x = np.array([keys[i][1] for i in idx])
        a = area[idx]
        nz = np.nonzero(a > 1e-6)[0]
        if len(nz) < 3:
            continue
        sl = slice(max(nz[0] - 1, 0), min(nz[-1] + 2, len(a)))
        out.append(wave_drag_area(x[sl], a[sl]))
    return float(np.mean(out)) if out else 0.0


@lru_cache(maxsize=256)
def _geo_cached(key: str, text: str, h: float) -> GeoProps:
    design = json.loads(text)
    info = kernel.info(design, h=h, nx=80)
    v = design["variables"]
    body_only = {**design, "features": [f for f in design["features"] if f.get("type") not in vlm.WING_TYPES]}
    body = kernel.info(body_only, h=h, nx=8)
    env = D.envelope_points(v)
    pts = np.vstack([env["engine"], env["cockpit"]])
    dist = kernel.distance(D.outer(design), pts)
    viol = float(max(0.0, (dist + 0.02).max()))  # 2 cm skin
    outer_vol = kernel.info(D.outer(design), h=h, nx=8)["volume"] if any(f["id"] in ("inlet_duct", "bay") for f in design["features"]) else info["volume"]
    duct = max(0.0, outer_vol - info["volume"] - (v["bay_volume"] if v["bay_volume"] > D.OPTION_ON["bay_volume"] else 0.0))
    z_lo, z_hi = info["bbox_lo"][2], info["bbox_hi"][2]
    mp = {f"{m:.1f}": _mach_plane_area(design, m, z_lo, z_hi) for m in (1.2, 1.6, 2.0)}
    return GeoProps(info["volume"], body["wetted_area"], info["frontal_area"], info["planform_area"], info["length"], info["area_x"], info["area_A"], mp, viol, duct,
                    v["bay_volume"] if v["bay_volume"] > D.OPTION_ON["bay_volume"] else 0.0)


def geometry(design: dict[str, Any], h: float = 0.08) -> GeoProps:
    return _geo_cached(_hash(design["features"]) + f"{h}", json.dumps(design), h)


# --------------------------------------------------------------------------- propulsion
def thrust_lapse(air: Air, M: float, setting: str, TR: float = 1.07) -> tuple[float, float]:
    """Installed thrust / sea-level static maximum, and TSFC (kg/N/s). Mattingly, Aircraft Engine Design
    (2002), low-bypass mixed-flow turbofan; constants from memory, not checked in this project."""
    th0 = air.T / 288.15 * (1 + 0.2 * M**2)
    d0 = air.p / 101325.0 * (1 + 0.2 * M**2) ** 3.5
    if setting == "max":
        a = d0 if th0 <= TR else d0 * (1 - 3.5 * (th0 - TR) / th0)
        c = (1.6 + 0.27 * M) * np.sqrt(air.T / 288.15)
    else:
        a = 0.6 * d0 if th0 <= TR else 0.6 * d0 * (1 - 3.8 * (th0 - TR) / th0)
        c = (0.9 + 0.30 * M) * np.sqrt(air.T / 288.15)
    return float(max(a, 0.0)), float(c / 3600.0 / G0)  # 1/h -> kg/(N s)


# --------------------------------------------------------------------------- aerodynamics
@dataclass
class Aero:
    S: float
    b: float
    c: float
    AR: float
    x_mac_le: float
    derivs: dict[str, vlm.Derivatives]
    cd0_sub: float
    parts: dict[str, float]
    Mdd: float
    sweep_le: float
    tc: float
    sharp_le: bool
    x_np_sub: float
    Kv: float
    lam_vortex: float
    body_munk: float  # m the body moves the neutral point forward
    vortex: list[tuple[float, float, float]] = field(default_factory=list)  # (Kv share, sweep deg, suction lost) per surface
    strake_ratio: float = 0.0
    stall_attached: float = 14.0


def _surfaces(design: dict[str, Any]) -> list[dict[str, Any]]:
    return vlm.surfaces(design["features"])


def aero_model(design: dict[str, Any], g: GeoProps, mission: Mission, rough_um: float = 1.0) -> Aero:
    v = design["variables"]
    ref = D.reference(design)
    S, b, c = ref.S, ref.b, ref.c_mac
    x_ref = ref.x_mac_le + 0.25 * c
    ders = {f"{m:.2f}": vlm.derivatives(design["features"], m, S, c, b, x_ref, nc=3, ns=8) for m in (0.3, 0.85)}
    air = isa(11000.0)
    M = 0.85
    parts: dict[str, float] = {}
    s_wet_surf = 0.0
    for f in _surfaces(design):
        span, cr, lam = float(f["span"]), float(f["root_chord"]), float(f["taper"])
        n = (2 if f.get("mirror") else 1) * max(int(f.get("pattern", {}).get("count", 1)), 1)
        Sexp = 0.5 * (1 + lam) * cr * span * n
        swet = 2.04 * Sexp
        s_wet_surf += swet
        mac = 2 / 3 * cr * (1 + lam + lam**2) / (1 + lam)
        tc = float(f.get("tc", 0.05))
        sw_m = np.arctan(np.tan(np.radians(float(f.get("sweep", 30)))) - 4 * 0.3 * (1 - lam) / (1 + lam) * cr / max(2 * span, 1e-6))
        ff = (1 + 0.6 / 0.3 * tc + 100 * tc**4) * (1.34 * M**0.18 * np.cos(sw_m) ** 0.28)  # Raymer eq. 12.30
        cf = friction(air, M, mac, 0.05, roughness=6e-6 * rough_um)
        Q = 1.1 if f["id"] == "wing" else 1.05
        parts[f["id"]] = parts.get(f["id"], 0.0) + cf * ff * Q * swet / S
    body_wet = g.wetted
    fr = g.length / np.sqrt(4 * max(g.frontal, 1e-3) / np.pi)
    ff_b = 1 + 60 / fr**3 + fr / 400  # Raymer eq. 12.31
    cf_b = friction(air, M, g.length, 0.03, roughness=6e-6 * rough_um)
    parts["body"] = cf_b * ff_b * body_wet / S
    if g.duct_volume > 0:  # inlet lip, diverter and spillage at cruise
        parts["inlet"] = 0.0012 * (g.duct_volume / 1.5) ** 0.5
    parts["excrescence"] = 0.06 * sum(parts.values())  # gaps, steps, antennas: about 6 % (Raymer's typical range 2-10 %)
    cd0 = float(sum(parts.values()))
    sw = float(v["w_sweep"])
    tc = float(v["w_tc"])
    sw_q = np.arctan(np.tan(np.radians(sw)) - (1 - v["w_taper"]) / ((1 + v["w_taper"]) * v["w_ar"]))
    Mdd = float(0.87 / np.cos(sw_q) - tc / np.cos(sw_q) ** 2 - 0.3 / (10 * np.cos(sw_q) ** 3))  # Korn, CL 0.3
    # body upwash moves the neutral point forward: Munk moment of the fuselage (slender body, k2-k1 ~ 0.9)
    munk = 2 * 0.9 * g.volume * 0.6 / (S * c)  # per rad, about 60 % of the body volume is fuselage
    d = ders["0.30"]
    body_munk = munk / max(d.CL_alpha, 1e-6) * c
    # vortex lift per horizontal surface (Polhamus), with its own sweep and breakdown
    vort = []
    for k, f in enumerate(_surfaces(design)):
        if abs(float(f.get("dihedral", 0))) > 60 or k >= len(d.CL_surf_alpha):
            continue
        span_f = float(f["span"]) * (2 if f.get("mirror") else 1)
        Sf = 0.5 * (1 + float(f["taper"])) * float(f["root_chord"]) * span_f
        ARf = span_f**2 / max(Sf, 1e-9)
        kp = d.CL_surf_alpha[k] * S / max(Sf, 1e-9)
        swf = float(f.get("sweep", 30))
        kv = max((kp - kp**2 / (np.pi * ARf)) / np.cos(np.radians(swf)), 0.0) * Sf / S
        sharp = f.get("profile") in ("biconvex", "diamond") or float(f.get("tc", 0.05)) < 0.04
        vort.append((float(kv), swf, 1.0 if sharp else 0.3))
    s_ratio = v["s_size"] if v["s_size"] > D.OPTION_ON["s_size"] else 0.0
    stall = 13 + 100 * max(tc - 0.03, 0) + 4 * max(3.0 - b * b / S, 0) + min(12.0, 150 * s_ratio) + (5.0 if v["c_size"] > D.OPTION_ON["c_size"] else 0.0)
    return Aero(S, b, c, b * b / S, ref.x_mac_le, ders, cd0, parts, Mdd, sw, tc, v["w_tc"] <= 0.045, d.x_np - body_munk, d.Kv, d.lam_le, body_munk, vort, s_ratio, float(stall))


def _derivs_at(a: Aero, M: float) -> vlm.Derivatives:
    d0, d1 = a.derivs["0.30"], a.derivs["0.85"]
    t = float(np.clip((M - 0.3) / 0.55, 0, 1))
    mix = lambda x, y: (1 - t) * x + t * y  # noqa: E731
    return vlm.Derivatives(mix(d0.CL_alpha, d1.CL_alpha), mix(d0.K, d1.K), mix(d0.x_np, d1.x_np), mix(d0.Cn_beta, d1.Cn_beta), mix(d0.Cl_beta, d1.Cl_beta),
                           mix(d0.CY_beta, d1.CY_beta), mix(d0.CL0, d1.CL0), mix(d0.Cm0, d1.Cm0), d0.Kv, d0.lam_le, d0.CL_surf_alpha, d0.names)


def cl_alpha(a: Aero, M: float) -> float:
    """Lift slope per rad: lattice below M 0.85, supersonic linear theory above M 1.2, blended between."""
    if M <= 0.85:
        return _derivs_at(a, M).CL_alpha
    sub = a.derivs["0.85"].CL_alpha
    m2 = max(M, 1.2)
    B = np.sqrt(m2**2 - 1)
    sup = 4 / B * (1 - 1 / (2 * a.AR * B))  # Ackeret with tip relief
    if B / np.tan(np.radians(90 - a.sweep_le)) < 1:  # subsonic leading edge: delta-wing value
        sup = min(sup, 2 * np.pi * np.tan(np.radians(90 - a.sweep_le)) * 0.9)
    sup *= 1.15  # strakes, tails and body add about 15 %
    if M >= 1.2:
        return float(sup)
    t = (M - 0.85) / 0.35
    peak = 1.1 * sub
    return float((1 - t) ** 2 * sub + 2 * t * (1 - t) * peak + t**2 * sup)


def k_induced(a: Aero, M: float, CL: float) -> float:
    """Drag-due-to-lift factor with partial leading-edge suction (Raymer 12.6): K = s K100 + (1-s) K0."""
    cla = cl_alpha(a, M)
    K0 = 1 / max(cla, 1e-6)
    if M >= 1.0:
        B = np.sqrt(max(M**2 - 1, 1e-3))
        Ks = a.AR * B**2 * np.cos(np.radians(a.sweep_le)) / max(4 * a.AR * B - 2, 1e-3)
        return float(max(Ks, 0.5 * K0))  # Raymer eq. 12.51
    K100 = max(_derivs_at(a, M).K, 1 / (np.pi * a.AR * 1.0))
    s_max = 0.6 if a.sharp_le else 0.92
    s = s_max * np.exp(-((CL / 0.7) ** 2))
    return float(s * K100 + (1 - s) * K0)


def cd_wave(a: Aero, g: GeoProps, M: float, CL: float) -> float:
    if a.Mdd - 0.08 > M:
        return 0.0
    mcr = a.Mdd - (0.1 / 80) ** (1 / 3)
    E = 1.1  # linear area-rule theory on the real shape slightly under-predicts measured wave drag
    w12 = E * g.mach_plane["1.2"] / a.S
    if a.Mdd > M:
        return float(20 * max(M - mcr, 0.0) ** 4)  # Lock's fourth-power rise up to the divergence Mach number
    if M < 1.0:
        return float(min(20 * (a.Mdd - mcr) ** 4 + 0.1 * (M - a.Mdd), 0.7 * w12))
    if M < 1.2:
        lo = 20 * (a.Mdd - mcr) ** 4 + 0.1 * (1.0 - a.Mdd)
        return float(lo + (w12 - lo) * np.sin(0.5 * np.pi * (M - 1.0) / 0.2) if lo < w12 else w12)
    ms = np.array([1.2, 1.6, 2.0])
    ws = E * np.array([g.mach_plane[k] for k in ("1.2", "1.6", "2.0")]) / a.S
    return float(np.interp(M, ms, ws))


def cd0_at(a: Aero, air: Air, M: float) -> float:
    """Friction scales with Reynolds number and Mach through Eckert's reference temperature."""
    base_air = isa(11000.0)
    ref = friction(base_air, 0.85, a.c, 0.05)
    now = friction(air, max(M, 0.1), a.c, 0.05)
    return float(a.cd0_sub * now / ref)


def vortex_breakdown_alpha(lam_rad: float, tc: float) -> float:
    """Angle (deg) at which the leading-edge vortex bursts over the trailing edge; rises with sweep. A rough fit
    to delta-wing data (about 15 deg at 55 deg sweep, 38 deg at 75 deg); from memory, not checked here."""
    lam = np.degrees(lam_rad)
    return float(np.clip(15 + 1.15 * (lam - 55), 8, 42) + 40 * max(0.0, 0.04 - tc))


def lift_curve(a: Aero, M: float, alpha_deg: float) -> float:
    """CL at incidence: potential lift that separates gradually past the attached-flow stall angle (raised by
    strake and canard vortices), plus Polhamus vortex lift of each swept surface until its vortex bursts."""
    al = np.radians(alpha_deg)
    cla = cl_alpha(a, min(M, 0.85)) if M < 1 else cl_alpha(a, M)
    d = _derivs_at(a, min(M, 0.85))
    sep = 1 / (1 + np.exp((alpha_deg - a.stall_attached) / 3.5))
    f_pot = sep + (1 - sep) * 0.6
    CL = d.CL0 + cla * np.sin(al) * np.cos(al) ** 2 * f_pot
    for kv, sw, lost in a.vortex:
        a_bd = vortex_breakdown_alpha(np.radians(sw), a.tc)
        CL += kv * lost * np.sin(al) ** 2 * np.cos(al) / (1 + np.exp((alpha_deg - a_bd) / 4.0))
    return float(CL)


def cl_max(a: Aero, M: float) -> tuple[float, float]:
    al = np.linspace(0, 50, 101)
    cl = np.array([lift_curve(a, M, x) for x in al])
    i = int(np.argmax(cl))
    return float(cl[i]), float(al[i])


# --------------------------------------------------------------------------- weights
def empty_mass(design: dict[str, Any], a: Aero, g: GeoProps, W0: float, T_sl: float, mission: Mission, mat_factor: float) -> dict[str, float]:
    """Raymer (2018) fighter/attack group weights, eqs. 15.1-15.5 (lb, ft); exponents quoted from memory and
    calibrated once to a public empty mass. Structure groups scale with the selected materials."""
    v = design["variables"]
    Wdg, Nz = W0 / LB, mission.n_ult
    Sw = (a.S + v["s_size"] * a.S) / FT**2
    delta = v["w_taper"] < 0.15 and v["w_sweep"] > 45
    wing = 0.0103 * (0.768 if delta else 1.0) * (Wdg * Nz) ** 0.5 * Sw**0.622 * a.AR**0.785 * v["w_tc"] ** -0.4 * (1 + v["w_taper"]) ** 0.05 / np.cos(np.radians(v["w_sweep"] * 0.8)) * (0.1 * Sw) ** 0.04
    tails = 0.0
    for key, frac in (("h_size", 1.0), ("c_size", 1.0)):
        if v[key] > D.OPTION_ON[key]:
            Sh = v[key] * a.S / FT**2
            tails += 3.316 * (1 + (2 * v["width"] / FT) / (np.sqrt(Sh * 3) + 1e-6)) ** -2.0 * (Wdg * Nz / 1000) ** 0.26 * Sh**0.806 * frac
    Sv = v["v_size"] * a.S / FT**2
    Lt = max(0.35 * v["length"], 1.0) / FT
    fins = 0.452 * 1.047 * (Wdg * Nz) ** 0.488 * Sv**0.718 * 2.0**0.341 / Lt * 1.25**0.348 * 1.4**0.223 * 1.3**0.25 / np.cos(np.radians(v["v_sweep"])) ** 0.323
    fins *= 1.08 if v["v_cant"] > 5 else 1.0  # two fins: more joints
    fins += 0.25 * Sv * 2.0 * (v["vf_size"] / max(v["v_size"], 1e-6)) if v["vf_size"] > D.OPTION_ON["vf_size"] else 0.0
    Lf, Dd, Wd = v["length"] / FT, 2 * v["height"] / FT, 2 * v["width"] / FT
    fus = 0.499 * (0.774 if delta else 1.0) * Wdg**0.35 * Nz**0.25 * Lf**0.5 * Dd**0.849 * Wd**0.685
    fus *= 1 + 0.15 * (g.duct_volume > 0) + 0.06 * (g.bay_volume > 0)  # duct walls, bay doors and frames
    blades = 4.0 * v["b_count"] * 2 * v["b_height"] if v["b_count"] > D.OPTION_ON["b_count"] else 0.0
    struct = (wing + tails + fins + fus + blades) * LB * mat_factor
    gear = 0.033 * W0
    engine = T_sl / 7.5 / G0 * 1.15  # engine thrust/weight 7.5, installation 15 %
    systems = 0.11 * W0 + mission.avionics_kg
    total = mission.calib * (struct + gear + engine + systems)
    return {"structure": struct, "wing": wing * LB * mat_factor, "tails": tails * LB * mat_factor, "fins": fins * LB * mat_factor, "fuselage": fus * LB * mat_factor,
            "gear": gear, "engine": engine, "systems": systems, "empty": total}


# --------------------------------------------------------------------------- evaluation
@dataclass
class State:
    key: str
    name: str
    mach: float
    altitude: float
    W: float
    CL: float
    alpha: float
    CD: float
    CD0: float
    CDi: float
    CDw: float
    drag_N: float
    thrust_N: float
    LD: float
    Ps: float  # specific excess power, m/s
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class Result:
    family: str
    variables: dict[str, float]
    W0: float
    empty: float
    fuel: float
    fuel_capacity: float
    T_sl: float
    S: float
    span: float
    CD0: float
    static_margin: float
    states: dict[str, State]
    masses: dict[str, float]
    drag_parts: dict[str, float]
    materials: list[dict[str, Any]]
    constraints: dict[str, float]  # >0 = violated (normalised)
    metrics: dict[str, float]
    geometry: dict[str, Any]
    feasible: bool = True


def flight_state(a: Aero, g: GeoProps, s: Situation, W: float, T_sl: float, x_cg: float) -> State:
    air = isa(s.altitude, s.dT)
    V = s.mach * air.a
    q = 0.5 * air.rho * V**2
    if np.isfinite(s.alpha):
        CL = lift_curve(a, s.mach, s.alpha)
        alpha = s.alpha
    else:
        CL = s.n * W * G0 / (q * a.S)
        alpha = float(np.degrees(CL / max(cl_alpha(a, s.mach), 1e-6)))
    cd0 = cd0_at(a, air, s.mach)
    K = k_induced(a, s.mach, CL)
    cdi = K * CL**2 if not np.isfinite(s.alpha) else CL * np.tan(np.radians(alpha))
    cdw = cd_wave(a, g, s.mach, CL)
    if s.beta:
        cd0 *= 1 + 0.5 * np.radians(s.beta) ** 2 * 10  # crossflow drag of the body and fins
    CD = cd0 + cdi + cdw
    D_ = CD * q * a.S
    lapse, tsfc = thrust_lapse(air, s.mach, s.thrust)
    T = lapse * T_sl
    st = State(s.key, s.name, s.mach, s.altitude, W, float(CL), float(alpha), float(CD), float(cd0), float(cdi), float(cdw), float(D_), float(T), float(CL / max(CD, 1e-9)), float((T - D_) * V / (W * G0)))
    st.extra["q_Pa"] = q
    st.extra["tsfc"] = tsfc
    st.extra["V"] = V
    st.extra["T_recovery_K"] = t_adiabatic(air.T, s.mach)
    return st


def evaluate(family: str, d: dict[str, float], mission: Mission | None = None, iters: int = 6, h: float = 0.09, extreme_seed: int = 1) -> Result:
    mission = mission or Mission()
    d = D.complete(family, d)
    design = D.build(d, family)
    g = geometry(design, h)
    # materials follow the temperatures the airframe sees (worst supersonic state), then roughness and mass factor
    T_hot = max(t_adiabatic(isa(s.altitude, s.dT).T, s.mach) for s in SITUATIONS)
    zone_T = {"radome": T_hot, "fwd_fuselage": T_hot, "centre_fuselage": T_hot - 10, "wing_box": T_hot - 15, "wing_le": T_hot + 15, "control": T_hot - 10, "aft_fuselage": T_hot, "nozzle": T_hot, "gear": 300.0}
    choices = mt.select(zone_T)
    mat_factor = mt.structure_factor(choices)
    rough = float(np.mean([mt.LIBRARY[c.material].rough_um for c in choices]))
    a = aero_model(design, g, mission, rough)
    states: dict[str, State] = {}
    W0 = 12000.0
    fuel = 3000.0
    masses: dict[str, float] = {}
    for _ in range(iters):
        T_sl = d["tw"] * W0 * G0
        masses = empty_mass(design, a, g, W0, T_sl, mission, mat_factor)
        # mission fuel: out and back at cruise, a supersonic leg each way, combat at max thrust, loiter reserve
        Wm = W0 - 0.5 * fuel
        cr = flight_state(a, g, next(s for s in SITUATIONS if s.key == "cruise"), Wm, T_sl, 0)
        ds = flight_state(a, g, Situation("dash_leg", "", 1.6, 12000.0, thrust="max"), Wm, T_sl, 0)
        dry = flight_state(a, g, Situation("sc_leg", "", 1.6, 12000.0, thrust="mil"), Wm, T_sl, 0)
        leg_super = dry if dry.thrust_N >= dry.drag_N else ds
        f_cruise = cr.drag_N * cr.extra["tsfc"] * 2 * (mission.radius_m - mission.dash_m) / cr.extra["V"]
        f_dash = leg_super.drag_N * leg_super.extra["tsfc"] * 2 * mission.dash_m / leg_super.extra["V"]
        comb = flight_state(a, g, next(s for s in SITUATIONS if s.key == "turn_sustained"), Wm, T_sl, 0)
        f_comb = comb.thrust_N * comb.extra["tsfc"] * mission.combat_s
        f_res = cr.drag_N * 0.8 * thrust_lapse(isa(3000.0), 0.5, "mil")[1] * mission.reserve_s
        fuel = 1.06 * (f_cruise + f_dash + f_comb + f_res) + 0.03 * W0  # start, taxi, take-off, climb
        W0 = masses["empty"] + fuel + mission.payload_kg + mission.crew_kg
    T_sl = d["tw"] * W0 * G0
    for s in SITUATIONS:
        W = masses["empty"] + mission.payload_kg * (0.0 if s.key in ("approach", "crosswind") else 1.0) + mission.crew_kg + s.fuel_frac * fuel
        states[s.key] = flight_state(a, g, s, W, T_sl, 0)
    # ---- metrics
    m: dict[str, float] = {}
    clm, alm = cl_max(a, 0.2)
    Wto, rho0 = states["takeoff"].W * G0, isa(0.0).rho
    rho_hot = isa(1500.0, 35.0).rho
    T_to = thrust_lapse(isa(0.0), 0.15, "max")[0] * T_sl
    T_hot_to = thrust_lapse(isa(1500.0, 35.0), 0.15, "max")[0] * T_sl
    cl_to = 0.8 * clm + mission.flap_dCL  # tail clearance limits the rotation angle
    m["takeoff_m"] = float(1.44 * Wto**2 / (G0 * rho0 * a.S * cl_to * max(T_to - 0.05 * Wto, 1.0)) + 3 * 1.1 * np.sqrt(2 * Wto / (rho0 * a.S * cl_to)))
    m["hot_takeoff_m"] = float(1.44 * Wto**2 / (G0 * rho_hot * a.S * cl_to * max(T_hot_to - 0.05 * Wto, 1.0)) + 3 * 1.1 * np.sqrt(2 * Wto / (rho_hot * a.S * cl_to)))
    Wl = states["approach"].W * G0
    m["approach_mps"] = float(1.3 * np.sqrt(2 * Wl / (rho0 * a.S * (0.85 * clm + mission.flap_dCL))))
    m["CL_max"], m["alpha_CL_max"] = clm, alm
    cr = states["cruise"]
    m["cruise_LD"] = cr.LD
    m["cruise_drag_N"] = cr.drag_N
    m["specific_range_km_per_kg"] = float(cr.extra["V"] / (cr.drag_N * cr.extra["tsfc"]) / 1000)
    m["Ps_climb"] = states["climb"].Ps
    # sustained turn: thrust = drag at the largest load factor
    st = states["turn_sustained"]
    q = st.extra["q_Pa"]
    K = k_induced(a, 0.9, 0.8)
    n_s = float(np.sqrt(max((st.thrust_N / (q * a.S) - st.CD0 - st.CDw) / K, 0.0)) * q * a.S / (st.W * G0))
    n_s = min(n_s, 9.0, q * a.S * cl_max(a, 0.9)[0] / (st.W * G0))
    m["n_sustained"] = n_s
    m["turn_rate_sustained_dps"] = float(np.degrees(G0 * np.sqrt(max(n_s**2 - 1, 0)) / st.extra["V"]))
    it = states["turn_instant"]
    n_i = min(9.0, it.extra["q_Pa"] * a.S * cl_max(a, 0.5)[0] / (it.W * G0))
    m["turn_rate_instant_dps"] = float(np.degrees(G0 * np.sqrt(max(n_i**2 - 1, 0)) / it.extra["V"]))
    m["supercruise_margin"] = float(states["supercruise"].thrust_N / states["supercruise"].drag_N - 1)
    m["dash_Ps"] = states["dash"].Ps
    m["dash_drag_N"] = states["dash"].drag_N
    m["dash_LD"] = states["dash"].LD
    m["cold_high_Ps"] = states["cold_high"].Ps
    m["max_q_kPa"] = states["max_q"].extra["q_Pa"] / 1000
    m["T_recovery_max_K"] = float(max(s.extra["T_recovery_K"] for s in states.values()))
    m["CL_35deg"] = states["high_alpha"].CL
    # directional stability: fins blanketed at high incidence unless canted outboard into clean air
    d03 = _derivs_at(a, 0.3)
    al = np.radians(35.0)
    blanket = 1 / (1 + np.exp((35.0 - (24 + 0.3 * d["v_cant"])) / 4))
    cnb_body = -1.3 * g.volume * 0.6 / (a.S * a.b) * 0.8  # Raymer-style fuselage destabilising term, per rad
    cnb = d03.Cn_beta + cnb_body
    m["Cn_beta"] = float(cnb)
    m["Cn_beta_dyn_35"] = float((d03.Cn_beta * blanket + cnb_body) * np.cos(al) - 5.0 * d03.Cl_beta * np.sin(al))  # Iz/Ix about 5
    # gusts (Pratt): load-factor increment
    gs = states["gust"]
    air_g = isa(1000.0)
    cla = cl_alpha(a, 0.8)
    WS = gs.W * G0 / a.S
    mu = 2 * WS / (air_g.rho * a.c * cla * G0)
    Kg = 0.88 * mu / (5.3 + mu)
    m["gust_dn"] = float(Kg * air_g.rho * 15.0 * gs.extra["V"] * cla / (2 * WS))
    tb = states["turbulence"]
    air_t = isa(300.0)
    WSt = tb.W * G0 / a.S
    cla_t = cl_alpha(a, 0.9)
    mu_t = 2 * WSt / (air_t.rho * a.c * cla_t * G0)
    m["turbulence_rms_g"] = float(0.88 * mu_t / (5.3 + mu_t) * air_t.rho * mission.turbulence_sigma * tb.extra["V"] * cla_t / (2 * WSt))
    m["crosswind_CY"] = float(abs(d03.CY_beta) * np.radians(12.0))
    m["Cl_beta"] = float(d03.Cl_beta)
    # static margin with the centre of gravity placed by the mass layout
    x_lay = float(_cg(d, masses, fuel, mission, a))
    sm_lay = (a.x_np_sub - x_lay) / a.c
    sm = float(np.clip(sm_lay, -0.05, 0.05))  # fly-by-wire: aim for a margin near zero by arranging tanks and equipment
    x_cg = a.x_np_sub - sm * a.c
    m["x_cg"], m["x_cg_layout"], m["static_margin_layout"] = x_cg, x_lay, float(sm_lay)
    shift = abs(x_cg - x_lay) / d["length"]
    # random extreme states (seeded): drag over the thrust available; the worst 20 % (CVaR) measures robustness
    m["extreme_cvar"], m["extreme_flyable"] = extreme(a, g, masses, fuel, mission, T_sl, extreme_seed)
    # ---- constraints (normalised, > 0 violated)
    span = a.b
    cap = max(0.0, 0.55 * (g.volume - 0.25 * np.pi * D.ENGINE["diameter"] ** 2 * D.ENGINE["length"] - 2.5)) * mission.fuel_density
    con = {
        "engine_and_cockpit_fit": g.envelope_violation / 0.05,
        "fuel_volume": (fuel - cap) / max(fuel, 1.0),
        "balance": (shift - mission.cg_reach) / mission.cg_reach,
        "takeoff": (m["takeoff_m"] - mission.field_m) / mission.field_m,
        "hot_takeoff": (m["hot_takeoff_m"] - 1.6 * mission.field_m) / mission.field_m,
        "approach": (m["approach_mps"] - mission.v_approach_max) / mission.v_approach_max,
        "dash_thrust": -states["dash"].Ps / 50.0,
        "sustained_turn": (4.5 - n_s) / 1.0,
        "directional_stability": (0.06 - cnb) / 0.05,
        "high_alpha_departure": -m["Cn_beta_dyn_35"] / 0.05,
        "span": (span - mission.span_max) / mission.span_max,
        "length": (d["length"] - mission.length_max) / mission.length_max,
        "materials_temperature": max(0.0, -min(c.margin_K for c in choices)) / 20.0,
    }
    con = {k: float(v) for k, v in con.items()}
    feasible = all(v <= 0 for v in con.values())
    geo = {"volume": g.volume, "wetted": g.wetted, "frontal": g.frontal, "planform": g.planform, "length": g.length, "duct_volume": g.duct_volume, "bay_volume": g.bay_volume,
           "mach_plane_Dq": g.mach_plane, "envelope_violation_m": g.envelope_violation, "neutral_point": a.x_np_sub, "mac": a.c, "x_mac_le": a.x_mac_le, "Mdd": a.Mdd, "Kv": a.Kv}
    mats = [{"zone": c.name, "material": mt.LIBRARY[c.material].name, "T_K": round(c.T_K, 1), "margin_K": c.margin_K, "index": c.index} for c in choices]
    return Result(family, d, float(W0), float(masses["empty"]), float(fuel), float(cap), float(T_sl), a.S, float(span), float(a.cd0_sub), float(sm), states, masses, a.parts, mats, con, m, geo, feasible)


def extreme_states(n: int, seed: int) -> list[Situation]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(Situation(f"x{i}", "random", float(rng.uniform(0.25, 2.0)), float(rng.uniform(0, 14000)), n=float(rng.uniform(1, 4)), dT=float(rng.uniform(-30, 35)),
                             fuel_frac=float(rng.uniform(0.2, 1.0))))
    return out


def extreme(a: Aero, g: GeoProps, masses: dict[str, float], fuel: float, mission: Mission, T_sl: float, seed: int, n: int = 60) -> tuple[float, float]:
    """CVaR (worst 20 %) of drag / maximum thrust over random states, and the share of them the design can sustain."""
    r = []
    for s in extreme_states(n, seed):
        W = masses["empty"] + mission.payload_kg + mission.crew_kg + s.fuel_frac * fuel
        st = flight_state(a, g, s, W, T_sl, 0)
        clm = cl_max(a, min(s.mach, 0.9))[0]
        ratio = st.drag_N / max(st.thrust_N, 1.0)
        if clm < st.CL:  # beyond maximum lift the state cannot be flown: penalise smoothly by the lift shortfall
            ratio = max(ratio, 1.0) + 2.0 * (st.CL / clm - 1.0)
        r.append(min(ratio, 10.0))
    r_ = np.sort(np.array(r))[::-1]
    k = max(1, int(0.2 * n))
    return float(r_[:k].mean()), float(np.mean(np.array(r) <= 1.0))  # flyable: sustainable within maximum lift and thrust


def _cg(d: dict[str, float], masses: dict[str, float], fuel: float, mission: Mission, a: Aero) -> float:
    """Centre of gravity from the main masses; fuel and payload are placed near the wing's mean chord."""
    L = d["length"]
    x_w = a.x_mac_le + 0.4 * a.c
    items = [
        (masses["wing"], x_w), (masses["tails"], 0.9 * L), (masses["fins"], 0.88 * L), (masses["fuselage"], 0.47 * L), (masses["gear"], x_w),
        (masses["engine"], L - D.ENGINE["aft_gap"] - 0.45 * D.ENGINE["length"]), (masses["systems"], 0.4 * L), (mission.crew_kg, D.COCKPIT_X * L + 1.0),
        (0.5 * fuel, x_w), (mission.payload_kg, x_w),
    ]
    m = sum(w for w, _ in items)
    return sum(w * x for w, x in items) / m


def summary(r: Result) -> dict[str, Any]:
    out = asdict(r)
    out["states"] = {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in asdict(s).items()} for k, s in r.states.items()}
    return out


def calibrate(family: str, d: dict[str, float], empty_target: float, mission: Mission | None = None) -> Mission:
    """Set the empty-mass factor so the reference design has the target empty mass (take-off weight closes
    around it, so this is solved as a root, not a ratio)."""
    from scipy.optimize import brentq

    mission = mission or Mission()

    def f(c: float) -> float:
        mission.calib = c
        return evaluate(family, d, mission).empty - empty_target

    mission.calib = float(brentq(f, 0.8, 1.3, xtol=1e-4))
    return mission
