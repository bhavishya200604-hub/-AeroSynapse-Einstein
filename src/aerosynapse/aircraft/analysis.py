r"""Whole-vehicle analysis: drag, thrust, weight, stability, instability margins and temperature.

Drag is built up from its physical sources, with each term stated:

* friction: per-strip flat-plate laws with transition, compressibility and wall temperature, times
  Raymer's form factors (a square-sided body takes the 30-40 % correction);
* induced drag: Trefftz-plane result of the non-planar lifting line, so winglets count;
* transonic wave drag: Korn drag-divergence Mach with Lock's fourth-power law, strip by strip;
* supersonic wave drag: slender-body theory on the real area law, with
  :math:`x=\tfrac L2(1-\cos\theta)`, :math:`S'=\sum a_n\sin n\theta`,
  :math:`D/q=\tfrac\pi4\sum n\,a_n^2`, which returns :math:`9\pi A^2/2L^2` for Sears-Haack;
* supersonic lift drag: :math:`K=AR(M^2-1)\cos\Lambda/(4AR\beta-2)`, whose infinite-aspect-ratio
  limit is the exact Ackeret value :math:`\beta/4`;
* closure penalty: a body whose tail is too steep separates, and its form drag grows.

Turbulence and instability are scored directly: the share of wetted area in turbulent flow
(transition falls with Reynolds number, surface state and sweep through crossflow), the tail
closure margin against separation, gust load, flutter, divergence, static margin and short-period
damping.

Limits: conceptual-design accuracy. Compare designs on the same model. Weights are calibrated on
the baseline in :func:`calibrate`, so they are relative, not predicted.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerosynapse.aircraft import layers as Y
from aerosynapse.aircraft.atmosphere import G0, Air, isa
from aerosynapse.aircraft.propulsion import Engine, available_thrust, cycle
from aerosynapse.aircraft.vehicle import Vehicle
from aerosynapse.shapeopt import dynamics as dy
from aerosynapse.shapeopt import wing as wg

FloatArr = NDArray[np.float64]


@dataclass
class Mission:
    """What the aircraft must do. Defaults describe a generic 150-seat airliner."""

    name: str = "150-seat airliner"
    payload: float = 15000.0
    range_m: float = 3.7e6
    mach: float = 0.78
    altitude: float = 11000.0
    floor_required: float = 95.0  # m^2 of cabin with headroom
    fuel_density: float = 800.0
    v_approach: float = 72.0
    span_limit: float = 36.0
    engine: Engine = field(default_factory=Engine)
    n_engines: int = 2
    material: str = "aluminium"
    k_cl: float = 0.92  # Korn factor: 0.87 conventional, 0.95 supercritical sections
    cl_max_landing: float = 2.4
    n_limit: float = 2.5
    # calibration, set against the baseline
    k_wing: float = 1.0
    k_body: float = 1.0
    k_torsion: float = 1.0  # scales torsion stiffness so the baseline has the stated flutter margin
    k_flat: float = 0.10  # extra shell mass for flat pressurised bodies: an assumption, see the sensitivity study
    w_systems_frac: float = 0.17


SUPERSONIC: dict[str, Any] = dict(
    name="Supersonic business jet", payload=2000.0, range_m=7.0e6, mach=1.6, altitude=16000.0, floor_required=26.0, v_approach=75.0, span_limit=30.0,
    engine=Engine(opr=20.0, fan_pr=2.5, bpr=0.6, T4=1750.0), n_engines=3, material="titanium", k_cl=0.95, cl_max_landing=1.5, n_limit=2.5, w_systems_frac=0.20,
)  # fmt: skip


@dataclass
class Flight:
    """One flight state. Surface state and turbulence change where the boundary layer transitions."""

    name: str = "cruise"
    mach: float | None = None
    altitude: float | None = None
    dT: float = 0.0
    n: float = 1.0
    contaminated: bool = False
    re_tr: float = 2.5e6
    roughness: float = 30e-6  # m, used when contaminated: insects, rain erosion, dirt


def x_transition(Re_c: float, sweep_le: float, re_tr: float, contaminated: bool) -> float:
    """Transition position on a strip as a fraction of chord."""
    if contaminated:
        return 0.02
    x = min(re_tr / max(Re_c, 1.0), 0.45)
    # crossflow: swept leading edges at high Reynolds number trip early
    if sweep_le > 18.0:
        x *= float(np.clip(1.0 - (sweep_le - 18.0) / 12.0, 0.1, 1.0))
    return float(max(x, 0.02))


def wave_drag_area(x: FloatArr, A: FloatArr, modes: int = 14) -> float:
    r"""Slender-body D/q from an area law, via :math:`D/q=\tfrac\pi4\sum n a_n^2`."""
    L = float(x[-1] - x[0])
    if L <= 0:
        return 0.0
    th = np.linspace(1e-3, np.pi - 1e-3, 400)
    xs = x[0] + 0.5 * L * (1 - np.cos(th))
    S = np.interp(xs, x, A)
    k = 5
    S = np.convolve(np.pad(S, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")  # area laws from strips are rough
    dS = np.gradient(S, xs)
    d = 0.0
    for n in range(1, modes + 1):
        an = 2 / np.pi * float(np.trapezoid(dS * np.sin(n * th), th))
        d += n * an**2
    return float(np.pi / 4 * d)


@dataclass
class DragResult:
    CL: float
    CD: float
    CD_friction: float
    CD_form: float
    CD_induced: float
    CD_wave: float
    CD_misc: float
    alpha: float
    L_over_D: float
    turbulent_share: float
    Re_mac: float


def drag(v: Vehicle, air: Air, M: float, CL: float, f: Flight, Tw: float | None = None, kappa: float = 0.95) -> DragResult:
    s = v.strips
    S = v.S_ref
    q_s = np.cos(np.radians(v.sweep_mid))
    # friction and form, strip by strip
    fric = form = 0.0
    wet_turb = wet_all = 0.0
    rough = f.roughness if f.contaminated else 0.0
    Vel = M * air.a
    for j in range(len(s.y)):
        if s.is_body[j]:
            continue
        cj, tcj = s.c[j], s.tc[j]
        Re_c = air.rho * Vel * cj / air.mu
        xt = x_transition(Re_c, v.sweep_le, f.re_tr, f.contaminated)
        cf = Y.friction(air, M, cj, xt, Tw, rough)
        Swet = 2 * 2 * cj * s.dy[j] * (1 + 0.2 * tcj)
        ff = (1 + 0.6 / 0.35 * tcj + 100 * tcj**4) * 1.34 * min(max(M, 0.3), 1.0) ** 0.18 * q_s**0.28
        ff = max(ff, 1.02)
        fric += cf * Swet / S
        form += cf * (ff - 1) * Swet / S
        wet_all += Swet
        wet_turb += Swet * (1 - xt)
    fin = max(v.fineness, 1.2)
    Rb = air.rho * Vel * v.d["L"] / air.mu
    xt_b = 0.02 if f.contaminated else min(f.re_tr / max(Rb, 1.0), 0.5)
    cf_b = Y.friction(air, M, v.d["L"], xt_b, Tw, rough)
    ff_tube = (1 + 60 / fin**3 + fin / 400) * (1 + 0.35 * min(v.squareness, 1.0))
    # a wide flat body behaves as a thick wing, so its thickness ratio sets the form drag
    t_b = 2 * v.d["H"] / v.d["L"]
    ff_flat = 1 + 0.6 / 0.35 * t_b + 100 * t_b**4
    flat = float(np.clip((v.d["W"] / max(v.d["H"], 1e-6) - 1.0) / 1.5, 0.0, 1.0))
    ff_b = (1 - flat) * ff_tube + flat * ff_flat
    # closure: beyond about 14 degrees the aft body separates and pressure drag rises fast
    over = max(v.closure_deg - 14.0, 0.0)
    ff_b *= 1 + 0.03 * over + 0.004 * over**2
    fric += cf_b * v.S_wet_body / S
    form += cf_b * (ff_b - 1) * v.S_wet_body / S
    wet_all += v.S_wet_body
    wet_turb += v.S_wet_body * (1 - xt_b)
    Rt = air.rho * Vel * max(0.4 * v.mac, 0.5) / air.mu
    cf_t = Y.cf_turbulent(Rt)
    fric += cf_t * v.S_wet_tail / S
    form += cf_t * 0.12 * v.S_wet_tail / S
    wet_all += v.S_wet_tail
    wet_turb += v.S_wet_tail
    # lift and induced drag
    AR = v.span**2 / S
    m = v.mesh
    assert m is not None
    beta_sub = np.sqrt(max(1 - min(M, 0.95) ** 2 * q_s**2, 0.1))
    cl_a = 2 * np.pi * q_s / beta_sub * (np.pi / 180) * 0.95
    a_l0 = -2.0
    if M < 1.05:
        cl, Gm, a, _ = wg.linear_load(m, CL, a_l0, cl_a)
        w = Gm @ m.A.T
        cdi = -2 * 2 * float((Gm * w * m.ds).sum()) / S
        # the lifting line is unswept, so sweep beyond 30 degrees costs span efficiency separately
        cdi /= min(1.0, (np.cos(np.radians(v.sweep_le)) / np.cos(np.radians(30.0))) ** 0.15)
    else:
        np.full(len(s.y), CL)
        beta = np.sqrt(max(M**2 - 1, 0.05))
        K = AR * (M**2 - 1) * np.cos(np.radians(v.sweep_le)) / max(4 * AR * beta - 2, 0.2)
        cdi = K * CL**2
        a = CL / (4 / beta * np.pi / 180) + 1.0
    # viscous lift-dependent drag
    cd0 = fric + form
    cdi += 0.38 * cd0 * CL**2 * 0.25
    # wave drag
    cdw = 0.0
    if M < 1.2:
        cosL = max(np.cos(np.radians(v.sweep_mid)), 0.2)
        tc = v.tc_wing
        c_loc = float(np.clip(CL, 0.0, 1.0))
        Mdd = kappa / cosL - tc / cosL**2 - c_loc / (10 * cosL**3)
        Mcr = Mdd - (0.1 / 80) ** (1 / 3)
        cdw_sub = 20 * max(M - Mcr, 0.0) ** 4
        cdw = cdw_sub
    if M >= 1.0:
        Dq = wave_drag_area(v.area_x, v.area_law)
        sup = 1.3 * Dq / S
        cdw = max(cdw, sup) if M >= 1.2 else cdw + (sup - cdw) * max(min((M - 1.0) / 0.2, 1.0), 0.0)
    misc = 0.05 * cd0
    CD = cd0 + cdi + cdw + misc
    Re_mac = air.rho * Vel * v.mac / air.mu
    return DragResult(CL, CD, fric, form, cdi, cdw, misc, float(a), CL / CD, wet_turb / max(wet_all, 1e-9), float(Re_mac))


@dataclass
class Result:
    d: dict[str, float]
    mtow: float
    oew: float
    fuel: float
    wing_mass: float
    body_mass: float
    drag: DragResult
    L_over_D: float
    tsfc: float
    range_factor: float
    thrust_cruise: float
    thrust_avail: float
    mdot: float
    v_stall: float
    static_margin: float
    short_period_zeta: float
    short_period_hz: float
    cn_beta: float
    n_gust: float
    v_flutter: float
    v_divergence: float
    v_dive: float
    closure_deg: float
    closure_margin: float
    thermal: Y.ThermalState
    span: float
    S_ref: float
    floor: float
    fuel_volume_ratio: float
    constraints: dict[str, float]


def structure(v: Vehicle, m: Mission, W_total: float, cl_a: float) -> tuple[float, float, float, float, float]:
    """Wing and body mass (relative model), gust factor, flutter and divergence speeds."""
    mat = Y.MATERIALS[m.material]
    s = v.strips
    msh = v.mesh
    assert msh is not None
    air0 = isa(0.0)
    cl, Gm, *_ = wg.linear_load(msh, 0.5, -2.0, cl_a)
    cl_rad = cl_a * 180 / np.pi
    AR = v.span**2 / v.S_ref
    n_g = dy.gust_load_factor(1.0, 0.6 * m.mach * 340.0 + 40.0, cl_rad * AR / (AR + 2), W_total / v.S_ref, v.mac)
    n_des = max(m.n_limit, n_g) * 1.5
    lift = 2 * Gm * msh.ds
    lift = lift / max(2 * lift.sum(), 1e-9) * (W_total * n_des)
    arm = np.where(msh.is_wing, s.y, v.span / 2)
    Mb = np.array([np.sum(lift * np.maximum(arm - yi, 0.0)) for yi in s.y])
    h_box = np.maximum(0.85 * s.tc * s.c, 0.05)
    A_cap = np.where(msh.is_wing, Mb / (mat.sigma * h_box), 0.0)
    cosS = max(np.cos(np.radians(v.sweep_mid)), 0.3)
    mass_w = 2 * 2 * mat.rho * float(np.sum(A_cap * msh.ds)) / cosS * m.k_wing + 12.0 * v.S_ref * (0.4 if v.lifting_body else 1.0)
    # the pressurised shell: a flat one needs far more material than a circular one
    noncirc = float(np.clip(v.d["W"] / max(v.d["H"], 0.1) - 1.0, 0.0, 8.0))
    mass_b = m.k_body * 11.0 * v.S_wet_body**1.15 * (1 + m.k_flat * noncirc**0.9) * (1 + 0.15 * v.squareness)
    # aeroelastic: typical section at 70 % semi-span
    idx = np.where(msh.is_wing & (np.abs(msh.phi) < 0.3))[0]
    i = idx[np.argmin(np.abs(s.y[idx] - 0.7 * v.span / 2))]
    j = idx[np.argmin(np.abs(s.y[idx] - 0.35 * v.span / 2))]
    half = v.span / 2
    b = float(s.c[i]) / 2
    hj = float(h_box[j])
    cj = float(s.c[j])
    EI = 2 * float(max(A_cap[j], 1e-5)) * mat.E * (hj / 2) ** 2
    GJ = m.k_torsion * 4 * (0.5 * cj * hj) ** 2 * (mat.E * 0.38) * 0.003 / (2 * (0.5 * cj + hj))
    m_len = max(mass_w / max(v.span, 1.0), 5.0) * 1.4
    I_a = m_len * 0.25 * b**2
    w_h = 1.875**2 * np.sqrt(EI / (m_len * half**4))
    w_a = (np.pi / 2) * np.sqrt(GJ / (I_a * half**2))
    sec = dy.Section2DOF(mu=m_len / (np.pi * air0.rho * b**2), a=-0.2, x_alpha=0.1, r2=0.25, sigma=float(w_h / max(w_a, 1e-6)))
    u, _ = dy.flutter_k(sec, np.geomspace(2.0, 0.03, 40))
    v_f = float(u * b * w_a)
    q_d = (GJ / half) * (np.pi / 2) ** 2 / (0.15 * float(s.c[i]) ** 2 * half * 0.5 * cl_rad)
    return mass_w, mass_b, n_g, v_f, float(np.sqrt(2 * max(q_d, 1.0) / air0.rho))


def weights(v: Vehicle, m: Mission, cl_a: float, ld: float, tsfc: float, V: float) -> tuple[float, float, float, float, float, float, float, float]:
    """Fixed-point take-off mass from structure, systems and Breguet fuel."""
    mtow = 3.5 * m.payload + 25000.0
    mw = mb = fuel = ng = vf = vd = 0.0
    for _ in range(5):
        mw, mb, ng, vf, vd = structure(v, m, mtow * G0, cl_a)
        ff = 1 - np.exp(-m.range_m * G0 * tsfc / (V * max(ld, 1.0)))
        fuel = 1.10 * ff * mtow
        mtow = 0.5 * mtow + 0.5 * (mw + mb + m.w_systems_frac * mtow + m.payload + fuel + 0.05 * mtow)
    oew = mtow - m.payload - fuel
    return mtow, oew, fuel, mw, mb, ng, vf, vd


def stability(v: Vehicle, mtow: float, air: Air, V: float) -> tuple[float, float, float, float]:
    """Static margin, short-period damping and frequency, and weathercock stability."""
    S, c = v.S_ref, v.mac
    a_w = 2 * np.pi * (v.span**2 / S) / (2 + np.sqrt(4 + (v.span**2 / S) ** 2))
    x_ac = v.x_mac + 0.25 * c
    a_t = 4.0
    de = 0.35
    l_t = max(v.x_tail - x_ac, 0.1)
    Vh = v.d["Vh"]
    x_np = x_ac + c * Vh * (a_t / a_w) * (1 - de)
    # a body ahead of the wing is destabilising (Munk), a wide body more so
    x_np -= 0.4 * (2 * v.d["W"]) ** 2 * (x_ac * 0.6) / (S * a_w) * 0.5
    # balance: the designer places equipment and fuel to put the centre of gravity near a target
    sm_target = 0.12 if Vh > 0.2 else 0.06
    x_cg = x_np - sm_target * c
    # how far the mass can reasonably move: within the body's middle
    lo, hi = 0.28 * v.d["L"], 0.72 * v.d["L"]
    x_cg = float(np.clip(x_cg, lo, hi))
    sm = (x_np - x_cg) / c
    q = 0.5 * air.rho * V**2
    Iy = mtow * (0.28 * v.d["L"]) ** 2
    Cma = -a_w * sm
    Cmq = -2 * a_t * Vh * l_t / c * (1 if Vh > 0.05 else 0) - 0.6 * a_w * (0.5 + 0.1 * (v.span**2 / S))
    wn2 = -Cma * q * S * c / Iy
    wn = float(np.sqrt(max(wn2, 1e-6)))
    zeta = float(-Cmq * q * S * c**2 / (2 * V * Iy) / (2 * wn) + a_w * q * S / (mtow * V) / (2 * wn))
    Cnb = 3.0 * v.d["Vv"] - 0.9 * v.V_body / (S * v.span) + 0.25 * v.d["winglet"] * 3
    return float(sm), zeta, wn / (2 * np.pi), float(Cnb)


def evaluate(v: Vehicle, m: Mission, f: Flight | None = None) -> Result:
    f = f or Flight()
    M = f.mach or m.mach
    h = f.altitude or m.altitude
    air = isa(h, f.dT)
    V = M * air.a
    q_s = np.cos(np.radians(v.sweep_mid))
    cl_a = 2 * np.pi * q_s / np.sqrt(max(1 - min(M, 0.9) ** 2 * q_s**2, 0.15)) * (np.pi / 180) * 0.95
    cyc = cycle(m.engine, air, M)
    tsfc = cyc.tsfc if cyc.valid else 3e-5
    # outer loop: drag and weight depend on each other
    ld = 16.0 if M < 1 else 7.0
    W = 0.0
    dr = None
    for _ in range(3):
        mtow, oew, fuel, mw, mb, ng, vf, vd = weights(v, m, cl_a, ld, tsfc, V)
        W = 0.93 * mtow * G0
        CL = f.n * W / (0.5 * air.rho * V**2 * v.S_ref)
        dr = drag(v, air, M, CL, f, kappa=m.k_cl)
        ld = dr.L_over_D
    assert dr is not None
    D = dr.CD * 0.5 * air.rho * V**2 * v.S_ref
    mdot = 1.15 * D / max(cyc.spec_thrust, 1.0)
    T_avail, _ = available_thrust(m.engine, air, M, mdot, air, M, f.dT)
    air_sl = isa(0.0)
    # maximum lift falls with sweep; the mission value is quoted for 25 degrees
    cl_land = m.cl_max_landing * min(np.cos(np.radians(v.sweep_mid)) / np.cos(np.radians(25.0)), 1.1)
    v_stall = float(np.sqrt(2 * mtow * 0.85 * G0 / (air_sl.rho * v.S_ref * cl_land)))
    sm, zeta, f_sp, cnb = stability(v, mtow, air, V)
    mat = Y.MATERIALS[m.material]
    R_nose = max(0.25 * v.d["H"], 0.03) if M < 1 else max(0.02 * v.d["H"], 0.02)
    th = Y.thermal_state(air, M, v.d["L"], v.S_wet_body + v.S_wet_wing, R_nose, mat, drag=D)
    # where the wing stalls first: tip stall loses roll control and, on swept wings, pitches up
    assert v.mesh is not None
    _, eta_stall = wg.wing_clmax(v.mesh, -2.0, cl_a, 1.5)
    fuel_vol = fuel / m.fuel_density
    V_avail = 0.75 * v.V_wing + (0.15 * v.V_body if v.lifting_body else 0.05 * v.V_body)
    v_dive = 1.2 * V
    cons = {
        "cabin_floor": 1 - v.floor_area / m.floor_required,
        "fuel_volume": fuel_vol / max(V_avail, 1e-6) - 1,
        "span": v.span / m.span_limit - 1,
        "approach_speed": 1.23 * v_stall / m.v_approach - 1,
        "static_margin_low": (0.03 - sm) / 0.05,
        "static_margin_high": (sm - 0.35) / 0.1,
        "short_period_damping": (0.25 - zeta) / 0.25,
        "directional": (0.05 - cnb) / 0.05,
        "flutter": 1.15 * v_dive / max(vf, 1.0) - 1,
        "divergence": 1.15 * v_dive / max(vd, 1.0) - 1,
        "skin_temperature": (th.T_nose - mat.T_max) / 100.0,
        "tail_closure": (v.closure_deg - 20.0) / 10.0,
        "lift_limit": dr.CL / (1.1 if M < 1 else 0.5) - 1,
        # delta wings hold lift on leading-edge vortices, so this rule does not apply to them
        "tip_stall": (eta_stall - 0.70) / 0.7 if v.sweep_le < 50.0 else -1.0,
    }
    return Result(
        v.d, mtow, oew, fuel, mw, mb, dr, dr.L_over_D, tsfc, V * dr.L_over_D / (G0 * tsfc), D, T_avail, mdot, v_stall, sm, zeta, f_sp, cnb, ng, vf, vd, v_dive,
        v.closure_deg, 14.0 - v.closure_deg, th, v.span, v.S_ref, v.floor_area, fuel_vol / max(V_avail, 1e-6), cons,
    )  # fmt: skip


def calibrate(v: Vehicle, m: Mission, oew_target: float | None = None, flutter_margin: float = 1.3) -> Mission:
    """Fix the baseline's empty mass and flutter margin, so both become comparative quantities.

    A strip-theory typical section cannot predict an absolute flutter speed, so the baseline is
    defined to have the stated margin over dive speed, and designs are then compared against it.
    """
    if oew_target is not None:
        lo, hi = 0.2, 6.0
        for _ in range(26):
            k = 0.5 * (lo + hi)
            if evaluate(v, replace(m, k_wing=k, k_body=k)).oew > oew_target:
                hi = k
            else:
                lo = k
        m = replace(m, k_wing=0.5 * (lo + hi), k_body=0.5 * (lo + hi))
    for _ in range(4):
        r = evaluate(v, m)
        if not np.isfinite(r.v_flutter) or r.v_flutter <= 0:
            break
        m = replace(m, k_torsion=m.k_torsion * (flutter_margin * r.v_dive / r.v_flutter) ** 2)
    return m


def summary(r: Result) -> dict[str, Any]:
    d = r.drag
    t = r.thermal
    return {
        "MTOW_t": r.mtow / 1000, "OEW_t": r.oew / 1000, "fuel_t": r.fuel / 1000, "L_over_D": r.L_over_D, "CD_counts": d.CD * 1e4, "friction": d.CD_friction * 1e4, "form": d.CD_form * 1e4,
        "induced": d.CD_induced * 1e4, "wave": d.CD_wave * 1e4, "misc": d.CD_misc * 1e4, "CL": d.CL, "turbulent_share": d.turbulent_share, "span_m": r.span, "S_ref_m2": r.S_ref,
        "floor_m2": r.floor, "static_margin": r.static_margin, "short_period_zeta": r.short_period_zeta, "cn_beta": r.short_period_hz * 0 + r.cn_beta, "n_gust": r.n_gust, "v_flutter": r.v_flutter, "v_dive": r.v_dive,
        "closure_deg": r.closure_deg, "T_nose_C": t.T_nose - 273.15, "T_skin_C": t.T_skin - 273.15, "T_total_C": t.T_total - 273.15, "q_nose_kW_m2": t.q_nose / 1e3, "delta_aft_m": t.delta_aft,
        "delta_T_aft_m": t.delta_T_aft, "expansion_mm": t.expansion_m * 1e3, "E_ratio": t.E_ratio, "thermal_stress_MPa": t.stress_MPa, "thrust_kN": r.thrust_cruise / 1e3, "tsfc_mg_Ns": r.tsfc * 1e6,
    }  # fmt: skip
