r"""Wing box and fuselage of a feature-tree design: loads, sizing to zero margin, stiffness, modes, flutter.

Loads (limit, per side): the lift :math:`nW/2` is spread by Schrenk's approximation (mean of the planform and
the elliptic shape), less the inertia relief of the wing's own structure. Lift acts at the quarter chord and the
elastic axis is at 40 % chord, so the box also carries the torque :math:`0.15\,c\,l`. Shear, bending moment and
torque are integrated inboard along the swept elastic axis.

Section: a two-spar box between 15 % and 65 % chord whose covers follow the real airfoil of the wing feature
(NACA, biconvex, diamond or a section table such as an optimised airfoil). At each station the cover thickness is
the smallest that keeps both margins non-negative at ultimate load: strength :math:`\sigma=Mz/I\le\sigma_{allow}(T)`
and buckling of the cover between stringers at 120 mm pitch, combined with the torsion shear flow. Each spar web is
sized for :math:`\tau=V/(2ht)+T/(2A_et)` against shear strength and shear buckling between ribs at 500 mm.
Minimum gauge 0.8 mm. Ribs add 15 % to the box mass.

Stiffness and dynamics: the sized box becomes a 3D frame finite-element model along the elastic axis (bending,
chordwise bending, torsion), clamped at the root, carrying structure and fuel mass: tip deflection and twist at
limit load and the natural modes. Flutter: the typical section at 70 % semispan with the first bending and
torsion frequencies (k method with Theodorsen aerodynamics) at sea level, against the code's margin on V_D.
Divergence: uniform cantilever :math:`q_D=\pi^2GJ/(4L^2ecC_{L\alpha})` (unswept value: sweep-back raises it).

Fuselage: a free beam carrying its mass distribution (proportional to the local cross-section) at the ultimate
load factor, reacted by the wing lift at its aerodynamic centre and a trimming tail load (force and moment balance); the bending moment gives the skin thickness of an
elliptical shell at the most loaded station.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aerosynapse.aircraft import materials as mt
from aerosynapse.aircraft.atmosphere import isa
from aerosynapse.structures import aeroelastic as ae
from aerosynapse.structures import codes as C
from aerosynapse.structures import fem, panels

FRONT, REAR, EA = 0.15, 0.65, 0.40
STRINGER_PITCH, RIB_PITCH, MIN_GAUGE = 0.12, 0.5, 0.8e-3
RIB_FACTOR = 1.15
FUEL_DENSITY, FUEL_FILL = 800.0, 0.85


# ------------------------------------------------------------------ the wing's own airfoil
@dataclass
class Airfoil:
    kind: str
    tc: float
    camber: float
    campos: float
    u: np.ndarray | None = None
    t: np.ndarray | None = None
    zc: np.ndarray | None = None

    @classmethod
    def of(cls, f: dict[str, Any]) -> Airfoil:
        prof = f.get("profile", "naca")
        if prof == "table" and "section" in f:
            s = f["section"]
            return cls("table", float(max(s["t"])), 0.0, 0.4, np.asarray(s["u"], float), np.asarray(s["t"], float), np.asarray(s.get("zc", [0.0] * len(s["u"])), float))
        return cls(prof, float(f.get("tc", 0.06)), float(f.get("camber", 0.0)), float(f.get("camber_pos", 0.4)))

    def half(self, u: np.ndarray, tcl: float | None = None) -> np.ndarray:
        tcl = self.tc if tcl is None else tcl
        u = np.clip(u, 0, 1)
        if self.kind == "table":
            return 0.5 * np.interp(u, self.u, self.t) * tcl / self.tc
        if self.kind == "biconvex":
            return 2 * tcl * u * (1 - u)
        if self.kind == "diamond":
            return tcl * np.where(u < 0.5, u, 1 - u)
        return 5 * tcl * (0.2969 * np.sqrt(u) - 0.1260 * u - 0.3516 * u**2 + 0.2843 * u**3 - 0.1036 * u**4)

    def camber_line(self, u: np.ndarray) -> np.ndarray:
        if self.kind == "table":
            return np.interp(u, self.u, self.zc)
        if self.camber <= 0:
            return np.zeros_like(u)
        m, p = self.camber, np.clip(self.campos, 0.05, 0.95)
        return np.where(u < p, m / p**2 * (2 * p * u - u * u), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * u - u * u))


def box_contour(af: Airfoil, chord: float, tcl: float, n: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed polygon of the box (chordwise y, vertical z), front spar to rear spar along the upper cover and back.
    Returns y, z and a mask: True for cover segments, False for the two spar webs."""
    u = np.linspace(FRONT, REAR, n)
    zc, h = af.camber_line(u), af.half(u, tcl)
    up_y, up_z = u * chord, (zc + h) * chord
    lo_y, lo_z = u[::-1] * chord, (zc - h)[::-1] * chord
    y = np.r_[up_y, lo_y]
    z = np.r_[up_z, lo_z]
    seg_cover = np.r_[np.ones(n - 1, bool), [False], np.ones(n - 1, bool), [False]]  # segment k joins point k to k+1
    return y, z, seg_cover


def main_wing(design: dict[str, Any]) -> dict[str, Any]:
    ws = [f for f in design["features"] if f.get("type") == "wing" and f.get("op", "add") == "add" and f.get("enabled", True)]
    if not ws:
        raise ValueError("the design has no wing feature to size")
    return max(ws, key=lambda f: float(f["span"]) * float(f["root_chord"]))


def helmbold(ar: float, sweep_half: float) -> float:
    return float(2 * np.pi * ar / (2 + np.sqrt(4 + ar**2 * (1 + np.tan(sweep_half) ** 2))))


# ------------------------------------------------------------------ sizing
def size_wing(design: dict[str, Any], W0_kg: float, code: C.Code, material: str = "al7075", T_K: float = 293.0, n_st: int = 24,
              v_d_eas: float = 400.0, cl_max: float = 1.2, fuel_kg: float = 0.0) -> dict[str, Any]:
    f = main_wing(design)
    mat = mt.LIBRARY[material]
    E, sig = mat.modulus(T_K), mat.strength(T_K)
    G = E / (2 * (1 + panels.NU))
    tau_allow = sig / np.sqrt(3)
    span, cr, lam = float(f["span"]), float(f["root_chord"]), float(f.get("taper", 0.5))
    sweep = np.radians(float(f.get("sweep", 30.0)))
    tc, tip_tc = float(f.get("tc", 0.05)), float(f.get("tip_tc", -1))
    af = Airfoil.of(f)
    if af.kind == "table":
        tc = af.tc
    y0 = float(f.get("translate", [0, 0, 0])[1])
    eta = np.linspace(0, 1, n_st + 1)
    c = cr * (1 - (1 - lam) * eta)
    tcl = tc + ((tip_tc if tip_tc > 0 else tc) - tc) * eta
    xle = eta * span * np.tan(sweep)
    tan_ea = (span * np.tan(sweep) + EA * cr * lam - EA * cr) / span
    sweep_ea = float(np.arctan(tan_ea))
    ds = span / n_st / np.cos(sweep_ea)
    S_side = 0.5 * (1 + lam) * cr * span
    S_ref = 2 * S_side + 2 * y0 * cr  # both sides, carried through the body
    AR = (2 * (span + y0)) ** 2 / S_ref
    sweep_half = np.arctan(np.tan(sweep) - 0.5 * cr * (1 - lam) / span)
    a3d = helmbold(AR, sweep_half)

    n_lim = code.n_pos
    W = W0_kg * C.G
    L_side = n_lim * W / 2
    # Schrenk: mean of the planform and an ellipse over the exposed semispan, scaled to carry L_side
    ell = 4 / np.pi * np.sqrt(np.clip(1 - eta**2, 0, 1)) / span
    plan = c / S_side
    shape = 0.5 * (ell + plan)
    shape /= np.trapezoid(shape, eta * span)
    wing_mass_guess = 0.035 * W0_kg  # per side, refined below
    relief = None
    out: dict[str, Any] = {}
    for _it in range(4):
        m_struct = np.full_like(eta, wing_mass_guess / span) if relief is None else relief
        w = L_side * shape - n_lim * C.G * m_struct  # net upward load per span, N/m
        torque_arm = (EA - 0.25) * c  # lift at c/4 ahead of the elastic axis: nose-up torque
        tq = L_side * shape * torque_arm
        # integrate from the tip inboard along the elastic axis
        V, M, T = np.zeros_like(eta), np.zeros_like(eta), np.zeros_like(eta)
        for i in range(n_st - 1, -1, -1):
            dy = span / n_st
            V[i] = V[i + 1] + 0.5 * (w[i] + w[i + 1]) * dy
            M[i] = M[i + 1] + 0.5 * (V[i] + V[i + 1]) * ds
            T[i] = T[i + 1] + 0.5 * (tq[i] + tq[i + 1]) * dy
        ts, tw, props, ms, contours = [], [], [], [], []
        for i in range(n_st + 1):
            y, z, cov = box_contour(af, c[i], tcl[i])
            h_front = z[0] - z[-1]
            h_rear = z[len(z) // 2 - 1] - z[len(z) // 2]
            h = 0.5 * (h_front + h_rear)
            width = (REAR - FRONT) * c[i]

            def section(t_s: float, t_w: float, y=y, z=z, cov=cov) -> dict[str, float]:
                return panels.closed_section(y, z, np.where(cov, t_s, t_w))

            def cover_ms(t_s: float, t_w: float, i=i, h=h, width=width, M=M, T=T) -> tuple[float, float]:
                p = section(t_s, t_w)
                sigma = abs(M[i]) * (0.5 * h) / p["I_bending"]
                q_t = abs(T[i]) / (2 * p["A_enclosed"])
                ms_str = C.margin(sig, np.hypot(sigma, np.sqrt(3) * q_t / t_s), code)
                s_cr = min(panels.plate_compression(E, t_s, STRINGER_PITCH), sig)
                t_cr = min(panels.plate_shear(E, t_s, STRINGER_PITCH, RIB_PITCH), tau_allow)
                ms_buck = panels.interaction(sigma * code.factor_of_safety, s_cr, q_t / t_s * code.factor_of_safety, t_cr)
                return ms_str, ms_buck

            def web_ms(t_w: float, t_s: float, i=i, h=h, V=V, T=T) -> tuple[float, float]:
                p = section(t_s, t_w)
                tau = max(abs(V[i]) / (2 * h * t_w) + abs(T[i]) / (2 * p["A_enclosed"] * t_w), 1e-9)
                t_cr = min(panels.plate_shear(E, t_w, h, RIB_PITCH), tau_allow)
                return C.margin(tau_allow, tau, code), t_cr / (tau * code.factor_of_safety) - 1

            t_s, t_w = 2e-3, 2e-3
            for _ in range(3):  # covers and webs depend on each other only through J and I
                lo, hi = MIN_GAUGE, 0.06
                if min(cover_ms(lo, t_w)) >= 0:
                    t_s = lo
                else:
                    for _b in range(40):
                        mid = 0.5 * (lo + hi)
                        lo, hi = (lo, mid) if min(cover_ms(mid, t_w)) >= 0 else (mid, hi)
                    t_s = hi
                lo, hi = MIN_GAUGE, 0.06
                if min(web_ms(lo, t_s)) >= 0:
                    t_w = lo
                else:
                    for _b in range(40):
                        mid = 0.5 * (lo + hi)
                        lo, hi = (lo, mid) if min(web_ms(mid, t_s)) >= 0 else (mid, hi)
                    t_w = hi
            p = section(t_s, t_w)
            cs, cb = cover_ms(t_s, t_w)
            wsm, wbm = web_ms(t_w, t_s)
            ts.append(t_s)
            tw.append(t_w)
            contours.append((y, z, cov, h, width))
            props.append({**p, "h": h, "width": width})
            ms.append({"cover_strength": cs, "cover_buckling": cb, "web_strength": wsm, "web_buckling": wbm})
        area = np.array([p["area"] for p in props])
        m_box = mat.rho * area * RIB_FACTOR  # kg per metre of axis
        relief = m_box / np.cos(sweep_ea)
        box_mass_side = float(np.trapezoid(m_box, eta * span / np.cos(sweep_ea)))
        if abs(box_mass_side - wing_mass_guess) < 1e-3 * max(wing_mass_guess, 1):
            break
        wing_mass_guess = box_mass_side
    box_vol = np.array([p["A_enclosed"] for p in props])
    fuel_cap = float(np.trapezoid(box_vol, eta * span / np.cos(sweep_ea)) * FUEL_DENSITY * FUEL_FILL) * 2
    fuel_in_wing = min(fuel_kg, fuel_cap)
    fuel_per_len = box_vol / max(np.trapezoid(box_vol, eta * span / np.cos(sweep_ea)), 1e-9) * fuel_in_wing / 2
    strength_mass_side = box_mass_side
    rho0 = isa(0.0).rho
    q_d = 0.5 * C.RHO0 * v_d_eas**2
    axis = np.c_[xle + EA * c, y0 + eta * span, np.zeros_like(eta)]
    dy = span / n_st

    def dynamics(scale: float) -> dict[str, Any]:
        """Frame model, modes, flutter and divergence of the box with every gauge multiplied by ``scale``."""
        pr = [{**panels.closed_section(yy, zz, np.where(cv, ts[i] * scale, tw[i] * scale)), "h": hh, "width": wd} for i, (yy, zz, cv, hh, wd) in enumerate(contours)]
        mb = mat.rho * np.array([q["area"] for q in pr]) * RIB_FACTOR
        frame = fem.Frame(axis)
        for i in range(n_st):
            p0, p1 = pr[i], pr[i + 1]
            avg = lambda k, p0=p0, p1=p1: 0.5 * (p0[k] + p1[k])  # noqa: E731
            fuel = 0.5 * (fuel_per_len[i] + fuel_per_len[i + 1])
            sec = fem.Section(E, G, avg("area"), avg("I_chordwise"), avg("I_bending"), avg("J"), rho=mat.rho * RIB_FACTOR, m_per_length=fuel,
                              Ip=mat.rho * RIB_FACTOR * (avg("I_chordwise") + avg("I_bending")) + fuel * (0.1 * c[i]) ** 2)
            frame.add(i, i + 1, sec, np.array([0.0, 0.0, 1.0]))
        loads = np.zeros((n_st + 1, 6))
        for i in range(n_st + 1):
            wgt = 0.5 if i in (0, n_st) else 1.0
            loads[i, 2] = w[i] * dy * wgt
            loads[i, 4] = tq[i] * dy * wgt  # nose-up torque: positive rotation about +y (x aft, y outboard, z up)
        u = frame.solve(loads, list(range(6)))["u"]
        modes = frame.modes(list(range(6)), n_modes=8)
        fr, shp = modes["freq_hz"], modes["shapes"]
        bend_f = tors_f = None
        for k in range(len(fr)):
            s_ = shp[k]
            wz, rot = np.abs(s_[:, 2]).max(), np.abs(s_[:, 3:6]).max() * np.mean(c) / 2
            if bend_f is None and wz > rot and wz > np.abs(s_[:, 0:2]).max():
                bend_f = float(fr[k])
            elif tors_f is None and rot > wz:
                tors_f = float(fr[k])
        bend_f = bend_f or float(fr[0])
        tors_f = tors_f or float(fr[min(1, len(fr) - 1)])
        k70 = int(round(0.7 * n_st))
        b = 0.5 * c[k70]
        mu = float(mb[k70] + fuel_per_len[k70]) / (np.pi * rho0 * b * b)
        fl = ae.flutter(mu, (EA - 0.5) / 0.5, 0.05 / 0.5, 0.25, bend_f / tors_f)  # centre of mass at 45 % chord, r_alpha^2 = 0.25
        U_f = fl["U_flutter"] * b * 2 * np.pi * tors_f
        GJ = float(np.mean([G * q["J"] for q in pr[: int(0.7 * n_st) + 1]]))
        L_eff = span / np.cos(sweep_ea)
        e = (EA - 0.25) * float(np.mean(c))
        q_div = float(np.pi**2 * GJ / (4 * L_eff**2 * e * np.mean(c) * a3d))
        return {"props": pr, "u": u, "freq": fr, "bend_f": bend_f, "tors_f": tors_f, "mu": mu, "fl": fl, "U_f": float(U_f), "q_div": q_div,
                "mass_side": float(np.trapezoid(mb, eta * span / np.cos(sweep_ea))),
                "flutter_ms": float(U_f / (code.flutter_margin * v_d_eas) - 1), "div_ms": float(q_div / (code.flutter_margin**2 * q_d) - 1)}

    # stiffness sizing: if the strength-sized box flutters or diverges inside the code's margin, thicken it
    dyn, scale = dynamics(1.0), 1.0
    if min(dyn["flutter_ms"], dyn["div_ms"]) < 0:
        lo, hi = 1.0, 2.0
        while min(dynamics(hi)["flutter_ms"], dynamics(hi)["div_ms"]) < 0 and hi < 8:
            lo, hi = hi, hi * 1.6
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if min((d_ := dynamics(mid))["flutter_ms"], d_["div_ms"]) >= 0 else (mid, hi)
        scale, dyn = hi, dynamics(hi)
    props, u = dyn["props"], dyn["u"]
    box_mass_side = dyn["mass_side"]
    tip_defl, tip_twist_deg = float(u[-1, 2]), float(np.degrees(u[-1, 4]))
    fr, bend_f, tors_f, mu, fl, U_f, q_div = dyn["freq"], dyn["bend_f"], dyn["tors_f"], dyn["mu"], dyn["fl"], dyn["U_f"], dyn["q_div"]
    ts, tw = [t * scale for t in ts], [t * scale for t in tw]
    margins = {k: float(min(m[k] for m in ms)) for k in ms[0]}
    margins["flutter"] = dyn["flutter_ms"]
    margins["divergence"] = dyn["div_ms"]
    out = {
        "material": mat.name, "material_key": material, "temperature_K": T_K, "allowable_MPa": sig / 1e6, "E_GPa": E / 1e9,
        "code": code.name, "n_limit": n_lim, "n_ultimate": code.ultimate(n_lim), "W0_kg": W0_kg,
        "span_m": span, "root_chord_m": cr, "elastic_axis_sweep_deg": float(np.degrees(sweep_ea)), "aspect_ratio": AR, "CL_alpha_per_rad": a3d,
        "stations": {"eta": eta.tolist(), "chord": c.tolist(), "shear_kN": (V / 1e3).tolist(), "moment_kNm": (M / 1e3).tolist(), "torque_kNm": (T / 1e3).tolist(),
                     "cover_mm": (np.array(ts) * 1e3).tolist(), "web_mm": (np.array(tw) * 1e3).tolist(), "box_height_m": [p["h"] for p in props],
                     "EI_MNm2": [E * p["I_bending"] / 1e6 for p in props], "GJ_MNm2": [G * p["J"] / 1e6 for p in props],
                     "deflection_m": u[:, 2].tolist(), "margins": ms},
        "root_bending_moment_kNm": float(M[0] / 1e3), "root_shear_kN": float(V[0] / 1e3),
        "box_mass_both_sides_kg": 2 * box_mass_side, "strength_sized_box_mass_kg": 2 * strength_mass_side, "stiffness_scale": scale,
        "wing_mass_with_secondary_kg": 2 * box_mass_side * 1.45,  # leading/trailing edges, control surfaces, fittings
        "fuel_capacity_kg": fuel_cap, "fuel_in_wing_kg": fuel_in_wing,
        "tip_deflection_limit_m": tip_defl, "tip_deflection_pct_semispan": 100 * tip_defl / span, "tip_twist_deg": tip_twist_deg,
        "modes_hz": fr.tolist(), "first_bending_hz": bend_f, "first_torsion_hz": tors_f,
        "flutter": {"typical_section_station_eta": 0.7, "mass_ratio": mu, "frequency_ratio": bend_f / tors_f, "U_flutter_nondim": fl["U_flutter"],
                    "V_flutter_eas_ms": float(U_f), "V_D_eas_ms": v_d_eas, "required_ms": code.flutter_margin * v_d_eas,
                    "vg_speed_eas_ms": (np.array(fl["vg"]["U"]) * 0.5 * c[int(round(0.7 * n_st))] * 2 * np.pi * tors_f).tolist(), "vg_damping": fl["vg"]["g"],
                    "vg_freq_hz": (np.array(fl["vg"]["omega"]) * tors_f).tolist()},
        "divergence": {"q_div_Pa": q_div, "q_D_Pa": q_d, "V_div_eas_ms": float(np.sqrt(2 * q_div / C.RHO0))},
        "margins_min": margins,
        "all_margins_positive": bool(all(v >= -1e-6 for v in margins.values())),
    }
    return out


def fuselage_bending(design: dict[str, Any], W0_kg: float, code: C.Code, x_wing_ac: float, material: str = "al7075", T_K: float = 293.0) -> dict[str, Any]:
    body = next((f for f in design["features"] if f.get("type") == "fuselage"), None)
    if body is None:
        return {"note": "no fuselage feature"}
    st = np.array(body["stations"], float)
    x = np.linspace(st[0, 0], st[-1, 0], 1201)
    A = np.interp(x, st[:, 0], st[:, 1])
    B = np.interp(x, st[:, 0], st[:, 2])
    t0 = np.array(body.get("translate", [0, 0, 0]), float)
    x = x + t0[0]
    area = np.pi * A * B
    n_ult = code.ultimate(code.n_pos)
    m = area / np.trapezoid(area, x) * W0_kg  # mass distributed like the volume (conservative: the wing mass is in the body)
    q = -n_ult * C.G * m  # N/m downward
    # trimmed: wing lift at its aerodynamic centre and a tail load near the tail balance both the force and the
    # pitching moment of the weight, so the free beam's shear and moment return to zero at both ends
    W = -np.trapezoid(q, x)
    x_cg = float(np.trapezoid(-q * x, x) / W)
    x_tail = float(x[-1] - 0.08 * (x[-1] - x[0]))
    L_tail = W * (x_cg - x_wing_ac) / (x_tail - x_wing_ac)
    Lift = W - L_tail
    Vx = np.concatenate([[0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(x))])
    Vx = Vx + np.where(x >= x_wing_ac, Lift, 0.0) + np.where(x >= x_tail, L_tail, 0.0)
    Mx = np.concatenate([[0], np.cumsum(0.5 * (Vx[1:] + Vx[:-1]) * np.diff(x))])
    k = int(np.argmax(np.abs(Mx)))
    mat = mt.LIBRARY[material]
    sig = mat.strength(T_K)
    a_, b_ = A[k], B[k]
    t_req = 4 * abs(Mx[k]) / (np.pi * b_ * (3 * a_ + b_) * sig)
    return {"tail_load_kN": float(L_tail / 1e3), "wing_lift_kN": float(Lift / 1e3), "x_cg_m": x_cg, "n_ultimate": n_ult, "max_moment_kNm": float(abs(Mx[k]) / 1e3), "at_x_m": float(x[k]), "skin_required_mm": float(max(t_req, MIN_GAUGE) * 1e3),
            "material": mat.name, "x": x.tolist(), "moment_kNm": (Mx / 1e3).tolist(), "shear_kN": (Vx / 1e3).tolist()}


def analyse(design: dict[str, Any], W0_kg: float, code_key: str = "mil_fighter", material: str = "al7075", T_K: float = 293.0, v_d_eas: float = 400.0,
            v_c_eas: float | None = None, cl_max: float = 1.2, fuel_kg: float = 0.0) -> dict[str, Any]:
    code = C.CODES[code_key]
    wing = size_wing(design, W0_kg, code, material, T_K, v_d_eas=v_d_eas, cl_max=cl_max, fuel_kg=fuel_kg)
    f = main_wing(design)
    span, cr, lam = float(f["span"]), float(f["root_chord"]), float(f.get("taper", 0.5))
    S = 2 * 0.5 * (1 + lam) * cr * span + 2 * float(f.get("translate", [0, 0, 0])[1]) * cr
    cbar = 2 / 3 * cr * (1 + lam + lam**2) / (1 + lam)
    vn = C.vn_diagram(code, W0_kg * C.G, S, cl_max, -0.6 * cl_max, wing["CL_alpha_per_rad"], cbar, isa(0.0).rho, v_c_eas or v_d_eas / 1.25, v_d_eas)
    x_ac = float(f.get("translate", [0, 0, 0])[0]) + 0.25 * cbar + 0.4 * span * np.tan(np.radians(float(f.get("sweep", 30))))
    fus = fuselage_bending(design, W0_kg, code, x_ac, material, T_K)
    return {"wing": wing, "vn": vn, "fuselage": fus, "code": C.CODES[code_key].name}
