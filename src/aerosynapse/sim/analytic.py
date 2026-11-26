r"""Analytic simulation of a design: classical closed-form and semi-analytic solutions applied to its own geometry.

1. **Panel method** (Hess-Smith, :mod:`aerosynapse.numerics.panel`) on the main wing's own section: pressure
   distribution and lift curve, against thin-airfoil theory :math:`c_{l\alpha}=2\pi`.
2. **Prandtl lifting line** (Glauert's series, 31 odd terms) on the main wing's planform including the part carried
   through the body: spanwise load against the elliptic load, :math:`C_{L\alpha}`, span efficiency.
3. **Sears-Haack area rule**: the design's normal-section area distribution against the minimum-wave-drag body of the
   same volume and length, :math:`D/q=128V^2/(\pi L^4)`.
4. **Taylor-Maccoll** cone flow on the nose and **normal-shock** relations for a pitot inlet at the dash Mach number.
5. **Ackeret** linear supersonic theory on the section at the normal Mach number of the swept wing.
6. **Exact transient heating** (eigen-series) of the skin panels through a supersonic dash (Mach 1.8 at 12 km for 600 s unless the setup says otherwise).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aerosynapse.sim.common import NEON, finish, plt, save

G = 1.4


def section_xy(f: dict[str, Any], n: int = 121) -> np.ndarray:
    from aerosynapse.structures.wing import Airfoil

    af = Airfoil.of(f)
    u = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    yu, yl = af.camber_line(u) + af.half(u), af.camber_line(u) - af.half(u)
    return np.c_[np.r_[u[::-1], u[1:]], np.r_[yu[::-1], yl[1:]]]  # Selig order


def lifting_line(f: dict[str, Any], alpha_deg: float, a0: float = 2 * np.pi, n_terms: int = 31) -> dict[str, Any]:
    span, cr, lam = float(f["span"]), float(f["root_chord"]), float(f.get("taper", 0.5))
    y0 = float(f.get("translate", [0, 0, 0])[1])
    twist = float(f.get("twist", 0.0))
    s = span + y0
    b = 2 * s
    ns = np.arange(1, 2 * n_terms, 2)
    th = np.linspace(0, np.pi / 2, n_terms + 1)[1:]
    y = s * np.cos(th)  # 0 at the root .. s at the tip
    eta = np.clip((y - y0) / span, 0, 1)
    c = np.where(y < y0, cr, cr * (1 - (1 - lam) * eta))
    a_loc = np.radians(alpha_deg + twist * eta)
    M = np.sin(np.outer(th, ns)) * (4 * b / (a0 * c))[:, None] + np.sin(np.outer(th, ns)) * (ns[None, :] / np.sin(th)[:, None])
    A = np.linalg.solve(M, a_loc)
    S = 2 * (0.5 * (1 + lam) * cr * span + y0 * cr)
    AR = b * b / S
    CL = np.pi * AR * A[0]
    CDi = np.pi * AR * float(np.sum(ns * A**2))
    tt = np.linspace(1e-3, np.pi - 1e-3, 201)
    gam = 2 * b * np.sin(np.outer(tt, ns)) @ A  # Gamma / V
    return {"CL": float(CL), "CDi": float(CDi), "e": float(CL**2 / (np.pi * AR * CDi)) if CDi > 0 else 1.0, "AR": float(AR), "S": float(S), "b": float(b),
            "y": (-s * np.cos(tt)).tolist(), "gamma": gam.tolist()}


def normal_shock(M: float) -> dict[str, float]:
    m2 = np.sqrt((1 + (G - 1) / 2 * M**2) / (G * M**2 - (G - 1) / 2))
    p21 = 1 + 2 * G / (G + 1) * (M**2 - 1)
    r21 = (G + 1) * M**2 / ((G - 1) * M**2 + 2)
    p0r = r21 ** (G / (G - 1)) * ((G + 1) / (2 * G * M**2 - (G - 1))) ** (1 / (G - 1))
    return {"M2": float(m2), "p2_p1": float(p21), "rho2_rho1": float(r21), "T2_T1": float(p21 / r21), "p02_p01": float(p0r)}


def write_viz(f: dict[str, Any], ll: dict[str, Any], xs: np.ndarray, sh: np.ndarray, out: Path) -> None:
    """3D view: the lifting-line load as a ribbon above the wing, and the Sears-Haack body of equal volume and length."""
    import json

    t = f.get("translate", [0, 0, 0])
    sw = np.tan(np.radians(float(f.get("sweep", 35))))
    yy, gg = np.array(ll["y"]), np.array(ll["gamma"])
    y0w, spanw, crw, lam = float(t[1]), float(f["span"]), float(f["root_chord"]), float(f.get("taper", 0.5))
    eta = np.clip((np.abs(yy) - y0w) / spanw, 0, 1)
    xq = float(t[0]) + np.clip(np.abs(yy) - y0w, 0, None) * sw + 0.25 * crw * (1 - (1 - lam) * eta)
    zw = float(t[2]) + 0.15
    gs = 0.6 * spanw / max(float(gg.max()), 1e-9)
    ribbon = [{"pts": [[float(a), float(b), zw] for a, b in zip(xq, yy, strict=True)], "color": [0.13, 0.83, 0.93]},
              {"pts": [[float(a), float(b), zw + float(c) * gs] for a, b, c in zip(xq, yy, gg, strict=True)], "color": [0.98, 0.75, 0.14]}]
    for k in range(0, len(yy), 10):
        ribbon.append({"pts": [[float(xq[k]), float(yy[k]), zw], [float(xq[k]), float(yy[k]), zw + float(gg[k]) * gs]], "color": [0.98, 0.75, 0.14]})
    r_sh = np.sqrt(np.maximum(sh, 0) / np.pi)
    zc = float(t[2])
    n_a = 32
    lines = []
    for x_, r_ in zip(xs, r_sh, strict=True):
        for a_ in range(n_a):
            ang = 2 * np.pi * a_ / n_a
            lines.append(f"v {x_:.4f} {r_ * np.cos(ang):.4f} {zc + r_ * np.sin(ang):.4f}")
    for i_ in range(len(xs) - 1):
        for a_ in range(n_a):
            p0, p1 = i_ * n_a + a_ + 1, i_ * n_a + (a_ + 1) % n_a + 1
            lines.append(f"f {p0} {p0 + n_a} {p1 + n_a}")
            lines.append(f"f {p0} {p1 + n_a} {p1}")
    (out / "sears_haack.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
    viz = {"title": "Analytic: lifting-line span load (gold) and the Sears-Haack body of equal volume and length (ghost)", "lines": ribbon,
           "meshes": [{"obj": "sears_haack.obj", "color": [0.4, 0.85, 1.0, 0.22], "label": "Sears-Haack body"}]}
    (out / "viz3d.json").write_text(json.dumps(viz), encoding="utf-8")


def run(design: dict[str, Any], out: str | Path, setup: dict[str, Any] | None = None) -> dict[str, Any]:
    from aerosynapse.aircraft import materials as mt
    from aerosynapse.aircraft.analysis import wave_drag_area
    from aerosynapse.aircraft.atmosphere import isa
    from aerosynapse.aircraft.layers import q_convective, t_adiabatic
    from aerosynapse.cfd.exact import taylor_maccoll
    from aerosynapse.numerics.panel import solve_panel
    from aerosynapse.shapeopt.geometry import to_panel_order
    from aerosynapse.sim import setup as SU
    from aerosynapse.structures.wing import main_wing
    from aerosynapse.thermal import bridge as B
    from aerosynapse.workflow.stages import area_distribution

    S = SU.merged("analytic", setup)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pl = plt()
    lines: list[str] = ["Setup:", *SU.describe("analytic", S), ""]
    res: dict[str, Any] = {"figures": [], "setup": S}
    f = main_wing(design)
    a_ll = float(S["ll_alpha"])
    a_mid = round(a_ll / 2, 3)
    # 1 panel method on the wing's own section
    xy = section_xy(f)
    X, Y = to_panel_order(xy)
    if float(S["alpha_max"]) <= float(S["alpha_min"]):
        raise ValueError("panel method: 'angle to' must be larger than 'angle from'")
    alphas = np.arange(float(S["alpha_min"]), float(S["alpha_max"]) + 1e-9, float(S["alpha_step"]))
    cls = [solve_panel(X, Y, float(a)).cl for a in alphas]
    fit = np.polyfit(np.radians(alphas), cls, 1)
    slope, a_zero = float(fit[0]), float(-fit[1] / fit[0])  # lift slope, zero-lift angle (rad)
    pr = {a: solve_panel(X, Y, a) for a in sorted({0.0, a_mid, a_ll})}
    fig, ax = pl.subplots(1, 2, figsize=(10, 3.4))
    for k, (a, r) in enumerate(pr.items()):
        ax[0].plot(r.xc, -r.cp, color=NEON[k], label=f"alpha {a:g} deg, cl {r.cl:.3f}")
    ax[0].set(xlabel="x/c", ylabel="-Cp", title=f"Panel method on the '{f.get('id')}' section")
    ax[0].legend(fontsize=7.5)
    ax[1].plot(alphas, cls, "o-", label=f"panel method: {slope:.3f} per rad")
    ax[1].plot(alphas, 2 * np.pi * (np.radians(alphas) - a_zero), "--", color=NEON[3], label="thin airfoil: 2 pi per rad")
    ax[1].set(xlabel="alpha, deg", ylabel="cl", title="Lift curve")
    ax[1].legend(fontsize=7.5)
    res["figures"].append(save(fig, out, "1_panel_method"))
    res["panel"] = {"lift_slope_per_rad": slope, "thin_airfoil": 2 * np.pi, "alpha_deg": a_mid, "cl": pr[a_mid].cl, "cm_c4": pr[a_mid].cm_c4}
    lines += ["1. Panel method (Hess-Smith) on the wing's own section",
              f"   lift slope {slope:.3f} per rad against 2 pi = {2 * np.pi:.3f} from thin-airfoil theory ({100 * (slope / (2 * np.pi) - 1):+.1f} %, the thickness effect)",
              f"   at {a_mid:g} deg: cl {pr[a_mid].cl:.4f}, cm about the quarter chord {pr[a_mid].cm_c4:+.4f}", ""]
    # 2 lifting line
    ll2, ll4 = lifting_line(f, a_mid, a0=slope), lifting_line(f, a_ll, a0=slope)
    cla = (ll4["CL"] - ll2["CL"]) / np.radians(a_ll - a_mid)
    fig, ax = pl.subplots(figsize=(8, 3.2))
    y = np.array(ll4["y"])
    g = np.array(ll4["gamma"])
    ax.plot(y, g, label="lifting line (with taper, twist and body carry-through)")
    ax.plot(y, g.max() * np.sqrt(np.clip(1 - (2 * y / ll4["b"]) ** 2, 0, 1)), "--", label="elliptic load, same peak")
    ax.set(xlabel="span station y, m", ylabel="circulation / V, m", title=f"Prandtl lifting line at {a_ll:g} deg")
    ax.legend(fontsize=7.5)
    res["figures"].append(save(fig, out, "2_lifting_line"))
    res["lifting_line"] = {"CL_alpha_per_rad": cla, "span_efficiency": ll4["e"], "aspect_ratio": ll4["AR"], "alpha_deg": a_ll, "CL": ll4["CL"], "CDi": ll4["CDi"]}
    lines += ["2. Prandtl lifting line (Glauert series) on the main wing planform",
              f"   aspect ratio {ll4['AR']:.2f}, CL_alpha {cla:.3f} per rad (Helmbold-style finite-wing value), span efficiency {ll4['e']:.3f}",
              f"   at {a_ll:g} deg: CL {ll4['CL']:.4f}, induced drag CDi {ll4['CDi']:.5f}", ""]
    # 3 Sears-Haack
    xs, A = area_distribution(design)
    L = float(xs[-1] - xs[0])
    V = float(np.trapezoid(A, xs))
    xi = (xs - xs[0]) / L
    sh = 16 * V / (3 * np.pi * L) * np.clip(4 * xi * (1 - xi), 0, None) ** 1.5
    dq_sh_exact = 128 * V**2 / (np.pi * L**4)
    dq_sh_num = wave_drag_area(xs, sh)
    dq_design = wave_drag_area(xs, A)
    fig, ax = pl.subplots(figsize=(8, 3.2))
    ax.fill_between(xs, A, alpha=0.25, color=NEON[0])
    ax.plot(xs, A, label=f"design, D/q {dq_design:.3f} m2")
    ax.plot(xs, sh, "--", label=f"Sears-Haack, D/q {dq_sh_exact:.3f} m2")
    ax.set(xlabel="x, m", ylabel="area, m2", title="Supersonic area rule (normal sections)")
    ax.legend(fontsize=7.5)
    res["figures"].append(save(fig, out, "3_area_rule"))
    res["area_rule"] = {"volume_m3": V, "length_m": L, "D_over_q_design_m2": dq_design, "D_over_q_sears_haack_m2": dq_sh_exact, "D_over_q_sears_haack_numeric_m2": dq_sh_num}
    lines += ["3. Sears-Haack area rule",
              f"   volume {V:.2f} m3, length {L:.2f} m: ideal body D/q = 128 V^2/(pi L^4) = {dq_sh_exact:.4f} m2 (the code's slender-body sum gives {dq_sh_num:.4f})",
              f"   the design's area distribution: D/q {dq_design:.4f} m2 = {dq_design / dq_sh_exact:.2f} x the ideal", ""]
    # 4 nose cone and inlet shock
    M = float(S["nose_mach"])
    body = next((x for x in design["features"] if x.get("type") == "fuselage"), None)
    cone = None
    if body:
        st = np.array(body["stations"], float)
        half = 0.5 * (st[1, 1] + st[1, 2]) - 0.5 * (st[0, 1] + st[0, 2])
        cone = float(np.degrees(np.arctan2(half, st[1, 0] - st[0, 0])))
    lines.append(f"4. Nose cone and inlet at Mach {M}")
    try:
        tm = taylor_maccoll(M, cone) if cone else None
        res["nose_cone"] = {"half_angle_deg": cone, **(tm or {})}
        if tm:
            lines.append(f"   nose half-angle {cone:.1f} deg (Taylor-Maccoll): shock {tm['shock_deg']:.1f} deg, surface Cp {tm['cp']:.4f}, surface Mach {tm['surface_mach']:.3f}")
    except Exception as e:
        res["nose_cone"] = {"half_angle_deg": cone, "note": f"no attached conical shock: {e}"}
        lines.append(f"   nose half-angle {cone:.1f} deg: the conical shock detaches at Mach {M} (blunt nose)")
    ns_ = normal_shock(M)
    res["normal_shock"] = ns_
    lines += [f"   pitot inlet normal shock: M2 {ns_['M2']:.3f}, p2/p1 {ns_['p2_p1']:.3f}, T2/T1 {ns_['T2_T1']:.3f}, total-pressure recovery {ns_['p02_p01']:.4f}", ""]
    # 5 Ackeret
    sweep = np.radians(float(f.get("sweep", 35)))
    Ma = float(S["ackeret_mach"])
    Mn = Ma * np.cos(sweep)
    x = np.linspace(0.002, 0.998, 400)
    yu = np.interp(x, xy[:121, 0][::-1], xy[:121, 1][::-1])
    yl = np.interp(x, xy[120:, 0], xy[120:, 1])
    yt, yc = 0.5 * (yu - yl), 0.5 * (yu + yl)
    st_t, st_c = float(np.mean(np.gradient(yt, x) ** 2)), float(np.mean(np.gradient(yc, x) ** 2))
    if Mn > 1.02:
        beta = np.sqrt(Mn**2 - 1)
        res["ackeret"] = {"normal_mach": Mn, "cl_alpha_per_rad": 4 / beta, "cd_thickness": 4 / beta * st_t, "cd_camber": 4 / beta * st_c}
        lines += [f"5. Ackeret supersonic section theory at Mach {Ma:g} (normal Mach {Mn:.3f} across the {np.degrees(sweep):.0f} deg sweep)",
                  f"   cl_alpha {4 / beta:.3f} per rad, wave drag from thickness {4 / beta * st_t:.5f}, from camber {4 / beta * st_c:.5f}", ""]
    else:
        res["ackeret"] = {"normal_mach": Mn, "note": "subsonic leading edge: linear 2D supersonic theory does not apply"}
        lines += [f"5. Ackeret: at Mach {Ma:g} the normal Mach number is {Mn:.3f}, a subsonic leading edge, so the 2D theory does not apply", ""]
    # 6 exact skin heating through a dash
    Md, hd, td = float(S["dash_mach"]), float(S["dash_altitude"]), float(S["dash_seconds"])
    air = isa(hd)
    Taw = t_adiabatic(air.T, Md)
    T0 = air.T + float(S["skin_start_offset"])
    fig, ax = pl.subplots(figsize=(8, 3.2))
    heat = {}
    skin = design.get("skin") or {}
    for k, (zone, xst, L_def, key) in enumerate((("fwd_fuselage", 2.5, 2.5e-3, "cfrp_bmi"), ("wing_box", 1.5, 6.0e-3, "cfrp_bmi"), ("aft_fuselage", 12.0, 2.6e-3, "cmc_csic"))):
        zs = skin.get(zone) or {}
        m = next((mm for mm in mt.LIBRARY.values() if mm.name == zs.get("material")), mt.LIBRARY[key])
        th = float(zs.get("thickness_mm", L_def * 1e3)) * 1e-3
        q = q_convective(air, Md, xst, 0.5 * (T0 + Taw))
        hc = B.h_coefficient(Taw, q, 0.5 * (T0 + Taw))
        r = B.panel(None, None, m, th, hc, T0, Taw, td, n_t=60)
        heat[zone] = {"material": m.name, "thickness_mm": th * 1e3, "Bi": r["Bi"], "T_max_K": r["T_max_K"], "dT_max_K": r["dT_max_K"]}
        ax.plot(r["time_s"], r["T_outer_K"], color=NEON[k], label=f"{zone} outer ({m.name.split()[0]}, {th * 1e3:.1f} mm)")
        ax.plot(r["time_s"], r["T_inner_K"], "--", color=NEON[k], label=f"{zone} inner")
    ax.axhline(Taw, color=NEON[3], lw=0.8, ls=":", label=f"adiabatic wall {Taw:.0f} K")
    ax.set(xlabel="time in the dash, s", ylabel="temperature, K", title=f"Exact transient skin heating, Mach {Md:g} at {hd / 1000:g} km")
    ax.legend(fontsize=7, ncol=2)
    res["figures"].append(save(fig, out, "4_skin_heating_exact"))
    res["skin_heating"] = heat
    lines.append(f"6. Exact transient skin heating (eigen-series), Mach {Md:g} dash at {hd / 1000:g} km for {td:g} s")
    lines += [f"   {z}: {v['material']} {v['thickness_mm']:.1f} mm, Bi {v['Bi']:.3f}, peak {v['T_max_K']:.0f} K, through-thickness difference {v['dT_max_K']:.1f} K" for z, v in heat.items()]
    write_viz(f, ll4, xs, sh, out)
    return finish(out, "Analytic simulation", lines, res)
