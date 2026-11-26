"""Structural simulation of a design: design-code loads, wing box sizing, finite elements and the V-g flutter diagram."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aerosynapse.sim.common import NEON, finish, plt, save


def run(design: dict[str, Any], out: str | Path, W0_kg: float = 12000.0, code: str = "mil_fighter", material: str = "al7075", T_K: float = 293.0,
        v_d_eas: float = 400.0, setup: dict[str, Any] | None = None) -> dict[str, Any]:
    from aerosynapse.sim import setup as SU
    from aerosynapse.structures import wing as SW

    S = SU.merged("structure", setup if setup is not None else {"W0_kg": W0_kg, "code": code, "material": material, "T_K": T_K, "v_d_eas": v_d_eas})
    W0_kg, code, material, T_K, v_d_eas = float(S["W0_kg"]), str(S["code"]), str(S["material"]), float(S["T_K"]), float(S["v_d_eas"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    r = SW.analyse(design, W0_kg, code, material, T_K, v_d_eas=v_d_eas, cl_max=float(S["cl_max"]), fuel_kg=float(S["fuel_fraction"]) / 100 * W0_kg)
    w, s, vn, fus = r["wing"], r["wing"]["stations"], r["vn"], r["fuselage"]
    pl = plt()
    figs = []
    fig, ax = pl.subplots(1, 2, figsize=(11, 3.4))
    ax[0].plot(vn["V_eas"], vn["n_manoeuvre_pos"], color=NEON[0])
    ax[0].plot(vn["V_eas"], vn["n_manoeuvre_neg"], color=NEON[0])
    ax[0].fill_between(vn["V_eas"], vn["n_manoeuvre_neg"], vn["n_manoeuvre_pos"], color=NEON[0], alpha=0.12)
    for g in vn["gust_points"].values():
        ax[0].plot([0, g["V"]], [1, g["n_up"]], ":", color=NEON[3])
        ax[0].plot([0, g["V"]], [1, g["n_down"]], ":", color=NEON[3])
    ax[0].set(xlabel="equivalent airspeed, m/s", ylabel="load factor", title=f"Flight envelope: {vn['code']['name']}")
    ax[1].plot(s["eta"], s["moment_kNm"], label="bending moment, kNm")
    ax[1].plot(s["eta"], s["shear_kN"], label="shear, kN")
    ax[1].plot(s["eta"], np.array(s["torque_kNm"]) * 5, label="torque x5, kNm")
    ax[1].set(xlabel="span station", title="Limit loads per side")
    ax[1].legend(fontsize=7.5)
    figs.append(save(fig, out, "1_envelope_and_loads"))
    fig, ax = pl.subplots(1, 2, figsize=(11, 3.4))
    ax[0].plot(s["eta"], s["cover_mm"], label="cover")
    ax[0].plot(s["eta"], s["web_mm"], label="spar web")
    ax[0].set(xlabel="span station", ylabel="mm", title=f"Sized gauges ({w['material']})")
    ax[0].legend(fontsize=7.5)
    ax[1].plot(s["eta"], np.array(s["deflection_m"]) * 1e3, color=NEON[2])
    ax[1].set(xlabel="span station", ylabel="mm", title=f"Deflection at limit load (finite elements): tip {w['tip_deflection_pct_semispan']:.1f} % of the semispan")
    figs.append(save(fig, out, "2_sizing_and_deflection"))
    fl = w["flutter"]
    fig, ax = pl.subplots(1, 2, figsize=(11, 3.4))
    U, g, f = np.array(fl["vg_speed_eas_ms"]), np.array(fl["vg_damping"]), np.array(fl["vg_freq_hz"])
    for j in range(U.shape[1]):
        ax[0].plot(U[:, j], g[:, j], color=NEON[j], label=f"branch {j + 1}")
        ax[1].plot(U[:, j], f[:, j], color=NEON[j])
    for a in ax:
        a.axvline(fl["V_flutter_eas_ms"], color=NEON[5], ls="--", lw=1)
        a.axvline(fl["required_ms"], color=NEON[3], ls=":", lw=1)
        a.set_xlim(0, 1.6 * max(fl["V_flutter_eas_ms"], fl["required_ms"]))
    ax[0].axhline(0, color="#8aa0bd", lw=0.8)
    ax[0].set(xlabel="equivalent airspeed, m/s", ylabel="damping needed g", ylim=(-0.6, 0.3), title=f"V-g diagram: flutter at {fl['V_flutter_eas_ms']:.0f} m/s (required {fl['required_ms']:.0f})")
    ax[0].legend(fontsize=7.5)
    ax[1].set(xlabel="equivalent airspeed, m/s", ylabel="frequency, Hz", ylim=(0, 3 * w["first_torsion_hz"]), title="V-f diagram")
    figs.append(save(fig, out, "3_flutter_vg"))
    if "x" in fus:
        fig, ax = pl.subplots(figsize=(9, 3.0))
        ax.plot(fus["x"], fus["moment_kNm"], color=NEON[4])
        ax.fill_between(fus["x"], fus["moment_kNm"], color=NEON[4], alpha=0.15)
        ax.set(xlabel="x, m", ylabel="kNm", title=f"Fuselage bending at n = {fus['n_ultimate']:.1f}, trimmed by a {fus['tail_load_kN']:.0f} kN tail load")
        figs.append(save(fig, out, "4_fuselage"))
    lines = ["Setup:", *SU.describe("structure", S), "", f"{r['code']}: limit n {w['n_limit']}, ultimate {w['n_ultimate']}; W0 {W0_kg:.0f} kg; {w['material']} at {T_K:.0f} K",
             f"root bending moment {w['root_bending_moment_kNm']:.0f} kNm, shear {w['root_shear_kN']:.0f} kN per side (limit)",
             f"box mass {w['box_mass_both_sides_kg']:.0f} kg (strength-sized {w['strength_sized_box_mass_kg']:.0f} kg, stiffness factor {w['stiffness_scale']:.2f})",
             f"modes: {', '.join(f'{x:.1f}' for x in w['modes_hz'][:6])} Hz",
             f"flutter {fl['V_flutter_eas_ms']:.0f} m/s EAS (required {fl['required_ms']:.0f}); divergence {w['divergence']['V_div_eas_ms']:.0f} m/s EAS",
             "minimum margins of safety: " + ", ".join(f"{k} {v:+.3f}" for k, v in w["margins_min"].items())]
    if "max_moment_kNm" in fus:
        lines.append(f"fuselage: {fus['max_moment_kNm']:.0f} kNm at x {fus['at_x_m']:.1f} m, skin {fus['skin_required_mm']:.2f} mm")
    # 3D view: the wing deformed under limit load (exaggerated) and coloured by the bending stress of its covers
    from aerosynapse.structures.wing import main_wing

    f = main_wing(design)
    y0, span = float(f.get("translate", [0, 0, 0])[1]), float(f["span"])
    eta = np.asarray(s["eta"])
    Mk, EI, hb = np.asarray(s["moment_kNm"]), np.asarray(s["EI_MNm2"]), np.asarray(s["box_height_m"])
    sigma = Mk * 1e3 * (0.5 * hb) / np.maximum(EI * 1e6 / (w["E_GPa"] * 1e9), 1e-12) / 1e6  # MPa at limit load
    dz = np.asarray(s["deflection_m"])
    scale = float(0.12 * span / max(abs(dz).max(), 1e-6))
    viz = {"title": "Structure: limit-load deflection and cover bending stress", "deform": {"y": (y0 + eta * span).tolist(), "dz": dz.tolist(), "y0": y0, "scale": scale},
           "colour": {"kind": "span", "label": f"cover bending stress at limit load, MPa (deflection x{scale:.1f})", "y": (y0 + eta * span).tolist(), "value": sigma.tolist(), "y0": y0}}
    (out / "viz3d.json").write_text(json.dumps(viz), encoding="utf-8")
    return finish(out, "Structural simulation", lines, {**r, "figures": figs})
