r"""Airframe skin design with the parametric PINN: a convergence study of the network and a per-zone optimisation.

Convergence: the skin PINN (parametric in :math:`\ln Bi`, :mod:`aerosynapse.thermal.bridge`) is trained at
increasing Adam budgets with the same seed, and its error against the exact eigen-series is measured on a grid it
never saw (four Biot numbers, 25 Fourier times, 21 depths). The error is fitted with the project's scaling laws
(:func:`aerosynapse.thermal.pinn.scaling.fit`: a power law, or a power law with a floor when another error source
takes over), so the report says how fast the network converges and where it stops improving.

Design: for every airframe zone, every candidate material whose service limit is above the zone's temperature and
a range of skin thicknesses, the PINN gives the outer and inner face temperatures through each hot flight state;
the through-thickness difference gives the thermal stress. The lightest panel (areal mass :math:`\rho t`) is kept
that satisfies, in every state, thermal stress x factor of safety <= allowable at temperature, peak temperature
<= service limit - 20 K, and thickness >= the structural minimum from the sizing stage. Each query outside the
network's trained box (Bi range or Fourier horizon) is answered by the analytic series and counted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from aerosynapse.aircraft import materials as mt
from aerosynapse.aircraft.atmosphere import isa
from aerosynapse.aircraft.layers import q_convective, q_stagnation, t_adiabatic
from aerosynapse.thermal import bridge as B

# panel: (nominal thickness m, heating station m from the nose or "stag", nose radius m)
PANELS: dict[str, tuple[float, Any, float | None]] = {
    "radome": (0.008, "stag", 0.10), "fwd_fuselage": (0.0025, 2.5, None), "centre_fuselage": (0.003, 7.0, None), "wing_box": (0.004, 1.5, None),
    "wing_le": (0.003, "stag", 0.015), "control": (0.002, 1.0, None), "aft_fuselage": (0.003, 12.0, None), "nozzle": (0.004, 14.0, None),
}
# state: (Mach, altitude m, hot/cold day dT K, duration s)
STATES: dict[str, tuple[float, float, float, float]] = {"dash": (1.8, 12000.0, 0.0, 600.0), "cold_high": (1.6, 15000.0, -20.0, 600.0),
                                                          "max_q": (1.2, 300.0, 0.0, 120.0), "hot_day_dash": (1.8, 12000.0, 25.0, 600.0)}
FS = 1.5
T_MARGIN = 20.0
# practical minimum gauges (handling, hail and rain erosion, radar transparency of the radome), m
MIN_GAUGE = {"radome": 3.0e-3, "wing_le": 1.6e-3, "nozzle": 1.2e-3}
DEFAULT_GAUGE = 0.8e-3


def equivalent_thickness(t_ref: float, mat: mt.Mat, T: float, ref: str = "al7075") -> float:
    """Thickness of ``mat`` at temperature T that matches a panel of the reference material (thickness t_ref, same
    temperature) in both strength (t sigma) and panel buckling stiffness (E t^3)."""
    r = mt.LIBRARY[ref]
    return max(t_ref * r.strength(T) / max(mat.strength(T), 1.0), t_ref * (r.modulus(T) / mat.modulus(T)) ** (1 / 3))


def error_vs_exact(net: Any, prob: Any) -> dict[str, float]:
    fo = np.linspace(0.02, B.HORIZON, 25)
    x = np.linspace(0, 1, 21)
    errs, rel = [], []
    for bi in (0.3, 1.0, 3.0, 10.0):
        ex = B.theta_exact(bi, fo, x)
        pn = B.theta_pinn(net, prob, bi, fo, x)
        errs.append(float(np.abs(pn - ex).max()))
        rel.append(float(np.linalg.norm(pn - ex) / np.linalg.norm(ex)))
    return {"max_abs": float(max(errs)), "rel_l2": float(np.mean(rel))}


def convergence(out: Path, levels: list[int], lbfgs: int = 50, log: Any = print) -> dict[str, Any]:
    """Train the skin PINN at each Adam budget; error vs the exact series; power-law fit of the convergence."""
    from aerosynapse.thermal.pinn import scaling

    rows = []
    best = None
    for ep in levels:
        d = out / f"adam_{ep}"
        d.mkdir(parents=True, exist_ok=True)
        done = d / "level.json"
        if done.exists() and (d / "skin_pinn.pt").exists() and json.loads(done.read_text())["lbfgs"] == lbfgs:
            rows.append(json.loads(done.read_text()))  # trained before with the same settings: reuse
            e = {"max_abs": rows[-1]["max_abs"], "rel_l2": rows[-1]["rel_l2"]}
        else:
            t0 = time.perf_counter()
            tr = B.train(d, epochs=ep, lbfgs=lbfgs)
            e = error_vs_exact(tr["net"], tr["problem"])
            rows.append({"adam_epochs": ep, "lbfgs": lbfgs, "wall_s": time.perf_counter() - t0, **e})
            done.write_text(json.dumps(rows[-1]))
        log(f"  PINN adam {ep}: max error {e['max_abs']:.2e}, rel L2 {e['rel_l2']:.2e}, {rows[-1]['wall_s']:.0f} s")
        if best is None or e["max_abs"] < best[0]:
            best = (e["max_abs"], d / "skin_pinn.pt")
    x = np.array([r["adam_epochs"] for r in rows], float)
    fit = scaling.fit(x, [np.array([r["max_abs"]]) for r in rows], n_boot=50)
    return {"levels": rows, "fit": {k: v for k, v in fit.items() if not isinstance(v, (np.ndarray, list)) or k in ("ci",)}, "best_checkpoint": str(best[1]) if best else None}


def zone_temperatures() -> dict[str, float]:
    T_hot = max(t_adiabatic(isa(h, dT).T, M) for M, h, dT, _ in STATES.values())
    return {"radome": T_hot, "fwd_fuselage": T_hot, "centre_fuselage": T_hot - 10, "wing_box": T_hot - 15, "wing_le": T_hot + 15, "control": T_hot - 10,
            "aft_fuselage": T_hot, "nozzle": T_hot, "gear": 300.0}


def panel_worst(net: Any, prob: Any, zone: str, mat: mt.Mat, t: float) -> dict[str, Any]:
    _, where, R = PANELS[zone]
    T_extra = next(z.T_extra for z in mt.ZONES if z.key == zone)
    res = []
    for st, (M, h, dT, dur) in STATES.items():
        air = isa(h, dT)
        Taw = t_adiabatic(air.T, M)
        T0 = air.T + 5.0
        Tw_ref = 0.5 * (T0 + Taw)
        if where == "stag":
            T_sink = air.T * (1 + 0.2 * M**2)
            q = q_stagnation(air, M, float(R), Tw_ref)
        else:
            T_sink = Taw + (0.15 * T_extra if zone in ("aft_fuselage", "nozzle") else 0.0)
            q = q_convective(air, M, float(where), Tw_ref)
        hc = B.h_coefficient(T_sink if where == "stag" else Taw, q, Tw_ref)
        r = B.panel(net, prob, mat, t, hc, T0, T_sink, dur, n_t=24)
        r["state"] = st
        res.append(r)
    w = max(res, key=lambda r: r["stress_ratio"])
    src: dict[str, int] = {}
    for r in res:
        for k, v in r["source"].items():
            src[k] = src.get(k, 0) + v
    return {"state": w["state"], "Bi": max(r["Bi"] for r in res), "dT_max_K": max(r["dT_max_K"] for r in res), "thermal_stress_MPa": w["thermal_stress_MPa"],
            "allowable_MPa": w["allowable_MPa"], "stress_ratio": w["stress_ratio"], "T_max_K": max(r["T_max_K"] for r in res), "source": src}


def optimise(net: Any, prob: Any, t_min: dict[str, float] | None = None, n_t: int = 22, log: Any = print, ref: str = "al7075") -> dict[str, Any]:
    """Lightest feasible (material, thickness) per zone with the PINN in the loop. ``t_min``: structural thickness per
    zone from the sizing stage, in the reference material ``ref`` (converted to each candidate material)."""
    t_min = t_min or {}
    zT = zone_temperatures()
    thick = np.geomspace(0.6e-3, 16e-3, n_t)
    out: dict[str, Any] = {}
    n_eval = 0
    t0 = time.perf_counter()
    for zone, (t_nom, _, _) in PANELS.items():
        z = next(zz for zz in mt.ZONES if zz.key == zone)
        T_zone = zT[zone] + z.T_extra
        cands = [mt.LIBRARY[k] for k in (z.allowed or mt.LIBRARY) if mt.LIBRARY[k].T_max - T_MARGIN >= T_zone]
        best = None
        tried = []
        for m in cands:
            t_floor = max(MIN_GAUGE.get(zone, DEFAULT_GAUGE), equivalent_thickness(t_min.get(zone, 0.0), m, T_zone, ref))
            for t in np.r_[t_floor, thick[thick > t_floor]]:
                w = panel_worst(net, prob, zone, m, float(t))
                n_eval += 1
                ok = w["stress_ratio"] * FS <= 1.0 and w["T_max_K"] <= m.T_max - T_MARGIN
                if ok:
                    areal = m.rho * t
                    tried.append({"material": m.name, "thickness_mm": t * 1e3, "areal_kg_m2": areal})
                    if best is None or areal < best["areal_kg_m2"]:
                        best = {"material": m.name, "material_key": m.key, "thickness_mm": float(t * 1e3), "areal_kg_m2": float(areal), **w}
                    break  # thicker panels of this material are heavier
        nominal = None
        sel = next((c for c in mt.select({k: v for k, v in zT.items()}) if c.zone == zone), None)
        if sel is not None:
            m0 = mt.LIBRARY[sel.material]
            nominal = {"material": m0.name, "thickness_mm": t_nom * 1e3, "areal_kg_m2": m0.rho * t_nom, **panel_worst(net, prob, zone, m0, t_nom)}
        out[zone] = {"zone_T_K": T_zone, "candidates": len(cands), "optimum": best, "nominal": nominal, "feasible_by_material": tried}
        if best:
            log(f"  {zone}: {best['material']} {best['thickness_mm']:.2f} mm ({best['areal_kg_m2']:.1f} kg/m2), stress ratio {best['stress_ratio']:.2f}, T {best['T_max_K']:.0f} K")
        else:
            log(f"  {zone}: no feasible panel in the range")
    return {"zones": out, "panel_evaluations": n_eval, "seconds": time.perf_counter() - t0, "factor_of_safety": FS, "temperature_margin_K": T_MARGIN}
