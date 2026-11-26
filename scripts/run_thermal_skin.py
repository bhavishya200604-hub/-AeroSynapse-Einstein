"""Skin heating of the fighter's panels with the merged parametric PINN, checked against exact and FD solutions.

    python scripts/run_thermal_skin.py --profile standard

1. Train one parametric PINN for the convectively heated slab over Bi in [0.2, 20] (project trainer, Adam then
   L-BFGS); the checkpoint is reused if present.
2. Verify it against the project's analytic eigen-series and finite-difference solver at Biot numbers inside
   the range and at its edges.
3. For every airframe zone, with the material the study selected, the zone's panel thickness and the hottest
   flight state's heat transfer: outer and inner face temperatures through the state, the through-thickness
   difference and the thermal stress against the allowable at temperature.
Results: results/thermal_skin/skin.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from _common import RESULTS

from aerosynapse.aircraft import materials as mt
from aerosynapse.aircraft.atmosphere import isa
from aerosynapse.aircraft.layers import q_convective, q_stagnation, t_adiabatic
from aerosynapse.thermal import bridge as B

# panel thickness and where its heating comes from: (thickness m, station m from the nose or "le", nose radius m)
PANELS = {
    "radome": (0.008, "stag", 0.10), "fwd_fuselage": (0.0025, 2.5, None), "centre_fuselage": (0.003, 7.0, None), "wing_box": (0.004, 1.5, None),
    "wing_le": (0.003, "stag", 0.015), "control": (0.002, 1.0, None), "aft_fuselage": (0.003, 12.0, None), "nozzle": (0.004, 14.0, None),
}
STATES = {"dash": (1.8, 12000.0, 0.0, 600.0), "cold_high": (1.6, 15000.0, -20.0, 600.0), "max_q": (1.2, 300.0, 0.0, 120.0), "hot_day_dash": (1.8, 12000.0, 25.0, 600.0)}
PROFILES = {"quick": (800, 60), "standard": (2500, 200), "full": (12000, 800)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="standard", choices=list(PROFILES))
    a = ap.parse_args()
    out = RESULTS / "thermal_skin"
    out.mkdir(parents=True, exist_ok=True)
    ck = out / "skin_pinn.pt"
    rec: dict = {"profile": a.profile, "bi_range": B.BI_RANGE, "horizon_fourier": B.HORIZON}
    if ck.exists():
        net, prob = B.load(ck)
        rec["training"] = json.loads((out / "training.json").read_text()) if (out / "training.json").exists() else {"reused": True}
    else:
        ep, lb = PROFILES[a.profile]
        tr = B.train(out, epochs=ep, lbfgs=lb)
        net, prob = tr["net"], tr["problem"]
        rec["training"] = {"wall_s": tr["wall_s"], "checks_vs_analytic": tr["checks"], "adam_epochs": ep, "lbfgs_iterations": lb}
        (out / "training.json").write_text(json.dumps(rec["training"], indent=1), encoding="utf-8")
    # verification on a grid the network never saw, against the exact series and the FD solver
    fo = np.linspace(0.02, B.HORIZON, 25)
    x = np.linspace(0, 1, 21)
    ver = {}
    for bi in (0.25, 0.7, 2.0, 6.0, 19.0):
        ex = B.theta_exact(bi, fo, x)
        fd = B.theta_fd(bi, fo, x)
        pn = B.theta_pinn(net, prob, bi, fo, x)
        ver[f"Bi={bi}"] = {"pinn_max_abs_error": float(np.abs(pn - ex).max()), "pinn_rel_l2": float(np.linalg.norm(pn - ex) / np.linalg.norm(ex)), "fd_max_abs_error": float(np.abs(fd - ex).max())}
    t0 = time.perf_counter()
    for _ in range(200):
        B.theta_pinn(net, prob, 2.0, fo, x)
    t_pinn = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(20):
        B.theta_exact(2.0, fo, x)
    t_ex = (time.perf_counter() - t0) / 20
    rec["verification"] = ver
    rec["query_seconds"] = {"pinn": t_pinn, "analytic_series": t_ex}
    # zones of the airframe, materials as the fighter study selected them for its hottest state
    T_hot = max(t_adiabatic(isa(h, dT).T, M) for M, h, dT, _ in STATES.values())
    zone_T = {"radome": T_hot, "fwd_fuselage": T_hot, "centre_fuselage": T_hot - 10, "wing_box": T_hot - 15, "wing_le": T_hot + 15, "control": T_hot - 10, "aft_fuselage": T_hot, "nozzle": T_hot, "gear": 300.0}
    choices = {c.zone: c for c in mt.select(zone_T)}
    zones = {}
    for st, (M, h, dT, dur) in STATES.items():
        air = isa(h, dT)
        Taw = t_adiabatic(air.T, M)
        T0 = air.T + 5.0  # soaked at the ambient temperature of the previous leg
        for z, (L, where, R) in PANELS.items():
            c = choices[z]
            m = mt.LIBRARY[c.material]
            T_extra = next(zz.T_extra for zz in mt.ZONES if zz.key == z)
            Tw_ref = 0.5 * (T0 + Taw)
            q = q_stagnation(air, M, float(R), Tw_ref) if where == "stag" else q_convective(air, M, float(where), Tw_ref)
            Taw_z = Taw + (0.15 * T_extra if z in ("aft_fuselage", "nozzle") else 0.0)  # engine-bay heat raises the effective sink
            hcoef = B.h_coefficient(Taw if where != "stag" else air.T * (1 + 0.2 * M**2), q, Tw_ref)
            res = B.panel(net, prob, m, L, hcoef, T0, Taw_z if where != "stag" else air.T * (1 + 0.2 * M**2), dur)
            res.update({"material": m.name, "thickness_mm": L * 1e3, "h_W_m2K": hcoef, "T_limit_K": m.T_max, "temperature_margin_K": m.T_max - res["T_max_K"]})
            for k in ("time_s", "T_outer_K", "T_inner_K"):
                res[k] = [round(v, 2) for v in res[k]]
            zones.setdefault(st, {})[z] = res
    rec["zones"] = zones
    (out / "skin.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({"training": {k: v for k, v in rec["training"].items() if k != "checks_vs_analytic"}, "verification": ver, "query_seconds": rec["query_seconds"]}, indent=1))
    for st, zz in zones.items():
        worst = max(zz.items(), key=lambda kv: kv[1]["stress_ratio"])
        hottest = min(zz.items(), key=lambda kv: kv[1]["temperature_margin_K"])
        print(f"{st}: hottest {hottest[0]} {hottest[1]['T_max_K']:.0f} K (margin {hottest[1]['temperature_margin_K']:.0f} K, {hottest[1]['material']}); "
              f"largest thermal stress {worst[0]} {worst[1]['thermal_stress_MPa']:.1f} MPa = {100 * worst[1]['stress_ratio']:.1f} % of allowable")


if __name__ == "__main__":
    main()
