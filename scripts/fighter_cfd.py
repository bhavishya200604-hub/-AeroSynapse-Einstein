"""3D flow simulation of the reference and the optimised fighters, and a check of the fast model against it.

    python scripts/fighter_cfd.py --profile standard [--designs reference conventional ...]

For each design and state (subsonic cruise, transonic manoeuvre, supersonic dash, high incidence, and a gust
encounter run time-accurately) the solver runs on the same grid spacing, so designs are compared like for like.
At each state the fast performance model is evaluated at the same Mach number, altitude and incidence, and the
lift and drag of both are recorded.
Results: results/fighter/cfd/<design>_<state>/ and results/fighter/cfd/summary.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from _common import RESULTS

from aerosynapse.aircraft.atmosphere import isa
from aerosynapse.cfd import runner
from aerosynapse.fighter import design as D
from aerosynapse.fighter import performance as P

STATES = {
    "cruise": dict(mach=0.85, alpha=2.0, altitude=11000.0),
    "cruise_a5": dict(mach=0.85, alpha=5.0, altitude=11000.0),  # second polar point: drag compared at equal lift
    "manoeuvre": dict(mach=0.9, alpha=8.0, altitude=4600.0),
    "dash": dict(mach=1.6, alpha=2.0, altitude=12000.0),
    "high_alpha": dict(mach=0.3, alpha=25.0, altitude=6000.0),
    "gust": dict(mach=0.8, alpha=2.0, altitude=1000.0, gust_amplitude=0.06, gust_wavelength=30.0, unsteady=True),
}
PROFILES = {"quick": dict(h=0.3, steps=800), "standard": dict(h=0.2, steps=1500), "full": dict(h=0.13, steps=3000)}


def fast_model(fam: str, v: dict, mission: P.Mission, mach: float, alpha: float, alt: float) -> dict:
    d = D.complete(fam, v)
    des = D.build(d, fam)
    g = P.geometry(des)
    a = P.aero_model(des, g, mission)
    air = isa(alt)
    CL = P.lift_curve(a, mach, alpha) if alpha > 12 else (P._derivs_at(a, min(mach, 0.85)).CL0 + P.cl_alpha(a, mach) * np.radians(alpha))
    cd0 = P.cd0_at(a, air, mach)
    CD = cd0 + (CL * np.tan(np.radians(alpha)) if alpha > 12 else P.k_induced(a, mach, CL) * CL**2) + P.cd_wave(a, g, mach, CL)
    return {"CL": float(CL), "CD": float(CD), "CD0": float(cd0), "CDw": float(P.cd_wave(a, g, mach, CL)), "S": a.S}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="standard", choices=list(PROFILES))
    ap.add_argument("--designs", nargs="*", default=None)
    ap.add_argument("--states", nargs="*", default=None)
    a = ap.parse_args()
    prof = PROFILES[a.profile]
    out = RESULTS / "fighter" / "cfd"
    out.mkdir(parents=True, exist_ok=True)
    study = json.loads((RESULTS / "fighter" / "study.json").read_text())
    mission = P.Mission(calib=study["calibration"]["calib"])
    designs = {"reference": ("conventional", json.loads((RESULTS / "fighter" / "baseline_design.json").read_text())["variables"])}
    best = sorted(study["families"].items(), key=lambda kv: (not kv[1]["feasible"], kv[1]["J"]))
    for fam, r in best:
        designs[fam] = (fam, r["variables"])
    pick = a.designs or ["reference", best[0][0]]
    summ_path = out / "summary.json"
    summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    for name in pick:
        fam, v = designs[name]
        des = D.build(D.complete(fam, v), name)
        ref_geo = D.reference(des)
        L = des["variables"]["length"]
        ref = {"length": L, "area": ref_geo.S, "chord": ref_geo.c_mac, "span": ref_geo.b, "moment_point": [ref_geo.x_mac_le + 0.25 * ref_geo.c_mac, 0.0, 0.0]}
        for st in a.states or list(STATES):
            key = f"{name}_{st}"
            if key in summ and "cfd" in summ[key]:
                print(f"{key}: already computed (results/fighter/cfd/summary.json); delete that entry to rerun it", flush=True)
                continue
            s = dict(STATES[st])
            unsteady = bool(s.pop("unsteady", False))
            fl = runner.Flight(**s)
            num = runner.Numerics(cells_per_length=L / prof["h"], steps=prof["steps"], unsteady=unsteady, max_cells=2.4e6, cfl=1.2 if not unsteady else 0.8, residual_drop=2e-4)
            t0 = time.time()
            try:
                res = runner.run(des, fl, ref, out / key, num)
            except Exception as e:  # a failed run is recorded, the others go on
                summ[key] = {"error": str(e)[:400]}
                summ_path.write_text(json.dumps(summ, indent=1), encoding="utf-8")
                continue
            fm = fast_model(fam, v, mission, fl.mach, fl.alpha, fl.altitude)
            summ[key] = {"design": name, "family": fam, "state": st, "flight": s, "cfd": res["coefficients"], "rms": res.get("rms"), "converged": res["converged"], "cells": res["cells"],
                         "thin_surface_cells": res.get("thin_surface_cells"), "seconds": time.time() - t0, "separated_fraction": res.get("separated_fraction"), "T_wall_max_K": res.get("T_wall_max_K"),
                         "fast_model": fm}
            summ_path.write_text(json.dumps(summ, indent=1), encoding="utf-8")
            print(key, json.dumps({"cfd": {k: round(v_, 4) for k, v_ in res["coefficients"].items() if k in ("CL", "CD", "CD_friction", "Cm")}, "fast": {k: round(v_, 4) for k, v_ in fm.items()}}), flush=True)


if __name__ == "__main__":
    main()
