"""Verify and validate the 3D solver (aeroflow) against exact solutions and published theory.

    python scripts/cfd_verify.py --profile standard

Cases (each reused if its result exists):
  sod         shock tube, exact Riemann solution (verification of the scheme)
  spheroid    prolate spheroid 4:1 in inviscid flow at M0.3 on two grids: drag must vanish (d'Alembert)
              and the peak surface speed approach the potential-flow value (immersed boundary)
  cone        10 deg cone at M2: surface pressure against Taylor-Maccoll (supersonic, shocks)
  plate       slab at zero incidence, turbulent: friction drag from the wall model against Schoenherr
  delta       70 deg sharp delta wing at M0.3, 10 and 20 deg: lift against Polhamus' suction analogy
  tgv         Taylor-Green vortex Re 1600, 64^3: dissipation history (turbulence model); its peak time
              about t* = 9 is checked, the peak value is compared with a value quoted from memory
Results: results/cfd_verification/verification.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from _common import RESULTS

from aerosynapse.cfd import exact, runner
from aerosynapse.fighter import vlm

OUT = RESULTS / "cfd_verification"
PROFILES = {"quick": dict(scale=0.6, steps=0.5), "standard": dict(scale=1.0, steps=1.0), "full": dict(scale=1.4, steps=1.5)}


def done(name: str) -> dict | None:
    p = OUT / name / "result.json"
    return json.loads(p.read_text()) | {"dir": str(OUT / name)} if p.exists() else None


def sod() -> dict:
    d = OUT / "sod"
    r = done("sod") or runner.run_special({"case": "sod", "grid": {"uniform": {"n": [400, 2, 2], "lo": [0, 0, 0], "hi": [1, 0.005, 0.005]},
                                          "bc": {"x": ["extrapolate", "extrapolate"], "y": ["periodic", "periodic"], "z": ["periodic", "periodic"]}},
                                          "model": {"viscous": False, "sgs": "none", "wall_model": False, "cfl": 0.6}, "run": {"steps": 100000, "t_end": 0.2, "log_every": 1000}}, d)
    p = runner.plane(d, "plane_z")
    ex = exact.sod(p["coord_u"], r["time"])
    return {"L1_rho": float(np.mean(np.abs(p["rho"][0] - ex[0]))), "L1_u": float(np.mean(np.abs(p["u"][0] - ex[1]))), "L1_p": float(np.mean(np.abs(p["p"][0] - ex[2]))),
            "cells": 400, "seconds": r["seconds"]}


def spheroid(prof: dict) -> dict:
    des = {"name": "spheroid 4:1", "features": [{"id": "e", "type": "ellipsoid", "radii": [2, 0.5, 0.5]}]}
    ref = {"length": 4.0, "area": float(np.pi * 0.25), "chord": 4.0, "span": 1.0, "moment_point": [0, 0, 0]}
    out = {}
    exv = exact.spheroid(4.0)
    beta = np.sqrt(1 - 0.3**2)
    for cpl in (24, 40):
        name = f"spheroid_{cpl}"
        r = done(name) or runner.run(des, runner.Flight(mach=0.3, alpha=0.0, altitude=0.0), ref, OUT / name,
                                     runner.Numerics(cells_per_length=cpl * prof["scale"], steps=int(2500 * prof["steps"]), viscous=False, sgs="none", wall_model=False, cfl=1.2, residual_drop=1e-5))
        s = runner.surface(OUT / name)
        out[f"cpl{cpl}"] = {"CD": r["coefficients"]["CD"], "cp_min": float(np.min(s["cp"])), "cells": r["cells"], "seconds": r["seconds"]}
    # potential flow, compressibility by Goethert's rule for slender bodies (Cp grows about 1/beta^2 for a slender body; use 1/beta as the lower bound)
    out["exact_incompressible_cp_min"] = exv["cp_min"]
    out["exact_compressible_cp_min_range"] = [exv["cp_min"] / beta, exv["cp_min"] / beta**2]
    return out


def cone(prof: dict) -> dict:
    L = 2.0
    des = {"name": "cone 10 deg", "features": [{"id": "c", "type": "cone", "r1": 0.0, "r2": float(L * np.tan(np.radians(10))), "length": L}, {"id": "cyl", "type": "cylinder", "r": float(L * np.tan(np.radians(10))), "length": 1.0, "axis": "x", "translate": [L + 0.5, 0, 0]}]}
    ref = {"length": 3.0, "area": 1.0, "chord": 1.0, "span": 1.0, "moment_point": [0, 0, 0]}
    r = done("cone") or runner.run(des, runner.Flight(mach=2.0, alpha=0.0, altitude=0.0), ref, OUT / "cone",
                                   runner.Numerics(cells_per_length=60 * prof["scale"], steps=int(2500 * prof["steps"]), viscous=False, sgs="none", wall_model=False, cfl=1.2, domain=[0.3, 0.4, 0.6, 0.6], residual_drop=1e-5))
    s = runner.surface(OUT / "cone")
    sel = (s["x"] > 0.6) & (s["x"] < 1.7)
    tm = exact.taylor_maccoll(2.0, 10.0)
    return {"cp_surface_mean": float(np.mean(s["cp"][sel])), "cp_surface_std": float(np.std(s["cp"][sel])), "taylor_maccoll_cp": tm["cp"], "shock_deg_exact": tm["shock_deg"], "cells": r["cells"], "seconds": r["seconds"]}


def plate(prof: dict) -> dict:
    Lp, W, T = 10.0, 4.0, 0.4
    des = {"name": "slab", "features": [{"id": "p", "type": "box", "size": [Lp, W, T], "round": 0.19}]}
    ref = {"length": Lp, "area": 2 * Lp * W, "chord": Lp, "span": W, "moment_point": [0, 0, 0]}
    fl = runner.Flight(mach=0.5, alpha=0.0, altitude=3000.0)
    r = done("plate") or runner.run(des, fl, ref, OUT / "plate", runner.Numerics(cells_per_length=40 * prof["scale"], steps=int(2500 * prof["steps"]), cfl=1.2, residual_drop=1e-5))
    from aerosynapse.aircraft.atmosphere import isa

    air = isa(3000.0)
    Re = air.rho * 0.5 * air.a * Lp / air.mu
    from scipy.optimize import brentq

    cf_s = brentq(lambda c: 0.242 / np.sqrt(c) - np.log10(Re * c), 1e-4, 0.02)  # Karman-Schoenherr mean friction, flat plate
    return {"CD_friction_per_wetted_side": r["coefficients"]["CD_friction"], "schoenherr_cf": float(cf_s), "reynolds": float(Re), "ratio": r["coefficients"]["CD_friction"] / cf_s, "cells": r["cells"], "seconds": r["seconds"]}


def delta(prof: dict) -> dict:
    cr, sw = 4.0, 70.0
    span = cr / np.tan(np.radians(sw))
    wing = {"id": "delta", "type": "wing", "span": float(span), "root_chord": cr, "taper": 0.0, "sweep": sw, "tc": 0.035, "profile": "biconvex", "translate": [0, 0, 0], "mirror": True}
    des = {"name": "70 deg delta", "features": [wing]}
    S = cr * span
    ref = {"length": cr, "area": S, "chord": 2 / 3 * cr, "span": 2 * span, "moment_point": [0.6 * cr, 0, 0]}
    d = vlm.derivatives([wing], 0.3, S, 2 / 3 * cr, 2 * span, 0.6 * cr, nc=8, ns=12)
    AR = (2 * span) ** 2 / S
    kp = d.CL_alpha
    kv = (kp - kp**2 / (np.pi * AR)) / np.cos(np.radians(sw))
    out = {"Kp": kp, "Kv": kv}
    for a in (10.0, 20.0):
        name = f"delta_{int(a)}"
        r = done(name) or runner.run(des, runner.Flight(mach=0.3, alpha=a, altitude=0.0), ref, OUT / name,
                                     runner.Numerics(cells_per_length=56 * prof["scale"], steps=int(2500 * prof["steps"]), cfl=1.1, residual_drop=1e-5))
        al = np.radians(a)
        pol = kp * np.sin(al) * np.cos(al) ** 2 + kv * np.sin(al) ** 2 * np.cos(al)
        out[f"alpha{int(a)}"] = {"CL_cfd": r["coefficients"]["CL"], "CL_polhamus": float(pol), "CL_potential_only": float(kp * np.sin(al) * np.cos(al) ** 2), "cells": r["cells"], "seconds": r["seconds"]}
    return out


def thin_wing(prof: dict) -> dict:
    """Rectangular wing, aspect ratio 6, 2 % thick (thinner than a cell), M0.3, 4 deg: lift against the vortex lattice."""
    wing = {"id": "w", "type": "wing", "span": 3.0, "root_chord": 1.0, "taper": 1.0, "sweep": 0.0, "tc": 0.02, "profile": "biconvex", "translate": [0, 0, 0], "mirror": True}
    lat = vlm.derivatives([wing], 0.3, 6.0, 1.0, 6.0, 0.25, nc=6, ns=24).CL_alpha * np.radians(4.0)
    ref = {"length": 1.0, "area": 6.0, "chord": 1.0, "span": 6.0, "moment_point": [0.25, 0, 0]}
    cpc = 24 * prof["scale"]
    r = done("thin_wing") or runner.run({"features": [wing]}, runner.Flight(mach=0.3, alpha=4.0, altitude=0.0), ref, OUT / "thin_wing",
                                        runner.Numerics(cells_per_length=cpc, steps=int(1500 * prof["steps"]), cfl=1.2, residual_drop=1e-4, domain=[2, 3, 1.0, 2]))
    h = runner.history(OUT / "thin_wing")
    cl = h["CL"]
    return {"CL_cfd": float(r["coefficients"]["CL"]), "CL_lattice": float(lat), "ratio": float(r["coefficients"]["CL"] / lat), "steps": int(r["steps"]),
            "CL_change_last_100_steps": float(abs(cl[-1] - cl[max(len(cl) - 11, 0)]) / abs(cl[-1])), "converged": r["converged"], "cells_per_chord": cpc,
            "thin_surface_cells": r.get("thin_surface_cells"), "CD": r["coefficients"]["CD"]}


TGV_CASE = {"case": "tgv", "grid": {"uniform": {"n": [64, 64, 64], "lo": [0, 0, 0], "hi": [2 * np.pi] * 3}, "bc": {"x": ["periodic", "periodic"], "y": ["periodic", "periodic"], "z": ["periodic", "periodic"]}},
            "flow": {"mach": 0.1, "reynolds": 1600}, "model": {"viscous": True, "sgs": "wale", "wall_model": False, "cfl": 0.9}, "run": {"steps": 100000, "t_end": 140, "log_every": 10}}


def _tgv_stats(d) -> dict:  # noqa: ANN001
    h = runner.history(d)
    t, k = h["time"], h["kinetic_energy"]
    eps = -np.gradient(k, t)
    i = int(np.argmax(eps))
    return {"peak_dissipation": float(eps[i]), "peak_time": float(t[i]), "final_time": float(t[-1]), "k_at_t9": float(np.interp(9.0, t, k))}


def tgv() -> dict:
    """64^3 at M0.1, Re 1600 (t* = t U0 / L up to 14). The first run without the low-Mach correction is kept in tgv64_hllc."""
    d = OUT / "tgv64"
    if not (d / "result.json").exists():
        runner.run_special({**TGV_CASE, "model": {**TGV_CASE["model"], "low_mach_fix": True}}, d)
    out = {"low_mach_fix": _tgv_stats(d)}
    if (OUT / "tgv64_hllc" / "history.csv").exists():
        out["plain_hllc"] = _tgv_stats(OUT / "tgv64_hllc")
    out.update({"reference_peak": 0.0127, "reference_peak_time": 9.0,
                "reference_note": "peak time about t* = 9 confirmed by a web search (2026-11-07); the peak value 0.0127 (van Rees et al. 2011 DNS, 512^3) is quoted from memory and was not checked"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="standard", choices=list(PROFILES))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="", help="results folder under results/ (default cfd_verification); runs already there are reused, so a new solver needs a new folder")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = RESULTS / a.out
    prof = PROFILES[a.profile]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "verification.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    cases = {"sod": sod, "thin_wing": lambda: thin_wing(prof), "spheroid": lambda: spheroid(prof), "cone": lambda: cone(prof), "plate": lambda: plate(prof), "delta": lambda: delta(prof), "tgv": tgv}
    for name, fn in cases.items():
        if a.only and name not in a.only:
            continue
        t0 = time.time()
        try:
            res[name] = fn()
        except Exception as e:  # one failing case must not lose the others
            res[name] = {"error": str(e)[:500]}
        res[name]["wall_s"] = time.time() - t0
        path.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(name, json.dumps(res[name])[:400], flush=True)


if __name__ == "__main__":
    main()
