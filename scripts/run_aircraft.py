"""Whole-aircraft study: optimise every body family for a mission, then compare and verify.

    python scripts/run_aircraft.py --profile standard

Results go to results/aircraft/: study.json, and for each winner an STL surface and a design file.
Finished jobs are reused, so a stopped run continues where it left off.
"""

from __future__ import annotations

import argparse
import json
import time

from _common import RESULTS
from _jobs import run_jobs

from aerosynapse.aircraft import analysis as an
from aerosynapse.aircraft import layers as Y
from aerosynapse.aircraft import panel3d as P3
from aerosynapse.aircraft import vehicle as vh
from aerosynapse.aircraft.atmosphere import isa
from aerosynapse.shapeopt.geometry import write_stl

PROFILES = {"quick": dict(budget=320, seeds=1, pop=12, restarts=0), "standard": dict(budget=4000, seeds=2, pop=16, restarts=2), "full": dict(budget=12000, seeds=3, pop=20, restarts=3)}
MISSIONS = {
    "airliner": dict(baseline="tube_and_wing", oew=41000.0, families=["tube_and_wing", "blended_wing_body", "flying_wing", "freeform", "freeform_euclidean"]),
    "supersonic": dict(baseline="supersonic_jet", oew=None, families=["supersonic_jet", "supersonic_freeform"]),
}


def thermal_sweep(d: dict, material: str) -> list[dict]:
    """Skin temperature and thermal layer against Mach number, for the temperature study."""
    v = vh.build(d)
    out = []
    for M, h in ((0.8, 11000), (1.2, 13000), (1.6, 16000), (2.0, 17000), (2.5, 19000), (3.0, 22000), (3.5, 25000)):
        for dT in (0.0, 25.0):
            air = isa(h, dT)
            t = Y.thermal_state(air, M, d["L"], v.S_wet_body + v.S_wet_wing, max(0.02 * d["H"], 0.02), Y.MATERIALS[material])
            out.append({"mach": M, "altitude": h, "dT": dT, "T_air_C": air.T - 273.15, "T_total_C": t.T_total - 273.15, "T_nose_C": t.T_nose - 273.15, "T_skin_C": t.T_skin - 273.15, "T_aft_C": t.T_skin_aft - 273.15, "q_nose_kW": t.q_nose / 1e3, "expansion_mm": t.expansion_m * 1e3, "E_ratio": t.E_ratio})
    return out


def verify_body(d: dict, Re: float) -> dict:
    """Panel method and boundary layer on the real body shape, top and side meridians."""
    v = vh.build(d)
    b = P3.from_grid(P3.body_grid(v, 34))
    f = P3.solve(b)
    nj = b.shape[1]
    return {"panels": len(b.A), "cp_min": float(f.cp.min()), "top": P3.meridian_layer(b, f, Re, j=nj // 4), "side": P3.meridian_layer(b, f, Re, j=0), "bottom": P3.meridian_layer(b, f, Re, j=3 * nj // 4)}


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--profile", default="standard", choices=list(PROFILES))
    a.add_argument("--workers", type=int, default=4)
    n = a.parse_args()
    p = PROFILES[n.profile]
    out = RESULTS / "aircraft"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = []
    for mname, ms in MISSIONS.items():
        for fam in ms["families"]:
            # the Euclidean against non-Euclidean question needs more seeds than the others
            n_seeds = max(p["seeds"], 5) if fam in ("freeform", "freeform_euclidean") and n.profile != "quick" else p["seeds"]
            for seed in range(n_seeds):
                jobs.append((f"{mname}_{fam}_{seed}", "aircraft", {"mission": mname, "family": fam, "baseline": ms["baseline"], "oew": ms["oew"], "seed": seed, "budget": p["budget"], "pop": p["pop"], "restarts": p["restarts"]}))
    res = run_jobs(jobs, out / "jobs" / n.profile, max_workers=n.workers)
    study: dict = {"profile": n.profile, "missions": {}, "wall_s": time.time() - t0}
    for mname, ms in MISSIONS.items():
        fams: dict = {}
        rows = [r for r in res if r["mission"] == mname]
        for r in rows:
            f = fams.setdefault(r["family"], [])
            f.append(r)
        best = {k: min(v, key=lambda r: r["J"]) for k, v in fams.items()}
        mat = (an.SUPERSONIC if mname == "supersonic" else {}).get("material", "aluminium")
        out_m: dict = {"baseline": rows[0]["baseline_summary"], "baseline_constraints": rows[0]["baseline_constraints"], "families": {}}
        for k, r in best.items():
            d = r["design"]
            v = vh.build(d)
            V, F = vh.surface(v)
            write_stl(out / f"{mname}_{k}.stl", V, F, k)
            (out / f"{mname}_{k}.design.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
            Re = 1.2e7 * d["L"]
            try:
                ver = verify_body(d, Re)
            except (ValueError, IndexError, FloatingPointError) as e:
                ver = {"error": str(e)}
            out_m["families"][k] = {**{q: r[q] for q in ("J", "design", "summary", "constraints", "off_design", "off_names", "history", "evals", "wall_s")}, "seeds_J": [q["J"] for q in fams[k]], "verify": ver, "thermal": thermal_sweep(d, mat)}
        base_d = vh.PRESETS[ms["baseline"]]
        out_m["baseline_design"] = base_d
        out_m["baseline_verify"] = verify_body(base_d, 1.2e7 * base_d["L"])
        out_m["baseline_thermal"] = thermal_sweep(base_d, mat)
        V, F = vh.surface(vh.build(base_d))
        write_stl(out / f"{mname}_baseline.stl", V, F, "baseline")
        study["missions"][mname] = out_m
    (out / "study.json").write_text(json.dumps(study, default=float), encoding="utf-8")
    for mname, m in study["missions"].items():
        print(f"\n{mname}: baseline L/D {m['baseline']['L_over_D']:.2f} fuel {m['baseline']['fuel_t']:.2f} t")
        for k, f in m["families"].items():
            s = f["summary"]
            bad = [q for q, x in f["constraints"].items() if x > 1e-3]
            print(f"  {k:24s} J {f['J']:.4f} L/D {s['L_over_D']:.2f} fuel {s['fuel_t']:.2f} t  span {s['span_m']:.1f}  violated {bad}")


if __name__ == "__main__":
    main()
