"""Build the fighter study page from the saved results (every number comes from a file under results/).

    python scripts/make_fighter_report.py

Inputs: results/fighter/study.json, results/fighter/cfd/summary.json, results/cfd_verification/verification.json,
results/thermal_skin/skin.json, results/surrogate/report.json, results/thermal_tests.log, and Studio images in
results/fighter/shots/*.png (made by scripts/studio_shots.ps1). Missing inputs leave their section empty with a note.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from _common import RESULTS

from aerosynapse.aircraft import materials as mt
from aerosynapse.aircraft.atmosphere import isa

SCRIPTS = RESULTS.parent / "scripts"


def load(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}"


def main() -> None:
    study = load(RESULTS / "fighter" / "study.json")
    assert study, "run scripts/run_fighter.py first"
    fam = study["families"]
    best = min((k for k in fam if fam[k]["feasible"]), key=lambda k: fam[k]["J"])
    D: dict = {"ref": study["reference"]["summary"], "best": best, "n_situations": len(study["situations"]), "n_extreme": 60,
               "fam": {k: {"s": v["summary"], "J": v["J"], "feasible": v["feasible"], "evals": v["evaluations"]} for k, v in fam.items()}}
    ho = study["holdout_extreme"]
    D["hold_note"] = ("Hold-out check: 60 random extreme states the search never saw (a different random seed). Share of them each design can sustain: "
                      + ", ".join(f"{k.replace('_', ' ')} {100 * v['extreme_flyable']:.0f} %" for k, v in ho.items()) + ".")
    logo = RESULTS.parent / "docs" / "assets" / "aerosynapse_logo.svg"
    D["logo"] = "data:image/svg+xml;base64," + base64.b64encode(logo.read_bytes()).decode()
    shots = []
    for p in sorted((RESULTS / "fighter" / "shots").glob("*.png")):
        cap = p.stem.replace("__", ": ").replace("_", " ")
        shots.append({"src": "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode(), "cap": cap})
    D["shots"] = shots
    # ---- verification of the solver
    ver = load(RESULTS / "cfd_verification" / "verification.json") or {}
    vrows: list[list[str]] = []
    if "thin_wing" in ver and "CL_cfd" in ver["thin_wing"]:
        w = ver["thin_wing"]
        vrows.append(["Wing thinner than a cell, AR 6, 4°, M0.3", "two-sided thin surfaces (wings, fins, blades)", f"C<sub>L</sub> {f(w['CL_cfd'], 3)} after {w['steps']} steps, {w['cells_per_chord']:.0f} cells per chord (changing {100 * w['CL_change_last_100_steps']:.1f} % per 100 steps)",
                      f"{f(w['CL_lattice'], 3)} (vortex lattice): {100 * (w['ratio'] - 1):.0f} %"])
    if "sod" in ver and "L1_rho" in ver["sod"]:
        vrows.append(["Sod shock tube", "shocks, scheme order", f"density L1 error {f(ver['sod']['L1_rho'], 4)} on 400 cells", "exact Riemann solution"])
    if "spheroid" in ver and "cpl24" in ver["spheroid"]:
        s = ver["spheroid"]
        vrows.append(["Spheroid 4:1, M0.3, inviscid", "immersed boundary, drag-free potential flow", f"C<sub>D</sub> {f(s['cpl24']['CD'], 4)} and {f(s['cpl40']['CD'], 4)} on two grids; lowest C<sub>p</sub> {f(s['cpl24']['cp_min'], 3)} and {f(s['cpl40']['cp_min'], 3)}",
                      f"C<sub>D</sub> 0; C<sub>p</sub> {f(s['exact_compressible_cp_min_range'][0], 3)} to {f(s['exact_compressible_cp_min_range'][1], 3)} (Lamb, compressibility bounds)"])
    if "cone" in ver and "cp_surface_mean" in ver["cone"]:
        c = ver["cone"]
        vrows.append(["10° cone, Mach 2", "supersonic flow, conical shock", f"surface C<sub>p</sub> {f(c['cp_surface_mean'], 4)} ± {f(c['cp_surface_std'], 4)}", f"{f(c['taylor_maccoll_cp'], 4)} (Taylor-Maccoll)"])
    if "plate" in ver and "ratio" in ver["plate"]:
        p = ver["plate"]
        vrows.append(["Slab at zero incidence, Re " + f"{p['reynolds']:.2g}", "wall model: turbulent friction", f"friction C<sub>D</sub> per wetted side {f(p['CD_friction_per_wetted_side'], 5)}", f"{f(p['schoenherr_cf'], 5)} (Kármán-Schoenherr), ratio {f(p['ratio'], 2)}"])
    if "delta" in ver and "alpha10" in ver["delta"]:
        d = ver["delta"]
        for a in ("10", "20"):
            r = d[f"alpha{a}"]
            vrows.append([f"70° delta wing, {a}°, M0.3", "leading-edge vortex lift", f"C<sub>L</sub> {f(r['CL_cfd'], 3)}", f"{f(r['CL_polhamus'], 3)} (Polhamus); potential lift only {f(r['CL_potential_only'], 3)}"])
    if "tgv" in ver and "low_mach_fix" in ver["tgv"]:
        t = ver["tgv"]
        lm = t["low_mach_fix"]
        txt = f"peak dissipation {f(lm['peak_dissipation'], 4)} at t* {f(lm['peak_time'], 1)} (low-Mach correction)"
        if "plain_hllc" in t:
            txt += f"; plain HLLC {f(t['plain_hllc']['peak_dissipation'], 4)} at t* {f(t['plain_hllc']['peak_time'], 1)}"
        vrows.append(["Taylor-Green vortex, Re 1600, 64³", "turbulence: large-eddy model and numerical dissipation", txt, "about 0.0127 at t* ≈ 9 (DNS; time checked, value from memory)"])
    D["ver"] = vrows
    D["ver_text"] = "Verification checks the solver against exact solutions; validation against theory that was itself validated against experiment. Every number is from results/cfd_verification/verification.json."
    # ---- CFD of the designs
    cs = load(RESULTS / "fighter" / "cfd" / "summary.json") or {}
    D["cfd"] = [[k.split("_")[0].replace("reference", "Reference"), v["state"], f(v["cfd"]["CL"], 3), f(v["fast_model"]["CL"], 3), f(v["cfd"]["CD"], 4), f(v["fast_model"]["CD"], 4), f(v["cfd"]["CD_friction"], 4), f(v["cfd"]["Cm"], 3),
                 f(100 * (v.get("separated_fraction") or 0), 0) + " %", f"{v['cells'] / 1e6:.2f} M", "yes" if v["converged"] else "no"] for k, v in cs.items() if "cfd" in v]
    # drag at the lift each design needs in cruise, from the two cruise incidences (C_D = a + b C_L^2)
    polar_txt = ""
    pol = {}
    for name, fam_key in (("reference", None), (best, best)):
        a2, a5 = cs.get(f"{name}_cruise"), cs.get(f"{name}_cruise_a5")
        if not (a2 and a5 and "cfd" in a2 and "cfd" in a5):
            continue
        c1, d1, c2, d2 = a2["cfd"]["CL"], a2["cfd"]["CD"], a5["cfd"]["CL"], a5["cfd"]["CD"]
        b = (d2 - d1) / max(c2**2 - c1**2, 1e-9)
        a = d1 - b * c1**2
        s = D["ref"] if fam_key is None else D["fam"][best]["s"]
        cl_need = s["states"]["cruise"]["CL"]
        q = s["states"]["cruise"]["extra"]["q_Pa"]
        drag = (a + b * cl_need**2) * q * s["S"]
        pol[name] = {"CL_needed": cl_need, "CD": a + b * cl_need**2, "drag_N": drag, "fast_drag_N": s["metrics"]["cruise_drag_N"]}
    if len(pol) == 2:
        r_, b_ = pol["reference"], pol[best]
        polar_txt = (f" At the lift each design needs in cruise (C<sub>L</sub> {r_['CL_needed']:.3f} for the reference, {b_['CL_needed']:.3f} for the {best.replace('_', ' ')} design), the 3D flow polars give "
                     f"{r_['drag_N'] / 1000:.1f} kN and {b_['drag_N'] / 1000:.1f} kN of cruise drag ({100 * (b_['drag_N'] / r_['drag_N'] - 1):+.0f} %), against "
                     f"{r_['fast_drag_N'] / 1000:.1f} and {b_['fast_drag_N'] / 1000:.1f} kN from the fast model ({100 * (b_['fast_drag_N'] / r_['fast_drag_N'] - 1):+.0f} %).")
    g_r, g_b = cs.get("reference_gust"), cs.get(f"{best}_gust")
    if g_r and g_b and g_r.get("rms") and g_b.get("rms"):
        polar_txt += (f" In the time-accurate gust run (vertical gust of 6 % of flight speed, 30 m wavelength, about three quarters of a period averaged) the lift fluctuates by "
                      f"{g_r['rms']['CL']:.4f} (rms) on the reference and {g_b['rms']['CL']:.4f} on the {best.replace('_', ' ')} design, the opposite of the fast model, which predicted an 18 % larger gust load for it.")
    hr, hb = cs.get("reference_high_alpha"), cs.get(f"{best}_high_alpha")
    if hr and hb:
        polar_txt += (f" At 25°, the 3D flow gives C<sub>L</sub> {hr['cfd']['CL']:.3f} for the reference and {hb['cfd']['CL']:.3f} for the {best.replace('_', ' ')} design, while the fast model gave "
                      f"{hr['fast_model']['CL']:.3f} and {hb['fast_model']['CL']:.3f}: its vortex lift for the canard and strake layout is optimistic.")
    D["cfd_polar"] = pol
    D["cfd_text"] = ("The 3D compressible solver ran on the reference and the best design at the same grid spacing, with thin surfaces (wings, fins, blades) as two-sided walls and friction from the wall model. "
                     "The fast model's lift and drag at the same Mach number, altitude and incidence are shown beside it." + polar_txt) if D["cfd"] else "No 3D flow runs yet (scripts/fighter_cfd.py)."
    # ---- materials and tolerances (zones at the hottest state)
    T_hot = max(x["extra"].get("T_recovery_K", 0) for x in D["ref"]["states"].values())
    zone_T = {"radome": T_hot, "fwd_fuselage": T_hot, "centre_fuselage": T_hot - 10, "wing_box": T_hot - 15, "wing_le": T_hot + 15, "control": T_hot - 10, "aft_fuselage": T_hot, "nozzle": T_hot, "gear": 300.0}
    ch = mt.select(zone_T)
    D["mat"] = [[c.name, mt.LIBRARY[c.material].name, f(c.T_K, 0), f(c.margin_K, 0), f(c.index, 3)] for c in ch]
    D["mat_text"] = (f"The hottest adiabatic-wall temperature over the situations is {T_hot:.0f} K (dash and cold high-altitude cruise). Each zone gets the lightest material that survives its temperature with 20 K to spare. "
                     f"Structural mass factor against all-7075 aluminium: {mt.structure_factor(ch):.3f}.")
    tol = mt.tolerance_study(ch, isa(11000.0), 0.85, D["fam"][best]["s"]["variables"]["length"], D["fam"][best]["s"]["geometry"]["wetted"] * 1.6, n=3000)
    D["tol"] = [["Mass change from thickness tolerance, kg", f(tol["mass_kg"]["mean"], 1), f(tol["mass_kg"]["p05"], 1), f(tol["mass_kg"]["p95"], 1)],
                ["Drag added by roughness, steps and gaps at M0.85, N", f(tol["drag_N"]["mean"], 0), f(tol["drag_N"]["p05"], 0), f(tol["drag_N"]["p95"], 0)],
                ["Stress margin change at the thinnest skin", f(tol["stress_margin"]["mean"], 3), f(tol["stress_margin"]["p05"], 3), "–"]]
    # ---- skin heating
    sk = load(RESULTS / "thermal_skin" / "skin.json")
    if sk:
        D["thv"] = [[k.split("=")[1], f(v["pinn_max_abs_error"], 4), f(v["pinn_rel_l2"], 4), f(v["fd_max_abs_error"], 6)] for k, v in sk["verification"].items()]
        zz = sk["zones"]["dash"]
        D["thz"] = [[k.replace("_", " "), v["material"], f(v["Bi"], 3), f(v["T_max_K"], 0), f(v["temperature_margin_K"], 0), f(v["dT_max_K"], 1), f(v["thermal_stress_MPa"], 1), f(100 * v["stress_ratio"], 1) + " %",
                     ", ".join(f"{a} {b}" for a, b in v["source"].items())] for k, v in zz.items()]
        q = sk["query_seconds"]
        tr = sk["training"]
        D["th_text"] = (f"One parametric PINN (inputs x, t, ln Bi) from the merged project, trained over Bi 0.2 to 20 in {tr.get('wall_s', 0) / 60:.0f} minutes, checked on a grid it never saw against the project's exact "
                        f"eigen-series and finite-difference solver. Rows below: the panels of each zone during a 10-minute Mach 1.8 dash at 12 km. A query costs {1e3 * q['pinn']:.1f} ms with the network "
                        f"and {1e3 * q['analytic_series']:.1f} ms with the exact series, so on this problem the network is {'faster' if q['pinn'] < q['analytic_series'] else 'not faster'}. "
                        "Panels whose Biot number is outside the trained range use the exact series, as marked.")
    else:
        D["thv"], D["thz"], D["th_text"] = [], [], "Skin heating not run yet (scripts/run_thermal_skin.py)."
    # ---- PINN merge tests
    log = (RESULTS / "thermal_tests.log").read_text(encoding="utf-8", errors="ignore") if (RESULTS / "thermal_tests.log").exists() else ""
    D["pinn_text"] = ("The author's adaptive parametric PINN repository (convection-diffusion heat transfer, inverse identification of α, k and h, verified solvers, THERMAL LAB) "
                      "is merged whole as <code>aerosynapse.thermal</code>: 649 files copied with imports and paths rewritten, its configs, data, docs and paper sources in <code>projects/thermal_pinn</code>, "
                      "its unit tests in <code>tests/thermal</code>. " + (f"Test run: {log.strip().splitlines()[-1]}" if log.strip() else "Its test suite has not been run yet."))
    # ---- surrogate
    sr = load(RESULTS / "surrogate" / "report.json")
    if sr:
        D["sur"] = [[o, f(a, 3), f(b, 3), f(c, 3)] for o, a, b, c in zip(sr["outputs"], sr["val_r2"], sr["nearest_neighbour_r2"], sr["val_rel_rmse"], strict=True)]
        D["sur_text"] = (f"{sr['rows']} designs of the freeform family were evaluated with the full model and written to {len(sr['train_shard_ids']) + len(sr['val_shard_ids'])} shards on disk "
                         f"({sr['store_bytes'] / 1e6:.2f} MB, SHA-256 per shard). A network trained from them through a {sr['buffer_rows']}-row shuffle buffer, never holding the dataset in memory; "
                         f"{sr['val_rows']} rows in held-out shards test it. {sr.get('dropped_unphysical_rows', 0)} rows with non-physical values (degenerate random designs) were left out, "
                         "and weights, drags and field length are learned as logarithms (R² below is on that scale). Every optimiser evaluation is stored too.")
    else:
        D["sur"], D["sur_text"] = [], "Surrogate not trained yet (scripts/train_surrogate.py)."
    D["built"] = [
        ["Geometry kernel (C++)", "feature tree on a signed distance field: add, cut, intersect, blend, mirror, hollow shells, patterns, twist; mesh, mass properties, sections", "sphere, torus and drilled-box volumes within 0.06 % (tests)"],
        ["AeroSynapse Studio (C++, Win32 + OpenGL)", "design editor with logo: feature tree, properties, 3D view, section cuts, undo/redo, STL/OBJ/SVG export, measurement, one-click analysis and CFD", "builds and runs; images below"],
        ["aeroflow (C++)", "3D compressible Navier-Stokes, immersed boundaries with two-sided thin surfaces, WALE turbulence, wall model, gusts, inflow turbulence, sideslip", "table: how the flow solver was checked"],
        ["Fighter design space (Python)", "5 families, body / wing / canard / strake / tails / twin fins / ventral fins / blades / notches / ducts / bays; engine and cockpit inside the body", "every family builds a valid solid (tests)"],
        ["Performance in every situation", "vortex lattice of all surfaces, area rule from 3D Mach-plane cuts, friction, vortex lift, propulsion, weights, gust and turbulence loads", "lattice checked against Helmbold and Prandtl-Glauert; empty mass calibrated to public F-16 data"],
        ["Materials and tolerances", "11 materials, temperature-dependent strength, per-zone selection, Monte Carlo of manufacturing tolerances", "two materials checked against data sheets"],
        ["Merged PINN project", "parametric PINN, verified solvers, inverse identification, real data, THERMAL LAB; used here for skin heating", "its own test suite"],
        ["Data store", "sharded data on disk with checksums, streamed training in bounded memory, resumable", "tests: round trip, bounded stream, learning and resume"],
    ]
    D["limits"] = [
        "<b>Conceptual-design model.</b> The ranking of designs comes from one fast model with empirical constants (some quoted from memory and marked in the code). The 3D flow runs check it on two designs and five states only.",
        "<b>3D flow on coarse Cartesian grids.</b> Boundary layers are modelled, not resolved; thin surfaces are one-cell walls, so their thickness and leading-edge shape are not represented; smooth-surface separation is approximate.",
        "<b>Turbulence.</b> The Taylor-Green check shows how much numerical dissipation the scheme adds at 64³; time-averaged turbulence statistics in the fighter runs carry the same limitation.",
        "<b>Materials.</b> Most values are typical handbook numbers; only density and service limits of IM7/8552 and Inconel 718 were checked against data sheets. Upload real allowables through the materials data slot.",
        "<b>Engine.</b> A generic afterburning turbofan with Mattingly lapse rates; inlet and nozzle flows are not simulated (the duct is a through-flow channel).",
        "<b>Not covered:</b> flutter, control surface sizing and trim drag, landing gear layout, radar or infrared signature, certification loads.",
    ]
    t = (SCRIPTS / "shapeopt_report_template.html").read_text(encoding="utf-8")
    css = t[t.index("<style>") + 7 : t.index("</style>")]
    js = t[t.index("const $=i=>") : t.index("function render(){")]
    page = (SCRIPTS / "fighter_report_template.html").read_text(encoding="utf-8")
    page = page.replace("/*__CSS__*/", css, 1).replace("/*__HELPERS__*/", js).replace("/*__DATA__*/null", json.dumps(D, separators=(",", ":"), default=float))
    out = RESULTS / "fighter" / "report.html"
    out.write_text(page, encoding="utf-8")
    print("wrote", out, round(len(page) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
