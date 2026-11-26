"""Fighter workspaces of the Designer app: feature-tree editor, study results, 3D flow, materials, skin heating
(parametric PINN) and the on-disk data store. The native editor (AeroSynapse Studio) opens the same files."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aerosynapse.aircraft import materials as mt
from aerosynapse.aircraft.atmosphere import isa
from aerosynapse.cad import kernel
from aerosynapse.cfd import runner
from aerosynapse.fighter import design as D
from aerosynapse.fighter import performance as P
from aerosynapse.ui import jobs

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "results" / "fighter"
LOGO = ROOT / "docs" / "assets" / "aerosynapse_logo.png"


def stl_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    n = int(np.frombuffer(raw[80:84], np.uint32)[0])
    rec = np.frombuffer(raw[84 : 84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    v = rec["v"].reshape(-1, 3).astype(float)
    keys, inv = np.unique(np.round(v, 5), axis=0, return_inverse=True)
    return keys, inv.reshape(-1, 3)


def design_figure(design: dict[str, Any], h: float = 0.07, colour: np.ndarray | None = None, title: str = "") -> go.Figure:
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "m.stl"
        kernel.mesh(design, f, h=h)
        V, T = stl_mesh(f)
    kw: dict[str, Any] = {"color": "#9fb3c8"} if colour is None else {"intensity": colour, "colorscale": "RdBu_r", "showscale": True}
    fig = go.Figure(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=T[:, 0], j=T[:, 1], k=T[:, 2], flatshading=False, lighting={"ambient": 0.45, "diffuse": 0.8, "specular": 0.3}, **kw))
    fig.update_layout(scene={"aspectmode": "data", "xaxis_title": "x (aft), m", "yaxis_title": "y, m", "zaxis_title": "z, m", "camera": {"eye": {"x": -1.4, "y": -1.3, "z": 0.8}}},
                      margin={"l": 0, "r": 0, "t": 30, "b": 0}, title=title, height=520)
    return fig


def page_editor() -> None:
    st.header("Fighter design editor")
    st.caption("A design is a feature tree: bodies, wings, canards, tails, fins, strakes, blades and primitives that are added, cut or intersected, with blends, mirror, hollow shells, patterns and twist. "
               "The same file opens in AeroSynapse Studio (native editor) and runs in the 3D flow solver.")
    files = sorted(FIG.glob("*.design.json")) + sorted(FIG.glob("baseline_design.json")) + sorted((ROOT / "projects").glob("*.json"))
    c1, c2 = st.columns([2, 1])
    pick = c1.selectbox("Design file", [str(f.relative_to(ROOT)) for f in files] or ["(none)"])
    if c2.button("Open in AeroSynapse Studio", help="the native C++ editor"):
        exe = kernel.build("AeroSynapseStudio")
        subprocess.Popen([str(exe), str(ROOT / pick)])
        st.success("Studio opened in its own window.")
    if not files:
        return
    if st.session_state.get("ed_file") != pick:
        st.session_state["ed_file"] = pick
        st.session_state["ed_doc"] = json.loads((ROOT / pick).read_text(encoding="utf-8"))
    doc = st.session_state["ed_doc"]
    feats = doc["features"]
    left, right = st.columns([1, 2])
    with left:
        names = [f"{'-' if f.get('op') == 'cut' else '+'} {f.get('id')} ({f.get('type')})" for f in feats]
        k = st.radio("Features", range(len(feats)), format_func=lambda i: names[i], key="ed_sel")
        kinds = list(kernel.templates())
        add = st.selectbox("Insert feature", ["(choose)"] + kinds)
        if add != "(choose)" and st.button("Insert"):
            t = kernel.templates()[add]
            t["id"] = f"{add}_{len(feats)}"
            feats.append(t)
            st.rerun()
        if st.button("Delete selected") and feats:
            feats.pop(int(k))
            st.rerun()
        txt = st.text_area("Selected feature (JSON)", json.dumps(feats[int(k)], indent=1), height=300)
        if st.button("Apply"):
            try:
                new = json.loads(txt)
                trial = {**doc, "features": feats[: int(k)] + [new] + feats[int(k) + 1 :]}
                chk = kernel.check(trial)
                if chk["ok"]:
                    feats[int(k)] = new
                    st.rerun()
                else:
                    st.error("; ".join(chk["errors"]))
            except json.JSONDecodeError as e:
                st.error(f"not valid JSON: {e}")
        name = st.text_input("Save as (projects/...)", "my_fighter.json")
        if st.button("Save"):
            (ROOT / "projects").mkdir(exist_ok=True)
            (ROOT / "projects" / name).write_text(json.dumps(doc, indent=1), encoding="utf-8")
            st.success(f"saved projects/{name}")
    with right:
        res = st.select_slider("Mesh resolution, m", [0.12, 0.09, 0.07, 0.05], value=0.07)
        try:
            st.plotly_chart(design_figure(doc, res), use_container_width=True)
            info = kernel.info(doc, h=res, nx=8)
            st.write(f"Volume {info['volume']:.2f} m³ · wetted area {info['wetted_area']:.1f} m² · frontal area {info['frontal_area']:.2f} m² · length {info['length']:.2f} m")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
        ax = st.radio("Section", ["x", "y", "z"], horizontal=True)
        at = st.slider("Section position, m", -10.0, 25.0, 8.0 if ax == "x" else 0.0, 0.1)
        seg = kernel.section(doc, ax, at, 0.02)
        fig = go.Figure()
        for s in seg:
            fig.add_trace(go.Scatter(x=[s[0], s[2]], y=[s[1], s[3]], mode="lines", line={"color": "#1a5fb4", "width": 2}, showlegend=False))
        fig.update_layout(height=300, yaxis={"scaleanchor": "x"}, margin={"t": 10, "b": 10})
        st.plotly_chart(fig, use_container_width=True)


def page_study() -> None:
    st.header("Fighter study: every family against every situation")
    p = FIG / "study.json"
    if not p.exists():
        st.info("No study yet. Run: python -m aerosynapse.cli fighter --profile standard")
        return
    s = json.loads(p.read_text(encoding="utf-8"))
    ref = s["reference"]["summary"]
    rows = [{"Design": "Reference (F-16-class)", "Objective": s["reference"]["J"], "Feasible": True, "Take-off mass, t": ref["W0"] / 1e3, "Cruise drag, kN": ref["metrics"]["cruise_drag_N"] / 1e3,
             "Dash drag, kN": ref["metrics"]["dash_drag_N"] / 1e3, "Sustained turn, deg/s": ref["metrics"]["turn_rate_sustained_dps"], "Climb Ps, m/s": ref["metrics"]["Ps_climb"], "Gust load, g": ref["metrics"]["gust_dn"]}]
    for fam, r in s["families"].items():
        x = r["summary"]
        rows.append({"Design": fam.replace("_", " "), "Objective": r["J"], "Feasible": r["feasible"], "Take-off mass, t": x["W0"] / 1e3, "Cruise drag, kN": x["metrics"]["cruise_drag_N"] / 1e3,
                     "Dash drag, kN": x["metrics"]["dash_drag_N"] / 1e3, "Sustained turn, deg/s": x["metrics"]["turn_rate_sustained_dps"], "Climb Ps, m/s": x["metrics"]["Ps_climb"], "Gust load, g": x["metrics"]["gust_dn"]})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
    fam = st.selectbox("Family", list(s["families"]))
    r = s["families"][fam]["summary"]
    c1, c2 = st.columns([3, 2])
    with c1:
        f = FIG / f"{fam}.design.json"
        if f.exists():
            st.plotly_chart(design_figure(json.loads(f.read_text()), 0.07, title=fam), use_container_width=True)
    with c2:
        st.subheader("Situations")
        st.dataframe(pd.DataFrame([{"State": v["name"] or k, "CL": v["CL"], "CD": v["CD"], "L/D": v["LD"], "Ps, m/s": v["Ps"]} for k, v in r["states"].items()]).round(3), hide_index=True)
        st.subheader("Requirements")
        st.dataframe(pd.DataFrame([{"Requirement": k.replace("_", " "), "Met": v <= 0, "Margin": -v} for k, v in r["constraints"].items()]).round(3), hide_index=True)
    st.subheader("Materials chosen per zone")
    st.dataframe(pd.DataFrame(r["materials"]).round(3), hide_index=True, use_container_width=True)


def page_cfd() -> None:
    st.header("3D flow simulation")
    st.caption("Compressible Navier-Stokes on a Cartesian grid, the design as an immersed boundary, WALE large-eddy turbulence and a wall model for friction (C++ solver aeroflow).")
    files = sorted(FIG.glob("*.design.json")) + [FIG / "baseline_design.json"]
    pick = st.selectbox("Design", [str(f.relative_to(ROOT)) for f in files if f.exists()])
    c = st.columns(5)
    mach = c[0].number_input("Mach", 0.2, 2.5, 0.85, 0.05)
    alpha = c[1].number_input("Incidence, deg", -10.0, 45.0, 2.0, 1.0)
    beta = c[2].number_input("Sideslip, deg", -20.0, 20.0, 0.0, 1.0)
    alt = c[3].number_input("Altitude, m", 0.0, 20000.0, 11000.0, 500.0)
    cpl = c[4].number_input("Cells per length", 20.0, 80.0, 36.0, 4.0)
    c2 = st.columns(3)
    gust = c2[0].number_input("Gust (w/V)", 0.0, 0.3, 0.0, 0.01)
    tu = c2[1].number_input("Inflow turbulence intensity", 0.0, 0.3, 0.0, 0.01)
    steps = c2[2].number_input("Steps", 200, 8000, 1500, 100)
    out = ROOT / "results" / "studio" / f"ui_cfd_M{mach:.2f}_a{alpha:.0f}"
    if st.button("Run"):
        des = json.loads((ROOT / pick).read_text())
        info = kernel.info(des, nx=8)
        ref = {"length": info["length"], "area": des.get("variables", {}).get("w_area", 0.6 * info["planform_area"]), "chord": 0.25 * info["length"], "span": info["bbox_hi"][1] - info["bbox_lo"][1],
               "moment_point": [info["bbox_lo"][0] + 0.55 * info["length"], 0, 0]}

        def work(job: jobs.Job) -> dict[str, Any]:
            return runner.run(des, runner.Flight(mach=mach, alpha=alpha, beta=beta, altitude=alt, gust_amplitude=gust, turbulence_intensity=tu), ref, out, runner.Numerics(cells_per_length=cpl, steps=int(steps)))

        st.session_state["cfd_job"] = jobs.submit("cfd", work)
    job = st.session_state.get("cfd_job")
    if job is not None and not job.done:
        st.info("running... (refresh to update)")
    if (out / "result.json").exists():
        res = json.loads((out / "result.json").read_text())
        st.json({k: res[k] for k in ("coefficients", "converged", "steps", "seconds", "cells", "separated_fraction", "T_wall_max_K") if k in res})
        s = runner.surface(out)
        fig = go.Figure(go.Scatter3d(x=s["x"], y=s["y"], z=s["z"], mode="markers", marker={"size": 2, "color": s["cp"], "colorscale": "RdBu_r", "colorbar": {"title": "Cp"}}))
        fig.update_layout(scene={"aspectmode": "data"}, height=480, margin={"t": 10})
        st.plotly_chart(fig, use_container_width=True)
        var = st.radio("Field on the symmetry plane", ["mach", "p", "vort", "tke"], horizontal=True)
        pl = runner.plane(out, "plane_symmetry")
        if var in pl:
            st.plotly_chart(go.Figure(go.Heatmap(x=pl["coord_u"], y=pl["coord_v"], z=pl[var], colorscale="Viridis")).update_layout(height=380, yaxis={"scaleanchor": "x"}), use_container_width=True)


def page_materials() -> None:
    st.header("Materials and tolerances")
    T = st.slider("Hottest skin temperature the airframe sees, K", 250.0, 900.0, 360.0, 5.0)
    zone_T = {z.key: T for z in mt.ZONES}
    ch = mt.select(zone_T)
    st.dataframe(pd.DataFrame([{"Zone": c.name, "Material": mt.LIBRARY[c.material].name, "Temperature, K": c.T_K, "Margin to service limit, K": c.margin_K, "Mass index vs 7075": c.index} for c in ch]).round(3),
                 hide_index=True, use_container_width=True)
    st.write(f"Structural mass factor against an all-aluminium airframe: **{mt.structure_factor(ch):.3f}**")
    st.subheader("Library")
    st.dataframe(pd.DataFrame([{"Material": m.name, "Density": m.rho, "E, GPa": m.E / 1e9, "Allowable, MPa": m.sigma / 1e6, "Service limit, K": m.T_max, "Strength at T": m.strength(T) / 1e6,
                                "Tolerance, mm": m.tol_mm, "Roughness Ra, um": m.rough_um, "Source": m.source, "Checked": m.verified} for m in mt.LIBRARY.values()]).round(2), hide_index=True, use_container_width=True)
    st.subheader("Tolerance study (Monte Carlo)")
    n = st.slider("Samples", 200, 5000, 1000, 100)
    t = mt.tolerance_study(ch, isa(11000.0), 0.85, 15.0, 150.0, n=n)
    st.json(t)


def page_thermal() -> None:
    st.header("Skin heating: parametric PINN from the thermal project")
    p = ROOT / "results" / "thermal_skin" / "skin.json"
    if not p.exists():
        st.info("Run: python -m aerosynapse.cli thermal")
        return
    s = json.loads(p.read_text())
    st.subheader("Verification against the exact series and the finite-difference solver")
    st.dataframe(pd.DataFrame([{"Case": k, **v} for k, v in s["verification"].items()]), hide_index=True)
    state = st.selectbox("Flight state", list(s["zones"]))
    z = s["zones"][state]
    st.dataframe(pd.DataFrame([{"Zone": k, "Material": v["material"], "Bi": v["Bi"], "Max T, K": v["T_max_K"], "Margin, K": v["temperature_margin_K"], "dT across, K": v["dT_max_K"],
                                "Thermal stress, MPa": v["thermal_stress_MPa"], "Share of allowable": v["stress_ratio"], "Model": ", ".join(v["source"])} for k, v in z.items()]).round(3), hide_index=True, use_container_width=True)
    pick = st.selectbox("Zone", list(z))
    v = z[pick]
    fig = go.Figure([go.Scatter(x=v["time_s"], y=v["T_outer_K"], name="outer face"), go.Scatter(x=v["time_s"], y=v["T_inner_K"], name="inner face")])
    st.plotly_chart(fig.update_layout(xaxis_title="time, s", yaxis_title="temperature, K", height=360), use_container_width=True)


def page_data_store() -> None:
    from aerosynapse.store.shards import store_root

    st.header("Data store and surrogates")
    root = store_root()
    st.write(f"Store: `{root}` (set AEROSYNAPSE_STORE to move it to another drive)")
    rows = []
    for m in sorted(root.rglob("manifest.json")):
        man = json.loads(m.read_text())
        rows.append({"Dataset": str(m.parent.relative_to(root)), "Kind": man.get("kind"), "Shards": len(man["shards"]), "Rows": sum(s.get("records", s.get("samples", 0)) for s in man["shards"]),
                     "MB": sum(s["bytes"] for s in man["shards"]) / 1e6})
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, use_container_width=True)
    rep = ROOT / "results" / "surrogate" / "report.json"
    if rep.exists():
        r = json.loads(rep.read_text())
        st.subheader("Surrogate (held-out shards)")
        st.dataframe(pd.DataFrame({"Output": r.get("outputs", list(range(len(r["val_r2"])))), "R² network": r["val_r2"], "R² nearest neighbour": r.get("nearest_neighbour_r2", [None] * len(r["val_r2"])),
                                   "Relative RMSE": r["val_rel_rmse"]}).round(3), hide_index=True)


TASKS = [
    # group, name, command (python arguments), what it optimises or simulates, result file
    ("Airfoil", "Airfoil shape optimisation", "-m aerosynapse.cli shapeopt --shape naca2412 --profile quick", "CST section, planform and twist over laminar, turbulent and extreme states", "results/shapeopt/naca2412/search.json"),
    ("Airfoil", "Airfoil check with XFOIL", "-m aerosynapse.cli xfoil --shape naca2412", "independent polar of the optimised section", "results/shapeopt/naca2412/xfoil.json"),
    ("Topology", "Topology optimisation of a 2D body", "-m aerosynapse.cli topology --shape square --profile quick", "material layout (density, Brinkman Navier-Stokes, adjoint)", "results/shapeopt/topology"),
    ("Aircraft", "Whole aircraft study", "-m aerosynapse.cli aircraft --profile quick", "tube-and-wing, blended wing body, flying wing, free-form", "results/aircraft/study.json"),
    ("Aircraft", "Fighter study", "-m aerosynapse.cli fighter --profile quick", "five topology families against every flight situation", "results/fighter/study.json"),
    ("3D flow", "3D flow verification", "-m aerosynapse.cli cfd-verify --profile quick", "exact and published cases", "results/cfd_verification/verification.json"),
    ("3D flow", "Fighter 3D flow comparison", "scripts/fighter_cfd.py --profile quick", "reference and best design in five states", "results/fighter/cfd/summary.json"),
    ("Thermal", "Skin heating with the PINN", "-m aerosynapse.cli thermal --profile quick", "panel temperatures and thermal stress per zone", "results/thermal_skin/skin.json"),
    ("Thermal", "Parametric 2D PINN (merged project)", "-m aerosynapse.thermal.experiments.p4_5_parametric --profile smoke", "one network over Re, Pr, Bi", "projects/thermal_pinn/results"),
    ("Thermal", "Inverse identification of alpha, k, h", "-m aerosynapse.thermal.experiments.p5_2_pinn_inverse --profile smoke", "material properties from sparse data", "projects/thermal_pinn/results"),
    ("Prediction", "Surrogate from the data store", "-m aerosynapse.cli surrogate --profile quick", "network trained from shards on disk", "results/surrogate/report.json"),
]
JOBS = ROOT / "results" / "studio" / "jobs"


def page_centre() -> None:
    import sys

    st.header("Optimisation and simulation centre")
    st.caption("Every optimisation and simulation in the project, in one place. Runs start in the background; their output is written to results/studio/jobs/. The same list is in AeroSynapse Studio's Optimise menu.")
    JOBS.mkdir(parents=True, exist_ok=True)
    rows = []
    for g, n, _cmd, what, res in TASKS:
        rp = ROOT / res
        rows.append({"Group": g, "Task": n, "Optimises or simulates": what, "Latest result": time_of(rp)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    names = [t[1] for t in TASKS]
    pick = st.selectbox("Task", names)
    g, n, cmd, what, res = TASKS[names.index(pick)]
    args = st.text_input("Arguments", cmd)
    log = JOBS / f"web_{names.index(pick)}.log"
    c1, c2 = st.columns(2)
    if c1.button("Run in the background"):
        with open(log, "w", encoding="utf-8") as fh:
            subprocess.Popen([sys.executable, *args.split()], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
        st.success(f"started; output in {log.relative_to(ROOT)}")
    refresh = c2.button("Refresh output")
    if log.exists() or refresh:
        st.code(log.read_text(encoding="utf-8", errors="ignore")[-6000:] if log.exists() else "(no output yet)")


def time_of(p: Path) -> str:
    import datetime

    if not p.exists():
        return "not run yet"
    return datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def page_workflow() -> None:
    import os
    import sys

    from aerosynapse.workflow import STAGES, Project
    from aerosynapse.workflow.project import PROJECTS

    st.header("Design workflow")
    st.caption("One project from a basic design to an exported, documented aircraft: design, wing and body optimisation, structure and design codes, "
               "PINN skin design, 3D flow with grid convergence, then STEP, glTF, STL, DXF, drawing, bill of materials and report. "
               "The same workflow is in AeroSynapse Studio (Workflow menu, F9).")
    names = sorted(d.name for d in PROJECTS.iterdir() if (d / "project.json").exists()) if PROJECTS.exists() else []
    c1, c2, c3 = st.columns([2, 2, 1])
    new = c1.text_input("New project name", "")
    fam = c2.selectbox("Start from", ["conventional", "canard_delta", "twin_tail_lerx", "tailless_blended", "freeform"])
    if c3.button("Create") and new:
        try:
            Project.create(new, family=fam)
            st.success(f"created projects/{new}")
            names.append(new)
        except Exception as e:
            st.error(str(e))
    if not names:
        st.info("No projects yet.")
        return
    pick = st.selectbox("Project", names, index=len(names) - 1)
    pr = Project(pick)
    rows = [{"#": i + 1, "Stage": t, "Status": pr.status(k), "Seconds": round(pr.meta["stages"][k].get("seconds", 0) or 0)} for i, (k, t) in enumerate(STAGES)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    prof = st.radio("Profile", ["quick", "standard"], horizontal=True)
    b1, b2, b3 = st.columns(3)
    log = pr.dir / "web_run.log"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    stage = b1.selectbox("Stage", [k for k, _ in STAGES], format_func=lambda k: dict(STAGES)[k])
    if b2.button("Run this stage"):
        with open(log, "w", encoding="utf-8") as fh:
            subprocess.Popen([sys.executable, "-m", "aerosynapse.cli", "workflow", "run", str(pr.dir), "--stage", stage, "--profile", prof], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
        st.success("started")
    if b3.button("Run all remaining"):
        with open(log, "w", encoding="utf-8") as fh:
            subprocess.Popen([sys.executable, "-m", "aerosynapse.cli", "workflow", "run", str(pr.dir), "--profile", prof], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
        st.success("started")
    wl = pr.dir / "workflow.log"
    if wl.exists():
        st.code(wl.read_text(encoding="utf-8", errors="ignore")[-5000:])
    for k, t in STAGES:
        d = pr.dir / f"{[x for x, _ in STAGES].index(k) + 1:02d}_{k}"
        pngs = sorted(d.glob("*.png")) if d.exists() else []
        if pngs:
            with st.expander(f"{t}: figures"):
                for f in pngs:
                    st.image(str(f))
    rep = pr.dir / "08_export" / "report.html"
    if rep.exists():
        st.download_button("Download the report (HTML)", rep.read_bytes(), file_name=f"{pick}_report.html")
        pkg = pr.dir / f"{pick}_package.zip"
        if pkg.exists():
            st.download_button("Download the project package (zip)", pkg.read_bytes(), file_name=pkg.name)


PAGES = {"Design workflow": page_workflow, "Optimisation centre": page_centre, "Fighter editor": page_editor, "Fighter study": page_study, "3D flow": page_cfd, "Materials": page_materials, "Skin heating (PINN)": page_thermal, "Data store": page_data_store}
_ = (D, P)
