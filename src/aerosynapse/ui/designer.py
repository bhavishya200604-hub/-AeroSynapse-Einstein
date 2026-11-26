"""Aircraft Designer: define a body, analyse it, optimise it, compare, and manage data.

Run: ``streamlit run src/aerosynapse/ui/designer.py``. Designs save to ``projects/`` as plain JSON.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aerosynapse.aircraft import analysis as an  # noqa: E402
from aerosynapse.aircraft import datahub as dh  # noqa: E402
from aerosynapse.aircraft import layers as Y  # noqa: E402
from aerosynapse.aircraft import optimise as op  # noqa: E402
from aerosynapse.aircraft import panel3d as p3  # noqa: E402
from aerosynapse.aircraft import vehicle as vh  # noqa: E402
from aerosynapse.aircraft.atmosphere import isa  # noqa: E402
from aerosynapse.safety import DISCLAIMER  # noqa: E402
from aerosynapse.shapeopt.cmaes import cmaes  # noqa: E402
from aerosynapse.shapeopt.geometry import write_stl  # noqa: E402
from aerosynapse.ui import jobs  # noqa: E402

PROJECTS = Path("projects")
LABELS = {
    "L": "Body length, m", "W": "Body half width, m", "H": "Body half height, m", "fn": "Nose length, fraction", "ft": "Tail length, fraction", "p": "Nose fullness", "tq": "Tail fullness",
    "tract": "Tractrix tail blend", "n1": "Section squareness (superformula)", "wi": "Hyperbolic warp", "f2": "Fourier 2-lobe", "f3": "Fourier 3-lobe",
    "upsweep": "Tail upsweep", "S": "Wing area, m²", "AR": "Aspect ratio", "taper": "Taper ratio", "sweep": "Sweep, degrees", "tc": "Thickness ratio", "twist": "Tip twist, degrees",
    "winglet": "Winglet height, fraction", "x_wing": "Wing position, fraction", "Vh": "Horizontal tail volume", "Vv": "Vertical tail volume",
}  # fmt: skip
GROUPS = {"Body length": ("L", "W", "H", "fn", "ft", "p", "tq", "tract", "upsweep"), "Cross section": ("n1", "wi", "f2", "f3"), "Wing": ("S", "AR", "taper", "sweep", "tc", "twist", "winglet", "x_wing"), "Tail": ("Vh", "Vv")}
NICE = {
    "cabin_floor": "Cabin floor area", "fuel_volume": "Fuel fits", "span": "Span limit", "approach_speed": "Approach speed", "static_margin_low": "Static margin, minimum", "static_margin_high": "Static margin, maximum",
    "short_period_damping": "Short-period damping", "directional": "Directional stability", "flutter": "Flutter margin", "divergence": "Divergence margin", "skin_temperature": "Skin temperature", "tail_closure": "Tail closure",
    "lift_limit": "Cruise lift limit", "tip_stall": "Wing stalls inboard first",
}  # fmt: skip


@st.cache_resource(show_spinner="Calibrating the baseline")
def default_mission(kind: str) -> an.Mission:
    """Missions start calibrated on their baseline, so the app agrees with the study results."""
    if kind == "supersonic":
        return an.calibrate(vh.build(vh.PRESETS["supersonic_jet"]), an.Mission(**an.SUPERSONIC))
    return an.calibrate(vh.build(vh.PRESETS["tube_and_wing"]), an.Mission(), 41000.0)


def mission() -> an.Mission:
    if "mission" not in st.session_state:
        st.session_state.mission = default_mission("airliner")
    return st.session_state.mission


def design() -> dict[str, float]:
    return st.session_state.setdefault("design", dict(vh.PRESETS["tube_and_wing"]))


@st.cache_data(show_spinner=False)
def evaluate_cached(d_json: str, m_json: str) -> dict[str, Any]:
    d = json.loads(d_json)
    m = _mission_from(json.loads(m_json))
    v = vh.build(d)
    r = an.evaluate(v, m)
    return {"summary": an.summary(r), "constraints": r.constraints}


def _mission_from(p: dict[str, Any]) -> an.Mission:
    return an.Mission(**{**default_mission("supersonic" if p["mach"] > 1 else "airliner").__dict__, **p})


def _mission_json() -> str:
    m = mission()
    return json.dumps({k: getattr(m, k) for k in ("name", "payload", "range_m", "mach", "altitude", "floor_required", "span_limit", "material", "k_wing", "k_body", "k_torsion")})


def mesh_figure(d: dict[str, float], title: str = "") -> go.Figure:
    v = vh.build(d)
    V, F = vh.surface(v)
    fig = go.Figure(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2], color="#7fb4e6", flatshading=True, lighting={"ambient": 0.55, "diffuse": 0.8, "specular": 0.15}))
    fig.update_layout(
        title=title,
        scene={"aspectmode": "data", "xaxis_title": "x, m", "yaxis_title": "y, m", "zaxis_title": "z, m", "camera": {"eye": {"x": -1.15, "y": -1.25, "z": 0.75}}},
        margin={"l": 0, "r": 0, "t": 30 if title else 0, "b": 0},
        height=460,
    )
    return fig


def status_table(cons: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{"Requirement": NICE.get(k, k), "Status": "Met" if x <= 1e-3 else "Not met", "Margin": f"{-x:+.2f}"} for k, x in cons.items()])


def page_design() -> None:
    st.header("Design")
    st.caption("Conventional, blended and free-form bodies are regions of one design space, so you can move between them with these controls.")
    c = st.columns([2, 1, 1])
    preset = c[0].selectbox("Start from", list(vh.PRESETS), format_func=lambda s: s.replace("_", " ").capitalize(), key="preset")
    if c[1].button("Load preset", use_container_width=True):
        st.session_state.design = dict(vh.PRESETS[preset])
        for k in vh.NAMES:
            st.session_state.pop(f"v_{k}", None)
        st.rerun()
    d = design()
    PROJECTS.mkdir(exist_ok=True)
    tabs = st.tabs(list(GROUPS))
    for t, (_, keys) in zip(tabs, GROUPS.items(), strict=False):
        with t:
            cols = st.columns(3)
            for i, k in enumerate(keys):
                lo, hi = next((a, b) for n, a, b in vh.VARS if n == k)
                d[k] = cols[i % 3].slider(LABELS[k], float(lo), float(hi), float(np.clip(d[k], lo, hi)), key=f"v_{k}")
    st.plotly_chart(mesh_figure(d), use_container_width=True)
    r = evaluate_cached(json.dumps(d, sort_keys=True), _mission_json())
    s = r["summary"]
    m = st.columns(5)
    m[0].metric("L/D", f"{s['L_over_D']:.1f}")
    m[1].metric("Mission fuel", f"{s['fuel_t']:.1f} t")
    m[2].metric("Take-off mass", f"{s['MTOW_t']:.1f} t")
    m[3].metric("Span", f"{s['span_m']:.1f} m")
    m[4].metric("Requirements met", f"{sum(x <= 1e-3 for x in r['constraints'].values())} of {len(r['constraints'])}")
    c = st.columns([2, 1])
    name = c[0].text_input("Project name", value="my_design")
    if c[1].button("Save project", use_container_width=True):
        (PROJECTS / f"{name}.json").write_text(json.dumps({"design": d}, indent=1), encoding="utf-8")
        st.success(f"Saved projects/{name}.json")
    saved = sorted(p.stem for p in PROJECTS.glob("*.json"))
    if saved:
        c = st.columns([2, 1])
        pick = c[0].selectbox("Open project", saved)
        if c[1].button("Open", use_container_width=True):
            st.session_state.design = json.loads((PROJECTS / f"{pick}.json").read_text(encoding="utf-8"))["design"]
            for k in vh.NAMES:
                st.session_state.pop(f"v_{k}", None)
            st.rerun()


def page_mission() -> None:
    st.header("Mission and environment")
    m = mission()
    c = st.columns(3)
    kind = c[0].selectbox("Mission type", ["Airliner", "Supersonic jet"], index=1 if m.mach > 1 else 0)
    base = default_mission("supersonic" if kind.startswith("Super") else "airliner")
    if (m.mach > 1) != (base.mach > 1):
        st.session_state.mission = m = base
    mach = c[1].number_input("Cruise Mach", 0.3, 3.5, float(m.mach), 0.01)
    alt = c[2].number_input("Cruise altitude, m", 1000.0, 26000.0, float(m.altitude), 500.0)
    c = st.columns(3)
    pay = c[0].number_input("Payload, kg", 500.0, 80000.0, float(m.payload), 500.0)
    rng = c[1].number_input("Range, km", 500.0, 16000.0, m.range_m / 1000, 100.0)
    floor = c[2].number_input("Cabin floor required, m²", 5.0, 500.0, float(m.floor_required), 1.0)
    c = st.columns(3)
    span = c[0].number_input("Span limit, m", 10.0, 90.0, float(m.span_limit), 1.0)
    mat = c[1].selectbox("Structure", list(Y.MATERIALS), index=list(Y.MATERIALS).index(m.material), format_func=lambda k: Y.MATERIALS[k].name)
    vap = c[2].number_input("Approach speed limit, m/s", 40.0, 110.0, float(m.v_approach), 1.0)
    st.session_state.mission = an.Mission(**{**m.__dict__, "mach": mach, "altitude": alt, "payload": pay, "range_m": rng * 1000, "floor_required": floor, "span_limit": span, "material": mat, "v_approach": vap})
    st.subheader("Atmosphere at cruise")
    dT = st.slider("Temperature offset from standard day, K", -40, 40, 0)
    a = isa(alt, dT)
    t = st.columns(5)
    t[0].metric("Air temperature", f"{a.T - 273.15:.1f} °C")
    t[1].metric("Pressure", f"{a.p / 1000:.1f} kPa")
    t[2].metric("Density", f"{a.rho:.3f} kg/m³")
    t[3].metric("Speed of sound", f"{a.a:.0f} m/s")
    t[4].metric("True airspeed", f"{mach * a.a:.0f} m/s")


def page_analysis() -> None:
    st.header("Analysis")
    d, m = design(), mission()
    res = evaluate_cached(json.dumps(d, sort_keys=True), _mission_json())
    s = res["summary"]
    t = st.tabs(["Drag and thrust", "Boundary and thermal layers", "Stability and instability", "Surface pressure (3D panels)", "Requirements"])
    with t[0]:
        parts = {"Friction": s["friction"], "Form": s["form"], "Induced": s["induced"], "Wave": s["wave"], "Other": s["misc"]}
        fig = go.Figure(go.Bar(x=list(parts.values()), y=list(parts), orientation="h", text=[f"{x:.1f}" for x in parts.values()], textposition="auto"))
        fig.update_layout(xaxis_title="Drag, counts (1 count = 0.0001)", height=280, margin={"t": 10, "b": 10})
        st.plotly_chart(fig, use_container_width=True)
        c = st.columns(4)
        c[0].metric("Total drag, counts", f"{s['CD_counts']:.0f}")
        c[1].metric("Thrust required", f"{s['thrust_kN']:.1f} kN")
        c[2].metric("Fuel use, mg/(N s)", f"{s['tsfc_mg_Ns']:.1f}")
        c[3].metric("Turbulent wetted area", f"{s['turbulent_share'] * 100:.0f} %")
    with t[1]:
        c = st.columns(4)
        c[0].metric("Boundary layer at tail", f"{s['delta_aft_m'] * 100:.0f} cm")
        c[1].metric("Thermal layer at tail", f"{s['delta_T_aft_m'] * 100:.0f} cm")
        c[2].metric("Nose skin", f"{s['T_nose_C']:.0f} °C")
        c[3].metric("Mid-body skin", f"{s['T_skin_C']:.0f} °C")
        c = st.columns(4)
        c[0].metric("Total air temperature", f"{s['T_total_C']:.0f} °C")
        c[1].metric("Nose heat flux", f"{s['q_nose_kW_m2']:.1f} kW/m²")
        c[2].metric("Body length change", f"{s['expansion_mm']:.0f} mm")
        c[3].metric("Stiffness remaining", f"{s['E_ratio'] * 100:.0f} %")
        st.subheader("How temperature changes with speed")
        rows = []
        v = vh.build(d)
        for M, h in ((0.8, 11000), (1.2, 13000), (1.6, 16000), (2.0, 17000), (2.5, 19000), (3.0, 22000), (3.5, 25000)):
            th = Y.thermal_state(isa(h), M, d["L"], v.S_wet_body + v.S_wet_wing, max(0.02 * d["H"], 0.02), Y.MATERIALS[m.material])
            rows.append({"Mach": M, "Total air °C": th.T_total - 273.15, "Nose °C": th.T_nose - 273.15, "Skin °C": th.T_skin - 273.15})
        df = pd.DataFrame(rows)
        fig = go.Figure([go.Scatter(x=df["Mach"], y=df[c_], name=c_, mode="lines+markers") for c_ in df.columns[1:]])
        fig.add_hline(y=Y.MATERIALS[m.material].T_max - 273.15, line_dash="dash", annotation_text=f"{Y.MATERIALS[m.material].name} limit")
        fig.update_layout(xaxis_title="Mach number", yaxis_title="°C", height=340, margin={"t": 10})
        st.plotly_chart(fig, use_container_width=True)
    with t[2]:
        c = st.columns(4)
        c[0].metric("Static margin", f"{s['static_margin'] * 100:.0f} % chord")
        c[1].metric("Short-period damping", f"{s['short_period_zeta']:.2f}")
        c[2].metric("Directional stability", f"{s['cn_beta']:.3f} /rad")
        c[3].metric("Gust load factor", f"{s['n_gust']:.2f}")
        c = st.columns(3)
        c[0].metric("Tail closure angle", f"{s['closure_deg']:.1f}°", help="Above about 14 degrees the aft body tends to separate")
        c[1].metric("Flutter speed", f"{s['v_flutter']:.0f} m/s")
        c[2].metric("Dive speed", f"{s['v_dive']:.0f} m/s")
    with t[3]:
        st.caption("Inviscid source panels on the real body shape, then a boundary layer along three lines of the body. Validated on the exact sphere and ellipsoid solutions.")
        if st.button("Run panel analysis"):
            with st.spinner("Solving"):
                v = vh.build(d)
                b = p3.from_grid(p3.body_grid(v, 34))
                f = p3.solve(b)
                nj = b.shape[1]
                fig = go.Figure(go.Scatter3d(x=b.c[:, 0], y=b.c[:, 1], z=b.c[:, 2], mode="markers", marker={"size": 3, "color": f.cp, "colorscale": "RdBu_r", "colorbar": {"title": "Cp"}}))
                fig.update_layout(scene={"aspectmode": "data"}, height=420, margin={"l": 0, "r": 0, "t": 0, "b": 0})
                st.plotly_chart(fig, use_container_width=True)
                Re = 1.2e7 * d["L"]
                st.dataframe(pd.DataFrame([{"Line": n, **{k: round(x, 4) for k, x in p3.meridian_layer(b, f, Re, j=j).items()}} for n, j in (("Top", nj // 4), ("Side", 0), ("Bottom", 3 * nj // 4))]), hide_index=True)
    with t[4]:
        st.dataframe(status_table(res["constraints"]), hide_index=True, use_container_width=True)


def page_optimise() -> None:
    st.header("Optimise")
    st.caption("The search changes every free variable together, scored on mission fuel and robust drag in off-design conditions. The current design is the baseline.")
    c = st.columns(3)
    fam = c[0].selectbox("Shape freedom", list(op.FAMILIES), index=3, format_func=lambda s: s.replace("_", " ").capitalize())
    budget = c[1].slider("Designs to try", 100, 6000, 800, 100)
    seed = c[2].number_input("Random seed", 0, 999, 0)
    if st.button("Start optimisation", type="primary"):
        base = dict(design())
        m = mission()

        def work(job: jobs.Job) -> dict[str, Any]:
            stdy = op.Study(m, base)
            sp = op.space(fam)
            seen = [0]

            def f(Z: np.ndarray) -> np.ndarray:
                out = np.empty(len(Z))
                for i, z in enumerate(Z):
                    out[i] = stdy.score(sp.decode(z))[0]
                seen[0] += len(Z)
                job.report(f"{min(seen[0], budget)} of {budget} designs", seen[0] / budget)
                if job.cancelled():
                    raise RuntimeError("cancelled")
                return out

            r = cmaes(f, np.clip(sp.encode({**sp.base, **{k: base[k] for k in sp.names}}), 0, 1), sigma=0.15, seed=int(seed), budget=budget, pop=14, restarts=1)
            d = sp.decode(r.x)
            J, res, rr = stdy.score(d)
            return {"design": d, "J": J, "summary": an.summary(res), "base": an.summary(stdy.r0), "constraints": res.constraints, "off": dict(zip([q.name for q in stdy.fl], rr.tolist(), strict=False))}

        st.session_state.opt_job = jobs.submit("optimise", work)
    job = st.session_state.get("opt_job")
    if job is not None:
        st.progress(job.progress, text=job.message)
        if not job.done:
            if st.button("Stop"):
                job.cancel_event.set()
            time.sleep(1.5)  # the page keeps itself up to date while the search runs
            st.rerun()
        if job.error:
            st.warning("The run was stopped." if "cancelled" in job.error else f"The run failed: {job.error.splitlines()[-1]}")
            return
        r = job.result()
        st.session_state.last_result = r
        a, b = r["base"], r["summary"]
        st.subheader("Baseline against optimised")
        keys = [("L_over_D", "L/D"), ("fuel_t", "Mission fuel, t"), ("MTOW_t", "Take-off mass, t"), ("CD_counts", "Drag, counts"), ("friction", "Friction, counts"), ("induced", "Induced, counts"), ("wave", "Wave, counts"), ("turbulent_share", "Turbulent wetted share"), ("closure_deg", "Tail closure, degrees"), ("static_margin", "Static margin"), ("T_nose_C", "Nose skin, °C")]
        st.dataframe(pd.DataFrame([{"Quantity": n, "Baseline": round(a[k], 3), "Optimised": round(b[k], 3), "Change %": round(100 * (b[k] / a[k] - 1), 1) if abs(a[k]) > 1e-9 else None} for k, n in keys]), hide_index=True, use_container_width=True)
        st.plotly_chart(mesh_figure(r["design"], "Optimised shape"), use_container_width=True)
        st.dataframe(status_table(r["constraints"]), hide_index=True, use_container_width=True)
        c = st.columns(2)
        if c[0].button("Use this design", use_container_width=True):
            st.session_state.design = r["design"]
            for k in vh.NAMES:
                st.session_state.pop(f"v_{k}", None)
            st.rerun()
        out = Path("results/aircraft/app_export.stl")
        out.parent.mkdir(parents=True, exist_ok=True)
        V, F = vh.surface(vh.build(r["design"]))
        write_stl(out, V, F, "design")
        c[1].download_button("Download STL", out.read_bytes(), "design.stl", use_container_width=True)


def page_shapes() -> None:
    from aerosynapse.aircraft import curves as C

    st.header("Shape library")
    st.caption("Mathematical shapes the bodies are built from. Geometry alone does not reduce drag: these give the optimiser more shapes to try, and each run shows whether they helped.")
    t = st.tabs(["Cross sections", "Curved surfaces"])
    with t[0]:
        c = st.columns(4)
        n1 = c[0].slider("Superformula exponent", 1.0, 8.0, 2.0)
        wi = c[1].slider("Hyperbolic warp", -0.6, 0.6, 0.0)
        f2 = c[2].slider("Fourier 2-lobe", -0.2, 0.2, 0.0)
        f3 = c[3].slider("Fourier 3-lobe", -0.15, 0.15, 0.0)
        y, z = vh.section({"n1": n1, "wi": wi, "f2": f2, "f3": f3}, 200)
        fig = go.Figure(go.Scatter(x=np.r_[y, y[0]], y=np.r_[z, z[0]], fill="toself"))
        fig.update_layout(yaxis={"scaleanchor": "x"}, height=380, margin={"t": 10})
        st.plotly_chart(fig, use_container_width=True)
    with t[1]:
        kind = st.selectbox("Surface", ["Pseudosphere (hyperbolic, K < 0)", "Sphere (elliptic, K > 0)", "Catenoid (minimal surface)"])
        ph = np.linspace(0, 2 * np.pi, 60)
        if kind.startswith("Pseudo"):
            X, Yy, Z = C.pseudosphere(np.linspace(0.2, 3, 60), ph)
        elif kind.startswith("Sphere"):
            X, Yy, Z = C.spherical_cap(np.linspace(0.05, 3.09, 60), ph)
        else:
            X, Yy, Z = C.catenoid(np.linspace(-1.2, 1.2, 60), ph)
        K = C.gaussian_curvature(X, Yy, Z)
        fig = go.Figure(go.Surface(x=X, y=Yy, z=Z, surfacecolor=K, colorscale="RdBu", colorbar={"title": "Gaussian curvature"}))
        fig.update_layout(scene={"aspectmode": "data"}, height=460, margin={"l": 0, "r": 0, "t": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)


def page_data() -> None:
    st.header("Data")
    st.caption("Upload measured data here when you have it. Empty slots stay open for a later phase, and the models carry on with their built-in estimates.")
    st.dataframe(pd.DataFrame(dh.status()), hide_index=True, use_container_width=True)
    keys = {s.title: s for s in dh.SLOTS}
    pick = st.selectbox("Slot", list(keys))
    s = keys[pick]
    st.write(s.purpose)
    st.code(",".join(s.columns), language="text")
    up = st.file_uploader("CSV file with these columns", type=["csv"])
    if up is not None and st.button("Check and store"):
        try:
            n = dh.store(s.key, up.getvalue().decode("utf-8"))
            st.success(f"Stored {n} rows.")
        except ValueError as e:
            st.error(str(e))
    st.subheader("Built-in public reference values")
    ref = dh.reference()
    st.dataframe(pd.DataFrame(ref["entries"]).drop(columns=["url"]), hide_index=True, use_container_width=True)
    st.write("Still missing: " + "; ".join(ref["missing"]) + ".")


def page_results() -> None:
    st.header("Study results")
    p = Path("results/aircraft/study.json")
    if not p.exists():
        st.info("No study yet. Run: python scripts/run_aircraft.py --profile standard")
        return
    s = json.loads(p.read_text(encoding="utf-8"))
    for name, m in s["missions"].items():
        st.subheader(name.capitalize())
        rows = [{"Design": "Baseline", "L/D": m["baseline"]["L_over_D"], "Fuel, t": m["baseline"]["fuel_t"], "Take-off mass, t": m["baseline"]["MTOW_t"], "Span, m": m["baseline"]["span_m"], "Requirements not met": 0}]
        for k, f in m["families"].items():
            x = f["summary"]
            rows.append({"Design": k.replace("_", " ").capitalize(), "L/D": x["L_over_D"], "Fuel, t": x["fuel_t"], "Take-off mass, t": x["MTOW_t"], "Span, m": x["span_m"], "Requirements not met": sum(q > 1e-3 for q in f["constraints"].values())})
        st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, use_container_width=True)
        pick = st.selectbox("View shape", list(m["families"]), key=f"view_{name}")
        st.plotly_chart(mesh_figure(m["families"][pick]["design"]), use_container_width=True)


PAGES = {"Design": page_design, "Mission": page_mission, "Analysis": page_analysis, "Optimise": page_optimise, "Study results": page_results, "Shape library": page_shapes, "Data": page_data}
try:  # fighter workspaces need the C++ kernel (built on first use)
    from aerosynapse.ui.fighter_pages import PAGES as FIGHTER_PAGES  # noqa: E402

    PAGES = {**FIGHTER_PAGES, **PAGES}
except Exception as _e:  # noqa: BLE001
    FIGHTER_PAGES = {}


def main() -> None:
    st.set_page_config(page_title="AeroSynapse Designer", page_icon=str(Path(__file__).resolve().parents[3] / "docs" / "assets" / "aerosynapse_logo.png"), layout="wide")
    logo = Path(__file__).resolve().parents[3] / "docs" / "assets" / "aerosynapse_logo.png"
    if logo.exists():
        st.sidebar.image(str(logo), width=96)
    st.sidebar.title("AeroSynapse Designer")
    choice = st.sidebar.radio("Workspace", list(PAGES), key="page")
    st.sidebar.caption(DISCLAIMER)
    PAGES[choice]()


if __name__ == "__main__":
    main()
