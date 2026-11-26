"""Phase 8: interactive interface to the project — verified solvers, saved PINN ensembles, their comparison, inverse
identification, grid convergence, training logs, uncertainty / validity and export.

    streamlit run src/app/main.py          (or: python tasks.py app / make app)

Deep links: ?case=B_slab&section=Comparison&Bi=2 opens a case on a page with parameter values. Nothing is trained
here; every computation is in src/app/backend.py (shared with the CLI, src/app/cli.py, and the tests).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
if str(ROOT) not in sys.path:  # `streamlit run` puts only the script's folder on the path
    sys.path.insert(0, str(ROOT))

from aerosynapse.thermal.app import backend as B  # noqa: E402
from aerosynapse.thermal.viz import style as S  # noqa: E402

SECTIONS = [
    "Overview",
    "Solver",
    "PINN",
    "Comparison",
    "Inverse",
    "Convergence",
    "Training monitor",
    "Uncertainty & validity",
    "Export",
]
SLUG = {s: s.split()[0].lower() for s in SECTIONS}  # deep-link names: overview, solver, ..., uncertainty, export
COL = {"ref": S.REF, "solver": S.SOLVER, "pinn": S.PINN, "data": S.DATA, "muted": S.MUTED}

st.set_page_config(page_title="PINNs for heat transfer", page_icon="🔥", layout="wide")


# ============================================================================ cached computations
@st.cache_data(show_spinner="Running the verified solver ...", max_entries=32)
def run_solver(key: str, params: dict, opts: dict) -> dict:
    return B.solve(key, params, opts)


@st.cache_data(show_spinner="Evaluating the saved networks ...", max_entries=32)
def run_pinn(key: str, params: dict, opts: dict) -> dict:
    return B.predict(key, params, opts)


@st.cache_data(show_spinner="Comparing ...", max_entries=16)
def run_compare(key: str, params: dict, opts: dict) -> dict:
    return B.compare(key, params, opts)


@st.cache_data(show_spinner="Refining the grid ...", max_entries=16)
def run_convergence(key: str, params: dict, Ns: tuple, schemes: tuple, opts: dict) -> dict:
    return B.convergence(key, params, list(Ns), list(schemes), opts)


@st.cache_data(show_spinner="Fitting ...", max_entries=16)
def run_inverse(q_vol, noise, n_sensors, unknown, seed, data_file) -> dict:
    return B.inverse(q_vol, noise, n_sensors, tuple(unknown), seed=seed, data_file=data_file)


@st.cache_data(show_spinner="Sweeping the parameter ...", max_entries=8)
def run_sweep(key: str, lo: float, hi: float, n: int, use: tuple) -> dict:
    return B.sweep_1d(key, np.geomspace(lo, hi, n), list(use) or None)


# ============================================================================ helpers
def fig_layout(fig: go.Figure, x: str, y: str, logx: bool = False, logy: bool = False, height: int = 420):
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title=x,
        yaxis_title=y,
        legend={"orientation": "h", "y": -0.2},
        template="plotly_white",
        font={"family": "Georgia, serif"},
    )
    if logx:
        fig.update_xaxes(type="log")
    if logy:
        fig.update_yaxes(type="log")
    return fig


def show_validity(msgs: list[dict]) -> bool:
    """Validity guard banner; returns True if a 'stop' message was shown."""
    stop = False
    for m in msgs:
        if m["level"] == "stop":
            st.error(m["text"], icon="⛔")
            stop = True
        elif m["level"] == "warn":
            st.warning(m["text"], icon="⚠️")
        else:
            st.caption("✓ " + m["text"])
    return stop


def mpl_scale(name: str, n: int = 11) -> list:
    """A matplotlib colormap as a Plotly colour scale, so the app uses the same maps as the project's figures
    (inferno for temperature, RdBu_r centred at zero for signed fields, viridis for magnitudes)."""
    import matplotlib

    cm = matplotlib.colormaps[name]
    # rgb() strings: the browser's Plotly drew inferno's first stop '#000004' as its fallback colour (lavender)
    rgb = [tuple(int(round(255 * c)) for c in cm(i / (n - 1))[:3]) for i in range(n)]
    return [[i / (n - 1), f"rgb({r},{g},{b})"] for i, (r, g, b) in enumerate(rgb)]


def heatmap(x, y, z, title: str, cmap: str = "inferno", zmid=None, height: int | None = None) -> go.Figure:
    """Field on a cell-centred grid, drawn at its true aspect ratio and fitted to its domain."""
    x, y = np.asarray(x), np.asarray(y)
    fig = go.Figure(go.Heatmap(x=x, y=y, z=z, colorscale=mpl_scale(cmap), zmid=zmid, colorbar={"title": title}))
    w, h = float(x.max() - x.min()), float(y.max() - y.min())
    fig.update_xaxes(range=[x.min(), x.max()], constrain="domain")
    fig.update_yaxes(range=[y.min(), y.max()], scaleanchor="x", scaleratio=1, constrain="domain")
    return fig_layout(fig, "x", "y", height=height or int(min(520, max(220, 900 * h / max(w, 1e-9) + 90))))


def run_on_request(label: str, key: str, note: str) -> None:
    """A run button whose 'requested' state lasts for the session (per case and page): a plain button is pressed for one
    rerun only, so any later widget change would hide the results again. Stops the page until requested."""
    flag = f"requested_{key}_{st.session_state.get('case')}"
    if st.button(label, type="primary", key=key):
        st.session_state[flag] = True
    if not st.session_state.get(flag):
        st.info(note)
        st.stop()


def section_index(value: str) -> int:
    """Section from a deep link: its name or its slug (first word, lower case), e.g. ?section=uncertainty."""
    v = (value or "").strip().lower()
    for i, s in enumerate(SECTIONS):
        if v in (s.lower(), SLUG[s]):
            return i
    return 0


def remember(name: str, result) -> None:
    st.session_state.setdefault("results", {})[name] = result


# ============================================================================ sidebar: case, parameters, conditions
qp = st.query_params
keys = list(B.CASES)
case_key = st.sidebar.selectbox(
    "Case",
    keys,
    index=keys.index(qp.get("case", "B_slab")) if qp.get("case", "B_slab") in keys else 1,
    format_func=lambda k: B.CASES[k].title,
    key="case",
)
C = B.case(case_key)
section = st.sidebar.radio(
    "Section",
    SECTIONS,
    index=section_index(qp.get("section", "Overview")),
    key="section",
)
st.query_params.update({"case": case_key, "section": SLUG[section]})

st.sidebar.markdown("**Parameters**")
params: dict[str, float] = {}
for p in C.params:
    params[p.name] = float(
        st.sidebar.number_input(
            p.name + (f"  (PINN range {p.train[0]:g} - {p.train[1]:g})" if p.train and C.pinn else ""),
            min_value=float(p.lo),
            max_value=float(p.hi),
            value=min(max(float(qp.get(p.name, p.default)), float(p.lo)), float(p.hi)),
            format="%.4g",
            key=f"p_{case_key}_{p.name}",
            help=p.help,
        )
    )

opts: dict = {}
with st.sidebar.expander("Conditions and grid", expanded=False):
    if case_key == "A_steady":
        opts["theta_left"] = st.number_input("θ at x = 0", value=0.0, key="a_tl")
        opts["theta_right"] = st.number_input("θ at x = 1", value=1.0, key="a_tr")
        opts["source"] = st.number_input("uniform source q", value=0.0, key="a_q")
        opts["N"] = int(st.number_input("cells N", 4, 4000, 40, key="a_N"))
        opts["scheme"] = st.selectbox("convection scheme", ["central", "upwind1", "upwind2", "quick"], key="a_s")
    elif case_key == "B_slab":
        opts["Q"] = st.number_input("heat source Q", value=1.0, key="b_Q")
        opts["m"] = st.number_input("fin parameter m", value=1.0, min_value=0.0, key="b_m")
        opts["theta0"] = st.number_input("initial temperature θ0", value=1.0, key="b_t0")
        opts["T"] = st.number_input("end time T (Fourier)", value=0.5, min_value=0.01, key="b_T")
        opts["N"] = int(st.number_input("cells N", 4, 2000, 40, key="b_N"))
    elif case_key == "D_graetz":
        opts["cells_per_unit"] = int(st.selectbox("solver cells per unit length", [8, 16, 32], index=1, key="d_c"))
    else:
        opts["N"] = int(st.selectbox("grid N x N", [16, 32, 48, 64, 96], index=1, key="c_N"))
    if C.pinn:
        names = B.member_names(case_key)
        opts["members"] = st.multiselect("networks (checkpoints) in the ensemble", names, default=names, key="mem")
        if not opts["members"]:
            st.warning("select at least one network")
            opts["members"] = names
    if C.pinn_note:
        st.caption(C.pinn_note)

st.sidebar.caption("Every result is computed by the project's code; nothing is trained here.")

# ============================================================================ header
st.title("Physics-informed neural networks for heat transfer")
st.caption(
    f"**{C.title}** · section **{section}** · solvers verified in Phase 1 · PINNs from saved checkpoints "
    "(Steps 4.7, 6.2) · inverse from Phase 5"
)
msgs = B.validity(case_key, params, opts)

# ============================================================================ sections
if section == "Overview":
    st.subheader(C.title)
    st.latex(C.equation)
    ok, why = B.pinn_available(case_key)
    c1, c2, c3 = st.columns(3)
    c1.metric("dimension", f"{C.dim}D" + (" + time" if C.transient else ""))
    c2.metric("PINN", "available" if ok else "none", help=C.pinn or C.pinn_note)
    c3.metric("parameters", ", ".join(p.name for p in C.params))
    st.markdown(f"**PINN source:** {C.pinn or '—'}  \n**Note:** {C.pinn_note or '—'}  \n**Status:** {why}")
    st.markdown("**Validity of the current request**")
    show_validity(msgs)
    st.markdown(
        "**Sections.** *Solver*: the verified numerical solution. *PINN*: the saved networks' prediction with the "
        "ensemble spread. *Comparison*: both, their difference and query times. *Inverse*: recover α, k, h from sensor "
        "data. *Convergence*: grid refinement and observed order. *Training monitor*: the saved loss histories. "
        "*Uncertainty & validity*: error and spread across the parameter range. *Export*: download everything."
    )

elif section == "Solver":
    show_validity([m for m in msgs if "PINN" not in m["text"]])
    heavy = case_key in ("D_graetz", "cavity")
    if heavy:
        run_on_request(
            "Run solver", "run_solver", "2D cases run on request (seconds to a minute). Press **Run solver**."
        )
    t0 = time.perf_counter()
    r = run_solver(case_key, params, opts)
    remember("solver", r)
    c1, c2, c3 = st.columns(3)
    c1.metric("solver wall time", f"{r['wall_s']:.3g} s")
    if case_key == "A_steady":
        if "err_inf" in r:
            c2.metric("max error vs exact", f"{r['err_inf']:.3g}")
        fig = go.Figure()
        if "exact" in r:
            fig.add_scatter(x=r["x_fine"], y=r["exact"], name="exact", line={"color": COL["ref"], "width": 2})
        fig.add_scatter(
            x=r["x"],
            y=r["theta"],
            name=f"{r['scheme']}, N = {r['N']}",
            mode="lines+markers",
            line={"color": COL["solver"]},
            marker={"size": 5},
        )
        st.plotly_chart(fig_layout(fig, "x", "θ"), key="sol_a", theme=None)
    elif case_key == "B_slab":
        c2.metric("max error vs eigen-series", f"{r['err_inf']:.3g}")
        c3.metric("time step", f"{r['dt']:.3g}")
        fig = go.Figure()
        for i, tt in enumerate(r["t"]):
            fig.add_scatter(
                x=r["x"],
                y=r["exact"][i],
                name=f"exact t = {tt:.3g}",
                line={"color": COL["ref"], "width": 1},
                showlegend=i == 0,
                legendgroup="ex",
            )
            fig.add_scatter(
                x=r["x"],
                y=r["theta"][i],
                mode="markers",
                marker={"size": 4, "color": COL["solver"]},
                name="Crank-Nicolson",
                showlegend=i == 0,
                legendgroup="cn",
            )
        st.plotly_chart(fig_layout(fig, "x (centre → cooled surface)", "θ"), key="sol_b", theme=None)
    elif case_key == "D_graetz":
        c2.metric("mean Nusselt number (bottom)", f"{r['Nu']:.4g}")
        c3.metric("Newton iterations", r["newton_iterations"])
        st.plotly_chart(heatmap(r["x"], r["y"], r["theta"], "θ"), key="sol_d", theme=None)
        st.plotly_chart(heatmap(r["x"], r["y"], r["u"], "u", "Blues"), key="sol_du", theme=None)
        st.caption(f"{r['kind']}; max |div u| = {r['max_div']:.2g}")
    else:
        c2.metric("ψ_min", f"{r['psi_min']:.5f}")
        c3.metric("Nusselt number at the lid", f"{r['Nu_lid']:.4g}")
        a, b = st.columns(2)
        with a:
            st.plotly_chart(heatmap(r["x"], r["y"], r["theta"], "θ"), key="sol_ct", theme=None)
        with b:
            st.plotly_chart(
                heatmap(
                    np.linspace(0, 1, r["psi"].shape[1]),
                    np.linspace(0, 1, r["psi"].shape[0]),
                    r["psi"],
                    "ψ",
                    "RdBu_r",
                    zmid=0.0,
                ),
                key="sol_cp",
                theme=None,
            )
        st.caption(
            f"steady Newton {'converged' if r['converged'] else 'NOT converged'} (residual {r['newton_residual']:.2g})"
            + (f"; continuation in Re through {r['continuation']}" if r["continuation"] else "")
        )
    st.caption(f"page computed in {time.perf_counter() - t0:.2f} s (cached results return immediately)")

elif section in ("PINN", "Comparison", "Uncertainty & validity") and not C.pinn:
    st.info(f"No PINN for this case: {C.pinn_note}. Use the **Solver** section.")

elif section == "PINN":
    stop = show_validity(msgs)
    ok, why = B.pinn_available(case_key)
    if not ok:
        st.error(f"saved networks not found: {why}")
        st.stop()
    if case_key == "D_graetz":
        run_on_request(
            "Evaluate networks",
            "run_pinn",
            "Loading the channel networks takes about 30 s the first time. Press **Evaluate networks**.",
        )
    r = run_pinn(case_key, params, opts)
    remember("pinn", r)
    c1, c2, c3 = st.columns(3)
    c1.metric("networks", r["n_members"])
    c2.metric("query time (all networks)", f"{r['wall_s'] * 1e3:.3g} ms")
    c3.metric("mean ensemble spread", f"{float(np.mean(r['sd'])):.3g}")
    if stop:
        st.error("This is an EXTRAPOLATION: the curves below are not a valid prediction.", icon="⛔")
    if case_key == "A_steady":
        fig = go.Figure()
        for u in r["members"]:
            fig.add_scatter(x=r["x"], y=u, line={"color": COL["pinn"], "width": 0.6}, opacity=0.35, showlegend=False)
        fig.add_scatter(x=r["x"], y=r["mean"], name="ensemble mean", line={"color": COL["pinn"], "width": 2.5})
        fig.add_scatter(
            x=np.r_[r["x"], r["x"][::-1]],
            y=np.r_[r["mean"] + 2 * r["sd"], (r["mean"] - 2 * r["sd"])[::-1]],
            fill="toself",
            fillcolor="rgba(213,94,0,0.15)",
            line={"width": 0},
            name="± 2 sd",
        )
        st.plotly_chart(fig_layout(fig, "x", "θ"), key="pinn_a", theme=None)
    elif case_key == "B_slab":
        fig = go.Figure()
        for i, tt in enumerate(r["t"]):
            fig.add_scatter(
                x=r["x"], y=r["mean"][i], name=f"t = {tt:.3g}", line={"color": COL["pinn"], "width": 1 + i * 0.4}
            )
            fig.add_scatter(
                x=np.r_[r["x"], r["x"][::-1]],
                y=np.r_[r["mean"][i] + 2 * r["sd"][i], (r["mean"][i] - 2 * r["sd"][i])[::-1]],
                fill="toself",
                fillcolor="rgba(213,94,0,0.12)",
                line={"width": 0},
                showlegend=False,
            )
        st.plotly_chart(fig_layout(fig, "x", "θ"), key="pinn_b", theme=None)
    else:
        st.metric("mean Nusselt number (bottom wall)", f"{r['Nu']:.4g} ± {r['Nu_sd']:.2g}")
        st.plotly_chart(heatmap(r["x"], r["y"], r["mean"], "θ (mean)"), key="pinn_d", theme=None)
        st.plotly_chart(heatmap(r["x"], r["y"], r["sd"], "ensemble sd", "viridis"), key="pinn_dsd", theme=None)

elif section == "Comparison":
    stop = show_validity(msgs)
    if case_key == "D_graetz":
        run_on_request(
            "Run solver and networks",
            "run_cmp",
            "The 2D comparison runs the solver and loads the networks (about 40 s the first time).",
        )
    r = run_compare(case_key, params, opts)
    remember("comparison", {k: v for k, v in r.items() if k not in ("solver", "pinn")})
    show = st.segmented_control("show", ["solver", "PINN", "both"], default="both", key="cmp_show") or "both"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PINN error (rel. L2 vs solver)", f"{100 * r['rel_l2']:.3g} %")
    c2.metric("solver time", f"{r['solver_s']:.3g} s")
    c3.metric("PINN query time", f"{r['pinn_s']:.3g} s")
    c4.metric("speed-up per query", f"{r['speedup']:.3g}x", help="solver time / PINN time for this one query")
    if stop:
        st.error("Extrapolation: the PINN error here is not representative of its trained accuracy.", icon="⛔")
    s, p = r["solver"], r["pinn"]
    if case_key == "A_steady":
        fig = go.Figure()
        if show in ("solver", "both"):
            fig.add_scatter(
                x=s["x"],
                y=s["theta"],
                name="solver",
                mode="lines+markers",
                line={"color": COL["solver"]},
                marker={"size": 4},
            )
        if show in ("PINN", "both"):
            fig.add_scatter(x=p["x"], y=p["mean"], name="PINN ensemble", line={"color": COL["pinn"], "width": 2.5})
        st.plotly_chart(fig_layout(fig, "x", "θ"), key="cmp_a", theme=None)
        fe = go.Figure(go.Scatter(x=p["x"], y=r["error"], line={"color": COL["pinn"]}, name="PINN − solver"))
        st.plotly_chart(fig_layout(fe, "x", "error", height=260), key="cmp_ae", theme=None)
    elif case_key == "B_slab":
        fig = go.Figure()
        for i in range(len(p["t"])):
            if show in ("solver", "both"):
                fig.add_scatter(
                    x=s["x"],
                    y=s["theta"][i],
                    mode="markers",
                    marker={"size": 4, "color": COL["solver"]},
                    name="solver",
                    showlegend=i == 0,
                    legendgroup="s",
                )
            if show in ("PINN", "both"):
                fig.add_scatter(
                    x=p["x"],
                    y=p["mean"][i],
                    line={"color": COL["pinn"]},
                    name="PINN ensemble",
                    showlegend=i == 0,
                    legendgroup="p",
                )
        st.plotly_chart(fig_layout(fig, "x", "θ"), key="cmp_b", theme=None)
        fe = go.Figure()
        for i, tt in enumerate(p["t"]):
            fe.add_scatter(x=p["x"], y=r["error"][i], name=f"t = {tt:.3g}")
        st.plotly_chart(fig_layout(fe, "x", "PINN − solver", height=260), key="cmp_be", theme=None)
    else:
        c1.metric(
            "Nusselt number: solver / PINN",
            f"{r['Nu_solver']:.4g} / {r['Nu_pinn']:.4g}",
            delta=f"{100 * r['Nu_rel_err']:.2g} % error",
            delta_color="off",
        )
        if show in ("solver", "both"):
            st.plotly_chart(heatmap(p["x"], p["y"], r["reference_on_pinn_grid"], "θ solver"), key="cmp_ds", theme=None)
        if show in ("PINN", "both"):
            st.plotly_chart(heatmap(p["x"], p["y"], p["mean"], "θ PINN"), key="cmp_dp", theme=None)
        st.plotly_chart(
            heatmap(p["x"], p["y"], r["error"], "PINN − solver", "RdBu_r", zmid=0.0), key="cmp_de", theme=None
        )
    st.caption(
        "Query times are single queries on this machine; Step 6.7 measures the break-even number of queries "
        "including the training cost."
    )

elif section == "Inverse":
    st.subheader("Identify α, k, h of a heated slab from temperature sensors (Phase 5, least squares)")
    a, b = st.columns([1, 2])
    with a:
        q = st.number_input("known heat source q_vol [W/m³]", 0.0, 1e7, 1e5, format="%.4g", key="inv_q")
        noise = st.number_input("noise level (fraction of θ range)", 0.0, 0.5, 0.02, format="%.3g", key="inv_n")
        ns_ = int(st.number_input("interior sensors", 1, 40, 8, key="inv_s"))
        unknown = st.multiselect("unknowns", ["alpha", "k", "h"], default=["alpha", "k", "h"], key="inv_u")
        seed = int(st.number_input("noise seed", 0, 10_000, 0, key="inv_seed"))
        up = st.file_uploader("or your data: CSV with x [m], t [s], T [, sigma]", type=["csv"], key="inv_file")
    data_file = None
    if up is not None:
        tmp = ROOT / "cache" / "app_uploads"
        tmp.mkdir(parents=True, exist_ok=True)
        data_file = str(tmp / Path(up.name).name)
        Path(data_file).write_bytes(up.getvalue())
    if not unknown:
        st.warning("choose at least one unknown")
        st.stop()
    r = run_inverse(q, noise, ns_, tuple(unknown), seed, data_file)
    remember("inverse", r)
    with b:
        if not r["identifiable"]:
            st.error(r["message"], icon="⛔")
            st.write("null direction (α, k, h in log space):", np.round(r["null_directions"], 3).tolist())
        elif not r["rows"]:
            st.error(r["message"])
        else:
            st.success(f"identifiable (rank {r['rank']} of {r['n_unknown']}); {r['message']}; Bi = {r['Bi']:.4g}")
            st.dataframe(
                [
                    {
                        "parameter": x["parameter"],
                        "estimate": f"{x['estimate']:.4g}",
                        "95 % interval": f"[{x['ci95'][0]:.4g}, {x['ci95'][1]:.4g}]",
                        "truth": "—" if x["truth"] is None else f"{x['truth']:.4g}",
                        "error": "—" if x["error_pct"] is None else f"{x['error_pct']:+.3g} %",
                    }
                    for x in r["rows"]
                ],
                hide_index=True,
            )
            d = r["data"]
            fig = go.Figure()
            xs = r["sensor_x"]
            for j, xv in enumerate(xs[:: max(1, len(xs) // 4)]):
                jj = int(np.argmin(np.abs(xs - xv)))
                sel = np.isclose(np.asarray(d["r"]), xv)
                fig.add_scatter(
                    x=np.asarray(d["t"])[sel],
                    y=np.asarray(d["T"])[sel],
                    mode="markers",
                    marker={"size": 4, "color": COL["data"]},
                    name="sensor data",
                    showlegend=j == 0,
                    legendgroup="d",
                )
                fig.add_scatter(
                    x=r["times"],
                    y=r["fitted"][:, jj],
                    line={"color": COL["solver"]},
                    name="fitted model",
                    showlegend=j == 0,
                    legendgroup="f",
                )
            st.plotly_chart(fig_layout(fig, "time t [s]", "temperature T [°C]"), key="inv_fit", theme=None)
            st.caption(f"data: {r['source']}; synthetic data = verification, not validation against measurements")

elif section == "Convergence":
    if case_key not in ("A_steady", "B_slab"):
        st.info(
            "Grid convergence against an exact solution is shown for the 1D cases (Phase 1.7 verified the 2D solvers)."
        )
        st.stop()
    Ns = st.multiselect("grids N", [5, 10, 20, 40, 80, 160, 320, 640], default=[10, 20, 40, 80, 160, 320], key="cv_N")
    schemes = ["central", "upwind1"] if case_key == "A_steady" else ["central"]
    if case_key == "A_steady":
        schemes = st.multiselect("schemes", ["central", "upwind1", "upwind2", "quick"], default=schemes, key="cv_s")
    if len(Ns) < 3 or not schemes:
        st.warning("choose at least three grids and one scheme")
        st.stop()
    r = run_convergence(case_key, params, tuple(sorted(Ns)), tuple(schemes), opts)
    remember("convergence", r)
    fig = go.Figure()
    for i, (sch, v) in enumerate(r["schemes"].items()):
        fig.add_scatter(
            x=r["h"],
            y=v["err_inf"],
            mode="lines+markers",
            name=f"{sch}: order {v['observed_order']:.2f}",
            line={"color": S.PALETTE[i % len(S.PALETTE)]},
        )
    h = np.array(r["h"])
    e0 = max(max(v["err_inf"]) for v in r["schemes"].values())
    for order, dash in ((1, "dot"), (2, "dash")):
        fig.add_scatter(
            x=h, y=e0 * (h / h[0]) ** order, name=f"slope {order}", line={"color": COL["muted"], "dash": dash}
        )
    st.plotly_chart(fig_layout(fig, "grid spacing h", "max error vs exact", True, True), key="cv", theme=None)
    st.dataframe(
        [{"scheme": k, "observed order": round(v["observed_order"], 3)} for k, v in r["schemes"].items()],
        hide_index=True,
    )

elif section == "Training monitor":
    runs = B.training_runs(case_key)
    if not runs:
        st.info("No saved training logs for this case.")
        st.stop()
    name = st.selectbox("network", list(runs), key="tm_run")
    log = B.read_log(runs[name])
    num = [k for k, v in log.items() if np.issubdtype(v.dtype, np.number) and k not in ("step", "wall_s", "lr")]
    default = [k for k in ("total", "pde", "rel_l2") if k in num] or num[:3]
    cols = st.multiselect("curves", num, default=default, key="tm_cols")
    xk = st.radio("against", ["step", "wall_s"], horizontal=True, key="tm_x")
    fig = go.Figure()
    for i, k in enumerate(cols):
        y = log[k]
        ok = np.isfinite(y) & (y > 0)
        fig.add_scatter(x=log[xk][ok], y=y[ok], name=k, mode="lines", line={"color": S.PALETTE[i % len(S.PALETTE)]})
    st.plotly_chart(
        fig_layout(fig, "optimiser step" if xk == "step" else "wall time [s]", "value", logy=True), key="tm", theme=None
    )
    ph = log.get("phase")
    if ph is not None:
        st.caption("phases: " + ", ".join(dict.fromkeys(ph.tolist())) + f" · log: {runs[name]}")

elif section == "Uncertainty & validity":
    show_validity(msgs)
    if case_key == "D_graetz":
        st.info(
            "For the channel, the ensemble spread field is on the **PINN** page; a parameter sweep of the 2D "
            "networks is in Step 6.3 / 6.6b."
        )
        st.stop()
    p0 = C.params[0]
    lo, hi = p0.train
    rng = st.slider(
        f"sweep {p0.name} (log scale, decades around the training range)", -2.0, 2.0, (-1.0, 1.0), 0.5, key="uv_rng"
    )
    n = int(st.number_input("values", 5, 60, 25, key="uv_n"))
    r = run_sweep(case_key, lo * 10 ** rng[0], hi * 10 ** rng[1], n, tuple(opts.get("members") or []))
    remember("sweep", r)
    fig = go.Figure()
    # training range and the user's value as legend entries: Plotly misplaces shape annotations on log axes
    top = float(max(r["member_max"].max(), r["rel_l2"].max(), r["rms_sd"].max())) * 2
    bot = float(min(r["member_min"].min(), r["rel_l2"].min(), r["rms_sd"].min())) / 2
    fig.add_scatter(
        x=[lo, hi, hi, lo, lo],
        y=[bot, bot, top, top, bot],
        fill="toself",
        fillcolor="rgba(0,158,115,0.12)",
        line={"width": 0},
        name="training range",
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=np.r_[r["value"], r["value"][::-1]],
        y=np.r_[r["member_max"], r["member_min"][::-1]],
        fill="toself",
        fillcolor="rgba(213,94,0,0.15)",
        line={"width": 0},
        name="members (min-max)",
    )
    fig.add_scatter(
        x=r["value"], y=r["rel_l2"], name="ensemble error (rel. L2)", line={"color": COL["pinn"], "width": 2.5}
    )
    fig.add_scatter(x=r["value"], y=r["rms_sd"], name="ensemble spread", line={"color": COL["solver"], "dash": "dash"})
    v = params[p0.name]
    fig.add_scatter(
        x=[v, v], y=[bot, top], mode="lines", line={"color": COL["ref"], "width": 1}, name=f"your {p0.name}"
    )
    fig = fig_layout(fig, p0.name, "error / spread", True, True)
    fig.update_xaxes(range=[np.log10(r["value"].min()), np.log10(r["value"].max())])
    fig.update_yaxes(range=[np.log10(bot), np.log10(top)])
    st.plotly_chart(fig, key="uv", theme=None)
    inside = (r["value"] >= lo) & (r["value"] <= hi)
    c1, c2 = st.columns(2)
    c1.metric("largest error inside the range", f"{100 * r['rel_l2'][inside].max():.3g} %" if inside.any() else "—")
    c2.metric("largest error outside", f"{100 * r['rel_l2'][~inside].max():.3g} %" if (~inside).any() else "—")
    st.caption(
        "Under the training conditions. A spread that grows with the error is the warning sign the model can "
        "give itself (Step 6.3); inside the range the spread is not a calibrated error bar."
    )

elif section == "Export":
    res = st.session_state.get("results", {})
    st.markdown("Results computed in this session (visit a section first to compute it):")
    if not res:
        st.info("Nothing computed yet.")
    for name, r in res.items():
        a, b = st.columns(2)
        a.download_button(
            f"{name}: JSON",
            json.dumps(B.jsonable(r), indent=1).encode(),
            f"{case_key}_{name}.json",
            "application/json",
            key=f"dl_j_{name}",
        )
        if name in ("solver", "pinn"):
            b.download_button(
                f"{name}: CSV",
                B.to_csv(B.field_columns(case_key, r, "theta" if name == "solver" else "mean")),
                f"{case_key}_{name}.csv",
                "text/csv",
                key=f"dl_c_{name}",
            )
    st.markdown("**The same from the command line** (`src/app/cli.py`):")
    sets = " ".join(f"--set {k}={v:g}" for k, v in params.items())
    st.code(f"python -m aerosynapse.thermal.app.cli compare --case {case_key} {sets} --out {case_key}_compare.json", language="bash")
