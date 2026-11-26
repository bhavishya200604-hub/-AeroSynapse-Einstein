"""THERMAL LAB's features as pages (Streamlit-style code), so they run in Studio's native Labs window.

Each page asks for the same inputs as the THERMAL LAB web page, runs the lab's own calculation
(``labs.run_kind``: solve, converge, sweep, inverse, real_fit, lumped, flow, dataset; the real-data PINN training job)
and shows the lab's own figures (``labs.figures_for``), its tables (``labs._tables``) and the main numbers. Every page
can also save the lab's ZIP export (README, request, result, tables, figures, reproduce.py).
"""

from __future__ import annotations

import json
import time
from typing import Any

import streamlit as st  # the native stand-in inside Studio (nativelab.shim)


def _labs():
    from aerosynapse.thermal.web import labs as L
    from aerosynapse.thermal.web import server as S

    return L, S


def _show(kind: str, q: dict, r: dict) -> None:
    L, S = _labs()
    clean = S.clean(r) if hasattr(S, "clean") else r
    nums = {k: v for k, v in clean.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if nums:
        cols = st.columns(min(4, len(nums)))
        for i, (k, v) in enumerate(list(nums.items())[:8]):
            cols[i % len(cols)].metric(k.replace("_", " "), f"{v:.5g}")
    try:
        figs = L.figures_for(kind, q, r)
    except Exception as e:  # noqa: BLE001
        figs = {}
        st.warning(f"the lab's figures could not be drawn: {type(e).__name__}: {e}")
    import io

    from PIL import Image

    for name, png in figs.items():
        st.subheader(name.replace("_", " "))
        st.image(Image.open(io.BytesIO(png)))
    try:
        import pandas as pd

        big = []
        for name, text in list(L._tables(clean).items())[:16]:
            n = text.count("\n")
            if n > 300:  # whole fields: kept in the ZIP export, not listed here
                big.append(f"{name} ({n} rows)")
                continue
            st.subheader("table: " + name.replace("_", " ").removesuffix(".csv"))
            st.dataframe(pd.read_csv(io.StringIO(text)))
        if big:
            st.caption("Whole fields, in the ZIP export (tick it above): " + ", ".join(big))
    except Exception as e:  # noqa: BLE001
        st.caption(f"(tables: {e})")
    for k in ("validity", "warnings", "notes", "note"):
        if k in clean and clean[k]:
            st.info(f"{k}: " + (json.dumps(clean[k])[:1500] if not isinstance(clean[k], str) else clean[k]))
    if st.checkbox(
        "Save the lab's ZIP export of this result (README, request, result, tables, figures, reproduce.py)",
        value=False,
        key=f"zip_{kind}",
    ):
        data, name = L.export({"kind": kind, "request": q})
        st.download_button("ZIP export", data, file_name=name)


def _case_params(key: str, prefix: str) -> dict[str, float]:
    from aerosynapse.thermal.app import backend as B

    c = B.case(key)
    out = {}
    for p in c.params:
        out[p.name] = st.number_input(
            f"{p.name} ({p.help})",
            min_value=float(p.lo),
            max_value=float(p.hi),
            value=float(p.default),
            key=f"{prefix}_{key}_{p.name}",
        )
    return out


def _cases() -> dict[str, str]:
    from aerosynapse.thermal.app import backend as B

    return {k: c.title for k, c in B.CASES.items()}


def page_overview() -> None:
    _, S = _labs()
    st.header("THERMAL LAB: the convection-diffusion PINN project")
    info = S.api_info({})
    import pandas as pd

    st.subheader("Problems")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "case": c["key"],
                    "title": c["title"],
                    "dim": c["dim"],
                    "transient": c["transient"],
                    "PINN": c["pinn"],
                    "about the PINN": c["pinn_note"],
                    "parameters": ", ".join(f"{p['name']} [{p['lo']:g}, {p['hi']:g}]" for p in c["params"]),
                }
                for c in info["cases"]
            ]
        )
    )
    st.subheader("Soil stations on disk (real data)")
    st.dataframe(pd.DataFrame(info.get("stations") or []))
    if info.get("findings"):
        st.subheader("Findings")
        st.json(info["findings"])
    st.caption(f"build {info.get('build', '')}, mode {info.get('mode', '')}")


def page_solve() -> None:
    st.header("Solve: verified solver, PINN and their comparison")
    cases = _cases()
    key = st.selectbox("Problem", list(cases), format_func=lambda k: f"{k}: {cases[k]}", key="solve_case")
    params = _case_params(key, "solve")
    mode = st.radio(
        "Mode",
        ["compare", "solver"],
        format_func=lambda m: {"compare": "solver and PINN compared", "solver": "solver only"}[m],
        key="solve_mode",
    )
    opts: dict[str, Any] = {}
    if key in ("A_steady", "B_slab"):
        opts["N"] = st.number_input("Grid cells N", 8, 4000, 200, key="solve_N")
    if key == "B_slab":
        opts["T"] = st.number_input("End time (dimensionless)", 0.01, 5.0, 0.5, key="solve_T")
        opts["n_out"] = st.number_input("Output times", 2, 60, 11, key="solve_nout")
    if key == "D_graetz":
        opts["cells_per_unit"] = st.number_input("Cells per unit length", 4, 64, 16, key="solve_cpu")
    if key == "cavity":
        opts["N"] = st.number_input("Grid N x N", 8, 128, 32, key="solve_Nc")
    q = {"case": key, "params": params, "opts": opts, "mode": mode}
    L, _ = _labs()
    r = L.run_kind("solve", q)
    _show("solve", q, r)


def page_converge() -> None:
    st.header("Grid convergence: order of accuracy of the solvers")
    cases = _cases()
    key = st.selectbox("Problem", list(cases), format_func=lambda k: f"{k}: {cases[k]}", key="conv_case")
    params = _case_params(key, "conv")
    Ns = st.text_input("Grids (cells, comma separated)", "16, 32, 64, 128, 256", key="conv_Ns")
    schemes = st.multiselect(
        "Schemes",
        ["upwind1", "central", "upwind2", "quick", "tvd_minmod", "tvd_vanleer", "tvd_superbee", "tvd_vanalbada"],
        default=["upwind1", "central", "quick"],
        key="conv_sch",
    )
    q = {
        "case": key,
        "params": params,
        "Ns": [int(x) for x in Ns.replace(";", ",").split(",") if x.strip()],
        "schemes": schemes,
    }
    L, _ = _labs()
    _show("converge", q, L.run_kind("converge", q))


def page_sweep() -> None:
    st.header("Parameter sweep of the PINN ensemble (accuracy and spread)")
    key = st.selectbox("Problem", ["A_steady", "B_slab"], key="sweep_case")
    from aerosynapse.thermal.app import backend as B

    p = B.case(key).params[0]
    lo = st.number_input(f"{p.name} from", float(p.lo), float(p.hi), float(p.lo), key="sweep_lo")
    hi = st.number_input(f"{p.name} to", float(p.lo), float(p.hi), float(p.hi), key="sweep_hi")
    n = st.slider("Values", 3, 60, 25, key="sweep_n")
    q = {"case": key, "lo": lo, "hi": hi, "n": n}
    L, _ = _labs()
    _show("sweep", q, L.run_kind("sweep", q))


def page_inverse() -> None:
    st.header("Inverse identification of alpha, k and h from temperature sensors")
    un = st.multiselect("Unknowns", ["alpha", "k", "h"], default=["alpha", "k", "h"], key="inv_un")
    qv = st.number_input("q_vol, volumetric heat source (W/m3)", 0.0, 1e9, 1e5, key="inv_q")
    noise = st.slider("Sensor noise (fraction)", 0.0, 0.5, 0.02, key="inv_noise")
    ns = st.slider("Sensors", 1, 64, 8, key="inv_ns")
    seed = st.number_input("Seed", 0, 10**6, 0, key="inv_seed")
    f = st.file_uploader(
        "Measured temperatures (CSV, optional: otherwise synthetic sensors)", type=["csv"], key="inv_csv"
    )
    q: dict[str, Any] = {"unknown": un, "q_vol": qv, "noise": noise, "n_sensors": ns, "seed": seed}
    if f is not None:
        q["csv"] = f.getvalue().decode("utf-8", "replace")
    L, _ = _labs()
    _show("inverse", q, L.run_kind("inverse", q))


def _stations() -> list[dict]:
    _, S = _labs()
    return S.api_real_stations({}).get("stations", [])


def page_real_fit() -> None:
    st.header("Real data: soil thermal diffusivity from buried sensors")
    src = st.radio(
        "Data",
        ["station", "table"],
        format_func=lambda s: {"station": "a soil station on disk (USCRN)", "table": "my own table (CSV)"}[s],
        key="rf_src",
    )
    q: dict[str, Any] = {"source": src}
    if src == "station":
        sts = _stations()
        names = sorted({s["station"] for s in sts}) or ["NE_Lincoln_11_SW"]
        q["station"] = st.selectbox("Station", names, key="rf_station")
        q["year"] = st.number_input("Year", 2000, 2030, 2023, key="rf_year")
        q["start"] = st.text_input("Start (date)", "2023-07-01", key="rf_start")
    else:
        f = st.file_uploader(
            "Temperature table (CSV: a time column and one column per depth)", type=["csv"], key="rf_csv"
        )
        if f is None:
            _, S = _labs()
            ex = S.api_real_example({})
            st.info(f"No table chosen: the lab's example ({ex['name']}) is used. {ex.get('note', '')}")
            csv = ex["csv"]
        else:
            csv = f.getvalue().decode("utf-8", "replace")
        q["csv"] = csv
        q["time_col"] = st.text_input("Time column", "time", key="rf_tcol")
        pos = st.text_input("Columns and depths (m)", "T_5cm=0.05, T_10cm=0.10, T_20cm=0.20, T_50cm=0.50", key="rf_pos")
        q["positions"] = {a.split("=")[0].strip(): float(a.split("=")[1]) for a in pos.split(",") if "=" in a}
    q["days"] = st.slider("Window (days)", 1, 120, 14, key="rf_days")
    q["max_gap_h"] = st.number_input("Largest gap to fill (hours)", 0.0, 48.0, 3.0, key="rf_gap")
    q["bottom_m"] = st.number_input("Bottom boundary depth (m)", 0.05, 2.0, 0.5, key="rf_bottom")
    q["dz_m"] = st.number_input("Grid spacing (m)", 0.002, 0.1, 0.01, key="rf_dz")
    q["dt_s"] = st.number_input("Time step (s)", 10.0, 7200.0, 600.0, key="rf_dt")
    q["spinup_h"] = st.number_input("Spin-up (hours)", 0.0, 240.0, 24.0, key="rf_spin")
    L, _ = _labs()
    _show("real_fit", q, L.run_kind("real_fit", q))


def page_lumped() -> None:
    st.header("Real data: lumped cooling (Newton) fit of one temperature record")
    f = st.file_uploader("Temperature record (CSV)", type=["csv"], key="lp_csv")
    _, S = _labs()
    if f is None:
        ex = S.api_real_example({"kind": "lumped"})
        st.info(f"No table chosen: the lab's example ({ex['name']}) is used.")
        csv = ex["csv"]
    else:
        csv = f.getvalue().decode("utf-8", "replace")
    cols = csv.splitlines()[0].split(",") if csv else []
    tcol = st.selectbox("Temperature column", cols[1:] or ["T"], key="lp_t")
    acol = st.selectbox("Ambient column", ["(none)"] + cols[1:], key="lp_a")
    q = {
        "csv": csv,
        "temp_col": tcol,
        "ambient_col": None if acol == "(none)" else acol,
        "time_col": cols[0] if cols else "time",
        "fit_source": st.checkbox("Also fit a heat source", value=False, key="lp_src"),
        "hac_lag": st.number_input("HAC lag (serial correlation)", 0, 200, 0, key="lp_hac"),
    }
    L, _ = _labs()
    _show("lumped", q, L.run_kind("lumped", q))


def page_flow() -> None:
    L, _ = _labs()
    st.header("Cavity flow: Navier-Stokes and heat in time (frames)")
    q = {
        "Re": st.number_input("Re (Reynolds)", 1.0, 3000.0, 400.0, key="fl_Re"),
        "Pr": st.number_input("Pr (Prandtl)", 0.01, 100.0, 0.71, key="fl_Pr"),
        "Bi": st.number_input("Bi (Biot, the convective wall)", 1e-3, 1e3, 1.0, key="fl_Bi"),
        "Ri": st.number_input("Ri (Richardson: buoyancy / inertia)", -10.0, 100.0, 0.0, key="fl_Ri"),
        "heating": st.selectbox("Heating", list(L.HEATING), key="fl_heat"),
        "lid": st.selectbox("Lid", list(L.LIDS), key="fl_lid"),
        "lid_speed": st.slider("Lid speed", -3.0, 3.0, 1.0, key="fl_ls"),
        "grid": st.slider("Grid", 8, int(L.lim("flow_grid")), 32, key="fl_grid"),
        "frames": st.slider("Frames", 2, int(L.lim("flow_frames")), 12, key="fl_frames"),
        "t_end": st.number_input("End time", 0.1, float(L.lim("flow_t_end")), 8.0, key="fl_tend"),
    }
    _show("flow", q, L.run_kind("flow", q))


def page_dataset() -> None:
    L, _ = _labs()
    st.header("Dataset pipeline: design of experiments, ground truth, sensors and noise")
    spec = L._data_cfg()
    case = st.selectbox("Case", list(spec["cases"]), key="ds_case")
    q = {
        "case": case,
        "seed": st.number_input("Seed", 0, 10**6, int(spec["seed"]), key="ds_seed"),
        "method": st.selectbox("Design method", ["sobol", "lhs", "grid", "random"], key="ds_method"),
        "u_lim": st.slider(
            "Training box (fraction of each range)", 0.3, 0.95, float(spec["design"]["unit_train_limit"]), key="ds_ulim"
        ),
        "noise_model": st.selectbox("Noise model", ["gaussian", "relative", "heteroscedastic"], key="ds_noise"),
        "n_sensors": st.slider("Sensors", 1, 64, 8, key="ds_ns"),
        "n_times": st.slider("Output times", 1, 40, int(spec["cases"][case]["times"]["n"]), key="ds_nt"),
    }
    _show("dataset", q, L.run_kind("dataset", q))


def page_train() -> None:
    L, S = _labs()
    st.header("Real-data PINN training (soil temperature, diffusivity identified)")
    sts = _stations()
    names = sorted({s["station"] for s in sts}) or ["NE_Lincoln_11_SW"]
    q: dict[str, Any] = {
        "source": "station",
        "station": st.selectbox("Station", names, key="tr_station"),
        "year": st.number_input("Year", 2000, 2030, 2023, key="tr_year"),
        "start": st.text_input("Start (date)", "2023-07-01", key="tr_start"),
        "days": st.slider("Window (days)", 1, 60, 14, key="tr_days"),
    }
    for k, lab, lo, hi, d in (
        ("width", "Network width", 4, 128, 32),
        ("depth", "Hidden layers", 1, 8, 3),
        ("harmonics", "Fourier harmonics of the day", 0, 6, 2),
        ("n_colloc", "Collocation points", 100, int(L.lim("train_colloc")), 1500),
        ("warmup", "Warm-up steps (data only)", 0, 10000, 500),
        ("adam_steps", "Adam steps", 10, int(L.lim("train_adam")), 3000),
        ("lbfgs_steps", "L-BFGS steps", 0, int(L.lim("train_lbfgs")), 300),
        ("seed", "Seed", 0, 10**6, 0),
    ):
        q[k] = st.number_input(lab, lo, hi, d, key=f"tr_{k}")
    q["lr"] = st.number_input("Learning rate", 1e-5, 0.1, 3e-3, format="%.5f", key="tr_lr")
    q["w_pde"] = st.number_input("Weight of the heat equation", 0.0, 1e4, 1.0, key="tr_wpde")
    if not st.button("Train now", key="tr_go"):
        st.info("Set the inputs and press Train now: the training runs here and shows its progress (several minutes).")
        return
    t_start = time.time()
    job = S.api_train_start(q)
    jid = job["id"]
    bar = st.progress(0.0, "starting")
    while True:
        s = L.train_status({"id": jid})
        tot = max(1, int(s.get("total") or 1))
        bar.progress(min(1.0, s.get("step", 0) / tot), f"{s.get('state')}: step {s.get('step', 0)} of {tot}")
        if s.get("state") in ("done", "error", "cancelled"):
            break
        time.sleep(1.5)
    if s.get("state") != "done":
        st.error(f"training {s.get('state')}: {s.get('error', '')}")
        return
    st.success(
        f"trained: {tot} steps, finished at {time.strftime('%H:%M:%S')} on {time.strftime('%Y-%m-%d')} after {(time.time() - t_start) / 60:.1f} min"
    )
    _show("train", {**q, "id": jid}, s)


PAGES = {
    "Overview": page_overview,
    "Solve (solver, PINN, compare)": page_solve,
    "Grid convergence": page_converge,
    "PINN parameter sweep": page_sweep,
    "Inverse identification": page_inverse,
    "Real data: soil diffusivity": page_real_fit,
    "Real data: lumped cooling": page_lumped,
    "Cavity flow": page_flow,
    "Dataset pipeline": page_dataset,
    "Real-data PINN training": page_train,
}
