"""The five workspaces of the workbench (spec Section 14)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from aerosynapse.safety import DISCLAIMER

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]  # Okabe-Ito


def _plot(xs: list[np.ndarray], ys: list[np.ndarray], labels: list[str], xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, (x, y, lab) in enumerate(zip(xs, ys, labels, strict=True)):
        ax.plot(x, y, color=CB[i % len(CB)], label=lab, lw=1.6, ls="-" if i == 0 else "--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


def _field(Z: np.ndarray, title: str, extent: tuple[float, float, float, float], label: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    im = ax.imshow(Z.T, origin="lower", extent=extent, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------- Physics
def physics() -> None:
    from aerosynapse.physics import library as L  # noqa: F401  (registers problems)
    from aerosynapse.physics.bc import BC_CATALOGUE
    from aerosynapse.registry import PHYSICS

    st.header("Physics workspace")
    name = st.selectbox("Problem", [n for n in PHYSICS.names() if n != "harmonic_oscillator"], key="phys_problem")
    p = PHYSICS.get(name)()
    st.code(p.equation)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Domain and parameters")
        st.json(
            {
                "domain": p.domain,
                "t_final": p.t_final,
                "parameters": {k: f"{v.value} [{v.unit}]" for k, v in p.parameters.items()},
            }
        )
        st.subheader("Dimensionless numbers")
        st.json(p.summary()["dimensionless"])
    with c2:
        st.subheader("Boundary conditions")
        for side, bc in p.bcs.items():
            st.markdown(f"**{side}**: {bc.kind} — {BC_CATALOGUE[bc.kind]['definition']}")
        if p.initial_condition is not None and p.dim == 1:
            ((a, b),) = p.domain
            x = np.linspace(a, b, 200)
            st.subheader("Initial condition")
            _plot([x], [p.initial_condition(x)], ["u0(x)"], "x", "u0")


# ---------------------------------------------------------------- Simulation
def simulation() -> None:
    st.header("Simulation workspace")
    solver = st.selectbox(
        "Solver",
        ["conv-diff FV", "heat 1D FD", "Poisson 2D FD", "lid-driven cavity (NS)", "airfoil panel method"],
        key="sim_solver",
    )
    if solver == "conv-diff FV":
        from aerosynapse.numerics.convdiff import solve_convdiff_fv
        from aerosynapse.physics.library import convdiff_1d

        pe = st.slider("Péclet number", 1.0, 500.0, 50.0, key="sim_pe")
        scheme = st.selectbox("Scheme", ["vanleer", "quick", "central", "upwind"], key="sim_scheme")
        n = st.select_slider("Cells (mesh)", [32, 64, 128, 256, 512], 128, key="sim_n")
        if st.button("Run simulation", key="sim_run"):
            p = convdiff_1d(pe=pe)
            r = solve_convdiff_fv(0, 1, n, 1.0, 1 / pe, p.initial_condition, 0.5, scheme=scheme)
            ex = p.exact(r.x, 0.5)
            _plot([r.x, r.x], [r.u, ex], ["numerical", "exact"], "x", "u(x, T)")
            st.metric("max error", f"{np.abs(r.u - ex).max():.3e}")
            st.subheader("Conservation diagnostic")
            _plot([np.arange(len(r.mass))], [np.array(r.mass) - r.mass[0]], ["mass - mass(0)"], "step", "drift")
    elif solver == "heat 1D FD":
        from aerosynapse.numerics.heat import solve_heat_1d
        from aerosynapse.physics.library import heat_1d

        theta = st.select_slider("theta", [0.0, 0.5, 1.0], 0.5, key="sim_theta")
        n = st.slider("Nodes", 8, 256, 64, key="sim_heat_n")
        if st.button("Run simulation", key="sim_heat_run"):
            p = heat_1d()
            try:
                x, s = solve_heat_1d(p, n, nt=n, theta=theta)
                _plot([x, x], [s.u, p.exact(x, p.t_final)], ["numerical", "exact"], "x", "T")
            except Exception as exc:  # stability guard surfaces as a message, not a plausible field
                st.error(f"{type(exc).__name__}: {exc}")
    elif solver == "Poisson 2D FD":
        from aerosynapse.numerics.poisson import solve_poisson_2d
        from aerosynapse.physics.library import poisson_2d

        n = st.slider("Intervals", 8, 64, 32, key="sim_pn")
        if st.button("Run simulation", key="sim_prun"):
            p = poisson_2d()
            X, Y, U = solve_poisson_2d(p, n)
            _field(U.T, "u", (0, 1, 0, 1), "u")
            _field(np.abs(U - p.exact(X, Y)).T, "|error|", (0, 1, 0, 1), "abs error")
    elif solver == "lid-driven cavity (NS)":
        from aerosynapse.numerics.ns2d import lid_driven_cavity

        re = st.select_slider("Reynolds number", [10, 100, 400], 100, key="sim_re")
        if st.button("Run simulation", key="sim_cav"):
            r = lid_driven_cavity(float(re), n=24, t_end=30)
            uc, vc = r.centre_velocity()
            _field(np.hypot(uc, vc), "velocity magnitude", (0, 1, 0, 1), "|u|")
            _field(r.P, "pressure", (0, 1, 0, 1), "p")
            st.metric("max divergence", f"{r.divergence_max:.2e}")
    else:
        from aerosynapse.numerics.panel import solve_panel
        from aerosynapse.physics.airfoil import naca4

        code = st.text_input("NACA 4-digit", "2412", key="sim_naca")
        alpha = st.slider("alpha [deg]", -6.0, 10.0, 4.0, key="sim_alpha")
        if st.button("Run simulation", key="sim_panel"):
            X, Y = naca4(code, 101)
            r = solve_panel(X, Y, alpha)
            _plot([r.xc], [-r.cp], ["-Cp"], "x/c", "-Cp")
            st.metric("C_L (inviscid)", f"{r.cl:.4f}")
            st.caption("Inviscid model: no profile drag, no stall.")


# ---------------------------------------------------------------- ML
def ml() -> None:
    from aerosynapse.ui.jobs import submit

    st.header("ML workspace")
    pe = st.slider("Péclet number (PINN)", 1.0, 1000.0, 10.0, key="ml_pe")
    sampler = st.selectbox("Collocation sampler", ["uniform", "rad", "rar", "r3", "lhs", "sobol"], key="ml_sampler")
    iters = st.slider("Iterations", 50, 3000, 300, key="ml_iters")

    def train(job: Any) -> dict[str, Any]:
        from aerosynapse.ml.nets import MLP, set_seed
        from aerosynapse.ml.pinn import PINNConfig, predict, train_pinn
        from aerosynapse.ml.problems import ConvDiff1D

        set_seed(0)
        p = ConvDiff1D(pe=pe, sigma0=0.1, t_final=1.0)
        m = MLP(2, 1, 32, 3, lo=p.lo, hi=p.hi, periodic_dims=(0,))
        r = train_pinn(
            p,
            m,
            PINNConfig(iterations=iters, n_collocation=1000, sampler=sampler, log_every=10),
            callback=lambda it, row: job.report(f"iteration {it}", it / iters),
        )
        if job.cancelled():
            raise RuntimeError("cancelled")
        G = p.eval_grid(81)
        return {"hist": r.history, "pred": predict(p, m, G).reshape(81, 81), "ref": p.exact(G).reshape(81, 81)}

    c1, c2 = st.columns(2)
    if c1.button("Train PINN in background", key="ml_train"):
        st.session_state["ml_job"] = submit("pinn", train)
    job = st.session_state.get("ml_job")
    if job is not None:
        if c2.button("Cancel", key="ml_cancel"):
            job.cancel_event.set()
        st.progress(job.progress, text=job.message)
        if job.done and job.error is None:
            out = job.result()
            h = out["hist"]
            _plot(
                [np.array([r["it"] for r in h])],
                [np.log10([r["loss"] for r in h])],
                ["log10 loss"],
                "iteration",
                "log10 training loss",
            )
            _field(out["pred"], "PINN prediction", (0, 1, 0, 1), "u")
            err = np.abs(out["pred"] - out["ref"])
            _field(err, "|error| vs exact (validation, not training loss)", (0, 1, 0, 1), "abs error")
            st.metric("relative L2 error", f"{np.linalg.norm(err) / np.linalg.norm(out['ref']):.3e}")
        elif job.error:
            st.error(job.error)
    p4 = RESULTS / "phase4_operators_uq" / "metrics.json"
    if p4.exists():
        st.subheader("Generalisation and uncertainty (Phase 4 results)")
        m = json.loads(p4.read_text())
        rows = {
            k: {s: v[s]["median"] for s in ("test_id", "ood_pe", "ood_input")}
            for k, v in m["operators"].items()
            if isinstance(v, dict) and "test_id" in v
        }
        st.dataframe(rows)
        st.json({k: v.get("coverage_95") for k, v in m.get("ensemble_uq", {}).items()})


# ---------------------------------------------------------------- Optimisation
def optimization() -> None:
    from aerosynapse.optim import demos as D
    from aerosynapse.optim.protocol import OptimizationStudy

    st.header("Optimization workspace")
    method = st.selectbox("Optimiser", ["de", "ga", "sqp", "bo", "cbo", "surrogate_tr"], key="opt_method")
    budget = st.slider("Evaluation budget", 20, 600, 100, key="opt_budget")
    st.latex(r"\min_{m,x_s,\mathrm{mat}}\ \theta_{max}/\theta_{ref}+0.1(m/4)^2\quad\text{s.t.}\ \theta(1)\le0.02")
    if st.button("Run optimisation (12-step protocol)", key="opt_run"):
        space = D.thermal_space(with_material=method not in ("sqp",))
        base = {"m": 1.0, "x_s": 0.8} | ({"material": "aluminium"} if method != "sqp" else {})
        stdy = OptimizationStudy(
            "ui_thermal",
            baseline_design=base,
            space=space,
            objective="thermal rod",
            constraints=["tip temperature"],
            constraints_declared=True,
            budget=budget,
            stopping="budget",
            model=D.thermal_model,
            trusted=lambda d: D.thermal_model(d, n=1600),
        )
        rep = stdy.run(method, "de", seed=0, n_cand=300, pop=10)
        st.session_state["opt_rep"] = rep
    rep = st.session_state.get("opt_rep")
    if rep:
        st.success(rep["verdict"]) if rep["verdict"].startswith("ACCEPT") else st.error(rep["verdict"])
        hist = rep["method"]["history"]
        f = np.array([h["f"][0] for h in hist])
        feas = np.array([h["feasible"] for h in hist])
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.scatter(np.where(feas)[0], f[feas], s=8, c=CB[0], label="feasible")
        ax.scatter(np.where(~feas)[0], f[~feas], s=8, c=CB[3], marker="x", label="infeasible")
        ax.plot(rep["method"]["trace"], c=CB[2], label="best feasible so far")
        ax.set_xlabel("evaluation")
        ax.set_ylabel("objective")
        ax.legend(frameon=False)
        st.pyplot(fig)
        plt.close(fig)
        st.subheader("Best feasible candidate (re-verified with the trusted solver)")
        st.json(rep["method"]["final"])
        sig = [h["sigma"] for h in hist if h["sigma"]]
        st.caption(f"Surrogate uncertainty recorded for {len(sig)} evaluations")
    p5 = RESULTS / "phase5_optimization" / "metrics.json"
    if p5.exists():
        m = json.loads(p5.read_text())
        pts = m["multiobjective"]["beam_points"]
        F = np.array([p["f"] for p in pts])
        ok = np.array([p["feasible"] for p in pts])
        import matplotlib.pyplot as plt

        st.subheader("Pareto front (cantilever: mass, deflection, stress; Phase 5)")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(F[~ok, 0], F[~ok, 1], s=5, c="#999999", label="infeasible")
        ax.scatter(F[ok, 0], F[ok, 1], s=5, c=CB[0], label="feasible")
        P = np.array(m["multiobjective"]["beam_pareto"])
        ax.scatter(P[:, 0], P[:, 1], s=12, c=CB[3], label="non-dominated")
        ax.set_xlabel("mass [kg]")
        ax.set_ylabel("tip deflection [mm]")
        ax.set_yscale("log")
        ax.legend(frameon=False)
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------------- Research
def research() -> None:
    from aerosynapse.ui.jobs import submit

    st.header("Research workspace")
    recs = sorted(p.parent.name for p in RESULTS.glob("*/record.json"))
    if not recs:
        st.info("No experiment records yet.")
    sel = st.selectbox("Experiment", recs, key="res_exp") if recs else None
    if sel:
        rec = json.loads((RESULTS / sel / "record.json").read_text())
        st.markdown(f"**Research question:** {rec['research_question']}")
        st.markdown(f"**Hypothesis (recorded before running):** {rec['hypothesis']}")
        st.markdown(f"**Status:** {rec['status']} — git `{rec['git_commit'][:10]}` — seed {rec['seed']}")
        st.code(rec["reproducibility"], language="bash")
        st.json(rec["software_versions"])
        metrics = json.loads((RESULTS / sel / "metrics.json").read_text())
        st.download_button(
            "Export metrics (JSON)", json.dumps(metrics, indent=1, default=str), f"{sel}.json", key="res_dl_json"
        )
        flat = {k: v for k, v in metrics.items() if isinstance(v, int | float)}
        st.download_button(
            "Export scalar metrics (CSV)",
            "key,value\n" + "\n".join(f"{k},{v}" for k, v in flat.items()),
            f"{sel}.csv",
            key="res_dl_csv",
        )
    st.subheader("End-to-end workflow (define → simulate → train → validate → optimise → report)")

    def wf(job: Any) -> dict[str, Any]:
        from aerosynapse.config import load_config
        from aerosynapse.experiments.runner import run_workflow

        cfg = load_config(ROOT / "configs" / "workflow_quick.yaml").model_copy(update={"experiment_id": "workflow_ui"})
        return run_workflow(cfg, RESULTS, progress=job.report, cancel=job.cancelled)

    if st.button("Run quick workflow in background", key="res_wf"):
        st.session_state["wf_job"] = submit("workflow", wf)
    job = st.session_state.get("wf_job")
    if job is not None:
        st.progress(job.progress, text=job.message)
        if job.done and job.error is None:
            out = job.result()
            st.json({k: out[k] for k in ("simulate", "validate")} | {"optimise": out["optimise"]["verdict"]})
            st.download_button(
                "Download HTML report", Path(out["report_html"]).read_text(), "report.html", key="res_dl_html"
            )
        elif job.error:
            st.error(job.error)
    st.caption(DISCLAIMER)
