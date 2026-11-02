"""All paper figures (A–G) and documentation data figures. Each function returns written paths or a skip reason."""

from __future__ import annotations

from typing import Any

import numpy as np
from common import DOUBLE, OKABE, SINGLE, load, plt, save, save_html


def fig_a_architecture() -> dict[str, Any]:
    """Fig. A (schematic): isometric stack of layers A–F with data/control flow and the validation gate.
    Fig. B (pipeline) is merged into this figure as the right-hand arrow sequence."""
    import plotly.graph_objects as go

    layers = [
        "A  Physics definition",
        "B  Classical solver",
        "C  Scientific ML",
        "D  Optimisation",
        "E  Validation engine",
        "F  Workbench",
    ]
    fig = plt.figure(figsize=(DOUBLE, DOUBLE * 0.55))
    ax = fig.add_subplot(111, projection="3d")
    for i, name in enumerate(layers):
        z = i * 1.0
        xs, ys = np.array([0, 4, 4, 0, 0]), np.array([0, 0, 2.5, 2.5, 0])
        ax.plot(xs, ys, z, color=OKABE[i % 7], lw=1.2)
        ax.plot_surface(
            np.array([[0, 4], [0, 4]]),
            np.array([[0, 0], [2.5, 2.5]]),
            np.full((2, 2), z),
            color=OKABE[i % 7],
            alpha=0.25,
            shade=False,
        )
        ax.text(4.3, 1.2, z, name, fontsize=7, color="black")
        if i < len(layers) - 1:
            ax.quiver(1.0, 1.25, z + 0.08, 0, 0, 0.8, color="#333333", arrow_length_ratio=0.25, lw=0.8)
    ax.quiver(3.0, 1.25, 3.9, 0, 0, -3.7, color=OKABE[3], arrow_length_ratio=0.08, lw=1.0)
    ax.text(3.1, 1.25, 2.1, "validation gate\n(trust / fallback)", fontsize=6, color=OKABE[3])
    pipeline = ["Mathematics", "Physics", "Simulation", "Sci-ML", "Optimisation", "Engineering\ndecision"]
    for i, s in enumerate(pipeline):
        ax.text(-2.6, 0, i * 1.0, s, fontsize=6, color="#555555")
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-60)
    ax.set_title("Schematic (not data): layered architecture and research pipeline", fontsize=8)
    paths = save(fig, "fig_a_architecture")
    g = go.Figure()
    for i, name in enumerate(layers):
        g.add_trace(
            go.Mesh3d(
                x=[0, 4, 4, 0],
                y=[0, 0, 2.5, 2.5],
                z=[i] * 4,
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=OKABE[i % 7],
                opacity=0.35,
                name=name,
                showlegend=True,
            )
        )
    g.update_layout(title="Schematic: AeroSynapse–Einstein layers", scene={"zaxis": {"title": "layer"}})
    paths.append(save_html(g, "fig_a_architecture"))
    return {"paths": paths}


def fig_c_spacetime() -> dict[str, Any]:
    fields = load("phase3_extras", "fields.json")
    if fields is None:
        return {"skipped": "phase3_extras not run"}
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = plt.figure(figsize=(DOUBLE, DOUBLE * 0.85))
    gh = make_subplots(
        rows=len(fields),
        cols=3,
        specs=[[{"type": "surface"}] * 3] * len(fields),
        subplot_titles=[f"{k} Pe={f['pe']:g}" for f in fields for k in ("reference", "PINN", "|error|")],
    )
    for r, f in enumerate(fields):
        X, T = np.meshgrid(f["x"], f["t"], indexing="ij")
        ref, pred = np.array(f["ref"]), np.array(f["pred"])
        for c, (Z, title) in enumerate(
            ((ref, "reference (exact)"), (pred, "PINN (RAD)"), (np.abs(pred - ref), "|error|"))
        ):
            ax = fig.add_subplot(len(fields), 3, 3 * r + c + 1, projection="3d")
            ax.plot_surface(X, T, Z, cmap="viridis", linewidth=0, antialiased=True, rstride=2, cstride=2)
            if c == 2:
                P = np.array(f["points"])
                ax.scatter(P[:, 0], P[:, 1], np.full(len(P), Z.max()), s=0.3, c=OKABE[3], alpha=0.5)
            ax.set_xlabel("$x$", labelpad=-8)
            ax.set_ylabel("$t$", labelpad=-8)
            ax.set_title(f"{title}, Pe={f['pe']:g}" + (f"\nrel. $L_2$={f['rel_l2']:.3f}" if c == 1 else ""), fontsize=7)
            ax.tick_params(pad=-3, labelsize=5)
            gh.add_trace(go.Surface(x=X, y=T, z=Z, showscale=False), row=r + 1, col=c + 1)
    paths = save(fig, "fig_c_spacetime_surfaces")
    paths.append(save_html(gh, "fig_c_spacetime_surfaces"))
    return {"paths": paths}


def fig_d_performance() -> dict[str, Any]:
    m = load("phase3_extras")
    if m is None:
        return {"skipped": "phase3_extras not run"}
    import plotly.graph_objects as go

    pes, ns = [1.0, 10.0, 100.0, 1000.0], [500, 1000, 2000]
    fig = plt.figure(figsize=(SINGLE * 1.4, SINGLE * 1.15))
    ax = fig.add_subplot(111, projection="3d")
    g = go.Figure()
    for k, (meth, col) in enumerate((("uniform", OKABE[0]), ("rad", OKABE[3]))):
        Z = np.array([[np.log10(m["budget_sweep"][f"{pe}/{n}/{meth}"]["median"]) for n in ns] for pe in pes])
        P, N = np.meshgrid(np.log10(pes), np.log10(ns), indexing="ij")
        ax.plot_wireframe(P, N, Z, color=col, label=meth, lw=0.9)
        g.add_trace(go.Surface(x=P, y=N, z=Z, name=meth, showscale=False, opacity=0.7, colorscale=[[0, col], [1, col]]))
        _ = k
    ax.set_xlabel(r"$\log_{10}$ Pe")
    ax.set_ylabel(r"$\log_{10}$ $N_{coll}$")
    ax.set_zlabel(r"$\log_{10}$ median rel. $L_2$")
    ax.legend(loc="upper left", frameon=False)
    paths = save(fig, "fig_d_performance_surface")
    g.update_layout(title="Median PINN rel. L2 vs (Pe, collocation budget, sampler)")
    paths.append(save_html(g, "fig_d_performance_surface"))
    return {"paths": paths}


def fig_e_pareto() -> dict[str, Any]:
    m = load("phase5_optimization")
    if m is None:
        return {"skipped": "phase5_optimization not run"}
    import plotly.graph_objects as go

    pts = m["multiobjective"]["beam_points"]
    F = np.array([p["f"] for p in pts])
    ok = np.array([p["feasible"] for p in pts])
    P = np.array(m["multiobjective"]["beam_pareto"])
    fig = plt.figure(figsize=(SINGLE * 1.4, SINGLE * 1.15))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(F[~ok, 0], np.log10(F[~ok, 1]), F[~ok, 2], s=1, c="#BBBBBB", label="infeasible")
    ax.scatter(F[ok, 0], np.log10(F[ok, 1]), F[ok, 2], s=1, c=OKABE[0], alpha=0.4, label="feasible, dominated")
    ax.scatter(P[:, 0], np.log10(P[:, 1]), P[:, 2], s=4, c=OKABE[3], label="non-dominated (feasible)")
    ax.set_xlabel("mass [kg]")
    ax.set_ylabel(r"$\log_{10}$ deflection [mm]")
    ax.set_zlabel("max stress [MPa]")
    ax.legend(loc="upper left", frameon=False, markerscale=3)
    paths = save(fig, "fig_e_pareto_3d")
    g = go.Figure(
        [
            go.Scatter3d(x=F[ok, 0], y=F[ok, 1], z=F[ok, 2], mode="markers", marker={"size": 2}, name="feasible"),
            go.Scatter3d(
                x=F[~ok, 0],
                y=F[~ok, 1],
                z=F[~ok, 2],
                mode="markers",
                marker={"size": 2, "color": "#bbb"},
                name="infeasible",
            ),
            go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", marker={"size": 4}, name="Pareto"),
        ]
    )
    paths.append(save_html(g, "fig_e_pareto_3d"))
    return {"paths": paths}


def fig_f_airfoil() -> dict[str, Any]:
    m = load("phase6_fluids_aero")
    if m is None:
        return {"skipped": "phase6_fluids_aero not run"}
    import matplotlib.colors as mcolors
    import plotly.graph_objects as go

    d = m["panel"]["naca2412_cp_alpha4"]
    x, y, cp = (np.array(d[k]) for k in ("x", "y", "cp"))
    fig = plt.figure(figsize=(DOUBLE * 0.7, DOUBLE * 0.42))
    ax = fig.add_subplot(111, projection="3d")
    span = np.linspace(0, 3, 12)
    Xs, Ss = np.meshgrid(x, span, indexing="ij")
    Ys = np.repeat(y[:, None], len(span), 1)
    C = np.repeat(cp[:, None], len(span), 1)
    norm = mcolors.Normalize(C.min(), C.max())
    ax.plot_surface(Xs, Ss, Ys, facecolors=plt.cm.coolwarm(norm(C)), linewidth=0, shade=False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    fig.colorbar(sm, ax=ax, shrink=0.6, label="$C_p$ (inviscid panel method)")
    ax.set_box_aspect((1, 2, 0.3))
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("span (extrusion)")
    ax.set_zlabel("$y/c$")
    ax.set_title(f"NACA 2412, alpha = 4 deg, $C_L$ = {d['cl']:.3f} (2D section extruded; no 3D effects)", fontsize=7)
    paths = save(fig, "fig_f_airfoil_cp_3d")
    g = go.Figure(go.Surface(x=Xs, y=Ss, z=Ys, surfacecolor=C, colorscale="RdBu_r", colorbar={"title": "Cp"}))
    paths.append(save_html(g, "fig_f_airfoil_cp_3d"))
    return {"paths": paths}


def fig_g_generalisation() -> dict[str, Any]:
    m = load("phase4_operators_uq")
    if m is None or "generalisation_map" not in m:
        return {"skipped": "phase4_operators_uq (generalisation_map) not run"}
    import plotly.graph_objects as go

    g4 = m["generalisation_map"]
    pes, ks = np.array(g4["pe"]), np.array(g4["k_max"])
    E = np.log10(np.array(g4["median_rel_l2"]))
    P, K = np.meshgrid(np.log10(pes), ks, indexing="ij")
    fig = plt.figure(figsize=(SINGLE * 1.4, SINGLE * 1.15))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(P, K, E, cmap="viridis", alpha=0.85, linewidth=0)
    lo, hi = np.log10(5), np.log10(50)
    zb = E.min()
    for (a, b), (c, d2) in (((lo, hi), (1, 1)), ((lo, hi), (4, 4)), ((lo, lo), (1, 4)), ((hi, hi), (1, 4))):
        ax.plot([a, b], [c, d2], [zb, zb], color=OKABE[3], lw=1.5)
    ax.text(lo, 4.3, zb, "training region", color=OKABE[3], fontsize=6)
    ax.set_xlabel(r"$\log_{10}$ Pe")
    ax.set_ylabel(r"input bandwidth $k_{max}$")
    ax.set_zlabel(r"$\log_{10}$ median rel. $L_2$")
    paths = save(fig, "fig_g_generalisation_map")
    gg = go.Figure(go.Surface(x=P, y=K, z=E))
    gg.update_layout(title="FNO generalisation map (training region Pe 5-50, k_max <= 4)")
    paths.append(save_html(gg, "fig_g_generalisation_map"))
    return {"paths": paths}


def fig_convergence() -> dict[str, Any]:
    m = load("phase1_verification")
    if m is None:
        return {"skipped": "phase1 not run"}
    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.8))
    for i, k in enumerate(
        [
            "heat1d_cn",
            "convdiff_fv_upwind_pe50",
            "convdiff_fv_vanleer_pe50",
            "burgers_fv",
            "wave_leapfrog",
            "poisson2d_fd",
            "poisson2d_fem_H1",
        ]
    ):
        ax.loglog(
            m[k]["h"],
            m[k]["errors"],
            "o-",
            ms=2.5,
            lw=1,
            label=f"{k} (p={m[k]['pairwise_orders'][-1]:.2f})",
            color=OKABE[i % 8],
        )
    ax.set_xlabel("$h$")
    ax.set_ylabel("error")
    ax.legend(frameon=False, fontsize=5)
    return {"paths": save(fig, "fig_convergence_phase1")}


def fig_cavity() -> dict[str, Any]:
    m = load("phase6_fluids_aero")
    import json
    from pathlib import Path

    if m is None:
        return {"skipped": "phase6 not run"}
    g = json.loads((Path(__file__).resolve().parents[2] / "src/aerosynapse/data/ghia1982.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE * 0.7, DOUBLE * 0.3))
    for i, re in enumerate((100, 400)):
        key = f"re{re}_n64"
        if key not in m["cavity"]:
            continue
        pr = m["cavity"][key]["profile"]
        axs[0].plot(pr["u"], pr["y"], color=OKABE[i], label=f"Re={re} (64$^2$)")
        axs[0].plot(
            g["u_vertical_centerline"][str(re)],
            g["u_vertical_centerline"]["coord"],
            "o",
            ms=2.5,
            color=OKABE[i],
            mfc="none",
        )
        axs[1].plot(pr["x"], pr["v"], color=OKABE[i])
        axs[1].plot(
            g["v_horizontal_centerline"]["coord"],
            g["v_horizontal_centerline"][str(re)],
            "o",
            ms=2.5,
            color=OKABE[i],
            mfc="none",
        )
    axs[0].set_xlabel("$u(0.5, y)$")
    axs[0].set_ylabel("$y$")
    axs[1].set_xlabel("$x$")
    axs[1].set_ylabel("$v(x, 0.5)$")
    axs[0].legend(frameon=False, title="lines: this solver, o: Ghia et al. (1982)", title_fontsize=6)
    return {"paths": save(fig, "fig_cavity_vs_ghia")}


def fig_phase3_bars() -> dict[str, Any]:
    m = load("phase3_adaptive_collocation")
    if m is None:
        return {"skipped": "phase3 not run"}
    t = m["table"]
    pes = sorted({float(k.split("/")[0]) for k in t})
    meths = ["uniform", "uniform_resample", "lhs", "sobol", "rar", "rad", "r3"]
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.75, DOUBLE * 0.3))
    w = 0.11
    for j, me in enumerate(meths):
        med = [t[f"{pe}/{me}"]["median"] for pe in pes]
        lo = [t[f"{pe}/{me}"]["median"] - t[f"{pe}/{me}"]["ci95"][0] for pe in pes]
        hi = [t[f"{pe}/{me}"]["ci95"][1] - t[f"{pe}/{me}"]["median"] for pe in pes]
        ax.bar(np.arange(len(pes)) + (j - 3) * w, med, w, yerr=[lo, hi], label=me, color=OKABE[j % 8], capsize=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(pes)), [f"Pe={p:g}" for p in pes])
    ax.set_ylabel("median rel. $L_2$ (95% bootstrap CI)")
    ax.legend(ncol=4, frameon=False, fontsize=6)
    return {"paths": save(fig, "fig_phase3_samplers")}


ALL = [
    fig_a_architecture,
    fig_c_spacetime,
    fig_d_performance,
    fig_e_pareto,
    fig_f_airfoil,
    fig_g_generalisation,
    fig_convergence,
    fig_cavity,
    fig_phase3_bars,
]
