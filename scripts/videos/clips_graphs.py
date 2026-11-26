"""Animated result graphs — every value is read from results/<id>/metrics.json (or fields.json)."""

from __future__ import annotations

import json

import numpy as np
from _base import CLIPS, ROOT, load

from aerosynapse.viz import neon as N

FPS = 30


def convergence() -> str:
    m = load("phase1_verification")
    keys = [
        ("heat1d_cn", "heat 1D CN"),
        ("convdiff_fv_upwind_pe50", "conv-diff upwind"),
        ("convdiff_fv_vanleer_pe50", "conv-diff van Leer"),
        ("burgers_fv", "Burgers FV"),
        ("wave_leapfrog", "wave leapfrog"),
        ("poisson2d_fd", "Poisson 2D FD"),
        ("poisson2d_fem_H1", "P1 FEM (H1)"),
    ]
    fig, ax = N.figure()
    fig.subplots_adjust(top=0.84, left=0.08, right=0.7)
    N.header(
        fig,
        "Verification: every solver reaches its theoretical order",
        "error vs grid spacing (log–log) · observed order on the finest pair",
        "results/phase1_verification",
    )
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlabel("h"), ax.set_ylabel("error")
    allh = np.concatenate([m[k]["h"] for k, _ in keys])
    alle = np.concatenate([m[k]["errors"] for k, _ in keys])
    ax.set_xlim(allh.min() * 0.7, allh.max() * 1.4), ax.set_ylim(alle.min() * 0.3, alle.max() * 3)
    lines = [N.glow(ax, np.ones(1), np.ones(1), N.NEON[i % 6], 2.0) for i in range(len(keys))]
    labels = [fig.text(0.72, 0.8 - 0.07 * i, "", color=N.NEON[i % 6], fontsize=12) for i in range(len(keys))]
    frames = 210

    def frame(k: int) -> None:
        for i, (key, name) in enumerate(keys):
            prog = N.ease((k - i * 15) / 60)
            h, e = np.array(m[key]["h"]), np.array(m[key]["errors"])
            npts = 1 + prog * (len(h) - 1)
            j = int(np.floor(npts))
            frac = npts - j
            hh, ee = list(h[:j]), list(e[:j])
            if j < len(h) and j > 0:
                hh.append(np.exp(np.log(h[j - 1]) + frac * (np.log(h[j]) - np.log(h[j - 1]))))
                ee.append(np.exp(np.log(e[j - 1]) + frac * (np.log(e[j]) - np.log(e[j - 1]))))
            N.set_glow(lines[i], np.array(hh), np.array(ee))
            if prog >= 1:
                labels[i].set_text(f"{name}:  p = {m[key]['pairwise_orders'][-1]:.3f}")

    return str(N.write(fig, frame, frames, CLIPS / "30_convergence.mp4", FPS))


def samplers() -> str:
    t = load("phase3_adaptive_collocation")["table"]
    pes = ["1.0", "10.0", "100.0", "1000.0"]
    meth = ["uniform", "uniform_resample", "lhs", "sobol", "rar", "rad", "r3"]
    fig, ax = N.figure()
    fig.subplots_adjust(top=0.82, bottom=0.12)
    N.header(
        fig,
        "Core study: does adaptive collocation help at high Péclet number?",
        "median rel. L2 error · 95% bootstrap CI · 8 paired seeds · equal budget  →  no significant gain (negative result)",
        "results/phase3_adaptive_collocation (Wilcoxon + Holm; 272 PINN runs)",
    )
    w = 0.11
    ax.set_yscale("log"), ax.set_ylim(0.005, 0.5), ax.set_xticks(range(4), [f"Pe = {float(p):g}" for p in pes])
    ax.set_ylabel("median relative L2 error")
    bars = []
    for j, me in enumerate(meth):
        b = ax.bar(
            np.arange(4) + (j - 3) * w,
            [0.005] * 4,
            w,
            color=N.NEON[j % 6],
            alpha=0.85,
            label=me,
            edgecolor="white",
            linewidth=0.3,
        )
        bars.append(b)
    ax.legend(ncol=7, loc="upper center", fontsize=8, bbox_to_anchor=(0.5, 1.0))
    errs = []

    def frame(k: int) -> None:
        nonlocal errs
        prog = N.ease(k / 90)
        for j, me in enumerate(meth):
            for i, pe in enumerate(pes):
                med = t[f"{pe}/{me}"]["median"]
                bars[j][i].set_height(np.exp(np.log(0.005) + prog * (np.log(med) - np.log(0.005))) - 0.0)
        if k == 95:
            for j, me in enumerate(meth):
                lo = [t[f"{pe}/{me}"]["median"] - t[f"{pe}/{me}"]["ci95"][0] for pe in pes]
                hi = [t[f"{pe}/{me}"]["ci95"][1] - t[f"{pe}/{me}"]["median"] for pe in pes]
                errs.append(
                    ax.errorbar(
                        np.arange(4) + (j - 3) * w,
                        [t[f"{pe}/{me}"]["median"] for pe in pes],
                        yerr=[lo, hi],
                        fmt="none",
                        ecolor="white",
                        elinewidth=0.8,
                        capsize=2,
                    )
                )
        if k == 130:
            ax.text(3, 0.42, "no sampler significantly\nbetter than uniform", ha="center", color=N.GOLD, fontsize=12)
            ax.text(0, 0.3, "initial-layer\nfailure", ha="center", color=N.ORANGE, fontsize=11)

    return str(N.write(fig, frame, 200, CLIPS / "31_samplers.mp4", FPS))


def _rotating_surface(
    P: np.ndarray,
    Q: np.ndarray,
    Z: np.ndarray,
    title: str,
    sub: str,
    src: str,
    labels: tuple[str, str, str],
    out: str,
    cmap=N.AURORA,
    extra=None,
) -> str:
    fig = N.plt.figure(figsize=(N.W, N.H), dpi=N.DPI)
    N.header(fig, title, sub, src)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.85], projection="3d")

    def frame(k: int) -> None:
        ax.cla()
        N.style_3d(ax)
        grow = N.ease(k / 60)
        Zs = Z.min() + grow * (Z - Z.min())
        ax.plot_surface(
            P,
            Q,
            Zs,
            facecolors=N.shaded(Z, cmap),
            rstride=1,
            cstride=1,
            linewidth=0.2,
            edgecolor=(1, 1, 1, 0.08),
            shade=False,
        )
        if extra:
            extra(ax, Z)
        ax.set_xlabel(labels[0]), ax.set_ylabel(labels[1]), ax.set_zlabel(labels[2])
        ax.set_zlim(Z.min(), Z.max())
        ax.view_init(26 + 8 * np.sin(k / 50), -70 + 1.2 * k)

    return str(N.write(fig, frame, 240, CLIPS / out, FPS))


def generalisation() -> str:
    g = load("phase4_generalisation_map")["generalisation_map"]
    P, K = np.meshgrid(np.log10(g["pe"]), g["k_max"], indexing="ij")
    Z = np.log10(np.array(g["median_rel_l2"]))

    def box(ax, Z):  # noqa: ANN001
        lo, hi, zb = np.log10(5), np.log10(50), Z.min()
        for (a, b), (c, d) in (((lo, hi), (2, 2)), ((lo, hi), (4, 4)), ((lo, lo), (2, 4)), ((hi, hi), (2, 4))):
            ax.plot([a, b], [c, d], [zb, zb], color=N.LIME, lw=3)
        ax.text(lo, 4.4, zb, "training region", color=N.LIME)

    return _rotating_surface(
        P,
        K,
        Z,
        "Where does the neural operator stop being valid?",
        "FNO error over (Péclet number, input bandwidth) · trained on Pe 5–50 · error explodes outside",
        "results/phase4_generalisation_map",
        ("log10 Pe", "k_max", "log10 error"),
        "32_generalisation.mp4",
        extra=box,
    )


def performance() -> str:
    s = load("phase3_extras")["budget_sweep"]
    pes, ns = [1.0, 10.0, 100.0, 1000.0], [500, 1000, 2000]
    P, Nn = np.meshgrid(np.log10(pes), np.log10(ns), indexing="ij")
    Z = np.log10(np.array([[s[f"{pe}/{n}/uniform"]["median"] for n in ns] for pe in pes]))
    Zr = np.log10(np.array([[s[f"{pe}/{n}/rad"]["median"] for n in ns] for pe in pes]))

    def rad(ax, Z):  # noqa: ANN001
        ax.plot_wireframe(P, Nn, Zr, color=N.CYAN, lw=1.2)

    return _rotating_surface(
        P,
        Nn,
        Z,
        "PINN error vs Péclet number and collocation budget",
        "surface: uniform sampling · cyan wireframe: RAD · budget barely matters (optimisation-limited)",
        "results/phase3_extras",
        ("log10 Pe", "log10 N_coll", "log10 error"),
        "33_performance.mp4",
        extra=rad,
    )


def pareto() -> str:
    m = load("phase5_optimization")["multiobjective"]
    F = np.array([p["f"] for p in m["beam_points"]])
    ok = np.array([p["feasible"] for p in m["beam_points"]])
    Pp = np.array(m["beam_pareto"])
    fig = N.plt.figure(figsize=(N.W, N.H), dpi=N.DPI)
    N.header(
        fig,
        "Multi-objective design: cantilever beam Pareto front (NSGA-II)",
        "mass × tip deflection × max stress · points appear in evaluation order · magenta = non-dominated feasible",
        "results/phase5_optimization (3000 evaluations; illustrative material data)",
    )
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.85], projection="3d")
    order = np.arange(len(F))

    def frame(k: int) -> None:
        ax.cla()
        N.style_3d(ax)
        n = int(len(F) * N.ease(k / 150))
        sel = order[:n]
        a, b = sel[~ok[sel]], sel[ok[sel]]
        ax.scatter(F[a, 0], np.log10(F[a, 1]), F[a, 2], s=1, c="#3a4566", alpha=0.5)
        ax.scatter(F[b, 0], np.log10(F[b, 1]), F[b, 2], s=2, c=F[b, 0], cmap=N.ICE, alpha=0.6)
        if k > 150:
            ax.scatter(Pp[:, 0], np.log10(Pp[:, 1]), Pp[:, 2], s=6, c=N.MAGENTA, depthshade=True)
        ax.set_xlabel("mass [kg]"), ax.set_ylabel("log10 deflection [mm]"), ax.set_zlabel("stress [MPa]")
        ax.view_init(22, -60 + 0.9 * k)

    return str(N.write(fig, frame, 260, CLIPS / "34_pareto.mp4", FPS))


def spacetime() -> str:
    fields = json.loads((ROOT / "results" / "phase3_extras" / "fields.json").read_text())
    f = fields[-1]
    Xg, Tg = np.meshgrid(f["x"], f["t"], indexing="ij")
    Zp, Zr = np.array(f["pred"]), np.array(f["ref"])
    fig = N.plt.figure(figsize=(N.W, N.H), dpi=N.DPI)
    N.header(
        fig,
        f"PINN space–time solution vs exact, Pe = {f['pe']:g}",
        f"left: exact · right: PINN (RAD) · relative L2 = {f['rel_l2']:.3f}",
        "results/phase3_extras/fields.json",
    )
    a1 = fig.add_axes([0.0, 0.05, 0.5, 0.82], projection="3d")
    a2 = fig.add_axes([0.5, 0.05, 0.5, 0.82], projection="3d")

    def frame(k: int) -> None:
        for a, Z, ttl in ((a1, Zr, "exact"), (a2, Zp, "PINN")):
            a.cla()
            N.style_3d(a)
            a.plot_surface(
                Xg, Tg, Z, facecolors=N.shaded(Z, N.AURORA, 0, 1), rstride=1, cstride=1, linewidth=0, shade=False
            )
            a.set_xlabel("x"), a.set_ylabel("t"), a.set_title(ttl, color=N.CYAN), a.set_zlim(0, 1)
            a.view_init(30, -120 + 1.1 * k)

    return str(N.write(fig, frame, 200, CLIPS / "35_spacetime.mp4", FPS))


def trust() -> str:
    fb = load("phase7_validation_engine")["fallback"]
    groups = [("in_distribution", "in-distribution"), ("ood_pe", "OOD Péclet"), ("ood_input", "OOD inputs")]
    fig, (a1, a2) = N.figure(1, 2)
    fig.subplots_adjust(top=0.8, bottom=0.14, wspace=0.25)
    N.header(
        fig,
        "Trust engine: when to fall back to the classical solver",
        "worst-case error of the raw surrogate vs the guarded system · fallback rate per query set",
        "results/phase7_validation_engine (300 logged decisions)",
    )
    a1.set_yscale("log"), a1.set_ylim(1e-4, 0.5), a1.set_ylabel("worst-case relative L2")
    a1.set_xticks(range(3), [g[1] for g in groups])
    a2.set_ylim(0, 1.05), a2.set_xticks(range(3), [g[1] for g in groups]), a2.set_ylabel("fallback rate")
    raw = a1.bar(np.arange(3) - 0.18, [1e-4] * 3, 0.34, color=N.ORANGE, label="raw surrogate")
    gd = a1.bar(np.arange(3) + 0.18, [1e-4] * 3, 0.34, color=N.CYAN, label="guarded system")
    fr = a2.bar(range(3), [0] * 3, 0.5, color=N.MAGENTA)
    a1.legend(fontsize=9)

    def frame(k: int) -> None:
        p = N.ease(k / 80)
        for i, (g, _) in enumerate(groups):
            raw[i].set_height(np.exp(np.log(1e-4) + p * (np.log(fb[g]["surrogate_err_max"]) - np.log(1e-4))))
            gd[i].set_height(np.exp(np.log(1e-4) + p * (np.log(fb[g]["guarded_err_max"]) - np.log(1e-4))))
            fr[i].set_height(p * fb[g]["fallback_rate"])

    return str(N.write(fig, frame, 150, CLIPS / "36_trust.mp4", FPS))


def ghia() -> str:
    m = load("phase6_fluids_aero")["cavity"]
    g = json.loads((ROOT / "src" / "aerosynapse" / "data" / "ghia1982.json").read_text())
    fig, (a1, a2) = N.figure(1, 2)
    fig.subplots_adjust(top=0.82, bottom=0.12, wspace=0.22)
    N.header(
        fig,
        "Validation: lid-driven cavity vs Ghia et al. (1982)",
        "lines: this solver (64²) · circles: published benchmark table · Re = 100 and 400",
        "results/phase6_fluids_aero · src/aerosynapse/data/ghia1982.json (fetched transcription)",
    )
    a1.set_xlabel("u(0.5, y)"), a1.set_ylabel("y"), a2.set_xlabel("x"), a2.set_ylabel("v(x, 0.5)")
    arts = []
    for i, re in enumerate((100, 400)):
        pr = m[f"re{re}_n64"]["profile"]
        c = (N.CYAN, N.MAGENTA)[i]
        arts.append(
            (
                pr,
                c,
                N.glow(a1, np.zeros(1), np.zeros(1), c, 1.8, f"Re = {re}"),
                N.glow(a2, np.zeros(1), np.zeros(1), c, 1.8),
            )
        )
        a1.scatter(
            g["u_vertical_centerline"][str(re)],
            g["u_vertical_centerline"]["coord"],
            s=30,
            facecolors="none",
            edgecolors=c,
        )
        a2.scatter(
            g["v_horizontal_centerline"]["coord"],
            g["v_horizontal_centerline"][str(re)],
            s=30,
            facecolors="none",
            edgecolors=c,
        )
    a1.legend(fontsize=9)

    def frame(k: int) -> None:
        p = N.ease(k / 90)
        for pr, _, g1, g2 in arts:
            n = max(2, int(p * len(pr["y"])))
            N.set_glow(g1, np.array(pr["u"][:n]), np.array(pr["y"][:n]))
            N.set_glow(g2, np.array(pr["x"][:n]), np.array(pr["v"][:n]))

    return str(N.write(fig, frame, 150, CLIPS / "37_ghia.mp4", FPS))


ALL = [convergence, samplers, generalisation, performance, pareto, spacetime, trust, ghia]
