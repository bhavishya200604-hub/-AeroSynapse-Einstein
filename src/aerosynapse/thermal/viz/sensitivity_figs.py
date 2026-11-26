"""Step 6.4 figures (pure functions: metrics dict in, Figure out), in the project style of src/viz/inverse_figs.py."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from aerosynapse.thermal.viz.inverse_figs import AXIS, GRID, INK, INK2, SURF, style  # noqa: F401  (style re-exported)

NAMES = ["Re", "Pr", "Bi"]
SOLVER, PCE, PINN, MEMBER = "#eb6834", "#f2a37d", "#2a78d6", "#1a4f99"
INPUT = {"Re": "#2a78d6", "Pr": "#eb6834", "Bi": "#1baf7a"}


def indices(cases: dict) -> plt.Figure:
    """rows = cases, columns = first-order S and total ST: solver Monte Carlo (95 % interval), solver PCE, PINN
    ensemble (95 % interval), PINN members (dots)."""
    names = list(cases)
    fig, axs = plt.subplots(len(names), 2, figsize=(10.5, 3.3 * len(names)), squeeze=False, sharey="row")
    w = 0.26
    x = np.arange(len(NAMES))
    for i, c in enumerate(names):
        r = cases[c]
        mem = [k for k in r["pinn"] if k not in ("ensemble", "ensemble_same_design", "n", "n_runs", "wall_s")]
        for j, (idx, cik) in enumerate((("S", "S_ci95"), ("ST", "ST_ci95"))):
            ax = axs[i][j]
            for k, (vals, ci, col) in enumerate(
                (
                    (r["solver"][idx], r["solver"][cik], SOLVER),
                    (r["pce"][idx], None, PCE),
                    (r["pinn"]["ensemble"][idx], r["pinn"]["ensemble"][cik], PINN),
                )
            ):
                xs = x + (k - 1) * w
                ax.bar(xs, vals, w * 0.9, color=col, zorder=2)
                if ci:
                    lo = [v - a for v, (a, _) in zip(vals, ci, strict=True)]
                    hi = [b - v for v, (_, b) in zip(vals, ci, strict=True)]
                    ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
            for m in mem:
                ax.plot(x + w, r["pinn"][m][idx], "o", ms=3.5, color=MEMBER, mec=SURF, mew=0.5, zorder=4)
            ax.axhline(0, color=AXIS, lw=0.8)
            ax.set_xticks(x, NAMES)
            ax.set_title(f"{c}: {'first-order S' if idx == 'S' else 'total ST'}", loc="left", fontsize=9, color=INK2)
            if j == 0:
                ax.set_ylabel("Sobol index")
    hs = [
        plt.Rectangle((0, 0), 1, 1, color=SOLVER, label="solver, Monte Carlo (95 % interval)"),
        plt.Rectangle((0, 0), 1, 1, color=PCE, label="solver, polynomial chaos"),
        plt.Rectangle((0, 0), 1, 1, color=PINN, label="PINN ensemble mean (95 % interval)"),
        plt.Line2D([], [], marker="o", ls="", color=MEMBER, ms=4, label="PINN members"),
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=2, fontsize=8)
    return fig


def convergence(cases: dict) -> plt.Figure:
    """rows = cases: solver indices vs the base size N of the design (ST solid with its 95 % band, S dashed)."""
    names = list(cases)
    fig, axs = plt.subplots(1, len(names), figsize=(5.2 * len(names), 3.6), squeeze=False)
    for ax, c in zip(axs[0], names, strict=True):
        conv = cases[c]["convergence"]
        N = [r["N"] for r in conv]
        for i, k in enumerate(NAMES):
            st = [r["ST"][i] for r in conv]
            lo = [r["ST_ci95"][i][0] for r in conv]
            hi = [r["ST_ci95"][i][1] for r in conv]
            ax.fill_between(N, lo, hi, color=INPUT[k], alpha=0.15, lw=0)
            ax.plot(N, st, "-o", color=INPUT[k], ms=4, mec=SURF, lw=1.4)
            ax.plot(N, [r["S"][i] for r in conv], color=INPUT[k], lw=1.0, ls=(0, (4, 2)))
        ax.set(xscale="log", xlabel="Saltelli base size N (solver runs = 5 N)", ylabel="Sobol index", title=c)
        ax.set_xticks(N, [str(n) for n in N])
        ax.minorticks_off()
    hs = [plt.Line2D([], [], color=INPUT[k], lw=1.5, label=k) for k in NAMES]
    hs += [
        plt.Line2D([], [], color=INK2, lw=1.4, marker="o", ms=4, label="total ST (band: 95 %)"),
        plt.Line2D([], [], color=INK2, lw=1.0, ls=(0, (4, 2)), label="first order S"),
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=5, fontsize=8)
    return fig


def paired(cases: dict) -> plt.Figure:
    """columns = cases: PINN ensemble QoI vs solver QoI on the solver's design points (1:1 line)."""
    names = list(cases)
    fig, axs = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.0), squeeze=False)
    for ax, c in zip(axs[0], names, strict=True):
        s = np.asarray(cases[c]["paired"]["sample"]["solver"])
        p = np.asarray(cases[c]["paired"]["sample"]["pinn"])
        ax.plot(s, p, "o", ms=2.5, color=PINN, alpha=0.4, mew=0)
        lo, hi = min(s.min(), p.min()), max(s.max(), p.max())
        log = lo > 0 and hi / lo > 20
        g = np.geomspace(lo, hi, 50) if log else np.linspace(lo, hi, 50)
        ax.plot(g, g, color=INK2, lw=0.9)
        if log:
            ax.set(xscale="log", yscale="log")
        pr = cases[c]["paired"]
        ax.set(xlabel=f"solver {cases[c]['qoi']}", ylabel=f"PINN ensemble {cases[c]['qoi']}")
        ax.set_title(f"{c}: median rel. error {pr['rel_err_median']:.2g}", loc="left", fontsize=9, color=INK2)
    return fig
