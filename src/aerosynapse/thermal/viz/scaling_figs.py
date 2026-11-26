"""Phase 6 figures (pure functions: data in, Figure out), in the project style of src/viz/inverse_figs.py."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from aerosynapse.thermal.pinn import scaling as S
from aerosynapse.thermal.viz.inverse_figs import (  # noqa: F401  (style re-exported)
    AXIS,
    BASE,
    EMPH,
    GRID,
    INK,
    INK2,
    MUTED,
    SURF,
    style,
)

METHOD = {"pinn": "#2a78d6", "fd": "#eb6834", "spectral": "#1baf7a"}


def _log_labels(ax) -> None:
    """Label minor ticks (2, 5 x 10^n) too when a log y-axis spans less than two decades (no major tick otherwise);
    within one decade matplotlib falls back to linear ticks, hence 4 significant digits (1.625, not 1.63)."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    lo, hi = ax.get_ylim()
    if lo > 0 and hi / lo < 100:
        ax.yaxis.set_minor_locator(LogLocator(subs=(2.0, 5.0)))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:.4g}"))


def _fit_curve(f: dict, xs: np.ndarray) -> np.ndarray:
    if f["best"] == "saturating" and f["saturating"]:
        s = f["saturating"]
        return s["A"] * xs ** (-s["q"]) + s["floor"]
    return f["loglog"]["C"] * xs ** f["loglog"]["slope"]


def scaling_grid(runs: list[dict], fits: dict, base: dict, label: dict) -> plt.Figure:
    """rows = problems, columns = swept factors: relative L2 per seed (dots), median (line), preferred fit (dashed),
    base level (vertical line); log-log axes."""
    probs = list(fits)
    factors = list(next(iter(fits.values())))
    fig, axs = plt.subplots(
        len(probs), len(factors), figsize=(3.0 * len(factors), 2.7 * len(probs)), squeeze=False, sharey="row"
    )
    for i, p in enumerate(probs):
        for j, fac in enumerate(factors):
            ax = axs[i][j]
            f = fits[p][fac]
            x = np.asarray(f["levels"], float)
            for lv in x:
                is_base = int(lv) == int(base[fac])
                e = [
                    r["rel_l2"]
                    for r in runs
                    if r["problem"] == p
                    and ((r["factor"] == "base") if is_base else (r["factor"] == fac and r["level"] == int(lv)))
                ]
                ax.plot(np.full(len(e), lv), e, "o", ms=4, color=BASE, alpha=0.6, mew=0, zorder=2)
            ax.plot(x, f["median"], "-o", color=INK, lw=1.3, ms=5, mec=SURF, zorder=3)
            xs = np.geomspace(x.min(), x.max(), 100)
            ax.plot(xs, _fit_curve(f, xs), color=EMPH, lw=1.3, ls=(0, (4, 3)), zorder=4)
            ci = f["exponent_ci95"]
            if f["best"] == "saturating" and f["saturating"] and not f.get("resolved", True):
                txt = f"step to floor {f['saturating']['floor']:.2g} (exponent unresolved)"
                ci = None
            elif f["best"] == "saturating" and f["saturating"]:
                txt = f"x^-{abs(f['exponent']):.2f} + {f['saturating']['floor']:.1g}"
            else:
                txt = f"slope {f['exponent']:+.2f}"
            if ci:
                lo = f"≤{ci[0]:+.0f}" if f.get("ci_at_bound") else f"{ci[0]:+.2f}"
                txt += f"  [{lo}, {ci[1]:+.2f}]"
            ax.set_title(txt, loc="left", fontsize=7, color=INK2)  # above the panel: never on top of the data
            ax.axvline(base[fac], color=AXIS, lw=0.8, ls=(0, (2, 2)), zorder=1)
            ax.set(xscale="log", yscale="log")
            ax.set_xticks(x, [str(int(v)) for v in x], fontsize=7)
            ax.minorticks_off()
            if i == len(probs) - 1:
                ax.set_xlabel(label.get(fac, fac))
            if j == 0:
                ax.set_ylabel(f"{p}: relative L2")
        _log_labels(axs[i][0])
    hs = [
        plt.Line2D([], [], marker="o", ls="", color=BASE, ms=4, label="seed"),
        plt.Line2D([], [], marker="o", color=INK, ms=5, label="median"),
        plt.Line2D([], [], color=EMPH, ls=(0, (4, 3)), label="preferred fit (AIC)"),
        plt.Line2D([], [], color=AXIS, ls=(0, (2, 2)), label="base setting"),
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=4, fontsize=8)
    return fig


def cost_vs_error(runs: list[dict], solvers: dict, targets: list[float], settings: dict | None = None) -> plt.Figure:
    """one panel per problem: every PINN run (faint), the per-setting medians over seeds with their Pareto front (or the
    runs' front when no settings are given), FD and Chebyshev solves of increasing size (error vs solve time); target
    errors as dotted lines."""
    probs = list(solvers)
    fig, axs = plt.subplots(1, len(probs), figsize=(5.0 * len(probs), 3.8), squeeze=False)
    for ax, p in zip(axs[0], probs, strict=True):
        pr = [r for r in runs if r["problem"] == p]
        e, t = np.array([r["rel_l2"] for r in pr]), np.array([r["wall_s"] for r in pr])
        ax.plot(t, e, "o", ms=3, color=METHOD["pinn"], alpha=0.25, mew=0, zorder=2)
        if settings:
            e = np.array([s["rel_l2"] for s in settings[p]])
            t = np.array([s["wall_s"] for s in settings[p]])
            ax.plot(t, e, "s", ms=4.5, color=METHOD["pinn"], mec=SURF, mew=0.6, zorder=3)
        fr = S.pareto(e, t)
        ax.plot(t[fr], e[fr], "-", color=METHOD["pinn"], lw=1.4, zorder=3)
        for kind, name in (("fd", "FD"), ("spectral", "Chebyshev")):
            c = solvers[p][kind]
            ax.plot(
                [x["wall_s"] for x in c],
                [x["rel_l2"] for x in c],
                "-o",
                color=METHOD[kind],
                ms=5,
                mec=SURF,
                lw=1.3,
                zorder=3,
            )
            ax.annotate(
                f"{name} N {c[-1]['N']}",
                (c[-1]["wall_s"], c[-1]["rel_l2"]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=7,
                color=INK2,
                va="center",
            )
        for tg in targets:
            ax.axhline(tg, color=GRID, lw=0.9, ls=(0, (2, 2)), zorder=1)
        ax.set(xscale="log", yscale="log", xlabel="wall time [s] (training or solve)", ylabel="relative L2", title=p)
    hs = [
        plt.Line2D([], [], marker="o", ls="", color=METHOD["pinn"], alpha=0.35, ms=4, label="PINN runs"),
        plt.Line2D([], [], marker="s", ls="", color=METHOD["pinn"], ms=5, label="PINN setting (median of seeds)"),
        plt.Line2D([], [], color=METHOD["pinn"], lw=1.4, label="PINN Pareto front"),
        plt.Line2D([], [], marker="o", color=METHOD["fd"], ms=5, label="FD (2nd order)"),
        plt.Line2D([], [], marker="o", color=METHOD["spectral"], ms=5, label="Chebyshev"),
        plt.Line2D([], [], color=GRID, ls=(0, (2, 2)), label="target errors"),
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=3, fontsize=8)
    return fig
