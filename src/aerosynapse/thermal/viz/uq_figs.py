"""Step 6.2 figures (pure functions: metrics dict in, Figure out), in the project style of src/viz/inverse_figs.py."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from aerosynapse.thermal.viz.inverse_figs import AXIS, GRID, INK, INK2, SURF, style  # noqa: F401  (style re-exported)

ENS = "#2a78d6"
MCD = ["#eb6834", "#b04a1e", "#f2a37d"]  # one shade per dropout rate (fixed order)


def _methods(an: dict) -> list[tuple[str, str, str]]:
    """(key, label, colour) for the largest ensemble and every MC-dropout network, in a fixed order."""
    ms = an["methods"]
    ens = sorted((k for k in ms if k.startswith("ensemble_M")), key=lambda k: int(k[10:]))
    out = [(ens[-1], f"deep ensemble (M = {ens[-1][10:]})", ENS)] if ens else []
    mcd = [k for k in ms if k.startswith("mcd_")]
    for i, k in enumerate(mcd):
        out.append((k, f"MC-dropout p = {k.split('_')[1]}", MCD[i % len(MCD)]))
    return out


def reliability(analysis: dict) -> plt.Figure:
    """empirical vs nominal coverage of central intervals on the test values; solid = raw Gaussian intervals,
    dashed = after split-conformal recalibration on the calibration values."""
    probs = list(analysis)
    fig, axs = plt.subplots(1, len(probs), figsize=(4.0 * len(probs), 3.9), squeeze=False)
    for ax, p in zip(axs[0], probs, strict=True):
        ax.plot([0, 1], [0, 1], color=AXIS, lw=0.9, zorder=1)
        for key, _, col in _methods(analysis[p]):
            r = analysis[p]["methods"][key]["reliability"]
            ax.plot(r["levels"], r["raw"], "-o", color=col, ms=3.5, lw=1.4, mec=SURF, zorder=3)
            ax.plot(r["levels"], r["conformal"], color=col, lw=1.2, ls=(0, (4, 2)), zorder=2)
        ax.set(
            xlim=(0, 1), ylim=(0, 1.02), xlabel="nominal coverage", ylabel="empirical coverage (test values)", title=p
        )
        ax.set_aspect("equal")
    hs = [plt.Line2D([], [], color=c, marker="o", ms=4, label=lab) for _, lab, c in _methods(analysis[probs[0]])]
    hs += [
        plt.Line2D([], [], color=INK2, ls=(0, (4, 2)), label="after conformal recalibration"),
        plt.Line2D([], [], color=AXIS, label="perfect calibration"),
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=3, fontsize=8)
    return fig


def spread_vs_error(analysis: dict) -> plt.Figure:
    """rows = problems: (a) rms error vs rms sd per test value (1:1 = consistent), (b) dispersion vs ensemble size,
    (c) 95 % coverage vs ensemble size, raw and conformal."""
    probs = list(analysis)
    fig, axs = plt.subplots(len(probs), 3, figsize=(11.5, 3.3 * len(probs)), squeeze=False)
    for i, p in enumerate(probs):
        an = analysis[p]
        ax = axs[i][0]
        lo, hi = np.inf, 0.0
        for key, _, col in _methods(an):
            pv = an["methods"][key]["per_value"]
            s, e = [v["rms_sd"] for v in pv], [v["rmse"] for v in pv]
            ax.plot(s, e, "o", color=col, ms=5, mec=SURF, zorder=3)
            lo, hi = min(lo, *s, *e), max(hi, *s, *e)
        g = np.array([lo / 2, hi * 2])
        ax.plot(g, g, color=AXIS, lw=0.9, zorder=1)
        ax.set(xscale="log", yscale="log", xlabel="rms sd (per grid value)", ylabel=f"{p}: rms error")
        ens = sorted((k for k in an["methods"] if k.startswith("ensemble_M")), key=lambda k: int(k[10:]))
        M = [int(k[10:]) for k in ens]
        ax = axs[i][1]
        ax.plot(M, [an["methods"][k]["dispersion"] for k in ens], "-o", color=ENS, ms=5, mec=SURF)
        ax.axhline(1.0, color=AXIS, lw=0.9)
        ax.set(xlabel="ensemble size M", ylabel="dispersion (rms error / rms sd)")
        ax.set_ylim(bottom=0)
        ax.set_xticks(M)
        ax = axs[i][2]
        ax.plot(M, [an["methods"][k]["coverage"]["0.95"] for k in ens], "-o", color=ENS, ms=5, mec=SURF)
        ax.plot(M, [an["methods"][k]["conformal"]["0.05"]["test_coverage"] for k in ens], color=ENS, ls=(0, (4, 2)))
        ax.axhline(0.95, color=AXIS, lw=0.9)
        ax.set(ylim=(0, 1.02), xlabel="ensemble size M", ylabel="coverage of the 95 % interval")
        ax.set_xticks(M)
    hs = [plt.Line2D([], [], color=c, marker="o", ls="", ms=5, label=lab) for _, lab, c in _methods(analysis[probs[0]])]
    hs += [
        plt.Line2D([], [], color=ENS, ls=(0, (4, 2)), label="ensemble, conformal 95 %"),
        plt.Line2D([], [], color=AXIS, label="consistent / nominal"),
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=4, fontsize=8)
    return fig


def bands(analysis: dict) -> plt.Figure:
    """columns = problems, at the middle test value (transient: the time slice of the largest ensemble error):
    top = truth and mean +- 2 sd per method; bottom = |error| and 2 sd on a log axis."""
    probs = list(analysis)
    fig, axs = plt.subplots(2, len(probs), figsize=(4.8 * len(probs), 5.6), squeeze=False, sharex="col")
    for j, p in enumerate(probs):
        ex = analysis[p]["example"]
        y = np.asarray(ex["truth"])
        x = np.linspace(0, 1, len(y))
        top, bot = axs[0][j], axs[1][j]
        top.plot(x, y, color=INK, lw=1.4, zorder=20)
        for z, (key, _, col) in enumerate(reversed(_methods(analysis[p]))):  # MC-dropout first, ensemble on top
            mu, sd = np.asarray(ex["methods"][key]["mu"]), np.asarray(ex["methods"][key]["sd"])
            top.fill_between(x, mu - 2 * sd, mu + 2 * sd, color=col, alpha=0.3, lw=0, zorder=2 + 2 * z)
            top.plot(x, mu, color=col, lw=1.1, ls=(0, (4, 2)), zorder=3 + 2 * z)
            bot.plot(x, np.abs(mu - y), color=col, lw=1.2)
            bot.plot(x, 2 * sd, color=col, lw=1.0, ls=(0, (1, 1.5)))
        sl = "" if ex["slice"] is None else f", time slice {ex['slice'] + 1}"
        top.set(ylabel=r"$\theta$", title=f"{p}: parameter {ex['value']:.3g}{sl}")
        bot.set(yscale="log", xlabel="$x^*$", ylabel="|error| (solid), 2 sd (dotted)")
    hs = [plt.Line2D([], [], color=INK, lw=1.4, label="analytic truth")]
    hs += [
        plt.Line2D([], [], color=c, lw=6, alpha=0.4, label=f"{lab}: mean ± 2 sd")
        for _, lab, c in _methods(analysis[probs[0]])
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=2, fontsize=8)
    return fig
