"""Step 6.3 figures (pure functions: metrics dict in, Figure out), in the project style of src/viz/inverse_figs.py."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from aerosynapse.thermal.viz.inverse_figs import AXIS, GRID, INK, INK2, SURF, style  # noqa: F401  (style re-exported)

IN, OUT = "#2a78d6", "#eb6834"  # inside / outside the training range
IND = {
    "rms_sd": ("ensemble spread", "#2a78d6"),
    "residual": ("physics residual", "#1baf7a"),
    "combined": ("combined", "#52514e"),
}
MCD = ["#eb6834", "#b04a1e"]
SEQ = ["#9ec5f0", "#5a9be3", "#2a78d6", "#1a4f99", "#0d2d5c"]  # sequential blues (light -> dark), fixed order


def _ind_items(r: dict) -> list[tuple[str, str, str]]:
    out = [(k, *IND[k]) for k in ("rms_sd", "residual", "combined") if k in r["scores"]]
    for i, k in enumerate(k for k in r["scores"] if k.startswith("sd_mcd")):
        out.append((k, f"MC-dropout spread (p {k.split('_')[2]})", MCD[i % len(MCD)]))
    return out


def _narrow_log_x(ax) -> None:
    """A log x-axis spanning less than a decade: plain tick labels at 1, 1.2, 1.5, 2, ... x 10^n, no minor labels."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    lo, hi = ax.get_xlim()
    if lo > 0 and hi / lo < 10:
        ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 7.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())


def _shade_range(ax, r):
    ax.axvspan(r["range"][0], r["range"][1], color=GRID, alpha=0.6, lw=0, zorder=0)


def extrapolation(par: dict) -> plt.Figure:
    """rows = problems: (a) error of the ensemble mean (line) and the member range (band) vs the parameter, training
    range shaded; (b) indicators / their in-range threshold (1 = threshold); (c) coverage of the raw 95 % band."""
    probs = list(par)
    fig, axs = plt.subplots(len(probs), 3, figsize=(12.5, 3.4 * len(probs)), squeeze=False)
    for i, p in enumerate(probs):
        r = par[p]
        v = r["values"]
        x = np.asarray(v["value"])
        ax = axs[i][0]
        _shade_range(ax, r)
        ax.fill_between(x, v["member_rel_l2_min"], v["member_rel_l2_max"], color=IN, alpha=0.2, lw=0)
        ax.plot(x, v["rel_l2"], "-o", color=IN, ms=3, lw=1.4, mec=SURF)
        for t in next(iter(r["scores"].values()))["detection"]:
            ax.axhline(t["tol"], color=AXIS, lw=0.8, ls=(0, (2, 2)))
        ax.set(xscale="log", yscale="log", xlabel=r["param_name"], ylabel=f"{p}: relative L2 error")
        ax = axs[i][1]
        _shade_range(ax, r)
        for k, _lab, col in _ind_items(r):
            s = np.asarray(v[k]) if k in v else None
            if k == "combined":
                s = np.sqrt(
                    np.asarray(v["rms_sd"])
                    / r["scores"]["rms_sd"]["threshold"]
                    * np.asarray(v["residual"])
                    / r["scores"]["residual"]["threshold"]
                )
            ax.plot(x, s / r["scores"][k]["threshold"], color=col, lw=1.3)
        ax.axhline(1.0, color=INK2, lw=0.9)
        ax.set(xscale="log", yscale="log", xlabel=r["param_name"], ylabel="indicator / in-range threshold")
        ax = axs[i][2]
        _shade_range(ax, r)
        ax.plot(x, v["coverage_95"], "-o", color=IN, ms=3, lw=1.3, mec=SURF)
        ax.axhline(0.95, color=AXIS, lw=0.9)
        ax.set(xscale="log", ylim=(-0.02, 1.02), xlabel=r["param_name"], ylabel="coverage of the 95 % band")
    hs = [
        plt.Line2D([], [], color=IN, marker="o", ms=4, label="ensemble mean (band: member range)"),
        plt.Line2D([], [], color=GRID, lw=8, label="training range"),
        plt.Line2D([], [], color=AXIS, ls=(0, (2, 2)), label="error tolerances"),
    ]
    hs += [plt.Line2D([], [], color=c, lw=1.5, label=lab) for _, lab, c in _ind_items(par[probs[0]])]
    fig.legend(handles=hs, loc="outside lower center", ncol=4, fontsize=8)
    return fig


def indicators(par: dict) -> plt.Figure:
    """rows = problems, columns = indicators: indicator / threshold vs error, inside (blue) and outside (orange) the
    training range; the vertical line is the flag threshold, horizontal lines the error tolerances."""
    probs = list(par)
    items = _ind_items(par[probs[0]])
    fig, axs = plt.subplots(len(probs), len(items), figsize=(3.1 * len(items), 3.0 * len(probs)), squeeze=False)
    for i, p in enumerate(probs):
        r = par[p]
        v, d = r["values"], np.asarray(r["distance"])
        e = np.asarray(v["rel_l2"])
        for j, (k, lab, _) in enumerate(_ind_items(r)):
            ax = axs[i][j]
            if k == "combined":
                s = np.sqrt(
                    np.asarray(v["rms_sd"])
                    / r["scores"]["rms_sd"]["threshold"]
                    * np.asarray(v["residual"])
                    / r["scores"]["residual"]["threshold"]
                )
            else:
                s = np.asarray(v[k])
            s = s / r["scores"][k]["threshold"]
            ax.plot(s[d == 0], e[d == 0], "o", color=IN, ms=4, mec=SURF, mew=0.5)
            ax.plot(s[d > 0], e[d > 0], "o", color=OUT, ms=4, mec=SURF, mew=0.5)
            ax.axvline(1.0, color=INK2, lw=0.9)
            for t in r["scores"][k]["detection"]:
                ax.axhline(t["tol"], color=AXIS, lw=0.8, ls=(0, (2, 2)))
            sc = r["scores"][k]
            ax.set_title(f"{lab}\nSpearman {sc['spearman']:.2f}", loc="left", fontsize=8, color=INK2)
            ax.set(xscale="log", yscale="log", xlabel="indicator / threshold")
            _narrow_log_x(ax)
            if j == 0:
                ax.set_ylabel(f"{p}: relative L2 error")
    hs = [
        plt.Line2D([], [], color=IN, marker="o", ls="", ms=5, label="inside the training range"),
        plt.Line2D([], [], color=OUT, marker="o", ls="", ms=5, label="outside"),
        plt.Line2D([], [], color=INK2, label="flag threshold (in-range quantile)"),
        plt.Line2D([], [], color=AXIS, ls=(0, (2, 2)), label="error tolerances"),
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=4, fontsize=8)
    return fig


def time_extrapolation(par: dict) -> plt.Figure:
    """transient problems: per in-range parameter value (light -> dark = small -> large), (a) slice error, (b) slice
    spread / in-range threshold, (c) slice residual / in-range threshold vs t / T; the trained window ends at 1."""
    probs = [p for p in par if "time" in par[p]]
    fig, axs = plt.subplots(len(probs), 3, figsize=(12.0, 3.3 * len(probs)), squeeze=False)
    for i, p in enumerate(probs):
        tm = par[p]["time"]
        vals = sorted({r["value"] for r in tm["rows"]})
        for c, v in enumerate(vals):
            rows = [r for r in tm["rows"] if r["value"] == v]
            tt = [r["t_over_T"] for r in rows]
            col = (
                SEQ[min(c, len(SEQ) - 1)]
                if len(vals) <= len(SEQ)
                else plt.cm.Blues(0.35 + 0.6 * c / max(len(vals) - 1, 1))
            )
            axs[i][0].plot(tt, [r["rel_l2"] for r in rows], color=col, lw=1.3)
            axs[i][1].plot(
                tt, np.array([r["rms_sd"] for r in rows]) / tm["scores"]["rms_sd"]["threshold"], color=col, lw=1.3
            )
            axs[i][2].plot(
                tt, np.array([r["residual"] for r in rows]) / tm["scores"]["residual"]["threshold"], color=col, lw=1.3
            )
        for j, lab in enumerate(("relative L2 error (slice)", "spread / threshold", "residual / threshold")):
            ax = axs[i][j]
            ax.axvspan(0, 1, color=GRID, alpha=0.6, lw=0, zorder=0)
            if j:
                ax.axhline(1.0, color=INK2, lw=0.9)
            ax.set(yscale="log", xlabel="t / T", ylabel=f"{p}: {lab}" if j == 0 else lab)
    hs = [
        plt.Line2D([], [], color=SEQ[2], lw=1.5, label="one line per in-range parameter value (darker = larger)"),
        plt.Line2D([], [], color=GRID, lw=8, label="trained time window"),
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=2, fontsize=8)
    return fig


def cavity(cav: dict) -> plt.Figure:
    """lid-driven cavity: (a) psi_min vs Re per grid, (b) growth rate of the rightmost eigenvalue vs Re per grid;
    hatched = Re >= label_from_Re ('no laminar-validity claim')."""
    Ns = sorted(cav["grids"], key=int)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), squeeze=False)
    lab = cav["label_from_Re"]
    for k, n in enumerate(Ns):
        recs = [r for r in cav["grids"][n]["records"] if r["converged"]]
        col = SEQ[1 + min(k, 3)] if len(Ns) <= 4 else SEQ[min(k, 4)]
        Re = [r["Re"] for r in recs]
        axs[0][0].plot(Re, [r["psi_min"] for r in recs], "-o", color=col, ms=4, mec=SURF, lw=1.3, label=f"N = {n}")
        axs[0][1].plot(Re, [r["stability"]["growth_rate"] for r in recs], "-o", color=col, ms=4, mec=SURF, lw=1.3)
    for ax in axs[0]:
        lo, hi = ax.get_xlim()
        ax.axvspan(lab, max(hi, lab * 1.01), facecolor="none", edgecolor=AXIS, hatch="////", lw=0, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel("Re")
    axs[0][1].axhline(0.0, color=INK2, lw=0.9)
    axs[0][0].set_ylabel(r"$\psi_{min}$ (steady solution)")
    axs[0][1].set_ylabel("growth rate of the rightmost eigenvalue")
    hs, _ = axs[0][0].get_legend_handles_labels()
    hs.append(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="none",
            edgecolor=AXIS,
            hatch="////",
            label=f"Re ≥ {lab:g}: no laminar-validity claim",
        )
    )
    fig.legend(handles=hs, loc="outside lower center", ncol=len(hs), fontsize=8)
    return fig
