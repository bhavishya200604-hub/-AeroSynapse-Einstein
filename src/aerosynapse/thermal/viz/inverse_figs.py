"""Phase 5 figures (pure functions: data in, Figure out).

Style follows the project's data-visualisation rules: one fixed colour per parameter (alpha / k / h = categorical
slots 1-3 of the reference palette, validated all-pairs in its documentation: blue, orange, aqua), designs compared as
emphasis (optimised = blue, base = muted grey), solid hairline grids, thin lines, >= 8 px markers, legends plus
selective direct labels in text ink (never in the series colour), small multiples instead of crowded panels, log
axes where values span decades.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from aerosynapse.thermal.viz.style import AXIS, GRID, INK, INK2, MUTED, PALETTE, SURF, apply_style  # noqa: E402  (project style, 7.1)

PARAM_COLOR = {"alpha": PALETTE[0], "k": PALETTE[1], "h": PALETTE[2]}
PARAM_LABEL = {"alpha": "α", "k": "k", "h": "h"}
EMPH, BASE = PALETTE[0], MUTED
BAND = {"well": "#f3f8f3", "weak": "#fbf6ea", "not": "#fbeeee"}


def style() -> None:
    """The project publication style (src/viz/style.py; Phase 7, Step 7.1)."""
    apply_style()


def _end_label(ax, x, y, text: str, color: str, dx: float = 4) -> None:
    """Direct label at a line end: a coloured dot beside text in ink (text never wears the series colour)."""
    ax.plot([x], [y], "o", color=color, ms=4.5, mec=SURF, mew=1.2, zorder=5, clip_on=False)
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)


def _spread(ys: list[float], gap: float) -> list[float]:
    """Label y positions nudged apart by at least `gap`, keeping their order (centred on the original mean)."""
    order = np.argsort(ys)
    out = np.asarray(ys, float)[order].copy()
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + gap)
    out += np.mean(ys) - out.mean()
    res = np.empty_like(out)
    res[order] = out
    return res.tolist()


def sensitivities(panels: list[dict], params: list[str]) -> plt.Figure:
    """panels: [{title, t [s], J (n_t, n_x, n_p), x_frac, sigma_K}] -> one panel per scenario; per parameter the
    sensitivity at the surface-most sensor (solid) and the centre-most sensor (light); grey band = +-1 noise sigma."""
    fig, axs = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.3), squeeze=False, sharey=True)
    for ax, pnl in zip(axs[0], panels, strict=True):
        t, J = np.asarray(pnl["t"]), np.asarray(pnl["J"])
        ax.axhspan(-pnl["sigma_K"], pnl["sigma_K"], color="#efeee9", lw=0, zorder=0)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        ends = []
        for j, k in enumerate(params):
            c = PARAM_COLOR[k]
            ax.plot(t, J[:, 0, j], color=c, lw=1.0, alpha=0.45)
            ax.plot(t, J[:, -1, j], color=c, lw=1.8)
            ax.plot([t[-1]], [J[-1, -1, j]], "o", color=c, ms=4.5, mec=SURF, mew=1.2, zorder=5)
            ends.append(float(J[-1, -1, j]))
        rng = float(np.ptp(J[:, -1, :])) or 1.0
        for k, y in zip(params, _spread(ends, 0.07 * rng), strict=True):  # labels nudged apart where curves meet
            ax.annotate(
                PARAM_LABEL[k],
                (t[-1], y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=INK2,
            )
        ax.set(xlabel="time t [s]", title=pnl["title"])
    axs[0][0].set_ylabel("sensitivity ∂T/∂ln p [K]")
    h1 = plt.Line2D([], [], color=INK2, lw=1.8, label=f"sensor at r/L = {panels[0]['x_frac'][-1]:.2f}")
    h2 = plt.Line2D([], [], color=INK2, lw=1.0, alpha=0.45, label=f"sensor at r/L = {panels[0]['x_frac'][0]:.2f}")
    sig = ", ".join(f"{p['sigma_K']:.2f}" for p in panels)
    h3 = plt.Rectangle((0, 0), 1, 1, color="#efeee9", label=f"±1 noise σ ({sig} K)")
    axs[0][0].legend(handles=[h1, h2, h3], loc="lower left")
    return fig


def crlb_sweeps(sweeps: dict[str, dict], params: list[str], thresholds: list[float], level: float) -> plt.Figure:
    """sweeps: name -> {x: [...], xlabel, log_x (True | False | 'symlog'), title, crlb: {param: [...]}} (inf =
    structurally not identifiable, drawn as an arrow at the top). Small multiples on a 3-column grid; the spare
    cell carries the legend and the band key. Bands: well < thr0 <= weak < thr1 <= not identifiable."""
    names = list(sweeps)
    ncol = 3
    nrow = int(np.ceil((len(names) + 1) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(10.2, 2.75 * nrow), squeeze=False, sharey=True)
    fin = [v for sw in sweeps.values() for k in params for v in sw["crlb"][k] if np.isfinite(v)]
    bottom = 10 ** np.floor(np.log10(max(min(fin) * 0.6, 1e-3)))
    top = 10 ** np.ceil(np.log10(max(max(fin) * 1.6, thresholds[1] * 2)))
    for idx, n in enumerate(names):
        ax = axs[idx // ncol][idx % ncol]
        sw = sweeps[n]
        ax.set_yscale("log")
        ax.axhspan(bottom, thresholds[0], color=BAND["well"], lw=0, zorder=0)
        ax.axhspan(thresholds[0], thresholds[1], color=BAND["weak"], lw=0, zorder=0)
        ax.axhspan(thresholds[1], top, color=BAND["not"], lw=0, zorder=0)
        cat = isinstance(sw["x"][0], str)
        xs = np.arange(len(sw["x"])) if cat else np.asarray(sw["x"], float)
        inf_at = {}
        for k in params:
            y = np.asarray(sw["crlb"][k], float)
            ok = np.isfinite(y)
            ax.plot(
                xs[ok], y[ok], "-" if not cat else "", marker="o", color=PARAM_COLOR[k], mec=SURF, mew=1.0, zorder=3
            )
            for xi in xs[~ok]:
                inf_at.setdefault(float(xi), []).append(PARAM_LABEL[k])
        for xi, ks in inf_at.items():
            ax.annotate(
                "",
                (xi, top * 0.92),
                xytext=(xi, top * 0.35),
                arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 0.9},
            )
            ax.annotate(
                f"{', '.join(ks)}: ∞",
                (xi, top * 0.3),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=7,
                color=INK2,
                va="center",
            )
        if sw.get("base") is not None:  # the base design's value of the swept setting
            bx = sw["x"].index(sw["base"]) if cat else float(sw["base"])
            ax.axvline(bx, color=AXIS, lw=0.9, zorder=1)
            ax.annotate("base", (bx, bottom), xytext=(3, 3), textcoords="offset points", fontsize=7, color=MUTED)
        if cat:
            ax.set_xticks(xs, [s.replace("_", "\n") for s in sw["x"]])
            ax.set_xlim(-0.5, len(xs) - 0.5)
            ax.grid(axis="x", visible=False)
        elif sw.get("log_x") == "symlog":
            pos = sorted(v for v in xs if v > 0)
            ax.set_xscale("symlog", linthresh=pos[0], linscale=0.6)
            ax.set_xlim(-0.4 * pos[0], pos[-1] * 2.0)
            ax.set_xticks([0.0] + pos, ["0"] + [f"$10^{{{int(round(np.log10(v)))}}}$" for v in pos])
            ax.minorticks_off()
        elif sw.get("log_x"):
            ax.set_xscale("log")
            ax.set_xticks(xs, [f"{v:g}" for v in xs])
            ax.minorticks_off()
        ax.set(xlabel=sw["xlabel"], title=sw.get("title", n), ylim=(bottom, top))
        if idx % ncol == 0:
            ax.set_ylabel(f"CRLB [% of p], noise {100 * level:g} %")
    for idx in range(len(names), nrow * ncol):  # spare cells: legend + band key
        ax = axs[idx // ncol][idx % ncol]
        ax.axis("off")
        if idx == len(names):
            hs = [
                plt.Line2D([], [], color=PARAM_COLOR[k], marker="o", mec=SURF, label=f"{PARAM_LABEL[k]}")
                for k in params
            ]
            hs += [
                plt.Rectangle((0, 0), 1, 1, color=BAND[b], label=lab)
                for b, lab in (
                    ("well", f"well identifiable (< {thresholds[0]:g} %)"),
                    ("weak", f"weak ({thresholds[0]:g}-{thresholds[1]:g} %)"),
                    ("not", f"practically not identifiable (> {thresholds[1]:g} %)"),
                )
            ]
            ax.legend(handles=hs, loc="center left", fontsize=8)
    return fig


def _ellipse(ax, mean, cov, color, lw, label=None, nsig=2.4477):
    """95% confidence ellipse of a 2D Gaussian (chi2_2(0.95) = 5.991 -> 2.4477 sigma)."""
    w, v = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax.add_patch(
        Ellipse(
            mean,
            2 * nsig * np.sqrt(w[1]),
            2 * nsig * np.sqrt(w[0]),
            angle=ang,
            fill=False,
            ec=color,
            lw=lw,
            label=label,
        )
    )


def confidence_ellipses(designs: list[dict], params: list[str]) -> plt.Figure:
    """designs: [{name, color, cov (log-params, F^-1), samples (R, n_p) log-estimate minus truth or None}] -> one panel
    per parameter pair: 95% CRLB ellipse and the Monte-Carlo least-squares estimates (in % of the parameter)."""
    pairs = [(i, j) for i in range(len(params)) for j in range(i + 1, len(params))]
    fig, axs = plt.subplots(1, len(pairs), figsize=(3.3 * len(pairs), 3.2), squeeze=False)
    for ax, (i, j) in zip(axs[0], pairs, strict=True):
        lim = 0.0
        for d in designs:
            C = 1e4 * np.asarray(d["cov"])[np.ix_([i, j], [i, j])]  # (100 ln p)^2 ~ %^2
            if d.get("samples") is not None:
                S = 100 * np.asarray(d["samples"])
                ax.plot(S[:, i], S[:, j], "o", ms=2.6, color=d["color"], alpha=0.35, mew=0, zorder=2)
                lim = max(lim, np.abs(S[:, [i, j]]).max())
            _ellipse(ax, (0, 0), C, d["color"], 1.8, d["name"])
            lim = max(lim, 2.6 * np.sqrt(np.diag(C)).max())
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.axvline(0, color=AXIS, lw=0.8)
        lim *= 1.08
        ax.set(
            xlim=(-lim, lim),
            ylim=(-lim, lim),
            xlabel=f"{PARAM_LABEL[params[i]]} error [%]",
            ylabel=f"{PARAM_LABEL[params[j]]} error [%]",
            title=f"{PARAM_LABEL[params[i]]} vs {PARAM_LABEL[params[j]]}",
        )
        ax.set_aspect("equal")
    axs[0][0].legend(loc="upper left")
    return fig


def design_comparison(base: dict, opt: dict, params: list[str], L: float) -> plt.Figure:
    """base / opt: {positions (r/L), times [s], crlb {param: %}} -> sensor positions, measurement times, bounds."""
    fig = plt.figure(figsize=(9.6, 3.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0])
    for row, (key, xl) in enumerate(
        (("positions", "sensor position r/L  (0 = centre, 1 = surface)"), ("times", "measurement time t [s]"))
    ):
        ax = fig.add_subplot(gs[row, 0])
        for y, d, c in ((1, opt, EMPH), (0, base, BASE)):
            v = np.asarray(d[key], float)
            u, cnt = np.unique(np.round(v, 2 if key == "positions" else 4), return_counts=True)
            ax.scatter(u, np.full(len(u), y), s=18 + 22 * (cnt - 1), color=c, edgecolor=SURF, linewidth=1.0, zorder=3)
            for ui, ci in zip(u, cnt, strict=True):
                if ci > 1:
                    ax.annotate(
                        f"×{ci}",
                        (ui, y),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                        color=INK2,
                    )
        ax.set_yticks([0, 1], ["base", "optimised"])
        ax.set_ylim(-0.6, 1.8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xl)
        if key == "positions":
            ax.set_xlim(-0.05, 1.05)
            ax.set_title("measurement design (same number of readings)")
    ax = fig.add_subplot(gs[:, 1])
    xs = np.arange(len(params))
    wdt = 0.36
    for off, d, c, n in ((-wdt / 2, base, BASE, "base"), (wdt / 2, opt, EMPH, "optimised")):
        vals = [d["crlb"][k] for k in params]
        ax.bar(xs + off, vals, wdt * 0.92, color=c, label=n, zorder=3)
        for x, v in zip(xs + off, vals, strict=True):
            ax.annotate(
                f"{v:.2g}", (x, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=INK2
            )
    ax.set_xticks(xs, [PARAM_LABEL[k] for k in params])
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="Cramér–Rao bound [%]", title="bounds at 2 % noise")
    ax.legend(loc="upper right")
    return fig


def convergence(
    histories: dict[str, list[list[dict]]],
    params: list[str],
    truth: dict,
    crlb: dict | None,
    xlabel: str = "logged step",
    ref: str = "true",
) -> plt.Figure:
    """histories: variant -> list (seeds) of histories [{step index, param values}] -> small multiples
    (rows = parameters, columns = variants): estimate / truth on a log axis, seeds as thin lines, +-1 CRLB band."""
    names = list(histories)
    fig, axs = plt.subplots(
        len(params),
        len(names),
        figsize=(2.3 * len(names), 1.9 * len(params)),
        squeeze=False,
        sharex="col",
        sharey="row",
    )
    for c, n in enumerate(names):
        for r, k in enumerate(params):
            ax = axs[r][c]
            if crlb:
                ax.axhspan(1 - crlb[k] / 100, 1 + crlb[k] / 100, color="#efeee9", lw=0, zorder=0)
            ax.axhline(1.0, color=AXIS, lw=0.9)
            for h in histories[n]:
                y = np.array([x[k] / truth[k] for x in h])
                ax.plot(np.arange(len(y)), y, color=PARAM_COLOR[k], lw=1.1, alpha=0.8)
            ax.set_yscale("log")
            if r == 0:
                ax.set_title(n.replace("_", " "))
            if c == 0:
                ax.set_ylabel(f"{PARAM_LABEL[k]} / {PARAM_LABEL[k]}_{ref}")
            if r == len(params) - 1:
                ax.set_xlabel(xlabel)
    return fig


def error_dots(rows: list[dict], params: list[str], crlb: dict | None) -> plt.Figure:
    """rows: [{name, errors: {param: [% per seed]}, optional ci: {param: [lo %, hi %]}}] -> one panel per parameter: a
    dot per seed, median tick, an interval bar where given, +-1 and +-2 CRLB bands; symlog y (linear within +-10 %)."""
    fig, axs = plt.subplots(1, len(params), figsize=(3.4 * len(params), 3.2), squeeze=False)
    for ax, k in zip(axs[0], params, strict=True):
        if crlb:
            ax.axhspan(-2 * crlb[k], 2 * crlb[k], color="#f4f3ef", lw=0, zorder=0)
            ax.axhspan(-crlb[k], crlb[k], color="#e9e8e2", lw=0, zorder=0)
        ax.axhline(0, color=AXIS, lw=0.8)
        for x, rw in enumerate(rows):
            e = np.asarray(rw["errors"][k], float)
            if rw.get("ci"):
                lo, hi = np.clip(rw["ci"][k], -1e6, 1e6)
                ax.plot([x, x], [lo, hi], color=INK2, lw=1.0, zorder=2)
            ax.plot(np.full(len(e), x), e, "o", color=PARAM_COLOR[k], mec=SURF, mew=1.0, ms=5.5, zorder=3)
            ax.plot([x - 0.25, x + 0.25], [np.median(e)] * 2, color=INK2, lw=1.2, zorder=4)
        ax.set_yscale("symlog", linthresh=10.0)
        ax.set_xticks(range(len(rows)), [r["name"].replace("_", "\n") for r in rows], fontsize=7)
        ax.grid(axis="x", visible=False)
        ax.set(
            ylabel=f"{PARAM_LABEL[k]} error [%]", title=f"{PARAM_LABEL[k]}" + ("  (bands: ±1, ±2 CRLB)" if crlb else "")
        )
    return fig


def group_consistency(rows: list[dict]) -> plt.Figure:
    """rows: [{variant, Bi_ratio, Q_ratio, pinn_rmse, model_rmse (one value per seed each), sigma}] ->
    three panels: estimated / true Bi, estimated / true Q (log axes), and the data misfit of the PINN field vs the exact
    model evaluated at the PINN's own estimates (a consistent estimate has both near the noise sigma)."""
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.3))
    names = [r["variant"].replace("_", "\n") for r in rows]
    xs = np.arange(len(rows))
    for ax, key, title in (
        (axs[0], "Bi_ratio", "Biot number Bi = hL/k"),
        (axs[1], "Q_ratio", "source group Q = q L²/(k ΔT)"),
    ):
        ax.axhline(1.0, color=AXIS, lw=0.9)
        for x, r in zip(xs, rows, strict=True):
            v = np.asarray(r[key], float)
            ax.plot(np.full(len(v), x), v, "o", color=EMPH, mec=SURF, mew=1.0, ms=5.5, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(xs, names, fontsize=7)
        ax.grid(axis="x", visible=False)
        ax.set(ylabel="estimate / truth", title=title)
    ax = axs[2]
    for x, r in zip(xs, rows, strict=True):
        ax.plot(
            np.full(len(r["pinn_rmse"]), x - 0.12), r["pinn_rmse"], "o", color=EMPH, mec=SURF, mew=1.0, ms=5.5, zorder=3
        )
        ax.plot(
            np.full(len(r["model_rmse"]), x + 0.12),
            r["model_rmse"],
            "o",
            color="#eb6834",
            mec=SURF,
            mew=1.0,
            ms=5.5,
            zorder=3,
        )
    ax.axhline(rows[0]["sigma"], color=AXIS, lw=0.9)
    ax.annotate(
        "noise σ",
        (len(rows) - 0.5, rows[0]["sigma"]),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        fontsize=7,
        color=MUTED,
    )
    ax.set_xticks(xs, names, fontsize=7)
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="data misfit RMSE [K]", title="misfit: PINN field vs exact model at its estimates", ylim=(0, None))
    ax.legend(
        handles=[
            plt.Line2D([], [], ls="", marker="o", color=EMPH, label="PINN field"),
            plt.Line2D([], [], ls="", marker="o", color="#eb6834", label="exact model at the PINN's estimates"),
        ],
        loc="upper left",
    )
    return fig


def multistart_kh(cases: dict[str, dict], truth: dict, ref: str = "true") -> plt.Figure:
    """cases: name -> {starts, estimates: [param dicts], chi2_per_reading, n_at_best} -> one panel per case on log
    axes of k / k_true and h / h_true: each start (hollow) joined to its converged estimate (filled), the truth, and
    the line of constant Biot number h/k through the truth (the direction least squares cannot resolve without a
    source)."""
    names = list(cases)
    fig, axs = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.5), squeeze=False)
    for ax, n in zip(axs[0], names, strict=True):  # own axis range per case: the no-source valley spans decades
        d = cases[n]
        lim = [1.0, 1.0]
        at_best = d.get("at_best") or [True] * len(d["starts"])
        for s, e, b in zip(d["starts"], d["estimates"], at_best, strict=True):
            xs, ys = [s["k"] / truth["k"]], [s["h"] / truth["h"]]
            ax.plot(xs[0], ys[0], "o", ms=5, mfc=SURF, mec=MUTED, mew=1.0, zorder=2)
            if e is not None:  # None: numerical failure, start only
                xs.append(e["k"] / truth["k"])
                ys.append(e["h"] / truth["h"])
                ax.plot(xs, ys, color=GRID, lw=0.9, zorder=1)
                if b:
                    ax.plot(xs[1], ys[1], "o", ms=6, color=EMPH, mec=SURF, mew=1.0, zorder=3)
                else:
                    ax.plot(xs[1], ys[1], "X", ms=7, color=INK2, mec=SURF, mew=0.8, zorder=3)
            lim = [min(lim[0], *xs, *ys), max(lim[1], *xs, *ys)]
        ax.plot(1, 1, "+", ms=11, mew=1.6, color=INK, zorder=4)
        lo, hi = lim[0] / 1.5, lim[1] * 1.5
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=0)
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel=f"k / k_{ref}")
        ax.set_aspect("equal")
        ax.set_title(f"{n}\n{d['n_at_best']}/{len(d['estimates'])} starts reach the best fit", fontsize=9)
        ax.annotate(
            "constant Bi",
            (hi / 1.8, hi / 1.8),
            xytext=(-4, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7,
            color=MUTED,
            rotation=45,
            rotation_mode="anchor",
        )
    axs[0][0].set_ylabel(f"h / h_{ref}")
    hs = [
        plt.Line2D([], [], marker="o", ls="", ms=5, mfc=SURF, mec=MUTED, label="start"),
        plt.Line2D([], [], marker="o", ls="", ms=6, color=EMPH, mec=SURF, label="estimate at the best fit"),
        plt.Line2D([], [], marker="X", ls="", ms=7, color=INK2, mec=SURF, label="stopped elsewhere (higher χ²)"),
        plt.Line2D(
            [], [], marker="+", ls="", ms=11, mew=1.6, color=INK, label="truth" if ref == "true" else "reference"
        ),
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=4, fontsize=8)
    return fig


def noise_robustness(designs: dict[str, dict], pinn: dict | None, params: list[str]) -> plt.Figure:
    """designs: name -> {levels [%], crlb {param: [%]}, std {param: [%]} (Monte-Carlo sample std), floor {param: |error|
    % of the noise-free FDM fit} or None}; pinn: {levels [%], err {param: [|error| %]}} or None -> one panel per
    parameter: error vs noise level on log axes; the noise-free point sits at the left edge (symlog x)."""
    fig, axs = plt.subplots(1, len(params), figsize=(3.6 * len(params), 3.4), squeeze=False, sharey=True)
    names = list(designs)
    for ax, k in zip(axs[0], params, strict=True):
        for i, n in enumerate(names):
            d = designs[n]
            c = EMPH if "optimised" in n else BASE
            ax.plot(d["levels"], d["crlb"][k], color=c, lw=1.2, ls=(0, (4, 3)), zorder=2)
            ax.plot(d["levels"], d["std"][k], "o", color=c, mec=SURF, mew=1.0, ms=6.5, zorder=3)
            if d.get("floor"):
                ax.plot(0, max(d["floor"][k], 1e-4), "s", color=c, mec=SURF, mew=1.0, ms=6.5, zorder=3)
            if i == 0 and k == params[0]:
                ax.annotate(
                    "noise-free\nFDM fit",
                    (0, max(d["floor"][k], 1e-4) if d.get("floor") else 1e-3),
                    xytext=(6, -2),
                    textcoords="offset points",
                    fontsize=7,
                    color=INK2,
                    va="top",
                )
        if pinn:
            ax.plot(
                np.maximum(pinn["levels"], 0),
                np.maximum(pinn["err"][k], 1e-4),
                "D",
                color=INK2,
                mec=SURF,
                mew=1.0,
                ms=6,
                zorder=4,
            )
        ax.set_xscale("symlog", linthresh=1.0, linscale=0.5)
        ax.set_yscale("log")
        ax.set_xticks([0, 1, 2, 5, 10], ["0", "1", "2", "5", "10"])
        ax.set(xlim=(-0.3, 13), xlabel="noise level σ / RMS signal [%]", title=PARAM_LABEL[k])
        ax.grid(axis="x", visible=False)
    axs[0][0].set_ylabel("error [%]")
    hs = []
    for n in names:
        c = EMPH if "optimised" in n else BASE
        hs.append(plt.Line2D([], [], color=c, lw=1.2, ls=(0, (4, 3)), label=f"{n}: Cramér–Rao bound"))
        hs.append(plt.Line2D([], [], marker="o", ls="", color=c, mec=SURF, ms=6.5, label=f"{n}: least-squares std"))
    if pinn:
        hs.append(plt.Line2D([], [], marker="D", ls="", color=INK2, mec=SURF, ms=6, label="PINN |error| (base)"))
    fig.legend(handles=hs, loc="outside lower center", ncol=min(len(hs), 5), fontsize=7.5)
    return fig


def sensor_studies(number: list[dict], placement: list[dict], params: list[str]) -> plt.Figure:
    """number / placement: [{label, crlb {param: %} or None, std {param: %} or None, rank}] -> left: bound (dashed
    line) and Monte-Carlo std (dots) vs the number of sensors (log-log); right: bound (ring) and std (dot) per
    placement (rows), log x. One shared legend below."""
    from matplotlib.ticker import FixedLocator, NullLocator

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1.0, 1.25]})
    if number:
        ns = [float(r["label"]) for r in number]
        for k in params:
            c = PARAM_COLOR[k]
            a1.plot(ns, [r["crlb"][k] if r["crlb"] else np.nan for r in number], color=c, lw=1.2, ls=(0, (4, 3)))
            a1.plot(ns, [r["std"][k] if r["std"] else np.nan for r in number], "o", color=c, mec=SURF, mew=1.0, ms=6.5)
        a1.set(
            xscale="log",
            yscale="log",
            xlabel="number of sensors (uniform, 60 times each)",
            ylabel="error [%]",
            title="sensor number (2 % noise)",
        )
        a1.xaxis.set_major_locator(FixedLocator(ns))
        a1.xaxis.set_minor_locator(NullLocator())
        a1.set_xticklabels([r["label"] for r in number])
    if placement:
        ys = np.arange(len(placement))[::-1]
        off = {k: ((len(params) - 1) / 2 - i) * 0.2 for i, k in enumerate(params)}  # first parameter on top
        vals = []
        for k in params:
            c = PARAM_COLOR[k]
            for y, r in zip(ys, placement, strict=True):
                if r["crlb"] and np.isfinite(r["crlb"][k]):
                    a2.plot(r["crlb"][k], y + off[k], "o", mfc=SURF, mec=c, mew=1.4, ms=7, zorder=2)
                    vals.append(r["crlb"][k])
                if r["std"]:
                    a2.plot(r["std"][k], y + off[k], "o", color=c, mec=SURF, mew=0.8, ms=5, zorder=3)
                    vals.append(r["std"][k])
        a2.set_yticks(
            ys, [r["label"] + ("" if r["rank"] == len(params) else f"  (rank {r['rank']})") for r in placement]
        )
        a2.set(xscale="log", xlabel="error [%]", title="placement (8 sensors, 2 % noise)")
        if vals:
            ticks = [
                t
                for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500)
                if min(vals) / 1.5 <= t <= max(vals) * 1.5
            ]
            a2.xaxis.set_major_locator(FixedLocator(ticks))
            a2.xaxis.set_minor_locator(NullLocator())
            a2.set_xticklabels([f"{t:g}" for t in ticks])
        a2.grid(axis="y", visible=False)
    hs = [
        plt.Line2D([], [], marker="o", ls="", color=PARAM_COLOR[k], mec=SURF, ms=6, label=PARAM_LABEL[k])
        for k in params
    ]
    hs += [
        plt.Line2D(
            [],
            [],
            color=MUTED,
            lw=1.2,
            ls=(0, (4, 3)),
            marker="o",
            mfc=SURF,
            mec=MUTED,
            mew=1.4,
            ms=7,
            label="Cramér–Rao bound (line / ring)",
        ),
        plt.Line2D([], [], marker="o", ls="", color=MUTED, mec=SURF, ms=5.5, label="least-squares std (Monte Carlo)"),
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=len(hs), fontsize=8)
    return fig


def calibration(sections: list[dict], params: list[str]) -> plt.Figure:
    """sections: [{title, labels, ratio {param: [std / CRLB]}, coverage {param: [%]}, ratio_band, coverage_band}] ->
    two rows (sample std / CRLB, target 1; 95 % interval coverage, target 95 %) x one column per study; shaded: the
    Monte-Carlo sampling uncertainty (+-1.96 x) around the target."""
    widths = [max(len(s["labels"]), 3) for s in sections]
    fig, axs = plt.subplots(
        2,
        len(sections),
        figsize=(2.0 + 0.55 * sum(widths), 5.0),
        squeeze=False,
        gridspec_kw={"width_ratios": widths},
        sharey="row",
    )
    off = {k: (i - (len(params) - 1) / 2) * 0.18 for i, k in enumerate(params)}
    for j, s in enumerate(sections):
        xs = np.arange(len(s["labels"]))
        for row, key, target, band in (
            (0, "ratio", 1.0, 1.96 * s["ratio_band"]),
            (1, "coverage", 95.0, s["coverage_band"]),
        ):
            ax = axs[row][j]
            ax.axhspan(target - band, target + band, color="#efeee9", lw=0, zorder=0)
            ax.axhline(target, color=AXIS, lw=0.9, zorder=1)
            for k in params:
                ax.plot(xs + off[k], s[key][k], "o", color=PARAM_COLOR[k], mec=SURF, mew=0.8, ms=5.5, zorder=3)
            ax.set_xticks(
                xs,
                s["labels"] if row == 1 else [""] * len(xs),
                fontsize=7,
                rotation=0 if max(map(len, s["labels"])) < 6 else 35,
                ha="center" if max(map(len, s["labels"])) < 6 else "right",
            )
            ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(s["title"], fontsize=9)
        axs[0][0].set_ylabel("sample std / CRLB")
        axs[1][0].set_ylabel("95 % interval coverage [%]")
    hs = [
        plt.Line2D([], [], marker="o", ls="", color=PARAM_COLOR[k], mec=SURF, ms=6, label=PARAM_LABEL[k])
        for k in params
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=len(params), fontsize=8)
    return fig


MCOL = {"gauss_newton": INK2, "bootstrap_parametric": "#eb6834", "bootstrap_residual": "#1baf7a", "posterior": EMPH}
MMARK = {"gauss_newton": "s", "bootstrap_parametric": "^", "bootstrap_residual": "v", "posterior": "o"}
QLABEL = {**PARAM_LABEL, "Bi": "Bi"}


def reliability(rel: dict[str, dict], quantities: list[str], labels: dict[str, str]) -> plt.Figure:
    """rel: case -> method -> {levels: [nominal], quantity: [empirical coverage]} -> rows = quantities, columns = cases:
    empirical vs nominal coverage (diagonal = calibrated; below = overconfident intervals)."""
    cases = list(rel)
    fig, axs = plt.subplots(
        len(quantities),
        len(cases),
        figsize=(3.0 * len(cases), 2.5 * len(quantities)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for j, c in enumerate(cases):
        for i, q in enumerate(quantities):
            ax = axs[i][j]
            ax.plot([0.4, 1.0], [0.4, 1.0], color=AXIS, lw=0.9, zorder=1)
            for m, d in rel[c].items():
                lv = np.asarray(d["levels"]) * 100
                ax.plot(
                    lv,
                    np.asarray(d[q]) * 100,
                    marker=MMARK[m],
                    color=MCOL[m],
                    mec=SURF,
                    mew=0.8,
                    ms=6,
                    lw=1.1,
                    zorder=3,
                )
            if i == 0:
                ax.set_title(c, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{QLABEL[q]}: empirical [%]")
            if i == len(quantities) - 1:
                ax.set_xlabel("nominal coverage [%]")
    for ax in axs.flat:
        ax.set(xlim=(45, 100), ylim=(None, 101))
        ax.plot([45, 100], [45, 100], color=AXIS, lw=0.9, zorder=1)
    hs = [plt.Line2D([], [], marker=MMARK[m], color=MCOL[m], mec=SURF, ms=6, lw=1.1, label=labels[m]) for m in MCOL]
    fig.legend(handles=hs, loc="outside lower center", ncol=len(hs), fontsize=8)
    return fig


def uq_intervals(iv: dict[str, dict], quantities: list[str], labels: dict[str, str], ref: str = "truth") -> plt.Figure:
    """iv: case -> {method: {"95": {q: [lo, hi]}, "68": {...} or None}, "_point": {q: x}} in ln(value / truth) ->
    rows = cases, columns = quantities: 95 % (thin) and 68 % (thick) intervals per method on a log ratio axis."""
    cases = list(iv)
    fig, axs = plt.subplots(
        len(cases), len(quantities), figsize=(2.9 * len(quantities), 1.7 + 0.45 * 4 * len(cases)), squeeze=False
    )
    methods = [m for m in MCOL if m in next(iter(iv.values()))]
    lim = 1e3
    for i, c in enumerate(cases):
        for j, q in enumerate(quantities):
            ax = axs[i][j]
            ax.axvline(1.0, color=AXIS, lw=0.9, zorder=1)
            for y, m in enumerate(methods[::-1]):
                d = iv[c][m]
                lo, hi = np.clip(np.exp(np.asarray(d["95"][q], float)), 1 / lim, lim)
                ax.plot([lo, hi], [y, y], color=MCOL[m], lw=1.4, solid_capstyle="butt", zorder=2)
                if d.get("68"):
                    lo, hi = np.clip(np.exp(np.asarray(d["68"][q], float)), 1 / lim, lim)
                    ax.plot([lo, hi], [y, y], color=MCOL[m], lw=4.5, solid_capstyle="butt", zorder=2)
            ax.plot(
                [np.exp(iv[c]["_point"][q])] * len(methods),
                range(len(methods)),
                "|",
                color=INK,
                ms=9,
                mew=1.4,
                zorder=3,
            )
            ax.set_xscale("log")
            _ratio_ticks(ax)
            ax.set_yticks(
                range(len(methods)), [labels[m] for m in methods[::-1]] if j == 0 else [""] * len(methods), fontsize=7.5
            )
            ax.grid(axis="y", visible=False)
            ax.set_ylim(-0.6, len(methods) - 0.4)
            if i == 0:
                ax.set_title(QLABEL[q])
            if j == len(quantities) - 1:
                ax.annotate(
                    c, (1.02, 0.5), xycoords="axes fraction", rotation=270, va="center", fontsize=8.5, color=INK2
                )
            if i == len(cases) - 1:
                ax.set_xlabel(f"value / {ref}")
    return fig


def posterior_pairs(pairs: dict[str, dict], params: list[str], ref: str = "true") -> plt.Figure:
    """pairs: case -> {posterior (n, p), bootstrap (m, p) samples of ln(p / truth), gn_cov (p, p), x_hat (p)} ->
    rows = cases, columns = parameter pairs: MCMC samples, parametric-bootstrap refits, Gauss-Newton 95 % ellipse."""
    cases = list(pairs)
    pp = [(a, b) for a in range(len(params)) for b in range(a + 1, len(params))]
    fig, axs = plt.subplots(len(cases), len(pp), figsize=(3.2 * len(pp), 3.1 * len(cases)), squeeze=False)
    for i, c in enumerate(cases):
        d = pairs[c]
        P, Bt = np.asarray(d["posterior"]), np.asarray(d["bootstrap"])
        for j, (a, b) in enumerate(pp):
            ax = axs[i][j]
            ax.plot(Bt[:, a], Bt[:, b], "o", ms=2.6, color=MCOL["bootstrap_parametric"], alpha=0.45, mew=0, zorder=2)
            ax.plot(P[:, a], P[:, b], "o", ms=1.8, color=MCOL["posterior"], alpha=0.25, mew=0, zorder=3)
            cov = np.asarray(d["gn_cov"])[np.ix_([a, b], [a, b])]
            if np.all(np.isfinite(cov)):
                _ellipse(ax, (d["x_hat"][a], d["x_hat"][b]), cov, MCOL["gauss_newton"], 1.4)
            ax.plot(0, 0, "+", ms=11, mew=1.6, color=INK, zorder=5)
            lo = np.percentile(np.r_[P[:, [a, b]].ravel(), Bt[:, [a, b]].ravel()], [0.5, 99.5])
            pad = 0.1 * (lo[1] - lo[0])
            ax.set(xlim=(lo[0] - pad, lo[1] + pad), ylim=(lo[0] - pad, lo[1] + pad))
            ax.set_xlabel(f"ln({PARAM_LABEL[params[a]]} / {ref})")
            ax.set_ylabel(f"ln({PARAM_LABEL[params[b]]} / {ref})")
            if j == 0:
                ax.set_title(c, fontsize=9)
    hs = [
        plt.Line2D([], [], marker="o", ls="", color=MCOL["posterior"], ms=5, label="posterior samples (MCMC)"),
        plt.Line2D(
            [], [], marker="o", ls="", color=MCOL["bootstrap_parametric"], ms=5, label="parametric bootstrap refits"
        ),
        plt.Line2D([], [], color=MCOL["gauss_newton"], lw=1.4, label="Gauss–Newton 95 % ellipse"),
        plt.Line2D([], [], marker="+", ls="", color=INK, ms=10, mew=1.6, label="truth"),
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=4, fontsize=8)
    return fig


def pinn_ensemble(p: dict, ls: dict[str, list[float]], ref: str = "truth") -> plt.Figure:
    """p: {members: [{seed, log_ratio}], ensemble: {mean_log_ratio, interval95_log_ratio}}; ls: q -> least-squares
    posterior 95 % interval in ln ratio -> one panel per quantity on a log ratio axis."""
    e = p["ensemble"]
    q = list(e["mean_log_ratio"])
    fig, axs = plt.subplots(1, len(q), figsize=(2.8 * len(q), 2.6), squeeze=False, sharey=True)
    for j, qq in enumerate(q):
        ax = axs[0][j]
        ax.axvline(1.0, color=AXIS, lw=0.9, zorder=1)
        x = np.exp([m["log_ratio"][j] for m in p["members"]])
        ax.plot(x, np.full(len(x), 2), "o", color=INK2, mec=SURF, mew=0.8, ms=6, zorder=3)
        lo, hi = np.exp(e["interval95_log_ratio"][qq])
        ax.plot([lo, hi], [1, 1], color=INK2, lw=1.6, zorder=2)
        ax.plot(np.exp(e["mean_log_ratio"][qq]), 1, "D", color=INK2, mec=SURF, ms=6, zorder=3)
        lo, hi = np.exp(ls[qq])
        ax.plot([lo, hi], [0, 0], color=EMPH, lw=1.6, zorder=2)
        ax.set_xscale("log")
        _ratio_ticks(ax)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, 2.6)
        ax.set(title=QLABEL[qq], xlabel=f"value / {ref}")
    axs[0][0].set_yticks(
        [0, 1, 2],
        ["least squares\n(posterior 95 %)", f"PINN ensemble\n(mean ± t·sd, {e['members']} seeds)", "PINN members"],
        fontsize=7.5,
    )
    for ax in axs[0][1:]:
        ax.tick_params(labelleft=False)
    return fig


def _ratio_ticks(ax, axis: str = "x") -> None:
    """Plain ratio labels on a log axis (x or y), from a ladder chosen by the span of the current limits."""
    from matplotlib.ticker import FixedLocator, NullLocator

    lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
    span = hi / lo
    if span > 50:
        ladder = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
    elif span > 4:
        ladder = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    elif span > 1.6:
        ladder = [0.25, 0.5, 0.7, 1.0, 1.4, 2.0, 4.0]
    elif span > 1.12:
        ladder = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25]
    else:
        ladder = [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06]
    ticks = [v for v in ladder if lo <= v <= hi]
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)([f"{v:g}" for v in ticks])


SEQ_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # sequential ramp (reference palette, steps 250-650)


def profiles(cases: list[dict], ref: str = "true") -> plt.Figure:
    """cases: [{label, unknown, profiles: {param: {x: ln(p / reference), d: delta chi2}}}] -> one panel per parameter:
    every case's profile (q_vol cases on a sequential blue ramp, others in ink), the 95 % threshold 3.84."""
    params = list(dict.fromkeys(k for c in cases for k in c["unknown"]))
    fig, axs = plt.subplots(1, len(params), figsize=(3.6 * len(params), 3.5), squeeze=False, sharey=True)
    q_cases = [c for c in cases if c["label"].startswith("q_vol")]
    style = {}
    for i, c in enumerate(q_cases):
        style[c["label"]] = dict(color=SEQ_BLUE[min(i, len(SEQ_BLUE) - 1)], lw=1.4, ls="-")
    other_ls = [(0, (5, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2))]
    j = 0
    for c in cases:
        if c["label"] not in style:
            style[c["label"]] = dict(
                color=INK if c["label"] == "configured" else INK2,
                lw=1.6 if c["label"] == "configured" else 1.2,
                ls="-" if c["label"] == "configured" else other_ls[j % len(other_ls)],
            )
            j += c["label"] != "configured"
    for ax, k in zip(axs[0], params, strict=True):
        ax.axhline(DCHI2, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
        for c in cases:
            if k not in c["profiles"]:
                continue
            x = np.exp(np.asarray(c["profiles"][k]["x"], float))
            y = np.clip(np.asarray(c["profiles"][k]["d"], float), 1e-3, 1e4)
            ax.plot(x, y, zorder=3, **style[c["label"]])
        ax.set(xscale="log", yscale="symlog", title=PARAM_LABEL[k], xlabel=f"{PARAM_LABEL[k]} / {PARAM_LABEL[k]}_{ref}")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylim(-0.1, 1.2e4)
        _ratio_ticks(ax)
    axs[0][0].set_ylabel("Δχ² (profile likelihood)")
    axs[0][0].annotate(
        "95 %: Δχ² = 3.84",
        (0.02, DCHI2),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=7,
        color=MUTED,
    )
    hs = [plt.Line2D([], [], label=c["label"], **style[c["label"]]) for c in cases]
    fig.legend(handles=hs, loc="outside lower center", ncol=min(len(hs), 5), fontsize=7.5)
    return fig


DCHI2 = 3.841458820694124


def identifiability_map(sweeps: dict[str, dict]) -> plt.Figure:
    """sweeps: name -> {x (Q or Bi), xlabel, crlb {param: [%]}, profile {param: [half-width %, inf = open]}, bi [1σ %]}
    -> one panel per sweep: Cramér–Rao bound (dashed), profile-likelihood half-width (dots; open intervals as
    arrows at the top), and the Biot-number bound (grey)."""
    names = list(sweeps)
    fig, axs = plt.subplots(1, len(names), figsize=(4.8 * len(names), 3.7), squeeze=False)
    top = 1e3
    for ax, n in zip(axs[0], names, strict=True):
        s = sweeps[n]
        x = np.asarray(s["x"], float)
        for k, c in s["crlb"].items():
            col = PARAM_COLOR[k]
            cr = np.asarray(c, float)
            ax.plot(
                x, np.where(np.isfinite(cr), np.minimum(cr, top), np.nan), color=col, lw=1.2, ls=(0, (4, 3)), zorder=2
            )
            pw = np.asarray(s["profile"][k], float)
            fin = np.isfinite(pw) & (pw < top)
            ax.plot(x[fin], pw[fin], "o", color=col, mec=SURF, mew=1.0, ms=6.5, zorder=3)
            ax.plot(
                x[~fin], np.full((~fin).sum(), top), "^", color=col, mec=SURF, mew=0.8, ms=8, zorder=3, clip_on=True
            )
        bi = np.asarray(s["bi"], float)
        ax.plot(x, bi, color=BASE, lw=1.0, marker="s", ms=4, zorder=2)
        if np.any(x == 0):
            ax.set_xscale("symlog", linthresh=float(np.min(x[x > 0])) / 2 if np.any(x > 0) else 1e-4)
        else:
            ax.set_xscale("log")
        ax.set(yscale="log", ylim=(0.1, top * 1.6), xlabel=s["xlabel"], ylabel="uncertainty [%]")
        ax.annotate(
            "open (unbounded) ▲",
            (0.98, top),
            xycoords=("axes fraction", "data"),
            xytext=(0, -10),
            textcoords="offset points",
            ha="right",
            fontsize=7,
            color=INK2,
        )
    hs = [
        plt.Line2D([], [], color=PARAM_COLOR[k], marker="o", ls="", ms=6, label=PARAM_LABEL[k])
        for k in ("alpha", "k", "h")
    ]
    hs += [
        plt.Line2D([], [], color=MUTED, lw=1.2, ls=(0, (4, 3)), label="Cramér–Rao bound (1σ)"),
        plt.Line2D([], [], color=MUTED, marker="o", ls="", ms=6, label="profile 95 % half-width"),
        plt.Line2D([], [], color=BASE, marker="s", ms=4, lw=1.0, label="Bi = hL/k (1σ)"),
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=6, fontsize=7.5)
    return fig


def prior_knowledge(known: list[dict], priors: list[dict]) -> plt.Figure:
    """known: case rows with unknown subsets (profile 95 % intervals); priors: rows with a posterior -> left: the
    profile intervals of the remaining unknowns; right: posterior 95 % intervals vs the prior width."""
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 3.9), gridspec_kw={"width_ratios": [1.2, 1.0]})
    ax = axs[0]
    lim = 100.0
    ys = np.arange(len(known))[::-1]
    for y, r in zip(ys, known, strict=True):
        for i, k in enumerate(r["unknown"]):
            lo, hi = r["profiles"][k]["ci95_log"]
            xt = np.log(r["parameters"][k])
            a = np.exp(lo - xt) if lo is not None else 1 / lim
            b = np.exp(hi - xt) if hi is not None else lim
            yy = y + (i - (len(r["unknown"]) - 1) / 2) * 0.22
            ax.plot([a, b], [yy, yy], color=PARAM_COLOR[k], lw=2.2, solid_capstyle="butt")
            for v, op in ((a, lo is None), (b, hi is None)):
                if op:
                    ax.plot(v, yy, "<" if v < 1 else ">", color=PARAM_COLOR[k], ms=6)
    ax.axvline(1.0, color=AXIS, lw=0.9)
    ax.set_xscale("log")
    ax.set_xlim(1 / lim / 1.5, lim * 1.5)
    _ratio_ticks(ax)
    ax.set_yticks(ys, [r["label"] for r in known], fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="profile 95 % interval / truth (arrows: open side)", title="all unknown vs one parameter known")
    ax = axs[1]
    if priors:
        sds = [100 * r["posterior"]["priors"][next(iter(r["posterior"]["priors"]))][1] for r in priors]
        for k in priors[0]["unknown"]:
            for x, r in zip(sds, priors, strict=True):
                p = r["posterior"]
                xt = np.log(r["parameters"][k])
                lo, hi = np.exp(np.asarray(p["ci95_log"][k]) - xt)
                off = {"alpha": 0.92, "k": 1.0, "h": 1.08}[k]
                ax.plot([x * off, x * off], [lo, hi], color=PARAM_COLOR[k], lw=2.0)
                ax.plot(x * off, np.exp(p["median_log"][k] - xt), "o", color=PARAM_COLOR[k], mec=SURF, ms=6)
        ax.axhline(1.0, color=AXIS, lw=0.9)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel=f"prior sd on {next(iter(priors[0]['posterior']['priors']))} [%] (mean off by the offset)",
            ylabel="posterior 95 % interval / truth",
            title="Bayesian prior (no source)",
        )
        ax.set_xticks(sds, [f"{v:g}" for v in sds])
        ax.minorticks_off()
        _ratio_ticks(ax, "y")
    hs = [plt.Line2D([], [], color=PARAM_COLOR[k], lw=2.2, label=PARAM_LABEL[k]) for k in ("alpha", "k", "h")]
    fig.legend(handles=hs, loc="outside lower center", ncol=3, fontsize=8)
    return fig


def gate_convergence(histories: dict[str, list[list[dict]]], truth: dict, crlb_log: dict) -> plt.Figure:
    """histories: estimator -> list (seeds) of histories [{alpha, k, h, ...}] -> rows α, k, h, Bi; columns least squares
    (model evaluations, best fit so far) | PINN (logged training steps, every variant and seed; joint_w100 emphasised):
    estimate / truth on log axes with the ±1.96 CRLB band."""
    qs = ("alpha", "k", "h", "Bi")
    fig, axs = plt.subplots(len(qs), 2, figsize=(9.6, 2.0 * len(qs)), squeeze=False, sharex="col", sharey="row")
    lab = {**PARAM_LABEL, "Bi": "Bi"}

    def ratio(h, q):
        if q == "Bi":
            return np.array([(x["h"] / x["k"]) / (truth["h"] / truth["k"]) for x in h])
        return np.array([x[q] / truth[q] for x in h])

    for i, q in enumerate(qs):
        for j in range(2):
            ax = axs[i][j]
            b = float(np.exp(1.96 * crlb_log[q]))
            ax.axhspan(1 / b, b, color="#efeee9", lw=0, zorder=0)
            ax.axhline(1.0, color=AXIS, lw=0.9, zorder=1)
        for name, hs in histories.items():
            ls = name.startswith("least squares")
            ax = axs[i][0 if ls else 1]
            emph = ls or "joint_w100" in name
            for h in hs:
                y = ratio(h, q)
                ax.plot(
                    np.arange(len(y)),
                    y,
                    color=INK if ls else (EMPH if emph else BASE),
                    lw=1.4 if emph else 0.8,
                    alpha=1.0 if emph else 0.6,
                    zorder=3 if emph else 2,
                )
        axs[i][0].set_ylabel(f"{lab[q]} / {lab[q]}_true")
        for j in range(2):
            axs[i][j].set_yscale("log")
    axs[0][0].set_title("least squares (FDM), best fit so far")
    axs[0][1].set_title("PINN, all variants and seeds")
    axs[-1][0].set_xlabel("model evaluation")
    axs[-1][1].set_xlabel("logged training step (every 50 steps)")
    hs = [
        plt.Line2D([], [], color=INK, lw=1.4, label="least squares"),
        plt.Line2D([], [], color=EMPH, lw=1.4, label="PINN joint_w100"),
        plt.Line2D([], [], color=BASE, lw=0.8, label="other PINN variants"),
        plt.Rectangle((0, 0), 1, 1, color="#efeee9", label="±1.96 CRLB (95 %)"),
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=4, fontsize=8)
    return fig


def gate_errors(rows: list[dict], crlb_pct: dict) -> plt.Figure:
    """rows: [{name, group classical | pinn | pinn_fewer, err {q: %}, ci {q: [lo, hi] %}, seeds {q: [%]}}] -> one panel
    per quantity: error % (symlog) with 95 % bands, seed values, and the ±1.96 CRLB band around zero."""
    qs = ("alpha", "k", "h", "Bi")
    col = {"classical": EMPH, "pinn": INK2, "pinn_fewer": "#1baf7a"}
    fig, axs = plt.subplots(1, len(qs), figsize=(3.2 * len(qs), 0.42 * len(rows) + 1.6), squeeze=False, sharey=True)
    ys = np.arange(len(rows))[::-1]
    for ax, q in zip(axs[0], qs, strict=True):
        b0, b1 = (100 * (np.exp(s * 1.96 * crlb_pct[q] / 100) - 1) for s in (-1, 1))  # crlb_pct = 100 sigma_ln
        ax.axvspan(b0, b1, color="#efeee9", lw=0, zorder=0)
        ax.axvline(0, color=AXIS, lw=0.9, zorder=1)
        for y, r in zip(ys, rows, strict=True):
            if q not in r["err"]:
                ax.annotate("known", (0, y), ha="center", va="center", fontsize=7, color=MUTED)
                continue
            c = col[r["group"]]
            if q in r.get("ci", {}):
                lo, hi = r["ci"][q]
                ax.plot([lo, hi], [y, y], color=c, lw=1.6, zorder=2, solid_capstyle="butt")
            for v in r.get("seeds", {}).get(q, []):
                ax.plot(v, y, "o", ms=3.5, color=c, alpha=0.5, mew=0, zorder=2)
            ax.plot(r["err"][q], y, "o", ms=6.5, color=c, mec=SURF, mew=1.0, zorder=3)
        ax.set_xscale("symlog", linthresh=10.0)
        ax.set(title={**PARAM_LABEL, "Bi": "Bi"}[q], xlabel="error [%]")
        ax.grid(axis="y", visible=False)
    axs[0][0].set_yticks(ys, [r["name"] for r in rows], fontsize=7)
    hs = [
        plt.Line2D([], [], marker="o", ls="", color=col[g], mec=SURF, ms=6, label=t)
        for g, t in (
            ("classical", "classical (least squares, bootstrap, posterior)"),
            ("pinn", "PINN, all three unknown"),
            ("pinn_fewer", "PINN, fewer unknowns"),
        )
    ]
    hs.append(plt.Rectangle((0, 0), 1, 1, color="#efeee9", label="±1.96 CRLB"))
    fig.legend(handles=hs, loc="outside lower center", ncol=4, fontsize=8)
    return fig
