"""Step 6.7 figures: cumulative cost vs number of queries (break-even), cost breakdown, inference latency vs batch and
threads, accuracy-cost trade-off (solver grid ladder vs the amortised PINN), and two 3D maps — which tool is cheaper
over (number of queries, required accuracy), and the break-even number of queries over (offline cost, solver time)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from aerosynapse.thermal.viz.diagnostics_figs import _plain_log  # noqa: F401
from aerosynapse.thermal.viz.inverse_figs import AXIS, GRID, INK, INK2, SURF, style  # noqa: F401  (style re-exported)

PINN, MEMBER, SOLVER, EQUAL, DATA = "#eb6834", "#f2a97f", "#2a78d6", "#1baf7a", "#8a5cc2"
TITLE = {
    "D_graetz": "Graetz channel (2D)",
    "C_cavity": "heated cavity (2D)",
    "A_steady": "A_steady (1D)",
    "B_slab": "B_slab (1D)",
}


def _t(c: str) -> str:
    return TITLE.get(c, c)


def _grid(n: int) -> tuple[int, int]:
    return (1, n) if n <= 2 else (2, int(np.ceil(n / 2)))


def _axes(n: int, w: float = 5.0, h: float = 4.2, **kw):
    r, c = _grid(n)
    fig, axs = plt.subplots(r, c, figsize=(w * c, h * r), squeeze=False, **kw)
    flat = axs.ravel()
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat


def _cost_pinn(a: dict, N, member: bool = False):
    if member:
        return a["offline_member_s"] + N * a["pinn_query_member_s"]
    return a["offline_ensemble_s"] + N * a["pinn_query_ensemble_s"]


def amortisation(an: dict) -> plt.Figure:
    """Per case: total cost vs number of queries — PINN ensemble / one member (offline + per query), solver at the case
    grid and at the cheapest grid as accurate as the PINN; break-even points marked."""
    cs = [c for c, a in an.items() if a.get("pinn_query_ensemble_s") and a.get("solver_s")]
    fig, axs = _axes(len(cs))
    N = np.logspace(0, 7, 200)
    for ax, c in zip(axs, cs, strict=False):
        a = an[c]
        ax.loglog(
            N,
            _cost_pinn(a, N),
            color=PINN,
            lw=2.0,
            label=f"PINN ensemble ({a['n_members']} members, error {100 * a['pinn_error']:.2g} %)",
        )
        ax.loglog(N, _cost_pinn(a, N, True), color=MEMBER, lw=1.4, ls=(0, (5, 2)), label="PINN one member")
        ax.loglog(
            N,
            N * a["solver_s"],
            color=SOLVER,
            lw=2.0,
            label=f"solver at its grid (error {100 * (a.get('solver_err') or 0):.2g} %)",
        )
        eq = a.get("equal_accuracy")
        if eq and abs(eq["time_s"] - a["solver_s"]) / a["solver_s"] > 0.01:
            ax.loglog(
                N,
                N * eq["time_s"],
                color=EQUAL,
                lw=1.4,
                ls=(0, (2, 2)),
                label=f"solver as accurate as the PINN ({100 * eq['rel_err']:.2g} %)",
            )
        for key, col in (("break_even_ensemble", PINN), ("break_even_member", MEMBER)):
            nb = a.get(key)
            if nb and nb < N[-1]:
                ax.axvline(nb, color=col, lw=0.8, ls=(0, (1, 2)))
                ax.annotate(
                    f"N* {nb:,.0f}",
                    (nb, a["solver_s"] * nb),
                    textcoords="offset points",
                    xytext=(5, -12),
                    fontsize=7,
                    color=col,
                )
        ax.set(xlabel="number of queries N", ylabel="total cost [s]")
        ax.set_title(f"{_t(c)}: speed-up per query {a['speedup_per_query']:.3g}x", fontsize=9)
        ax.legend(fontsize=6.5, loc="upper left")
    return fig


def breakdown(an: dict, n_ref: int = 1000) -> plt.Figure:
    """Per case: where the time goes for n_ref queries: PINN (data, training logged / alone, inference) vs solver."""
    cs = [c for c, a in an.items() if a.get("pinn_query_ensemble_s") and a.get("solver_s")]
    fig, axs = _axes(len(cs), 5.4, 3.2)
    for ax, c in zip(axs, cs, strict=False):
        a = an[c]
        rows = [
            (
                "PINN (training alone)",
                [a["data_s"], a["offline_ensemble_s"] - a["data_s"], n_ref * a["pinn_query_ensemble_s"]],
            ),
            (
                "PINN (as logged)",
                [a["data_s"], a["offline_ensemble_logged_s"] - a["data_s"], n_ref * a["pinn_query_ensemble_s"]],
            ),
            (f"solver ({a['solver_grid']})", [0.0, 0.0, n_ref * a["solver_s"]]),
        ]
        eq = a.get("equal_accuracy")
        if eq:
            rows.append(("solver, PINN accuracy", [0.0, 0.0, n_ref * eq["time_s"]]))
        h = 1 / 3600
        for k, (lab, v) in enumerate(rows):
            left = 0.0
            for val, col, name in zip(
                v,
                (DATA, PINN, SOLVER if lab.startswith("solver") else MEMBER),
                ("solver snapshot data", "training", "solver queries" if lab.startswith("solver") else "PINN queries"),
                strict=True,
            ):
                if val > 0:
                    ax.barh(
                        k,
                        val * h,
                        left=left,
                        color=col,
                        height=0.6,
                        edgecolor=SURF,
                        linewidth=2,
                        label=name,
                    )
                    left += val * h
            ax.text(left, k, f"  {left:.3g} h", va="center", fontsize=7, color=INK)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
        ax.invert_yaxis()
        ax.set(xlabel=f"hours for {n_ref:,} queries")
        ax.set_title(_t(c), fontsize=9)
        hl, ll = ax.get_legend_handles_labels()
        uniq = dict(zip(ll, hl, strict=True))
        ax.legend(uniq.values(), uniq.keys(), fontsize=6.5, loc="lower right")
    return fig


def latency(an: dict) -> plt.Figure:
    """Per case: PINN ensemble time per query vs batch size (queries per call), per thread setting; solver per query."""
    cs = [c for c, a in an.items() if a.get("threads")]
    fig, axs = _axes(len(cs))
    cols = [PINN, "#b04a1e", MEMBER]
    for ax, c in zip(axs, cs, strict=False):
        a = an[c]
        for k, (th, v) in enumerate(sorted(a["threads"].items(), key=lambda z: int(z[0]))):
            ax.loglog(
                a["batches"], v, "-o", color=cols[k % len(cols)], ms=4, mec=SURF, label=f"PINN ensemble, {th} thread(s)"
            )
        if a.get("solver_s"):
            ax.axhline(a["solver_s"], color=SOLVER, lw=1.4, label=f"solver per query ({a['solver_grid']})")
        eq = a.get("equal_accuracy")
        if eq and a.get("solver_s") and abs(eq["time_s"] - a["solver_s"]) / a["solver_s"] > 0.01:
            ax.axhline(eq["time_s"], color=EQUAL, lw=1.2, ls=(0, (2, 2)), label="solver as accurate as the PINN")
        if a.get("pinn_field_ensemble_s"):
            ax.axhline(
                a["pinn_field_ensemble_s"], color=INK2, lw=0.9, ls=(0, (4, 2)), label="PINN full field (1 query)"
            )
        ax.set(xlabel="queries per call (batch)", ylabel="time per query [s]")
        ax.set_title(_t(c), fontsize=9)
        ax.legend(fontsize=6.5)
    return fig


def tradeoff(an: dict) -> plt.Figure:
    """Per case: error vs time per query — the solver ladder (one point per grid / N) and the PINN ensemble, whose
    cost per query falls with the number of queries it is amortised over (offline / N + inference)."""
    cs = [c for c, a in an.items() if a.get("ladder") and a.get("pinn_query_ensemble_s")]
    fig, axs = _axes(len(cs))
    for ax, c in zip(axs, cs, strict=False):
        a = an[c]
        L = [r for r in a["ladder"] if r.get("rel_err")]
        kinds = sorted({r.get("kind", "solver") for r in L})
        for kd, col in zip(kinds, (SOLVER, EQUAL), strict=False):
            rr = sorted(
                (r for r in L if r.get("kind", "solver") == kd), key=lambda r: r.get("grid", r.get("N"))
            )  # by resolution
            ax.loglog(
                [r["rel_err"] for r in rr],
                [r["time_s"] for r in rr],
                "-o",
                color=col,
                ms=4,
                mec=SURF,
                label=f"{kd if kd != 'solver' else 'solver'} ladder",
            )
            for r in rr[:: max(1, len(rr) // 4)]:
                lab = r.get("grid", r.get("N"))
                ax.annotate(
                    str(lab),
                    (r["rel_err"], r["time_s"]),
                    textcoords="offset points",
                    xytext=(4, 3),
                    fontsize=6,
                    color=INK2,
                )
        e = a["pinn_error"]
        Ns = [1, 10, 100, 1e3, 1e4, 1e5, 1e6]
        ys = [a["offline_ensemble_s"] / n + a["pinn_query_ensemble_s"] for n in Ns]
        ax.loglog([e] * len(Ns), ys, "s", color=PINN, ms=6, mec=SURF, label="PINN ensemble, amortised over N")
        for n, y in zip(Ns, ys, strict=True):
            ax.annotate(f"N={n:.0e}", (e, y), textcoords="offset points", xytext=(6, -2), fontsize=6, color=PINN)
        ax.set(xlabel="relative QoI error (median)", ylabel="cost per query [s]")
        ax.set_title(f"{_t(c)}: accuracy vs cost", fontsize=9)
        ax.legend(fontsize=6.5, loc="lower left")
    return fig


def _solver_cost_at(a: dict, e_req: np.ndarray) -> np.ndarray:
    """Cheapest ladder entry with error <= the required error (NaN if none)."""
    L = [r for r in a["ladder"] if r.get("rel_err") is not None]
    out = np.full(e_req.shape, np.nan)
    for idx, e in np.ndenumerate(e_req):
        ok = [r["time_s"] for r in L if r["rel_err"] <= e]
        if ok:
            out[idx] = min(ok)
    return out


def decision3d(an: dict) -> plt.Figure:
    """Per case: log10(solver cost / PINN cost) over (log10 N, log10 required error). > 0: the PINN is cheaper.
    The PINN is only admissible where the required error is at least its own error (elsewhere: no surface)."""
    cs = [c for c, a in an.items() if a.get("ladder") and a.get("pinn_query_ensemble_s")]
    r, cc = _grid(len(cs))
    fig = plt.figure(figsize=(6.0 * cc, 5.0 * r))
    lN = np.linspace(0, 7, 41)
    le = np.linspace(-6, 0.5, 41)
    A, B = np.meshgrid(lN, le)
    for k, c in enumerate(cs):
        a = an[c]
        ts = _solver_cost_at(a, 10**B)
        Cs = 10**A * ts
        Cp = _cost_pinn(a, 10**A)
        Z = np.log10(Cs / Cp)
        Z[10**B < a["pinn_error"]] = np.nan  # the PINN cannot deliver this accuracy
        ax = fig.add_subplot(r, cc, k + 1, projection="3d")
        lim = np.nanmax(np.abs(Z)) if np.isfinite(Z).any() else 1.0
        ax.plot_surface(A, B, Z, cmap="RdBu_r", vmin=-lim, vmax=lim, linewidth=0, rcount=41, ccount=41)
        if np.isfinite(Z).any() and np.nanmin(Z) < 0 < np.nanmax(Z):
            ax.contour(A, B, np.ma.masked_invalid(Z), levels=[0.0], zdir="z", offset=-lim, colors=INK, linewidths=1.2)
        ax.plot(
            [0, 7],
            [np.log10(a["pinn_error"])] * 2,
            [-lim, -lim],
            color=PINN,
            lw=1.5,
            label="PINN error (min. admissible)",
        )
        ax.set(
            xlabel="log10 queries N",
            ylabel="log10 required rel. error",
            zlabel="log10 (solver / PINN cost)",
            zlim=(-lim, lim),
        )
        ax.set_title(f"{_t(c)}: red/positive = PINN cheaper", fontsize=9)
        ax.legend(fontsize=6.5, loc="upper left")
        ax.view_init(elev=26, azim=-125)
    return fig


def breakeven3d(an: dict) -> plt.Figure:
    """Per case: log10 break-even queries N* over (offline cost x factor, solver time per query); the case's own
    position marked. No surface where the solver is faster per query than the PINN (no break-even)."""
    cs = [c for c, a in an.items() if a.get("pinn_query_ensemble_s") and a.get("solver_s")]
    r, cc = _grid(len(cs))
    fig = plt.figure(figsize=(6.0 * cc, 5.0 * r))
    for k, c in enumerate(cs):
        a = an[c]
        f = np.linspace(-1, 1, 31)  # log10 factor on the offline cost
        ts0 = np.log10(a["solver_s"])
        g = np.linspace(ts0 - 2, ts0 + 2, 31)  # log10 solver time per query
        Fm, Gm = np.meshgrid(f, g)
        den = 10**Gm - a["pinn_query_ensemble_s"]
        Z = np.where(
            den > 0,
            np.log10(np.maximum(10**Fm * a["offline_ensemble_s"], 1e-300) / np.where(den > 0, den, 1.0)),
            np.nan,
        )
        ax = fig.add_subplot(r, cc, k + 1, projection="3d")
        ax.plot_surface(Fm, Gm, Z, cmap="viridis", linewidth=0, rcount=31, ccount=31, alpha=0.9)
        nb = a.get("break_even_ensemble")
        if nb:
            ax.scatter([0], [ts0], [np.log10(nb)], color=PINN, s=40, depthshade=False, label=f"this case: N* {nb:,.0f}")
            ax.legend(fontsize=6.5, loc="upper left")
        ax.set(xlabel="log10 offline-cost factor", ylabel="log10 solver time / query [s]", zlabel="log10 break-even N*")
        ax.set_title(_t(c), fontsize=9)
        ax.view_init(elev=26, azim=-130)
    return fig
