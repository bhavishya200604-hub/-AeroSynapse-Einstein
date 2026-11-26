"""Step 6.6 figures: Graetz Nusselt numbers vs x* (2D and 3D), channel temperature (3D), PINN vs solver local Nu,
natural-convection Nu vs Ra with correlations and benchmarks, cavity isotherms / streamlines (2D) and surfaces (3D)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from aerosynapse.thermal.physics import graetz as G
from aerosynapse.thermal.viz.diagnostics_figs import _plain_log
from aerosynapse.thermal.viz.inverse_figs import AXIS, GRID, INK, INK2, SURF, style  # noqa: F401  (style re-exported)

SOLVER, PINN, SERIES, CORR, BENCH = "#2a78d6", "#eb6834", "#1a1a19", "#1baf7a", "#b04a1e"


def _pe_colours(pes):
    pes = sorted(set(pes))
    return {p: plt.cm.viridis(0.1 + 0.8 * i / max(len(pes) - 1, 1)) for i, p in enumerate(pes)}


def _upto(met: dict, r: dict) -> np.ndarray:
    """Mask of the stations up to the evaluation station (the last cells feel the zero-gradient outflow)."""
    x = np.asarray(r["xstar"]) * 4.0 * r["Pe"]
    st = [e["x_station"] for e in (met.get("analysis") or {}).get("graetz", []) if "x_station" in e]
    return x <= (st[0] if st else np.inf)


def graetz_nu(met: dict) -> plt.Figure:
    """(a) local and (b) mean Nu_Dh vs x*: solver (colour = Pe), eigen-series (black), Leveque asymptote (dashed),
    fully developed 7.541 (dotted), Phase 4 gate PINNs; (c) solver / series − 1 (axial conduction, low Pe)."""
    fig, axs = plt.subplots(1, 3, figsize=(16.0, 4.4))
    s = met["series"]
    xs = np.asarray(s["xstar"])
    cols = _pe_colours([r["Pe"] for r in met["graetz"]])
    for k, key, lev in ((0, "Nu_local", G.leveque_local), (1, "Nu_mean", G.leveque_mean)):
        ax = axs[k]
        for r in sorted(met["graetz"], key=lambda z: z["Pe"]):
            m = _upto(met, r)
            ax.plot(np.asarray(r["xstar"])[m], np.asarray(r[key])[m], color=cols[r["Pe"]], lw=1.4)
        ax.plot(xs, s[key], color=SERIES, lw=1.2, label="eigen-series (no axial conduction)")
        ax.plot(xs, lev(xs), color=CORR, lw=1.1, ls=(0, (5, 3)), label="Leveque asymptote")
        ax.axhline(G.NU_FD_CWT, color=CORR, lw=1.0, ls=(0, (1, 2)), label="fully developed 7.541")
        gp = (met.get("gate_pinns") or {}).get("graetz")
        if gp and key == "Nu_local":
            for i, rr in enumerate(gp["runs"]):
                ax.plot(
                    rr["xstar"],
                    rr["Nu_x"],
                    ".",
                    color=PINN,
                    ms=3,
                    label="Phase 4 gate PINN (Pe 20)" if i == 0 else None,
                )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="x* = x / (D_h Re_Dh Pr)",
            ylabel=f"{'local' if k == 0 else 'mean'} Nu_Dh",
        )
        ax.set_ylim(5, 80)
        _plain_log(ax, "y")
        ax.set_title(f"({'a' if k == 0 else 'b'}) {'local' if k == 0 else 'mean'} Nusselt number", fontsize=9)
    axs[0].legend(fontsize=7, loc="upper right")
    ax = axs[2]
    for r in sorted(met["graetz"], key=lambda z: z["Pe"]):
        x = np.asarray(r["xstar"])
        ok = (x >= 1e-4) & _upto(met, r)
        ser = G.entry_series(x[ok])
        ax.plot(
            x[ok],
            np.asarray(r["Nu_mean"])[ok] / ser["Nu_mean"] - 1,
            color=cols[r["Pe"]],
            lw=1.4,
            label=f"Pe {r['Pe']:g} (Re {r['Re']:g}, Pr {r['Pr']:g})",
        )
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set(xscale="log", xlabel="x*", ylabel="solver / series − 1 (mean Nu)")
    ax.set_title("(c) axial conduction: deviation from the classical result", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    return fig


def graetz_surface3d(met: dict) -> plt.Figure:
    """3D: local Nu_Dh over (log10 x*, log10 Pe): solver ribbons per Pe and the Pe-independent series."""
    fig = plt.figure(figsize=(9.5, 6.5))
    ax = fig.add_subplot(projection="3d")
    cols = _pe_colours([r["Pe"] for r in met["graetz"]])
    for r in sorted(met["graetz"], key=lambda z: z["Pe"]):
        x = np.asarray(r["xstar"])
        ok = (x >= 1e-5) & _upto(met, r)
        lx, nu = np.log10(x[ok]), np.log10(np.asarray(r["Nu_local"])[ok])
        lp = np.full_like(lx, np.log10(r["Pe"]))
        ax.plot(lx, lp, nu, color=cols[r["Pe"]], lw=2.0)
        ser = G.entry_series(x[ok])
        ax.plot(lx, lp, np.log10(ser["Nu_local"]), color=SERIES, lw=0.8, ls=(0, (2, 2)))
    ax.set(xlabel="log10 x*", ylabel="log10 Pe_H", zlabel="log10 local Nu_Dh")
    ax.set_title(
        "Graetz channel: solver (colour) vs eigen-series (dashed) — divergence at low Pe = axial conduction", fontsize=9
    )
    ax.view_init(elev=24, azim=-58)
    return fig


def graetz_field3d(met: dict, fields: dict) -> plt.Figure:
    """θ(x, y) of the channel for the plotted (Re, Pr) pair as a 3D surface with isotherms projected below."""
    keys = sorted({k.rsplit("_", 1)[0] for k in fields if k.startswith("graetz_")})
    if not keys:
        return plt.figure()
    k = keys[len(keys) // 2]
    want = met.get("plot_pair")
    if want:
        cand = f"graetz_{want[0]:g}_{want[1]:g}"
        k = cand if f"{cand}_theta" in fields else k
    x, y, th = fields[f"{k}_x"], fields[f"{k}_y"], fields[f"{k}_theta"]
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(11.0, 5.0))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, th, cmap="inferno", linewidth=0, rcount=60, ccount=160, alpha=0.95)
    ax.contour(X, Y, th, levels=12, zdir="z", offset=float(np.nanmin(th)) - 0.25, cmap="inferno", linewidths=0.8)
    ax.set(xlabel="x / H", ylabel="y / H", zlabel="θ", zlim=(float(np.nanmin(th)) - 0.25, 1.0))
    ax.set_box_aspect((3.0, 1.0, 0.9))
    Re, Pr = k.split("_")[1:3]
    ax.set_title(f"Graetz channel, isothermal walls, Re_H {Re}, Pr {Pr}: temperature (isotherms below)", fontsize=9)
    ax.view_init(elev=30, azim=-62)
    return fig


def pinn_nu(pinn: list[dict]) -> plt.Figure:
    """columns = parameter points: local Nu along the channel, solver vs PINN ensemble (members thin); bottom: error."""
    n = len(pinn)
    fig, axs = plt.subplots(2, n, figsize=(4.6 * n, 6.0), squeeze=False, sharex="col")
    for j, r in enumerate(pinn):
        x = np.asarray(r["xstar"])
        ax = axs[0][j]
        for m in r["members"]:
            ax.plot(x, m, color=PINN, lw=0.6, alpha=0.4)
        ax.plot(x, r["solver"]["Nu_local"], color=SOLVER, lw=1.8, label="solver")
        ax.plot(x, r["pinn"]["Nu_local"], color=PINN, lw=1.4, ls=(0, (5, 2)), label="PINN ensemble (mean field)")
        ax.plot([], [], color=PINN, lw=0.6, alpha=0.4, label="members")
        ax.set_ylim(
            0, 1.3 * float(np.max(r["solver"]["Nu_local"]))
        )  # members whose theta_w - theta_b crosses 0 leave it
        p = r["point"]
        ax.set(xscale="log", ylabel="local Nu_Dh (Robin walls)")
        ax.set_title(f"Re {p['Re']:.3g}, Pr {p['Pr']:.3g}, Bi {p['Bi']:.3g}", fontsize=9)
        ax.legend(fontsize=7)
        e = np.abs(np.asarray(r["pinn"]["Nu_local"]) - np.asarray(r["solver"]["Nu_local"])) / np.abs(
            r["solver"]["Nu_local"]
        )
        axs[1][j].plot(x, e, color=INK, lw=1.2)
        axs[1][j].set(xscale="log", yscale="log", xlabel="x*", ylabel="relative error")
    return fig


def natural_nu(met: dict) -> plt.Figure:
    """(a) hot-wall Nu vs Ra per Pr: solver (Richardson), Berkovsky–Polevikov (line), de Vahl Davis (□), Phase 4 PINNs
    (×); (b) deviation from Berkovsky–Polevikov; (c) grid convergence of Nu (Pr of the first entry)."""
    an = met["analysis"]["natural"]
    prs = sorted({e["Pr"] for e in an.values()})
    cols = {p: c for p, c in zip(prs, [SOLVER, PINN, CORR, BENCH], strict=False)}
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for Pr in prs:
        es = sorted((e for e in an.values() if e["Pr"] == Pr), key=lambda z: z["Ra"])
        Ra = np.array([e["Ra"] for e in es])
        axs[0].plot(Ra, [e["best"] for e in es], "o", color=cols[Pr], ms=6, mec=SURF, label=f"solver, Pr {Pr:g}")
        rr = np.geomspace(Ra.min() / 1.5, Ra.max() * 1.5, 100)
        bp = 0.18 * (Pr / (0.2 + Pr) * rr) ** 0.29
        axs[0].plot(rr, bp, color=cols[Pr], lw=1.2, ls=(0, (5, 3)), label=f"Berkovsky–Polevikov, Pr {Pr:g}")
        axs[1].plot(Ra, [100 * e["dev_BP"] for e in es], "-o", color=cols[Pr], ms=5, mec=SURF, label=f"Pr {Pr:g}")
    dv = [e for e in an.values() if "dVD" in e]
    if dv:
        axs[0].plot(
            [e["Ra"] for e in dv], [e["dVD"] for e in dv], "s", mfc="none", mec=INK, ms=9, label="de Vahl Davis (1983)"
        )
    gp = (met.get("gate_pinns") or {}).get("dvd")
    if gp:
        axs[0].plot([gp["Ra"]] * len(gp["Nu_hot"]), gp["Nu_hot"], "x", color=BENCH, ms=8, label="Phase 4 gate PINNs")
    axs[0].set(xscale="log", yscale="log", xlabel="Ra", ylabel="hot-wall Nu")
    _plain_log(axs[0], "y")
    axs[0].legend(fontsize=7)
    axs[1].axhline(0, color=AXIS, lw=0.8)
    axs[1].set(xscale="log", xlabel="Ra", ylabel="solver vs Berkovsky–Polevikov [%]")
    axs[1].legend(fontsize=7)
    Pr0 = prs[0]
    for k, e in enumerate(sorted((e for e in an.values() if e["Pr"] == Pr0), key=lambda z: z["Ra"])):
        col = plt.cm.viridis(0.1 + 0.8 * k / max(len(an) // len(prs) - 1, 1))
        axs[2].plot(
            1.0 / np.asarray(e["N"]),
            np.abs(np.asarray(e["Nu"]) / e["best"] - 1) + 1e-12,
            "-o",
            color=col,
            ms=4,
            mec=SURF,
            label=f"Ra {e['Ra']:.0e}",
        )
    axs[2].set(xscale="log", yscale="log", xlabel="h = 1 / N", ylabel=f"|Nu / Richardson − 1| (Pr {Pr0:g})")
    _plain_log(axs[2], "x")
    axs[2].invert_xaxis()
    axs[2].legend(fontsize=6, ncol=2)
    for ax, t in zip(axs, ("(a) Nu vs Ra", "(b) deviation from the correlation", "(c) grid convergence"), strict=True):
        ax.set_title(t, fontsize=9)
    return fig


def _nat(fields: dict, Pr: float, Ra: float):
    k = f"nat_{Pr:g}_{Ra:g}"
    if f"{k}_theta" not in fields:
        return None
    return fields[f"{k}_xc"], fields[f"{k}_yc"], fields[f"{k}_theta"], fields[f"{k}_psi"]


def natural_fields(met: dict, fields: dict) -> plt.Figure:
    """rows = Pr, columns = Ra: isotherms (colour) and streamlines (ψ contours) of the cavity."""
    prs = sorted({e["Pr"] for e in met["analysis"]["natural"].values()})
    ras = met.get("plot_Ra") or sorted({e["Ra"] for e in met["analysis"]["natural"].values()})[::2]
    fig, axs = plt.subplots(
        len(prs), len(ras), figsize=(3.2 * len(ras), 3.2 * len(prs)), squeeze=False, layout="constrained"
    )
    for i, Pr in enumerate(prs):
        for j, Ra in enumerate(ras):
            ax = axs[i][j]
            d = _nat(fields, Pr, Ra)
            if d is None:
                ax.axis("off")
                continue
            x, y, th, psi = d
            X, Y = np.meshgrid(x, y)
            ax.contourf(X, Y, th, levels=21, cmap="RdBu_r")
            xs = np.linspace(0, 1, psi.shape[1])
            ys = np.linspace(0, 1, psi.shape[0])
            a = float(np.abs(psi).max())
            lv = np.linspace(-a, a, 22)[1:-1]
            lv = lv[np.abs(lv) > 0.02 * a]  # no zero level: psi = 0 on the walls draws broken lines along them
            ax.contour(xs, ys, psi, levels=lv, colors=INK, linewidths=0.6, linestyles="solid")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Pr {Pr:g}, Ra {Ra:.0e}", fontsize=8)
            ax.grid(False)
    return fig


def natural_field3d(met: dict, fields: dict) -> plt.Figure:
    """3D surfaces θ(x, y) of the cavity for increasing Ra (first Pr): the thermal boundary layers sharpen."""
    prs = sorted({e["Pr"] for e in met["analysis"]["natural"].values()})
    ras = met.get("plot_Ra") or sorted({e["Ra"] for e in met["analysis"]["natural"].values()})[::2]
    fig = plt.figure(figsize=(4.2 * len(ras), 4.2))
    for j, Ra in enumerate(ras):
        d = _nat(fields, prs[0], Ra)
        ax = fig.add_subplot(1, len(ras), j + 1, projection="3d")
        if d is None:
            continue
        x, y, th, _ = d
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, th, cmap="RdBu_r", linewidth=0, rcount=64, ccount=64)
        ax.set(xlabel="x", ylabel="y", zlabel="θ")
        ax.set_title(f"Ra {Ra:.0e}, Pr {prs[0]:g}", fontsize=9)
        ax.view_init(elev=28, azim=-130)
    return fig
