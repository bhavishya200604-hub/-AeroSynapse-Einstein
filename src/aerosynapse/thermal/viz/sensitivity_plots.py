"""Step 6.4b figures: response surfaces and maps, main effects, interactions, QoI distribution, fields (2D / 3D),
solver cost with the large-grid prediction, grid convergence. Pure functions (metrics in, Figure out)."""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from aerosynapse.thermal.utils.costmodel import human_time, predict
from aerosynapse.thermal.viz.inverse_figs import AXIS, GRID, INK, INK2, SURF, style  # noqa: F401  (style re-exported)
from aerosynapse.thermal.viz.style import DIV_CMAP, SEQ_CMAP

NAMES = ["Re", "Pr", "Bi"]
SOLVER, PINN, MEMBER = "#eb6834", "#2a78d6", "#9ec5f0"
POINT = ["#2a78d6", "#1baf7a", "#eb6834", "#52514e", "#b04a1e"]


def _plain_log(ax, which: str = "xy") -> None:
    """Log axes spanning less than ~1.5 decades: plain labels at 1, 1.5, 2, 3, 5, 7 x 10^n, no minor labels."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    for w in which:
        axis = ax.xaxis if w == "x" else ax.yaxis
        lo, hi = ax.get_xlim() if w == "x" else ax.get_ylim()
        if lo > 0 and hi / lo < 30:
            axis.set_major_locator(LogLocator(subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())


# --------------------------------------------------------------------------- PCE helpers
def decompose(m: dict) -> dict:
    """Variance shares of a PCE: first order, every pair S_ij, the triple term."""
    al = [tuple(a) for a in m["alphas"]]
    c2 = np.asarray(m["coef"]) ** 2
    V = float(sum(c for c, a in zip(c2, al, strict=True) if any(a)))
    S_ij = {}
    for i, j in itertools.combinations(range(3), 2):
        S_ij[f"{NAMES[i]}×{NAMES[j]}"] = float(
            sum(c for c, a in zip(c2, al, strict=True) if a[i] > 0 and a[j] > 0 and sum(a) == a[i] + a[j]) / V
        )
    first = [float(sum(c for c, a in zip(c2, al, strict=True) if a[i] > 0 and sum(a) == a[i]) / V) for i in range(3)]
    triple = float(sum(c for c, a in zip(c2, al, strict=True) if all(x > 0 for x in a)) / V)
    return {"S": first, "first": sum(first), "S_ij": S_ij, "pairs": sum(S_ij.values()), "triple": triple}


# --------------------------------------------------------------------------- response surfaces and maps
def surfaces3d(pl: dict) -> plt.Figure:
    """per case: 3D surface of the QoI over the surface pair (solver PCE, coloured) with the PINN ensemble as a
    wireframe, and the signed PINN - solver difference surface (diverging colours, 0 plane)."""
    names = list(pl)
    fig = plt.figure(figsize=(11.0, 4.6 * len(names)))
    for i, name in enumerate(names):
        c = pl[name]
        key = next(k for k in c["maps"] if k == "Re-Bi") if "Re-Bi" in c["maps"] else next(iter(c["maps"]))
        mp = c["maps"][key]
        a, b = key.split("-")
        X, Y = np.meshgrid(np.log10(mp["x"]), np.log10(mp["y"]))
        S, P = np.asarray(mp["solver"]), np.asarray(mp["pinn"])
        ax = fig.add_subplot(len(names), 2, 2 * i + 1, projection="3d")
        ax.plot_surface(X, Y, S, cmap=SEQ_CMAP, linewidth=0, antialiased=True, alpha=0.95, rcount=60, ccount=60)
        ax.plot_wireframe(X, Y, P, color=PINN, linewidth=0.5, rcount=14, ccount=14)
        ax.set(xlabel=f"log10 {a}", ylabel=f"log10 {b}", zlabel=c["qoi"])
        ax.set_title(f"{name}: solver (surface) vs PINN (wireframe)", fontsize=9)
        ax.view_init(elev=24, azim=-128)
        ax2 = fig.add_subplot(len(names), 2, 2 * i + 2, projection="3d")
        D = P - S
        lim = float(np.nanmax(np.abs(D)))
        ax2.plot_surface(
            X,
            Y,
            D,
            cmap=DIV_CMAP,
            norm=mcolors.TwoSlopeNorm(0, -lim, lim),
            linewidth=0,
            alpha=0.95,
            rcount=60,
            ccount=60,
        )
        ax2.plot_surface(X, Y, 0 * D, color=GRID, alpha=0.25, linewidth=0)
        ax2.set(xlabel=f"log10 {a}", ylabel=f"log10 {b}", zlabel="PINN − solver")
        ax2.set_title(f"{name}: signed difference (third input at the box centre)", fontsize=9)
        ax2.view_init(elev=24, azim=-128)
    return fig


def response_maps(c: dict, name: str) -> plt.Figure:
    """rows = input pairs; columns = solver PCE, PINN ensemble PCE (shared colour scale, iso-lines), relative
    difference in % (diverging); the third input at the box centre."""
    pairs = list(c["maps"])
    allv = np.concatenate([np.ravel(c["maps"][k][s]) for k in pairs for s in ("solver", "pinn")])
    norm = mcolors.Normalize(np.nanmin(allv), np.nanmax(allv))
    fig, axs = plt.subplots(len(pairs), 3, figsize=(12.0, 3.4 * len(pairs)), squeeze=False, constrained_layout=True)
    for i, k in enumerate(pairs):
        mp = c["maps"][k]
        a, b = k.split("-")
        X, Y = np.meshgrid(mp["x"], mp["y"])
        S, P = np.asarray(mp["solver"]), np.asarray(mp["pinn"])
        for j, (Z, ttl) in enumerate(((S, "solver (PCE)"), (P, "PINN ensemble (PCE)"))):
            ax = axs[i][j]
            cs = ax.contourf(X, Y, Z, levels=24, cmap=SEQ_CMAP, norm=norm)
            ax.contour(X, Y, S, levels=8, colors="white", linewidths=0.5, alpha=0.7)
            if j == 1:
                ax.contour(X, Y, Z, levels=cs.levels[::3], colors=INK, linewidths=0.6, linestyles="dashed")
            ax.set(xscale="log", yscale="log", xlabel=a, ylabel=b)
            ax.grid(False)
            _plain_log(ax)
            ax.set_title(f"{ttl}: {c['qoi']}", fontsize=9)
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=SEQ_CMAP), ax=axs[i][:2].tolist(), shrink=0.85, label=c["qoi"]
        )
        R = 100 * (P - S) / np.maximum(np.abs(S), 1e-300)
        lim = float(np.nanmax(np.abs(R)))
        ax = axs[i][2]
        cs = ax.contourf(X, Y, R, levels=24, cmap=DIV_CMAP, norm=mcolors.TwoSlopeNorm(0, -lim, lim))
        ax.contour(X, Y, R, levels=[0], colors=INK, linewidths=0.8)
        ax.set(xscale="log", yscale="log", xlabel=a, ylabel=b)
        ax.grid(False)
        _plain_log(ax)
        ax.set_title("PINN − solver [% of solver]", fontsize=9)
        fig.colorbar(cs, ax=ax, shrink=0.85)
    fig.suptitle(
        f"{name}: response maps (third input at the centre of the box)", x=0.01, ha="left", fontsize=10, color=INK
    )
    return fig


def main_effects(pl: dict) -> plt.Figure:
    """rows = cases, columns = inputs: E[QoI | x_i] - E[QoI] (univariate PCE terms) for the solver and the PINN
    ensemble, members as thin lines."""
    names = list(pl)
    fig, axs = plt.subplots(len(names), 3, figsize=(11.5, 3.0 * len(names)), squeeze=False, sharey="row")
    for i, name in enumerate(names):
        for j, k in enumerate(NAMES):
            ax = axs[i][j]
            me = pl[name]["main_effects"][k]
            for m in me["members"]:
                ax.plot(me["x"], m, color=MEMBER, lw=0.8)
            ax.plot(me["x"], me["solver"], color=SOLVER, lw=2.0)
            ax.plot(me["x"], me["pinn"], color=PINN, lw=1.6, ls=(0, (5, 2)))
            ax.axhline(0, color=AXIS, lw=0.8)
            ax.set(xscale="log", xlabel=k)
            _plain_log(ax, "x")
            if j == 0:
                ax.set_ylabel(f"{name}: main effect on {pl[name]['qoi']}")
    hs = [
        plt.Line2D([], [], color=SOLVER, lw=2, label="solver (PCE)"),
        plt.Line2D([], [], color=PINN, lw=1.6, ls=(0, (5, 2)), label="PINN ensemble (PCE)"),
        plt.Line2D([], [], color=MEMBER, lw=0.8, label="PINN members"),
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=3, fontsize=8)
    return fig


def interactions(pl: dict) -> plt.Figure:
    """rows = cases: (a) heatmaps of first-order (diagonal) and second-order (off-diagonal) Sobol indices from the
    PCE, solver and PINN; (b) variance decomposition into first order, pairs and the triple term."""
    names = list(pl)
    fig, axs = plt.subplots(
        len(names),
        3,
        figsize=(11.5, 3.4 * len(names)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1, 1, 1.1]},
        layout="constrained",
    )
    for i, name in enumerate(names):
        dec = pl[name]["decomposition"]
        for j, who in enumerate(("solver", "pinn")):
            d = dec[who]
            M = np.full((3, 3), np.nan)
            for a in range(3):
                M[a, a] = d["S"][a]
            for (a, b), key in zip(itertools.combinations(range(3), 2), d["S_ij"], strict=True):
                M[a, b] = M[b, a] = d["S_ij"][key]
            ax = axs[i][j]
            im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
            for a in range(3):
                for b in range(3):
                    ax.text(
                        b,
                        a,
                        f"{M[a, b]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=INK if M[a, b] < 0.55 else SURF,
                    )
            ax.set_xticks(range(3), NAMES)
            ax.set_yticks(range(3), NAMES)
            ax.set_title(f"{name}: {'solver' if who == 'solver' else 'PINN'} — S_i (diag), S_ij", fontsize=9)
            ax.grid(False)
        fig.colorbar(im, ax=axs[i][:2].tolist(), shrink=0.8)
        ax = axs[i][2]
        parts = [("first order", "first", "#2a78d6"), ("pairs", "pairs", "#eb6834"), ("triple", "triple", "#1baf7a")]
        for j, who in enumerate(("solver", "pinn")):
            left = 0.0
            for lab, key, col in parts:
                v = dec[who][key]
                ax.barh(j, v, left=left, color=col, height=0.55, label=lab if j == 0 else None)
                if v > 0.09:
                    ax.text(left + v / 2, j, f"{v:.2f}", ha="center", va="center", color=SURF, fontsize=8)
                left += v
        ax.set_yticks([0, 1], ["solver", "PINN"])
        ax.set(xlim=(0, 1.0), xlabel="share of the QoI variance")
        ax.set_title(f"{name}: variance decomposition", fontsize=9)
        ax.grid(False)
    hs = [plt.Rectangle((0, 0), 1, 1, color=col, label=lab) for lab, _, col in parts]
    fig.legend(handles=hs, loc="outside lower right", ncol=3, fontsize=8, title="variance shares")
    return fig


def distribution(pl: dict) -> plt.Figure:
    """columns = cases: distribution of the QoI under the input uncertainty (log-uniform box): histogram + empirical
    CDF, solver vs PINN ensemble (same design points)."""
    names = list(pl)
    fig, axs = plt.subplots(2, len(names), figsize=(5.0 * len(names), 6.0), squeeze=False)
    for j, name in enumerate(names):
        s, p = np.asarray(pl[name]["distribution"]["solver"]), np.asarray(pl[name]["distribution"]["pinn"])
        lo, hi = min(s.min(), p.min()), max(s.max(), p.max())
        bins = np.linspace(lo, hi, 40)
        ax = axs[0][j]
        ax.hist(s, bins=bins, color=SOLVER, alpha=0.55, density=True, label="solver")
        ax.hist(p, bins=bins, color=PINN, alpha=0.55, density=True, label="PINN ensemble")
        ax.set(xlabel=pl[name]["qoi"], ylabel="density")
        ax.set_title(f"{name}: QoI under input uncertainty", fontsize=9)
        ax.legend(fontsize=7)
        ax = axs[1][j]
        for v, col, lab in ((s, SOLVER, "solver"), (p, PINN, "PINN ensemble")):
            v = np.sort(v)
            ax.step(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=1.5, where="post", label=lab)
        ax.set(xlabel=pl[name]["qoi"], ylabel="empirical CDF")
    return fig


# --------------------------------------------------------------------------- fields
def fields2d(c: dict, name: str) -> plt.Figure:
    """θ from the solver and the PINN ensemble (shared scale) with solver streamlines, and |θ_PINN − θ_solver|."""
    f = c["fields"]
    x, y = np.asarray(f["x"]), np.asarray(f["y"])
    X, Y = np.meshgrid(x, y)
    ts, tp = np.asarray(f["solver"]["theta"]), np.asarray(f["pinn"]["theta"])
    us, vs = np.asarray(f["solver"]["u"]), np.asarray(f["solver"]["v"])
    wide = (x.max() - x.min()) / (y.max() - y.min()) > 2
    fig, axs = plt.subplots(
        3 if wide else 1,
        1 if wide else 3,
        figsize=(12.0, 7.2) if wide else (13.0, 4.0),
        squeeze=False,
        constrained_layout=True,
    )
    axs = axs.ravel()
    norm = mcolors.Normalize(np.nanmin([ts, tp]), np.nanmax([ts, tp]))
    lab0 = "solver θ + isotherms" if wide else "solver θ + streamlines"
    lab1 = "PINN ensemble θ (dashed: solver isotherms)" if wide else "PINN ensemble θ"
    for ax, Z, ttl in ((axs[0], ts, lab0), (axs[1], tp, lab1)):
        im = ax.pcolormesh(X, Y, Z, cmap="inferno", norm=norm, shading="auto")
        if ax is axs[0] and not wide:  # recirculating cavity flow: streamlines
            ax.streamplot(
                x, y, np.nan_to_num(us), np.nan_to_num(vs), color="white", linewidth=0.6, density=1.1, arrowsize=0.6
            )
        elif wide:  # parallel channel flow: solver isotherms (solid on the solver, dashed on the PINN)
            ax.contour(
                X, Y, ts, levels=10, colors="white", linewidths=0.6, linestyles="solid" if ax is axs[0] else "dashed"
            )
        ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=9)
    fig.colorbar(im, ax=list(axs[:2]), shrink=0.85, label="θ")
    fig.suptitle(
        f"{name} at Re {f['point']['Re']:.3g}, Pr {f['point']['Pr']:.3g}, Bi {f['point']['Bi']:.3g}",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    E = np.abs(tp - ts)
    im = axs[2].pcolormesh(
        X,
        Y,
        E,
        cmap="magma",
        norm=mcolors.LogNorm(max(np.nanmin(E[E > 0]), np.nanmax(E) * 1e-4), np.nanmax(E)),
        shading="auto",
    )
    axs[2].set_aspect("equal")
    axs[2].set_title("|θ_PINN − θ_solver| (log scale)", fontsize=9)
    fig.colorbar(im, ax=axs[2], shrink=0.85)
    for ax in axs:
        ax.set(xlabel="x", ylabel="y")
        ax.grid(False)
    return fig


def field3d(pl: dict) -> plt.Figure:
    """per case: θ(x, y) as a 3D surface (solver, coloured) with the PINN ensemble as a wireframe."""
    names = list(pl)
    fig = plt.figure(figsize=(6.0 * len(names), 5.0))
    for i, name in enumerate(names):
        f = pl[name]["fields"]
        X, Y = np.meshgrid(f["x"], f["y"])
        ts, tp = np.asarray(f["solver"]["theta"]), np.asarray(f["pinn"]["theta"])
        ax = fig.add_subplot(1, len(names), i + 1, projection="3d")
        ax.plot_surface(X, Y, np.nan_to_num(ts), cmap="inferno", linewidth=0, alpha=0.95, rcount=80, ccount=80)
        ax.plot_wireframe(X, Y, np.nan_to_num(tp), color=PINN, linewidth=0.4, rcount=16, ccount=16)
        ax.set(xlabel="x", ylabel="y", zlabel="θ")
        ax.set_title(f"{name}: θ solver (surface) vs PINN (wireframe)", fontsize=9)
        ax.view_init(elev=28, azim=-60)
        if (np.ptp(f["x"]) / np.ptp(f["y"])) > 2:
            ax.set_box_aspect((3.0, 1.0, 0.8))
    return fig


# --------------------------------------------------------------------------- cost and grid convergence
def cost(grid: dict) -> plt.Figure:
    """rows = cases: (a) wall time per solve and (b) peak memory vs number of cells, measured (points) with the
    power-law fit, its 95 % prediction band out to the target grid (star) and the machine's RAM."""
    names = [k for k in grid if k not in ("machine", "memory_limit_bytes")]
    ram = grid["machine"].get("total_memory")
    fig, axs = plt.subplots(len(names), 2, figsize=(11.0, 3.6 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        c = grid[name]
        n = np.array([r["cells"] for r in c["rows"]])
        p = c["prediction"]
        nn = np.geomspace(n.min(), p["cells"] * 1.3, 200)
        base = min(r["baseline_bytes"] for r in c["rows"])
        for j, (key, fit, yl, scale) in enumerate(
            (
                ("wall_s", c["fit_time"], "wall time per solve [s]", 1.0),
                ("mem_bytes", c["fit_memory"], "memory [GiB] (direct solver + process baseline)", 2.0**-30),
            )
        ):
            ax = axs[i][j]
            y = np.array([r[key] for r in c["rows"]]) * scale
            pr = predict(fit, nn)
            add = base * scale if key == "mem_bytes" else 0.0
            ax.fill_between(nn, pr["lo"] * scale + add, pr["hi"] * scale + add, color=PINN, alpha=0.15, lw=0)
            ax.plot(nn, pr["value"] * scale + add, color=PINN, lw=1.4)
            ax.plot(n, y + add, "o", color=INK, ms=4, mec=SURF)
            tv = p["time_per_solve_s"] if key == "wall_s" else [v * scale for v in p["memory_bytes"]]
            ax.errorbar(
                [p["cells"]],
                [tv[0]],
                yerr=[[tv[0] - tv[1]], [tv[2] - tv[0]]],
                fmt="*",
                ms=13,
                color=SOLVER,
                ecolor=SOLVER,
                capsize=3,
            )
            lab = human_time(tv[0]) if key == "wall_s" else f"{tv[0]:.3g} GiB"
            ax.annotate(
                f"{p['target']}{'²' if name != 'D_graetz' else ' cells/unit'}: {lab}",
                (p["cells"], tv[0]),
                xytext=(-8, 10),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color=INK2,
            )
            if key == "mem_bytes" and ram:
                ax.axhline(ram * scale, color=INK2, lw=1.0, ls=(0, (4, 2)))
                ax.annotate(
                    f"RAM {ram * scale:.1f} GiB",
                    (nn[0], ram * scale),
                    xytext=(2, 3),
                    textcoords="offset points",
                    fontsize=7,
                    color=INK2,
                )
            ax.set(xscale="log", yscale="log", xlabel="cells", ylabel=yl)
            ax.set_title(f"{name}: ∝ cells^{fit['b']:.2f} (± {fit['b_se']:.2f})", fontsize=9)
    hs = [
        plt.Line2D([], [], marker="o", ls="", color=INK, ms=4, label="measured (one solve per process)"),
        plt.Line2D([], [], color=PINN, lw=1.4, label="power-law fit (band: 95 % prediction)"),
        plt.Line2D([], [], marker="*", ls="", color=SOLVER, ms=10, label="predicted target grid"),
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=3, fontsize=8)
    return fig


def grid_convergence(grid: dict) -> plt.Figure:
    """rows = cases: QoI vs grid spacing h at every parameter point, with the Richardson value (dashed) and the
    relative difference to it (log-log: the slope is the observed order)."""
    names = [k for k in grid if k not in ("machine", "memory_limit_bytes")]
    fig, axs = plt.subplots(len(names), 2, figsize=(10.5, 3.4 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        for k, e in enumerate(grid[name]["convergence"]):
            col = POINT[k % len(POINT)]
            h = 1.0 / np.asarray(e["grid"], float)
            q = np.asarray(e["qoi"])
            lab = f"Re {e['point']['Re']:.3g}, Pr {e['point']['Pr']:.3g}, Bi {e['point']['Bi']:.3g}"
            fe = (e.get("gci") or {}).get("f_ext")
            if fe is not None and np.isfinite(fe):
                axs[i][0].plot(h, q / fe, "-o", color=col, ms=4, mec=SURF, label=lab)
                d = np.abs(q - fe) / abs(fe)
                ok = d > 0
                axs[i][1].plot(h[ok], d[ok], "-o", color=col, ms=4, mec=SURF)
        axs[i][0].axhline(1.0, color=AXIS, lw=0.9)
        axs[i][0].set(xscale="log", xlabel="h = 1 / grid", ylabel=f"{name}: QoI / Richardson value")
        axs[i][0].invert_xaxis()
        axs[i][0].legend(fontsize=7)
        axs[i][1].set(xscale="log", yscale="log", xlabel="h = 1 / grid", ylabel="|QoI − Richardson| / |Richardson|")
        axs[i][1].invert_xaxis()
    return fig
