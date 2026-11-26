"""Step 6.6b figures: UQ of the 2D ensembles (parity, reliability, spread vs error; field-level 3D), extrapolation
(error and indicators vs distance, ROC; 3D error surfaces over a parameter slice with the training box), training
diagnostics (gradient cosine matrices, NTK / Hessian spectra, loss along init -> final, 3D loss landscapes)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from aerosynapse.thermal.viz.diagnostics_figs import _plain_log
from aerosynapse.thermal.viz.inverse_figs import AXIS, GRID, INK, INK2, SURF, style  # noqa: F401  (style re-exported)

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"
MEM = [C1, C2, C3, C4, "#b04a1e"]
TITLE = {"D_graetz": "Graetz channel", "C_cavity": "heated cavity"}


def _t(case: str) -> str:
    return TITLE.get(case, case)


def uq(cases: dict, an: dict) -> plt.Figure:
    """columns = cases: (a) ensemble mean ± 2 sd vs the solver QoI; (b) reliability: empirical vs nominal coverage of
    Gaussian intervals, and the split-conformal coverage; (c) |error| vs spread."""
    cs = [c for c in cases if "uq" in an.get(c, {})]
    fig, axs = plt.subplots(3, len(cs), figsize=(4.8 * len(cs), 12.0), squeeze=False)
    for j, c in enumerate(cs):
        u = an[c]["uq"]
        y, mu, sd = (np.asarray(u["points"][k]) for k in ("y", "mu", "sd"))
        ax = axs[0][j]
        ax.errorbar(
            y, mu, yerr=2 * sd, fmt="o", ms=3, color=C1, ecolor=C1, elinewidth=0.7, alpha=0.8, label="mean ± 2 sd"
        )
        lim = [min(y.min(), mu.min()), max(y.max(), mu.max())]
        ax.plot(lim, lim, color=INK, lw=0.8)
        ax.set(xlabel=f"solver {cases[c]['qoi']}", ylabel="PINN ensemble")
        ax.set_title(f"(a) {_t(c)}: {u['n_members']} members, {u['n']} points", fontsize=9)
        ax.legend(fontsize=7)
        ax = axs[1][j]
        g = u["gaussian"]["coverage"]
        lv = np.array([float(k) for k in g])
        ax.plot([0, 1], [0, 1], color=INK, lw=0.8)
        ax.plot(lv, list(g.values()), "-o", color=C1, ms=5, mec=SURF, label="Gaussian (mean ± z sd)")
        cf = u["conformal"]
        ax.errorbar(
            [cf["target"]],
            [cf["coverage_mean"]],
            yerr=[[cf["coverage_mean"] - cf["coverage_p05_p95"][0]], [cf["coverage_p05_p95"][1] - cf["coverage_mean"]]],
            fmt="s",
            color=C2,
            ms=7,
            label="split conformal (5–95 % over splits)",
        )
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="nominal coverage", ylabel="empirical coverage")
        ax.set_title(f"(b) reliability (ECE {u['gaussian']['ece']:.2f})", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax = axs[2][j]
        e = np.abs(mu - y)
        ax.loglog(np.maximum(sd, 1e-12), np.maximum(e, 1e-12), "o", ms=3, color=C1)
        lo, hi = max(min(sd.min(), e.min()), 1e-8), max(sd.max(), e.max())
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, label="|error| = sd")
        ax.plot([lo, hi], [2 * lo, 2 * hi], color=INK2, lw=0.8, ls=(0, (4, 3)), label="|error| = 2 sd")
        ax.set(xlabel="ensemble sd", ylabel="|mean − solver|")
        _plain_log(ax, "xy")
        ax.set_title(f"(c) spread vs error (Spearman {u['gaussian']['rank_corr']:.2f})", fontsize=9)
        ax.legend(fontsize=7)
    return fig


def uq_field3d(cases: dict, fields: dict) -> plt.Figure:
    """Per case (first field point): 3D surfaces of the actual error |mean - solver| and of the ensemble sd of theta."""
    cs = [c for c in cases if f"{c}_0_theta" in fields]
    fig = plt.figure(figsize=(11.0, 4.6 * max(len(cs), 1)))
    for i, c in enumerate(cs):
        x, y = fields[f"{c}_0_x"], fields[f"{c}_0_y"]
        X, Y = np.meshgrid(x, y)
        err = np.abs(fields[f"{c}_0_theta_mean"] - fields[f"{c}_0_theta"])
        sd = fields[f"{c}_0_theta_sd"]
        top = float(max(err.max(), sd.max()))
        v = cases[c]["fields"][0]["values"] if cases[c].get("fields") else {}
        for k, (Z, lab, cm) in enumerate(((err, "|error| of θ", "magma"), (sd, "ensemble sd of θ", "viridis"))):
            ax = fig.add_subplot(len(cs), 2, 2 * i + k + 1, projection="3d")
            ax.plot_surface(X, Y, Z, cmap=cm, linewidth=0, rcount=48, ccount=96)
            ax.set(xlabel="x", ylabel="y", zlabel=lab, zlim=(0, top))
            if c == "D_graetz":
                ax.set_box_aspect((3.0, 1.0, 0.9))
            pt = ", ".join(f"{n} {v[n]:.3g}" for n in ("Re", "Pr", "Bi")) if v else ""
            ax.set_title(f"{_t(c)} ({pt}): {lab}", fontsize=8)
            ax.view_init(elev=30, azim=-60)
    return fig


def extrapolation(cases: dict, an: dict) -> plt.Figure:
    """rows = cases: (a) relative QoI error vs distance outside the box (axis rays filled, random rays open);
    (b) indicators vs distance, each over its in-range threshold; (c) ROC of the indicators (bad = error > tol)."""
    cs = [c for c in cases if "extrapolation" in an.get(c, {})]
    fig, axs = plt.subplots(len(cs), 3, figsize=(15.0, 4.3 * len(cs)), squeeze=False)
    for i, c in enumerate(cs):
        x = an[c]["extrapolation"]
        rr = [r for r in x["records"] if np.isfinite(r["y"])]
        meta = cases[c]["ray_meta"]
        ok = [k for k, r in enumerate(x["records"]) if np.isfinite(r["y"])]
        ax = axs[i][0]
        for k, r in zip(ok, rr, strict=True):
            ax.plot(r["d"], r["rel_err"], "o", ms=4, color=C1, mfc=C1 if meta[k]["axis"] else "none", alpha=0.7)
        b = x["by_distance"]
        ax.plot([e["d"] for e in b], [e["rel_err_median"] for e in b], "-", color=INK, lw=1.4, label="median")
        ax.axhline(x["tol"], color=C2, lw=1.0, ls=(0, (4, 3)), label=f"tolerance {x['tol']:g}")
        ax.set(yscale="log", xlabel="distance outside the training box [decades]", ylabel="relative QoI error")
        ax.set_title(f"(a) {_t(c)}: relative error vs distance outside the box", fontsize=9)
        ax.legend(fontsize=7)
        ax = axs[i][1]
        thr = {k: x["indicators"][k]["threshold"] for k in ("spread", "residual")}
        for k, col in (("spread", C1), ("residual", C3)):
            ax.plot(
                [e["d"] for e in b], [e[f"{k}_median"] / thr[k] for e in b], "-o", color=col, ms=4, mec=SURF, label=k
            )
        ax.axhline(1.0, color=INK2, lw=0.8, ls=(0, (2, 2)), label="in-range threshold")
        ax.set(yscale="log", xlabel="distance [decades]", ylabel="median indicator / threshold")
        ax.set_title("(b) truth-free indicators", fontsize=9)
        ax.legend(fontsize=7)
        ax = axs[i][2]
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8)
        for k, col in (("spread", C1), ("residual", C3), ("combined", C4)):
            det = x["indicators"][k]
            if det["roc"]["fpr"]:
                auc = det["auc"]
                ax.plot(det["roc"]["fpr"], det["roc"]["tpr"], color=col, lw=1.5, label=f"{k} (AUC {auc:.2f})")
        if not any(x["indicators"][k]["roc"]["fpr"] for k in ("spread", "residual", "combined")):
            msg = f"every point has error > {x['tol']:g}\n(no good cases): detection undefined"
            ax.text(0.5, 0.55, msg, ha="center", va="center", fontsize=9, color=INK)
        ax.set(xlabel="false-positive rate", ylabel="true-positive rate", xlim=(0, 1), ylim=(0, 1.02))
        ax.set_title(f"(c) detecting error > {x['tol']:g}", fontsize=9)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="lower right")
    return fig


def slice3d(cases: dict, an: dict) -> plt.Figure:
    """Per case: 3D surface of log10 relative error over the (p1, p2) slice through the box centre, coloured by the
    log10 ensemble spread; the training box drawn on the floor."""
    from matplotlib import cm, colors

    cs = [c for c in cases if "slice" in an.get(c, {})]
    fig = plt.figure(figsize=(7.5 * len(cs), 6.0))
    for j, c in enumerate(cs):
        sl, s = cases[c]["slice_grid"], an[c]["slice"]
        a, b = np.asarray(sl["a"]), np.asarray(sl["b"])
        A, B = np.meshgrid(a, b)
        E = np.log10(np.maximum(np.asarray(s["rel_err"], float).reshape(B.shape), 1e-6))
        S = np.log10(np.maximum(np.asarray(s["spread"], float).reshape(B.shape), 1e-8))
        ax = fig.add_subplot(1, len(cs), j + 1, projection="3d")
        nrm = colors.Normalize(np.nanmin(S), np.nanmax(S))
        ax.plot_surface(
            A,
            B,
            np.where(np.isfinite(E), E, np.nan),
            facecolors=cm.viridis(nrm(S)),
            linewidth=0.2,
            edgecolor=SURF,
            shade=False,
            rcount=len(b),
            ccount=len(a),
        )
        box = cases[c]["box"]
        p1, p2 = sl["pair"]
        z0 = float(np.nanmin(E)) - 0.3
        bx = np.log10([box[p1][0], box[p1][1], box[p1][1], box[p1][0], box[p1][0]])
        by = np.log10([box[p2][0], box[p2][0], box[p2][1], box[p2][1], box[p2][0]])
        ax.plot(bx, by, np.full(5, z0), color=C2, lw=2.0, label="training box")
        ax.contour(
            A, B, np.where(np.isfinite(E), E, np.nan), levels=8, zdir="z", offset=z0, cmap="magma", linewidths=0.7
        )
        ax.set(
            xlabel=f"log10 {p1}", ylabel=f"log10 {p2}", zlabel="log10 rel. error", zlim=(z0, float(np.nanmax(E)) + 0.1)
        )
        m = cm.ScalarMappable(norm=nrm, cmap="viridis")
        fig.colorbar(
            m, ax=ax, shrink=0.5, pad=0.02, orientation="horizontal", label="colour: log10 relative ensemble spread"
        )
        ax.set_title(f"{_t(c)}: QoI error over the ({p1}, {p2}) slice (other parameters at the centre)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.view_init(elev=27, azim=-130)
    return fig


def _short(g: str) -> str:
    return g.replace("pde_", "").replace("bc_", "bc ")


def conflict(cases: dict) -> plt.Figure:
    """rows = cases (first member): cosine matrix of the group gradients at init and final weights; gradient norms."""
    cs = [c for c in cases if cases[c].get("diagnostics")]
    fig, axs = plt.subplots(len(cs), 3, figsize=(15.0, 4.6 * len(cs)), squeeze=False)
    for i, c in enumerate(cs):
        r = cases[c]["diagnostics"][0]
        for k, tag in enumerate(("init", "final")):
            cf = r[tag]["conflict"]
            ax = axs[i][k]
            M = np.asarray(cf["cosine"])
            im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
            g = [_short(n) for n in cf["groups"]]
            ax.set_xticks(range(len(g)), g, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(g)), g, fontsize=7)
            for (p, q), v in np.ndenumerate(M):
                ax.text(q, p, f"{v:+.2f}", ha="center", va="center", fontsize=6, color=INK if abs(v) < 0.6 else SURF)
            ax.grid(False)
            u = cf["unweighted"]
            ax.set_title(
                f"({'ab'[k]}) {_t(c)} / {r['member']}, {tag}: "
                f"cos(PDE, BC) {u['cosine']:+.2f}, cancellation {u['cancellation']:.2f}",
                fontsize=8,
            )
            if k == 1:
                fig.colorbar(im, ax=ax, shrink=0.8, label="cosine of gradients")
        ax = axs[i][2]
        for m, rr in enumerate(cases[c]["diagnostics"]):
            for tag, ls in (("init", (0, (3, 2))), ("final", "-")):
                cf = rr[tag]["conflict"]
                ax.plot(
                    range(len(cf["groups"])),
                    cf["grad_norm"],
                    ls=ls,
                    marker="o",
                    ms=3,
                    color=MEM[m % len(MEM)],
                    label=f"{rr['member']} {tag}",
                )
        ax.set_xticks(range(len(cf["groups"])), [_short(n) for n in cf["groups"]], rotation=45, ha="right", fontsize=7)
        ax.set(yscale="log", ylabel="|gradient of the group loss|")
        ax.set_title("(c) gradient norms per group", fontsize=8)
        ax.legend(fontsize=6, ncol=2)
    return fig


def spectra(cases: dict) -> plt.Figure:
    """rows = cases: (a) residual-NTK eigenvalues (init dashed, final solid; per member); (b) Hessian spectrum at the
    final weights (sorted, symmetric log); (c) loss along the straight line init (s = 0) -> final (s = 1)."""
    cs = [c for c in cases if cases[c].get("diagnostics")]
    fig, axs = plt.subplots(len(cs), 3, figsize=(15.0, 4.2 * len(cs)), squeeze=False)
    for i, c in enumerate(cs):
        for m, r in enumerate(cases[c]["diagnostics"]):
            col = MEM[m % len(MEM)]
            for tag, ls in (("init", (0, (3, 2))), ("final", "-")):
                lam = np.asarray(r[tag]["ntk"]["eigenvalues"])
                lam = lam[lam > 0]
                axs[i][0].loglog(
                    np.arange(1, lam.size + 1),
                    lam,
                    ls=ls,
                    color=col,
                    lw=1.2,
                    label=f"{r['member']} {tag} (eff. rank {r[tag]['ntk']['effective_rank']:.3g})",
                )
            h = r["final"].get("hessian_spectrum")
            if h:
                axs[i][1].plot(np.linspace(0, 1, len(h)), h, color=col, lw=1.2, label=r["member"])
            else:  # full spectrum not computed: the Lanczos extremes
                e = r["final"]["hessian"]
                axs[i][1].plot(
                    [0, 1], [e["min"] or np.nan, e["max"] or np.nan], "o", color=col, label=f"{r['member']} (extremes)"
                )
            it = r["interpolation"]
            axs[i][2].semilogy(it["s"], it["loss"], color=col, lw=1.2, label=r["member"])
        axs[i][0].set(xlabel="index", ylabel="NTK eigenvalue")
        axs[i][0].set_title(f"(a) {_t(c)}: residual NTK spectra", fontsize=9)
        axs[i][0].legend(fontsize=6)
        axs[i][1].set_yscale("symlog", linthresh=1e-3)
        axs[i][1].axhline(0, color=AXIS, lw=0.8)
        axs[i][1].set(xlabel="fraction of eigenvalues (sorted)", ylabel="Hessian eigenvalue")
        axs[i][1].set_title("(b) Hessian spectrum at the final weights", fontsize=9)
        axs[i][1].legend(fontsize=7)
        for s in (0, 1):
            axs[i][2].axvline(s, color=INK2, lw=0.6, ls=(0, (2, 2)))
        axs[i][2].set(xlabel="s (0 = initialisation, 1 = trained)", ylabel="loss (fixed batch)")
        axs[i][2].set_title("(c) loss along the straight line init → final", fontsize=9)
        axs[i][2].legend(fontsize=7)
    return fig


def landscapes3d(cases: dict) -> plt.Figure:
    """rows = cases, columns = members: log10 loss around the final weights over two filter-normalised directions."""
    cs = [c for c in cases if cases[c].get("diagnostics")]
    nm = max(len(cases[c]["diagnostics"]) for c in cs) if cs else 1
    fig = plt.figure(figsize=(4.6 * nm, 4.4 * max(len(cs), 1)))
    for i, c in enumerate(cs):
        for m, r in enumerate(cases[c]["diagnostics"]):
            ls = r["landscape"]
            a = np.asarray(ls["a"])
            A, B = np.meshgrid(a, a)
            Z = np.log10(np.maximum(np.asarray(ls["loss"]), 1e-300))
            ax = fig.add_subplot(len(cs), nm, i * nm + m + 1, projection="3d")
            ax.plot_surface(A, B, Z, cmap="cividis", linewidth=0, rcount=len(a), ccount=len(a))
            ax.contour(A, B, Z, levels=10, zdir="z", offset=float(Z.min()) - 0.5, cmap="cividis", linewidths=0.6)
            ax.scatter([0], [0], [Z[len(a) // 2, len(a) // 2]], color=C2, s=25, depthshade=False)
            ax.set(xlabel="d1", ylabel="d2", zlabel="log10 loss", zlim=(float(Z.min()) - 0.5, float(Z.max())))
            ax.set_title(f"{_t(c)} / {r['member']}", fontsize=8)
            ax.view_init(elev=32, azim=-55)
    return fig


__all__ = ["uq", "uq_field3d", "extrapolation", "slice3d", "conflict", "spectra", "landscapes3d", "style", "_plain_log"]
