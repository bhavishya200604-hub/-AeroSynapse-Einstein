"""Step 6.5 figures: spectral bias (2D / 3D), NTK spectra and eigenfunctions (2D / 3D waterfalls), training dynamics,
Hessian spectra, loss landscapes with the optimisation path (2D / 3D), width and Fourier-sigma studies, toy problem."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from aerosynapse.thermal.viz.inverse_figs import AXIS, GRID, INK, INK2, SURF, style  # noqa: F401  (style re-exported)

VAR = ["#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#b04a1e", "#7fb3e8"]  # fixed order of variants


def _grid(cases):
    cs = list(dict.fromkeys(r["case"] for r in cases))
    vs = list(dict.fromkeys(r["variant"] for r in cases))
    get = {(r["case"], r["variant"]): r for r in cases}
    return cs, vs, get


def _vcol(vs, v):
    return VAR[vs.index(v) % len(VAR)]


def _plain_log(ax, which: str = "x") -> None:
    """Log axes spanning < 2 decades: plain labels, no minor labels (avoids crowded 2x10^0 3x10^0 ... ticks)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    for w in which:
        axis = ax.xaxis if w == "x" else ax.yaxis
        lo, hi = ax.get_xlim() if w == "x" else ax.get_ylim()
        if lo > 0 and hi / lo < 100:
            axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())


def bands(spec, n_modes: int | None = None) -> tuple[np.ndarray, list[str]]:
    """Octave bands of DCT modes: [0], [1], [2, 3], [4, 7], ... -> rms amplitude per band (rows = snapshots)."""
    S = np.asarray(spec, float)
    n = S.shape[-1] if n_modes is None else min(n_modes, S.shape[-1])
    edges = [0, 1, 2]
    while edges[-1] < n:
        edges.append(min(2 * edges[-1], n))
    out = np.stack(
        [np.sqrt(np.mean(S[..., a:b] ** 2, axis=-1)) for a, b in zip(edges[:-1], edges[1:], strict=True)], -1
    )
    labels = [f"{a}" if b - a == 1 else f"{a}-{b - 1}" for a, b in zip(edges[:-1], edges[1:], strict=True)]
    return out, labels


def _adam_end(r) -> int:
    return next((i for i, (ph, _) in enumerate(r["steps"]) if ph == "lbfgs"), len(r["steps"]))


# --------------------------------------------------------------------------- spectral bias
def spectral_surfaces(cases: list[dict], modes: int = 256) -> plt.Figure:
    """rows = cases, columns = variants: log10 |DCT coefficient of the error| over (snapshot, mode) as a 3D surface."""
    cs, vs, get = _grid(cases)
    fig = plt.figure(figsize=(4.2 * len(vs), 3.6 * len(cs)))
    for i, c in enumerate(cs):
        for j, v in enumerate(vs):
            r = get.get((c, v))
            ax = fig.add_subplot(len(cs), len(vs), i * len(vs) + j + 1, projection="3d")
            if r is None:
                continue
            B, labs = bands(r["spectrum"], modes)
            S = np.log10(np.maximum(B, 1e-12))
            Xs, Ks = np.meshgrid(np.arange(S.shape[0]), np.arange(S.shape[1]), indexing="ij")
            ax.plot_surface(Xs, Ks, S, cmap="viridis", linewidth=0.2, edgecolor=SURF, rcount=80, ccount=20, alpha=0.95)
            ax.set_yticks(range(len(labs))[::2], labs[::2], fontsize=6)
            ax.set(xlabel="snapshot", ylabel="DCT mode band", zlabel="log10 rms |ê|")
            ax.set_title(f"{c} / {v}", fontsize=8)
            ax.view_init(elev=28, azim=-55)
    fig.suptitle("Error spectrum during training (which frequencies are learnt when)", x=0.01, ha="left", fontsize=10)
    return fig


def error_spectra(cases: list[dict], modes: int = 256) -> plt.Figure:
    """rows = cases, columns = variants: heat map of log10 |ê_k| over (snapshot, mode); white line = Adam → L-BFGS."""
    cs, vs, get = _grid(cases)
    fig, axs = plt.subplots(
        len(cs), len(vs), figsize=(3.4 * len(vs), 2.8 * len(cs)), squeeze=False, layout="constrained"
    )
    allS = [np.log10(np.maximum(bands(r["spectrum"], modes)[0], 1e-12)) for r in cases]
    norm = mcolors.Normalize(np.percentile(np.concatenate([s.ravel() for s in allS]), 2), max(s.max() for s in allS))
    for i, c in enumerate(cs):
        for j, v in enumerate(vs):
            ax = axs[i][j]
            r = get.get((c, v))
            if r is None:
                ax.axis("off")
                continue
            B, labs = bands(r["spectrum"], modes)
            S = np.log10(np.maximum(B, 1e-12))
            im = ax.imshow(S.T, origin="lower", aspect="auto", cmap="magma", norm=norm, interpolation="nearest")
            ax.set_yticks(range(len(labs)), labs, fontsize=6)
            ax.axvline(_adam_end(r) - 0.5, color="white", lw=0.8, ls=(0, (3, 2)))
            ax.set_title(f"{c} / {v}: final rel L2 {r['rel_l2'][-1]:.2g}", fontsize=8)
            ax.set(xlabel="snapshot", ylabel="DCT mode band" if j == 0 else None)
            ax.grid(False)
    fig.colorbar(im, ax=axs, shrink=0.6, label="log10 rms |DCT coefficient of the error| per octave band")
    return fig


# --------------------------------------------------------------------------- NTK
def ntk_spectra(cases: list[dict], k: int = 120) -> plt.Figure:
    """rows = cases, columns = variants: NTK eigenvalues (descending, log) at each NTK snapshot, light → dark = time."""
    cs, vs, get = _grid(cases)
    fig, axs = plt.subplots(len(cs), len(vs), figsize=(3.4 * len(vs), 2.7 * len(cs)), squeeze=False, sharey="row")
    for i, c in enumerate(cs):
        for j, v in enumerate(vs):
            ax = axs[i][j]
            r = get.get((c, v))
            if r is None:
                ax.axis("off")
                continue
            n = len(r["ntk"])
            for m, e in enumerate(r["ntk"]):
                lam = np.asarray(e["eigenvalues"])[:k]
                ax.plot(
                    np.arange(1, len(lam) + 1),
                    np.maximum(lam, 1e-16),
                    color=plt.cm.Blues(0.3 + 0.7 * m / max(n - 1, 1)),
                    lw=1.1,
                )
            ax.set(yscale="log", xscale="log", xlabel="eigenvalue index")
            ax.set_title(f"{c} / {v}", fontsize=8)
            if j == 0:
                ax.set_ylabel("NTK eigenvalue")
    hs = [
        plt.Line2D([], [], color=plt.cm.Blues(0.3), lw=1.5, label="initial"),
        plt.Line2D([], [], color=plt.cm.Blues(1.0), lw=1.5, label="final"),
    ]
    fig.legend(handles=hs, loc="outside lower center", ncol=2, fontsize=8)
    return fig


def ntk_waterfall(cases: list[dict], k: int = 80) -> plt.Figure:
    """rows = cases, columns = variants: 3D waterfall of log10 NTK eigenvalue vs (index, NTK snapshot)."""
    cs, vs, get = _grid(cases)
    fig = plt.figure(figsize=(4.0 * len(vs), 3.5 * len(cs)))
    for i, c in enumerate(cs):
        for j, v in enumerate(vs):
            r = get.get((c, v))
            ax = fig.add_subplot(len(cs), len(vs), i * len(vs) + j + 1, projection="3d")
            if r is None:
                continue
            n = len(r["ntk"])
            for m, e in enumerate(r["ntk"]):
                lam = np.log10(np.maximum(np.asarray(e["eigenvalues"])[:k], 1e-16))
                idx = np.arange(1, len(lam) + 1)
                col = plt.cm.viridis(m / max(n - 1, 1))
                ax.plot(np.log10(idx), np.full_like(lam, m), lam, color=col, lw=1.2)
            ax.set(xlabel="log10 index", ylabel="NTK snapshot", zlabel="log10 λ")
            ax.set_title(f"{c} / {v}", fontsize=8)
            ax.view_init(elev=25, azim=-60)
    fig.suptitle("NTK spectrum evolution during training", x=0.01, ha="left", fontsize=10)
    return fig


def ntk_eigenfunctions(cases: list[dict], n_show: int = 5) -> plt.Figure:
    """rows = cases, columns = variants: leading eigenvectors of the PDE block of the NTK over x (final time slice for
    transient problems) — initial (dashed) and final (solid). Smooth leading modes = spectral bias."""
    cs, vs, get = _grid(cases)
    fig, axs = plt.subplots(len(cs), len(vs), figsize=(3.4 * len(vs), 2.6 * len(cs)), squeeze=False)
    for i, c in enumerate(cs):
        for j, v in enumerate(vs):
            ax = axs[i][j]
            r = get.get((c, v))
            if r is None:
                ax.axis("off")
                continue
            for e, ls in ((r["ntk"][0], (0, (3, 2))), (r["ntk"][-1], "-")):
                V = np.asarray(e.get("pde_eigenvectors") or [])
                for m, vec in enumerate(V[:n_show]):
                    if r["problem"] != "A_steady":  # transient: (n_t, n_x) ordering -> last time slice
                        vec = np.asarray(vec).reshape(int(r["points"]["n_t"]), -1)[-1]
                    xs = np.linspace(0, 1, len(vec) + 2)[1:-1]
                    s = np.sign(vec[np.argmax(np.abs(vec))])
                    ax.plot(
                        xs, s * np.asarray(vec) / np.max(np.abs(vec)) + 2.2 * m, color=VAR[m % len(VAR)], lw=1.0, ls=ls
                    )
            ax.set_yticks([])
            ax.set(xlabel="x")
            ax.set_title(f"{c} / {v}", fontsize=8)
    hs = [
        plt.Line2D([], [], color=INK2, ls=(0, (3, 2)), label="initial"),
        plt.Line2D([], [], color=INK2, label="final"),
    ]
    fig.legend(
        handles=hs,
        loc="outside lower center",
        ncol=2,
        fontsize=8,
        title="leading NTK eigenvectors (offset; mode 1 at the bottom)",
    )
    return fig


# --------------------------------------------------------------------------- dynamics and Hessian
def dynamics(cases: list[dict]) -> plt.Figure:
    """rows = cases; columns: error, loss, NTK drift, PDE / BC trace ratio, Hessian λmax over training."""
    cs, vs, get = _grid(cases)
    fig, axs = plt.subplots(len(cs), 5, figsize=(17.5, 2.9 * len(cs)), squeeze=False)
    for i, c in enumerate(cs):
        for v in vs:
            r = get.get((c, v))
            if r is None:
                continue
            col = _vcol(vs, v)
            s = np.arange(len(r["rel_l2"]))
            axs[i][0].plot(s, r["rel_l2"], color=col, lw=1.3)
            axs[i][1].plot(s, r["loss"], color=col, lw=1.3)
            sn = [e["snapshot"] for e in r["ntk"]]
            dr = [(s_, e["drift"]) for s_, e in zip(sn, r["ntk"], strict=True) if s_ > 0]  # drift is 0 at the start
            axs[i][2].plot([d[0] for d in dr], [d[1] for d in dr], "-o", color=col, ms=3, mec=SURF)
            tr = [
                e["trace_per_point"]["pde"]
                / np.mean([q for k, q in e["trace_per_point"].items() if k.startswith("bc")])
                for e in r["ntk"]
            ]
            axs[i][3].plot(sn, tr, "-o", color=col, ms=3, mec=SURF)
            hm = [(e.get("hessian") or {}).get("max") for e in r["ntk"]]
            if all(h is not None for h in hm):
                axs[i][4].plot(sn, np.maximum(hm, 1e-12), "-o", color=col, ms=3, mec=SURF)
            for ax in axs[i]:
                ax.axvline(_adam_end(r) - 0.5, color=GRID, lw=0.8)
        for ax, lab in zip(
            axs[i],
            (
                "relative L2 error",
                "loss (truth-free)",
                "NTK drift ‖K−K₀‖/‖K₀‖",
                "trace per point PDE / BC",
                "Hessian λmax",
            ),
            strict=True,
        ):
            ax.set(yscale="log", xlabel="snapshot")
            ax.set_title(f"{c}: {lab}", fontsize=8)
    hs = [plt.Line2D([], [], color=_vcol(vs, v), lw=1.5, label=v) for v in vs]
    hs.append(plt.Line2D([], [], color=GRID, lw=1.0, label="Adam → L-BFGS"))
    fig.legend(handles=hs, loc="outside lower center", ncol=len(hs), fontsize=8)
    return fig


def hessian(cases: list[dict]) -> plt.Figure:
    """rows = cases: full Hessian eigenvalue distributions at the final weights (symmetric-log x), per variant."""
    cs, vs, get = _grid(cases)
    fig, axs = plt.subplots(1, len(cs), figsize=(5.0 * len(cs), 3.6), squeeze=False)
    for ax, c in zip(axs[0], cs, strict=True):
        for v in vs:
            r = get.get((c, v))
            if r is None or not r.get("hessian_spectrum"):
                continue
            lam = np.asarray(r["hessian_spectrum"])
            t = np.sign(lam) * np.log10(1 + np.abs(lam))
            bins = np.linspace(-8, 8, 161)
            h, e = np.histogram(t, bins=bins)
            ax.step(
                0.5 * (e[1:] + e[:-1]),
                np.maximum(h, 0.5),
                color=_vcol(vs, v),
                lw=1.2,
                where="mid",
                label=f"{v}: λmax {lam[-1]:.3g}, λmin {lam[0]:.2g}",
            )
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set(yscale="log", xlabel="sign(λ) log10(1 + |λ|)", ylabel="count")
        ax.set_title(f"{c}: full Hessian spectrum (final weights)", fontsize=8)
        ax.legend(fontsize=6)
    return fig


# --------------------------------------------------------------------------- landscapes
def landscapes3d(cases: list[dict]) -> plt.Figure:
    """rows = cases, columns = variants: log10 loss over the 2D slice (surface) with the training path on it."""
    cs, vs, get = _grid(cases)
    fig = plt.figure(figsize=(4.2 * len(vs), 3.7 * len(cs)))
    for i, c in enumerate(cs):
        for j, v in enumerate(vs):
            r = get.get((c, v))
            ax = fig.add_subplot(len(cs), len(vs), i * len(vs) + j + 1, projection="3d")
            if r is None:
                continue
            ls = r["landscape"]
            A, B = np.meshgrid(ls["a"], ls["b"])
            Z = np.log10(np.maximum(np.asarray(ls["loss"]), 1e-14))
            ax.plot_surface(A, B, Z, cmap="viridis", linewidth=0, alpha=0.85, rcount=40, ccount=40)
            if r.get("trajectory"):
                tr = np.asarray(r["trajectory"])
                zt = np.log10(np.maximum(np.asarray(r["loss"]), 1e-14))
                ax.plot(tr[:, 0], tr[:, 1], zt, color="#eb6834", lw=1.6)
                ax.scatter([tr[0, 0]], [tr[0, 1]], [zt[0]], color="#eb6834", s=12)
                ax.scatter([0], [0], [zt[-1]], color=INK, s=14, marker="*")
            ax.set(
                xlabel="PC 1" if ls["directions"] == "pca" else "d1",
                ylabel="PC 2" if ls["directions"] == "pca" else "d2",
                zlabel="log10 loss",
            )
            ax.set_title(f"{c} / {v}", fontsize=8)
            ax.view_init(elev=32, azim=-50)
    fig.suptitle(
        "Loss landscape around the trained weights and the optimisation path (orange)", x=0.01, ha="left", fontsize=10
    )
    return fig


def landscapes2d(cases: list[dict]) -> plt.Figure:
    """rows = cases, columns = variants: contours of log10 loss, training path (start ●, end ★); bottom row of the
    figure legend explains the markers. A narrow valley along one direction = ill-conditioning."""
    cs, vs, get = _grid(cases)
    fig, axs = plt.subplots(
        len(cs), len(vs), figsize=(3.4 * len(vs), 3.0 * len(cs)), squeeze=False, layout="constrained"
    )
    for i, c in enumerate(cs):
        for j, v in enumerate(vs):
            ax = axs[i][j]
            r = get.get((c, v))
            if r is None:
                ax.axis("off")
                continue
            ls = r["landscape"]
            A, B = np.meshgrid(ls["a"], ls["b"])
            Z = np.log10(np.maximum(np.asarray(ls["loss"]), 1e-14))
            cf = ax.contourf(A, B, Z, levels=24, cmap="viridis")
            ax.contour(A, B, Z, levels=12, colors="white", linewidths=0.4, alpha=0.6)
            if r.get("trajectory"):
                tr = np.asarray(r["trajectory"])
                ax.plot(tr[:, 0], tr[:, 1], color="#eb6834", lw=1.3)
                ax.plot(tr[0, 0], tr[0, 1], "o", color="#eb6834", ms=5, mec=SURF)
            ax.plot(0, 0, "*", color="white", ms=10, mec=INK)
            pc = ls["directions"] == "pca"
            ax.set(xlabel="PC 1" if pc else "d1", ylabel="PC 2" if pc else "d2")
            ax.set_title(f"{c} / {v}", fontsize=8)
            ax.grid(False)
            fig.colorbar(cf, ax=ax, shrink=0.8)
    return fig


# --------------------------------------------------------------------------- width and sigma; toy
def width_sigma(width: list[dict], sigma: list[dict]) -> plt.Figure:
    """(a) NTK drift during training per width; (b) final drift vs width (log-log); (c) output-NTK alignment with a
    coarse FD solution at initialisation vs Fourier σ; (d) error after training vs σ."""
    fig, axs = plt.subplots(1, 4, figsize=(16.0, 3.6), squeeze=False)
    ax = axs[0]
    for k, w in enumerate(sorted(width, key=lambda z: z["width"])):
        ax[0].plot(  # from snapshot 1: the drift is 0 at the start
            np.arange(1, len(w["drift"])),
            w["drift"][1:],
            "-o",
            ms=3,
            color=VAR[k % len(VAR)],
            label=f"width {w['width']}",
        )
    ax[0].set(yscale="log", xlabel="snapshot", ylabel="NTK drift ‖K−K₀‖/‖K₀‖")
    ax[0].legend(fontsize=7)
    if width:
        ws = sorted(width, key=lambda z: z["width"])
        ax[1].plot(
            [w["width"] for w in ws], [w["drift"][-1] for w in ws], "-o", color=INK, ms=5, mec=SURF, label="NTK drift"
        )
        ax[1].plot(
            [w["width"] for w in ws],
            [w["param_change"] for w in ws],
            "-s",
            color="#1baf7a",
            ms=5,
            mec=SURF,
            label="relative weight change",
        )
        ax[1].set(xscale="log", yscale="log", xlabel="width", ylabel="relative change (end of training)")
        ax[1].legend(fontsize=7)
    if sigma:
        ss = sorted(sigma, key=lambda z: z["sigma"])
        sg = [s["sigma"] for s in ss]
        if all(s["alignment_fd"] is not None for s in ss):
            ax[2].plot(sg, [s["alignment_fd"] for s in ss], "-o", color="#2a78d6", ms=5, mec=SURF)
        ax[2].set(xscale="log", xlabel="Fourier σ", ylabel="kernel–target alignment (init)")
        if all(s.get("rel_l2") is not None for s in ss):
            ax[3].plot(sg, [s["rel_l2"] for s in ss], "-o", color="#eb6834", ms=5, mec=SURF)
        ax[3].set(xscale="log", yscale="log", xlabel="Fourier σ", ylabel="relative L2 after training")
    for a_ in ax[1:]:
        _plain_log(a_, "x")
    _plain_log(ax[1], "y")
    for a_, t in zip(
        ax,
        (
            "(a) kernel drift during training",
            "(b) lazy regime: change vs width",
            "(c) a-priori NTK alignment",
            "(d) trained error",
        ),
        strict=True,
    ):
        a_.set_title(t, fontsize=9)
    return fig


def toy(toy_runs: list[dict]) -> plt.Figure:
    """top: 3D surface of the residual spectrum (epoch, frequency, amplitude) per variant; bottom: residual amplitude
    of each target frequency vs epoch (spectral bias: low frequencies first)."""
    n = len(toy_runs)
    fig = plt.figure(figsize=(4.2 * n, 7.4))
    for j, t in enumerate(toy_runs):
        ax = fig.add_subplot(2, n, j + 1, projection="3d")
        S = np.asarray(t["spectrum"])
        E, Kf = np.meshgrid(t["epochs"], np.arange(S.shape[1]), indexing="ij")
        ax.plot_surface(
            E, Kf, np.log10(np.maximum(S, 1e-3)), cmap="viridis", linewidth=0, rcount=60, ccount=64
        )  # floor 1e-3: display only
        ax.set(xlabel="epoch", ylabel="frequency", zlabel="log10 amplitude")
        ax.set_title(f"toy fit: {t['variant']}", fontsize=8)
        ax.view_init(elev=28, azim=-60)
        ax2 = fig.add_subplot(2, n, n + j + 1)
        amp = np.asarray(t["amplitude"])
        for m, k in enumerate(t["frequencies"]):
            ax2.plot(t["epochs"], amp[:, m], color=VAR[m % len(VAR)], lw=1.4, label=f"k = {k}")
        ax2.set(yscale="log", xlabel="epoch", ylabel="residual amplitude")
        ax2.legend(fontsize=7)
    return fig
