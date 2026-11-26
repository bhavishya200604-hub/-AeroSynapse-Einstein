"""Learning scenes: how the networks train and how the inverse estimates converge, replayed from SAVED results (no
training): the optimiser's path on the loss landscape and the error's frequency content (Step 6.5 diagnostics), the
loss histories at the short and the full training budget (Steps 6.1 and 6B G3 logs), and the parameter estimates of
least squares and the PINN variants against the truth (Phase 5 gate)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from aerosynapse.thermal.viz import style as S
from aerosynapse.thermal.viz.anim.core import Scene

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
LABEL = {"mlp": "tanh MLP", "fourier": "Fourier features", "siren": "SIREN", "mlp_ntk_weights": "MLP + NTK weights"}


def _diag_cases(case: str, variants: list[str]) -> list[dict]:
    met = json.loads((ROOT / "results/p6_5_diagnostics_default/metrics.json").read_text(encoding="utf-8"))
    by = {(c["case"], c["variant"]): c for c in met["cases"]}
    return [by[(case, v)] for v in variants]


# ---------------------------------------------------------------------------- optimiser path on the loss landscape
def scene_training_landscape(p: dict, cache: Path) -> Scene:
    cs = _diag_cases(p["case"], p["variants"])
    n_snap = len(cs[0]["trajectory"])
    sub = int(p["substeps"])
    n_frames = (n_snap - 1) * sub + 1

    def pos(c, i):
        tr = np.asarray(c["trajectory"])
        k, f = divmod(i, sub)
        if k >= len(tr) - 1:
            return tr[-1]
        return tr[k] + (tr[k + 1] - tr[k]) * f / sub

    def make():
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10.0, 5.6))
        gs = fig.add_gridspec(2, len(cs), height_ratios=[1.35, 1])
        dots, paths, marks = [], [], []
        ax_e = fig.add_subplot(gs[1, :])
        for j, c in enumerate(cs):
            ax = fig.add_subplot(gs[0, j])
            ls = c["landscape"]
            Z = np.log10(np.maximum(np.asarray(ls["loss"]), 1e-14))
            cf = ax.contourf(ls["a"], ls["b"], Z, levels=24, cmap=S.SEQ_CMAP)
            ax.contour(ls["a"], ls["b"], Z, levels=12, colors="white", linewidths=0.3, alpha=0.5)
            S.colorbar(fig, cf, ax, r"$\log_{10}$ loss", shrink=0.85)
            (pl,) = ax.plot([], [], color="white", lw=1.4)
            (dt,) = ax.plot([], [], "o", color=S.PINN, ms=7, mec="white", mew=1.2)
            ax.plot(0, 0, "*", color="white", ms=11, mec=S.INK, mew=0.6)
            ax.set(title=f"{LABEL.get(c['variant'], c['variant'])}", xlabel="direction 1", ylabel="direction 2")
            ax.grid(False)
            paths.append(pl)
            dots.append(dt)
            y = np.asarray(c["rel_l2"])
            ax_e.plot(np.arange(len(y)), y, color=S.PALETTE[j], label=LABEL.get(c["variant"], c["variant"]))
            (mk,) = ax_e.plot([], [], "o", color=S.PALETTE[j], ms=6)
            marks.append(mk)
        ax_e.set(yscale="log", xlabel="training snapshot (Adam, then L-BFGS)", ylabel="relative error vs analytic")
        ax_e.legend(loc="center left", fontsize=8)
        cur = ax_e.axvline(0, color=S.MUTED, lw=0.8)
        sup = fig.suptitle("")

        def update(i):
            k = min(i // sub, n_snap - 1)
            for j, c in enumerate(cs):
                pts = np.array([pos(c, q) for q in range(i + 1)])
                paths[j].set_data(pts[:, 0], pts[:, 1])
                x, y = pos(c, i)
                dots[j].set_data([x], [y])
                marks[j].set_data([k], [c["rel_l2"][k]])
            cur.set_xdata([k, k])
            ph, st = cs[0]["steps"][k]
            sup.set_text(f"Training on the loss landscape, steady problem Pe = 100   ({ph} step {st})")

        return fig, update

    return Scene(
        "training_landscape",
        "Training on the loss landscape",
        "The optimiser's path (white) in the plane of the training trajectory, ending at the final weights (star), "
        "for three architectures on the same problem; the tanh MLP stalls in a flat valley with a large error, "
        "Fourier features and SIREN descend into narrow basins. Replayed from saved weight snapshots.",
        n_frames,
        p["fps"],
        make,
        "results/p6_5_diagnostics_default (Step 6.5, one seed per architecture)",
        category="learning",
    )


# ---------------------------------------------------------------------------- spectral bias during training
def scene_spectral_bias(p: dict, cache: Path) -> Scene:
    cs = _diag_cases(p["case"], p["variants"])
    km = int(p["kmax"])
    n = len(cs[0]["spectrum"])

    def make():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.6, 4.0))
        k = np.arange(km)
        tgt = np.asarray(cs[0]["spectrum_target"])[:km]
        ax.plot(k, np.maximum(tgt, 1e-16), color=S.MUTED, lw=1, ls="--", label="solution itself")
        lines = [
            ax.plot([], [], color=S.PALETTE[j], label=LABEL.get(c["variant"], c["variant"]))[0]
            for j, c in enumerate(cs)
        ]
        allv = np.concatenate([np.asarray(c["spectrum"])[:, :km].ravel() for c in cs] + [tgt])
        allv = allv[allv > 0]
        ax.set(
            yscale="log",
            xlim=(0, km - 1),
            ylim=(np.percentile(allv, 1), allv.max() * 2),
            xlabel="frequency index of the error (discrete cosine transform)",
            ylabel="error amplitude",
        )
        ax.legend(loc="upper right", fontsize=8)
        title = ax.set_title("")

        def update(i):
            for j, c in enumerate(cs):
                ln = np.maximum(np.asarray(c["spectrum"][i])[:km], 1e-16)
                lines[j].set_data(np.arange(km), ln)
            ph, st = cs[0]["steps"][i]
            title.set_text(f"Frequency content of the error during training ({ph} step {st})")

        return fig, update

    return Scene(
        "spectral_bias",
        "Spectral bias during training",
        "Error spectrum of each network against the exact solution, snapshot by snapshot: high frequencies of "
        "the error persist longest for the tanh MLP (spectral bias); Fourier features and SIREN reduce them "
        "early. Replayed from saved snapshots.",
        n,
        p["fps"],
        make,
        "results/p6_5_diagnostics_default (Step 6.5, one seed per architecture)",
        category="learning",
    )


# ---------------------------------------------------------------------------- loss histories, short vs full budget
def _log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    keys = [k for k in rows[0] if k.split("_")[0] in ("pde", "ic", "bc", "data") and not k.startswith("w_")]
    w = np.array([float(r["wall_s"]) for r in rows])
    loss = np.array([sum(float(r[k]) for k in keys if r[k] not in ("", None)) for r in rows])
    return w, loss


def scene_training_budget(p: dict, cache: Path) -> Scene:
    sets = {
        "slab, short budget": [
            ROOT / f"results/p6_1_scaling_default/runs/B_slab_base_base_s{s}/log.csv" for s in range(3)
        ],
        "slab, full budget": [ROOT / f"results/p6b_g3_budget_default/runs/B_slab_s{s}/log.csv" for s in range(3)],
        "Graetz, short budget": [ROOT / "results/p4_7_ablation_smoke/runs/full-exact_psi_s0/log.csv"],
        "Graetz, full budget": [ROOT / "results/p4_7_ablation_g3_full_s0/runs/full-exact_psi_s0/log.csv"],
    }
    data = {k: [_log(f) for f in v if f.exists()] for k, v in sets.items()}
    tmax = max(w.max() for v in data.values() for w, _ in v)
    tmin = min(w[w > 0].min() for v in data.values() for w, _ in v)
    times = np.geomspace(tmin, tmax, p["frames"])
    col = {"short": S.SOLVER, "full": S.PINN}

    def make():
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9))
        lines = []
        for k, runs in data.items():
            ax = axs[0] if k.startswith("slab") else axs[1]
            c = col["short" if "short" in k else "full"]
            for j, _ in enumerate(runs):
                (ln,) = ax.plot([], [], color=c, lw=1.1, label=k.split(", ")[1] if j == 0 else None)
                lines.append((ln, runs[j]))
        for ax, t in zip(axs, ("slab, Bi = 1 (three seeds per budget)", "Graetz channel, seed 0"), strict=True):
            lo = min(lv.min() for (ln, (w, lv)) in lines if ln.axes is ax)
            hi = max(lv.max() for (ln, (w, lv)) in lines if ln.axes is ax)
            ax.set(
                xscale="log",
                yscale="log",
                xlim=(tmin, tmax),
                ylim=(lo / 2, hi * 2),
                title=t,
                xlabel="training wall time [s]",
                ylabel="sum of residual losses",
            )
            ax.legend(loc="lower left", fontsize=8)
        cur = [ax.axvline(tmin, color=S.MUTED, lw=0.8) for ax in axs]
        sup = fig.suptitle("")

        def update(i):
            tt = times[i]
            for ln, (w, lv) in lines:
                m = w <= tt
                ln.set_data(w[m], lv[m])
            for c in cur:
                c.set_xdata([tt, tt])
            sup.set_text(f"Training at the short and the full budget   wall time {tt:,.0f} s")

        return fig, update

    return Scene(
        "training_budget",
        "Short versus full training budget",
        "Training losses replayed against wall time. For the slab the full budget keeps lowering the loss (and "
        "the error, 18-fold); for the parametric Graetz channel the loss also falls, but the error rose, because "
        "the larger network fitted its collocation points and not the equations between them.",
        len(times),
        p["fps"],
        make,
        "training logs of results/p6_1_scaling_default, p6b_g3_budget_default, p4_7_ablation_smoke, "
        "p4_7_ablation_g3_full_s0",
        category="learning",
    )


# ---------------------------------------------------------------------------- inverse estimates converging
def scene_inverse_convergence(p: dict, cache: Path) -> Scene:
    from aerosynapse.thermal.experiments import p5_gate as G

    g = G.gather()
    hist, truth, crlb = g["histories"], g["truth"], g["crlb_log"]
    qs = ("alpha", "k", "h", "Bi")
    lab = {"alpha": r"$\alpha$", "k": "$k$", "h": "$h$", "Bi": r"$\mathrm{Bi} = hL/k$"}

    def ratio(hh, q):
        if q == "Bi":
            return np.array([(x["h"] / x["k"]) / (truth["h"] / truth["k"]) for x in hh])
        return np.array([x[q] / truth[q] for x in hh])

    series = []
    for name, hs in hist.items():
        ls = name.startswith("least squares")
        for hh in hs:
            series.append((name, ls, "joint_w100" in name, {q: ratio(hh, q) for q in qs}))
    fr = np.linspace(0, 1, p["frames"])

    def make():
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 4, figsize=(11.0, 3.6), sharey=True)
        arts = []
        for ax, q in zip(axs, qs, strict=True):
            b = float(np.exp(1.96 * crlb[q]))
            ax.axhspan(1 / b, b, color="#e8eef5", lw=0, label="95 % Cramér-Rao band")
            ax.axhline(1, color=S.AXIS, lw=0.9)
            for _name, ls, emph, r in series:
                c = S.REF if ls else (S.PINN if emph else S.MUTED)
                (ln,) = ax.plot([], [], color=c, lw=1.6 if (ls or emph) else 0.7, alpha=1 if (ls or emph) else 0.6)
                arts.append((ln, r[q]))
            ax.set(yscale="log", xlim=(0, 1), title=lab[q], xlabel="fraction of the iterations")
        axs[0].set_ylabel("estimate / truth")
        axs[0].plot([], [], color=S.REF, label="least squares")
        axs[0].plot([], [], color=S.PINN, label="PINN (joint, w = 100)")
        axs[0].plot([], [], color=S.MUTED, label="other PINN variants")
        axs[0].legend(loc="upper left", fontsize=7)
        allv = np.concatenate([a[1] for a in arts])
        allv = allv[np.isfinite(allv) & (allv > 0)]
        axs[0].set_ylim(max(allv.min() / 1.5, 1e-3), min(allv.max() * 1.5, 1e4))
        sup = fig.suptitle("")

        def update(i):
            f = fr[i]
            for ln, y in arts:
                n = max(1, int(round(f * (len(y) - 1))) + 1)
                ln.set_data(np.linspace(0, 1, len(y))[:n], y[:n])
            sup.set_text(f"Identifying $\\alpha$, $k$ and $h$ of a heated slab   {100 * f:.0f} % of the iterations")

        return fig, update

    return Scene(
        "inverse_convergence",
        "Parameter identification: least squares versus PINN",
        "Estimates divided by the true values while each method iterates. Least squares reaches the truth "
        "within the Cramér-Rao band for all three parameters; the PINN variants recover the Biot number but "
        "not k and h separately, which the identifiability analysis predicts.",
        len(fr),
        p["fps"],
        make,
        "results/p5_* via experiments/p5_gate.gather() (Phase 5)",
        category="learning",
    )


SCENES = {
    "training_landscape": scene_training_landscape,
    "spectral_bias": scene_spectral_bias,
    "training_budget": scene_training_budget,
    "inverse_convergence": scene_inverse_convergence,
}
