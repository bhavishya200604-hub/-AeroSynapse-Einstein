"""ML / optimisation clips: PINN training live, FNO predictions (saved model), airfoil morph along history."""

from __future__ import annotations

import numpy as np
from _base import CLIPS, RES, load

from aerosynapse.viz import neon as N

FPS = 30


def pinn_training() -> str:
    import torch

    from aerosynapse.ml.nets import MLP, set_seed
    from aerosynapse.ml.pinn import PINNConfig, predict, train_pinn
    from aerosynapse.ml.problems import ConvDiff1D
    from aerosynapse.ml.sampling import RAD

    class RecRAD(RAD):
        last = None

        def initial(self):  # type: ignore[no-untyped-def]
            self.last = super().initial()
            return self.last

        def update(self, X, residual, step):  # type: ignore[no-untyped-def]
            self.last = super().update(X, residual, step)
            return self.last

    torch.set_num_threads(2)
    set_seed(0)
    p = ConvDiff1D(pe=100.0, sigma0=0.1, t_final=1.0)
    m = MLP(2, 1, 32, 3, lo=p.lo, hi=p.hi, periodic_dims=(0,))
    smp = RecRAD(p.lo, p.hi, 2000, 0)
    G = p.eval_grid(81)
    ref = p.exact(G).reshape(81, 81)
    snaps: list[tuple[int, np.ndarray, np.ndarray, float]] = []
    losses: list[tuple[int, float]] = []

    def cb(it: int, row: dict) -> None:
        losses.append((it, row["loss"]))
        snaps.append((it, predict(p, m, G).reshape(81, 81), smp.last.copy(), row["loss"]))

    train_pinn(
        p,
        m,
        PINNConfig(iterations=4000, n_collocation=2000, sampler="rad", resample_every=250, lr_decay=0.9, log_every=40),
        sampler=smp,
        callback=cb,
    )
    fig = N.plt.figure(figsize=(N.W, N.H), dpi=N.DPI)
    N.header(
        fig,
        "Physics-informed neural network learning convection–diffusion (Pe = 100)",
        "MLP 3×32 · residual-based adaptive collocation (RAD) · prediction vs exact as training proceeds",
        "computed live: aerosynapse.ml.pinn.train_pinn (4000 Adam iterations, 2000 collocation points)",
    )
    a1 = fig.add_axes([0.05, 0.1, 0.28, 0.72])
    a2 = fig.add_axes([0.37, 0.1, 0.28, 0.72])
    a3 = fig.add_axes([0.71, 0.5, 0.27, 0.32])
    a4 = fig.add_axes([0.71, 0.1, 0.27, 0.3])
    ext = (0, 1, 0, 1)
    im1 = a1.imshow(snaps[0][1].T, origin="lower", extent=ext, cmap=N.AURORA, vmin=0, vmax=1, aspect="auto")
    sc = a1.scatter([], [], s=1.5, c=N.CYAN, alpha=0.5)
    a2.imshow(ref.T, origin="lower", extent=ext, cmap=N.AURORA, vmin=0, vmax=1, aspect="auto")
    im4 = a4.imshow(
        np.abs(snaps[0][1] - ref).T, origin="lower", extent=ext, cmap=N.ICE, vmin=0, vmax=0.5, aspect="auto"
    )
    for a, t in ((a1, "PINN prediction u(x,t) + collocation points"), (a2, "exact solution"), (a4, "|error|")):
        a.set_title(t, fontsize=10), a.set_xlabel("x"), a.set_ylabel("t"), a.grid(False)
    a3.set_yscale("log"), a3.set_title("training loss", fontsize=10), a3.set_xlim(0, 4000)
    a3.set_ylim(min(v for _, v in losses) * 0.5, max(v for _, v in losses) * 2)
    g = N.glow(a3, np.zeros(1), np.ones(1), N.MAGENTA, 1.6)
    txt = fig.text(0.37, 0.86, "", color=N.GOLD, fontsize=13)

    def frame(k: int) -> None:
        it, pred, pts, _ = snaps[k]
        im1.set_data(pred.T)
        im4.set_data(np.abs(pred - ref).T)
        sc.set_offsets(pts[:: max(1, len(pts) // 700)])
        L = np.array(losses[: k + 1])
        N.set_glow(g, L[:, 0], L[:, 1])
        rl2 = np.linalg.norm(pred - ref) / np.linalg.norm(ref)
        txt.set_text(f"iteration {it:5d}   relative L2 error vs exact = {rl2:.3f}")

    return str(N.write(fig, frame, len(snaps), CLIPS / "20_pinn_training.mp4", 20))


def fno() -> str:
    import torch

    from aerosynapse.ml.datasets import convdiff_operator_dataset
    from aerosynapse.ml.fno import FNO1d
    from aerosynapse.ml.io import load_model
    from aerosynapse.ml.training import predict_operator

    m = FNO1d(2, 1, 32, 12, 4)
    load_model(m, RES / "phase4_operators_uq" / "fno.pt")
    torch.set_num_threads(2)
    idd = convdiff_operator_dataset(45, n=64, pe_range=(5, 50), seed=901)
    ood = convdiff_operator_dataset(45, n=64, pe_range=(200, 1000), seed=902)
    P1, P2 = predict_operator(m, idd.inputs), predict_operator(m, ood.inputs)
    fig, (a1, a2) = N.figure(1, 2)
    fig.subplots_adjust(top=0.8, bottom=0.12, wspace=0.18)
    N.header(
        fig,
        "Fourier Neural Operator: inside vs outside its training range",
        "operator (u₀, Pe) ↦ u(·, T)  ·  trained on Pe ∈ [5, 50]  ·  white dashed = FV solver truth",
        "model: results/phase4_operators_uq/fno.pt · fresh test fields generated live by the FV solver",
    )
    x = idd.grid
    arts = []
    for a, c, ttl in ((a1, N.CYAN, "in-distribution"), (a2, N.MAGENTA, "out-of-distribution Pe ∈ [200, 1000]")):
        a.set_xlim(0, 1), a.set_ylim(0.0, 2.0), a.set_xlabel("x"), a.set_title(ttl, color=c)
        u0 = a.plot([], [], color="#6b7a99", lw=1, label="input u₀")[0]
        tr = a.plot([], [], "--", color="white", lw=1.2, label="FV truth")[0]
        g = N.glow(a, np.zeros(2), np.zeros(2), c, 2.0, "FNO prediction")
        a.legend(loc="upper right", fontsize=8)
        arts.append((u0, tr, g, a))

    def frame(k: int) -> None:
        i = k // 6
        for (u0, tr, g, a), d, P in zip(arts, (idd, ood), (P1, P2), strict=True):
            u0.set_data(x, d.inputs[i, 0])
            tr.set_data(x, d.outputs[i, 0])
            N.set_glow(g, x, P[i, 0])
            e = np.linalg.norm(P[i, 0] - d.outputs[i, 0]) / np.linalg.norm(d.outputs[i, 0])
            a.set_xlabel(f"x        Pe = {d.params[i, 0]:.0f}    rel. L2 error = {e:.4f}")

    return str(N.write(fig, frame, 45 * 6, CLIPS / "21_fno.mp4", FPS))


def airfoil_morph() -> str:
    from aerosynapse.numerics.panel import solve_panel
    from aerosynapse.physics.airfoil import cst

    m = load("phase6_fluids_aero")["aero_optimisation"]
    hist = m["method"]["history"]
    best, seq = np.inf, []
    for h in hist:
        if h["feasible"] and h["f"][0] < best:
            best = h["f"][0]
        seq.append((h, best))
    fig = N.plt.figure(figsize=(N.W, N.H), dpi=N.DPI)
    N.header(
        fig,
        "Aerodynamic design optimisation (12-step protocol, trusted re-verification)",
        "maximise C_L at α = 4° · t/c ≥ 0.12 · Cp_min ≥ −2 · inviscid panel model (no drag)",
        "history: results/phase6_fluids_aero (GP trust-region surrogate, 60 evaluations); shapes + Cp recomputed live",
    )
    a1 = fig.add_axes([0.05, 0.42, 0.55, 0.42])
    a2 = fig.add_axes([0.05, 0.08, 0.55, 0.26])
    a3 = fig.add_axes([0.67, 0.1, 0.3, 0.72])

    def shape(d: dict) -> tuple[np.ndarray, np.ndarray]:
        return cst(np.array([d["au0"], d["au1"], d["au2"]]), np.array([d["al0"], d["al1"], d["al2"]]), 121)

    def frame(k: int) -> None:
        i = min(k // 4, len(seq) - 1)
        h, b = seq[i]
        X, Y = shape(h["design"])
        r = solve_panel(X, Y, 4.0)
        for a in (a1, a2, a3):
            a.cla()
        a1.set_facecolor(N.PANEL)
        a1.scatter(r.xc, r.yc, c=r.cp, cmap=N.NEON_DIVERGING, s=18, vmin=-2.5, vmax=1)
        N.glow(a1, X, Y, N.CYAN if h["feasible"] else N.ORANGE, 1.4)
        a1.set_xlim(-0.05, 1.05), a1.set_ylim(-0.2, 0.3), a1.set_aspect("equal")
        a1.set_title(
            f"evaluation {i + 1}/60   {'feasible' if h['feasible'] else 'INFEASIBLE'}   C_L = {-h['f'][0]:.3f}",
            color=N.LIME if h["feasible"] else N.ORANGE,
        )
        a2.plot(r.xc, -r.cp, color=N.MAGENTA, lw=1.5)
        a2.set_title("−Cp along the surface", fontsize=10), a2.set_xlabel("x/c")
        F = [-s[0]["f"][0] for s in seq[: i + 1]]
        Bv = [-s[1] if np.isfinite(s[1]) else np.nan for s in seq[: i + 1]]
        a3.scatter(range(1, i + 2), F, s=14, c=[N.CYAN if s[0]["feasible"] else N.ORANGE for s in seq[: i + 1]])
        N.glow(a3, np.arange(1, i + 2), np.array(Bv), N.LIME, 1.8)
        a3.set_xlim(0, 61), a3.set_ylim(0, 1.6), a3.set_xlabel("evaluation"), a3.set_ylabel("C_L")
        a3.set_title("best feasible so far (green)", fontsize=10)

    return str(N.write(fig, frame, len(seq) * 4 + 30, CLIPS / "22_airfoil_opt.mp4", FPS))


ALL = [pinn_training, fno, airfoil_morph]
