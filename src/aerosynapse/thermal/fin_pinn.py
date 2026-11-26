r"""Fin PINN study: data extracted from every optimisation setting, a parametric PINN trained on it, full analysis.

**Physics (dimensionless steady fin, the thermal-rod demo of ``aerosynapse.optim.demos``).**
:math:`-\theta'' + m^2\theta = q(x; x_s)` on [0, 1], heat sink :math:`\theta(0) = 0`, convective tip
:math:`\theta'(1) + Bi\,\theta(1) = 0` (Bi = 1), a Gaussian source of unit power at :math:`x_s` (width 0.05),
:math:`m^2 = hPL^2/(kA)`. Copper (k_rel = 1). Objective :math:`f = \theta_{max}/0.1 + 0.1 (m/4)^2`, constraint
:math:`\theta(1) \le 0.02`.

**Steps.**

1. *Extracted data.* Every optimisation setting (gradient descent, Adam, L-BFGS, SQP, differential evolution,
   genetic algorithm, Bayesian optimisation, constrained BO, trust-region surrogate) is run on the fin with the
   finite-difference model; every design it evaluates is kept with its temperature profile θ(x) (101 points). That is
   the training set. A 7 x 7 grid of designs that no optimiser was steered to is kept apart for testing.
2. *Parametric PINN.* One network θ(x; m, x_s) = x N(x, m, x_s) (the sink condition holds exactly), trained on the
   extracted data and on the fin equation's residual and the tip condition at random collocation points over the
   whole design box. A data-only network of the same size is trained too, to show what the physics adds.
3. *Analysis.* Errors on the held-out grid, the residual, the energy balance (heat into the sink + lateral
   convection + tip = source power), each optimiser's path and optimum, and every optimiser re-run with the PINN
   in the loop and its optimum checked with a fine finite-difference solve. Each optimisation setting has its own
   colour, marker and line style in every 2D and 3D figure.

Run: ``python -m aerosynapse.thermal.fin_pinn --out DIR [--profile quick|full]``. Writes PNG figures, an interactive
3D page (``FinPINN_ALL_interactive_3D.html``), ``fin_data.npz``, ``evaluations.csv``, ``result.json`` and ``summary.txt``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

M_LO, M_HI, XS_LO, XS_HI = 0.5, 4.0, 0.1, 0.9
SHORT = {
    "gd": "GD",
    "adam": "ADAM",
    "lbfgs": "LBFGS",
    "sqp": "SQP",
    "de": "DE",
    "ga": "GA",
    "bo": "BO",
    "cbo": "CBO",
    "surrogate_tr": "TR",
}
DPI = 300  # every PNG at print resolution; PDF (and SVG for the 2D figures) beside them in vector/
LOGO = Path(__file__).resolve().parents[3] / "assets" / "logo.png"
_trapz = getattr(np, "trapezoid", None) or np.trapz
BI, W_SRC, NX = 1.0, 0.05, 101
TIP_MAX, THETA_REF, LAM = 0.02, 0.1, 0.1

# one style per optimisation setting: label, colour, marker, matplotlib dash, plotly symbol, plotly dash
SETTINGS: list[tuple[str, str, str, str, Any, str, str, int]] = [
    ("gd", "Gradient descent", "#e11d48", "o", "-", "circle", "solid", 150),
    ("adam", "Adam", "#f59e0b", "s", "--", "square", "dash", 150),
    ("lbfgs", "L-BFGS (penalty)", "#10b981", "^", "-.", "diamond", "dashdot", 150),
    ("sqp", "SQP", "#3b82f6", "D", ":", "cross", "dot", 150),
    ("de", "Differential evolution", "#8b5cf6", "v", (0, (5, 2, 1, 2)), "x", "longdashdot", 300),
    ("ga", "Genetic algorithm", "#ec4899", "P", (0, (3, 1, 1, 1, 1, 1)), "circle-open", "longdash", 300),
    ("bo", "Bayesian optimisation", "#14b8a6", "X", (0, (8, 3)), "square-open", "dash", 45),
    ("cbo", "Constrained BO", "#a16207", "*", (0, (1, 1)), "diamond-open", "dot", 45),
    ("surrogate_tr", "Trust-region surrogate", "#6366f1", "h", (0, (4, 2, 4, 2, 1, 2)), "circle", "dashdot", 60),
]


# ----------------------------------------------------------------------------- physics
def source(x: np.ndarray, xs: float | np.ndarray) -> np.ndarray:
    return np.exp(-((x - xs) ** 2) / (2 * W_SRC**2)) / (W_SRC * math.sqrt(2 * math.pi))


def fd_field(m: float, xs: float, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    from aerosynapse.optim.demos import thermal_field

    return thermal_field(m, xs, 1.0, BI, n)


def objective(theta: np.ndarray, m: float) -> tuple[float, float]:
    f = float(np.max(theta)) / THETA_REF + LAM * m**2 / 16.0
    g = float(theta[-1]) / TIP_MAX - 1.0
    return f, g


def energy_balance(x: np.ndarray, th: np.ndarray, dth0: float, m: float, xs: float) -> dict[str, float]:
    """Heat into the sink (θ'(0)), lateral convection (m² ∫θ), tip (Bi θ(1)) against the source's power on [0, 1]."""
    qin = float(_trapz(source(x, xs), x))
    sink, lat, tip = float(dth0), float(m**2 * _trapz(th, x)), float(BI * th[-1])
    tot = sink + lat + tip
    return {"source": qin, "sink": sink, "lateral": lat, "tip": tip, "imbalance": (tot - qin) / qin}


# ----------------------------------------------------------------------------- 1 extracted data
def extract(seed: int = 0, log=print) -> dict[str, Any]:
    from aerosynapse.optim import algorithms  # noqa: F401  (registers the optimisers)
    from aerosynapse.optim import surrogate  # noqa: F401
    from aerosynapse.optim.algorithms import minimize
    from aerosynapse.optim.demos import thermal_model, thermal_space
    from aerosynapse.optim.problem import OptimizationProblem

    xg = np.linspace(0, 1, NX)
    runs: dict[str, Any] = {}
    for key, label, *_rest, budget in SETTINGS:
        p = OptimizationProblem(f"fin_{key}", thermal_space(False), thermal_model, budget=budget)
        t0 = time.perf_counter()
        try:
            r = minimize(p, key, z0=np.array([0.15, 0.85]), seed=seed)
            msg = r.message
        except Exception as e:  # noqa: BLE001  (a budget stop or a solver failure: keep what was evaluated)
            msg = f"stopped: {type(e).__name__}: {e}"[:160]
        H = p.history
        designs = np.array([[e.design["m"], e.design["x_s"]] for e in H]) if H else np.zeros((0, 2))
        fields = (
            np.array([np.interp(xg, *fd_field(d[0], d[1])) for d in designs]) if len(designs) else np.zeros((0, NX))
        )
        f = np.array([e.f[0] for e in H])
        g = np.array([e.g[0] for e in H])
        feas = np.array([e.feasible for e in H])
        best = int(np.argmin(np.where(feas, f, np.inf))) if feas.any() else int(np.argmin(f + 1e3 * np.maximum(g, 0)))
        bsf, b = [], np.inf
        for fi, ok in zip(f, feas):
            if ok:
                b = min(b, fi)
            bsf.append(b)
        runs[key] = {
            "label": label,
            "designs": designs,
            "fields": fields,
            "f": f,
            "g": g,
            "feasible": feas,
            "best": best,
            "best_so_far": np.array(bsf),
            "message": msg,
            "wall_s": time.perf_counter() - t0,
        }
        log(
            f"  {label:24s} {len(H):4d} evaluations, best f {f[best]:.4f} at m {designs[best, 0]:.3f}, x_s {designs[best, 1]:.3f} ({msg})"
        )
    return {"x": xg, "runs": runs}


def test_grid(n: int = 7) -> tuple[np.ndarray, np.ndarray]:
    ms, xss = np.linspace(M_LO + 0.2, M_HI - 0.2, n), np.linspace(XS_LO + 0.05, XS_HI - 0.05, n)
    D = np.array([[a, b] for a in ms for b in xss])
    xg = np.linspace(0, 1, NX)
    return D, np.array([np.interp(xg, *fd_field(a, b)) for a, b in D])


# ----------------------------------------------------------------------------- 2 PINN
def _net(width: int = 64, depth: int = 4):
    import torch

    layers: list[Any] = []
    d = 3
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
        d = width
    layers.append(torch.nn.Linear(d, 1))
    torch.manual_seed(0)
    return torch.nn.Sequential(*layers)


class FinPINN:
    """θ(x; m, x_s) = 0.1 x N(x, a(m), b(x_s)); the sink θ(0) = 0 holds by construction."""

    def __init__(self, width: int = 64, depth: int = 4) -> None:
        self.net = _net(width, depth)

    def __call__(self, x, m, xs):
        import torch

        a = (m - M_LO) / (M_HI - M_LO) * 2 - 1
        b = (xs - XS_LO) / (XS_HI - XS_LO) * 2 - 1
        return 0.1 * x * self.net(torch.cat([2 * x - 1, a, b], dim=1))

    def predict(self, x: np.ndarray, m: float, xs: float) -> np.ndarray:
        import torch

        with torch.no_grad():
            X = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
            M = torch.full_like(X, float(m))
            S = torch.full_like(X, float(xs))
            return self(X, M, S).numpy().ravel().astype(float)

    def derivatives(self, x: np.ndarray, m: float, xs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        X = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
        M = torch.full_like(X, float(m))
        S = torch.full_like(X, float(xs))
        th = self(X, M, S)
        d1 = torch.autograd.grad(th.sum(), X, create_graph=True)[0]
        d2 = torch.autograd.grad(d1.sum(), X)[0]
        return th.detach().numpy().ravel(), d1.detach().numpy().ravel(), d2.numpy().ravel()


def train(
    data_D: np.ndarray, data_T: np.ndarray, physics: bool, steps: int, lbfgs_iters: int, seed: int = 0, log=print
) -> tuple[FinPINN, dict[str, list[float]]]:
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FinPINN()
    xg = np.linspace(0, 1, NX)
    # the data: every extracted profile at every grid point
    Xd = np.repeat(xg[None, :], len(data_D), 0).ravel()
    Md = np.repeat(data_D[:, :1], NX, 1).ravel()
    Sd = np.repeat(data_D[:, 1:], NX, 1).ravel()
    Td = data_T.ravel()
    td = [torch.tensor(v, dtype=torch.float32).reshape(-1, 1) for v in (Xd, Md, Sd, Td)]
    nd, nb, nc = len(Td), 2048, 2048
    opt = torch.optim.Adam(model.net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    hist: dict[str, list[float]] = {"step": [], "data": [], "physics": [], "tip": [], "total": []}

    def colloc(n: int):
        m = rng.uniform(M_LO, M_HI, n)
        xs = rng.uniform(XS_LO, XS_HI, n)
        x = np.where(rng.random(n) < 0.5, rng.random(n), np.clip(xs + 0.08 * rng.standard_normal(n), 0, 1))
        return [torch.tensor(v, dtype=torch.float32).reshape(-1, 1) for v in (x, m, xs)]

    def losses(batch_d, batch_c, batch_t):
        X, M, S, T = batch_d
        ld = (((model(X, M, S) - T) / 0.1) ** 2).mean()
        if not physics:
            z = torch.zeros(())
            return ld, z, z
        x, m, xs = batch_c
        x = x.clone().requires_grad_(True)
        th = model(x, m, xs)
        d1 = torch.autograd.grad(th.sum(), x, create_graph=True)[0]
        d2 = torch.autograd.grad(d1.sum(), x, create_graph=True)[0]
        q = torch.exp(-((x - xs) ** 2) / (2 * W_SRC**2)) / (W_SRC * math.sqrt(2 * math.pi))
        lp = (((-d2 + m**2 * th - q) / 8.0) ** 2).mean()
        xt, mt, st = batch_t
        xt = xt.clone().requires_grad_(True)
        tt = model(xt, mt, st)
        dt = torch.autograd.grad(tt.sum(), xt, create_graph=True)[0]
        lt = (((dt + BI * tt) / 0.1) ** 2).mean()
        return ld, lp, lt

    def tip_batch(n: int):
        _, m, xs = colloc(n)
        return [torch.ones_like(m), m, xs]

    t0 = time.perf_counter()
    for k in range(steps):
        idx = torch.from_numpy(rng.integers(0, nd, nb))
        bd = [t[idx] for t in td]
        ld, lp, lt = losses(bd, colloc(nc), tip_batch(256))
        loss = ld + lp + lt
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if k % 50 == 0 or k == steps - 1:
            for kk, v in zip(
                ("step", "data", "physics", "tip", "total"),
                (k, ld.detach().item(), lp.detach().item(), lt.detach().item(), loss.detach().item()),
            ):
                hist[kk].append(float(v))
        if k % 500 == 0:
            log(
                f"    {'PINN' if physics else 'data-only'} step {k:5d}: data {ld.detach().item():.2e} physics {lp.detach().item():.2e} tip {lt.detach().item():.2e}  ({time.perf_counter() - t0:.0f} s)"
            )
    if lbfgs_iters > 0:  # polish on a fixed batch
        idx = torch.from_numpy(rng.integers(0, nd, 6000))
        bd, bc, bt = [t[idx] for t in td], colloc(6000), tip_batch(800)
        lb = torch.optim.LBFGS(
            model.net.parameters(), lr=0.5, max_iter=lbfgs_iters, history_size=50, line_search_fn="strong_wolfe"
        )

        def closure():
            lb.zero_grad()
            a, b, c = losses(bd, bc, bt)
            tot = a + b + c
            tot.backward()
            return tot

        lb.step(closure)
        ld, lp, lt = losses(bd, bc, bt)
        for kk, v in zip(
            ("step", "data", "physics", "tip", "total"),
            (
                steps + lbfgs_iters,
                ld.detach().item(),
                lp.detach().item(),
                lt.detach().item(),
                (ld + lp + lt).detach().item(),
            ),
        ):
            hist[kk].append(float(v))
        log(
            f"    after L-BFGS: data {ld.detach().item():.2e} physics {lp.detach().item():.2e} tip {lt.detach().item():.2e}"
        )
    return model, hist


# ----------------------------------------------------------------------------- 3 analysis
def pinn_in_loop(model: FinPINN, log=print) -> dict[str, Any]:
    """Every optimiser again, with the PINN as the model; its optimum then checked with a fine FD solve."""
    from aerosynapse.optim.algorithms import minimize
    from aerosynapse.optim.demos import thermal_space
    from aerosynapse.optim.problem import EvalResult, OptimizationProblem

    xg = np.linspace(0, 1, 201)

    def pinn_model(d: dict[str, Any]) -> EvalResult:
        th = model.predict(xg, d["m"], d["x_s"])
        f, g = objective(th, d["m"])
        return EvalResult(f=[f], g=[g], info={})

    out: dict[str, Any] = {}
    for key, label, *_rest, budget in SETTINGS:
        p = OptimizationProblem(f"fin_pinn_{key}", thermal_space(False), pinn_model, budget=budget)
        try:
            minimize(p, key, z0=np.array([0.15, 0.85]), seed=0)
        except Exception:  # noqa: BLE001
            pass
        H = p.history
        if not H:
            continue
        feas = [e for e in H if e.feasible] or H
        e = min(feas, key=lambda e: e.f[0])
        m, xs = e.design["m"], e.design["x_s"]
        xf, tf = fd_field(m, xs, 1600)
        f_fd, g_fd = objective(tf, m)
        out[key] = {
            "m": m,
            "x_s": xs,
            "f_pinn": e.f[0],
            "f_fd": f_fd,
            "g_fd": g_fd,
            "feasible_fd": g_fd <= 1e-9,
            "n_evals": len(H),
        }
        log(
            f"  {label:24s} PINN optimum m {m:.3f} x_s {xs:.3f}: f(PINN) {e.f[0]:.4f}, f(FD) {f_fd:.4f}, tip {'ok' if g_fd <= 1e-9 else 'VIOLATED'}"
        )
    return out


def stamp_and_save(fig, out: Path, name: str, what: str, dim: str) -> str:
    """The AeroSynapse logo and a footer on the figure, then PNG at DPI, PDF (and SVG for 2D) in out/vector/."""
    import datetime

    import matplotlib.pyplot as plt

    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    try:
        if LOGO.exists():
            w, h = fig.get_size_inches()
            s = 0.42  # inches
            ax = fig.add_axes((1 - (s + 0.08) / w, 1 - (s + 0.06) / h, s / w, s / h))
            ax.imshow(plt.imread(str(LOGO)))
            ax.axis("off")
    except Exception:  # noqa: BLE001  (a figure without the logo is still a figure)
        pass
    fig.text(
        0.008,
        0.008,
        f"AeroSynapse  |  Fin PINN  |  {what}  |  {dim}  |  {datetime.date.today():%Y-%m-%d}",
        fontsize=7,
        color="#64748b",
    )
    fig.savefig(out / f"{name}.png", dpi=DPI)
    vec = out / "vector"
    vec.mkdir(exist_ok=True)
    fig.savefig(vec / f"{name}.pdf")
    if dim == "2D":
        fig.savefig(vec / f"{name}.svg")
    plt.close(fig)
    return f"{name}.png"


def setting_figures(
    out: Path, ext: dict[str, Any], pinn: FinPINN, bal: dict[str, Any], loop: dict[str, Any]
) -> list[str]:
    """For every optimisation setting its own pair of figures: a 2D analysis (its path over the objective map,
    convergence, theta(x) by FD and PINN at its optimum with the error, the energy balance) and a 3D one (its path over
    the PINN objective surface, its optimum fin coloured by temperature)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    xg, runs = ext["x"], ext["runs"]
    mg, sg = np.linspace(M_LO, M_HI, 40), np.linspace(XS_LO, XS_HI, 36)
    F = np.zeros((len(sg), len(mg)))
    G = np.zeros_like(F)
    FP = np.zeros_like(F)
    for i, s_ in enumerate(sg):
        for j, m in enumerate(mg):
            _x, t = fd_field(m, s_, 300)
            F[i, j], G[i, j] = objective(t, m)
            FP[i, j] = objective(pinn.predict(xg, m, s_), m)[0]
    MM, SS = np.meshgrid(mg, sg)
    names: list[str] = []
    for key, label, col, mk, ls, *_ in SETTINGS:
        r = runs.get(key)
        if not r or not len(r["designs"]):
            continue
        K = SHORT[key]
        d, b = r["designs"], r["best"]
        m_b, xs_b = d[b]
        # 2D
        fig, axs = plt.subplots(2, 2, figsize=(14, 10.5))
        a = axs[0, 0]
        cs = a.contourf(mg, sg, F, 30, cmap="Greys", alpha=0.6)
        fig.colorbar(cs, ax=a, label="objective f (FD)")
        a.contour(mg, sg, G, [0], colors="crimson", linewidths=2)
        a.plot(d[:, 0], d[:, 1], color=col, linestyle=ls, lw=1.2, alpha=0.8)
        a.scatter(d[:, 0], d[:, 1], c=np.arange(len(d)), cmap="cool", marker=mk, s=26, edgecolors=col, linewidths=0.6)
        a.scatter(
            [d[0, 0]], [d[0, 1]], marker="o", s=140, facecolors="none", edgecolors="k", linewidths=2, label="start"
        )
        a.scatter(
            [m_b],
            [xs_b],
            marker=mk,
            s=260,
            c=col,
            edgecolors="k",
            linewidths=1.5,
            label=f"optimum m {m_b:.3f}, x_s {xs_b:.3f}",
        )
        a.set(
            xlabel="fin parameter m",
            ylabel="source position x_s",
            title=f"Its {len(d)} evaluations (colour = order); red line = tip limit",
        )
        a.legend(fontsize=8, loc="upper right")
        a = axs[0, 1]
        bsf = r["best_so_far"]
        a.plot(
            np.arange(1, len(bsf) + 1), bsf, color=col, linestyle=ls, marker=mk, markevery=max(1, len(bsf) // 12), lw=2
        )
        a.plot(
            np.arange(1, len(r["f"]) + 1),
            np.where(r["feasible"], r["f"], np.nan),
            ".",
            color=col,
            alpha=0.35,
            ms=4,
            label="each feasible evaluation",
        )
        a.set_xscale("log")
        a.set(xlabel="evaluations", ylabel="objective f", title=f"Convergence: best feasible f = {r['f'][b]:.4f}")
        a.legend(fontsize=8)
        a = axs[1, 0]
        tf = r["fields"][b]
        tp = pinn.predict(xg, m_b, xs_b)
        a.plot(xg, tf, color=col, linestyle=ls, lw=2.2, label="FD")
        a.plot(xg[::5], tp[::5], marker=mk, ls="none", color=col, mfc="white", ms=7, label="PINN")
        a.axhline(TIP_MAX, color="crimson", lw=1, ls="--", label="tip limit 0.02")
        a2 = a.twinx()
        a2.plot(xg, tp - tf, color="#475569", lw=1, ls=":")
        a2.set_ylabel("PINN - FD", color="#475569")
        a.set(
            xlabel="x (sink 0, tip 1)",
            ylabel="theta",
            title=f"theta(x) at the optimum; max |PINN - FD| = {np.max(np.abs(tp - tf)):.2e}",
        )
        a.legend(fontsize=8, loc="upper right")
        a = axs[1, 1]
        if key in bal:
            parts = ("sink", "lateral", "tip")
            xb = np.arange(3)
            a.bar(xb - 0.2, [bal[key]["fd"][q] for q in parts], 0.38, color=col, edgecolor="k", label="FD")
            a.bar(
                xb + 0.2,
                [bal[key]["pinn"][q] for q in parts],
                0.38,
                color=col,
                alpha=0.45,
                hatch="//",
                edgecolor="k",
                label="PINN",
            )
            a.set_xticks(xb, ["into the sink", "lateral convection", "tip"])
            L = loop.get(key, {})
            extra = (
                f"\nPINN in the loop: f(FD) {L['f_fd']:.4f}, tip {'ok' if L.get('feasible_fd') else 'VIOLATED'}"
                if L
                else ""
            )
            a.set(
                ylabel="heat / source power",
                title=f"Energy balance: imbalance FD {100 * bal[key]['fd']['imbalance']:+.2f} %, PINN {100 * bal[key]['pinn']['imbalance']:+.2f} %{extra}",
            )
            a.legend(fontsize=8)
        fig.suptitle(f"Fin PINN  |  {label} ({K})  |  2D analysis", fontsize=14, fontweight="bold", color=col)
        names.append(stamp_and_save(fig, out, f"FinPINN_{K}_analysis_2D", label, "2D"))
        # 3D
        fig = plt.figure(figsize=(14, 7))
        a = fig.add_subplot(1, 2, 1, projection="3d")
        cap = np.percentile(FP, 95)
        a.plot_surface(MM, SS, np.minimum(FP, cap), cmap=cm.viridis, alpha=0.35, linewidth=0)
        fz = np.minimum(r["f"], cap)
        a.plot(d[:, 0], d[:, 1], fz, color=col, linestyle=ls, lw=1.6)
        a.scatter(d[:, 0], d[:, 1], fz, c=col, marker=mk, s=18)
        a.scatter([m_b], [xs_b], [min(r["f"][b], cap)], c=col, marker=mk, s=220, edgecolors="k")
        a.set(xlabel="m", ylabel="x_s", zlabel="f", title=f"{label}: path over the PINN objective")
        a.view_init(32, -60)
        a = fig.add_subplot(1, 2, 2, projection="3d")
        norm = matplotlib.colors.Normalize(0, max(1e-9, float(np.max(tp))))
        X2, Y2 = np.meshgrid(xg, [0, 0.4])
        C = cm.inferno(norm(np.vstack([tp, tp])))
        a.plot_surface(X2, Y2, np.full_like(X2, 0.08), facecolors=C, shade=False, linewidth=0)
        a.plot_surface(
            X2,
            np.zeros_like(X2),
            np.vstack([np.zeros_like(xg), np.full_like(xg, 0.08)]),
            facecolors=C,
            shade=False,
            linewidth=0,
        )
        a.bar3d(-0.08, -0.1, -0.1, 0.08, 0.6, 0.28, color="#94a3b8", alpha=0.9)
        a.plot(
            xg, np.full_like(xg, 0.2), 0.08 + tp / max(1e-9, float(np.max(tp))) * 0.25, color=col, linestyle=ls, lw=2.2
        )
        a.scatter([xs_b], [0.2], [0.14], c=col, marker=mk, s=120, edgecolors="k")
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap="inferno"), ax=a, shrink=0.6, label="theta (PINN)")
        a.set(xlabel="x along the fin", title=f"its optimum fin (m {m_b:.2f}), coloured by theta; line = theta(x)")
        a.set_yticks([])
        a.set_zticks([])
        a.set_box_aspect((2.4, 0.9, 0.6))
        a.view_init(30, -65)
        fig.suptitle(f"Fin PINN  |  {label} ({K})  |  3D analysis", fontsize=14, fontweight="bold", color=col)
        names.append(stamp_and_save(fig, out, f"FinPINN_{K}_analysis_3D", label, "3D"))
    return names


def write_manifest(out: Path) -> None:
    """settings.json: the viewer's buttons (one per optimisation setting) and how the figures are named."""
    man = {
        "study": "Fin PINN",
        "naming": "FinPINN_<SETTING>_<figure>_<2D|3D>.png (ALL = every setting together); vector/ holds PDF (and SVG for 2D)",
        "settings": [{"key": SHORT[k], "label": lab, "colour": col, "marker": mk} for k, lab, col, mk, *_ in SETTINGS],
    }
    (out / "settings.json").write_text(json.dumps(man, indent=1), encoding="utf-8")


def figures(
    out: Path,
    ext: dict[str, Any],
    test: tuple[np.ndarray, np.ndarray],
    pinn: FinPINN,
    hist,
    hist_d,
    err: dict[str, Any],
    loop: dict[str, Any],
    bal: dict[str, Any],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    plt.rcParams.update(
        {"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "font.size": 9.5, "axes.titleweight": "bold"}
    )
    st = {s[0]: s for s in SETTINGS}
    runs = ext["runs"]
    xg = ext["x"]
    names: list[str] = []

    def save(fig, name: str) -> None:
        stem = name[:-4]
        nn, rest = stem.split("_", 1)
        dim = "3D" if rest.endswith("3d") else "2D"
        rest = rest[:-3] if rest.endswith("_3d") else rest
        names.append(
            stamp_and_save(
                fig, out, f"FinPINN_ALL_{nn}_{rest.replace('_', '-')}_{dim}", "all optimisation settings", dim
            )
        )

    def sty(k: str) -> dict[str, Any]:
        s = st[k]
        return {"color": s[2], "marker": s[3], "linestyle": s[4], "label": s[1]}

    # landscape of the FD objective (and the tip constraint) on the design box
    mg, sg = np.linspace(M_LO, M_HI, 36), np.linspace(XS_LO, XS_HI, 33)
    F = np.zeros((len(sg), len(mg)))
    G = np.zeros_like(F)
    FP = np.zeros_like(F)
    for i, s in enumerate(sg):
        for j, m in enumerate(mg):
            x, t = fd_field(m, s, 300)
            F[i, j], G[i, j] = objective(t, m)
            FP[i, j] = objective(pinn.predict(xg, m, s), m)[0]

    # 1: the extracted data in the design plane
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    cs = ax.contourf(mg, sg, F, 30, cmap="Greys", alpha=0.55)
    fig.colorbar(cs, ax=ax, label="objective f (FD)")
    ax.contour(mg, sg, G, [0], colors="crimson", linewidths=2)
    ax.text(
        M_LO + 0.1,
        XS_HI - 0.05,
        "red line: tip limit θ(1) = 0.02 (feasible to its right/below)",
        color="crimson",
        fontsize=8.5,
    )
    for k, r in runs.items():
        d = r["designs"]
        if len(d):
            s = sty(k)
            ax.plot(d[:, 0], d[:, 1], color=s["color"], linestyle=s["linestyle"], lw=0.8, alpha=0.6)
            ax.scatter(
                d[:, 0], d[:, 1], c=s["color"], marker=s["marker"], s=16, alpha=0.75, label=f"{s['label']} ({len(d)})"
            )
            b = r["best"]
            ax.scatter(
                d[b, 0], d[b, 1], c=s["color"], marker=s["marker"], s=170, edgecolors="k", linewidths=1.3, zorder=5
            )
    ax.scatter(test[0][:, 0], test[0][:, 1], marker="+", c="k", s=40, label="held-out test designs (49)")
    ax.set(
        xlabel="fin parameter m = sqrt(hP/kA) L",
        ylabel="source position x_s",
        title="1  Extracted data: every design each optimisation setting evaluated (large = its optimum)",
    )
    ax.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.85)
    save(fig, "01_extracted_data.png")

    # 2: convergence
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for k, r in runs.items():
        b = r["best_so_far"]
        if len(b) and np.isfinite(b).any():
            s = sty(k)
            n = np.arange(1, len(b) + 1)
            ax.plot(
                n,
                b,
                color=s["color"],
                linestyle=s["linestyle"],
                lw=1.8,
                marker=s["marker"],
                markevery=max(1, len(b) // 10),
                ms=6,
                label=s["label"],
            )
    ax.set_xscale("log")
    ax.set(
        xlabel="evaluations of the FD model",
        ylabel="best feasible objective so far",
        title="2  Convergence of each optimisation setting on the fin",
    )
    ax.legend(fontsize=8)
    save(fig, "02_convergence.png")

    # 3: temperature profiles at each optimum, FD and PINN
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for k, r in runs.items():
        if not len(r["designs"]):
            continue
        s = sty(k)
        m, x_s = r["designs"][r["best"]]
        tf = r["fields"][r["best"]]
        tp = pinn.predict(xg, m, x_s)
        axs[0].plot(
            xg, tf, color=s["color"], linestyle=s["linestyle"], lw=1.6, label=f"{s['label']}: m {m:.2f}, x_s {x_s:.2f}"
        )
        axs[0].plot(xg[::8], tp[::8], color=s["color"], marker=s["marker"], ls="none", ms=6, mfc="none")
        axs[1].plot(
            xg,
            tp - tf,
            color=s["color"],
            linestyle=s["linestyle"],
            marker=s["marker"],
            markevery=10,
            ms=5,
            lw=1.4,
            label=s["label"],
        )
    axs[0].axhline(TIP_MAX, color="crimson", lw=0.8, ls="--")
    axs[0].set(
        xlabel="x (sink at 0, tip at 1)", ylabel="θ", title="3a  θ(x) at each optimum: lines FD, open markers PINN"
    )
    axs[0].legend(fontsize=7)
    axs[1].set(xlabel="x", ylabel="θ PINN - θ FD", title="3b  PINN error along the fin at each optimum")
    axs[1].legend(fontsize=7)
    save(fig, "03_profiles_at_optima.png")

    # 4: training
    fig, ax = plt.subplots(figsize=(9, 5))
    for kk, c, ls in (("data", "#2563eb", "-"), ("physics", "#dc2626", "--"), ("tip", "#16a34a", "-.")):
        ax.semilogy(hist["step"], np.maximum(hist[kk], 1e-12), color=c, ls=ls, lw=1.6, label=f"PINN: {kk} loss")
    ax.semilogy(
        hist_d["step"],
        np.maximum(hist_d["data"], 1e-12),
        color="#64748b",
        ls=":",
        lw=1.8,
        label="data-only network: data loss",
    )
    ax.set(
        xlabel="step (the last point after L-BFGS)",
        ylabel="loss",
        title="4  Training: data, fin-equation residual and tip condition",
    )
    ax.legend(fontsize=8)
    save(fig, "04_training.png")

    # 5: error maps on the held-out designs
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2))
    D = test[0]
    for a, key, ttl in (
        (axs[0], "pinn_case_rel", "5a  PINN (data + physics)"),
        (axs[1], "data_case_rel", "5b  data-only network"),
    ):
        sc = a.scatter(
            D[:, 0],
            D[:, 1],
            c=100 * np.array(err[key]),
            cmap="magma_r",
            s=150,
            marker="s",
            vmin=0,
            vmax=max(1e-6, 100 * max(max(err["pinn_case_rel"]), max(err["data_case_rel"]))),
        )
        fig.colorbar(sc, ax=a, label="relative L2 error on the held-out design, %")
        for k, r in runs.items():
            if len(r["designs"]):
                s = sty(k)
                a.scatter(r["designs"][:, 0], r["designs"][:, 1], c=s["color"], marker=s["marker"], s=7, alpha=0.35)
        a.set(xlabel="m", ylabel="x_s", title=ttl + f": mean {100 * np.mean(err[key]):.2f} %")
    save(fig, "05_error_maps.png")

    # 6: residual of the fin equation along the fin at each optimum
    fig, ax = plt.subplots(figsize=(9, 5))
    xr = np.linspace(0, 1, 401)
    for k, r in runs.items():
        if not len(r["designs"]):
            continue
        s = sty(k)
        m, x_s = r["designs"][r["best"]]
        th, d1, d2 = pinn.derivatives(xr, m, x_s)
        res = -d2 + m**2 * th - source(xr, x_s)
        ax.plot(
            xr,
            res,
            color=s["color"],
            linestyle=s["linestyle"],
            marker=s["marker"],
            markevery=40,
            ms=5,
            lw=1.3,
            label=s["label"],
        )
    ax.set(
        xlabel="x",
        ylabel="-θ'' + m²θ - q",
        title="6  PINN residual of the fin equation at each optimum (0 = exact physics)",
    )
    ax.legend(fontsize=8)
    save(fig, "06_residual.png")

    # 7: energy balance at each optimum
    keys = [k for k in runs if k in bal]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    w = 0.38
    for i, k in enumerate(keys):
        s = sty(k)
        for j, (src, hatch) in enumerate((("fd", ""), ("pinn", "//"))):
            b = bal[k][src]
            x0 = i + (j - 0.5) * w
            bot = 0.0
            for part, alpha in (("sink", 1.0), ("lateral", 0.6), ("tip", 0.3)):
                ax.bar(
                    x0, b[part], w * 0.92, bottom=bot, color=s["color"], alpha=alpha, hatch=hatch, edgecolor="k", lw=0.5
                )
                bot += b[part]
            ax.text(x0, bot + 0.01, f"{100 * b['imbalance']:+.1f}%", ha="center", fontsize=7, rotation=90)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(keys)), [st[k][1] for k in keys], rotation=20, ha="right", fontsize=8)
    ax.set(
        ylabel="heat / source power",
        title="7  Energy balance at each optimum: sink (dark), lateral (mid), tip (light); left FD, right (hatched) PINN; label = imbalance",
    )
    save(fig, "07_energy_balance.png")

    # 8: PINN in the loop
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, k in enumerate([k for k in runs if k in loop]):
        s = sty(k)
        L = loop[k]
        r = runs[k]
        ax.scatter(
            i - 0.15,
            r["f"][r["best"]],
            marker=s["marker"],
            c=s["color"],
            s=110,
            edgecolors="k",
            label=f"{s['label']}" if i == 0 else None,
        )
        ax.scatter(i + 0.15, L["f_fd"], marker=s["marker"], c="white", edgecolors=s["color"], linewidths=2, s=110)
        ax.plot(
            [i - 0.15, i + 0.15], [r["f"][r["best"]], L["f_fd"]], color=s["color"], linestyle=s["linestyle"], lw=1.5
        )
        if not L["feasible_fd"]:
            ax.text(i + 0.18, L["f_fd"], " tip limit\n violated", fontsize=7, color="crimson")
    ks = [k for k in runs if k in loop]
    ax.set_xticks(range(len(ks)), [st[k][1] for k in ks], rotation=20, ha="right", fontsize=8)
    ax.set(
        ylabel="objective f checked with FD",
        title="8  Optimum with the FD model (filled) and with the PINN in the loop, re-checked by FD (open)",
    )
    save(fig, "08_pinn_in_the_loop.png")

    # 9 (3D): θ(x, m) surface at the best x_s with each optimum's profile as a 3D line
    kbest = min(runs, key=lambda k: runs[k]["f"][runs[k]["best"]] if len(runs[k]["f"]) else np.inf)
    xs_b = float(runs[kbest]["designs"][runs[kbest]["best"], 1])
    Xs, Ms = np.meshgrid(xg, mg)
    Th = np.array([pinn.predict(xg, m, xs_b) for m in mg])
    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(Xs, Ms, Th, cmap=cm.inferno, alpha=0.65, linewidth=0, antialiased=True)
    for k, r in runs.items():
        if not len(r["designs"]):
            continue
        s = sty(k)
        m, x_s = r["designs"][r["best"]]
        ax.plot(
            xg,
            np.full_like(xg, m),
            pinn.predict(xg, m, x_s),
            color=s["color"],
            linestyle=s["linestyle"],
            lw=2.2,
            label=s["label"],
        )
        ax.scatter([xg[-1]], [m], [r["fields"][r["best"]][-1]], c=s["color"], marker=s["marker"], s=60, edgecolors="k")
    ax.set(
        xlabel="x",
        ylabel="m",
        zlabel="θ",
        title=f"9  PINN θ(x, m) at x_s = {xs_b:.2f}; lines: each setting's optimum profile",
    )
    ax.view_init(26, -128)
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "09_theta_surface_3d.png")

    # 10 (3D): design paths over the objective
    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_subplot(projection="3d")
    MM, SS = np.meshgrid(mg, sg)
    ax.plot_surface(MM, SS, np.minimum(FP, np.percentile(FP, 95)), cmap=cm.viridis, alpha=0.35, linewidth=0)
    for k, r in runs.items():
        d = r["designs"]
        if not len(d):
            continue
        s = sty(k)
        fz = np.minimum(r["f"], np.percentile(FP, 95))
        ax.plot(d[:, 0], d[:, 1], fz, color=s["color"], linestyle=s["linestyle"], lw=1.2, alpha=0.8)
        ax.scatter(d[:, 0], d[:, 1], fz, c=s["color"], marker=s["marker"], s=14, label=s["label"])
    ax.set(
        xlabel="m", ylabel="x_s", zlabel="f", title="10  Every optimiser's path over the PINN objective surface (3D)"
    )
    ax.view_init(32, -60)
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "10_design_paths_3d.png")

    # 11 (3D): the optimum fins themselves, coloured by temperature
    fig = plt.figure(figsize=(12, 7.5))
    ax = fig.add_subplot(projection="3d")
    vmax = max(float(np.max(r["fields"][r["best"]])) for r in runs.values() if len(r["designs"]))
    norm = matplotlib.colors.Normalize(0, vmax)
    for i, (k, r) in enumerate(runs.items()):
        if not len(r["designs"]):
            continue
        s = sty(k)
        m, x_s = r["designs"][r["best"]]
        th = pinn.predict(xg, m, x_s)
        y0, wdt, hgt = 1.4 * i, 0.5, 0.12
        X2, Y2 = np.meshgrid(xg, [y0, y0 + wdt])
        C = cm.inferno(norm(np.vstack([th, th])))
        ax.plot_surface(X2, Y2, np.full_like(X2, hgt), facecolors=C, shade=False, linewidth=0)
        ax.plot_surface(
            X2,
            np.full_like(X2, y0),
            np.vstack([np.zeros_like(xg), np.full_like(xg, hgt)]),
            facecolors=C,
            shade=False,
            linewidth=0,
        )
        ax.bar3d(-0.08, y0 - 0.1, -0.1, 0.08, wdt + 0.2, hgt + 0.2, color="#94a3b8", alpha=0.9)
        ax.scatter([x_s], [y0 + wdt / 2], [hgt + 0.05], c=s["color"], marker=s["marker"], s=90, edgecolors="k")
        ax.text(1.05, y0 + wdt / 2, hgt, f"{s['label']}  m={m:.2f}", fontsize=7.5, color=s["color"])
    m_ = cm.ScalarMappable(norm=norm, cmap="inferno")
    fig.colorbar(m_, ax=ax, shrink=0.6, label="θ (PINN)")
    ax.set(
        xlabel="x along the fin",
        ylabel="",
        zlabel="",
        title="11  The optimum fin of each setting, coloured by θ (grey block = sink, marker = source)",
    )
    ax.set_yticks([])
    ax.view_init(38, -70)
    ax.set_box_aspect((2.2, 3.0, 0.5))
    save(fig, "11_fins_3d.png")

    # 12 (3D): PINN against FD objective surfaces
    fig = plt.figure(figsize=(12.5, 6))
    for j, (Z, ttl) in enumerate(((F, "FD objective"), (FP - F, "PINN - FD objective"))):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        ax.plot_surface(MM, SS, Z, cmap=cm.coolwarm if j else cm.viridis, linewidth=0, alpha=0.9)
        for k, r in runs.items():
            if len(r["designs"]):
                s = sty(k)
                m, x_s = r["designs"][r["best"]]
                ax.scatter(
                    [m],
                    [x_s],
                    [Z.max() * (1.05 if j == 0 else 1.0)],
                    c=s["color"],
                    marker=s["marker"],
                    s=50,
                    edgecolors="k",
                )
        ax.set(xlabel="m", ylabel="x_s", title=f"12{'ab'[j]}  {ttl}")
        ax.view_init(30, -55)
    save(fig, "12_objective_surfaces_3d.png")
    return names


def html3d(out: Path, ext: dict[str, Any], pinn: FinPINN) -> str | None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:  # noqa: BLE001
        return None
    runs, xg = ext["runs"], ext["x"]
    mg, sg = np.linspace(M_LO, M_HI, 30), np.linspace(XS_LO, XS_HI, 28)
    FP = np.array([[objective(pinn.predict(xg, m, s), m)[0] for m in mg] for s in sg])
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Optimiser paths over the PINN objective", "θ(x) of each optimum fin (PINN)"),
    )
    fig.add_trace(
        go.Surface(
            x=mg,
            y=sg,
            z=np.minimum(FP, np.percentile(FP, 95)),
            colorscale="Viridis",
            opacity=0.55,
            showscale=False,
            name="objective",
        ),
        1,
        1,
    )
    for k, label, col, _mk, _ls, sym, dash, _b in SETTINGS:
        r = runs.get(k)
        if not r or not len(r["designs"]):
            continue
        d = r["designs"]
        fig.add_trace(
            go.Scatter3d(
                x=d[:, 0],
                y=d[:, 1],
                z=np.minimum(r["f"], np.percentile(FP, 95)),
                mode="lines+markers",
                name=label,
                legendgroup=k,
                line={"color": col, "dash": dash, "width": 4},
                marker={"symbol": sym, "size": 3.5, "color": col},
            ),
            1,
            1,
        )
        m, x_s = d[r["best"]]
        th = pinn.predict(xg, m, x_s)
        fig.add_trace(
            go.Scatter3d(
                x=xg,
                y=np.full_like(xg, m),
                z=th,
                mode="lines+markers",
                name=f"{label} optimum",
                legendgroup=k,
                showlegend=False,
                line={"color": col, "dash": dash, "width": 6},
                marker={"symbol": sym, "size": 3, "color": col},
            ),
            1,
            2,
        )
    fig.update_layout(
        template="plotly_dark",
        height=760,
        title="Fin PINN: 3D analysis per optimisation setting (drag to turn, scroll to zoom)",
        scene={"xaxis_title": "m", "yaxis_title": "x_s", "zaxis_title": "f"},
        scene2={"xaxis_title": "x", "yaxis_title": "m", "zaxis_title": "θ"},
    )
    p = out / "FinPINN_ALL_interactive_3D.html"
    fig.write_html(str(p), include_plotlyjs=True, full_html=True)
    return p.name


PLAIN_ABOUT = """Physics: the steady cooling fin  -theta'' + m^2 theta = q(x; x_s)  on 0 <= x <= 1 (dimensionless),
heat sink theta(0) = 0, convective tip theta'(1) + Bi theta(1) = 0 with Bi = 1, a Gaussian heat source of unit power
at x_s (width 0.05), m^2 = h P L^2 / (k A), copper. Objective f = theta_max / 0.1 + 0.1 (m / 4)^2, tip limit theta(1) <= 0.02.

1  Extracted data: nine optimisation settings run on the fin with the finite-difference model; every design they
   evaluate is kept with its temperature profile (101 points). A 7 x 7 grid of other designs is held out for testing.
2  A parametric PINN theta(x; m, x_s) trained on that data and on the fin equation and tip condition, and a
   data-only network of the same size for comparison.
3  Analysis: held-out errors, the equation residual, the energy balance (sink + lateral + tip = source), each
   optimiser's path and optimum, and every optimiser re-run with the PINN in the loop, its optimum checked by FD."""


def run(out: Path, profile: str = "quick", seed: int = 0) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    t_all = time.perf_counter()
    steps, lb = (3000, 150) if profile == "quick" else (8000, 400)
    if profile == "smoke":
        steps, lb = 60, 0
    log("1  Extracting fin data from every optimisation setting (FD model, copper, Bi = 1)")
    ext = extract(seed, log)
    runs = ext["runs"]
    D = np.vstack([r["designs"] for r in runs.values() if len(r["designs"])])
    T = np.vstack([r["fields"] for r in runs.values() if len(r["designs"])])
    tD, tT = test_grid()
    np.savez_compressed(out / "fin_data.npz", x=ext["x"], designs=D, fields=T, test_designs=tD, test_fields=tT)
    with open(out / "evaluations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["setting", "index", "m", "x_s", "f", "g_tip", "feasible", "theta_max", "theta_tip"])
        for k, r in runs.items():
            for i, (d, f, g, ok, th) in enumerate(zip(r["designs"], r["f"], r["g"], r["feasible"], r["fields"])):
                w.writerow(
                    [
                        k,
                        i,
                        f"{d[0]:.6f}",
                        f"{d[1]:.6f}",
                        f"{f:.6f}",
                        f"{g:.6f}",
                        int(ok),
                        f"{th.max():.6f}",
                        f"{th[-1]:.6f}",
                    ]
                )
    log(f"   {len(D)} designs x {NX} points = {len(D) * NX} training values; {len(tD)} held-out designs")

    log(f"2  Training the parametric PINN ({steps} Adam steps + {lb} L-BFGS) and a data-only network")
    t0 = time.perf_counter()
    pinn, hist = train(D, T, True, steps, lb, seed, log)
    t_pinn = time.perf_counter() - t0
    t0 = time.perf_counter()
    dnet, hist_d = train(D, T, False, steps, lb, seed, log)
    t_data = time.perf_counter() - t0

    def rel(model: FinPINN) -> list[float]:
        return [
            float(np.linalg.norm(model.predict(ext["x"], a, b) - t) / max(1e-12, np.linalg.norm(t)))
            for (a, b), t in zip(tD, tT)
        ]

    err = {"pinn_case_rel": rel(pinn), "data_case_rel": rel(dnet)}
    xr = np.linspace(0, 1, 401)
    resid = []
    for a, b in tD:
        th, d1, d2 = pinn.derivatives(xr, a, b)
        resid.append(float(np.sqrt(np.mean((-d2 + a**2 * th - source(xr, b)) ** 2))))
    resid_d = []
    for a, b in tD:
        th, d1, d2 = dnet.derivatives(xr, a, b)
        resid_d.append(float(np.sqrt(np.mean((-d2 + a**2 * th - source(xr, b)) ** 2))))
    err["pinn_residual_rms"] = float(np.mean(resid))
    err["data_residual_rms"] = float(np.mean(resid_d))
    log(
        f"   held-out relative L2: PINN {100 * np.mean(err['pinn_case_rel']):.3f} % (max {100 * np.max(err['pinn_case_rel']):.3f} %), "
        f"data-only {100 * np.mean(err['data_case_rel']):.3f} % (max {100 * np.max(err['data_case_rel']):.3f} %)"
    )
    log(
        f"   fin-equation residual (rms): PINN {err['pinn_residual_rms']:.3e}, data-only {err['data_residual_rms']:.3e}"
    )

    log("3  Energy balance at each optimum (FD and PINN)")
    bal: dict[str, Any] = {}
    for k, r in runs.items():
        if not len(r["designs"]):
            continue
        m, x_s = r["designs"][r["best"]]
        xf, tf = fd_field(m, x_s, 1600)
        dfd = (tf[1] - tf[0]) / (xf[1] - xf[0])
        th, d1, _ = pinn.derivatives(xr, m, x_s)
        bal[k] = {"fd": energy_balance(xf, tf, dfd, m, x_s), "pinn": energy_balance(xr, th, d1[0], m, x_s)}
        b, p = bal[k]["fd"], bal[k]["pinn"]
        log(
            f"  {r['label']:24s} FD: sink {b['sink']:.3f} lateral {b['lateral']:.3f} tip {b['tip']:.4f} (imbalance {100 * b['imbalance']:+.2f} %); "
            f"PINN imbalance {100 * p['imbalance']:+.2f} %"
        )

    log("4  Every optimiser again with the PINN in the loop, optimum checked by FD (1600 cells)")
    loop = pinn_in_loop(pinn, log)

    log("5  Figures (2D and 3D, one style per setting)")
    names = figures(out, ext, (tD, tT), pinn, hist, hist_d, err, loop, bal)
    names += setting_figures(out, ext, pinn, bal, loop)
    write_manifest(out)
    (out / "training_history.json").write_text(json.dumps({"pinn": hist, "data_only": hist_d}), encoding="utf-8")
    h = html3d(out, ext, pinn)
    if h:
        names.append(h)
    try:
        import torch

        torch.save(pinn.net.state_dict(), out / "fin_pinn.pt")
        torch.save(dnet.net.state_dict(), out / "fin_data_only.pt")
    except Exception:  # noqa: BLE001
        pass

    res = {
        "profile": profile,
        "settings": {
            k: {
                "label": r["label"],
                "n_evals": int(len(r["f"])),
                "best_f": float(r["f"][r["best"]]) if len(r["f"]) else None,
                "best_m": float(r["designs"][r["best"], 0]) if len(r["f"]) else None,
                "best_x_s": float(r["designs"][r["best"], 1]) if len(r["f"]) else None,
                "feasible": bool(r["feasible"][r["best"]]) if len(r["f"]) else None,
                "message": r["message"],
                "wall_s": r["wall_s"],
            }
            for k, r in runs.items()
        },
        "n_training_designs": int(len(D)),
        "n_test_designs": int(len(tD)),
        "pinn": {
            "held_out_rel_l2_mean": float(np.mean(err["pinn_case_rel"])),
            "held_out_rel_l2_max": float(np.max(err["pinn_case_rel"])),
            "residual_rms": err["pinn_residual_rms"],
            "train_s": t_pinn,
        },
        "data_only": {
            "held_out_rel_l2_mean": float(np.mean(err["data_case_rel"])),
            "held_out_rel_l2_max": float(np.max(err["data_case_rel"])),
            "residual_rms": err["data_residual_rms"],
            "train_s": t_data,
        },
        "energy_balance": bal,
        "pinn_in_loop": loop,
        "figures": names,
        "wall_s": time.perf_counter() - t_all,
    }
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    summ = [
        "FIN PINN STUDY",
        "=" * 60,
        PLAIN_ABOUT,
        "",
        "RUN LOG",
        "-" * 60,
        *lines,
        "",
        "STYLES (the same in every figure)",
        "-" * 60,
    ]
    summ += [
        f"  {lab:24s} colour {col}  marker '{mk}'  line {ls if isinstance(ls, str) else 'custom dash'}"
        for _k, lab, col, mk, ls, *_ in SETTINGS
    ]
    summ += [
        "",
        "FIGURES",
        "-" * 60,
        *[f"  {n}" for n in names],
        "",
        "The interactive 3D page (FinPINN_ALL_interactive_3D.html) opens in the browser: drag to turn, scroll to zoom, click a legend entry to hide a setting.",
        f"Total time {res['wall_s']:.0f} s.",
    ]
    (out / "summary.txt").write_text("\n".join(summ), encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results/studio/fin_pinn")
    ap.add_argument("--profile", default="quick", choices=["smoke", "quick", "full"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    try:
        import torch

        torch.set_num_threads(4)
    except Exception:  # noqa: BLE001
        pass
    run(Path(a.out), a.profile, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
