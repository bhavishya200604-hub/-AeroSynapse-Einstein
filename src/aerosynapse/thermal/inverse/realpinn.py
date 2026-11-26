"""Phase 10.5: a physics-informed neural network TRAINED ON REAL MEASURED soil temperatures (inverse problem).

The temperature T(z, t) between the top and bottom sensors is a network and the thermal diffusivity alpha is a trainable
parameter. The loss is the misfit to the measured temperatures at the training sensors (the boundary sensors are data,
not imposed) plus the residual of T_t = alpha T_zz at collocation points, both in scaled units:

    zeta = (z - z_top) / L,  tau = (t - t0) / t_w,  T^ = (T - m) / s,
    T^_tau = kappa T^_zetazeta,  kappa = alpha t_w / L^2

The residual is divided by omega = 2 pi D (D = days in the window): the daily cycle makes T^_tau of order omega, and an
unscaled residual (about 10^3 times the data term at the start) drives the network to a flat solution that ignores the
data (seen in the first trial, D149). Training starts with `warmup` data-only steps, then adds the residual.

Inputs are zeta, tau and diurnal Fourier features sin / cos(2 pi k D tau), k = 1..K (D = days in the window): the
measured forcing is dominated by the daily cycle, which a plain MLP in tau would learn slowly (spectral bias, RQ2).

Validation is leave-one-sensor-out: one sensor is never shown to the network, and the verified classical estimator
(Crank-Nicolson model of src/inverse/realdata.py) is fitted on the same sensors and predicts the same held-out sensor,
so both are scored on identical data (after the same spin-up). Float64 throughout; deterministic for a given seed.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np

DAY = 86400.0


def _torch():
    import torch

    torch.set_default_dtype(torch.float64)
    return torch


class _Net:
    """MLP on (zeta, tau, diurnal features) -> T^; tanh; Glorot init from the seed."""

    def __init__(self, torch, width: int, depth: int, n_days: float, harmonics: int, seed: int):
        torch.manual_seed(seed)
        self.t, self.n_days, self.K = torch, float(n_days), int(harmonics)
        d_in = 2 + 2 * self.K
        sizes = [d_in] + [width] * depth + [1]
        self.layers = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:], strict=True))
        for lay in self.layers:
            torch.nn.init.xavier_normal_(lay.weight)
            torch.nn.init.zeros_(lay.bias)

    def features(self, zeta, tau):
        f = [zeta, tau]
        for k in range(1, self.K + 1):
            w = 2 * math.pi * k * self.n_days
            f += [self.t.sin(w * tau), self.t.cos(w * tau)]
        return self.t.cat(f, 1)

    def __call__(self, zeta, tau):
        h = self.features(zeta, tau)
        for lay in self.layers[:-1]:
            h = self.t.tanh(lay(h))
        return self.layers[-1](h)

    def parameters(self):
        return list(self.layers.parameters())


def train(
    win: dict,
    train_z: list[float],
    top: float,
    bottom: float,
    *,
    seed: int = 0,
    width: int = 32,
    depth: int = 3,
    harmonics: int = 2,
    n_colloc: int = 1500,
    warmup: int = 500,
    resample_every: int = 250,
    adam_steps: int = 3000,
    lr: float = 3e-3,
    lbfgs_steps: int = 300,
    w_pde: float = 1.0,
    alpha_init: float = 1e-6,
    progress: Callable[[dict], None] | None = None,
    every: int = 100,
) -> dict:
    """Train on the window `win` (realdata.window) using the sensors at depths `train_z` (must include top and bottom).

    Returns alpha, the loss history, the network's predictions at every sensor of the window (held-out ones included)
    and the wall time. `progress(record)` is called every `every` steps with the current losses and alpha."""
    torch = _torch()
    t0w = time.perf_counter()
    z_all, T_all, t = np.asarray(win["z"], float), np.asarray(win["T"], float), np.asarray(win["t"], float)
    cols = [int(np.argmin(np.abs(z_all - z))) for z in train_z]
    if any(abs(z_all[c] - z) > 1e-9 for c, z in zip(cols, train_z, strict=True)):
        raise ValueError(f"training depths {train_z} not all in the window's sensors {z_all.tolist()}")
    if min(train_z) > top + 1e-9 or max(train_z) < bottom - 1e-9:
        raise ValueError("the training sensors must include the top and bottom sensors")
    L, tw = bottom - top, t[-1] - t[0]
    n_days = tw / DAY
    Tm, Ts = float(np.mean(T_all[:, cols])), float(np.std(T_all[:, cols]))
    zz, tt = np.meshgrid(z_all[cols], t)
    X = torch.tensor(np.stack([(zz.ravel() - top) / L, (tt.ravel() - t[0]) / tw], 1))
    Y = torch.tensor(((T_all[:, cols] - Tm) / Ts).ravel()[:, None])
    net = _Net(torch, width, depth, n_days, harmonics, seed)
    log_alpha = torch.nn.Parameter(torch.tensor(math.log(alpha_init)))
    params = net.parameters() + [log_alpha]
    gen = torch.Generator().manual_seed(seed + 12345)

    def colloc():
        c = torch.rand(n_colloc, 2, generator=gen)
        return c.requires_grad_(True)

    Xc = colloc()

    def losses(Xc):
        pred = net(X[:, :1], X[:, 1:])
        l_data = torch.mean((pred - Y) ** 2)
        zeta, tau = Xc[:, :1], Xc[:, 1:]
        u = net(zeta, tau)
        u_t, u_z = torch.autograd.grad(u.sum(), (tau, zeta), create_graph=True)
        u_zz = torch.autograd.grad(u_z.sum(), zeta, create_graph=True)[0]
        kappa = torch.exp(log_alpha) * tw / L**2
        l_pde = torch.mean(((u_t - kappa * u_zz) / omega) ** 2)
        return l_data, l_pde

    omega = 2 * math.pi * n_days
    hist: list[dict] = []

    def record(step, phase, l_data, l_pde):
        rec = {
            "step": step,
            "phase": phase,
            "loss": float(l_data + w_pde * l_pde),
            "data": float(l_data),
            "pde": float(l_pde),
            "alpha": float(torch.exp(log_alpha.detach())),
            "wall_s": time.perf_counter() - t0w,
        }
        hist.append(rec)
        if progress:
            progress(rec)

    opt = torch.optim.Adam(params, lr=lr)
    # step decay of the learning rate at 60 % and 85 % of the Adam steps
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.6 * adam_steps), int(0.85 * adam_steps)], gamma=0.3)
    for k in range(adam_steps):
        if resample_every and k and k % resample_every == 0:
            Xc = colloc()
        opt.zero_grad()
        l_data, l_pde = losses(Xc)
        w = 0.0 if k < warmup else w_pde
        (l_data + w * l_pde).backward()
        opt.step()
        sched.step()
        if k % every == 0 or k == adam_steps - 1:
            record(k, "adam" if k >= warmup else "warm-up", l_data.detach(), l_pde.detach())
    if lbfgs_steps:
        Xc = colloc()
        lb = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=lbfgs_steps,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-15,
            history_size=50,
        )

        def closure():
            lb.zero_grad()
            l_data, l_pde = losses(Xc)
            loss = l_data + w_pde * l_pde
            loss.backward()
            return loss

        lb.step(closure)
        l_data, l_pde = losses(Xc)
        record(adam_steps + lbfgs_steps, "lbfgs", l_data.detach(), l_pde.detach())
    with torch.no_grad():
        zz, tt = np.meshgrid(z_all, t)
        Xa = torch.tensor(np.stack([(zz.ravel() - top) / L, (tt.ravel() - t[0]) / tw], 1))
        P = net(Xa[:, :1], Xa[:, 1:]).numpy().reshape(len(t), len(z_all)) * Ts + Tm
    return {
        "alpha": float(torch.exp(log_alpha.detach())),
        "history": hist,
        "prediction": P,
        "z": z_all,
        "train_z": list(train_z),
        "scales": {"L_m": L, "t_window_s": tw, "T_mean": Tm, "T_std": Ts},
        "config": {
            "seed": seed,
            "width": width,
            "depth": depth,
            "harmonics": harmonics,
            "n_colloc": n_colloc,
            "warmup": warmup,
            "adam_steps": adam_steps,
            "lr": lr,
            "lbfgs_steps": lbfgs_steps,
            "w_pde": w_pde,
            "alpha_init": alpha_init,
        },
        "n_parameters": int(sum(p.numel() for p in params)),
        "wall_s": time.perf_counter() - t0w,
    }


def rmse_after(t: np.ndarray, obs: np.ndarray, pred: np.ndarray, spinup_s: float) -> float:
    m = (t - t[0]) >= spinup_s
    return float(np.sqrt(np.mean((np.asarray(pred)[m] - np.asarray(obs)[m]) ** 2)))


def classical_loo(win: dict, train_inner: list[float], holdout_z: float, mcfg: dict, fcfg: dict) -> dict:
    """The verified Crank-Nicolson estimator fitted on the same sensors (top, bottom and train_inner) and its prediction
    at the held-out depth (the same model, read out there)."""
    from aerosynapse.thermal.inverse import realdata as R

    m = dict(mcfg, inner_m=list(train_inner))
    r = R.fit_conduction(win, m, fcfg)
    alpha = float(np.exp(r["x"][0]))
    model = R.Conduction1D(
        win["t"],
        win["z"],
        win["T"],
        m["top_m"],
        m["bottom_m"],
        [holdout_z],
        float(m.get("dz_m", 0.0025)),
        float(m.get("dt_s", 600)),
        1,
        float(m.get("spinup_h", 24)) * 3600,
    )
    return {"alpha": alpha, "converged": bool(r["fit"]["converged"]), "prediction": np.real(model.run(alpha))[:, 0]}
