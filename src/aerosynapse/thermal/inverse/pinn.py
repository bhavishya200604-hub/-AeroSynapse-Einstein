"""Step 5.2: PINN inverse solver for (alpha, k, h) of the 1D body (slab / cylinder / sphere).

Network ansatz, in r_hat = r/L in [0, 1], t_hat = t / t_ref (t_ref = L^2 / alpha_guess, so the diffusion factor starts
near 1) and s = r_hat^2:

    theta(r, t) = 1 + t_hat N(s, t_hat)

so the initial condition theta = 1 holds exactly and theta is even in r_hat (zero gradient at the centre for every
shape). With d/dr_hat = 2 r_hat d/ds the radial Laplacian is regular:
r^-n (r^n theta_r)_r = 2 (n+1) theta_s + 4 s theta_ss.
Residuals (theta = (T - T_inf)/(T0 - T_inf)):

    pde   theta_t_hat - F (2 (n+1) theta_s + 4 s theta_ss) - F Q,   F = alpha t_ref / L^2,   Q = q_vol L^2 / (k dT)
    bc    -2 theta_s(1, t) - Bi theta(1, t),                         Bi = h L / k
    data  theta(s_i, t_j) - theta_ij

Unknown parameters are trainable LOG-parameters (any subset of alpha, k, h; the others are known constants).
Modes:
  joint      network and parameters trained together on w_pde L_pde + w_bc L_bc + w_data L_data (Adam, then L-BFGS)
  two_stage  (1) network fitted to the data alone, (2) parameters fitted to the physics residuals of the FROZEN network
             (Adam, then L-BFGS on the log-parameters only)
Balancing: fixed weights, or grad_balance (w_data <- EMA of |grad L_phys| / |grad L_data| over the network weights,
every `every` Adam steps; frozen for L-BFGS).
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from aerosynapse.thermal.data.design import sobol_unit
from aerosynapse.thermal.inverse.body import PARAMS, Body
from aerosynapse.thermal.pinn.architectures import build_network

DT = torch.float64


def _grad(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    g = torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True, allow_unused=True)[0]
    return torch.zeros_like(x) if g is None else g  # f independent of x (e.g. a linear field's second derivative)


class InversePINN(nn.Module):
    def __init__(
        self,
        net_cfg: dict,
        body: Body,
        t_ref: float,
        t_hat_max: float,
        unknown: list[str],
        init: dict,
        known: dict,
        time_coordinate: str = "linear",
    ):
        super().__init__()
        if time_coordinate not in ("linear", "sqrt"):
            raise ValueError("time_coordinate: linear or sqrt")
        self.body, self.t_ref, self.n, self.coord = body, float(t_ref), body.n, time_coordinate
        self.c_max = float(t_hat_max) if time_coordinate == "linear" else math.sqrt(float(t_hat_max))
        self.net = build_network(net_cfg, [0.0, 0.0], [1.0, self.c_max], 1).to(DT)
        self.unknown = list(unknown)
        self.logp = nn.ParameterDict(
            {k: nn.Parameter(torch.tensor(math.log(float(init[k])), dtype=DT)) for k in self.unknown}
        )
        self.known = {k: math.log(float(known[k])) for k in PARAMS if k not in self.unknown}

    def params(self) -> dict[str, torch.Tensor]:
        return {
            k: torch.exp(self.logp[k]) if k in self.logp else torch.tensor(math.exp(self.known[k]), dtype=DT)
            for k in PARAMS
        }

    def estimates(self) -> dict[str, float]:
        return {k: float(v.detach()) for k, v in self.params().items()}

    def c_of(self, th: torch.Tensor) -> torch.Tensor:
        """Network time coordinate of t_hat: c = t_hat (linear) or c = sqrt(t_hat) (sqrt: the similarity variable of
        the early-time surface layer, where theta changes like sqrt(t))."""
        return th if self.coord == "linear" else torch.sqrt(th)

    def theta_c(self, s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return 1.0 + c * self.net(torch.stack([s, c], -1)).squeeze(-1)

    def theta(self, s: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        return self.theta_c(s, self.c_of(th))

    def groups(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p, b = self.params(), self.body
        F = p["alpha"] * self.t_ref / b.L**2
        return F, p["h"] * b.L / p["k"], F * b.q_vol * b.L**2 / (p["k"] * b.dT)

    def pde_residual(self, s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """At collocation points (s, c) in NETWORK coordinates. linear: theta_t_hat - F (Lap + Q); sqrt: the same
        multiplied by 2c (= d t_hat / d c), i.e. theta_c - 2 c F (Lap + Q), regular at c = 0."""
        s, c = s.detach().clone().requires_grad_(True), c.detach().clone().requires_grad_(True)
        u = self.theta_c(s, c)
        u_s, u_c = _grad(u, s), _grad(u, c)
        u_ss = _grad(u_s, s)
        F, _, FQ = self.groups()
        rhs = F * (2.0 * (self.n + 1) * u_s + 4.0 * s * u_ss) + FQ
        return u_c - (rhs if self.coord == "linear" else 2.0 * c * rhs)

    def bc_residual(self, c: torch.Tensor) -> torch.Tensor:
        """Robin condition at the surface, at network times c."""
        s = torch.ones_like(c).requires_grad_(True)
        u = self.theta_c(s, c.detach())
        _, Bi, _ = self.groups()
        return -2.0 * _grad(u, s) - Bi * u


def collocation(
    n_pde: int, n_bc: int, t_hat_max: float, seed: int, box: tuple | None = None
) -> dict[str, torch.Tensor]:
    """Sobol points: (s, t_hat) interior with r_hat in (0, 1] (or inside box = (r_lo, r_hi, t_lo, t_hi)) and t_hat on
    the surface."""
    u = sobol_unit(n_pde, 2, seed)
    r_lo, r_hi, t_lo, t_hi = box if box else (0.0, 1.0, 0.0, t_hat_max)
    r = r_lo + (r_hi - r_lo) * u[:, 0]
    t = t_lo + (t_hi - t_lo) * u[:, 1]
    tb = sobol_unit(n_bc, 1, seed + 1)[:, 0] * t_hat_max
    mk = lambda a: torch.as_tensor(a, dtype=DT)  # noqa: E731
    return {"s": mk(r**2), "t": mk(t), "t_bc": mk(tb)}


def _flat_grad(loss: torch.Tensor, params: list) -> torch.Tensor:
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([x.reshape(-1) for x in g if x is not None])


def train(
    cfg: dict,
    body: Body,
    data: dict,
    unknown: list[str],
    init: dict,
    known: dict,
    mode: str = "joint",
    seed: int = 0,
    eval_grid: dict | None = None,
) -> dict[str, Any]:
    """Train one inverse PINN. data: {r [m], t [s], T} (1D arrays of equal length). Returns estimates, losses, the
    parameter history (every log_every Adam steps and per L-BFGS stage) and the wall time."""
    torch.manual_seed(seed)
    t_ref = body.L**2 / float(init["alpha"] if "alpha" in unknown else known["alpha"])
    t_hat = np.asarray(data["t"], float) / t_ref
    t_hat_max = float(t_hat.max())
    model = InversePINN(
        cfg["network"], body, t_ref, t_hat_max, unknown, init, known, cfg.get("time_coordinate", "linear")
    )
    mk = lambda a: torch.as_tensor(np.asarray(a, float), dtype=DT)  # noqa: E731
    sd, td = mk((np.asarray(data["r"], float) / body.L) ** 2), mk(t_hat)
    thd = mk((np.asarray(data["T"], float) - body.T_inf) / body.dT)
    col = cfg["collocation"]
    C = collocation(int(col["n_pde"]), int(col["n_bc"]), model.c_max, int(col.get("seed", 0)) + seed)  # in network time
    w = {k: float(v) for k, v in cfg["weights"].items()}
    bal = cfg.get("balancing") or {"method": "fixed"}
    hist: list[dict] = []
    net_params = list(model.net.parameters())

    def losses() -> dict[str, torch.Tensor]:
        return {
            "pde": (model.pde_residual(C["s"], C["t"]) ** 2).mean(),
            "bc": (model.bc_residual(C["t_bc"]) ** 2).mean(),
            "data": ((model.theta(sd, td) - thd) ** 2).mean(),
        }

    def log(stage: str, step: int, L: dict) -> None:
        hist.append(
            {
                "stage": stage,
                "step": step,
                **{f"loss_{k}": float(v.detach()) for k, v in L.items()},
                **model.estimates(),
                "w_data": w["data"],
            }
        )

    def run_adam(epochs: int, total, params_groups, stage: str) -> None:
        opt = torch.optim.Adam(params_groups)
        for ep in range(epochs):
            opt.zero_grad()
            L = losses()
            if stage == "joint" and bal.get("method") == "grad_balance" and ep % int(bal.get("every", 50)) == 0:
                gp = _flat_grad(w["pde"] * L["pde"] + w["bc"] * L["bc"], net_params).norm()
                gd = _flat_grad(L["data"], net_params).norm()
                target = float(gp / max(float(gd), 1e-300))
                a = float(bal.get("alpha", 0.9))
                w["data"] = float(np.clip(a * w["data"] + (1 - a) * target, 1e-3, 1e6))
            tot = total(L)
            tot.backward()
            opt.step()
            if ep % int(cfg.get("log_every", 50)) == 0 or ep == epochs - 1:
                log(stage, ep, L)

    def run_lbfgs(iters: int, total, params, stage: str) -> None:
        if iters <= 0:
            return
        opt = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=iters,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-16,
        )

        def closure():
            opt.zero_grad()
            t = total(losses())
            t.backward()
            return t

        opt.step(closure)
        log(stage, iters, losses())

    t0 = time.perf_counter()
    ad, lb = cfg["adam"], cfg["lbfgs"]
    lr_net, lr_par = float(ad["lr"]), float(ad["lr_params"])
    if mode == "joint":
        tot = lambda L: w["pde"] * L["pde"] + w["bc"] * L["bc"] + w["data"] * L["data"]  # noqa: E731
        run_adam(
            int(ad["epochs"]),
            tot,
            [{"params": net_params, "lr": lr_net}, {"params": list(model.logp.values()), "lr": lr_par}],
            "joint",
        )
        run_lbfgs(int(lb["iterations"]), tot, list(model.parameters()), "joint_lbfgs")
    elif mode == "two_stage":
        ts = cfg["two_stage"]
        for p in model.logp.values():
            p.requires_grad_(False)
        run_adam(int(ts["fit_epochs"]), lambda L: L["data"], [{"params": net_params, "lr": lr_net}], "fit")
        run_lbfgs(int(ts["fit_lbfgs"]), lambda L: L["data"], net_params, "fit_lbfgs")
        for p in net_params:
            p.requires_grad_(False)
        for p in model.logp.values():
            p.requires_grad_(True)
        phys = lambda L: w["pde"] * L["pde"] + w["bc"] * L["bc"]  # noqa: E731
        run_adam(int(ts["param_epochs"]), phys, [{"params": list(model.logp.values()), "lr": lr_par}], "params")
        run_lbfgs(int(lb["iterations"]), phys, list(model.logp.values()), "params_lbfgs")
    elif mode == "staged":  # warm-up: network fitted to the data with the parameters frozen, then joint training
        for p in model.logp.values():
            p.requires_grad_(False)
        run_adam(
            int(cfg["staged"]["warmup_epochs"]), lambda L: L["data"], [{"params": net_params, "lr": lr_net}], "warmup"
        )
        for p in model.logp.values():
            p.requires_grad_(True)
        tot = lambda L: w["pde"] * L["pde"] + w["bc"] * L["bc"] + w["data"] * L["data"]  # noqa: E731
        run_adam(
            int(ad["epochs"]),
            tot,
            [{"params": net_params, "lr": lr_net}, {"params": list(model.logp.values()), "lr": lr_par}],
            "joint",
        )
        run_lbfgs(int(lb["iterations"]), tot, list(model.parameters()), "joint_lbfgs")
    elif mode == "forward":  # diagnostic: parameters frozen at `init`, only the network is trained
        for p in model.logp.values():
            p.requires_grad_(False)
        tot = lambda L: w["pde"] * L["pde"] + w["bc"] * L["bc"] + w["data"] * L["data"]  # noqa: E731
        run_adam(int(ad["epochs"]), tot, [{"params": net_params, "lr": lr_net}], "forward")
        run_lbfgs(int(lb["iterations"]), tot, net_params, "forward_lbfgs")
    else:
        raise ValueError(f"mode '{mode}': joint, staged, two_stage or forward")
    wall = time.perf_counter() - t0
    L = losses()
    with torch.no_grad():
        fit_K = float(torch.sqrt(((model.theta(sd, td) - thd) ** 2).mean())) * abs(body.dT)
    pred = None
    if eval_grid is not None:  # theta on a grid {r: [m], t: [s]} -> array (len(t), len(r))
        R, Tt = np.meshgrid(np.asarray(eval_grid["r"], float) / body.L, np.asarray(eval_grid["t"], float) / t_ref)
        with torch.no_grad():
            pred = model.theta(mk(R.ravel() ** 2), mk(Tt.ravel())).numpy().reshape(R.shape)
    out_res = {}
    for name, (s_, t_) in {"pde_rms": (C["s"], C["t"])}.items():
        out_res[name] = float(torch.sqrt((model.pde_residual(s_, t_).detach() ** 2).mean()))
    return {
        "theta_grid": pred,
        "residual": out_res,
        "estimates": model.estimates(),
        "losses": {k: float(v.detach()) for k, v in L.items()},
        "data_rmse_K": fit_K,
        "w_data_final": w["data"],
        "history": hist,
        "wall_s": wall,
        "t_ref": t_ref,
    }
