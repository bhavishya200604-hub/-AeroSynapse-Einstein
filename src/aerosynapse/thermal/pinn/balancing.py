"""Loss-balancing suite (Step 4.3) behind one interface.

Total loss = sum_k g_k * w_k * L_k with L_k = mean(r_k^2) the per-component losses (1D: pde, ic, bc_left, bc_right;
2D: pde_*, bc_*, gauge), g_k the FIXED group weights of ``loss.weights`` (a user prior; 1 by default) and w_k the
adaptive weights of the chosen method. Adaptive weights are updated during Adam only and FROZEN for L-BFGS (L-BFGS
needs a fixed objective). Formulas were checked against the arXiv PDFs (docs/LITERATURE.md); choices not fixed by the
papers are marked "(our choice)".

    fixed          w_k = 1 (Trainer keeps its original code path)
    lr_annealing   Wang, Teng & Perdikaris (arXiv:2001.04536, eq. 41): lam_hat_k = max|grad L_ref| / mean|grad L_k|,
                   lam_k <- (1 - alpha) lam_k + alpha lam_hat_k (alpha = 0.9 recommended); reference = PDE terms
                   (weight 1). Update interval `every` (our choice, default 10).
    gradnorm       Chen et al. (arXiv:1711.02257): G_k = ||grad_W w_k L_k||, r_k = (L_k/L_k(0)) / mean(.),
                   L_grad = sum_k |G_k - mean(G) r_k^alpha| with the target held constant; Adam on w; w renormalised
                   to sum to m. W = last parameterised layer; alpha, lr, every: our choices (1.5, 0.025, 10).
    ntk            Wang, Yu & Perdikaris (arXiv:2007.14527): lam_k = Tr(K) / Tr(K_kk), K_kk = J_k J_k^T. Traces are
                   estimated from n_points sampled residuals. trace: 'sum' = literal block trace (scales with the
                   number of points); 'mean' = per-point trace (default; for MEAN-squared losses the k-th block of
                   the gradient-flow kernel is w_k K_kk / N_k, so equalising w_k Tr(K_kk)/N_k is the analogue - D51).
    softadapt      Heydari et al. (arXiv:1912.12355, eq. 3 + Algorithm 1): alpha_k = softmax(beta s_k), s_k = rate of
                   change of L_k over a window; variants normalized / loss_weighted. Weights scaled by m (uniform -> 1;
                   the original sums to 1, D52).
    relobralo      Bischof & Kraus (arXiv:2110.09813, eq. 11): bal(t,t') = m softmax(L(t) / (T L(t'))),
                   hist = rho w(t-1) + (1 - rho) bal(t,0), w = alpha hist + (1 - alpha) bal(t,t-1), rho ~ Bernoulli.
    uncertainty    Kendall, Gal & Cipolla (arXiv:1705.07115): sum_k exp(-s_k)/2 L_k + s_k/2, s = log sigma^2 trained.
    self_adaptive  McClenny & Braga-Neto (arXiv:2009.04544, eqs. 10-17): L_k = mean(m(lam_ki) r_ki^2) with one weight
                   per collocation point, minimised over the network and MAXIMISED over lam (Adam ascent, lr 5e-3).
                   Requires a fixed collocation set (sampling.resample_every = 0). Mask m: our choice (square).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

METHODS = ("fixed", "lr_annealing", "gradnorm", "ntk", "softadapt", "relobralo", "uncertainty", "self_adaptive")


def group_of(name: str) -> str:
    for g in ("pde", "bc", "ic", "gauge", "data"):
        if name.startswith(g):
            return g
    raise ValueError(f"loss component {name!r} has no group (pde/bc/ic/gauge/data)")


def _params(model: torch.nn.Module) -> list[Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def _flat_grad(loss: Tensor, params: list[Tensor]) -> Tensor:
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat(
        [(gi if gi is not None else torch.zeros_like(p)).reshape(-1) for gi, p in zip(g, params, strict=True)]
    )


def last_layer_params(model: torch.nn.Module) -> list[Tensor]:
    """Parameters owned directly by the last module that owns any (the output layer for every architecture)."""
    last: list[Tensor] = []
    for m in model.modules():
        own = [p for p in m.parameters(recurse=False) if p.requires_grad]
        if own:
            last = own
    return last


class Balancer:
    method = "base"

    def __init__(self, cfg: dict[str, Any], group_weights: dict[str, float], seed: int):
        self.cfg = cfg
        self.mc = cfg.get(self.method, {}) or {}
        self.gw = dict(group_weights)
        self.seed = int(seed)
        self.lo, self.hi = (float(v) for v in cfg.get("weight_bounds", [1e-6, 1e6]))
        self.names: list[str] | None = None
        self.w: dict[str, float] = {}
        self.frozen = False
        self.n_updates = 0
        self.n_clamped = 0

    # --- interface used by the Trainer ---------------------------------------------------------------------------
    def prepare(self, model: torch.nn.Module, comp: dict[str, Tensor], vecs: dict[str, Tensor], step: int) -> None:
        """Called once per Adam step BEFORE the backward pass (may differentiate the components)."""
        if self.names is None:
            self.names = list(comp)
            self.w = {k: 1.0 for k in self.names}
            self._init(model, comp, vecs)
        elif list(comp) != self.names:
            raise RuntimeError(f"loss components changed during training: {list(comp)} vs {self.names}")
        if self.frozen:
            return
        self._observe(comp, step)
        if step % int(self.mc.get("every", 1)) == 0:
            self._update(model, comp, vecs, step)
            self.n_updates += 1

    def total(self, comp: dict[str, Tensor], vecs: dict[str, Tensor]) -> Tensor:
        return sum(self.gw[group_of(k)] * self.w[k] * comp[k] for k in comp)

    def after_backward(self, step: int) -> None:
        """Called after the network's optimiser step (methods with trainable weights step their own optimiser)."""

    def freeze(self) -> None:
        self.frozen = True

    def effective(self) -> dict[str, float]:
        """Current g_k * w_k (for logging and the report)."""
        return {k: self.gw[group_of(k)] * float(self.w[k]) for k in (self.names or [])}

    def state_dict(self) -> dict[str, Any]:
        return {
            "names": self.names,
            "w": dict(self.w),
            "frozen": self.frozen,
            "n_updates": self.n_updates,
            "n_clamped": self.n_clamped,
            "extra": self._extra_state(),
        }

    def load_state_dict(self, s: dict[str, Any]) -> None:
        self.names, self.w, self.frozen = s["names"], dict(s["w"]), s["frozen"]
        self.n_updates, self.n_clamped = s["n_updates"], s["n_clamped"]
        self._load_extra(s["extra"])

    # --- hooks ---------------------------------------------------------------------------------------------------
    def _init(self, model: torch.nn.Module, comp: dict[str, Tensor], vecs: dict[str, Tensor]) -> None:
        pass

    def _update(self, model: torch.nn.Module, comp: dict[str, Tensor], vecs: dict[str, Tensor], step: int) -> None:
        pass

    def _observe(self, comp: dict[str, Tensor], step: int) -> None:
        """Every Adam step (before a possible update)."""

    def _extra_state(self) -> dict[str, Any]:
        return {}

    def _load_extra(self, s: dict[str, Any]) -> None:
        pass

    def _clamp(self, v: float) -> float:
        if not math.isfinite(v) or v < self.lo or v > self.hi:
            self.n_clamped += 1
            return self.hi if (not math.isfinite(v) or v > self.hi) else self.lo
        return v


class LRAnnealing(Balancer):
    method = "lr_annealing"

    def _update(self, model, comp, vecs, step):
        params = _params(model)
        ref = self.mc.get("reference", "pde")
        refs = [k for k in self.names if group_of(k) == ref]
        g_ref = _flat_grad(sum(comp[k] for k in refs), params).abs()
        num = g_ref.max() if self.mc.get("statistic", "max") == "max" else g_ref.mean()
        a = float(self.mc.get("alpha", 0.9))
        for k in self.names:
            if k in refs:
                continue
            den = _flat_grad(comp[k], params).abs().mean()
            lam_hat = float(num / den) if float(den) > 0 else float("inf")
            self.w[k] = self._clamp((1.0 - a) * self.w[k] + a * lam_hat)


class GradNorm(Balancer):
    method = "gradnorm"

    def _init(self, model, comp, vecs):
        self.wt = torch.ones(len(self.names), dtype=torch.float64, requires_grad=True)
        self.opt = torch.optim.Adam([self.wt], lr=float(self.mc.get("lr", 0.025)))
        self.L0 = [float(comp[k].detach()) for k in self.names]

    def _update(self, model, comp, vecs, step):
        W = last_layer_params(model)
        g = []
        for k in self.names:
            gr = torch.autograd.grad(self.gw[group_of(k)] * comp[k], W, retain_graph=True, allow_unused=True)
            g.append(float(torch.sqrt(sum((x**2).sum() for x in gr if x is not None))))
        gn = torch.tensor(g, dtype=torch.float64)
        w = self.wt.detach()
        G = w * gn
        Lt = torch.tensor(
            [float(comp[k].detach()) / max(l0, 1e-300) for k, l0 in zip(self.names, self.L0, strict=True)]
        )
        r = Lt / Lt.mean()
        target = G.mean() * r ** float(self.mc.get("alpha", 1.5))  # held constant (as in the paper)
        self.opt.zero_grad()
        self.wt.grad = torch.sign(G - target) * gn  # d/dw_k |w_k g_k - target_k| (exact, no double backward)
        self.opt.step()
        with torch.no_grad():
            self.wt.clamp_(min=self.lo)
            self.wt.mul_(len(self.names) / self.wt.sum())
        self.w = {k: float(v) for k, v in zip(self.names, self.wt.detach(), strict=True)}

    def _extra_state(self):
        return {"wt": self.wt.detach().clone(), "opt": self.opt.state_dict(), "L0": self.L0} if self.names else {}

    def _load_extra(self, s):
        if s:
            self.wt = s["wt"].clone().requires_grad_(True)
            self.opt = torch.optim.Adam([self.wt], lr=float(self.mc.get("lr", 0.025)))
            self.opt.load_state_dict(s["opt"])
            self.L0 = s["L0"]


class NTKTrace(Balancer):
    method = "ntk"

    def _update(self, model, comp, vecs, step):
        params = _params(model)
        rng = np.random.default_rng(self.seed * 1_000_003 + 31 * step)
        n = int(self.mc.get("n_points", 16))
        tr = {}
        for k in self.names:
            v = vecs[k]
            idx = rng.choice(v.numel(), size=min(n, v.numel()), replace=False)
            sq = [float((_flat_grad(v.reshape(-1)[int(i)], params) ** 2).sum()) for i in idx]
            tr[k] = float(np.mean(sq)) * (v.numel() if self.mc.get("trace", "mean") == "sum" else 1.0)
        tot = sum(tr.values())
        for k in self.names:
            self.w[k] = self._clamp(tot / tr[k] if tr[k] > 0 else float("inf"))
        self.last_traces = tr


class SoftAdapt(Balancer):
    method = "softadapt"

    def _init(self, model, comp, vecs):
        self.hist: list[list[float]] = []

    def _observe(self, comp, step):
        self.hist.append([float(comp[k].detach()) for k in self.names])
        self.hist = self.hist[-int(self.mc.get("window", 10)) :]

    def _update(self, model, comp, vecs, step):
        if len(self.hist) < 2:
            return
        H = np.array(self.hist)
        s = (H[-1] - H[0]) / (len(H) - 1)  # first-order rate of change over the stored window
        if self.mc.get("normalized", True):
            s = s / (np.abs(s).sum() + 1e-8)
        a = np.exp(float(self.mc.get("beta", 0.1)) * (s - s.max()))
        a = a / (a.sum() + 1e-8)
        if self.mc.get("loss_weighted", False):
            f = H.mean(axis=0)
            a = f * a / ((f * a).sum() + 1e-8)
        m = len(self.names)
        self.w = {k: self._clamp(float(m * ai)) for k, ai in zip(self.names, a, strict=True)}

    def _extra_state(self):
        return {"hist": [list(h) for h in getattr(self, "hist", [])]}

    def _load_extra(self, s):
        self.hist = [list(h) for h in s.get("hist", [])]


class ReLoBRaLo(Balancer):
    method = "relobralo"

    def _init(self, model, comp, vecs):
        self.L0 = [float(comp[k].detach()) for k in self.names]
        self.Lprev = list(self.L0)

    def _bal(self, L: list[float], Lref: list[float]) -> np.ndarray:
        T = float(self.mc.get("temperature", 0.1))
        z = np.array([a / (T * max(b, 1e-300)) for a, b in zip(L, Lref, strict=True)])
        e = np.exp(z - z.max())
        return len(L) * e / e.sum()

    def _update(self, model, comp, vecs, step):
        L = [float(comp[k].detach()) for k in self.names]
        rho = float(np.random.default_rng(self.seed * 7919 + step).random() < float(self.mc.get("rho", 0.9999)))
        a = float(self.mc.get("alpha", 0.999))
        w = np.array([self.w[k] for k in self.names])
        hist = rho * w + (1.0 - rho) * self._bal(L, self.L0)
        new = a * hist + (1.0 - a) * self._bal(L, self.Lprev)
        self.w = {k: self._clamp(float(v)) for k, v in zip(self.names, new, strict=True)}
        self.Lprev = L

    def _extra_state(self):
        return {"L0": getattr(self, "L0", None), "Lprev": getattr(self, "Lprev", None)}

    def _load_extra(self, s):
        self.L0, self.Lprev = s["L0"], s["Lprev"]


class Uncertainty(Balancer):
    method = "uncertainty"

    def _init(self, model, comp, vecs):
        s0 = float(self.mc.get("init_log_var", 0.0))
        self.s = torch.full((len(self.names),), s0, dtype=torch.float64, requires_grad=True)
        self.opt = torch.optim.Adam([self.s], lr=float(self.mc.get("lr", 1e-3)))
        self.w = {k: 0.5 * math.exp(-s0) for k in self.names}

    def prepare(self, model, comp, vecs, step):
        super().prepare(model, comp, vecs, step)
        self.opt.zero_grad(set_to_none=True)

    def total(self, comp, vecs):
        s = self.s.detach() if self.frozen else self.s
        s = s.to(next(iter(comp.values())).dtype)
        return sum(
            0.5 * torch.exp(-s[i]) * self.gw[group_of(k)] * comp[k] + 0.5 * s[i] for i, k in enumerate(self.names)
        )

    def after_backward(self, step):
        if not self.frozen and self.s.grad is not None:
            self.opt.step()
        self.w = {k: 0.5 * math.exp(-float(v)) for k, v in zip(self.names, self.s.detach(), strict=True)}

    def _extra_state(self):
        return {"s": self.s.detach().clone(), "opt": self.opt.state_dict()} if self.names else {}

    def _load_extra(self, s):
        if s:
            self.s = s["s"].clone().requires_grad_(True)
            self.opt = torch.optim.Adam([self.s], lr=float(self.mc.get("lr", 1e-3)))
            self.opt.load_state_dict(s["opt"])


MASKS = {
    "square": lambda lam: lam**2,
    "identity": lambda lam: lam,
    "softplus": torch.nn.functional.softplus,
}


class SelfAdaptive(Balancer):
    method = "self_adaptive"

    def _init(self, model, comp, vecs):
        init = self.mc.get("init", 1.0)
        g = torch.Generator().manual_seed(self.seed * 97 + 5)
        self.lam = {}
        for k in self.names:
            n = vecs[k].numel()
            if isinstance(init, list):
                v = init[0] + (init[1] - init[0]) * torch.rand(n, generator=g, dtype=torch.float64)
            else:
                v = torch.full((n,), float(init), dtype=torch.float64)
            self.lam[k] = v.requires_grad_(True)
        self.opt = torch.optim.Adam(list(self.lam.values()), lr=float(self.mc.get("lr", 5e-3)), maximize=True)
        mask = MASKS[self.mc.get("mask", "square")]
        self.w = {k: float(mask(self.lam[k].detach()).mean()) for k in self.names}

    def prepare(self, model, comp, vecs, step):
        super().prepare(model, comp, vecs, step)
        for k in self.names:
            if vecs[k].numel() != self.lam[k].numel():
                raise RuntimeError(
                    f"self_adaptive: component {k} has {vecs[k].numel()} points but {self.lam[k].numel()} weights; "
                    "the collocation set must stay fixed (sampling.resample_every = 0)"
                )
        self.opt.zero_grad(set_to_none=True)

    def total(self, comp, vecs):
        mask = MASKS[self.mc.get("mask", "square")]
        tot = 0.0
        for k in self.names:
            lam = self.lam[k].detach() if self.frozen else self.lam[k]
            r = vecs[k].reshape(-1)
            tot = tot + self.gw[group_of(k)] * (mask(lam.to(r.dtype)) * r**2).mean()
        return tot

    def after_backward(self, step):
        if not self.frozen:
            self.opt.step()
        mask = MASKS[self.mc.get("mask", "square")]
        self.w = {k: float(mask(self.lam[k].detach()).mean()) for k in self.names}

    def _extra_state(self):
        if not self.names:
            return {}
        return {"lam": {k: v.detach().clone() for k, v in self.lam.items()}, "opt": self.opt.state_dict()}

    def _load_extra(self, s):
        if s:
            self.lam = {k: v.clone().requires_grad_(True) for k, v in s["lam"].items()}
            self.opt = torch.optim.Adam(list(self.lam.values()), lr=float(self.mc.get("lr", 5e-3)), maximize=True)
            self.opt.load_state_dict(s["opt"])


CLASSES = {c.method: c for c in (LRAnnealing, GradNorm, NTKTrace, SoftAdapt, ReLoBRaLo, Uncertainty, SelfAdaptive)}


def build_balancer(cfg: dict[str, Any] | None, group_weights: dict[str, float], seed: int) -> Balancer | None:
    """None for the fixed method (the Trainer then uses its original weighted sum)."""
    if not cfg or cfg.get("method", "fixed") == "fixed":
        return None
    m = cfg["method"]
    if m not in CLASSES:
        raise ValueError(f"unknown balancing method {m!r}; available: {list(METHODS)}")
    return CLASSES[m](cfg, group_weights, seed)
