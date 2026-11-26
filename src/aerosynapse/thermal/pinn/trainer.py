"""Trainer: Adam (stateless LR schedule) followed by L-BFGS (strong-Wolfe), with per-component loss logging,
checkpointing and resume.

Determinism: the collocation batch is a pure function of (seed, epoch // resample_every), the LR is a pure function of
the epoch, so a resumed run reproduces an uninterrupted run exactly (tested). L-BFGS uses the batch of the last Adam
epoch.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from aerosynapse.thermal.pinn import strategies as ST
from aerosynapse.thermal.pinn.balancing import build_balancer, group_of
from aerosynapse.thermal.pinn.problems import Problem, component_losses, residual_vectors, weighted_total


def lr_at(adam: dict, epoch: int) -> float:
    e, n = epoch, max(adam["epochs"], 1)
    lr0, lr1 = adam["lr"], adam["lr_min"]
    if adam["schedule"] == "constant":
        return lr0
    if adam["schedule"] == "cosine":
        return lr1 + 0.5 * (lr0 - lr1) * (1.0 + math.cos(math.pi * e / n))
    return lr0 * (max(lr1, 1e-300) / lr0) ** (e / n)  # exp: geometric decay lr0 -> lr_min


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        problem: Problem,
        spec: dict,
        out_dir: str | Path,
        eval_fn: Callable[[torch.nn.Module], dict] | None = None,
        weights: dict[str, float] | None = None,
        tag: str = "run",
        data_terms: list | None = None,
    ):
        self.model, self.problem, self.spec = model, problem, spec
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.eval_fn = eval_fn
        self.weights = dict(weights or spec["loss"]["weights"])
        self.dev = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.history: list[dict[str, Any]] = []
        self.tag = tag
        self.t_wall = 0.0
        # Step 4.3: adaptive loss weights (None = fixed weights, original code path)
        self.balancer = build_balancer(spec["loss"].get("balancing"), self.weights, spec["seed"])
        # user data used as training data (aerosynapse.thermal.pinn.userdata.DataTerm); none by default
        self.data_terms = list(data_terms or [])
        # Step 4.4: residual-based adaptive sampling of the PDE points and causal temporal weighting (off by default)
        ad = spec["sampling"].get("adaptive") or {"method": "none"}
        self.sampler = (
            ST.AdaptiveSampler(ad, spec["sampling"]["n_pde"], spec["training"]["adam"]["epochs"], spec["seed"])
            if ad.get("method", "none") != "none"
            else None
        )
        cz = spec["training"].get("causal") or {"enabled": False}
        self.causal = None
        if cz.get("enabled") and getattr(problem, "has_time", False):
            ti = problem.input_names.index("t")
            self.causal = ST.CausalWeights(cz, problem.lo[ti], problem.hi[ti])

    # --- batches --------------------------------------------------------------------------------
    def batch(self, epoch: int) -> dict:
        s = self.spec["sampling"]
        k = epoch // s["resample_every"] if s["resample_every"] > 0 else 0
        seed = self.spec["seed"] * 1_000_003 + 11 * k
        out = self.problem.sample(s["n_pde"], s["n_ic"], s["n_bc"], s["method"], seed, self.dev, self.dtype)
        if self.sampler is not None:  # adaptive PDE points replace the fixed set (IC/BC points unchanged)
            out["pde"] = self.sampler.initial(out["pde"]) if self.sampler.points is None else self.sampler.points
        return out

    def fixed_losses(self, batch: dict, include_data: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fixed-weight total (loss.weights only): the original objective, independent of any balancing method.
        ``include_data=False`` gives the physics-only loss (truth-free validation)."""
        if hasattr(self.problem, "loss_components"):  # 2D problems (aerosynapse.thermal.pinn.problems2d) define their own losses
            comp = self.problem.loss_components(self.model, batch)
            total = self.problem.weighted_total(comp, self.weights)
        else:
            comp = component_losses(self.model, self.problem, batch)
            total = weighted_total(comp, self.weights)
        if include_data and self.data_terms:
            for term in self.data_terms:
                for k, v in term.vectors(self.model).items():
                    comp[k] = (v**2).mean()
                    total = total + self.weights.get("data", 1.0) * comp[k]
        return total, comp

    def losses(self, batch: dict, step: int | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Training objective. With a balancer, ``step`` (Adam epoch) lets it update its weights first."""
        if self.balancer is None and self.causal is None:
            return self.fixed_losses(batch)
        if hasattr(self.problem, "residual_vectors"):
            vecs = self.problem.residual_vectors(self.model, batch)
        else:
            vecs = residual_vectors(self.model, self.problem, batch)
        for term in self.data_terms:
            vecs.update(term.vectors(self.model))
        if self.causal is not None:  # mean(scaled^2) = (1/N_t) sum_i w_i L_r(t_i)
            vecs["pde"] = self.causal.transform(vecs["pde"], batch["pde"]["t"], step)
        comp = {k: (v**2).mean() for k, v in vecs.items()}
        if self.balancer is None:
            return sum(self.weights.get(group_of(k), 1.0) * v for k, v in comp.items()), comp
        if step is not None:
            self.balancer.prepare(self.model, comp, vecs, step)
        return self.balancer.total(comp, vecs), comp

    # --- logging ---------------------------------------------------------------------------------
    def _log(self, phase: str, step: int, total: torch.Tensor, comp: dict, lr: float, evaluate: bool) -> None:
        row: dict[str, Any] = {"phase": phase, "step": step, "total": float(total.detach()), "lr": lr}
        row.update({k: float(v.detach()) for k, v in comp.items()})
        if self.balancer is not None:
            row.update({f"w_{k}": v for k, v in self.balancer.effective().items()})
        if self.causal is not None:
            row.update({"causal_eps": self.causal.eps, "causal_min_w": self.causal.last_min_w})
        if self.sampler is not None and self.sampler.points is not None:
            row["n_pde"] = ST._size(self.sampler.points)
        row["wall_s"] = self.t_wall + (time.perf_counter() - self._t0)
        if evaluate and self.eval_fn is not None:
            row.update({k: v for k, v in self.eval_fn(self.model).items() if not k.startswith("_")})
        self.history.append(row)

    def write_log(self) -> None:
        if not self.history:
            return
        keys: list[str] = []
        for r in self.history:
            keys += [k for k in r if k not in keys]
        with open(self.out / "log.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)

    # --- checkpoint -------------------------------------------------------------------------------
    def save_checkpoint(self, phase: str, step: int, adam: torch.optim.Optimizer | None, lbfgs: Any | None) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "adam": adam.state_dict() if adam else None,
                "lbfgs": lbfgs.state_dict() if lbfgs else None,
                "phase": phase,
                "step": step,
                "history": self.history,
                "wall": self.t_wall + (time.perf_counter() - self._t0),
                "spec_tag": self.tag,
                "balancer": self.balancer.state_dict() if self.balancer is not None else None,
                "sampler": self.sampler.state_dict() if self.sampler is not None else None,
                "causal": self.causal.state_dict() if self.causal is not None else None,
            },
            self.out / "checkpoint.pt",
        )
        self.write_log()

    # --- run ---------------------------------------------------------------------------------------
    def run(
        self, resume: bool = False, stop_after_adam_epoch: int | None = None, stop_after_lbfgs_step: int | None = None
    ) -> dict[str, Any]:
        """Train; ``stop_after_*`` write a checkpoint and return early (used to test/continue interrupted runs)."""
        t = self.spec["training"]
        adam_cfg, lb_cfg = t["adam"], t["lbfgs"]
        adam = torch.optim.Adam(self.model.parameters(), lr=adam_cfg["lr"])
        lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=lb_cfg["lr"],
            max_iter=lb_cfg["inner"],
            history_size=lb_cfg["history"],
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-14,
            tolerance_change=1e-16,
        )
        phase, start = "adam", 0
        ck = self.out / "checkpoint.pt"
        if resume and ck.exists():
            s = torch.load(ck, map_location=self.dev, weights_only=False)
            self.model.load_state_dict(s["model"])
            if s["adam"]:
                adam.load_state_dict(s["adam"])
            if s["lbfgs"]:
                lbfgs.load_state_dict(s["lbfgs"])
            phase, start, self.history, self.t_wall = s["phase"], s["step"], s["history"], s["wall"]
            if self.balancer is not None and s.get("balancer"):
                self.balancer.load_state_dict(s["balancer"])
            if self.sampler is not None and s.get("sampler"):
                self.sampler.load_state_dict(s["sampler"])
            if self.causal is not None and s.get("causal"):
                self.causal.load_state_dict(s["causal"])
        self._t0 = time.perf_counter()
        status = "ok"
        n_ad, n_lb = adam_cfg["epochs"], -(-lb_cfg["iterations"] // lb_cfg["inner"]) if lb_cfg["iterations"] else 0

        if phase == "adam":
            batch = self.batch(start)
            for ep in range(start, n_ad):
                if stop_after_adam_epoch is not None and ep >= stop_after_adam_epoch:
                    self.save_checkpoint("adam", ep, adam, None)
                    return {"status": "stopped", "epoch": ep}
                if self.spec["sampling"]["resample_every"] > 0 and ep % self.spec["sampling"]["resample_every"] == 0:
                    batch = self.batch(ep)
                if self.sampler is not None and ep > 0 and ep % self.sampler.period == 0:
                    self.sampler.update(
                        self.model, self.problem, ep, self.dev, self.dtype, self.spec["sampling"]["method"]
                    )
                    batch = self.batch(ep)
                lr = lr_at(adam_cfg, ep)
                for g in adam.param_groups:
                    g["lr"] = lr
                adam.zero_grad(set_to_none=True)
                total, comp = self.losses(batch, step=ep)
                if not torch.isfinite(total):
                    status = "diverged"
                    break
                total.backward()
                if adam_cfg["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), adam_cfg["grad_clip"])
                adam.step()
                if self.balancer is not None:
                    self.balancer.after_backward(ep)
                last = ep == n_ad - 1
                if ep % t["log_every"] == 0 or last:
                    self._log("adam", ep, total, comp, lr, evaluate=(ep % t["eval_every"] == 0 or last))
                if (ep + 1) % t["checkpoint_every"] == 0 and not last:
                    self.save_checkpoint("adam", ep + 1, adam, None)
            phase, start = "lbfgs", 0
            if status == "ok":
                self.save_checkpoint("lbfgs", 0, adam, None)

        if self.balancer is not None:
            self.balancer.freeze()  # L-BFGS needs a fixed objective: adaptive weights frozen at their last Adam value
        if self.causal is not None:
            self.causal.freeze()  # same for the causal weights
        if phase == "lbfgs" and status == "ok" and n_lb > 0:
            batch = self.batch(max(n_ad - 1, 0))  # fixed batch (that of the last Adam epoch)
            for k in range(start, n_lb):
                if stop_after_lbfgs_step is not None and k >= stop_after_lbfgs_step:
                    self.save_checkpoint("lbfgs", k, adam, lbfgs)
                    return {"status": "stopped", "step": k}

                def closure() -> torch.Tensor:
                    lbfgs.zero_grad(set_to_none=True)
                    tot, _ = self.losses(batch)
                    tot.backward()
                    return tot

                lbfgs.step(closure)
                total, comp = self.losses(batch)
                if not torch.isfinite(total):
                    status = "diverged"
                    break
                last = k == n_lb - 1
                self._log("lbfgs", k, total, comp, lb_cfg["lr"], evaluate=(k % 5 == 0 or last))
                if (k + 1) % max(t["checkpoint_every"] // lb_cfg["inner"], 1) == 0 and not last:
                    self.save_checkpoint("lbfgs", k + 1, adam, lbfgs)
        self.t_wall += time.perf_counter() - self._t0
        self._t0 = time.perf_counter()
        self.save_checkpoint("done", 0, adam, lbfgs)
        final = self.history[-1] if self.history else {}
        return {"status": status, "final": final, "wall_s": self.t_wall, "n_log_rows": len(self.history)}


def component_grad_norms(model: torch.nn.Module, problem: Problem, batch: dict) -> dict[str, float]:
    """||grad_theta L_k||_2 of each (unweighted) loss component: a direct measure of loss-term imbalance."""
    if hasattr(problem, "loss_components"):
        comp = problem.loss_components(model, batch)
    else:
        comp = component_losses(model, problem, batch)
    out = {}
    params = [p for p in model.parameters() if p.requires_grad]
    for k, v in comp.items():
        g = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
        out[k] = float(torch.sqrt(sum((gi**2).sum() for gi in g if gi is not None)))
    return out
