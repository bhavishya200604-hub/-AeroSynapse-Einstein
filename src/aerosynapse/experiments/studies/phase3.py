"""Phase 3: adaptive collocation for transient conv-diff across Péclet numbers (equal budget)."""

from __future__ import annotations

from typing import Any

from aerosynapse.ml.nets import MLP, set_seed
from aerosynapse.ml.pinn import PINNConfig, predict, train_pinn
from aerosynapse.ml.problems import ConvDiff1D
from aerosynapse.validation.metrics import error_distribution, field_metrics

PROFILES = {
    "quick": {"iters": 4000, "n": 2000, "seeds": 8, "pes": [1.0, 10.0, 100.0, 1000.0], "resample_every": 250},
    "standard": {
        "iters": 10000,
        "n": 4000,
        "seeds": 10,
        "pes": [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
        "resample_every": 500,
    },
    "full": {
        "iters": 20000,
        "n": 8000,
        "seeds": 20,
        "pes": [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0],
        "resample_every": 500,
    },
}
METHODS = ["uniform", "uniform_resample", "lhs", "sobol", "rar", "rad", "r3"]
SIGMA0, T_FINAL = 0.1, 1.0


def job(j: dict[str, Any]) -> dict[str, Any]:
    """One PINN run. Same seed => same network initialisation across methods (paired design)."""
    set_seed(j["seed"])
    p = ConvDiff1D(pe=j["pe"], sigma0=SIGMA0, t_final=T_FINAL, hard=j.get("hard", False))
    m = MLP(
        2,
        1,
        32,
        3,
        lo=p.lo,
        hi=p.hi,
        periodic_dims=(0,),
        seed=j["seed"],
        fourier_features=j.get("ff", 0),
        fourier_scale=j.get("ff_scale", 1.0),
    )
    cfg = PINNConfig(
        iterations=j["iters"],
        n_collocation=j["n"],
        sampler=j["method"],
        lr_decay=0.9,
        resample_every=j["resample_every"],
        log_every=max(1, j["iters"] // 10),
        causal_eps=j.get("causal_eps"),
    )
    r = train_pinn(p, m, cfg, seed=j["seed"])
    G = p.eval_grid(201)
    pred, ref = predict(p, m, G), p.exact(G)
    early = G[:, 1] <= 0.05
    late = G[:, 1] >= 0.9
    return {
        **j,
        **field_metrics(pred, ref),
        **error_distribution(pred, ref),
        "rel_l2_early_t": float(((pred[early] - ref[early]) ** 2).sum() ** 0.5 / (ref[early] ** 2).sum() ** 0.5),
        "rel_l2_late_t": float(((pred[late] - ref[late]) ** 2).sum() ** 0.5 / (ref[late] ** 2).sum() ** 0.5),
        "wall_s": r.wall_s,
        "residual_evals": r.residual_evals,
        "final_loss": r.history[-1]["loss"],
        "final_L_r": r.history[-1]["L_r"],
        "history": r.history,
    }
