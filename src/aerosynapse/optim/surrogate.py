r"""Surrogate-assisted optimisation with a trust region (spec Section 8, rule 11).

Loop: fit a GP to all *trusted* evaluations; minimise the lower confidence bound μ - κσ (penalised
by GP-predicted constraints) inside the trust region [z_c - r, z_c + r]; evaluate the proposal with
the trusted solver; ratio ρ = actual / predicted improvement grows (ρ > 0.75), keeps or shrinks
(ρ < 0.25) the radius. The returned best design is a trusted evaluation, never a surrogate value.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import qmc

from aerosynapse.exceptions import BudgetExhaustedError
from aerosynapse.optim.algorithms import OptResult, _best_so_far
from aerosynapse.optim.problem import Evaluation, OptimizationProblem
from aerosynapse.registry import OPTIMIZERS
from aerosynapse.uq.gp import GaussianProcess


def _merit(e: Evaluation, rho: float = 1e3) -> float:
    return e.f[0] + rho * sum(max(0.0, g) ** 2 for g in e.g)


@OPTIMIZERS.register("surrogate_tr")
def surrogate_trust_region(
    p: OptimizationProblem,
    z0: Any = None,
    seed: int = 0,
    r0: float = 0.25,
    kappa: float = 1.0,
    n_cand: int = 2000,
    **_: Any,
) -> OptResult:
    rng = np.random.default_rng(seed)
    d = p.space.dim
    log: list[dict[str, float]] = []
    try:
        for z in qmc.LatinHypercube(d=d, seed=seed).random(max(5, 2 * d + 1)):
            p.evaluate(z)
        r = r0
        while True:
            Z = np.array([e.z for e in p.history])
            M = np.array([_merit(e) for e in p.history])
            centre = Z[int(np.argmin(M))]
            gp = GaussianProcess(noise=1e-6, restarts=1, seed=seed).fit(Z, np.array([e.f[0] for e in p.history]))
            gps = [
                GaussianProcess(noise=1e-6, restarts=1, seed=seed).fit(Z, np.array([e.g[j] for e in p.history]))
                for j in range(len(p.history[0].g))
            ]
            C = np.clip(centre + (rng.random((n_cand, d)) * 2 - 1) * r, 0, 1)
            mu, sd = gp.predict(C)
            pen = np.zeros(n_cand)
            for g in gps:
                gm, _gs = g.predict(C)
                pen += 1e3 * np.maximum(gm, 0) ** 2
            score = mu - kappa * sd + pen
            k = int(np.argmin(score))
            mc, _cs = gp.predict(centre[None])
            pred_impr = float(mc[0] - mu[k])
            e = p.evaluate(C[k])
            act_impr = float(M.min() - _merit(e))
            rho = act_impr / pred_impr if pred_impr > 1e-12 else -1.0
            r = min(0.5, 2 * r) if rho > 0.75 else (max(1e-3, 0.5 * r) if rho < 0.25 else r)
            log.append({"radius": r, "rho": rho, "gp_sigma_at_proposal": float(sd[k])})
    except BudgetExhaustedError:
        pass
    res = OptResult("surrogate_tr", p.best_feasible(), p.n_evals, "budget exhausted", trace=_best_so_far(p))
    res.pareto = []
    res.trace_info = log  # type: ignore[attr-defined]
    return res
