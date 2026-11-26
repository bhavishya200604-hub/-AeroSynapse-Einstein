"""Phase 5 study: analytic benchmarks, engineering demos under the 8G protocol, multi-objective,
surrogate-assisted vs direct, robust optimisation, inverse problem (adjoint vs PINN)."""

from __future__ import annotations

from typing import Any

import numpy as np

from aerosynapse.optim import demos as D
from aerosynapse.optim.algorithms import minimize
from aerosynapse.optim.inverse import HeatInverseSetup, pinn_identify
from aerosynapse.optim.kkt import kkt_check
from aerosynapse.optim.pareto import hypervolume_2d, hypervolume_3d
from aerosynapse.optim.problem import EvalResult, OptimizationProblem
from aerosynapse.optim.protocol import OptimizationStudy
from aerosynapse.optim.robust import chance, cvar, expectation


def _branin_global_min() -> float:
    """Global minimum by dense grid + local polish (computed, not recalled)."""
    from scipy.optimize import minimize as spm

    s = D.branin_space()
    g = np.linspace(0, 1, 401)
    Z = np.array([[a, b] for a in g for b in g])
    vals = np.array([D.branin(s.decode(z)).f[0] for z in Z])
    best = np.inf
    for z in Z[np.argsort(vals)[:10]]:
        r = spm(
            lambda w: D.branin(s.decode(np.clip(w, 0, 1))).f[0],
            z,
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12},
        )
        best = min(best, r.fun)
    return float(best)


def benchmarks(seeds: int) -> dict[str, Any]:
    out: dict[str, Any] = {"rosenbrock": {}, "qp": {}, "branin": {}, "zdt1": {}}
    for m, budget in (("gd", 3000), ("adam", 3000), ("lbfgs", 400), ("sqp", 400), ("de", 3000), ("ga", 3000)):
        vals = []
        for s in range(seeds):
            p = OptimizationProblem("rosenbrock", D.rosenbrock_space(), D.rosenbrock, budget=budget)
            r = minimize(p, m, z0=np.array([0.2, 0.2]), seed=s)
            vals.append(r.best.f[0] if r.best else np.inf)
        out["rosenbrock"][m] = {"budget": budget, "best_f": vals, "median": float(np.median(vals)), "known_min": 0.0}
    for m, budget in (("sqp", 400), ("de", 2000), ("ga", 2000), ("cbo", 40), ("lbfgs", 400)):
        p = OptimizationProblem("qp", D.qp_space(), D.qp, budget=budget)
        r = minimize(p, m, seed=0)
        b = r.best
        rep = None
        if b is not None:
            x = np.array([b.design["x1"], b.design["x2"]])
            rep = kkt_check(lambda x: float(x @ x), x, g=lambda x: np.array([1 - x[0] - x[1]]), convex=True)
        out["qp"][m] = {
            "budget": budget,
            "best_f": b.f[0] if b else None,
            "known_min": 0.5,
            "design": b.design if b else None,
            "kkt_stationarity": rep.stationarity if rep else None,
            "kkt_ok": rep.kkt_ok if rep else None,
            "mu": rep.mu if rep else None,
            "known_mu": 1.0,
        }
    gmin = _branin_global_min()
    for m in ("bo", "de", "ga"):
        vals = []
        for s in range(seeds):
            p = OptimizationProblem("branin", D.branin_space(), D.branin, budget=40)
            r = minimize(p, m, seed=s, pop=12) if m != "bo" else minimize(p, m, seed=s)
            vals.append(r.best.f[0])
        out["branin"][m] = {"budget": 40, "best_f": vals, "median_gap": float(np.median(vals) - gmin)}
    out["branin"]["global_min_computed"] = gmin
    true_hv = D.zdt1_true_hv()
    for m, budget in (("nsga2", 4000), ("ehvi", 40), ("nsga2_small", 40)):
        p = OptimizationProblem("zdt1", D.zdt1_space(6 if m == "nsga2" else 3), D.zdt1, n_obj=2, budget=budget)
        r = minimize(
            p, "nsga2" if m.startswith("nsga2") else "ehvi", seed=0, pop=40 if m == "nsga2" else 10, n_cand=200, n_mc=16
        )
        F = np.array([e.f for e in r.pareto])
        out["zdt1"][m] = {
            "budget": budget,
            "hv": hypervolume_2d(F, np.array([1.1, 1.1])),
            "true_hv": true_hv,
            "n_pareto": len(F),
        }
    return out


def protocol_runs() -> dict[str, Any]:
    out = {}
    st = OptimizationStudy(
        "thermal_rod",
        baseline_design={"m": 1.0, "x_s": 0.8, "material": "aluminium"},
        space=D.thermal_space(),
        objective="min theta_max/theta_ref + 0.1 (m/4)^2 (dimensionless)",
        constraints=["theta(1)/0.02 - 1 <= 0 (tip temperature)"],
        constraints_declared=True,
        budget=600,
        stopping="budget or 30 stalled generations",
        model=D.thermal_model,
        trusted=lambda d: D.thermal_model(d, n=1600),
    )
    out["thermal"] = st.run("de", "ga", seed=0)
    st = OptimizationStudy(
        "source_placement",
        baseline_design={"s": 0.5, "q": 1.0},
        space=D.source_space(),
        objective="min max(u)/c_ref",
        constraints=["1 - u(0.7)/c_min <= 0"],
        constraints_declared=True,
        budget=300,
        stopping="budget or SLSQP ftol 1e-12",
        model=D.source_model,
        trusted=lambda d: D.source_model(d, n=1600),
    )
    out["source"] = st.run("sqp", "de", seed=0)
    st = OptimizationStudy(
        "cantilever",
        baseline_design={"b": 0.05, "t": 0.15, "material": "steel"},
        space=D.beam_space(),
        objective="min mass [kg]",
        constraints=["1.5 sigma/sigma_y - 1 <= 0", "delta/5mm - 1 <= 0"],
        constraints_declared=True,
        budget=1500,
        stopping="budget or 30 stalled generations",
        model=D.beam_model,
        trusted=D.beam_model,
    )
    out["beam"] = st.run("de", "ga", seed=0)
    return out


def multiobjective() -> dict[str, Any]:
    p = OptimizationProblem("beam3", D.beam_space(), D.beam_three_objectives, n_obj=3, budget=3000)
    r = minimize(p, "nsga2", seed=0, pop=60)
    F = np.array([e.f for e in r.pareto])
    allpts = [{"f": e.f, "feasible": e.feasible, "design": e.design} for e in p.history]
    ref = np.array([100.0, 50.0, 500.0])
    t = OptimizationProblem("thermal2", D.thermal_space(False), D.thermal_two_objectives, n_obj=2, budget=40)
    re = minimize(t, "ehvi", seed=0, n_cand=200, n_mc=16)
    t2 = OptimizationProblem("thermal2", D.thermal_space(False), D.thermal_two_objectives, n_obj=2, budget=40)
    rn = minimize(t2, "nsga2", seed=0, pop=10)
    refT = np.array([30.0, 1.1])
    return {
        "beam_pareto_size": len(F),
        "beam_hv": hypervolume_3d(F, ref),
        "beam_ref": ref.tolist(),
        "beam_points": allpts,
        "beam_pareto": [e.f for e in r.pareto],
        "thermal2_hv_ehvi_40": hypervolume_2d(np.array([e.f for e in re.pareto]), refT),
        "thermal2_hv_nsga2_40": hypervolume_2d(np.array([e.f for e in rn.pareto]), refT),
    }


def surrogate_vs_direct(seeds: int) -> dict[str, Any]:
    res: dict[str, list[float]] = {"surrogate_tr": [], "de": [], "cbo": []}
    trusted_fine = lambda d: D.thermal_model(d, n=1600)  # noqa: E731
    verified = []
    for s in range(seeds):
        for m in res:
            p = OptimizationProblem("thermal", D.thermal_space(False), D.thermal_model, budget=40)
            r = minimize(p, m, seed=s, n_cand=500, pop=8) if m != "de" else minimize(p, m, seed=s, pop=8)
            if r.best is None:
                res[m].append(np.inf)
                continue
            tv = trusted_fine(r.best.design)
            res[m].append(tv.f[0])
            verified.append(
                {"method": m, "seed": s, "feasible_fine": tv.g[0] <= 0, "f_fine": tv.f[0], "f_model": r.best.f[0]}
            )
    return {
        "budget_trusted_evals": 40,
        "f_reverified": res,
        "median": {k: float(np.median(v)) for k, v in res.items()},
        "reverification": verified,
    }


def robust() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    xi = np.exp(rng.normal(np.log(1.0), 0.5, 256))  # uncertain tip Biot number, lognormal (assumption)

    def stats(d: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        f, g = [], []
        for b in xi:
            r = D.thermal_model(d, bi=float(b), n=200)
            f.append(r.f[0])
            g.append(r.g[0])
        return np.array(f), np.array(g)

    def robust_model(d: dict[str, Any]) -> EvalResult:
        f, g = stats(d)
        ch = chance(g, eps=0.05)
        return EvalResult(f=[cvar(f, 0.9)], g=[0.95 - ch["p_feasible"]])

    space = D.thermal_space(False)
    det = OptimizationProblem("det", space, lambda d: D.thermal_model(d, n=200), budget=400)
    rd = minimize(det, "de", seed=0, pop=10)
    rob = OptimizationProblem("rob", space, robust_model, budget=200)
    rr = minimize(rob, "de", seed=0, pop=10)
    out = {}
    for tag, r in (("deterministic_opt", rd), ("robust_opt", rr)):
        assert r.best is not None
        f, g = stats(r.best.design)
        m, se = expectation(f)
        out[tag] = {
            "design": r.best.design,
            "E_f": m,
            "se": se,
            "CVaR90_f": cvar(f, 0.9),
            "chance_tip": chance(g, eps=0.05),
            "worst_f": float(f.max()),
        }
    out["xi_model"] = "Bi ~ LogNormal(log 1, 0.5), 256 common random samples"
    return out


def inverse(pinn_iters: int) -> dict[str, Any]:
    s = HeatInverseSetup(n=64, nt=100)
    rng = np.random.default_rng(0)
    clean = s.observe(s.forward(0.1))
    out: dict[str, Any] = {"alpha_true": 0.1, "noise_sigma": 0.01, "n_obs": int(clean.size)}
    d = clean + rng.normal(0, 0.01, clean.shape)
    out["adjoint_lsq"] = s.identify(d, noise_sigma=0.01)
    reps = [
        s.identify(clean + np.random.default_rng(k).normal(0, 0.01, clean.shape), noise_sigma=0.01)["alpha_hat"]
        for k in range(1, 21)
    ]
    out["adjoint_lsq_replicates"] = {"mean": float(np.mean(reps)), "std": float(np.std(reps, ddof=1)), "n": 20}
    out["pinn"] = pinn_identify(s, d, iterations=pinn_iters)
    return out
