r"""Optimisation algorithms behind one interface: ``minimize(problem, method, ...)``.

All methods work in the unit-cube code z ∈ [0, 1]^d and every model evaluation goes through
``problem.evaluate`` (recorded, budget-enforced). Constraint handling: penalty (GD/Adam/L-BFGS),
native (SLSQP), Deb's feasibility rules (GA/DE/NSGA-II), probability of feasibility (cBO).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize as sp_minimize
from scipy.stats import norm, qmc

from aerosynapse.exceptions import BudgetExhaustedError
from aerosynapse.optim.pareto import crowding, non_dominated_sort
from aerosynapse.optim.problem import Evaluation, OptimizationProblem
from aerosynapse.registry import OPTIMIZERS
from aerosynapse.uq.gp import GaussianProcess

FloatArr = NDArray[np.float64]


@dataclass
class OptResult:
    method: str
    best: Evaluation | None
    n_evals: int
    message: str
    pareto: list[Evaluation] = field(default_factory=list)
    trace: list[float] = field(default_factory=list)


def _best_so_far(p: OptimizationProblem) -> list[float]:
    out, b = [], np.inf
    for e in p.history:
        if e.feasible:
            b = min(b, e.f[0])
        out.append(b)
    return out


class _Cache:
    """Avoid re-evaluating the same z for objective and constraints (SLSQP calls both)."""

    def __init__(self, p: OptimizationProblem) -> None:
        self.p = p
        self.z: FloatArr | None = None
        self.e: Evaluation | None = None

    def __call__(self, z: FloatArr) -> Evaluation:
        if self.z is None or not np.array_equal(self.z, z):
            self.z, self.e = np.array(z), self.p.evaluate(z)
        assert self.e is not None
        return self.e


def _fd_grad(fun: Callable[[FloatArr], float], z: FloatArr, h: float = 1e-6) -> FloatArr:
    f0 = fun(z)
    g = np.zeros_like(z)
    for i in range(len(z)):
        zp = z.copy()
        step = h if z[i] + h <= 1 else -h
        zp[i] += step
        g[i] = (fun(zp) - f0) / step
    return g


def _deb_better(a: Evaluation, b: Evaluation) -> bool:
    if a.feasible and b.feasible:
        return a.f[0] < b.f[0]
    if a.feasible != b.feasible:
        return a.feasible
    return a.violation < b.violation


def _finish(p: OptimizationProblem, method: str, msg: str, **kw: Any) -> OptResult:
    return OptResult(method, p.best_feasible(), p.n_evals, msg, trace=_best_so_far(p), **kw)


@OPTIMIZERS.register("gd")
def gradient_descent(
    p: OptimizationProblem,
    z0: FloatArr,
    lr: float = 0.05,
    tol: float = 1e-8,
    rho: float = 1e3,
    adam: bool = False,
    **_: Any,
) -> OptResult:
    z = np.array(z0, float)
    m = v = np.zeros_like(z)
    fun = lambda w: p.penalised(w, rho)  # noqa: E731
    try:
        for k in range(1, 10**6):
            g = _fd_grad(fun, z)
            if np.linalg.norm(g) < tol:
                return _finish(p, "adam" if adam else "gd", "gradient tolerance reached")
            if adam:
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g**2
                step = lr * (m / (1 - 0.9**k)) / (np.sqrt(v / (1 - 0.999**k)) + 1e-12)
            else:
                step = lr * g
            z = np.clip(z - step, 0, 1)
    except BudgetExhaustedError:
        pass
    return _finish(p, "adam" if adam else "gd", "budget exhausted")


@OPTIMIZERS.register("adam")
def adam(p: OptimizationProblem, z0: FloatArr, lr: float = 0.02, **kw: Any) -> OptResult:
    return cast(OptResult, gradient_descent(p, z0, lr=lr, adam=True, **kw))


@OPTIMIZERS.register("lbfgs")
def lbfgs(p: OptimizationProblem, z0: FloatArr, rho: float = 1e3, **_: Any) -> OptResult:
    try:
        r = sp_minimize(
            lambda w: p.penalised(w, rho),
            z0,
            method="L-BFGS-B",
            bounds=[(0, 1)] * len(z0),
            options={"maxfun": p.budget - p.n_evals, "ftol": 1e-12, "gtol": 1e-9},
        )
        msg = str(r.message)
    except BudgetExhaustedError:
        msg = "budget exhausted"
    return _finish(p, "lbfgs", msg)


@OPTIMIZERS.register("sqp")
def sqp(p: OptimizationProblem, z0: FloatArr, **_: Any) -> OptResult:
    c = _Cache(p)
    cons = []
    ng, nh = len(c(z0).g), len(c(z0).h)
    if ng:
        cons.append({"type": "ineq", "fun": lambda w: -np.array(c(w).g)})
    if nh:
        cons.append({"type": "eq", "fun": lambda w: np.array(c(w).h)})
    try:
        r = sp_minimize(
            lambda w: c(w).f[0],
            z0,
            method="SLSQP",
            bounds=[(0, 1)] * len(z0),
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        msg = str(r.message)
    except BudgetExhaustedError:
        msg = "budget exhausted"
    return _finish(p, "sqp", msg)


@OPTIMIZERS.register("de")
def differential_evolution(
    p: OptimizationProblem,
    z0: FloatArr | None = None,
    seed: int = 0,
    pop: int | None = None,
    F: float = 0.7,
    CR: float = 0.9,
    stall: int = 30,
    **_: Any,
) -> OptResult:
    rng = np.random.default_rng(seed)
    d = p.space.dim
    n = pop or max(12, 8 * d)
    try:
        Z = qmc.LatinHypercube(d=d, seed=seed).random(n)
        if z0 is not None:
            Z[0] = z0
        E = [p.evaluate(z) for z in Z]
        best_hist, since = np.inf, 0
        while True:
            for i in range(n):
                a, b, cc = rng.choice([j for j in range(n) if j != i], 3, replace=False)
                mutant = np.clip(Z[a] + F * (Z[b] - Z[cc]), 0, 1)
                mask = rng.random(d) < CR
                mask[rng.integers(d)] = True
                trial = np.where(mask, mutant, Z[i])
                et = p.evaluate(trial)
                if _deb_better(et, E[i]) or et.f[0] == E[i].f[0]:
                    Z[i], E[i] = trial, et
            bf = p.best_feasible()
            cur = bf.f[0] if bf else np.inf
            since = since + 1 if cur >= best_hist - 1e-12 * max(1, abs(best_hist)) else 0
            best_hist = min(best_hist, cur)
            if since >= stall:
                return _finish(p, "de", f"stalled for {stall} generations")
    except BudgetExhaustedError:
        return _finish(p, "de", "budget exhausted")


def _sbx(a: FloatArr, b: FloatArr, rng: np.random.Generator, eta: float = 15.0) -> tuple[FloatArr, FloatArr]:
    u = rng.random(a.size)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * a + (1 - beta) * b)
    c2 = 0.5 * ((1 - beta) * a + (1 + beta) * b)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def _pm(z: FloatArr, rng: np.random.Generator, pm: float, eta: float = 20.0) -> FloatArr:
    u = rng.random(z.size)
    delta = np.where(u < 0.5, (2 * u) ** (1 / (eta + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta + 1)))
    return np.asarray(np.clip(np.where(rng.random(z.size) < pm, z + delta, z), 0, 1))


@OPTIMIZERS.register("ga")
def genetic_algorithm(
    p: OptimizationProblem, z0: FloatArr | None = None, seed: int = 0, pop: int | None = None, stall: int = 30, **_: Any
) -> OptResult:
    rng = np.random.default_rng(seed)
    d = p.space.dim
    n = pop or max(16, 8 * d) // 2 * 2
    try:
        Z = qmc.LatinHypercube(d=d, seed=seed).random(n)
        E = [p.evaluate(z) for z in Z]
        best_hist, since = np.inf, 0
        while True:

            def tour() -> int:
                i, j = rng.integers(n, size=2)
                return int(i if _deb_better(E[i], E[j]) else j)  # noqa: B023 (called within this iteration)

            kids = []
            for _k in range(n // 2):
                c1, c2 = _sbx(Z[tour()], Z[tour()], rng)
                kids += [_pm(c1, rng, 1 / d), _pm(c2, rng, 1 / d)]
            EK = [p.evaluate(k) for k in kids]
            allZ, allE = list(Z) + kids, E + EK
            order = sorted(range(len(allE)), key=lambda i: (not allE[i].feasible, allE[i].violation, allE[i].f[0]))
            Z = np.array([allZ[i] for i in order[:n]])
            E = [allE[i] for i in order[:n]]
            bf = p.best_feasible()
            cur = bf.f[0] if bf else np.inf
            since = since + 1 if cur >= best_hist - 1e-12 * max(1, abs(best_hist)) else 0
            best_hist = min(best_hist, cur)
            if since >= stall:
                return _finish(p, "ga", f"stalled for {stall} generations")
    except BudgetExhaustedError:
        return _finish(p, "ga", "budget exhausted")


def _cdom_key(E: list[Evaluation]) -> FloatArr:
    """Constraint-domination: infeasible points are pushed behind all feasible ones by violation."""
    F = np.array([e.f for e in E], float)
    v = np.array([e.violation for e in E])
    big = np.abs(F).max() + 1.0 if len(F) else 1.0
    return np.where(v[:, None] > 0, big * 10 + v[:, None], F)


@OPTIMIZERS.register("nsga2")
def nsga2(p: OptimizationProblem, z0: FloatArr | None = None, seed: int = 0, pop: int = 40, **_: Any) -> OptResult:
    rng = np.random.default_rng(seed)
    d = p.space.dim
    Z = qmc.LatinHypercube(d=d, seed=seed).random(pop)
    E: list[Evaluation] = []
    try:
        E = [p.evaluate(z) for z in Z]
        while True:
            K = _cdom_key(E)
            fronts = non_dominated_sort(K)
            rank = np.zeros(pop, int)
            cd = np.zeros(pop)
            for r, fr in enumerate(fronts):
                rank[fr] = r
                cd[fr] = crowding(K[fr])

            def tour() -> int:
                i, j = rng.integers(pop, size=2)
                return int(i if (rank[i], -cd[i]) < (rank[j], -cd[j]) else j)  # noqa: B023

            kids = []
            for _k in range(pop // 2):
                c1, c2 = _sbx(Z[tour()], Z[tour()], rng)
                kids += [_pm(c1, rng, 1 / d), _pm(c2, rng, 1 / d)]
            EK = [p.evaluate(k) for k in kids]
            allZ, allE = np.vstack([Z, kids]), E + EK
            K = _cdom_key(allE)
            sel: list[int] = []
            for fr in non_dominated_sort(K):
                if len(sel) + len(fr) <= pop:
                    sel += fr
                else:
                    c = crowding(K[fr])
                    sel += [fr[i] for i in np.argsort(-c)[: pop - len(sel)]]
                    break
            Z, E = allZ[sel], [allE[i] for i in sel]
    except BudgetExhaustedError:
        pass
    feas = [e for e in p.history if e.feasible]
    pf = [feas[i] for i in non_dominated_sort(np.array([e.f for e in feas]))[0]] if feas else []
    return OptResult("nsga2", None, p.n_evals, "budget exhausted", pareto=pf)


def _ei(mu: FloatArr, sd: FloatArr, best: float) -> FloatArr:
    z = (best - mu) / np.maximum(sd, 1e-12)
    return np.asarray((best - mu) * norm.cdf(z) + sd * norm.pdf(z))


def _initial_design(p: OptimizationProblem, seed: int, n0: int | None) -> list[Evaluation]:
    d = p.space.dim
    n0 = n0 or max(5, 2 * d + 1)
    return [p.evaluate(z) for z in qmc.LatinHypercube(d=d, seed=seed).random(n0)]


@OPTIMIZERS.register("bo")
def bayesian_optimization(
    p: OptimizationProblem,
    z0: FloatArr | None = None,
    seed: int = 0,
    n0: int | None = None,
    constrained: bool = False,
    n_cand: int = 2000,
    **_: Any,
) -> OptResult:
    """GP-EI (``constrained=False``) or EI × Π_i P(g_i ≤ 0) with one GP per constraint (cBO)."""
    rng = np.random.default_rng(seed)
    d = p.space.dim
    try:
        _initial_design(p, seed, n0)
        while True:
            Z = np.array([e.z for e in p.history])
            y = np.array([e.f[0] for e in p.history])
            gp = GaussianProcess(noise=1e-6, restarts=2, seed=seed).fit(Z, y)
            feas = [e.f[0] for e in p.history if e.feasible]
            best = min(feas) if feas else float(y.min())
            C = rng.random((n_cand, d))
            mu, sd = gp.predict(C)
            acq = _ei(mu, sd, best)
            if constrained and p.history[0].g:
                G = np.array([e.g for e in p.history])
                for j in range(G.shape[1]):
                    gj = GaussianProcess(noise=1e-6, restarts=1, seed=seed).fit(Z, G[:, j])
                    m, s = gj.predict(C)
                    acq = acq * norm.cdf(-m / np.maximum(s, 1e-12))
            p.evaluate(C[int(np.argmax(acq))])
    except BudgetExhaustedError:
        pass
    return _finish(p, "cbo" if constrained else "bo", "budget exhausted")


@OPTIMIZERS.register("cbo")
def constrained_bo(p: OptimizationProblem, z0: FloatArr | None = None, **kw: Any) -> OptResult:
    return cast(OptResult, bayesian_optimization(p, z0, constrained=True, **kw))


@OPTIMIZERS.register("ehvi")
def ehvi(
    p: OptimizationProblem,
    z0: FloatArr | None = None,
    seed: int = 0,
    ref: FloatArr | None = None,
    n_cand: int = 500,
    n_mc: int = 64,
    **_: Any,
) -> OptResult:
    """Monte-Carlo expected hypervolume improvement for 2 objectives (independent GPs)."""
    from aerosynapse.optim.pareto import hypervolume_2d

    rng = np.random.default_rng(seed)
    d = p.space.dim
    try:
        _initial_design(p, seed, None)
        while True:
            Z = np.array([e.z for e in p.history])
            F = np.array([e.f for e in p.history])
            r = ref if ref is not None else F.max(0) * 1.1 + 1e-9
            gps = [GaussianProcess(noise=1e-6, restarts=1, seed=seed).fit(Z, F[:, j]) for j in range(2)]
            C = rng.random((n_cand, d))
            ms = [g.predict(C) for g in gps]
            hv0 = hypervolume_2d(F, r)
            eps = rng.standard_normal((n_mc, 2))
            score = np.zeros(n_cand)
            for i in range(n_cand):
                samples = np.column_stack([ms[j][0][i] + ms[j][1][i] * eps[:, j] for j in range(2)])
                score[i] = np.mean([hypervolume_2d(np.vstack([F, s]), r) - hv0 for s in samples])
            p.evaluate(C[int(np.argmax(score))])
    except BudgetExhaustedError:
        pass
    feas = [e for e in p.history if e.feasible]
    pf = [feas[i] for i in non_dominated_sort(np.array([e.f for e in feas]))[0]] if feas else []
    return OptResult("ehvi", None, p.n_evals, "budget exhausted", pareto=pf)


def minimize(p: OptimizationProblem, method: str, z0: FloatArr | None = None, seed: int = 0, **kw: Any) -> OptResult:
    fn = OPTIMIZERS.get(method)
    z0 = np.full(p.space.dim, 0.5) if z0 is None else np.asarray(z0, float)
    return cast(OptResult, fn(p, z0=z0, seed=seed, **kw))


CAPABILITIES = {
    "gd": {
        "gradients": "yes (FD here)",
        "expensive_objective": "no",
        "constraints": "penalty",
        "noise": "poor",
        "multi_objective": "no",
        "complexity": "O(d) evals/iter",
        "failure_modes": "slow, step-size sensitive, local",
    },
    "adam": {
        "gradients": "yes",
        "expensive_objective": "no",
        "constraints": "penalty",
        "noise": "moderate",
        "multi_objective": "no",
        "complexity": "O(d) evals/iter",
        "failure_modes": "no convergence guarantee to KKT",
    },
    "lbfgs": {
        "gradients": "yes",
        "expensive_objective": "moderate",
        "constraints": "bounds + penalty",
        "noise": "poor",
        "multi_objective": "no",
        "complexity": "O(d) evals/iter + O(md)",
        "failure_modes": "local minima, penalty ill-conditioning",
    },
    "sqp": {
        "gradients": "yes",
        "expensive_objective": "moderate",
        "constraints": "native (KKT-based)",
        "noise": "poor",
        "multi_objective": "no",
        "complexity": "O(d) evals/iter + QP",
        "failure_modes": "infeasible start, non-smooth constraints, local",
    },
    "de": {
        "gradients": "no",
        "expensive_objective": "no",
        "constraints": "Deb rules",
        "noise": "moderate",
        "multi_objective": "no",
        "complexity": "pop × generations evals",
        "failure_modes": "many evaluations",
    },
    "ga": {
        "gradients": "no",
        "expensive_objective": "no",
        "constraints": "Deb rules",
        "noise": "moderate",
        "multi_objective": "no (see nsga2)",
        "complexity": "pop × generations",
        "failure_modes": "premature convergence",
    },
    "nsga2": {
        "gradients": "no",
        "expensive_objective": "no",
        "constraints": "constraint domination",
        "noise": "moderate",
        "multi_objective": "yes",
        "complexity": "O(M N^2) sort per generation",
        "failure_modes": "poor spread with >3 objectives",
    },
    "bo": {
        "gradients": "no",
        "expensive_objective": "yes",
        "constraints": "no (use cbo)",
        "noise": "yes (GP noise)",
        "multi_objective": "no",
        "complexity": "O(n^3) GP per iteration",
        "failure_modes": "high dimension (>~20)",
    },
    "cbo": {
        "gradients": "no",
        "expensive_objective": "yes",
        "constraints": "probability of feasibility",
        "noise": "yes",
        "multi_objective": "no",
        "complexity": "O((1+n_g) n^3)",
        "failure_modes": "tiny feasible set",
    },
    "ehvi": {
        "gradients": "no",
        "expensive_objective": "yes",
        "constraints": "filter after",
        "noise": "yes",
        "multi_objective": "yes (2 objectives)",
        "complexity": "MC × candidates × HV",
        "failure_modes": "cost",
    },
}
