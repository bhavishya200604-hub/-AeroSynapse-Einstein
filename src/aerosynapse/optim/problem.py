r"""Optimisation problem with provenance (spec Sections 8B, 8C):

.. math:: \min_x f(x)\ \text{s.t.}\ g_i(x)\le 0,\ h_j(x)=0,\ x_{min}\le x\le x_{max}

Every evaluation is recorded (design, objectives, constraints, feasibility, uncertainty, source,
wall time). The evaluation budget is enforced (``BudgetExhaustedError``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerosynapse.exceptions import BudgetExhaustedError
from aerosynapse.optim.variables import DesignSpace

FloatArr = NDArray[np.float64]


@dataclass
class EvalResult:
    """What a model returns for one design: objectives, inequality/equality constraints, uncertainty."""

    f: list[float]
    g: list[float] = field(default_factory=list)
    h: list[float] = field(default_factory=list)
    sigma: list[float] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evaluation:
    index: int
    z: list[float]
    design: dict[str, Any]
    f: list[float]
    g: list[float]
    h: list[float]
    feasible: bool
    violation: float
    sigma: list[float]
    source: str
    wall_s: float


class OptimizationProblem:
    def __init__(
        self,
        name: str,
        space: DesignSpace,
        model: Callable[[dict[str, Any]], EvalResult],
        n_obj: int = 1,
        objective_names: list[str] | None = None,
        constraint_names: list[str] | None = None,
        eq_tol: float = 1e-6,
        budget: int = 200,
        source: str = "trusted",
    ) -> None:
        self.name, self.space, self.model = name, space, model
        self.n_obj, self.eq_tol, self.budget, self.source = n_obj, eq_tol, budget, source
        self.objective_names = objective_names or [f"f{i}" for i in range(n_obj)]
        self.constraint_names = constraint_names or []
        self.history: list[Evaluation] = []

    def reset(self) -> None:
        self.history = []

    @property
    def n_evals(self) -> int:
        return len(self.history)

    def evaluate(self, z: FloatArr, source: str | None = None, count: bool = True) -> Evaluation:
        if count and self.n_evals >= self.budget:
            raise BudgetExhaustedError(f"{self.name}: evaluation budget {self.budget} exhausted")
        z = np.clip(np.asarray(z, float), 0, 1)
        d = self.space.decode(z)
        t0 = time.perf_counter()
        r = self.model(d)
        viol = max([0.0] + [max(0.0, gi) for gi in r.g] + [max(0.0, abs(hj) - self.eq_tol) for hj in r.h])
        e = Evaluation(
            len(self.history),
            z.tolist(),
            d,
            list(map(float, r.f)),
            list(map(float, r.g)),
            list(map(float, r.h)),
            viol == 0.0,
            float(viol),
            list(map(float, r.sigma)),
            source or self.source,
            time.perf_counter() - t0,
        )
        if count:
            self.history.append(e)
        return e

    def best_feasible(self) -> Evaluation | None:
        feas = [e for e in self.history if e.feasible]
        return min(feas, key=lambda e: e.f[0]) if feas else None

    def penalised(self, z: FloatArr, rho: float = 1e3) -> float:
        e = self.evaluate(z)
        return e.f[0] + rho * (sum(max(0.0, g) ** 2 for g in e.g) + sum(h**2 for h in e.h))
