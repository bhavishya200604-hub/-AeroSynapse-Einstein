r"""The 12-step optimisation validation protocol of spec Section 8G, enforced in code.

:class:`OptimizationStudy` refuses to run unless steps 1–6 are defined (baseline design, design space,
objective, constraints, budget, stopping rule). Running records every evaluation with feasibility
and uncertainty (7–9), runs a baseline optimiser (10), re-evaluates the final candidate with the
trusted solver (11) and issues a feasibility verdict (12). A surrogate is never the final authority.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aerosynapse.exceptions import ConfigurationError, ConstraintViolationError
from aerosynapse.optim.algorithms import OptResult, minimize
from aerosynapse.optim.problem import EvalResult, OptimizationProblem
from aerosynapse.optim.variables import DesignSpace


@dataclass
class OptimizationStudy:
    name: str
    baseline_design: dict[str, Any] | None = None  # step 1
    space: DesignSpace | None = None  # step 2
    objective: str = ""  # step 3 (mathematical statement)
    constraints: list[str] = field(default_factory=list)  # step 4 (statements; empty list must be explicit)
    constraints_declared: bool = False
    budget: int = 0  # step 5
    stopping: str = ""  # step 6
    model: Callable[[dict[str, Any]], EvalResult] | None = None  # model used by the optimiser (may be a surrogate)
    trusted: Callable[[dict[str, Any]], EvalResult] | None = None  # trusted numerical solver
    eq_tol: float = 1e-6

    def check(self) -> None:
        missing = [
            n
            for n, ok in (
                ("1 baseline design", self.baseline_design is not None),
                ("2 design space", self.space is not None),
                ("3 objective", bool(self.objective)),
                ("4 constraints", self.constraints_declared),
                ("5 budget", self.budget > 0),
                ("6 stopping criterion", bool(self.stopping)),
                ("model", self.model is not None),
                ("11 trusted solver", self.trusted is not None),
            )
            if not ok
        ]
        if missing:
            raise ConfigurationError(f"optimisation protocol incomplete, missing steps: {missing}")

    def run(self, method: str, baseline_method: str, seed: int = 0, **kw: Any) -> dict[str, Any]:
        self.check()
        assert self.space is not None and self.model is not None and self.trusted is not None
        assert self.baseline_design is not None
        out: dict[str, Any] = {
            "name": self.name,
            "objective": self.objective,
            "constraints": self.constraints,
            "budget": self.budget,
            "stopping": self.stopping,
        }
        z0 = self.space.encode(self.baseline_design)
        tb = self.trusted(self.baseline_design)
        out["baseline"] = {
            "design": self.baseline_design,
            "f": tb.f,
            "g": tb.g,
            "h": tb.h,
            "feasible": all(g <= 0 for g in tb.g),
        }
        results: dict[str, OptResult] = {}
        for tag, m in (("method", method), ("baseline_method", baseline_method)):
            prob = OptimizationProblem(
                self.name,
                self.space,
                self.model,
                eq_tol=self.eq_tol,
                budget=self.budget,
                source="surrogate" if self.model is not self.trusted else "trusted",
            )
            r = minimize(prob, m, z0=z0, seed=seed, **kw)
            results[tag] = r
            out[tag] = {
                "name": m,
                "n_evals": r.n_evals,
                "message": r.message,
                "trace": r.trace,
                "history": [
                    {
                        "i": e.index,
                        "design": e.design,
                        "f": e.f,
                        "g": e.g,
                        "feasible": e.feasible,
                        "sigma": e.sigma,
                        "source": e.source,
                    }
                    for e in prob.history
                ],
            }
            if r.best is None:
                out[tag]["final"] = None
                continue
            tv = self.trusted(r.best.design)  # step 11
            feas = all(g <= 0 for g in tv.g) and all(abs(h) <= self.eq_tol for h in tv.h)
            out[tag]["final"] = {
                "design": r.best.design,
                "model_f": r.best.f,
                "trusted_f": tv.f,
                "trusted_g": tv.g,
                "trusted_feasible": feas,  # step 12
                "model_vs_trusted_abs_diff": float(np.abs(np.array(r.best.f) - np.array(tv.f)).max()),
                "improvement_vs_baseline": float(tb.f[0] - tv.f[0]),
            }
        out["verdict"] = (
            "ACCEPT"
            if out["method"].get("final") and out["method"]["final"]["trusted_feasible"]
            else "REJECT: final candidate infeasible or missing under the trusted solver"
        )
        return out


def require_trusted_feasible(report: dict[str, Any]) -> None:
    if not report["verdict"].startswith("ACCEPT"):
        raise ConstraintViolationError(report["verdict"])
