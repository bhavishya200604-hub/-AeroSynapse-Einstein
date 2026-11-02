r"""Problem definition API: :math:`\mathcal N(u;\mu)=f`, :math:`\mathcal B(u;\mu)=g`, :math:`u(x,0)=u_0`.

A :class:`PDEProblem` stores the domain, the named parameters :math:`\mu` (value, unit, admissible
range), the boundary conditions per side, the initial condition, a source term and an optional
analytical solution. Solvers and PINNs consume the same object.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerosynapse.exceptions import ConfigurationError
from aerosynapse.physics.bc import BoundaryCondition

FloatArr = NDArray[np.float64]


@dataclass(frozen=True)
class Parameter:
    """A physical parameter with unit and admissible range (validated on construction)."""

    name: str
    value: float
    unit: str = "1"
    low: float = -np.inf
    high: float = np.inf
    description: str = ""

    def __post_init__(self) -> None:
        if not (self.low <= self.value <= self.high):
            raise ConfigurationError(
                f"parameter {self.name}={self.value} outside admissible range [{self.low}, {self.high}]"
            )


@dataclass
class PDEProblem:
    """Container for a PDE problem in non-dimensional or dimensional form."""

    name: str
    equation: str
    domain: tuple[tuple[float, float], ...]
    t_final: float | None = None
    parameters: dict[str, Parameter] = field(default_factory=dict)
    bcs: dict[str, BoundaryCondition] = field(default_factory=dict)
    initial_condition: Callable[..., FloatArr] | None = None
    source: Callable[..., FloatArr] | None = None
    exact: Callable[..., FloatArr] | None = None
    coordinates: str = "cartesian"
    constitutive: str = ""
    constraints: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for lo, hi in self.domain:
            if not hi > lo:
                raise ConfigurationError(f"degenerate domain interval [{lo}, {hi}]")
        if self.t_final is not None and self.t_final <= 0:
            raise ConfigurationError("t_final must be positive")
        sides = {"left", "right"} if self.dim == 1 else {"left", "right", "bottom", "top"}
        unknown = set(self.bcs) - sides
        if unknown:
            raise ConfigurationError(f"unknown boundary sides {sorted(unknown)}; valid: {sorted(sides)}")
        periodic = [s for s, b in self.bcs.items() if b.kind == "periodic"]
        for a, b in (("left", "right"), ("bottom", "top")):
            if (a in periodic) != (b in periodic):
                raise ConfigurationError(f"periodic BC must be applied to both {a} and {b}")

    @property
    def dim(self) -> int:
        return len(self.domain)

    def p(self, name: str) -> float:
        """Parameter value by name."""
        try:
            return self.parameters[name].value
        except KeyError:
            raise ConfigurationError(f"problem {self.name} has no parameter {name!r}") from None

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "equation": self.equation,
            "domain": self.domain,
            "t_final": self.t_final,
            "parameters": {k: (v.value, v.unit) for k, v in self.parameters.items()},
            "bcs": {k: v.kind for k, v in self.bcs.items()},
            "dimensionless": dimensionless_numbers(self),
        }


def dimensionless_numbers(problem: PDEProblem) -> dict[str, float]:
    r"""Dimensionless groups available from the problem parameters.

    * Péclet :math:`Pe = vL/D`
    * Reynolds :math:`Re = UL/\nu`
    * Fourier :math:`Fo = \alpha t_f / L^2`
    * Biot :math:`Bi = hL/k`
    """
    L = problem.domain[0][1] - problem.domain[0][0]
    prm = {k: v.value for k, v in problem.parameters.items()}
    out: dict[str, float] = {}
    D = prm.get("D", prm.get("alpha"))
    if "v" in prm and D:
        out["Pe"] = abs(prm["v"]) * L / D
    if "U" in prm and prm.get("nu"):
        out["Re"] = abs(prm["U"]) * L / prm["nu"]
    elif "nu" in prm and "u_ref" in prm:
        out["Re"] = abs(prm["u_ref"]) * L / prm["nu"]
    if D and problem.t_final:
        out["Fo"] = D * problem.t_final / L**2
    if "h" in prm and "k" in prm:
        out["Bi"] = prm["h"] * L / prm["k"]
    return out


def cfl_number(v: float, dt: float, dx: float) -> float:
    """Courant number :math:`C = |v|\\Delta t/\\Delta x`."""
    return abs(v) * dt / dx


def diffusion_number(D: float, dt: float, dx: float) -> float:
    """Diffusion number :math:`r = D\\Delta t/\\Delta x^2`."""
    return D * dt / dx**2


def cell_peclet(v: float, D: float, dx: float) -> float:
    """Cell Péclet number :math:`Pe_h = |v|\\Delta x/D` (central scheme oscillates for > 2)."""
    return abs(v) * dx / D
