r"""Boundary-condition framework (spec Section 6).

All conditions are written for an outward normal :math:`n` on a boundary side:

* Dirichlet  :math:`u = g`
* Neumann    :math:`\partial u/\partial n = g`
* Robin      :math:`a u + b\,\partial u/\partial n = g` (``b != 0``; ``b = 0`` is Dirichlet)
* Periodic   :math:`u(x_{min}) = u(x_{max})` (applies to a pair of opposite sides)
* Interface  continuity :math:`[u]=0` and flux :math:`[k\,\partial u/\partial n]=0`

``g`` may be a constant or a callable ``g(t)`` (1D) / ``g(s, t)`` (2D, ``s`` = boundary coordinate).
The ghost-node formulae used by the FD/FV solvers are implemented in ``ghost_value``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Union

import numpy as np
from numpy.typing import NDArray

from aerosynapse.exceptions import ConfigurationError

BCValue = Union[float, Callable[..., "float | NDArray[np.float64]"]]
FloatArr = NDArray[np.float64]


def _eval(g: BCValue, *args: float | FloatArr) -> float | FloatArr:
    return g(*args) if callable(g) else float(g)


@dataclass(frozen=True)
class BoundaryCondition:
    """Base class. ``kind`` is used by solvers to pick the discrete treatment."""

    kind: ClassVar[str] = "base"

    def value(self, *args: float | FloatArr) -> float | FloatArr:
        raise NotImplementedError("abstract boundary condition; use a concrete subclass")


@dataclass(frozen=True)
class Dirichlet(BoundaryCondition):
    g: BCValue = 0.0
    kind: ClassVar[str] = "dirichlet"

    def value(self, *args: float | FloatArr) -> float | FloatArr:
        return _eval(self.g, *args)


@dataclass(frozen=True)
class Neumann(BoundaryCondition):
    """Outward normal derivative prescribed."""

    g: BCValue = 0.0
    kind: ClassVar[str] = "neumann"

    def value(self, *args: float | FloatArr) -> float | FloatArr:
        return _eval(self.g, *args)


@dataclass(frozen=True)
class Robin(BoundaryCondition):
    a: float = 1.0
    b: float = 1.0
    g: BCValue = 0.0
    kind: ClassVar[str] = "robin"

    def __post_init__(self) -> None:
        if self.b == 0.0:
            raise ConfigurationError("Robin with b=0 is a Dirichlet condition; use Dirichlet(g/a)")
        if self.a == 0.0:
            raise ConfigurationError("Robin with a=0 is a Neumann condition; use Neumann(g/b)")

    def value(self, *args: float | FloatArr) -> float | FloatArr:
        return _eval(self.g, *args)


@dataclass(frozen=True)
class Periodic(BoundaryCondition):
    kind: ClassVar[str] = "periodic"

    def value(self, *args: float | FloatArr) -> float | FloatArr:
        raise ConfigurationError("periodic BCs have no prescribed value")


@dataclass(frozen=True)
class Interface(BoundaryCondition):
    """Interface between materials with conductivities ``k_left`` and ``k_right``."""

    k_left: float = 1.0
    k_right: float = 1.0
    kind: ClassVar[str] = "interface"

    def __post_init__(self) -> None:
        if self.k_left <= 0 or self.k_right <= 0:
            raise ConfigurationError("interface conductivities must be positive")

    def harmonic_conductivity(self) -> float:
        """Face conductivity enforcing flux continuity on a cell-centred grid: harmonic mean."""
        return 2.0 * self.k_left * self.k_right / (self.k_left + self.k_right)


def ghost_value(
    bc: BoundaryCondition, u_inner: float | FloatArr, h: float, t: float = 0.0, s: FloatArr | None = None
) -> float | FloatArr:
    r"""Ghost-node value for a node-centred boundary at distance ``h`` outside the boundary node.

    For node-centred grids the boundary node ``u_b`` itself is unknown for Neumann/Robin; the
    second-order central approximation :math:`(u_{g}-u_{i})/(2h) = \partial u/\partial n` is used,
    where ``u_inner`` is the first interior neighbour. Returns ``u_g``. For Robin:
    :math:`a u_b + b (u_g - u_i)/(2h) = g` requires ``u_b``; solvers handle Robin by assembling
    the boundary row directly, so this helper covers Neumann only.
    """
    args: tuple[float | FloatArr, ...] = (t,) if s is None else (s, t)
    if isinstance(bc, Neumann):
        return u_inner + 2.0 * h * bc.value(*args)
    raise ConfigurationError(f"ghost_value not defined for {bc.kind}")


BC_CATALOGUE: dict[str, dict[str, str]] = {
    "dirichlet": {
        "definition": "u = g on boundary",
        "meaning": "prescribed temperature / concentration / velocity",
        "implementation": "boundary node fixed (FD); ghost cell u_g = 2g - u_P (FV)",
        "compatibility": "g(t=0) should equal u0 on the boundary, else an initial-layer error appears",
        "failure_modes": "corner incompatibility, boundary layers thinner than h at high Pe",
    },
    "neumann": {
        "definition": "du/dn = g",
        "meaning": "prescribed flux (g=0: insulated / zero-gradient outflow)",
        "implementation": "ghost node u_g = u_i + 2 h g (FD), boundary face flux = D g (FV)",
        "compatibility": "pure-Neumann Poisson requires integral(f) + boundary flux = 0 and a gauge",
        "failure_modes": "singular system without gauge fixing; drift of mean value",
    },
    "robin": {
        "definition": "a u + b du/dn = g",
        "meaning": "convective heat transfer: -k dT/dn = h (T - T_inf) => a=h, b=k, g=h T_inf",
        "implementation": "boundary row a u_b + b (u_g - u_i)/(2h) = g with ghost eliminated",
        "compatibility": "a/b >= 0 for well-posedness of the heat equation",
        "failure_modes": "a/b < 0 gives growing solutions; large a/b approaches Dirichlet stiffly",
    },
    "periodic": {
        "definition": "u(x_min) = u(x_max) (and derivative)",
        "meaning": "spatially repeating domain",
        "implementation": "wrap-around indexing (np.roll) / circulant matrix",
        "compatibility": "initial and source data must be periodic",
        "failure_modes": "non-periodic data creates a spurious discontinuity",
    },
    "initial": {
        "definition": "u(x, 0) = u0(x)",
        "meaning": "state at t = 0",
        "implementation": "grid sampling of u0 (FD) / cell averages via midpoint (FV)",
        "compatibility": "consistent with BCs at t = 0",
        "failure_modes": "under-resolved u0 introduces aliasing",
    },
    "interface": {
        "definition": "[u] = 0 and [k du/dn] = 0 across an interface",
        "meaning": "perfect thermal contact between two materials",
        "implementation": "harmonic-mean face conductivity (FV)",
        "compatibility": "interface should coincide with a cell face",
        "failure_modes": "arithmetic mean violates flux continuity for large k contrast",
    },
}
