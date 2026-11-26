r"""Design variables and design space (spec Section 8A).

Continuous variables are normalised to [0, 1] by :math:`z=(x-lo)/(hi-lo)`; integers are decoded by
rounding; categoricals map a [0, 1) code onto ``len(choices)`` equal bins (index encoding — used
by GA/DE/BO alike). ``kind`` documents the physical role (geometric, material, operating, control).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerosynapse.exceptions import ConfigurationError

FloatArr = NDArray[np.float64]


@dataclass(frozen=True)
class Variable:
    name: str
    lo: float = 0.0
    hi: float = 1.0
    vtype: str = "continuous"  # continuous | integer | categorical
    choices: tuple[Any, ...] = ()
    unit: str = "1"
    kind: str = "geometric"

    def __post_init__(self) -> None:
        if self.vtype not in ("continuous", "integer", "categorical"):
            raise ConfigurationError(f"unknown variable type {self.vtype}")
        if self.vtype == "categorical" and not self.choices:
            raise ConfigurationError("categorical variable needs choices")
        if self.vtype != "categorical" and not self.hi > self.lo:
            raise ConfigurationError(f"variable {self.name}: hi must exceed lo")


@dataclass
class DesignSpace:
    variables: list[Variable] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return len(self.variables)

    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    def decode(self, z: FloatArr) -> dict[str, Any]:
        """Unit-cube code -> physical design (dict)."""
        z = np.clip(np.asarray(z, float), 0.0, 1.0)
        out: dict[str, Any] = {}
        for zi, v in zip(z, self.variables, strict=True):
            if v.vtype == "categorical":
                out[v.name] = v.choices[min(int(zi * len(v.choices)), len(v.choices) - 1)]
            else:
                x = v.lo + zi * (v.hi - v.lo)
                out[v.name] = int(round(x)) if v.vtype == "integer" else float(x)
        return out

    def encode(self, design: dict[str, Any]) -> FloatArr:
        z = []
        for v in self.variables:
            x = design[v.name]
            if v.vtype == "categorical":
                z.append((v.choices.index(x) + 0.5) / len(v.choices))
            else:
                z.append((float(x) - v.lo) / (v.hi - v.lo))
        return np.array(z)

    def sample(self, n: int, rng: np.random.Generator) -> FloatArr:
        return rng.random((n, self.dim))
