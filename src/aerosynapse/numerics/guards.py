"""Divergence guards: never return a plausible-looking field from a failed solve (spec Section 16)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aerosynapse.exceptions import SolverDivergenceError


def check_field(u: NDArray[np.float64], name: str = "u", step: int | None = None, bound: float = 1e8) -> None:
    """Raise :class:`SolverDivergenceError` if ``u`` is non-finite or exceeds ``bound``."""
    where = f" at step {step}" if step is not None else ""
    if not np.all(np.isfinite(u)):
        raise SolverDivergenceError(f"{name} became non-finite{where}")
    m = float(np.max(np.abs(u))) if u.size else 0.0
    if m > bound:
        raise SolverDivergenceError(f"{name} exceeded growth bound {bound:g} (max |{name}| = {m:.3e}){where}")
