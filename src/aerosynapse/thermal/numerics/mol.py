"""Independent method-of-lines cross-check: SciPy solve_ivp (BDF / Radau) on any semi-discretisation
exposing ``A`` (constant Jacobian) and ``b(t)`` or a ``rhs(y, t)``."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

Array = NDArray[np.float64]


def solve_mol(
    prob: Any,
    y0: Array,
    t0: float,
    t1: float,
    method: str = "Radau",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    t_eval: Array | None = None,
) -> Any:
    """Integrate y' = rhs(y, t) with SciPy's adaptive stiff solvers. Returns the solve_ivp result.

    For linear problems the exact constant Jacobian A is passed (sparse), making Newton exact.
    """
    jac = None
    try:
        jac = prob.A
    except (ValueError, AttributeError):
        pass
    sol = solve_ivp(
        lambda t, y: prob.rhs(y, t),
        (t0, t1),
        np.asarray(y0, float),
        method=method,
        rtol=rtol,
        atol=atol,
        jac=jac,
        t_eval=t_eval,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol
