"""Linear stability of a steady MAC Navier–Stokes solution (Step 6.3, labelled study beyond the laminar regime).

For :class:`SteadyNewtonNS` the velocity rows of the residual ARE the time derivative (R_u = N_u - G p = du/dt), the
continuity rows are the constraint D u = 0. Small perturbations w = (u', v', p') e^{lambda t} therefore satisfy the
generalized eigenproblem
    J w = lambda B w,     B = identity on the velocity unknowns, 0 on pressure (and temperature not included),
with J the exact Newton Jacobian at the steady state. Re(lambda) > 0 for some lambda: the steady solution is linearly
UNSTABLE, i.e. the flow does not stay at this steady state (a Hopf bifurcation if Im(lambda) != 0).
Rightmost eigenvalues: complex shift-invert Arnoldi around sigma = s + i w for several w (ARPACK through
scipy.sparse.linalg.eigs on the operator (J - sigma B)^{-1} B; lambda = sigma + 1/mu). Infinite eigenvalues of the
singular pencil map to mu = 0 and are never selected.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from aerosynapse.thermal.numerics.ns_steady import SteadyNewtonNS


def mass_matrix(N: SteadyNewtonNS) -> sps.csr_matrix:
    d = np.zeros(N.n)
    d[N.sl["u"]] = 1.0
    d[N.sl["v"]] = 1.0
    return sps.diags(d).tocsc()


def rightmost(
    N: SteadyNewtonNS, x: np.ndarray, shifts_imag: list[float], real_shift: float = 0.0, k: int = 6, tol: float = 1e-10
) -> dict:
    """Eigenvalues near each shift sigma = real_shift + i w: all found, the rightmost one, its growth rate."""
    J, _ = N.jacobian(x)
    J = sps.csc_matrix(J, dtype=complex)
    B = mass_matrix(N).astype(complex)
    found = []
    for w in shifts_imag:
        sigma = complex(real_shift, float(w))
        lu = spla.splu((J - sigma * B).tocsc())
        op = spla.LinearOperator(J.shape, matvec=lambda v, lu=lu: lu.solve(B @ v), dtype=complex)
        mu = spla.eigs(op, k=min(k, N.n - 2), which="LM", tol=tol, return_eigenvectors=False, maxiter=5000)
        lam = sigma + 1.0 / mu[np.abs(mu) > 1e-14]
        found.extend(lam.tolist())
    lam = np.array(found)
    # duplicates from overlapping shifts: keep unique to 1e-8
    uniq = []
    for v in lam[np.argsort(-lam.real)]:
        if not any(abs(v - u) < 1e-8 * max(1.0, abs(v)) for u in uniq):
            uniq.append(v)
    lead = uniq[0]
    return {
        "eigenvalues": [[float(v.real), float(v.imag)] for v in uniq],
        "growth_rate": float(lead.real),
        "frequency": float(abs(lead.imag)),
        "unstable": bool(lead.real > 0),
    }


def crossing(Re: list[float], growth: list[float]) -> float | None:
    """Re where the growth rate first changes sign from < 0 to > 0 (linear interpolation); None if it never does."""
    for i in range(1, len(Re)):
        if growth[i - 1] < 0 <= growth[i]:
            a, b = growth[i - 1], growth[i]
            return float(Re[i - 1] + (Re[i] - Re[i - 1]) * (-a) / (b - a))
    return None
