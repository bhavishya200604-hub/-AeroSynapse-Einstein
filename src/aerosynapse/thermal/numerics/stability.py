"""Stability and accuracy analysis of the 1D schemes in :mod:`aerosynapse.thermal.numerics.fd1d`.

von Neumann:  theta_j^n = G^n e^{i j xi}.  Semi-discrete symbol (per unit time)
    lambda(xi) dt = z(xi) = -C S_c(xi) + r S_d(xi) - sigma,   C = a dt/h, r = D dt/h^2, sigma = s dt
with S_c = sum_j c_j e^{i j xi} (convective stencil, derivative = (1/h) sum c_j theta_{i+j}) and
S_d = 2 cos xi - 2.  The fully discrete amplification factor is G = R(z) where R is the stability
function of the time integrator (BDF2: largest root of (3/2 - z) G^2 - 2 G + 1/2 = 0).

Modified equation: Taylor expansion of the stencil gives
    (1/h) sum_j c_j theta(x + j h) = theta_x + sum_{k>=2} e_k h^{k-1} d^k theta/dx^k
so the scheme solves theta_t + a theta_x = D theta_xx - a sum_k e_k h^{k-1} theta^{(k)} (+ time error).
For forward Euler the leading time error adds -dt/2 theta_tt = -(a^2 dt/2) theta_xx (pure advection),
giving the numerical diffusion  nu_num = -a h e_2 - a^2 dt / 2.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import sympy as sp

# derivative stencils for a > 0: d/dx ~ (1/h) sum_j c_j theta_{i+j}
CONV_STENCILS: dict[str, dict[int, sp.Rational]] = {
    "upwind1": {0: sp.Integer(1), -1: sp.Integer(-1)},
    "central": {1: sp.Rational(1, 2), -1: sp.Rational(-1, 2)},
    "upwind2": {0: sp.Rational(3, 2), -1: sp.Integer(-2), -2: sp.Rational(1, 2)},
    "quick": {1: sp.Rational(3, 8), 0: sp.Rational(3, 8), -1: sp.Rational(-7, 8), -2: sp.Rational(1, 8)},
}
DIFF_STENCIL = {1: sp.Integer(1), 0: sp.Integer(-2), -1: sp.Integer(1)}

xi, C, r, sig, z = sp.symbols("xi C r sigma z")


def fourier_symbol(stencil: dict[int, sp.Expr]) -> sp.Expr:
    """S(xi) = sum_j c_j exp(i j xi)."""
    return sp.expand(sum(c * sp.exp(sp.I * j * xi) for j, c in stencil.items()))


def semi_discrete_z(scheme: str) -> sp.Expr:
    """z(xi; C, r, sigma) = dt * (Fourier symbol of the semi-discrete operator)."""
    return -C * fourier_symbol(CONV_STENCILS[scheme]) + r * fourier_symbol(DIFF_STENCIL) - sig


STABILITY_FUNCTIONS: dict[str, Callable[[sp.Expr], sp.Expr]] = {
    "euler": lambda z_: 1 + z_,
    "ssprk3": lambda z_: 1 + z_ + z_**2 / 2 + z_**3 / 6,
    "rk4": lambda z_: 1 + z_ + z_**2 / 2 + z_**3 / 6 + z_**4 / 24,
    "implicit_euler": lambda z_: 1 / (1 - z_),
    "crank_nicolson": lambda z_: (1 + z_ / 2) / (1 - z_ / 2),
}


def amplification_factor(scheme: str, integrator: str) -> sp.Expr:
    """Symbolic G(xi; C, r, sigma) for single-step integrators."""
    return STABILITY_FUNCTIONS[integrator](semi_discrete_z(scheme))


def abs2_simplified(G: sp.Expr) -> sp.Expr:
    """|G|^2 simplified with real C, r, xi."""
    Cr, rr, xr = sp.symbols("C r xi", real=True)
    Gr = G.subs({C: Cr, r: rr, xi: xr, sig: 0})
    return sp.simplify(sp.expand(sp.re(Gr) ** 2 + sp.im(Gr) ** 2).rewrite(sp.cos)).subs({Cr: C, rr: r, xr: xi})


def _R_numeric(integrator: str, zv: np.ndarray) -> np.ndarray:
    if integrator == "bdf2":
        # roots of (3/2 - z) G^2 - 2 G + 1/2 = 0
        a2 = 1.5 - zv
        disc = np.sqrt(4.0 - 2.0 * a2 + 0j)
        return np.maximum(np.abs((2 + disc) / (2 * a2)), np.abs((2 - disc) / (2 * a2)))
    f = sp.lambdify(z, STABILITY_FUNCTIONS[integrator](z), "numpy")
    return np.abs(f(zv))


def max_amplification(scheme: str, integrator: str, Cv: float, rv: float, sv: float = 0.0, nxi: int = 1441) -> float:
    """max_xi |G(xi)| evaluated numerically."""
    fz = sp.lambdify((xi, C, r, sig), semi_discrete_z(scheme), "numpy")
    xs = np.linspace(0.0, np.pi, nxi)
    return float(np.max(_R_numeric(integrator, fz(xs, Cv, rv, sv))))


def stability_map(scheme: str, integrator: str, Cs: np.ndarray, rs: np.ndarray, nxi: int = 721) -> np.ndarray:
    """max|G| on a (len(rs), len(Cs)) grid."""
    fz = sp.lambdify((xi, C, r, sig), semi_discrete_z(scheme), "numpy")
    xs = np.linspace(0.0, np.pi, nxi)[None, None, :]
    Z = fz(xs, Cs[None, :, None], rs[:, None, None], 0.0)
    return np.max(_R_numeric(integrator, Z), axis=-1)


def max_stable_courant(scheme: str, integrator: str, rv: float = 0.0, tol: float = 1e-10, Cmax: float = 10.0) -> float:
    """Largest C in [0, Cmax] with max|G| <= 1 + tol (bisection; assumes a single threshold)."""
    lo, hi = 0.0, Cmax
    if max_amplification(scheme, integrator, hi, rv) <= 1 + tol:
        return np.inf
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if max_amplification(scheme, integrator, mid, rv) <= 1 + tol:
            lo = mid
        else:
            hi = mid
    return lo


# --- modified equation ------------------------------------------------------------------------


def stencil_taylor_coeffs(stencil: dict[int, sp.Expr], kmax: int = 5) -> dict[int, sp.Expr]:
    """e_k in (1/h) sum c_j theta(x+jh) = sum_k e_k h^{k-1} theta^{(k)}, k = 0..kmax (e_1 = 1)."""
    return {
        k: sp.nsimplify(sum(c * sp.Integer(j) ** k for j, c in stencil.items()) / sp.factorial(k))
        for k in range(kmax + 1)
    }


def modified_equation_terms(scheme: str, kmax: int = 4) -> dict[int, sp.Expr]:
    """Spatial truncation terms: returns {k: coefficient of d^k theta/dx^k} on the RHS of
    theta_t + a theta_x = D theta_xx + ...  i.e. -a e_k h^{k-1}, for k >= 2."""
    a, h = sp.symbols("a h", positive=True)
    e = stencil_taylor_coeffs(CONV_STENCILS[scheme], kmax)
    assert e[0] == 0 and e[1] == 1, "stencil must be a consistent first-derivative approximation"
    return {k: sp.simplify(-a * e[k] * h ** (k - 1)) for k in range(2, kmax + 1) if e[k] != 0}


def numerical_diffusion_euler(scheme: str) -> sp.Expr:
    """nu_num = -a h e_2 - a^2 dt/2 for forward Euler + given stencil (pure advection)."""
    a, h, dt = sp.symbols("a h dt", positive=True)
    e2 = stencil_taylor_coeffs(CONV_STENCILS[scheme], 2)[2]
    return sp.simplify(-a * h * e2 - a**2 * dt / 2)


# --- discrete diagnostics ---------------------------------------------------------------------


def total_variation(u: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(u))))


def count_oscillations(u: np.ndarray, tol: float = 1e-12) -> int:
    """Number of sign changes of the discrete slope (0 for a monotone profile)."""
    d = np.diff(u)
    d = d[np.abs(d) > tol]
    return int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))


def central_steady_ratio(Pe_h: float) -> float:
    """Root q of the homogeneous central-difference steady recurrence; q < 0 <=> Pe_h > 2 (wiggles)."""
    return (1 + Pe_h / 2) / (1 - Pe_h / 2)
