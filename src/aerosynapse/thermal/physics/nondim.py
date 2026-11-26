"""Nondimensionalisation: dimensional properties -> dimensionless groups, scalings, Buckingham pi.

Scales (convective): x* = x/L, u* = u/U, t* = t U / L, p* = p/(rho U^2), theta = (T - T_inf)/dT.
Diffusive scaling (conduction-only Case B): t_F = alpha t / L^2 (Fourier number).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class DimensionalParams:
    """Dimensional inputs (SI). ``k_solid`` is used for the Biot number when given."""

    L: float  # length scale [m]
    U: float  # velocity scale [m/s]
    nu: float  # kinematic viscosity [m^2/s]
    alpha: float  # thermal diffusivity of the fluid [m^2/s]
    k_fluid: float  # fluid conductivity [W/m/K]
    h: float = 0.0  # heat-transfer coefficient [W/m^2/K]
    k_solid: float | None = None  # solid conductivity [W/m/K] (Biot of a solid body)
    rho: float = 1.0  # density [kg/m^3]
    cp: float = 1.0  # specific heat [J/kg/K]
    dT: float = 1.0  # temperature scale [K]
    T_inf: float = 0.0  # reference temperature [K]
    g: float = 9.81  # gravity [m/s^2]
    beta: float = 0.0  # thermal expansion coefficient [1/K]
    q_vol: float = 0.0  # volumetric heat generation [W/m^3]


@dataclass(frozen=True)
class NondimNumbers:
    Re: float
    Pr: float
    Pe: float
    Bi: float
    Ri: float
    Q: float
    bi_conductivity: str  # "solid" or "fluid": which k defines Bi


def reynolds(U: float, L: float, nu: float) -> float:
    return U * L / nu


def prandtl(nu: float, alpha: float) -> float:
    return nu / alpha


def peclet(U: float, L: float, alpha: float) -> float:
    return U * L / alpha


def biot(h: float, L: float, k: float) -> float:
    return h * L / k


def richardson(g: float, beta: float, dT: float, L: float, U: float) -> float:
    return g * beta * dT * L / U**2


def source_number(q_vol: float, L: float, rho: float, cp: float, U: float, dT: float) -> float:
    """Q* = q''' L / (rho cp U dT) (convective scaling)."""
    return q_vol * L / (rho * cp * U * dT)


def to_nondim(p: DimensionalParams, bi_k: Literal["solid", "fluid"] = "solid") -> NondimNumbers:
    """Compute all groups. Bi uses k_solid if ``bi_k == 'solid'`` (must be provided)."""
    if bi_k == "solid":
        if p.k_solid is None:
            raise ValueError("bi_k='solid' requires k_solid")
        k = p.k_solid
    else:
        k = p.k_fluid
    Re = reynolds(p.U, p.L, p.nu)
    Pr = prandtl(p.nu, p.alpha)
    return NondimNumbers(
        Re=Re,
        Pr=Pr,
        Pe=Re * Pr,
        Bi=biot(p.h, p.L, k),
        Ri=richardson(p.g, p.beta, p.dT, p.L, p.U),
        Q=source_number(p.q_vol, p.L, p.rho, p.cp, p.U, p.dT),
        bi_conductivity=bi_k,
    )


# --- variable scalings ------------------------------------------------------------------------


def scale_x(x: np.ndarray | float, L: float) -> np.ndarray | float:
    return np.asarray(x) / L


def unscale_x(xs: np.ndarray | float, L: float) -> np.ndarray | float:
    return np.asarray(xs) * L


def scale_t(t: np.ndarray | float, L: float, U: float) -> np.ndarray | float:
    return np.asarray(t) * U / L


def unscale_t(ts: np.ndarray | float, L: float, U: float) -> np.ndarray | float:
    return np.asarray(ts) * L / U


def scale_T(T: np.ndarray | float, T_inf: float, dT: float) -> np.ndarray | float:
    return (np.asarray(T) - T_inf) / dT


def unscale_T(theta: np.ndarray | float, T_inf: float, dT: float) -> np.ndarray | float:
    return np.asarray(theta) * dT + T_inf


def scale_p(p: np.ndarray | float, rho: float, U: float) -> np.ndarray | float:
    return np.asarray(p) / (rho * U**2)


def convective_to_fourier_time(t_star: np.ndarray | float, Pe: float) -> np.ndarray | float:
    """t_F = alpha t / L^2 = t* / Pe."""
    return np.asarray(t_star) / Pe


# --- Buckingham pi ----------------------------------------------------------------------------

# dimension exponents over base (M, L, T, Theta)
DIMENSIONS: dict[str, tuple[int, int, int, int]] = {
    "L": (0, 1, 0, 0),
    "U": (0, 1, -1, 0),
    "nu": (0, 2, -1, 0),
    "alpha": (0, 2, -1, 0),
    "k": (1, 1, -3, -1),
    "h": (1, 0, -3, -1),
    "rho": (1, -3, 0, 0),
    "cp": (0, 2, -2, -1),
    "dT": (0, 0, 0, 1),
    "g_beta": (0, 1, -2, -1),  # g*beta combined: [m s^-2 K^-1]
    "q_vol": (1, -1, -3, 0),
}


def buckingham_pi(variables: list[str]) -> tuple[int, list[dict[str, sp.Rational]]]:
    """Return (number of pi groups, exponent vectors of a basis of dimensionless groups).

    n_groups = n_variables - rank(dimension matrix)  (Buckingham pi theorem).
    """
    M = sp.Matrix([list(DIMENSIONS[v]) for v in variables]).T  # 4 x n
    basis = M.nullspace()
    groups = []
    for vec in basis:
        # scale to integer exponents
        den = sp.ilcm(*[sp.fraction(e)[1] for e in vec])
        vec = vec * den
        groups.append({v: sp.Rational(vec[i]) for i, v in enumerate(variables) if vec[i] != 0})
    return len(variables) - M.rank(), groups
