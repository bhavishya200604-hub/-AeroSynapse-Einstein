r"""Boundary layer, thermal layer and skin temperature.

Friction uses mean flat-plate laws: Blasius :math:`C_f=1.328/\sqrt{Re}` when laminar and
Prandtl-Schlichting :math:`C_f=0.455/(\log_{10}Re)^{2.58}` when turbulent, joined at the transition
point. Compressibility and wall temperature enter through Eckert's reference temperature

.. math:: T^*=T_e\left[1+0.032M^2+0.58\left(T_w/T_e-1\right)\right],

so air properties are evaluated at :math:`T^*` and a hot wall lowers friction.

Heating uses the Reynolds analogy :math:`St=\tfrac12c_fPr^{-2/3}` with the adiabatic wall temperature
:math:`T_{aw}=T_e(1+r\tfrac{\gamma-1}2M^2)`, recovery factor :math:`r=Pr^{1/2}` laminar and
:math:`Pr^{1/3}` turbulent. Stagnation heating follows Fay-Riddell in its simple sphere form. The
skin temperature balances convection against radiation, with no internal cooling, which is the
conservative limit.

The thermal layer thickness is :math:`\delta_T=\delta\,Pr^{-1/3}` when laminar and close to
:math:`\delta` when turbulent.

Limits: flat-plate laws on each strip, so pressure gradients enter only through form factors here.
The integral boundary-layer solver in :mod:`aerosynapse.shapeopt.vii` handles those for 2D sections.
Above about Mach 5 real-gas effects matter and this model should not be used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from aerosynapse.aircraft.atmosphere import CP_AIR, GAMMA, PR, R_AIR, SIGMA_SB, Air, sutherland


def cf_laminar(Re: float) -> float:
    return 1.328 / np.sqrt(max(Re, 1.0))


def cf_turbulent(Re: float) -> float:
    return 0.455 / np.log10(max(Re, 10.0)) ** 2.58


def cf_mixed(Re: float, x_tr: float) -> float:
    """Mean friction for a surface with transition at fraction ``x_tr`` of its length."""
    if x_tr >= 1.0:
        return cf_laminar(Re)
    if x_tr <= 0.0:
        return cf_turbulent(Re)
    Rt = Re * x_tr
    return cf_turbulent(Re) - x_tr * (cf_turbulent(Rt) - cf_laminar(Rt))


def cf_rough(L: float, k: float) -> float:
    """Mean friction of a fully rough plate (Schlichting), for roughness height k."""
    return float((1.89 + 1.62 * np.log10(max(L / k, 10.0))) ** -2.5)


def recovery(turbulent: bool = True) -> float:
    return PR ** (1 / 3) if turbulent else PR**0.5


def t_adiabatic(T: float, M: float, turbulent: bool = True) -> float:
    return T * (1 + recovery(turbulent) * (GAMMA - 1) / 2 * M**2)


def t_total(T: float, M: float) -> float:
    return T * (1 + (GAMMA - 1) / 2 * M**2)


def t_reference(T: float, M: float, Tw: float) -> float:
    return T * (1 + 0.032 * M**2 + 0.58 * (Tw / T - 1))


def friction(air: Air, M: float, L: float, x_tr: float, Tw: float | None = None, roughness: float = 0.0) -> float:
    """Mean skin friction coefficient of a surface of length L, referred to free-stream q.

    Roughness only matters once it pokes through the viscous sublayer, so the rough-plate law
    acts as a floor. It is why a dirty wing has more drag even when it was already turbulent.
    """
    Tw = t_adiabatic(air.T, M) if Tw is None else Tw
    Ts = t_reference(air.T, M, Tw)
    rho_s = air.p / (R_AIR * Ts)
    Re_s = rho_s * M * air.a * L / sutherland(Ts)
    cf = cf_mixed(Re_s, x_tr)
    if roughness > 0:
        cf = max(cf, cf_rough(L, roughness))
    return float(cf * rho_s / air.rho)


def layer_thickness(Re_x: float, x: float, turbulent: bool) -> float:
    """Velocity boundary-layer thickness at distance x."""
    return 0.37 * x / Re_x**0.2 if turbulent else 5.0 * x / np.sqrt(Re_x)


def thermal_thickness(delta: float, turbulent: bool) -> float:
    return delta if turbulent else delta * PR ** (-1 / 3)


def q_convective(air: Air, M: float, x: float, Tw: float, turbulent: bool = True) -> float:
    """Local convective heat flux into the wall at distance x, W/m^2. Positive heats the skin."""
    Ts = t_reference(air.T, M, Tw)
    rho_s = air.p / (R_AIR * Ts)
    V = M * air.a
    Re = max(rho_s * V * x / sutherland(Ts), 1.0)
    cf = 0.0592 / Re**0.2 if turbulent else 0.664 / np.sqrt(Re)
    St = 0.5 * cf * PR ** (-2 / 3)
    return float(St * rho_s * V * CP_AIR * (t_adiabatic(air.T, M, turbulent) - Tw))


def q_stagnation(air: Air, M: float, R_nose: float, Tw: float) -> float:
    """Stagnation heating on a sphere of radius R_nose, W/m^2 (Fay-Riddell, simple form)."""
    if M < 0.3:
        return 0.0
    T0 = t_total(air.T, M)
    if M > 1:
        p0 = air.p * ((GAMMA + 1) ** 2 * M**2 / (4 * GAMMA * M**2 - 2 * (GAMMA - 1))) ** (GAMMA / (GAMMA - 1)) * (1 - GAMMA + 2 * GAMMA * M**2) / (GAMMA + 1)
    else:
        p0 = air.p * (T0 / air.T) ** (GAMMA / (GAMMA - 1))
    rho0 = p0 / (R_AIR * T0)
    mu0 = sutherland(T0)
    dudx = np.sqrt(2 * max(p0 - air.p, 1.0) / rho0) / R_nose
    return float(0.763 * PR**-0.6 * np.sqrt(rho0 * mu0 * dudx) * CP_AIR * (T0 - Tw))


def skin_temperature(flux: Callable[[float], float], T_hi: float, T_sky: float, emissivity: float = 0.85) -> float:
    """Wall temperature where convection ``flux(Tw)`` balances radiation to the sky."""
    lo, hi = T_sky, T_hi
    for _ in range(50):
        t = 0.5 * (lo + hi)
        if flux(t) - emissivity * SIGMA_SB * (t**4 - T_sky**4) > 0:
            lo = t
        else:
            hi = t
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Material:
    """Round-number structural properties for comparison. Upload real allowables for real work."""

    name: str
    rho: float  # kg/m^3
    E: float  # Pa at 293 K
    sigma: float  # working stress, Pa
    alpha: float  # expansion, 1/K
    T_max: float  # K, long-term service limit
    softening: float  # fractional loss of E per K above 293 K


MATERIALS = {
    "aluminium": Material("Aluminium alloy", 2800.0, 7.0e10, 3.0e8, 23e-6, 400.0, 6e-4),
    "cfrp": Material("Carbon fibre, epoxy", 1600.0, 7.0e10, 5.0e8, 2e-6, 420.0, 1e-3),
    "titanium": Material("Titanium alloy", 4500.0, 1.1e11, 6.0e8, 9e-6, 750.0, 4e-4),
}


@dataclass
class ThermalState:
    T_total: float
    T_adiabatic: float
    T_nose: float
    T_skin: float
    T_skin_aft: float
    q_nose: float
    q_mean: float
    heat_power: float  # W, heat convected into the skin
    delta_aft: float  # velocity layer thickness at the tail, m
    delta_T_aft: float
    expansion_m: float
    E_ratio: float
    stress_MPa: float
    margin_K: float
    wake_dT: float  # mean temperature rise of the disturbed air, K


def thermal_state(air: Air, M: float, L: float, S_wet: float, R_nose: float, mat: Material, drag: float = 0.0, x_tr_frac: float = 0.1) -> ThermalState:
    """Skin, thermal layer and structural temperature effects at one flight state."""
    Tt = t_total(air.T, M)
    Taw = t_adiabatic(air.T, M)
    T_sky = max(air.T - 20.0, 150.0)
    hi = max(Tt, air.T) + 5.0
    T_n = skin_temperature(lambda t: q_stagnation(air, M, R_nose, t), hi, T_sky)
    x_m, x_a = max(0.3 * L, 1e-3), max(0.9 * L, 1e-3)
    T_s = skin_temperature(lambda t: q_convective(air, M, x_m, t), hi, T_sky)
    T_a = skin_temperature(lambda t: q_convective(air, M, x_a, t), hi, T_sky)
    V = M * air.a
    Re_L = max(air.rho * V * L / air.mu, 1.0)
    d = layer_thickness(Re_L, L, True)
    q_s = q_stagnation(air, M, R_nose, T_n)
    q_m = q_convective(air, M, x_m, T_s)
    dT = T_s - 293.0
    # the power that went into the air: drag work ends up as heat in the wake
    wake_area = max(S_wet * 0.05, 1.0)
    return ThermalState(
        Tt, Taw, T_n, T_s, T_a, q_s, q_m, q_m * S_wet, d, thermal_thickness(d, True),
        mat.alpha * L * (T_s - 288.0), max(1 - mat.softening * max(dT, 0.0), 0.05), mat.E * mat.alpha * abs(dT) / 1e6,
        mat.T_max - T_n, drag * V / max(air.rho * V * wake_area * CP_AIR, 1e-9),
    )  # fmt: skip
