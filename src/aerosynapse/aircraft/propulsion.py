r"""Gas-turbine thrust from a thermodynamic cycle: turbojet or separate-flow turbofan.

Station temperatures follow the Brayton cycle with polytropic efficiencies. The fuel-air ratio
comes from the burner energy balance, the turbine work from the compressor and fan work, and each
nozzle expands fully to ambient pressure:

.. math:: \frac{F}{\dot m_0}=\frac{(1+f)V_9-V_0+\alpha(V_{19}-V_0)}{1+\alpha},\qquad
          TSFC=\frac{f}{(1+\alpha)\,F/\dot m_0}.

With perfect components and no bypass this reduces to the closed-form ideal turbojet, which the
tests check. Off-design mass flow scales with corrected flow, and the temperature limit falls with
the hot-day margin, so a hot day costs thrust as it does in service.

Limits: constant specific heats and full expansion, with no map of compressor stall or inlet
distortion. Use an engine deck through the data slots for real performance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from aerosynapse.aircraft.atmosphere import CP_AIR, GAMMA, Air

H_FUEL = 43.0e6
EX = (GAMMA - 1) / GAMMA


@dataclass(frozen=True)
class Engine:
    opr: float = 30.0
    fan_pr: float = 1.6
    bpr: float = 6.0
    T4: float = 1600.0
    eta_c: float = 0.90
    eta_f: float = 0.90
    eta_t: float = 0.90
    eta_b: float = 0.99
    pi_b: float = 0.96
    pi_d: float = 0.98
    afterburner_T: float = 0.0  # K, zero means off; only for bpr = 0

    @classmethod
    def ideal_turbojet(cls, opr: float, T4: float) -> Engine:
        return cls(opr=opr, fan_pr=1.0, bpr=0.0, T4=T4, eta_c=1.0, eta_f=1.0, eta_t=1.0, eta_b=1.0, pi_b=1.0, pi_d=1.0)


@dataclass
class Cycle:
    spec_thrust: float  # N per kg/s of total inlet flow
    tsfc: float  # kg/(N s)
    f: float
    V9: float
    V19: float
    T_exhaust: float
    eta_propulsive: float
    eta_thermal: float
    valid: bool


def cycle(e: Engine, air: Air, M: float) -> Cycle:
    V0 = M * air.a
    Tt0 = air.T * (1 + 0.2 * M**2)
    pt0 = air.p * (1 + 0.2 * M**2) ** (1 / EX) * e.pi_d
    Tt13 = Tt0 * e.fan_pr ** (EX / e.eta_f)
    Tt3 = Tt0 * e.opr ** (EX / e.eta_c)
    if Tt3 >= e.T4:
        return Cycle(0.0, 1.0, 0.0, 0.0, 0.0, Tt3, 0.0, 0.0, False)
    f = CP_AIR * (e.T4 - Tt3) / (e.eta_b * H_FUEL - CP_AIR * e.T4)
    Tt5 = e.T4 - ((Tt3 - Tt0) + e.bpr * (Tt13 - Tt0)) / (1 + f)
    if Tt5 <= 0.3 * e.T4:
        return Cycle(0.0, 1.0, f, 0.0, 0.0, Tt5, 0.0, 0.0, False)
    pi_t = (Tt5 / e.T4) ** (1 / (EX * e.eta_t))
    pt5 = pt0 * e.opr * e.pi_b * pi_t
    if e.afterburner_T > Tt5:
        f += CP_AIR * (e.afterburner_T - Tt5) / (0.95 * H_FUEL - CP_AIR * e.afterburner_T)
        Tt5 = e.afterburner_T
    V9 = float(np.sqrt(max(2 * CP_AIR * Tt5 * (1 - (air.p / pt5) ** EX), 0.0)))
    pt13 = pt0 * e.fan_pr
    V19 = float(np.sqrt(max(2 * CP_AIR * Tt13 * (1 - (air.p / pt13) ** EX), 0.0))) if e.bpr > 0 else 0.0
    a = e.bpr
    F = ((1 + f) * V9 - V0 + a * (V19 - V0)) / (1 + a)
    if F <= 0:
        return Cycle(0.0, 1.0, f, V9, V19, Tt5, 0.0, 0.0, False)
    tsfc = f / ((1 + a) * F)
    ke = 0.5 * ((1 + f) * V9**2 + a * V19**2 - (1 + a) * V0**2) / (1 + a)
    eta_p = F * V0 / ke if ke > 0 else 0.0
    eta_th = (1 + a) * ke / (f * H_FUEL) if f > 0 else 0.0
    return Cycle(float(F), float(tsfc), float(f), V9, V19, float(Tt5), float(eta_p), float(eta_th), True)


def ideal_turbojet_thrust(air: Air, M: float, opr: float, T4: float) -> float:
    """Closed-form specific thrust of the ideal turbojet (Mattingly), per kg/s."""
    tr = 1 + 0.2 * M**2
    tc = opr**EX
    tl = T4 / air.T
    tt = 1 - tr / tl * (tc - 1)
    v9a = np.sqrt(5 * tl / (tr * tc) * (tr * tc * tt - 1))
    return float(air.a * (v9a - M))


def available_thrust(e: Engine, air: Air, M: float, mdot_ref: float, ref_air: Air, ref_M: float, hot_dT: float = 0.0) -> tuple[float, float]:
    """Thrust and TSFC at an off-design point for an engine sized at (ref_air, ref_M)."""
    th = air.T * (1 + 0.2 * M**2) / (ref_air.T * (1 + 0.2 * ref_M**2))
    de = air.p * (1 + 0.2 * M**2) ** (1 / EX) / (ref_air.p * (1 + 0.2 * ref_M**2) ** (1 / EX))
    mdot = mdot_ref * de / np.sqrt(th)
    c = cycle(replace(e, T4=e.T4 - max(hot_dT, 0.0)), air, M)
    return c.spec_thrust * mdot, c.tsfc
