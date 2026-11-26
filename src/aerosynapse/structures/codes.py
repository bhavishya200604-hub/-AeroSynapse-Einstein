r"""Design codes and the flight envelope (V-n diagram).

Each code fixes the limit load factors, the factor of safety (ultimate = 1.5 x limit in every code here), the
fitting factor for joints (1.15) and the design gust velocities. Margins of safety are reported the usual way,

.. math:: MS=\frac{\text{allowable}}{\text{FS}\cdot\text{(fitting factor)}\cdot\text{limit stress}}-1\ \ge 0 .

Gust load factor (CS-25.341 / CS-23.341 Pratt formula, equivalent airspeed):

.. math:: n=1\pm\frac{K_g\,\rho_0\,U_{de}\,V_e\,a}{2\,W/S},\qquad K_g=\frac{0.88\mu_g}{5.3+\mu_g},\qquad
          \mu_g=\frac{2\,W/S}{\rho\,\bar c\,a\,g}.

References: MIL-A-8861B (fighter strength and rigidity, flight loads), CS-25 and CS-23 (EASA), FAR 25.303
(factor of safety 1.5), FAR 25.625 (fitting factor 1.15).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

G = 9.80665
RHO0 = 1.225


@dataclass(frozen=True)
class Code:
    key: str
    name: str
    n_pos: float  # limit load factor, positive
    n_neg: float  # limit load factor, negative
    factor_of_safety: float = 1.5
    fitting_factor: float = 1.15
    u_gust_vc: float = 15.24  # design gust at VC, EAS m/s (50 ft/s)
    u_gust_vd: float = 7.62  # design gust at VD, EAS m/s (25 ft/s)
    flutter_margin: float = 1.15  # flutter speed must exceed this x VD (EAS)
    reference: str = ""

    def ultimate(self, n: float) -> float:
        return n * self.factor_of_safety


CODES: dict[str, Code] = {
    "mil_fighter": Code("mil_fighter", "MIL-A-8861 fighter / attack", 9.0, -3.0, 1.5, 1.15, 20.1, 10.1, 1.15, "MIL-A-8861B, MIL-A-8870C (flutter margin 15 %)"),
    "cs25": Code("cs25", "CS-25 large aeroplane", 2.5, -1.0, 1.5, 1.15, 15.24, 7.62, 1.15, "CS-25.333, 25.337, 25.341, 25.303, 25.625, 25.629"),
    "cs23_utility": Code("cs23_utility", "CS-23 utility", 4.4, -1.76, 1.5, 1.15, 15.24, 7.62, 1.2, "CS-23.337, 23.341"),
    "aerobatic": Code("aerobatic", "CS-23 aerobatic", 6.0, -3.0, 1.5, 1.15, 15.24, 7.62, 1.2, "CS-23.337"),
}


def gust_factor(W: float, S: float, a: float, c: float, rho: float) -> float:
    mu = 2 * W / S / (rho * c * a * G)
    return 0.88 * mu / (5.3 + mu)


def vn_diagram(code: Code, W: float, S: float, cl_max: float, cl_min: float, a: float, c: float, rho: float, v_c: float, v_d: float, n: int = 200) -> dict[str, Any]:
    """Manoeuvre and gust envelopes in equivalent airspeed. W in N, S in m^2, a = lift slope per rad, speeds EAS m/s."""
    ws = W / S
    v = np.linspace(0, v_d, n)
    stall_pos = np.minimum(0.5 * RHO0 * v**2 * cl_max / ws, code.n_pos)
    stall_neg = np.maximum(0.5 * RHO0 * v**2 * cl_min / ws, code.n_neg)
    v_s1 = float(np.sqrt(2 * ws / (RHO0 * cl_max)))
    v_a = float(v_s1 * np.sqrt(code.n_pos))
    kg = gust_factor(W, S, a, c, rho)
    slope = kg * RHO0 * a / (2 * ws)
    gust = {"V_C": (v_c, 1 + slope * code.u_gust_vc * v_c, 1 - slope * code.u_gust_vc * v_c), "V_D": (v_d, 1 + slope * code.u_gust_vd * v_d, 1 - slope * code.u_gust_vd * v_d)}
    n_gust_max = max(gust["V_C"][1], gust["V_D"][1])
    n_gust_min = min(gust["V_C"][2], gust["V_D"][2])
    return {
        "code": asdict(code),
        "V_eas": v.tolist(),
        "n_manoeuvre_pos": stall_pos.tolist(),
        "n_manoeuvre_neg": stall_neg.tolist(),
        "V_S1": v_s1,
        "V_A": v_a,
        "V_C": v_c,
        "V_D": v_d,
        "K_g": kg,
        "gust_points": {k: {"V": a_[0], "n_up": a_[1], "n_down": a_[2]} for k, a_ in gust.items()},
        "n_limit_pos": max(code.n_pos, n_gust_max),
        "n_limit_neg": min(code.n_neg, n_gust_min),
        "gust_governs": bool(n_gust_max > code.n_pos or n_gust_min < code.n_neg),
    }


def margin(allowable: float, limit_stress: float, code: Code, fitting: bool = False) -> float:
    """Margin of safety at ultimate load; infinite when nothing is applied."""
    applied = abs(limit_stress) * code.factor_of_safety * (code.fitting_factor if fitting else 1.0)
    return float("inf") if applied <= 0 else allowable / applied - 1
