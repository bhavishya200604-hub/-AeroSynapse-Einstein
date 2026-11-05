r"""Engineering demonstration problems (spec Section 8 examples), each with stated physics and units.

**Thermal rod (dimensionless).** Steady fin equation :math:`-\theta''+m^2\theta=q(x)` on [0, 1],
θ = (T - T_∞)/ΔT_ref, x = X/L, m² = hPL²/(kA) (lateral convection), heat sink θ(0) = 0, convective tip
θ'(1) + Bi θ(1) = 0, Gaussian source of unit power at x_s (width 0.05). Material enters through the
conductivity ratio k_rel (θ scales as 1/k_rel). Objective :math:`f=\theta_{max}/\theta_{ref}+\lambda P/P_{ref}`
with cooling cost P ∝ m² (assumption: cost proportional to the lateral heat-transfer coefficient h),
λ = 0.1; both terms are dimensionless. Constraint: tip (sensitive component) θ(1) ≤ θ_tip,max.

**Source placement (conv-diff).** Steady :math:`v u'-Du''=q\,\phi(x-s)` on [0,1], u(0)=u(1)=0, Pe = 20.
Minimise the peak concentration (per c_ref) subject to delivering u(x_t) ≥ c_min at x_t = 0.7.

**Cantilever beam (Euler–Bernoulli).** Tip load P, length L, rectangular section b × t, material
category. Mass m = ρbtL; stress σ = 6PL/(bt²) ≤ σ_y/SF; deflection δ = 4PL³/(Ebt³) ≤ δ_max.
Material values are *illustrative representative values* (not certified data); SF = 1.5.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aerosynapse.numerics.fd1d import assemble_1d, steady_solve
from aerosynapse.optim.problem import EvalResult
from aerosynapse.optim.variables import DesignSpace, Variable
from aerosynapse.physics.bc import Dirichlet, Robin

K_REL = {"copper": 1.0, "aluminium": 0.6, "steel": 0.13}
MATERIALS = {  # illustrative: E [Pa], rho [kg/m^3], yield [Pa]
    "steel": {"E": 200e9, "rho": 7850.0, "sy": 250e6},
    "aluminium": {"E": 69e9, "rho": 2700.0, "sy": 240e6},
    "titanium": {"E": 110e9, "rho": 4500.0, "sy": 830e6},
}


def thermal_space(with_material: bool = True) -> DesignSpace:
    v = [Variable("m", 0.5, 4.0, unit="1", kind="operating"), Variable("x_s", 0.1, 0.9, unit="1", kind="geometric")]
    if with_material:
        v.append(Variable("material", vtype="categorical", choices=tuple(K_REL), kind="material"))
    return DesignSpace(v)


def thermal_field(
    m: float, x_s: float, k_rel: float = 1.0, bi: float = 1.0, n: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    op = assemble_1d(0, 1, n, 1.0, 0.0, Dirichlet(0.0), Robin(bi, 1.0, 0.0), kappa=m**2)
    w = 0.05
    q = lambda x: np.exp(-((x - x_s) ** 2) / (2 * w**2)) / (w * np.sqrt(2 * np.pi))  # noqa: E731
    return op.x, steady_solve(op, source=q) / k_rel


def thermal_model(
    d: dict[str, Any], bi: float = 1.0, n: int = 400, lam: float = 0.1, theta_ref: float = 0.1, tip_max: float = 0.02
) -> EvalResult:
    x, th = thermal_field(d["m"], d["x_s"], K_REL[d.get("material", "copper")], bi, n)
    th_max = float(np.max(th))
    P = d["m"] ** 2 / 4.0**2
    return EvalResult(
        f=[th_max / theta_ref + lam * P],
        g=[float(th[-1] / tip_max - 1.0)],
        info={"theta_max": th_max, "theta_tip": float(th[-1]), "P": P},
    )


def thermal_two_objectives(d: dict[str, Any]) -> EvalResult:
    r = thermal_model(d)
    return EvalResult(f=[r.info["theta_max"] / 0.1, r.info["P"]], g=r.g, info=r.info)


def source_space() -> DesignSpace:
    return DesignSpace([Variable("s", 0.05, 0.95, kind="geometric"), Variable("q", 0.1, 5.0, kind="operating")])


def source_model(
    d: dict[str, Any], pe: float = 20.0, n: int = 400, x_t: float = 0.7, c_min: float = 0.05, c_ref: float = 0.1
) -> EvalResult:
    op = assemble_1d(0, 1, n, 1.0 / pe, 1.0, Dirichlet(0.0), Dirichlet(0.0), scheme="central")
    w = 0.03
    src = lambda x: d["q"] * np.exp(-((x - d["s"]) ** 2) / (2 * w**2)) / (w * np.sqrt(2 * np.pi))  # noqa: E731
    u = steady_solve(op, source=src)
    ut = float(np.interp(x_t, op.x, u))
    return EvalResult(f=[float(np.max(u)) / c_ref], g=[float(1.0 - ut / c_min)], info={"u_target": ut})


def beam_space() -> DesignSpace:
    return DesignSpace(
        [
            Variable("b", 0.01, 0.10, unit="m"),
            Variable("t", 0.01, 0.20, unit="m"),
            Variable("material", vtype="categorical", choices=tuple(MATERIALS), kind="material"),
        ]
    )


def beam_response(d: dict[str, Any], P: float = 5e3, L: float = 1.0) -> dict[str, float]:
    m = MATERIALS[d["material"]]
    b, t = d["b"], d["t"]
    return {
        "mass": m["rho"] * b * t * L,
        "stress": 6 * P * L / (b * t**2),
        "defl": 4 * P * L**3 / (m["E"] * b * t**3),
        "sy": m["sy"],
    }


def beam_model(d: dict[str, Any], sf: float = 1.5, defl_max: float = 0.005) -> EvalResult:
    r = beam_response(d)
    return EvalResult(f=[r["mass"]], g=[r["stress"] * sf / r["sy"] - 1.0, r["defl"] / defl_max - 1.0], info=r)


def beam_three_objectives(d: dict[str, Any], sf: float = 1.5) -> EvalResult:
    r = beam_response(d)
    return EvalResult(f=[r["mass"], r["defl"] * 1e3, r["stress"] / 1e6], g=[r["stress"] * sf / r["sy"] - 1.0], info=r)


# --- analytic benchmarks ------------------------------------------------------------------------------
def rosenbrock_space() -> DesignSpace:
    return DesignSpace([Variable("x1", -2.0, 2.0), Variable("x2", -1.0, 3.0)])


def rosenbrock(d: dict[str, Any]) -> EvalResult:
    x, y = d["x1"], d["x2"]
    return EvalResult(f=[(1 - x) ** 2 + 100 * (y - x**2) ** 2])


def qp_space() -> DesignSpace:
    return DesignSpace([Variable("x1", -2.0, 2.0), Variable("x2", -2.0, 2.0)])


def qp(d: dict[str, Any]) -> EvalResult:
    """min x1² + x2² s.t. 1 - x1 - x2 ≤ 0 → x* = (0.5, 0.5), μ* = 1 (convex)."""
    return EvalResult(f=[d["x1"] ** 2 + d["x2"] ** 2], g=[1 - d["x1"] - d["x2"]])


def branin_space() -> DesignSpace:
    return DesignSpace([Variable("x1", -5.0, 10.0), Variable("x2", 0.0, 15.0)])


def branin(d: dict[str, Any]) -> EvalResult:
    x1, x2 = d["x1"], d["x2"]
    a, b, c, r, s, t = 1, 5.1 / (4 * np.pi**2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)
    return EvalResult(f=[float(a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)])


def zdt1_space(d: int = 6) -> DesignSpace:
    return DesignSpace([Variable(f"x{i}", 0.0, 1.0) for i in range(d)])


def zdt1(dd: dict[str, Any]) -> EvalResult:
    x = np.array(list(dd.values()), float)
    g = 1 + 9 * x[1:].mean()
    return EvalResult(f=[float(x[0]), float(g * (1 - np.sqrt(x[0] / g)))])


def zdt1_true_hv(ref: tuple[float, float] = (1.1, 1.1)) -> float:
    """Hypervolume of the analytic front f2 = 1 - sqrt(f1), f1 ∈ [0, 1] (numerical quadrature)."""
    f1 = np.linspace(0, 1, 200001)
    integrand = ref[1] - (1 - np.sqrt(f1))
    return float(np.trapezoid(integrand, f1) + (ref[0] - 1.0) * ref[1])
