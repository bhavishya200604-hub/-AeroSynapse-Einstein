r"""Predefined Level-1 problems with analytical / manufactured solutions (spec Section 5).

Each factory returns a :class:`PDEProblem`. The analytical solutions are exact solutions of the
continuous PDE; they are the "numerical truth" for verification.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aerosynapse.physics.bc import Dirichlet, Periodic
from aerosynapse.physics.problem import Parameter, PDEProblem
from aerosynapse.registry import PHYSICS

FloatArr = NDArray[np.float64]
PI = np.pi


@PHYSICS.register("heat1d")
def heat_1d(alpha: float = 0.1, t_final: float = 0.5, mode: int = 1) -> PDEProblem:
    r"""Heat equation :math:`T_t = \alpha T_{xx}` on [0,1], T=0 at both ends,
    :math:`T(x,0)=\sin(m\pi x)`. Exact: :math:`e^{-\alpha (m\pi)^2 t}\sin(m\pi x)`."""
    k = mode * PI
    return PDEProblem(
        name="heat1d",
        equation="T_t = alpha T_xx",
        domain=((0.0, 1.0),),
        t_final=t_final,
        parameters={"alpha": Parameter("alpha", alpha, "m^2/s", 1e-8, 1e3, "thermal diffusivity")},
        bcs={"left": Dirichlet(0.0), "right": Dirichlet(0.0)},
        initial_condition=lambda x: np.sin(k * x),
        exact=lambda x, t: np.exp(-alpha * k**2 * t) * np.sin(k * x),
    )


@PHYSICS.register("heat2d")
def heat_2d(alpha: float = 0.05, t_final: float = 0.5) -> PDEProblem:
    r"""2D heat on the unit square, homogeneous Dirichlet; exact
    :math:`e^{-2\alpha\pi^2 t}\sin\pi x\sin\pi y`."""
    return PDEProblem(
        name="heat2d",
        equation="T_t = alpha (T_xx + T_yy)",
        domain=((0.0, 1.0), (0.0, 1.0)),
        t_final=t_final,
        parameters={"alpha": Parameter("alpha", alpha, "m^2/s", 1e-8, 1e3)},
        bcs={s: Dirichlet(0.0) for s in ("left", "right", "bottom", "top")},
        initial_condition=lambda x, y: np.sin(PI * x) * np.sin(PI * y),
        exact=lambda x, y, t: np.exp(-2 * alpha * PI**2 * t) * np.sin(PI * x) * np.sin(PI * y),
    )


def convdiff_exact(
    x: FloatArr,
    t: float | FloatArr,
    v: float,
    D: float,
    x0: float = 0.25,
    sigma0: float = 0.05,
    periods: int = 3,
) -> FloatArr:
    r"""Exact solution of :math:`u_t + v u_x = D u_{xx}` on the periodic unit interval for a
    Gaussian initial pulse (method of images, ``periods`` images each side; error < 1e-12 for the
    default widths):

    .. math:: u = \sum_k \frac{\sigma_0}{\sigma(t)}
              \exp\!\Big(-\frac{(x - x_0 - vt - k)^2}{2\sigma(t)^2}\Big),\qquad
              \sigma(t)^2 = \sigma_0^2 + 2Dt.
    """
    s2 = sigma0**2 + 2.0 * D * np.asarray(t, dtype=float)
    out = np.zeros(np.broadcast(x, t).shape)
    for k in range(-periods, periods + 1):
        out = out + np.sqrt(sigma0**2 / s2) * np.exp(-((x - x0 - v * t - k) ** 2) / (2.0 * s2))
    return out


@PHYSICS.register("convdiff1d")
def convdiff_1d(pe: float = 10.0, t_final: float = 0.5, sigma0: float = 0.05) -> PDEProblem:
    r"""Transient convection-diffusion :math:`u_t + v u_x = D u_{xx}` on periodic [0,1) with v = 1,
    :math:`D = 1/Pe`. Pulse of width ``sigma0`` advected and spread."""
    v, D = 1.0, 1.0 / pe
    return PDEProblem(
        name="convdiff1d",
        equation="u_t + v u_x = D u_xx + s",
        domain=((0.0, 1.0),),
        t_final=t_final,
        parameters={"v": Parameter("v", v, "m/s"), "D": Parameter("D", D, "m^2/s", 1e-12, 1e6)},
        bcs={"left": Periodic(), "right": Periodic()},
        initial_condition=lambda x: convdiff_exact(x, 0.0, v, D, sigma0=sigma0),
        exact=lambda x, t: convdiff_exact(x, t, v, D, sigma0=sigma0),
    )


@PHYSICS.register("convdiff_steady")
def convdiff_steady(pe: float = 10.0) -> PDEProblem:
    r"""Steady boundary-layer problem :math:`v u_x = D u_{xx}`, u(0)=0, u(1)=1. Exact
    :math:`u=(e^{Pe x}-1)/(e^{Pe}-1)` (boundary layer of thickness ~1/Pe at x=1)."""
    D = 1.0 / pe

    def exact(x: FloatArr, t: float = 0.0) -> FloatArr:
        # overflow-safe form of (exp(Pe x) - 1) / (exp(Pe) - 1)
        return np.asarray((np.exp(pe * (x - 1.0)) - np.exp(-pe)) / (1.0 - np.exp(-pe)))

    return PDEProblem(
        name="convdiff_steady",
        equation="v u_x = D u_xx",
        domain=((0.0, 1.0),),
        parameters={"v": Parameter("v", 1.0, "m/s"), "D": Parameter("D", D, "m^2/s", 1e-12, 1e6)},
        bcs={"left": Dirichlet(0.0), "right": Dirichlet(1.0)},
        exact=exact,
    )


@PHYSICS.register("burgers1d")
def burgers_1d(nu: float = 0.05, t_final: float = 0.5) -> PDEProblem:
    r"""Viscous Burgers :math:`u_t + u u_x = \nu u_{xx}` on [0, 2π) periodic with the exact
    Cole–Hopf "sawtooth" solution :math:`u = -2\nu\phi_x/\phi + 4`, where
    :math:`\phi=\sum_k \exp(-(x-4t-2\pi k)^2/(4\nu(t+1)))` (Whitham-type benchmark)."""

    def exact(x: FloatArr, t: float | FloatArr) -> FloatArr:
        x = np.asarray(x, dtype=float)
        tt = np.asarray(t, dtype=float) + 1.0
        num = np.zeros(np.broadcast(x, tt).shape)
        den = np.zeros_like(num)
        for k in range(-4, 5):
            a = x - 4.0 * (tt - 1.0) - 2.0 * PI * k
            e = np.exp(-(a**2) / (4.0 * nu * tt))
            num = num + (-a / (2.0 * nu * tt)) * e
            den = den + e
        return np.asarray(-2.0 * nu * num / den + 4.0)

    return PDEProblem(
        name="burgers1d",
        equation="u_t + u u_x = nu u_xx",
        domain=((0.0, 2 * PI),),
        t_final=t_final,
        parameters={"nu": Parameter("nu", nu, "m^2/s", 1e-8, 1e3), "u_ref": Parameter("u_ref", 4.0)},
        bcs={"left": Periodic(), "right": Periodic()},
        initial_condition=lambda x: exact(x, 0.0),
        exact=exact,
    )


@PHYSICS.register("wave1d")
def wave_1d(c: float = 1.0, t_final: float = 1.0, mode: int = 1) -> PDEProblem:
    r"""Wave equation :math:`u_{tt}=c^2u_{xx}` on [0,1], fixed ends, standing wave
    :math:`u=\sin(m\pi x)\cos(m\pi c t)`, :math:`u_t(x,0)=0`."""
    k = mode * PI
    return PDEProblem(
        name="wave1d",
        equation="u_tt = c^2 u_xx",
        domain=((0.0, 1.0),),
        t_final=t_final,
        parameters={"c": Parameter("c", c, "m/s", 0.0, 1e6)},
        bcs={"left": Dirichlet(0.0), "right": Dirichlet(0.0)},
        initial_condition=lambda x: np.sin(k * x),
        exact=lambda x, t: np.sin(k * x) * np.cos(k * c * t),
    )


@PHYSICS.register("poisson2d")
def poisson_2d() -> PDEProblem:
    r"""Poisson :math:`-\nabla^2u=f` on the unit square with manufactured
    :math:`u=\sin(\pi x)\sin(2\pi y)`, :math:`f=5\pi^2u`, homogeneous Dirichlet."""

    def exact(x: FloatArr, y: FloatArr) -> FloatArr:
        return np.asarray(np.sin(PI * x) * np.sin(2 * PI * y))

    return PDEProblem(
        name="poisson2d",
        equation="-lap u = f",
        domain=((0.0, 1.0), (0.0, 1.0)),
        bcs={s: Dirichlet(0.0) for s in ("left", "right", "bottom", "top")},
        source=lambda x, y: 5 * PI**2 * exact(x, y),
        exact=exact,
    )
