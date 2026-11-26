"""Exact solutions the 3D solver is verified against.

* Sod shock tube: exact Riemann solution (Toro, ch. 4)
* oblique shock on a wedge: theta-beta-Mach relation
* supersonic cone: Taylor-Maccoll equation, integrated from the shock to the surface
* prolate spheroid in axial potential flow: peak surface speed and surface pressure (Lamb)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

G = 1.4


def sod(x: NDArray[np.float64], t: float, left: tuple[float, float, float] = (1.0, 0.0, 1.0), right: tuple[float, float, float] = (0.125, 0.0, 0.1), x0: float = 0.5) -> NDArray[np.float64]:
    """(rho, u, p) at x and time t."""
    rl, ul, pl = left
    rr, ur, pr = right
    cl, cr = np.sqrt(G * pl / rl), np.sqrt(G * pr / rr)

    def f(ps: float, r: float, p: float, c: float) -> float:
        if ps > p:
            A, B = 2 / ((G + 1) * r), (G - 1) / (G + 1) * p
            return float((ps - p) * np.sqrt(A / (ps + B)))
        return float(2 * c / (G - 1) * ((ps / p) ** ((G - 1) / (2 * G)) - 1))

    ps = brentq(lambda q: f(q, rl, pl, cl) + f(q, rr, pr, cr) + ur - ul, 1e-8, 100)
    us = 0.5 * (ul + ur) + 0.5 * (f(ps, rr, pr, cr) - f(ps, rl, pl, cl))
    rls = rl * (ps / pl) ** (1 / G)
    cls = cl * (ps / pl) ** ((G - 1) / (2 * G))
    rrs = rr * (ps / pr + (G - 1) / (G + 1)) / ((G - 1) / (G + 1) * ps / pr + 1)
    S = ur + cr * np.sqrt((G + 1) / (2 * G) * ps / pr + (G - 1) / (2 * G))
    out = np.zeros((3, len(x)))
    for i, s in enumerate((np.asarray(x) - x0) / t):
        if s < ul - cl:
            out[:, i] = rl, ul, pl
        elif s < us - cls:
            u = 2 / (G + 1) * (cl + (G - 1) / 2 * ul + s)
            c = 2 / (G + 1) * (cl + (G - 1) / 2 * (ul - s))
            out[:, i] = rl * (c / cl) ** (2 / (G - 1)), u, pl * (c / cl) ** (2 * G / (G - 1))
        elif s < us:
            out[:, i] = rls, us, ps
        elif s < S:
            out[:, i] = rrs, us, ps
        else:
            out[:, i] = rr, ur, pr
    return out


def oblique_shock(mach: float, theta_deg: float) -> dict[str, float]:
    """Weak oblique shock on a wedge: shock angle and pressure coefficient behind it."""
    th = np.radians(theta_deg)

    def g(b: float) -> float:
        return float(np.tan(th) - 2 / np.tan(b) * (mach**2 * np.sin(b) ** 2 - 1) / (mach**2 * (G + np.cos(2 * b)) + 2))

    mu = np.arcsin(1 / mach)
    bmax = mu + 0.5 * (np.pi / 2 - mu)
    grid = np.linspace(mu + 1e-6, np.pi / 2 - 1e-6, 2000)
    vals = np.array([g(b) for b in grid])
    i = int(np.argmax(np.sign(vals[:-1]) != np.sign(vals[1:])))
    b = brentq(g, grid[i], grid[i + 1])
    mn = mach * np.sin(b)
    p21 = 1 + 2 * G / (G + 1) * (mn**2 - 1)
    _ = bmax
    return {"beta_deg": float(np.degrees(b)), "p2_p1": float(p21), "cp": float((p21 - 1) / (0.5 * G * mach**2))}


def taylor_maccoll(mach: float, cone_deg: float) -> dict[str, float]:
    """Supersonic flow over a sharp cone at zero incidence: shock angle and surface pressure coefficient."""
    tc = np.radians(cone_deg)

    def surface_angle(beta: float) -> tuple[float, float]:
        mn = mach * np.sin(beta)
        m2 = np.sqrt((1 + (G - 1) / 2 * mn**2) / (G * mn**2 - (G - 1) / 2)) / np.sin(beta - _deflection(beta))
        d = _deflection(beta)
        V = (2 / ((G - 1) * m2**2) + 1) ** -0.5
        vr, vt = V * np.cos(beta - d), -V * np.sin(beta - d)

        def rhs(th: float, y: NDArray[np.float64]) -> list[float]:
            a, b = y
            k = (G - 1) / 2 * (1 - a**2 - b**2)
            return [b, (b**2 * a - k * (2 * a + b / np.tan(th))) / (k - b**2)]

        ev = lambda th, y: y[1]  # noqa: E731
        ev.terminal = True  # type: ignore[attr-defined]
        sol = solve_ivp(rhs, (beta, 1e-4), [vr, vt], events=ev, rtol=1e-10, atol=1e-12, max_step=1e-3)
        th_s = float(sol.t_events[0][0]) if len(sol.t_events[0]) else 0.0
        return th_s, float(sol.y_events[0][0][0]) if len(sol.t_events[0]) else 0.0

    def _deflection(beta: float) -> float:
        return float(np.arctan(2 / np.tan(beta) * (mach**2 * np.sin(beta) ** 2 - 1) / (mach**2 * (G + np.cos(2 * beta)) + 2)))

    mu = np.arcsin(1 / mach)
    b = brentq(lambda q: surface_angle(q)[0] - tc, mu + 1e-4, np.radians(80))
    _, vs = surface_angle(b)
    # isentropic from the shock to the surface; total pressure behind the shock
    mn = mach * np.sin(b)
    p2_p1 = 1 + 2 * G / (G + 1) * (mn**2 - 1)
    m2n = np.sqrt((1 + (G - 1) / 2 * mn**2) / (G * mn**2 - (G - 1) / 2))
    m2 = m2n / np.sin(b - _deflection(b))
    p02_p2 = (1 + (G - 1) / 2 * m2**2) ** (G / (G - 1))
    ms2 = 2 / (G - 1) * vs**2 / (1 - vs**2)
    ps_p1 = p2_p1 * p02_p2 / (1 + (G - 1) / 2 * ms2) ** (G / (G - 1))
    return {"shock_deg": float(np.degrees(b)), "cp": float((ps_p1 - 1) / (0.5 * G * mach**2)), "surface_mach": float(np.sqrt(ms2))}


def spheroid(fineness: float) -> dict[str, float]:
    """Prolate spheroid, axial potential flow: peak surface speed / free stream and the minimum Cp."""
    e = np.sqrt(1 - 1 / fineness**2)
    a0 = 2 * (1 - e**2) / e**3 * (0.5 * np.log((1 + e) / (1 - e)) - e)
    v = 2 / (2 - a0)
    return {"v_max": float(v), "cp_min": float(1 - v**2)}


def spheroid_cp(x_over_a: NDArray[np.float64], fineness: float) -> NDArray[np.float64]:
    """Surface Cp along a prolate spheroid (incompressible): u_s = (1+k) U cos(angle between surface and axis)."""
    e = np.sqrt(1 - 1 / fineness**2)
    a0 = 2 * (1 - e**2) / e**3 * (0.5 * np.log((1 + e) / (1 - e)) - e)
    k = a0 / (2 - a0)
    x = np.asarray(x_over_a)
    # surface slope dr/dx for r = b sqrt(1 - x^2), x in units of a
    b = 1 / fineness
    drdx = -b * x / np.sqrt(np.maximum(1 - x**2, 1e-12))
    cos = 1 / np.sqrt(1 + drdx**2)
    return np.asarray(1 - ((1 + k) * cos) ** 2)
