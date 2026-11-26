"""Method of Manufactured Solutions (MMS) via SymPy.

Target scalar equation (1D or 2D, nondimensional):
    theta_t + u . grad(theta) = D lap(theta) - s theta + Q(x, t)
Given a chosen exact theta, Q is derived symbolically so that theta solves the forced PDE exactly.
Boundary data for the general Robin form  a*theta + b*d(theta)/dn = g  are generated too.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

x, y, t = sp.symbols("x y t", real=True)
SPACE = (x, y)

NumFn = Callable[..., np.ndarray]


def _lambdify(expr: sp.Expr, args: Sequence[sp.Symbol]) -> NumFn:
    f = sp.lambdify(args, expr, modules="numpy")

    def wrapped(*vals: np.ndarray | float) -> np.ndarray:
        arrs = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in vals])
        return np.broadcast_to(np.asarray(f(*arrs), dtype=float), arrs[0].shape).copy()

    return wrapped


@dataclass
class ManufacturedSolution:
    """Symbolic exact solution + derived forcing, with numpy callables f(x[, y], t)."""

    dim: int
    theta: sp.Expr
    source: sp.Expr
    velocity: tuple[sp.Expr, ...]
    D: sp.Expr
    s: sp.Expr
    args: tuple[sp.Symbol, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.args = (*SPACE[: self.dim], t)
        self.exact = _lambdify(self.theta, self.args)
        self.forcing = _lambdify(self.source, self.args)
        self.grad = tuple(_lambdify(sp.diff(self.theta, v), self.args) for v in SPACE[: self.dim])

    def robin_data(self, normal: Sequence[int], a: float, b: float) -> NumFn:
        """g = a*theta + b*(n . grad theta) for an axis-aligned outward normal, e.g. (1,0)."""
        dn = sum(int(n) * sp.diff(self.theta, v) for n, v in zip(normal, SPACE[: self.dim], strict=False))
        return _lambdify(a * self.theta + b * dn, self.args)


def residual_operator(
    theta: sp.Expr, velocity: Sequence[sp.Expr], D: sp.Expr | float, s: sp.Expr | float = 0
) -> sp.Expr:
    """theta_t + u.grad(theta) - D lap(theta) + s theta  (the forcing Q that makes theta exact)."""
    dim = len(velocity)
    space = SPACE[:dim]
    conv = sum(u * sp.diff(theta, v) for u, v in zip(velocity, space, strict=True))
    lap = sum(sp.diff(theta, v, 2) for v in space)
    return sp.simplify(sp.diff(theta, t) + conv - D * lap + s * theta)


def manufacture(
    theta: sp.Expr,
    velocity: Sequence[sp.Expr | float],
    D: sp.Expr | float,
    s: sp.Expr | float = 0,
) -> ManufacturedSolution:
    """Build a ManufacturedSolution for the scalar transport equation."""
    vel = tuple(sp.sympify(u) for u in velocity)
    Dn, sn = sp.sympify(D), sp.sympify(s)
    Q = residual_operator(theta, vel, Dn, sn)
    return ManufacturedSolution(dim=len(vel), theta=theta, source=Q, velocity=vel, D=Dn, s=sn)


def divergence(velocity: Sequence[sp.Expr]) -> sp.Expr:
    return sp.simplify(sum(sp.diff(u, v) for u, v in zip(velocity, SPACE, strict=False)))


def velocity_from_streamfunction(psi: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """u = dpsi/dy, v = -dpsi/dx (divergence-free by construction)."""
    return sp.diff(psi, y), -sp.diff(psi, x)


@dataclass
class ManufacturedNS:
    """Exact (u, v, p[, theta]) for the 2D incompressible NS (+ Boussinesq energy) with derived forcing.

    u_t + (u.grad)u = -grad p + lap(u)/Re + Ri theta e_y + f
    theta_t + u.grad theta = lap(theta)/Pe + Q
    """

    psi: sp.Expr
    p: sp.Expr
    theta: sp.Expr | None
    Re: sp.Expr
    Ri: sp.Expr
    Pe: sp.Expr | None
    u: sp.Expr = field(init=False)
    v: sp.Expr = field(init=False)
    fx: sp.Expr = field(init=False)
    fy: sp.Expr = field(init=False)
    Q: sp.Expr | None = field(init=False)

    def __post_init__(self) -> None:
        self.u, self.v = velocity_from_streamfunction(self.psi)
        u, v = self.u, self.v
        lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2)  # noqa: E731
        th = self.theta if self.theta is not None else sp.Integer(0)
        # no sp.simplify here: it costs minutes on the default MMS and is irrelevant for lambdify;
        # callers that need an exact-zero check (e.g. Taylor–Green) simplify explicitly
        self.fx = sp.diff(u, t) + u * sp.diff(u, x) + v * sp.diff(u, y) + sp.diff(self.p, x) - lap(u) / self.Re
        self.fy = (
            sp.diff(v, t) + u * sp.diff(v, x) + v * sp.diff(v, y) + sp.diff(self.p, y) - lap(v) / self.Re - self.Ri * th
        )
        self.Q = None
        if self.theta is not None:
            self.Q = sp.diff(th, t) + u * sp.diff(th, x) + v * sp.diff(th, y) - lap(th) / self.Pe
        args = (x, y, t)
        self.u_fn, self.v_fn, self.p_fn = (_lambdify(e, args) for e in (u, v, self.p))
        self.fx_fn, self.fy_fn = _lambdify(self.fx, args), _lambdify(self.fy, args)
        self.theta_fn = _lambdify(th, args) if self.theta is not None else None
        self.Q_fn = _lambdify(self.Q, args) if self.Q is not None else None
        self.theta_ms = (
            ManufacturedSolution(dim=2, theta=th, source=self.Q, velocity=(u, v), D=1 / self.Pe, s=sp.Integer(0))
            if self.theta is not None
            else None
        )


def manufacture_ns(
    psi: sp.Expr,
    p: sp.Expr,
    Re: float,
    theta: sp.Expr | None = None,
    Ri: float = 0.0,
    Pe: float | None = None,
) -> ManufacturedNS:
    """Velocity from a stream function (divergence-free by construction); forcing derived symbolically."""
    return ManufacturedNS(
        psi=psi,
        p=p,
        theta=theta,
        Re=sp.nsimplify(Re),
        Ri=sp.nsimplify(Ri),
        Pe=sp.nsimplify(Pe) if Pe is not None else None,
    )


def taylor_green(Re: float) -> ManufacturedNS:
    """Decaying Taylor–Green vortex on [0,1]^2 (exact unforced NS solution; zero normal velocity on walls)."""
    F = sp.exp(-2 * sp.pi**2 * t / sp.nsimplify(Re))
    psi = sp.sin(sp.pi * x) * sp.sin(sp.pi * y) / sp.pi * F
    p = (sp.cos(2 * sp.pi * x) + sp.cos(2 * sp.pi * y)) / 4 * F**2
    return manufacture_ns(psi, p, Re)


def mms_ns_default(Re: float = 50.0, Ri: float = 0.5, Pr: float = 0.71) -> ManufacturedNS:
    """Asymmetric, time-dependent NS + Boussinesq MMS. psi = 0 on the unit-square boundary (closed box, zero
    normal velocity), non-zero tangential wall velocity, non-trivial pressure and temperature."""
    psi = sp.sin(sp.pi * x) * sp.sin(sp.pi * y) / sp.pi * (1 + sp.sin(2 * t) / 2) + 2 * x**2 * (1 - x) * y * (
        1 - y
    ) ** 2 * sp.cos(t)
    p = sp.cos(sp.pi * x) * sp.sin(sp.pi * y / 2 + sp.Rational(1, 3)) * sp.exp(-t) + x * y / 3
    th = (
        sp.exp(-t / 2) * sp.cos(sp.pi * x / 2 + sp.Rational(1, 5)) * sp.sin(sp.pi * y + sp.Rational(1, 7))
        + x * y**2 / 2
    )
    return manufacture_ns(psi, p, Re, theta=th, Ri=Ri, Pe=Re * Pr)


# --- standard manufactured solutions used in verification ------------------------------------


def mms_1d_default(Pe: float = 20.0, a: float = 1.0, s: float = 0.0) -> ManufacturedSolution:
    """Smooth, non-polynomial, time-dependent 1D solution (all derivatives non-trivial)."""
    th = sp.exp(-t) * sp.sin(sp.pi * x + sp.Rational(1, 3)) + sp.Rational(1, 2) * sp.cos(2 * x) * sp.cos(t)
    return manufacture(th, [a], 1 / sp.nsimplify(Pe), s)


def mms_2d_default(Pe: float = 50.0, s: float = 0.0) -> ManufacturedSolution:
    """2D solution advected by the divergence-free cellular flow psi = sin(pi x) sin(pi y)/pi."""
    psi = sp.sin(sp.pi * x) * sp.sin(sp.pi * y) / sp.pi
    u, v = velocity_from_streamfunction(psi)
    th = (
        sp.exp(-t / 2) * sp.cos(sp.pi * x / 2 + sp.Rational(1, 5)) * sp.sin(sp.pi * y + sp.Rational(1, 7))
        + x * y**2 / 2
    )
    return manufacture(th, [u, v], 1 / sp.nsimplify(Pe), s)
