r"""PINN problem definitions: residuals, IC/BC losses, optional hard constraints, exact solutions.

Each problem exposes ``lo``/``hi`` (space then time), ``u(model, X)`` (applies any hard-constraint
trial function), ``residual(model, X)``, ``constraint_losses(model)`` and ``exact(X)`` (NumPy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from aerosynapse.ml.autodiff import grad, second
from aerosynapse.physics.library import burgers_1d, convdiff_exact

FloatArr = NDArray[np.float64]


@dataclass
class PINNProblem:
    name: str
    lo: list[float]
    hi: list[float]
    periodic_dims: tuple[int, ...] = ()
    hard: bool = False
    n_constraint: int = 256
    seed: int = 0
    _cache: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    # --- interface -----------------------------------------------------------------
    def u(self, model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
        return model(X)

    def residual(self, model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclasses define the PDE residual")

    def constraint_losses(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {}

    def exact(self, X: FloatArr) -> FloatArr:
        raise NotImplementedError("subclasses define the exact solution")

    # --- helpers ---------------------------------------------------------------------
    def _rng(self, salt: int) -> np.random.Generator:
        return np.random.default_rng(self.seed * 1000 + salt)

    def tensor(self, a: FloatArr) -> torch.Tensor:
        return torch.tensor(a, dtype=torch.get_default_dtype())

    def eval_grid(self, n: int = 101) -> FloatArr:
        axes = [np.linspace(lo, hi, n) for lo, hi in zip(self.lo, self.hi, strict=True)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([m.ravel() for m in mesh])


@dataclass
class ConvDiff1D(PINNProblem):
    r""":math:`u_t + v u_x = D u_{xx}` on periodic [0,1) × [0,T]; Gaussian pulse (``physics.library``)."""

    pe: float = 10.0
    v: float = 1.0
    t_final: float = 0.5
    sigma0: float = 0.05
    name: str = "convdiff1d"
    lo: list[float] = field(default_factory=lambda: [0.0, 0.0])
    hi: list[float] = field(default_factory=lambda: [1.0, 0.5])
    periodic_dims: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        self.hi = [1.0, self.t_final]
        self.D = self.v / self.pe

    def u0(self, x: FloatArr) -> FloatArr:
        return convdiff_exact(x, 0.0, self.v, self.D, sigma0=self.sigma0)

    def u(self, model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
        if not self.hard:
            return model(X)
        x, t = X[:, :1], X[:, 1:2]
        s2 = self.sigma0**2
        u0 = sum(torch.exp(-((x - 0.25 - k) ** 2) / (2 * s2)) for k in (-1, 0, 1))
        return u0 + (t / self.t_final) * model(X)

    def residual(self, model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
        u = self.u(model, X)
        g = grad(u, X)
        return g[:, 1:2] + self.v * g[:, :1] - self.D * second(u, X, 0)

    def constraint_losses(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        if "ic" not in self._cache:
            x = np.linspace(0, 1, self.n_constraint, endpoint=False)
            self._cache["ic"] = self.tensor(np.column_stack([x, np.zeros_like(x)]))
            self._cache["u0"] = self.tensor(self.u0(x)[:, None])
        out = {}
        if not self.hard:
            out["ic"] = torch.mean((self.u(model, self._cache["ic"]) - self._cache["u0"]) ** 2)
        return out

    def conservation_loss(self, model: torch.nn.Module, n_t: int = 8, n_x: int = 64) -> torch.Tensor:
        """Squared deviation of ∫u dx from its initial value at ``n_t`` times (periodic mass)."""
        x = np.linspace(0, 1, n_x, endpoint=False)
        ts = np.linspace(0, self.t_final, n_t)
        X = self.tensor(np.column_stack([np.tile(x, n_t), np.repeat(ts, n_x)]))
        m = self.u(model, X).reshape(n_t, n_x).mean(dim=1)
        m0 = float(np.sqrt(2 * np.pi) * self.sigma0)  # exact mass of the initial pulse
        return torch.mean((m - m0) ** 2)

    def exact(self, X: FloatArr) -> FloatArr:
        return convdiff_exact(X[:, 0], X[:, 1], self.v, self.D, sigma0=self.sigma0)


@dataclass
class Heat1D(PINNProblem):
    r""":math:`u_t=\alpha u_{xx}`, u(0)=u(1)=0, u(x,0)=sin(πx). Hard: u = sin(πx) + t x(1-x) N."""

    alpha: float = 0.1
    t_final: float = 0.5
    name: str = "heat1d"
    lo: list[float] = field(default_factory=lambda: [0.0, 0.0])
    hi: list[float] = field(default_factory=lambda: [1.0, 0.5])

    def u(self, model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
        if not self.hard:
            return model(X)
        x, t = X[:, :1], X[:, 1:2]
        return torch.sin(math.pi * x) + t * x * (1 - x) * model(X)

    def residual(self, model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
        u = self.u(model, X)
        return grad(u, X)[:, 1:2] - self.alpha * second(u, X, 0)

    def constraint_losses(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        if self.hard:
            return {}
        if "ic" not in self._cache:
            n = self.n_constraint
            x = np.linspace(0, 1, n)
            t = np.linspace(0, self.t_final, n)
            self._cache["ic"] = self.tensor(np.column_stack([x, 0 * x]))
            self._cache["u0"] = self.tensor(np.sin(np.pi * x)[:, None])
            self._cache["bc"] = self.tensor(np.vstack([np.column_stack([0 * t, t]), np.column_stack([0 * t + 1, t])]))
        return {
            "ic": torch.mean((self.u(model, self._cache["ic"]) - self._cache["u0"]) ** 2),
            "bc": torch.mean(self.u(model, self._cache["bc"]) ** 2),
        }

    def exact(self, X: FloatArr) -> FloatArr:
        return np.asarray(np.exp(-self.alpha * np.pi**2 * X[:, 1]) * np.sin(np.pi * X[:, 0]))


@dataclass
class Poisson2D(PINNProblem):
    r""":math:`-\Delta u = 5\pi^2\sin\pi x\sin 2\pi y`, zero Dirichlet on the unit square."""

    name: str = "poisson2d"
    lo: list[float] = field(default_factory=lambda: [0.0, 0.0])
    hi: list[float] = field(default_factory=lambda: [1.0, 1.0])

    def u(self, model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
        if not self.hard:
            return model(X)
        x, y = X[:, :1], X[:, 1:2]
        return x * (1 - x) * y * (1 - y) * model(X)

    def residual(self, model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
        u = self.u(model, X)
        f = 5 * math.pi**2 * torch.sin(math.pi * X[:, :1]) * torch.sin(2 * math.pi * X[:, 1:2])
        return -(second(u, X, 0) + second(u, X, 1)) - f

    def constraint_losses(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        if self.hard:
            return {}
        if "bc" not in self._cache:
            s = np.linspace(0, 1, self.n_constraint // 4)
            z, o = 0 * s, 0 * s + 1
            pts = np.vstack([np.column_stack(p) for p in ((z, s), (o, s), (s, z), (s, o))])
            self._cache["bc"] = self.tensor(pts)
        return {"bc": torch.mean(self.u(model, self._cache["bc"]) ** 2)}

    def exact(self, X: FloatArr) -> FloatArr:
        return np.asarray(np.sin(np.pi * X[:, 0]) * np.sin(2 * np.pi * X[:, 1]))


@dataclass
class Burgers1D(PINNProblem):
    r"""Viscous Burgers on periodic [0, 2π) with the Cole–Hopf exact solution."""

    nu: float = 0.1
    t_final: float = 0.5
    name: str = "burgers1d"
    lo: list[float] = field(default_factory=lambda: [0.0, 0.0])
    hi: list[float] = field(default_factory=lambda: [2 * math.pi, 0.5])
    periodic_dims: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        self.hi = [2 * math.pi, self.t_final]
        self._p = burgers_1d(self.nu, self.t_final)

    def residual(self, model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
        u = self.u(model, X)
        g = grad(u, X)
        return g[:, 1:2] + u * g[:, :1] - self.nu * second(u, X, 0)

    def constraint_losses(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        if "ic" not in self._cache:
            x = np.linspace(0, 2 * np.pi, self.n_constraint, endpoint=False)
            self._cache["ic"] = self.tensor(np.column_stack([x, 0 * x]))
            assert self._p.exact is not None
            self._cache["u0"] = self.tensor(self._p.exact(x, 0.0)[:, None])
        return {"ic": torch.mean((self.u(model, self._cache["ic"]) - self._cache["u0"]) ** 2)}

    def exact(self, X: FloatArr) -> FloatArr:
        assert self._p.exact is not None
        return np.asarray(self._p.exact(X[:, 0], X[:, 1]))


@dataclass
class SteadyBoundaryLayer(PINNProblem):
    r""":math:`u_x = Pe^{-1}u_{xx}`, u(0)=0, u(1)=1. Hard: u = x + x(1-x)N(x)."""

    pe: float = 20.0
    name: str = "bl1d"
    lo: list[float] = field(default_factory=lambda: [0.0])
    hi: list[float] = field(default_factory=lambda: [1.0])

    def u(self, model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
        return X + X * (1 - X) * model(X) if self.hard else model(X)

    def residual(self, model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
        u = self.u(model, X)
        return grad(u, X) - second(u, X, 0) / self.pe

    def constraint_losses(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        if self.hard:
            return {}
        Xb = self.tensor(np.array([[0.0], [1.0]]))
        target = self.tensor(np.array([[0.0], [1.0]]))
        return {"bc": torch.mean((self.u(model, Xb) - target) ** 2)}

    def exact(self, X: FloatArr) -> FloatArr:
        x = X[:, 0]
        return np.asarray((np.exp(self.pe * (x - 1.0)) - np.exp(-self.pe)) / (1.0 - np.exp(-self.pe)))


@dataclass
class DiskPoisson(PINNProblem):
    r"""Complex-geometry demo: :math:`-\Delta u = 4` on the unit disk, u = 0 on r = 1, exact 1 - r².

    Hard enforcement uses the approximate distance function φ = 1 - x² - y² (exact SDF-like
    boundary representation): u = φ N. Penalty uses boundary points on the circle."""

    name: str = "disk_poisson"
    lo: list[float] = field(default_factory=lambda: [-1.0, -1.0])
    hi: list[float] = field(default_factory=lambda: [1.0, 1.0])

    def inside(self, X: FloatArr) -> NDArray[np.bool_]:
        return np.asarray(X[:, 0] ** 2 + X[:, 1] ** 2 < 1.0)

    def u(self, model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
        if not self.hard:
            return model(X)
        phi = 1 - X[:, :1] ** 2 - X[:, 1:2] ** 2
        return phi * model(X)

    def residual(self, model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
        u = self.u(model, X)
        return -(second(u, X, 0) + second(u, X, 1)) - 4.0

    def constraint_losses(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        if self.hard:
            return {}
        th = np.linspace(0, 2 * np.pi, self.n_constraint, endpoint=False)
        Xb = self.tensor(np.column_stack([np.cos(th), np.sin(th)]))
        return {"bc": torch.mean(self.u(model, Xb) ** 2)}

    def exact(self, X: FloatArr) -> FloatArr:
        return np.asarray(1 - X[:, 0] ** 2 - X[:, 1] ** 2)

    def eval_grid(self, n: int = 101) -> FloatArr:
        G = super().eval_grid(n)
        return G[self.inside(G)]
