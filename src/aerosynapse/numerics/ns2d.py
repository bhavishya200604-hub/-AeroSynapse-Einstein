r"""2D incompressible Navier–Stokes on a uniform MAC (staggered) grid, Chorin projection.

.. math:: u^* = u^n + \Delta t\,[-\nabla\cdot(u u) + \nu\nabla^2u + f],\quad
          \nabla^2 p = \nabla\cdot u^*/\Delta t,\quad u^{n+1}=u^*-\Delta t\nabla p

(ρ = 1, p kinematic pressure). Second-order central differences in space, first-order in time.
Steady states are reached by time marching (stop when max|Δu|/Δt < tol).

Storage: ``U`` (nx+1, ny+2): normal faces x = i h with ghost rows; ``V`` (nx+2, ny+1); ``P`` (nx, ny).

Boundary conditions per side (``left``/``right``/``bottom``/``top``), see :data:`FLUID_BC_CATALOGUE`:
``wall`` (no-slip, optional tangential wall speed), ``slip`` / ``symmetry`` (zero normal velocity,
zero tangential shear), ``inlet`` (prescribed velocity profile), ``pressure_inlet`` (prescribed
pressure, zero-gradient velocity), ``outlet`` (p = 0, zero-gradient velocity), ``periodic`` (x pair
only), ``farfield`` (prescribed free stream on all components — valid only far from bodies).
A solid mask (staircase immersed boundary, direct forcing) allows bodies such as a cylinder.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from aerosynapse.exceptions import ConfigurationError
from aerosynapse.numerics.guards import check_field
from aerosynapse.registry import SOLVERS

FloatArr = NDArray[np.float64]
BoolArr = NDArray[np.bool_]
VALID = ("wall", "slip", "symmetry", "inlet", "pressure_inlet", "outlet", "periodic", "farfield")


@dataclass
class FluidBC:
    kind: str
    value: float | Callable[[FloatArr], FloatArr] = 0.0  # wall: tangential speed; inlet: normal-velocity profile
    pressure: float = 0.0  # pressure_inlet / outlet value
    tangential: float = 0.0  # inlet / farfield tangential component

    def __post_init__(self) -> None:
        if self.kind not in VALID:
            raise ConfigurationError(f"unknown fluid BC {self.kind!r}; valid {VALID}")

    def profile(self, s: FloatArr) -> FloatArr:
        return np.asarray(self.value(s)) if callable(self.value) else np.full_like(s, float(self.value))


@dataclass
class NSResult:
    U: FloatArr
    V: FloatArr
    P: FloatArr
    h: float
    t: float
    steps: int
    converged: bool
    residual_history: list[float] = field(default_factory=list)
    divergence_max: float = 0.0
    body_force: tuple[float, float] = (0.0, 0.0)

    def centre_velocity(self) -> tuple[FloatArr, FloatArr]:
        uc = 0.5 * (self.U[:-1, 1:-1] + self.U[1:, 1:-1])
        vc = 0.5 * (self.V[1:-1, :-1] + self.V[1:-1, 1:])
        return uc, vc


class NSSolver:
    def __init__(
        self,
        nx: int,
        ny: int,
        lx: float,
        ly: float,
        nu: float,
        bcs: dict[str, FluidBC],
        body_force: tuple[float, float] = (0.0, 0.0),
        solid: BoolArr | None = None,
    ) -> None:
        h = lx / nx
        if abs(ly / ny - h) > 1e-12 * h:
            raise ConfigurationError("uniform square cells required (lx/nx == ly/ny)")
        if nu <= 0:
            raise ConfigurationError("viscosity must be positive")
        for s in ("left", "right", "bottom", "top"):
            if s not in bcs:
                raise ConfigurationError(f"missing fluid BC on side {s}")
        if (bcs["left"].kind == "periodic") != (bcs["right"].kind == "periodic"):
            raise ConfigurationError("periodic BC must be set on both left and right")
        if "periodic" in (bcs["bottom"].kind, bcs["top"].kind):
            raise ConfigurationError("periodicity supported in x only")
        self.nx, self.ny, self.h, self.nu, self.bcs = nx, ny, h, nu, bcs
        self.lx, self.ly = lx, ly
        self.periodic = bcs["left"].kind == "periodic"
        self.f = body_force
        self.solid = solid if solid is not None else np.zeros((nx, ny), bool)
        self.yc = np.asarray((np.arange(ny) + 0.5) * h, dtype=np.float64)
        self.xc = np.asarray((np.arange(nx) + 0.5) * h, dtype=np.float64)
        self._build_poisson()
        self.U = np.zeros((nx + 1, ny + 2))
        self.V = np.zeros((nx + 2, ny + 1))
        self.P = np.zeros((nx, ny))
        self._apply_bc(self.U, self.V)
        # faces touching solid cells (direct forcing)
        sm = self.solid
        su = np.zeros((nx + 1, ny), bool)
        su[1:-1] = sm[:-1] | sm[1:]
        su[0] |= sm[0]
        su[-1] |= sm[-1]
        sv = np.zeros((nx, ny + 1), bool)
        sv[:, 1:-1] = sm[:, :-1] | sm[:, 1:]
        sv[:, 0] |= sm[:, 0]
        sv[:, -1] |= sm[:, -1]
        self.su, self.sv = su, sv

    # ------------------------------------------------------------------ pressure Poisson
    def _pressure_dirichlet(self, side: str) -> bool:
        return self.bcs[side].kind in ("outlet", "pressure_inlet")

    def _build_poisson(self) -> None:
        nx, ny = self.nx, self.ny
        N = nx * ny
        idx = np.arange(N).reshape(nx, ny)
        rows, cols, vals = [], [], []
        self.p_dir_rhs = np.zeros(N)
        any_dir = any(self._pressure_dirichlet(s) for s in ("left", "right", "bottom", "top"))
        for i in range(nx):
            for j in range(ny):
                r = idx[i, j]
                diag = 0.0
                for di, dj, side, at_edge in (
                    (-1, 0, "left", i == 0),
                    (1, 0, "right", i == nx - 1),
                    (0, -1, "bottom", j == 0),
                    (0, 1, "top", j == ny - 1),
                ):
                    if not at_edge:
                        rows.append(r)
                        cols.append(idx[i + di, j + dj])
                        vals.append(1.0)
                        diag -= 1.0
                    elif side in ("left", "right") and self.periodic:
                        rows.append(r)
                        cols.append(idx[(i + di) % nx, j])
                        vals.append(1.0)
                        diag -= 1.0
                    elif self._pressure_dirichlet(side):
                        diag -= 2.0  # ghost = 2 p_b - p_P
                        self.p_dir_rhs[r] -= 2.0 * self.bcs[side].pressure
                rows.append(r)
                cols.append(r)
                vals.append(diag)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N)) / self.h**2
        self.p_dir_rhs /= self.h**2
        self.gauge = not any_dir
        if self.gauge:  # pure Neumann / periodic: pin cell 0 (compatible for closed or periodic domains)
            A = A.tolil()
            A.rows[0], A.data[0] = [0], [1.0]
            A = A.tocsr()
        self.lu = spla.splu(A.tocsc())

    # ------------------------------------------------------------------ boundary conditions
    def _apply_bc(self, U: FloatArr, V: FloatArr) -> None:
        b, h = self.bcs, self.h
        ys, xs = self.yc, self.xc
        # normal velocities on x-boundaries
        for side, i in (("left", 0), ("right", -1)):
            k = b[side].kind
            if k in ("wall", "slip", "symmetry"):
                U[i, 1:-1] = 0.0
            elif k in ("inlet", "farfield"):
                U[i, 1:-1] = b[side].profile(ys)
            elif k in ("outlet", "pressure_inlet"):
                pass  # computed by momentum + projection
        if self.periodic:
            U[-1] = U[0]
        for side, j in (("bottom", 0), ("top", -1)):
            k = b[side].kind
            if k in ("wall", "slip", "symmetry"):
                V[1:-1, j] = 0.0
            elif k in ("inlet", "farfield"):
                V[1:-1, j] = b[side].profile(xs)
        # tangential ghosts: u on bottom/top, v on left/right
        for side, g, inner in (("bottom", 0, 1), ("top", -1, -2)):
            bc = b[side]
            if bc.kind == "wall":
                U[:, g] = 2 * float(bc.value if not callable(bc.value) else 0.0) - U[:, inner]
            elif bc.kind in ("slip", "symmetry", "outlet", "pressure_inlet"):
                U[:, g] = U[:, inner]
            elif bc.kind in ("inlet", "farfield"):
                U[:, g] = 2 * bc.tangential - U[:, inner]
        if self.periodic:
            V[0] = V[-2]
            V[-1] = V[1]
        else:
            for side, g, inner in (("left", 0, 1), ("right", -1, -2)):
                bc = b[side]
                if bc.kind == "wall":
                    V[g] = 2 * float(bc.value if not callable(bc.value) else 0.0) - V[inner]
                elif bc.kind in ("slip", "symmetry", "outlet", "pressure_inlet"):
                    V[g] = V[inner]
                elif bc.kind in ("inlet", "farfield"):
                    V[g] = 2 * bc.tangential - V[inner]
        _ = h

    # ------------------------------------------------------------------ time step
    def stable_dt(self, safety: float = 0.5) -> float:
        h, nu = self.h, self.nu
        umax = max(float(np.abs(self.U).max()), float(np.abs(self.V).max()), 1e-12)
        return safety * min(0.25 * h * h / nu, h / umax, 2 * nu / umax**2)

    def _momentum(self, U: FloatArr, V: FloatArr) -> tuple[FloatArr, FloatArr]:
        h, nu = self.h, self.nu
        # extend U in x by one column each side (periodic wrap or zero-gradient)
        Ue = np.concatenate([U[-2:-1], U, U[1:2]], 0) if self.periodic else np.concatenate([U[:1], U, U[-1:]], 0)
        # --- u-momentum on all faces i = 0..nx, rows 1..ny
        uc = 0.5 * (Ue[:-1, 1:-1] + Ue[1:, 1:-1])  # cell-centre u (nx+2 cells incl. ext)
        duu = (uc[1:] ** 2 - uc[:-1] ** 2) / h  # (nx+1, ny)
        vcorner = 0.5 * (V[:-1] + V[1:])  # corners x = i h (ghost columns already wrapped / mirrored)
        ucorner = 0.5 * (U[:, :-1] + U[:, 1:])  # (nx+1, ny+1)
        duv = (ucorner[:, 1:] * vcorner[:, 1:] - ucorner[:, :-1] * vcorner[:, :-1]) / h
        lapu = (Ue[2:, 1:-1] - 2 * U[:, 1:-1] + Ue[:-2, 1:-1]) / h**2 + (U[:, 2:] - 2 * U[:, 1:-1] + U[:, :-2]) / h**2
        Fu = -duu - duv + nu * lapu + self.f[0]
        # --- v-momentum on faces j = 0..ny, columns 1..nx
        vc = 0.5 * (V[1:-1, :-1] + V[1:-1, 1:])  # (nx, ny)
        vce = np.concatenate([vc[:, :1], vc, vc[:, -1:]], 1)  # zero-gradient ext in y
        dvv = (vce[:, 1:] ** 2 - vce[:, :-1] ** 2) / h  # (nx, ny+1)
        ucr = 0.5 * (U[:, :-1] + U[:, 1:])  # corners x=i h, y=j h (nx+1, ny+1)
        vcr = 0.5 * (V[:-1] + V[1:])  # (nx+1, ny+1)
        uv = ucr * vcr
        duvx = (uv[1:] - uv[:-1]) / h  # (nx, ny+1)
        Ve = np.concatenate([V[:, :1], V, V[:, -1:]], 1)
        lapv = (V[2:] - 2 * V[1:-1] + V[:-2]) / h**2 + (Ve[1:-1, 2:] - 2 * V[1:-1] + Ve[1:-1, :-2]) / h**2
        Fv = -dvv - duvx + nu * lapv + self.f[1]
        return Fu, Fv

    def step(self, dt: float) -> tuple[float, float, tuple[float, float]]:
        U, V, h = self.U, self.V, self.h
        Fu, Fv = self._momentum(U, V)
        Us, Vs = U.copy(), V.copy()
        Us[:, 1:-1] += dt * Fu
        Vs[1:-1] += dt * Fv
        # direct forcing on the body: momentum removed = force on the body (per unit depth, rho = 1)
        fx = float(np.sum(Us[:, 1:-1][self.su]) * h * h / dt)
        fy = float(np.sum(Vs[1:-1][self.sv]) * h * h / dt)
        Us[:, 1:-1][self.su] = 0.0
        Vs[1:-1][self.sv] = 0.0
        self._apply_bc(Us, Vs)
        div = ((Us[1:, 1:-1] - Us[:-1, 1:-1]) + (Vs[1:-1, 1:] - Vs[1:-1, :-1])) / h
        rhs = div.ravel() / dt + self.p_dir_rhs  # A p = div/dt - (Dirichlet ghost terms)
        if self.gauge:
            rhs[0] = 0.0
        P = self.lu.solve(rhs).reshape(self.nx, self.ny)
        Un, Vn = Us, Vs
        Un[1:-1, 1:-1] -= dt * (P[1:] - P[:-1]) / h
        Vn[1:-1, 1:-1] -= dt * (P[:, 1:] - P[:, :-1]) / h
        for side, i, pi in (("left", 0, 0), ("right", -1, -1)):
            k = self.bcs[side].kind
            if k in ("outlet", "pressure_inlet"):
                pb = self.bcs[side].pressure
                sgn = -1.0 if side == "left" else 1.0
                Un[i, 1:-1] -= dt * sgn * (pb - P[pi]) / (0.5 * h)
        for side, j, pj in (("bottom", 0, 0), ("top", -1, -1)):
            if self.bcs[side].kind in ("outlet", "pressure_inlet"):
                pb = self.bcs[side].pressure
                sgn = -1.0 if side == "bottom" else 1.0
                Vn[1:-1, j] -= dt * sgn * (pb - P[:, pj]) / (0.5 * h)
        if self.periodic:
            Un[0, 1:-1] -= dt * (P[0] - P[-1]) / h
            Un[-1] = Un[0]
        Un[:, 1:-1][self.su] = 0.0
        Vn[1:-1][self.sv] = 0.0
        self._apply_bc(Un, Vn)
        change = max(float(np.abs(Un - U).max()), float(np.abs(Vn - V).max())) / dt
        self.U, self.V, self.P = Un, Vn, P
        return change, float(np.abs(self.divergence()).max()), (fx, fy)

    def divergence(self) -> FloatArr:
        d = ((self.U[1:, 1:-1] - self.U[:-1, 1:-1]) + (self.V[1:-1, 1:] - self.V[1:-1, :-1])) / self.h
        return np.asarray(np.where(self.solid, 0.0, d))

    def run(
        self, t_end: float, tol: float = 1e-6, dt: float | None = None, max_steps: int = 500_000, check_every: int = 50
    ) -> NSResult:
        t, k, hist, conv = 0.0, 0, [], False
        force = (0.0, 0.0)
        while t < t_end and k < max_steps:
            d = dt or self.stable_dt()
            change, _, force = self.step(d)
            t += d
            k += 1
            if k % check_every == 0:
                check_field(self.U, "u", k, bound=1e4)
                hist.append(change)
                if change < tol:
                    conv = True
                    break
        return NSResult(
            self.U.copy(),
            self.V.copy(),
            self.P.copy(),
            self.h,
            t,
            k,
            conv,
            hist,
            float(np.abs(self.divergence()).max()),
            force,
        )


@SOLVERS.register("ns2d_projection")
def lid_driven_cavity(re: float, n: int = 64, t_end: float = 60.0, tol: float = 1e-5) -> NSResult:
    """Unit cavity, lid speed 1 (top wall), ν = 1/Re."""
    bcs = {"left": FluidBC("wall"), "right": FluidBC("wall"), "bottom": FluidBC("wall"), "top": FluidBC("wall", 1.0)}
    return NSSolver(n, n, 1.0, 1.0, 1.0 / re, bcs).run(t_end, tol)


def cavity_centerlines(r: NSResult) -> tuple[FloatArr, FloatArr, FloatArr, FloatArr]:
    """u(x=0.5, y) and v(x, y=0.5) including wall values (n even)."""
    nx = r.U.shape[0] - 1
    ny = r.U.shape[1] - 2
    h = r.h
    u_mid = r.U[nx // 2, 1:-1]
    y = np.concatenate([[0.0], (np.arange(ny) + 0.5) * h, [1.0]])
    u = np.concatenate([[0.0], u_mid, [1.0]])
    v_mid = r.V[1:-1, ny // 2]
    x = np.concatenate([[0.0], (np.arange(nx) + 0.5) * h, [1.0]])
    v = np.concatenate([[0.0], v_mid, [0.0]])
    return y, u, x, v


FLUID_BC_CATALOGUE = {
    "wall": {
        "definition": "u = u_wall (no-slip)",
        "implementation": "normal face = 0; tangential ghost 2u_w - u_in",
        "compatibility": "closed domains need a pressure gauge",
        "failure": "corner singularity at moving lid",
    },
    "slip": {
        "definition": "u.n = 0, d(u.t)/dn = 0",
        "implementation": "normal face 0, mirrored tangential ghost",
        "compatibility": "free-slip only for inviscid-like boundaries",
        "failure": "misused at real walls",
    },
    "symmetry": {
        "definition": "as slip, on a symmetry plane",
        "implementation": "identical to slip",
        "compatibility": "flow must be symmetric",
        "failure": "suppresses asymmetric modes (vortex shedding)",
    },
    "inlet": {
        "definition": "u = u_in(s)",
        "implementation": "Dirichlet normal face, tangential ghost",
        "compatibility": "net inflow must leave through an outlet",
        "failure": "over-specification with outlet",
    },
    "pressure_inlet": {
        "definition": "p = p_0, du/dn = 0",
        "implementation": "Dirichlet p (ghost), velocity by projection",
        "compatibility": "needs a pressure outlet elsewhere",
        "failure": "reverse flow",
    },
    "outlet": {
        "definition": "p = 0, du/dn = 0",
        "implementation": "Dirichlet p, zero-gradient tangential",
        "compatibility": "place far downstream",
        "failure": "backflow into the domain",
    },
    "periodic": {
        "definition": "u(0, y) = u(L, y)",
        "implementation": "wrap-around indexing in x",
        "compatibility": "streamwise-periodic flows only; mean pressure drop via body force",
        "failure": "non-periodic physics",
    },
    "farfield": {
        "definition": "u = U_inf on all components",
        "implementation": "as inlet with tangential value",
        "compatibility": "boundary far from the body (blockage error otherwise)",
        "failure": "domain too small -> blockage",
    },
}
