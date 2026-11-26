"""2D incompressible Navier–Stokes on a staggered (MAC) grid with a projection method.

    u_t + div(u u) = -grad p + lap(u)/Re + Ri theta e_y + f,     div u = 0
    theta_t + div(u theta) = lap(theta)/Pe + Q                 (optional, via fvm2d.FVEnergy2D)

Layout (same as fvm2d): p, theta at cell centres (ny, nx); u on x-faces (ny, nx+1); v on y-faces (ny+1, nx).
Space: divergence-form second-order central convection (energy-conserving on MAC), 5-point viscous term,
tangential wall velocities through ghost values u_g = 2 u_w - u_int.
Time: explicit SSP Runge–Kutta (euler | ssprk2 | ssprk3) with a pressure-free projection after every stage.
For time-independent normal boundary velocities this is exactly RK applied to the projected ODE
u' = P N(u), so the velocity keeps the RK order; the pressure is recovered from the consistent discrete
pressure Poisson equation  D G p = D N(u)  (see :meth:`pressure`).
Projection: sparse D (cells x faces) and G (faces x cells) are assembled explicitly and L = D diag(active) G,
so div_h(u) = 0 holds to solver precision by construction. L is constant: SuperLU is factorised once
(default); 'mg' / 'mgcg' reuse the Step-1.5 multigrid when there is no obstacle and the grid coarsens.
Boundary kinds per side: 'velocity' (no-slip wall, moving wall, inflow), 'slip', 'outflow' (zero normal
gradient of velocity, phi = 0 Dirichlet for the projection). Obstacles: boolean solid-cell mask (staircase);
faces touching a solid cell are fixed (zero normal velocity on the surface); faces inside the solid adjacent
to fluid faces carry mirror values (-u_fluid), i.e. tangential no-slip exactly at the cell interface.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from aerosynapse.thermal.numerics.fvm2d import SIDES, FaceBC, FaceVelocity, FVEnergy2D, Grid2D, dirichlet2, neumann2
from aerosynapse.thermal.numerics.linalg import cg

Array = NDArray[np.float64]
Fn = Callable[[Array, Array, float], Array]
SideFn = float | Callable[[Array, float], Array]

RK_STAGES = {"euler": 1, "ssprk2": 2, "ssprk3": 3}
RK_ORDER = {"euler": 1, "ssprk2": 2, "ssprk3": 3}


@dataclass(frozen=True)
class VelBC:
    """Velocity BC on one side. For kind 'velocity', u and v are the prescribed velocity components along
    the side as functions of (s, t), s = coordinate along the side (normal AND tangential are prescribed)."""

    kind: str = "velocity"  # 'velocity' | 'slip' | 'outflow'
    u: SideFn = 0.0
    v: SideFn = 0.0
    convective: bool = False  # outflow only: du/dt + U_c du/dn = 0 on the boundary faces (U_c = mean outflow)

    def comp(self, which: str, s: Array, t: float) -> Array:
        f = self.u if which == "u" else self.v
        if callable(f):
            return np.broadcast_to(np.asarray(f(s, t), float), s.shape).astype(float)
        return np.full(s.shape, float(f))


def wall(u: SideFn = 0.0, v: SideFn = 0.0) -> VelBC:
    return VelBC("velocity", u, v)


def slip() -> VelBC:
    return VelBC("slip")


def outflow(convective: bool = False) -> VelBC:
    """Outflow: zero normal gradient ghosts, phi = 0 in the projection.

    convective=True replaces the advective /
    viscous operator on the outflow faces by -U_c du/dn (the projection still adds -G phi there), i.e. a
    convective outflow that lets vortices leave with backflow. It is a DIFFERENT outflow condition (steady
    states differ slightly), used for unsteady wakes only. Found necessary in Step 1.7 (Schäfer–Turek 2D-2
    blew up with the zero-gradient outflow once the vortex street reached the outlet)."""
    return VelBC("outflow", convective=convective)


@dataclass
class EnergySpec:
    Pe: float
    bcs: dict[str, FaceBC]
    source: Fn | None = None
    scheme: str = "central"
    solid_theta: float | None = None  # None: solid conducts like the fluid (u = 0 there); float: isothermal
    bc_order: int = 1  # FV wall closure; 2 = second-order wall gradients / Nusselt numbers (Step 1.7, D30)


class _LiveVelocity(FaceVelocity):
    """FaceVelocity whose arrays are swapped in by the NS solver at every RK stage."""

    def __init__(self, grid: Grid2D):
        self.U = np.zeros((grid.ny, grid.nx + 1))
        self.V = np.zeros((grid.ny + 1, grid.nx))
        super().__init__(grid, lambda t: (self.U, self.V), steady=False)


class MACNavierStokes2D:
    def __init__(
        self,
        grid: Grid2D,
        Re: float,
        bcs: dict[str, VelBC],
        force: tuple[Fn, Fn] | None = None,
        solid: Array | None = None,
        energy: EnergySpec | None = None,
        Ri: float = 0.0,
        poisson: str = "splu",
        rk: str = "ssprk3",
        cfl: float = 0.5,
        fo: float = 0.5,
        dt_max: float = np.inf,
    ):
        if rk not in RK_STAGES:
            raise ValueError(f"rk must be one of {list(RK_STAGES)}")
        self.g, self.Re, self.Ri, self.force, self.rk = grid, float(Re), float(Ri), force, rk
        self.cfl, self.fo, self.dt_max = cfl, fo, dt_max
        self.bcs = {k: wall() for k in SIDES} | dict(bcs)
        g = grid
        self.solid = np.zeros((g.ny, g.nx), bool) if solid is None else np.asarray(solid, bool)
        self.fluid = ~self.solid
        self._build_masks()
        self._build_operators()
        self._build_poisson(poisson)
        # state
        self.t = 0.0
        self.nsteps = 0
        self.u = np.zeros((g.ny, g.nx + 1))
        self.v = np.zeros((g.ny + 1, g.nx))
        self.energy_spec = energy
        self.theta: Array | None = None
        self.energy: FVEnergy2D | None = None
        if energy is not None:
            self._vel = _LiveVelocity(g)
            self.energy = FVEnergy2D(
                g,
                1.0 / energy.Pe,
                self._vel,
                source=energy.source,
                bcs=energy.bcs,
                scheme=energy.scheme,
                bc_order=energy.bc_order,
            )
            self.theta = np.zeros((g.ny, g.nx))
        # coordinates of face centres
        self.Xu, self.Yu = np.meshgrid(g.xf, g.yc)
        self.Xv, self.Yv = np.meshgrid(g.xc, g.yf)

    # --- masks & operators ------------------------------------------------------------------
    def _build_masks(self) -> None:
        g, s = self.g, self.solid
        au = np.zeros((g.ny, g.nx + 1), bool)
        au[:, 1:-1] = True
        au[:, 0] = self.bcs["left"].kind == "outflow"
        au[:, -1] = self.bcs["right"].kind == "outflow"
        av = np.zeros((g.ny + 1, g.nx), bool)
        av[1:-1, :] = True
        av[0, :] = self.bcs["bottom"].kind == "outflow"
        av[-1, :] = self.bcs["top"].kind == "outflow"
        # faces touching a solid cell are fixed
        su = np.zeros_like(au)
        su[:, :-1] |= s
        su[:, 1:] |= s
        sv = np.zeros_like(av)
        sv[:-1, :] |= s
        sv[1:, :] |= s
        self.act_u, self.act_v = au & ~su, av & ~sv
        self.solid_u, self.solid_v = su, sv
        # faces whose every existing neighbour cell is solid (incl. domain-boundary faces next to a solid
        # cell) are mirror-ghost candidates; e.g. an outflow face right below a solid surface needs one
        both_u = np.zeros_like(au)
        both_u[:, 1:-1] = s[:, :-1] & s[:, 1:]
        both_u[:, 0], both_u[:, -1] = s[:, 0], s[:, -1]
        both_v = np.zeros_like(av)
        both_v[1:-1, :] = s[:-1, :] & s[1:, :]
        both_v[0, :], both_v[-1, :] = s[0, :], s[-1, :]
        self._mirror_u = both_u
        self._mirror_v = both_v
        self.active = np.concatenate([self.act_u.ravel(), self.act_v.ravel()])

    def _build_operators(self) -> None:
        g = self.g
        nx, ny, dx, dy = g.nx, g.ny, g.dx, g.dy
        nu_f, nv_f, nc = ny * (nx + 1), (ny + 1) * nx, nx * ny
        J, I = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        c = (J * nx + I).ravel()
        uE = (J * (nx + 1) + I + 1).ravel()
        uW = (J * (nx + 1) + I).ravel()
        vN = nu_f + ((J + 1) * nx + I).ravel()
        vS = nu_f + (J * nx + I).ravel()
        rows = np.concatenate([c, c, c, c])
        cols = np.concatenate([uE, uW, vN, vS])
        vals = np.concatenate([np.full(nc, 1 / dx), np.full(nc, -1 / dx), np.full(nc, 1 / dy), np.full(nc, -1 / dy)])
        self.D = sps.csr_matrix((vals, (rows, cols)), shape=(nc, nu_f + nv_f))
        # gradient: interior faces, plus half-cell Dirichlet (phi=0) rows at outflow boundary faces
        r, cc, vv = [], [], []
        Ju, Iu = np.meshgrid(np.arange(ny), np.arange(nx + 1), indexing="ij")
        Ju, Iu = Ju.ravel(), Iu.ravel()
        fu = Ju * (nx + 1) + Iu
        inner = (Iu > 0) & (Iu < nx)
        r += [fu[inner], fu[inner]]
        cc += [(Ju * nx + Iu)[inner], (Ju * nx + Iu - 1)[inner]]
        vv += [np.full(inner.sum(), 1 / dx), np.full(inner.sum(), -1 / dx)]
        if self.bcs["left"].kind == "outflow":
            m = Iu == 0
            r.append(fu[m])
            cc.append((Ju * nx)[m])
            vv.append(np.full(m.sum(), 2 / dx))  # noqa: E702
        if self.bcs["right"].kind == "outflow":
            m = Iu == nx
            r.append(fu[m])
            cc.append((Ju * nx + nx - 1)[m])
            vv.append(np.full(m.sum(), -2 / dx))  # noqa: E702
        Jv, Iv = np.meshgrid(np.arange(ny + 1), np.arange(nx), indexing="ij")
        Jv, Iv = Jv.ravel(), Iv.ravel()
        fv_ = nu_f + Jv * nx + Iv
        inner = (Jv > 0) & (Jv < ny)
        r += [fv_[inner], fv_[inner]]
        cc += [(Jv * nx + Iv)[inner], ((Jv - 1) * nx + Iv)[inner]]
        vv += [np.full(inner.sum(), 1 / dy), np.full(inner.sum(), -1 / dy)]
        if self.bcs["bottom"].kind == "outflow":
            m = Jv == 0
            r.append(fv_[m])
            cc.append(Iv[m])
            vv.append(np.full(m.sum(), 2 / dy))  # noqa: E702
        if self.bcs["top"].kind == "outflow":
            m = Jv == ny
            r.append(fv_[m])
            cc.append(((ny - 1) * nx + Iv)[m])
            vv.append(np.full(m.sum(), -2 / dy))  # noqa: E702
        G = sps.csr_matrix((np.concatenate(vv), (np.concatenate(r), np.concatenate(cc))), shape=(nu_f + nv_f, nc))
        self.G = sps.diags(self.active.astype(float)) @ G  # rows of fixed faces removed
        self.fluid_idx = np.flatnonzero(self.fluid.ravel())
        Df = self.D[self.fluid_idx]
        self.L = sps.csr_matrix(Df @ self.G[:, self.fluid_idx])
        self.singular = not any(self.bcs[k].kind == "outflow" for k in SIDES)

    def _build_poisson(self, method: str) -> None:
        self.poisson_method = method
        n = self.fluid_idx.size
        if method == "splu":
            L = self.L.tolil()
            if self.singular:  # pin one cell (compatible RHS => the dropped equation holds automatically)
                L[0, :] = 0.0
                L[0, 0] = 1.0
            self._lu = spla.splu(sps.csc_matrix(L))
        elif method in ("mg", "mgcg"):
            from aerosynapse.thermal.numerics.multigrid import MultigridHelmholtz

            if self.solid.any():
                raise ValueError("multigrid pressure solver does not support obstacle masks; use 'splu'")
            bcs = {k: (dirichlet2(0.0) if self.bcs[k].kind == "outflow" else neumann2(0.0)) for k in SIDES}
            self._mg = MultigridHelmholtz(self.g, 0.0, 1.0, bcs)
            if abs(self._mg.M + self.L).max() > 1e-8 * abs(self.L).max():
                raise RuntimeError("multigrid operator does not match D G")
        else:
            raise ValueError(f"unknown poisson solver {method!r}")
        self._n_fluid = n

    def solve_poisson(self, rhs: Array) -> Array:
        """Solve L phi = rhs on fluid cells (zero-mean RHS/solution in the singular all-wall case)."""
        if self.singular:
            rhs = rhs - rhs.mean()
        if self.poisson_method == "splu":
            b = rhs.copy()
            if self.singular:
                b[0] = 0.0
            phi = self._lu.solve(b)
        elif self.poisson_method == "mg":
            phi, _ = self._mg.solve(-rhs, rtol=1e-12, maxiter=60)
        else:
            phi, _ = cg(self._mg.M, -rhs, M=self._mg.precondition, rtol=1e-12, maxiter=200)
        return phi - phi.mean() if self.singular else phi

    # --- boundary conditions ----------------------------------------------------------------
    def _set_fixed(self, u: Array, v: Array, t: float) -> None:
        """Normal velocities on non-outflow boundary faces, zero on solid faces (in place)."""
        g, B = self.g, self.bcs
        if B["left"].kind != "outflow":
            u[:, 0] = B["left"].comp("u", g.yc, t) if B["left"].kind == "velocity" else 0.0
        if B["right"].kind != "outflow":
            u[:, -1] = B["right"].comp("u", g.yc, t) if B["right"].kind == "velocity" else 0.0
        if B["bottom"].kind != "outflow":
            v[0, :] = B["bottom"].comp("v", g.xc, t) if B["bottom"].kind == "velocity" else 0.0
        if B["top"].kind != "outflow":
            v[-1, :] = B["top"].comp("v", g.xc, t) if B["top"].kind == "velocity" else 0.0
        u[self.solid_u] = 0.0
        v[self.solid_v] = 0.0

    def _update_mirrors(self, u: Array, v: Array) -> None:
        """Tangential no-slip at solid surfaces: interior-solid faces next to fluid faces get -u_fluid."""
        if not self.solid.any():
            return
        for arr, mirror, act, axis in (
            (u, self._mirror_u, self.act_u, 0),
            (v, self._mirror_v, self.act_v, 1),
        ):
            acc = np.zeros_like(arr)
            cnt = np.zeros_like(arr)
            for sh in (1, -1):
                nb = np.roll(arr, sh, axis=axis)
                nb_act = np.roll(act, sh, axis=axis)
                # exclude wrap-around
                if axis == 0:
                    edge = slice(0, 1) if sh == 1 else slice(-1, None)
                    nb_act[edge, :] = False
                else:
                    edge = slice(0, 1) if sh == 1 else slice(-1, None)
                    nb_act[:, edge] = False
                acc += np.where(nb_act, -nb, 0.0)
                cnt += nb_act
            sel = mirror & (cnt > 0)
            arr[sel] = acc[sel] / cnt[sel]

    def _pad_u(self, u: Array, t: float) -> Array:
        g, B = self.g, self.bcs
        ny, nx = g.ny, g.nx
        P = np.zeros((ny + 2, nx + 3))
        P[1:-1, 1:-1] = u
        # x ghosts (faces i=-1, i=nx+1)
        P[1:-1, 0] = u[:, 1] if B["left"].kind == "outflow" else 2 * u[:, 0] - u[:, 1]
        P[1:-1, -1] = u[:, -2] if B["right"].kind == "outflow" else 2 * u[:, -1] - u[:, -2]
        xe = np.concatenate([[-g.dx], g.xf, [g.Lx + g.dx]])
        for side, row, inner in (("bottom", 0, 1), ("top", -1, -2)):
            bc = B[side]
            if bc.kind == "velocity":
                P[row, :] = 2 * bc.comp("u", xe, t) - P[inner, :]
            else:  # slip / outflow: zero tangential gradient
                P[row, :] = P[inner, :]
        return P

    def _pad_v(self, v: Array, t: float) -> Array:
        g, B = self.g, self.bcs
        ny, nx = g.ny, g.nx
        P = np.zeros((ny + 3, nx + 2))
        P[1:-1, 1:-1] = v
        P[0, 1:-1] = v[1, :] if B["bottom"].kind == "outflow" else 2 * v[0, :] - v[1, :]
        P[-1, 1:-1] = v[-2, :] if B["top"].kind == "outflow" else 2 * v[-1, :] - v[-2, :]
        ye = np.concatenate([[-g.dy], g.yf, [g.Ly + g.dy]])
        for side, col, inner in (("left", 0, 1), ("right", -1, -2)):
            bc = B[side]
            if bc.kind == "velocity":
                P[:, col] = 2 * bc.comp("v", ye, t) - P[:, inner]
            else:
                P[:, col] = P[:, inner]
        return P

    # --- spatial operator -------------------------------------------------------------------
    def momentum_rhs(self, u: Array, v: Array, theta: Array | None, t: float) -> tuple[Array, Array]:
        """N(u) = -div(u u) + lap(u)/Re + Ri theta e_y + f  on all faces (pressure-free)."""
        g = self.g
        dx, dy = g.dx, g.dy
        Up, Vp = self._pad_u(u, t), self._pad_v(v, t)
        uc = 0.5 * (Up[1:-1, :-1] + Up[1:-1, 1:])  # (ny, nx+2) cell-centre u (incl. ghost cells)
        duu = (uc[:, 1:] ** 2 - uc[:, :-1] ** 2) / dx
        ucor = 0.5 * (Up[:-1, 1:-1] + Up[1:, 1:-1])  # (ny+1, nx+1) at corners
        vcor = 0.5 * (Vp[1:-1, :-1] + Vp[1:-1, 1:])  # (ny+1, nx+1)
        uv = ucor * vcor
        duv_y = (uv[1:, :] - uv[:-1, :]) / dy  # at u faces (ny, nx+1)
        duv_x = (uv[:, 1:] - uv[:, :-1]) / dx  # at v faces (ny+1, nx)
        vc = 0.5 * (Vp[:-1, 1:-1] + Vp[1:, 1:-1])  # (ny+2, nx)
        dvv = (vc[1:, :] ** 2 - vc[:-1, :] ** 2) / dy
        lap_u = (Up[1:-1, 2:] - 2 * u + Up[1:-1, :-2]) / dx**2 + (Up[2:, 1:-1] - 2 * u + Up[:-2, 1:-1]) / dy**2
        lap_v = (Vp[1:-1, 2:] - 2 * v + Vp[1:-1, :-2]) / dx**2 + (Vp[2:, 1:-1] - 2 * v + Vp[:-2, 1:-1]) / dy**2
        Nu = -(duu + duv_y) + lap_u / self.Re
        Nv = -(duv_x + dvv) + lap_v / self.Re
        if self.Ri != 0.0 and theta is not None and self.energy is not None:
            tb = self.energy.boundary_values(theta, t)
            th_face = np.empty((g.ny + 1, g.nx))
            th_face[1:-1, :] = 0.5 * (theta[:-1, :] + theta[1:, :])
            th_face[0, :], th_face[-1, :] = tb["bottom"], tb["top"]
            Nv = Nv + self.Ri * th_face
        if self.force is not None:
            Nu = Nu + self.force[0](self.Xu, self.Yu, t)
            Nv = Nv + self.force[1](self.Xv, self.Yv, t)
        # convective outflow: first-order upwind transport of the boundary-face normal velocity
        B = self.bcs
        if B["right"].kind == "outflow" and B["right"].convective:
            Uc = max(float(np.mean(u[:, -1])), 1e-12)
            Nu[:, -1] = -Uc * (u[:, -1] - u[:, -2]) / dx
        if B["left"].kind == "outflow" and B["left"].convective:
            Uc = max(float(-np.mean(u[:, 0])), 1e-12)
            Nu[:, 0] = Uc * (u[:, 1] - u[:, 0]) / dx
        if B["top"].kind == "outflow" and B["top"].convective:
            Uc = max(float(np.mean(v[-1, :])), 1e-12)
            Nv[-1, :] = -Uc * (v[-1, :] - v[-2, :]) / dy
        if B["bottom"].kind == "outflow" and B["bottom"].convective:
            Uc = max(float(-np.mean(v[0, :])), 1e-12)
            Nv[0, :] = Uc * (v[1, :] - v[0, :]) / dy
        return Nu, Nv

    def energy_rhs(self, u: Array, v: Array, theta: Array, t: float) -> Array:
        assert self.energy is not None
        # mirror ghosts inside solids are not physical velocities: the scalar sees u = 0 on all solid faces
        self._vel.U = np.where(self.solid_u, 0.0, u) if self.solid.any() else u
        self._vel.V = np.where(self.solid_v, 0.0, v) if self.solid.any() else v
        return self.energy.rhs(theta.ravel(), t).reshape(theta.shape)

    # --- projection -------------------------------------------------------------------------
    def divergence(self, u: Array, v: Array) -> Array:
        d = (self.D @ np.concatenate([u.ravel(), v.ravel()])).reshape(self.g.ny, self.g.nx)
        return np.where(self.fluid, d, 0.0)

    def project(self, u: Array, v: Array, t: float) -> tuple[Array, Array, Array]:
        """u <- u - G phi with L phi = D u (fixed faces set to their BC values at time t first)."""
        self._set_fixed(u, v, t)
        f = np.concatenate([u.ravel(), v.ravel()])
        rhs = self.D[self.fluid_idx] @ f
        phi = self.solve_poisson(rhs)
        f = f - self.G[:, self.fluid_idx] @ phi
        nu_f = u.size
        u, v = f[:nu_f].reshape(u.shape), f[nu_f:].reshape(v.shape)
        self._update_mirrors(u, v)
        return u, v, phi

    # --- time stepping ----------------------------------------------------------------------
    def set_state(self, u: Array, v: Array, theta: Array | None = None, t: float = 0.0, project: bool = True) -> None:
        self.t = float(t)
        u, v = np.array(u, float), np.array(v, float)
        if project:
            u, v, _ = self.project(u, v, self.t)
        else:
            self._set_fixed(u, v, self.t)
            self._update_mirrors(u, v)
            d = float(np.max(np.abs(self.divergence(u, v))))
            if d > 1e-10:
                # a non-solenoidal initial state makes the first stage evaluate N off the constraint
                # manifold: O(dt h^2) local error -> a spurious O(dt) global error (found in Step 1.6)
                warnings.warn(
                    f"initial state not discretely divergence-free (max|D u| = {d:.2e}); use project=True",
                    stacklevel=2,
                )
        self.u, self.v = u, v
        if theta is not None:
            self.theta = np.array(theta, float)
            self._fix_solid_theta(self.theta)

    def _fix_solid_theta(self, th: Array) -> None:
        if self.energy_spec is not None and self.energy_spec.solid_theta is not None:
            th[self.solid] = self.energy_spec.solid_theta

    def stable_dt(self) -> float:
        g = self.g
        umax = np.max(np.abs(self.u)) / g.dx + np.max(np.abs(self.v)) / g.dy
        dt_c = self.cfl / umax if umax > 0 else np.inf
        inv_h2 = 1 / g.dx**2 + 1 / g.dy**2
        diff = 1.0 / self.Re
        if self.energy_spec is not None:
            diff = max(diff, 1.0 / self.energy_spec.Pe)
        dt_v = self.fo / (2 * diff * inv_h2)
        return float(min(dt_c, dt_v, self.dt_max))

    def _stage(self, u: Array, v: Array, th: Array | None, t: float, dt: float) -> tuple:
        Nu, Nv = self.momentum_rhs(u, v, th, t)
        un, vn = u + dt * Nu, v + dt * Nv
        thn = th + dt * self.energy_rhs(u, v, th, t) if th is not None else None
        return un, vn, thn

    def step(self, dt: float) -> None:
        u0, v0, th0, t = self.u, self.v, self.theta, self.t
        lin = lambda a, b, wa, wb: None if a is None else wa * a + wb * b  # noqa: E731
        if self.rk == "euler":
            u1, v1, th1 = self._stage(u0, v0, th0, t, dt)
            u, v, _ = self.project(u1, v1, t + dt)
            th = th1
        elif self.rk == "ssprk2":
            u1, v1, th1 = self._stage(u0, v0, th0, t, dt)
            u1, v1, _ = self.project(u1, v1, t + dt)
            self._fix(th1)
            u2, v2, th2 = self._stage(u1, v1, th1, t + dt, dt)
            u, v, _ = self.project(0.5 * u0 + 0.5 * u2, 0.5 * v0 + 0.5 * v2, t + dt)
            th = lin(th0, th2, 0.5, 0.5)
        else:
            u1, v1, th1 = self._stage(u0, v0, th0, t, dt)
            u1, v1, _ = self.project(u1, v1, t + dt)
            self._fix(th1)
            u2, v2, th2 = self._stage(u1, v1, th1, t + dt, dt)
            u2, v2, _ = self.project(0.75 * u0 + 0.25 * u2, 0.75 * v0 + 0.25 * v2, t + 0.5 * dt)
            th2 = lin(th0, th2, 0.75, 0.25)
            self._fix(th2)
            u3, v3, th3 = self._stage(u2, v2, th2, t + 0.5 * dt, dt)
            u, v, _ = self.project(u0 / 3 + 2 * u3 / 3, v0 / 3 + 2 * v3 / 3, t + dt)
            th = lin(th0, th3, 1 / 3, 2 / 3)
        self._fix(th)
        self.u, self.v, self.theta = u, v, th
        self.t = t + dt
        self.nsteps += 1

    def _fix(self, th: Array | None) -> None:
        if th is not None:
            self._fix_solid_theta(th)

    def run(
        self,
        t_end: float,
        dt: float | None = None,
        max_steps: int = 10**9,
        steady_tol: float | None = None,
        callback: Callable[[MACNavierStokes2D, float], None] | None = None,
    ) -> dict:
        """Advance to t_end (fixed dt, or adaptive stable_dt() when dt is None).

        steady_tol: stop when max|u^{n+1}-u^n|/dt (and theta) < steady_tol. Returns run info.
        """
        change = np.inf
        n0 = self.nsteps
        dts = []
        while self.t < t_end - 1e-12 and self.nsteps - n0 < max_steps:
            h = dt if dt is not None else self.stable_dt()
            h = min(h, t_end - self.t)
            uo, vo, tho = self.u, self.v, self.theta
            self.step(h)
            dts.append(h)
            change = max(np.max(np.abs(self.u - uo)), np.max(np.abs(self.v - vo))) / h
            if tho is not None and self.theta is not None:
                change = max(change, np.max(np.abs(self.theta - tho)) / h)
            if callback is not None:
                callback(self, h)
            if steady_tol is not None and change < steady_tol:
                break
        return {
            "t": self.t,
            "steps": self.nsteps - n0,
            "last_change_rate": float(change),
            "dt_min": float(min(dts)) if dts else None,
            "dt_max": float(max(dts)) if dts else None,
            "max_div": self.max_div(),
        }

    # --- diagnostics ------------------------------------------------------------------------
    def max_div(self) -> float:
        return float(np.max(np.abs(self.divergence(self.u, self.v))))

    def net_boundary_flux(self, t: float | None = None) -> float:
        """Net outflow through the domain boundary (should be 0 for a closed or balanced domain)."""
        g = self.g
        return float(
            (self.u[:, -1].sum() - self.u[:, 0].sum()) * g.dy + (self.v[-1, :].sum() - self.v[0, :].sum()) * g.dx
        )

    def pressure(self) -> Array:
        """Consistent pressure from D G p = D (act*N(u) + fixed*du_fixed/dt); NaN in solid cells."""
        u, v, t = self.u, self.v, self.t
        Nu, Nv = self.momentum_rhs(u, v, self.theta, t)
        eps = 1e-6
        up, vp = np.zeros_like(u), np.zeros_like(v)
        um, vm = np.zeros_like(u), np.zeros_like(v)
        self._set_fixed(up, vp, t + eps)
        self._set_fixed(um, vm, t - eps)
        dfix = np.concatenate([(up - um).ravel(), (vp - vm).ravel()]) / (2 * eps)
        fN = np.concatenate([Nu.ravel(), Nv.ravel()])
        rhs_faces = np.where(self.active, fN, dfix)
        p_f = self.solve_poisson(self.D[self.fluid_idx] @ rhs_faces)
        p = np.full(self.g.n, np.nan)
        p[self.fluid_idx] = p_f
        return p.reshape(self.g.ny, self.g.nx)

    def body_force(self, p: Array | None = None) -> tuple[float, float]:
        """Force (per unit depth, rho = 1) exerted by the fluid on the solid mask.

        Every body face is held fixed, so the fluid there feels f = -(N - G p); the body feels
        F = sum_body (N - G p) dx dy. Because convective/viscous fluxes are shared between neighbouring
        faces and G p telescopes along each grid line, the (arbitrary) pressure inside the solid cancels
        and F equals the discrete momentum balance around the body (exact for the discrete solution).
        """
        if not self.solid.any():
            return 0.0, 0.0
        P = np.nan_to_num(self.pressure() if p is None else p, nan=0.0)
        P = np.where(self.solid, 0.0, P)
        g = self.g
        Nu, Nv = self.momentum_rhs(self.u, self.v, self.theta, self.t)
        Gx = np.zeros_like(self.u)
        Gx[:, 1:-1] = (P[:, 1:] - P[:, :-1]) / g.dx
        Gy = np.zeros_like(self.v)
        Gy[1:-1, :] = (P[1:, :] - P[:-1, :]) / g.dy
        bu = self.solid_u.copy()
        bu[:, 0] = bu[:, -1] = False  # bodies are interior; domain-boundary faces carry BCs
        bv = self.solid_v.copy()
        bv[0, :] = bv[-1, :] = False
        Fx = float(np.sum((Nu - Gx)[bu]) * g.vol)
        Fy = float(np.sum((Nv - Gy)[bv]) * g.vol)
        return Fx, Fy

    def cell_velocity(self) -> tuple[Array, Array]:
        return 0.5 * (self.u[:, 1:] + self.u[:, :-1]), 0.5 * (self.v[1:, :] + self.v[:-1, :])

    def kinetic_energy(self) -> float:
        uc, vc = self.cell_velocity()
        return float(0.5 * np.sum((uc**2 + vc**2)[self.fluid]) * self.g.vol)

    def vorticity(self) -> Array:
        """omega = v_x - u_y at cell corners (ny+1, nx+1); wall corners use the BC ghost values."""
        Up, Vp = self._pad_u(self.u, self.t), self._pad_v(self.v, self.t)
        return (Vp[1:-1, 1:] - Vp[1:-1, :-1]) / self.g.dx - (Up[1:, 1:-1] - Up[:-1, 1:-1]) / self.g.dy

    def streamfunction(self) -> Array:
        """psi at cell corners (ny+1, nx+1): psi(x,0) = -int v dx, then psi += int u dy (path-independent
        to round-off because div_h u = 0)."""
        g = self.g
        psi = np.zeros((g.ny + 1, g.nx + 1))
        psi[0, 1:] = -np.cumsum(self.v[0, :]) * g.dx
        psi[1:, :] = psi[0, :] + np.cumsum(self.u, axis=0) * g.dy
        return psi

    # --- checkpoint -------------------------------------------------------------------------
    def save_state(self, path: str | Path) -> None:
        np.savez(
            path,
            u=self.u,
            v=self.v,
            t=self.t,
            nsteps=self.nsteps,
            theta=self.theta if self.theta is not None else np.array([]),
        )

    def load_state(self, path: str | Path) -> None:
        d = np.load(path)
        self.u, self.v = d["u"].copy(), d["v"].copy()
        self.t, self.nsteps = float(d["t"]), int(d["nsteps"])
        self.theta = d["theta"].copy() if d["theta"].size else None
