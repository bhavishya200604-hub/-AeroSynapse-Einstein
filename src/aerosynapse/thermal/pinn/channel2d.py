"""Case D for the parametric 2D PINN (Step 4.5b): channel flows with heat transfer.

    D_graetz    channel [0, Lx] x [0, Ly]: fully developed parabolic inflow (mean 1) at theta_in, zero-gradient outflow,
                no-slip walls with thermal conditions (default Robin -dtheta/dn = Bi (theta - theta_inf): the
                thermally developing Graetz problem; Bi -> infinity = constant wall temperature)
    D_obstacle  the same channel around a grid-aligned rectangular obstacle (square cylinder, D22) held at theta_obs
                (isothermal), no-slip on all its faces

Steady, Boussinesq coupling off (Ri = 0). Inputs (x, y, log Re, log Pr, log Bi ...) exactly as for the parametric
cavity (ParamProblem2D: per-point residuals, collocation parameters in the training box); the geometry adds boundary
SEGMENTS (inlet, outlet, bottom, top and the four obstacle faces). Boundary conditions are soft, except with
streamfunction + hard_bc (Step 4.5b(a), D73): psi is then prescribed EXACTLY on the inlet, the walls and the obstacle,

    psi = psi_in(y) + (psi_obs - psi_in(y)) w + D_out phi N,   D_out = (x/Lx) 4 s (1 - s),  s = y/Ly,
    w = D_out / (D_out + phi^2),  phi = r/(h + r), r = R-function distance to the obstacle (0 on its faces)

(no obstacle: psi = psi_in(y) + D_out N), with psi_in(y) = int_0^y u_in, so psi = 0 on the bottom wall, Q = psi_in(Ly)
on the top wall: no flow through the walls, the inlet profile u = u_in and the flow rate Q at every section (mass
conservation) hold exactly. psi_obs: 'symmetric' = psi_in(Ly/2) = Q/2 (exact for a centred obstacle, a symmetric inlet
profile and Ri = 0), 'learned' = psi_in(y_c) + N_psi(obstacle centre, parameters), 'auto' = symmetric when exact,
'soft' = psi exact on inlet and walls only (psi = psi_in + D_out N; Q at the outlet still exact), the obstacle
conditions stay loss terms (D74 test i).
Still soft: no-slip (tangential velocity) on walls and obstacle, inlet v, outflow conditions, all thermal conditions.

References (Phase 1 solver, two grids): no obstacle -> steady Newton flow + linear-scheme energy (decoupled, D66);
with obstacle -> coupled steady Newton (central energy, isothermal solid cells). Quantities: fields on the fluid cells,
the bottom-wall heat flux q = mean d(theta)/dn_in, and for the obstacle the drag coefficient C_D = 2 F_x / (U_mean^2 D)
(PINN: surface integral of the fluid stress; reference: discrete momentum balance, body_force).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from aerosynapse.thermal.data.design import Dim, sobol_unit
from aerosynapse.thermal.data.generators import _thermal_bc, restrict_cell_average
from aerosynapse.thermal.data.spec import eval_profile
from aerosynapse.thermal.numerics import fvm2d as fv
from aerosynapse.thermal.numerics import ns_projection as ns
from aerosynapse.thermal.numerics.ns_steady import SteadyNewtonNS
from aerosynapse.thermal.pinn import param2d as PP
from aerosynapse.thermal.pinn import problems2d as P2

CHANNEL_PROBLEMS = ("D_graetz", "D_obstacle")
SEG_NORMAL = {"bottom": (0.0, -1.0), "top": (0.0, 1.0)}


def _obstacle(geom: dict) -> dict | None:
    return geom.get("obstacle") or None


@dataclass
class ChannelProblem(PP.ParamProblem2D):
    geom: dict = field(default_factory=dict)
    inlet: dict = field(default_factory=dict)
    thermal: dict = field(default_factory=dict)  # bottom / top: {type, value | Bi, theta_inf}
    hard_psi: bool = False  # streamfunction only: psi prescribed exactly on inlet, walls, obstacle (D73)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.Lx, self.Ly = float(self.geom["Lx"]), float(self.geom["Ly"])
        self.lo = (0.0, 0.0, *self.lo[2:])
        self.hi = (self.Lx, self.Ly, *self.hi[2:])
        self.obs = _obstacle(self.geom)
        self.segments = ["inlet", "outlet", "bottom", "top"] + (
            ["obs_left", "obs_right", "obs_bottom", "obs_top"] if self.obs else []
        )
        if self.hard_psi:
            if self.kind != "streamfunction":
                raise ValueError(f"{self.name}: hard psi constraints need the streamfunction formulation")
            self.Q = float(self.psi_inlet(torch.tensor([self.Ly], dtype=torch.float64))[0])
            self.psi_obs_mode = obstacle_psi_mode(self.geom, self.inlet, self.params, self.sweep) if self.obs else None

    # --- exact stream-function boundary values (hard_psi) ------------------------------------------------------
    def psi_inlet(self, y: Tensor) -> Tensor:
        """psi_in(y) = int_0^y u_in dy' (analytic for the named profiles, 64-point Gauss-Legendre for 'expr')."""
        vel, H = self.inlet["velocity"], self.Ly
        kind, a, s = vel.get("profile", "zero"), float(vel.get("amplitude", 1.0)), y / H
        if kind == "zero":
            return torch.zeros_like(y)
        if kind == "uniform":
            return a * y
        if kind == "parabolic":
            return a * H * (2.0 * s**2 - 4.0 * s**3 / 3.0)
        if kind == "cosh":
            C = float(vel.get("C", 50.0))
            return a * (y - H * (torch.sinh(C * (s - 0.5)) + math.sinh(0.5 * C)) / (C * math.cosh(0.5 * C)))
        xi, wq = np.polynomial.legendre.leggauss(64)
        xi, wq = torch.as_tensor(0.5 * (xi + 1.0)).to(y), torch.as_tensor(0.5 * wq).to(y)
        return y * (P2.profile_torch(vel, s[:, None] * xi[None, :]) * wq).sum(1)

    def obstacle_distance(self, x: Tensor, y: Tensor) -> Tensor:
        """Normalised distance to the rectangular obstacle: zero exactly on its faces, positive outside, smooth except
        at its four corners (R-function conjunction of the two strips, r = sqrt(sx^2 + sy^2) - sx - sy; phi = r/(h + r)
        with h = sum of the sides, so h + r > 0 also inside the obstacle)."""
        o = self.obs
        w, hh = o["x1"] - o["x0"], o["y1"] - o["y0"]
        sx, sy = (x - o["x0"]) * (o["x1"] - x) / w, (y - o["y0"]) * (o["y1"] - y) / hh
        r = torch.sqrt(sx**2 + sy**2) - sx - sy
        return r / (w + hh + r)

    def psi_obstacle(self, model: torch.nn.Module, X: Tensor) -> Tensor | float:
        o = self.obs
        yc = 0.5 * (o["y0"] + o["y1"])
        base = float(self.psi_inlet(torch.tensor([yc], dtype=torch.float64))[0])
        if self.psi_obs_mode == "symmetric":
            return base
        c = torch.tensor([[0.5 * (o["x0"] + o["x1"]), yc]], dtype=X.dtype, device=X.device)
        Xc = torch.cat([c.expand(X.shape[0], 2), X[:, 2:]], 1)  # obstacle centre at each row's parameters
        return base + model(Xc)[:, 0]

    def transform(self, model: torch.nn.Module, X: Tensor) -> dict[str, Tensor]:
        F = super().transform(model, X)
        if not self.hard_psi:
            return F
        x, y = X[:, 0], X[:, 1]
        s = y / self.Ly
        Dout = (x / self.Lx) * 4.0 * s * (1.0 - s)
        pin = self.psi_inlet(y)
        if not self.obs or self.psi_obs_mode == "soft":
            F["psi"] = pin + Dout * F["psi"]
            return F
        phi = self.obstacle_distance(x, y)
        w = Dout / (Dout + phi**2)
        F["psi"] = pin + (self.psi_obstacle(model, X) - pin) * w + Dout * phi * F["psi"]
        return F

    # --- geometry --------------------------------------------------------------------------------------------
    def inside_obstacle(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not self.obs:
            return np.zeros_like(x, bool)
        o = self.obs
        return (x > o["x0"]) & (x < o["x1"]) & (y > o["y0"]) & (y < o["y1"])

    def segment_points(self, seg: str, s: np.ndarray) -> np.ndarray:
        L, H, o = self.Lx, self.Ly, self.obs
        z = np.zeros_like(s)
        if seg == "inlet":
            return np.stack([z, H * s], 1)
        if seg == "outlet":
            return np.stack([z + L, H * s], 1)
        if seg == "bottom":
            return np.stack([L * s, z], 1)
        if seg == "top":
            return np.stack([L * s, z + H], 1)
        if seg in ("obs_left", "obs_right"):
            xx = o["x0"] if seg == "obs_left" else o["x1"]
            return np.stack([z + xx, o["y0"] + (o["y1"] - o["y0"]) * s], 1)
        yy = o["y0"] if seg == "obs_bottom" else o["y1"]
        return np.stack([o["x0"] + (o["x1"] - o["x0"]) * s, z + yy], 1)

    def sample(self, n_pde, n_ic, n_bc, method, seed, device, dtype):
        k = len(self.sweep)

        def unit(n: int, dim: int, s: int) -> np.ndarray:
            if method == "sobol":
                return sobol_unit(n, dim, s)
            if method == "lhs":
                from scipy.stats import qmc

                return qmc.LatinHypercube(d=dim, seed=s).random(n)
            return np.random.default_rng(s).uniform(size=(n, dim))

        def params_cols(u: np.ndarray) -> np.ndarray:
            uu = u * self.u_lim
            cols = [d.to_phys(uu[:, i]) for i, d in enumerate(self.sweep)]
            return (
                np.stack([np.log(c) if d.is_log else c for c, d in zip(cols, self.sweep, strict=True)], 1)
                if k
                else np.zeros((len(u), 0))
            )

        mk = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)  # noqa: E731
        u = unit(2 * n_pde + 64, 2 + k, seed)  # oversample, drop points inside the obstacle, keep the first n_pde
        xy = u[:, :2] * [self.Lx, self.Ly]
        keep = np.flatnonzero(~self.inside_obstacle(xy[:, 0], xy[:, 1]))[:n_pde]
        pde = np.concatenate([xy[keep], params_cols(u[keep, 2:])], 1)
        walls = {}
        for j, seg in enumerate(self.segments):
            uw = unit(n_bc, 1 + k, seed + 2 + j)
            walls[seg] = mk(np.concatenate([self.segment_points(seg, uw[:, 0]), params_cols(uw[:, 1:])], 1))
        return {"pde": mk(pde), "walls": walls}

    # --- boundary residuals (all soft) -----------------------------------------------------------------------
    def wall_residuals(self, model: torch.nn.Module, walls: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        out = {}
        for seg, Xs in walls.items():
            d = self.fields(model, Xs, order=1)
            y = d["y"]
            if seg == "inlet":
                u_in = P2.profile_torch(self.inlet["velocity"], y / self.Ly)
                out[seg] = {"u": d["u"] - u_in, "v": d["v"], "theta": d["theta"] - float(self.inlet["theta"])}
            elif seg == "outlet":
                out[seg] = {"u": d["u_x"], "v": d["v_x"], "theta": d["theta_x"]}
            elif seg in ("bottom", "top"):
                t = self.thermal[seg]
                nx_, ny_ = SEG_NORMAL[seg]
                dn = nx_ * d["theta_x"] + ny_ * d["theta_y"]
                if t["type"] == "dirichlet":
                    th = d["theta"] - float(t.get("value", 0.0))
                elif t["type"] == "neumann":
                    th = dn - float(t.get("value", 0.0))
                else:
                    a = d["par"][t["Bi"]] if isinstance(t["Bi"], str) else float(t["Bi"])
                    th = a * (d["theta"] - float(t.get("theta_inf", 0.0))) + dn
                out[seg] = {"u": d["u"], "v": d["v"], "theta": th}
            else:
                out[seg] = {"u": d["u"], "v": d["v"], "theta": d["theta"] - float(self.obs["theta"])}
        return out

    def residual_vectors(self, model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Tensor]:
        d = self.fields(model, batch["pde"], order=2)
        r = self.residuals(d)
        out: dict[str, Tensor] = {}
        for k in P2.RESIDUALS:
            if k == "cont" and self.kind == "streamfunction":
                continue
            out["pde_" + k] = r[k] / self.scales[k]
        wr = self.wall_residuals(model, batch["walls"])
        for c in ("u", "v", "theta"):
            out["bc_" + c] = torch.cat([wr[s][c] for s in wr])
        g = self.gauge
        if g["kind"] == "point" and g["mode"] == "soft":
            X0 = torch.cat([self.X0.to(batch["pde"]), batch["pde"][:1, 2:]], 1)
            out["gauge"] = self.transform(model, X0)["p"] - float(g["value"])
        elif g["kind"] == "mean":
            out["gauge"] = (d["p"].mean() - float(g["value"])).reshape(1)
        return out

    # --- inputs on a grid -------------------------------------------------------------------------------------
    def inputs(self, xy: np.ndarray, values: dict[str, float] | None, like: Tensor) -> Tensor:
        cols = [xy] + ([np.tile(self.encode(values), (len(xy), 1))] if self.sweep else [])
        return torch.as_tensor(np.concatenate(cols, 1), dtype=like.dtype, device=like.device)


def build_channel(
    name: str, pcfg: dict, form: dict, sweep: list[Dim], fixed: dict[str, float], u_lim: float
) -> ChannelProblem:
    """A Case D problem; ``sweep`` = [] gives a fixed-parameter (per-point) problem with ``fixed`` values. hard_bc is
    available with the streamfunction formulation only (exact psi on inlet, walls and obstacle; the unit-square
    transfinite constraints of Step 4.2 are not used)."""
    if form["hard_bc"] and form["kind"] != "streamfunction":
        raise ValueError(f"{name}: hard_bc for channel geometries needs formulation.kind = streamfunction")
    nominal = {**fixed, **{d.name: float(d.to_phys(0.5 * u_lim)) for d in sweep}}
    zero = lambda x, y: torch.zeros_like(x)  # noqa: E731
    walls = {w: P2.WallBC(zero, zero, 1.0, 0.0, zero) for w in P2.WALLS}  # placeholders (base class bookkeeping)
    return ChannelProblem(
        name,
        nominal,
        walls,
        {**form, "hard_bc": False},
        None,
        {},
        sweep=tuple(sweep),
        u_lim=float(u_lim),
        geom=pcfg["geometry"],
        inlet=pcfg["inlet"],
        thermal=pcfg["thermal"],
        hard_psi=bool(form["hard_bc"]),
    )


SYMMETRIC_PROFILES = ("zero", "uniform", "parabolic", "cosh")


def obstacle_psi_mode(geom: dict, inlet: dict, params: dict, swept: tuple | list = ()) -> str:
    """Resolve geometry.obstacle.psi (auto | symmetric | learned); ``swept`` = swept parameter names or Dims.
    'symmetric' (psi_obs = Q/2) is exact only for an obstacle centred in the channel, a symmetric inlet profile and no
    buoyancy (Ri fixed at 0)."""
    o = geom["obstacle"]
    mode = o.get("psi", "auto")
    names = [getattr(d, "name", d) for d in swept]
    ri_zero = "Ri" not in names and float(params.get("Ri", 0.0)) == 0.0
    sym = (
        abs(o["y0"] + o["y1"] - float(geom["Ly"])) < 1e-12
        and inlet["velocity"].get("profile", "zero") in SYMMETRIC_PROFILES
        and ri_zero
    )
    if mode in ("soft", "learned"):
        return mode
    if mode == "auto":
        return "symmetric" if sym else "learned"
    if mode == "symmetric" and not sym:
        raise ValueError(
            "psi = symmetric needs a centred obstacle (y0 + y1 = Ly), a symmetric inlet profile "
            f"{list(SYMMETRIC_PROFILES)} and Ri = 0; use 'learned' or 'auto'"
        )
    return mode


# --- reference solutions -------------------------------------------------------------------------------------------
def _solver(params: dict, pcfg: dict, n_per_unit: int, energy: bool, scheme: str) -> ns.MACNavierStokes2D:
    geom = pcfg["geometry"]
    Lx, Ly = float(geom["Lx"]), float(geom["Ly"])
    g = fv.Grid2D(int(round(Lx * n_per_unit)), int(round(Ly * n_per_unit)), Lx, Ly)
    vel = pcfg["inlet"]["velocity"]
    inflow = lambda s_, t, vel=vel, Ly=Ly: eval_profile(vel, np.asarray(s_) / Ly, t)  # noqa: E731
    bcs = {"left": ns.wall(u=inflow), "right": ns.outflow()}
    obs = _obstacle(geom)
    solid = None
    if obs:
        solid = (g.Xc > obs["x0"]) & (g.Xc < obs["x1"]) & (g.Yc > obs["y0"]) & (g.Yc < obs["y1"])
    en = None
    if energy:
        en = ns.EnergySpec(
            Pe=float(params["Re"]) * float(params["Pr"]),
            bcs=_thermal_bcs(params, pcfg),
            scheme=scheme,
            bc_order=2,
            solid_theta=float(obs["theta"]) if obs else None,
        )
    return ns.MACNavierStokes2D(g, float(params["Re"]), bcs, solid=solid, energy=en, Ri=float(params.get("Ri", 0.0)))


def _thermal_bcs(params: dict, pcfg: dict) -> dict[str, fv.FaceBC]:
    t = pcfg["thermal"]
    return {
        "left": fv.dirichlet2(float(pcfg["inlet"]["theta"])),
        "right": fv.neumann2(0.0),
        "bottom": _thermal_bc(t["bottom"], params),
        "top": _thermal_bc(t["top"], params),
    }


def solve_channel(params: dict, pcfg: dict, n_per_unit: int, tol: float = 1e-10, scheme: str = "quick") -> dict:
    """Steady reference on one grid. No obstacle: Newton flow + energy with ``scheme`` (decoupled, Ri = 0); obstacle:
    coupled Newton with central energy convection (the solid is held at theta_obs)."""
    t0 = time.perf_counter()
    obs = _obstacle(pcfg["geometry"])
    coupled = bool(obs) or float(params.get("Ri", 0.0)) != 0.0
    s = _solver(params, pcfg, n_per_unit, energy=coupled, scheme="central")
    N = SteadyNewtonNS(s)
    th0 = np.zeros((s.g.ny, s.g.nx)) if coupled else None
    info = N.solve(N.pack(s.u, s.v, None, th0), tol=tol, max_iter=40)
    if not info["converged"]:  # continuation in Re from rest (as the cavity reference)
        Re = float(params["Re"])
        x = N.pack(s.u * 0, s.v * 0, None, th0)
        for r in (Re / 8, Re / 4, Re / 2, Re):
            s.Re = r
            info = N.solve(x, tol=tol, max_iter=40)
            x = N.x
    if not info["converged"]:
        raise RuntimeError(
            f"channel reference did not converge (n={n_per_unit}): |R| = {info['residual_history'][-1]:.3e}"
        )
    p = N.store()
    if coupled:
        E = s.energy
        th = s.theta.ravel()
        used = "coupled steady Newton (central energy)"
    else:
        E = fv.FVEnergy2D(
            s.g,
            1.0 / (float(params["Re"]) * float(params["Pr"])),
            fv.FaceVelocity.from_arrays(s.g, s.u, s.v),
            bcs=_thermal_bcs(params, pcfg),
            scheme=scheme,
            bc_order=2,
        )
        th = E.steady()
        used = f"decoupled: steady Newton flow + {scheme} energy"
    uc, vc = s.cell_velocity()
    out = {
        "x": s.g.xc.copy(),
        "y": s.g.yc.copy(),
        "solid": s.solid.copy(),
        "u": uc,
        "v": vc,
        "p": p - np.nanmean(p),
        "theta": th.reshape(s.g.ny, s.g.nx),
        "q_bottom": float(np.mean(E.wall_gradient(th, 0.0, "bottom"))),
        # per-x wall data (Step 6.6: local Nusselt numbers): wall temperature and inward wall gradient, both walls
        **{
            f"theta_wall_{w}": np.asarray(E.boundary_values(th.reshape(s.g.ny, s.g.nx), 0.0)[w]).copy()
            for w in ("bottom", "top")
        },
        **{f"dtheta_dn_{w}": np.asarray(E.wall_gradient(th, 0.0, w)).copy() for w in ("bottom", "top")},
        "newton_iterations": info["iterations"],
        "newton_residual": info["residual_history"][-1],
        "max_div": s.max_div(),
        "kind": used,
        "wall_s": time.perf_counter() - t0,
    }
    if obs:
        Fx, Fy = s.body_force(p)
        D = float(obs["y1"] - obs["y0"])
        out["C_D"], out["C_L"] = 2.0 * Fx / D, 2.0 * Fy / D  # U_mean = 1, rho = 1
    return out


def channel_reference(
    params: dict, pcfg: dict, n_per_unit: int, refine: int = 2, tol: float = 1e-10, scheme: str = "quick"
) -> dict:
    """Fine solution (truth) + coarse-fine relative differences on the coarse fluid cells."""
    c = solve_channel(params, pcfg, n_per_unit, tol, scheme)
    f = solve_channel(params, pcfg, refine * n_per_unit, tol, scheme)
    fluid = ~c["solid"]
    fr = {
        k: restrict_cell_average(np.where(f["solid"], 0.0, np.nan_to_num(f[k])), refine)[fluid]
        for k in ("u", "v", "theta", "p")
    }
    vel = float(np.hypot(np.linalg.norm(fr["u"]), np.linalg.norm(fr["v"])))  # u, v relative to the velocity-vector norm
    diff = {}
    for k in ("u", "v", "theta", "p"):
        a, b = c[k][fluid], fr[k]
        if k == "p":
            a, b = a - a.mean(), b - b.mean()
        den = vel if k in ("u", "v") else np.linalg.norm(b)
        diff[k] = float(np.linalg.norm(a - b) / max(den, 1e-300))
    diff["q_bottom"] = abs(c["q_bottom"] - f["q_bottom"]) / max(abs(f["q_bottom"]), 1e-300)
    if "C_D" in f:
        diff["C_D"] = abs(c["C_D"] - f["C_D"]) / max(abs(f["C_D"]), 1e-300)
    return {
        **f,
        "n_coarse": n_per_unit,
        "n_fine": refine * n_per_unit,
        "coarse_fine_diff": diff,
        "newton": {
            "coarse": {"iterations": c["newton_iterations"], "residual": c["newton_residual"], "wall_s": c["wall_s"]},
            "fine": {"iterations": f["newton_iterations"], "residual": f["newton_residual"], "wall_s": f["wall_s"]},
        },
    }


# --- evaluation ----------------------------------------------------------------------------------------------------
def flow_rates(problem: ChannelProblem, model: torch.nn.Module, values: dict | None, n: int = 64) -> dict:
    """PINN volume flow rate Q = int u dy at the inlet and outlet sections (midpoint rule, n points) and their ratio;
    the exact flow conserves mass, so flux_ratio = Q_out / Q_in = 1."""
    p0 = next(model.parameters())
    yq = (np.arange(n) + 0.5) / n * problem.Ly
    q = []
    for xq in (0.0, problem.Lx):
        F = problem.fields(model, problem.inputs(np.stack([np.full(n, xq), yq], 1), values, p0), order=0)
        q.append(float(F["u"].detach().mean()) * problem.Ly)
    return {"flux_inlet": q[0], "flux_outlet": q[1], "flux_ratio": q[1] / q[0] if abs(q[0]) > 1e-300 else float("nan")}


def drag_coefficient(problem: ChannelProblem, model: torch.nn.Module, values: dict | None, n_face: int = 64) -> float:
    """C_D = 2 F_x / D, F_x = sum over the obstacle faces of (-p n_x + (1/Re)(2 u_x n_x + (u_y + v_x) n_y)) ds,
    n pointing OUT of the body (midpoint rule; the unknown pressure constant cancels on the closed surface)."""
    o = problem.obs
    p0 = next(model.parameters())
    Re = float(values["Re"]) if values else float(problem.params["Re"])
    s = (np.arange(n_face) + 0.5) / n_face
    Fx = 0.0
    for seg, (nx_, ny_), length in (
        ("obs_left", (-1.0, 0.0), o["y1"] - o["y0"]),
        ("obs_right", (1.0, 0.0), o["y1"] - o["y0"]),
        ("obs_bottom", (0.0, -1.0), o["x1"] - o["x0"]),
        ("obs_top", (0.0, 1.0), o["x1"] - o["x0"]),
    ):
        X = problem.inputs(problem.segment_points(seg, s), values, p0)
        d = problem.fields(model, X, order=1)
        tx = -d["p"] * nx_ + (2.0 * d["u_x"] * nx_ + (d["u_y"] + d["v_x"]) * ny_) / Re
        Fx += float(tx.detach().mean()) * length
    return 2.0 * Fx / float(o["y1"] - o["y0"])


def evaluate_channel(
    problem: ChannelProblem, model: torch.nn.Module, ref: dict, values: dict | None, exclude: np.ndarray | None = None
) -> dict[str, Any]:
    """Field errors on the fluid cells of the reference grid (``exclude``: grid mask of cells left out, e.g. the cells
    used as training data in Step 4.6), bottom-wall heat flux, flow rates and (obstacle) drag coefficient."""
    p0 = next(model.parameters())
    X, Y = np.meshgrid(ref["x"], ref["y"])
    fluid = ~np.asarray(ref["solid"])
    if exclude is not None:
        fluid = fluid & ~np.asarray(exclude, bool)
    xy = np.stack([X[fluid], Y[fluid]], 1)
    pred = P2._batched(problem, model, problem.inputs(xy, values, p0), 1, (*P2.FIELDS, "u_x", "v_y"))
    res: dict[str, Any] = {}
    vel = float(np.hypot(np.linalg.norm(np.asarray(ref["u"])[fluid]), np.linalg.norm(np.asarray(ref["v"])[fluid])))
    for k in P2.FIELDS:
        r = np.asarray(ref[k])[fluid]
        pk = pred[k]
        if k == "p":
            r, pk = r - r.mean(), pk - pk.mean()
        den = (
            vel if k in ("u", "v") else np.linalg.norm(r)
        )  # u, v relative to the velocity-vector norm (v ~ 0 in a channel)
        res[f"rel_l2_{k}"] = float(np.linalg.norm(pk - r) / max(den, 1e-300))
        res[f"linf_{k}"] = float(np.abs(pk - r).max())
    res["rel_l2_max"] = max(res["rel_l2_u"], res["rel_l2_v"], res["rel_l2_theta"])
    res["max_div"] = float(np.abs(pred["u_x"] + pred["v_y"]).max())
    xb = np.stack([np.asarray(ref["x"]), np.zeros(len(ref["x"]))], 1)
    q = float(problem.fields(model, problem.inputs(xb, values, p0), order=1)["theta_y"].detach().mean())
    res["q_bottom"], res["q_bottom_ref"] = q, float(ref["q_bottom"])
    res.update(flow_rates(problem, model, values))
    res["q_bottom_rel_err"] = abs(q - ref["q_bottom"]) / max(abs(ref["q_bottom"]), 1e-300)
    if problem.obs and "C_D" in ref:
        cd = drag_coefficient(problem, model, values)
        res["C_D"], res["C_D_ref"] = cd, float(ref["C_D"])
        res["C_D_rel_err"] = abs(cd - ref["C_D"]) / max(abs(ref["C_D"]), 1e-300)
    return res
