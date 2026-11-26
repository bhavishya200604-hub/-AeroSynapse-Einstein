"""2D steady PINN problems and formulations (Step 4.2).

Equations on the unit square (non-dimensional, steady, Boussinesq; same convention as aerosynapse.thermal.physics.mms.ManufacturedNS):
    u.grad u + grad p - lap(u)/Re - Ri theta e_y = f        momentum (mom_x, mom_y)
    u_x + v_y = 0                                           continuity (cont)
    u.grad theta - lap(theta)/Pe = Q,   Pe = Re Pr          energy
Walls: velocity Dirichlet (u, v given); thermal  alpha theta + beta dtheta/dn_out = g  per wall.

Problems
    C_mms     manufactured steady solution (exact truth; f, Q derived symbolically from user expressions)
    C_cavity  heated lid-driven cavity with the Case C wall conditions; truth = steady Newton FV (aerosynapse.thermal.pinn.ref2d)
Formulations (``formulation`` spec block)
    kind: primitive       network -> (u, v, p, theta); continuity is a loss term
          streamfunction  network -> (psi, p, theta); u = psi_y, v = -psi_x: divergence-free by construction
                          (needs third derivatives of psi)
    hard_bc               Dirichlet data imposed exactly: field = G + D * N with G the Boolean-sum transfinite
                          interpolant of the wall data and D a product of wall distances that vanishes on those walls.
                          primitive: u, v on all walls; streamfunction: psi = 0 on all walls (impermeability; the
                          tangential velocity stays a loss term); theta: on its Dirichlet (beta = 0) walls
    pressure_gauge        none | point (soft penalty or hard p = N_p(X) - N_p(X0) + value) | mean (soft penalty)
    normalise             none | init_rms (each PDE residual divided by its RMS for the initial network, then fixed)
                          | manual (user scales)
The truth-free validation loss uses the UNNORMALISED primitive residuals and ALL wall conditions, so it is comparable
across formulations (training losses are not: they contain different terms).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp
import torch
from torch import Tensor

from aerosynapse.thermal.data.design import sobol_unit
from aerosynapse.thermal.physics import mms

WALLS = ("left", "right", "bottom", "top")
NORMAL = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "bottom": (0.0, -1.0), "top": (0.0, 1.0)}
RESIDUALS = ("mom_x", "mom_y", "cont", "energy")
FIELDS = ("u", "v", "p", "theta")
Fn = Callable[[Tensor, Tensor], Tensor]


# --- symbolic / profile expressions evaluated on tensors ---------------------------------------------------------
def _num_or_tensor(name: str) -> Callable:
    tf, mf = getattr(torch, name), getattr(math, name)
    return lambda a: tf(a) if isinstance(a, Tensor) else mf(a)


TORCH_NS: dict[str, Any] = {
    n: _num_or_tensor(n)
    for n in ("sin", "cos", "tan", "exp", "log", "sqrt", "sinh", "cosh", "tanh", "atan", "asin", "acos")
}
TORCH_NS.update({"pi": math.pi, "E": math.e, "abs": abs, "Abs": abs})


def torch_fn(expr: sp.Expr) -> Fn:
    """SymPy expression in mms.x, mms.y -> f(x, y) on tensors (constants broadcast)."""
    f = sp.lambdify((mms.x, mms.y), expr, modules=[TORCH_NS])

    def g(xx: Tensor, yy: Tensor) -> Tensor:
        out = f(xx, yy)
        return out if isinstance(out, Tensor) and out.shape == xx.shape else torch.zeros_like(xx) + out

    return g


def parse_expr(s: str | float, path: str) -> sp.Expr:
    try:
        e = sp.sympify(str(s), locals={"x": mms.x, "y": mms.y, "t": mms.t, "pi": sp.pi})
    except (sp.SympifyError, SyntaxError, TypeError) as ex:
        raise ValueError(f"invalid setting at '{path}': cannot parse {s!r} ({ex})") from ex
    extra = e.free_symbols - {mms.x, mms.y}
    if extra:
        raise ValueError(f"invalid setting at '{path}': steady problem, expression may use only x and y, found {extra}")
    return e


def profile_torch(vel: dict[str, Any], s: Tensor) -> Tensor:
    """Torch twin of aerosynapse.thermal.data.spec.eval_profile (steady: t = 0)."""
    kind = vel.get("profile", "zero")
    a = float(vel.get("amplitude", 1.0))
    if kind == "zero":
        return torch.zeros_like(s)
    if kind == "uniform":
        return a * torch.ones_like(s)
    if kind == "cosh":
        C = float(vel.get("C", 50.0))
        return a * (1.0 - torch.cosh(C * (s - 0.5)) / math.cosh(0.5 * C))
    if kind == "parabolic":
        return a * 4.0 * s * (1.0 - s)
    out = eval(vel["expr"], {"__builtins__": {}} | TORCH_NS, {"s": s, "t": 0.0})  # noqa: S307
    return a * (out if isinstance(out, Tensor) else torch.full_like(s, float(out)))


# --- hard constraints ----------------------------------------------------------------------------------------------
def _blend(a: Tensor | None, b: Tensor | None, w: Tensor) -> Tensor | None:
    """Projector along one direction: both lines -> linear blend; one line -> constant extension; none -> None."""
    if a is not None and b is not None:
        return (1.0 - w) * a + w * b
    return a if a is not None else b


def transfinite(data: dict[str, Fn]) -> Fn:
    """Boolean-sum (Coons-type) interpolant G(x, y) of Dirichlet data given on any subset of the unit-square walls.

    G = P + Q - PQ with P the x-projector (left/right lines) and Q the y-projector (bottom/top lines); G equals the
    data on every given wall provided the data are single-valued at corners shared by two given walls."""
    L, R, B, T = (data.get(w) for w in WALLS)

    def q_at(x: Tensor, y: Tensor) -> Tensor | None:
        return _blend(B(x, torch.zeros_like(y)) if B else None, T(x, torch.ones_like(y)) if T else None, y)

    def G(x: Tensor, y: Tensor) -> Tensor:
        z0, z1 = torch.zeros_like(x), torch.ones_like(x)
        P = _blend(L(z0, y) if L else None, R(z1, y) if R else None, x)
        Q = q_at(x, y)
        if P is None:
            return Q if Q is not None else torch.zeros_like(x)
        if Q is None:
            return P
        PQ = _blend(q_at(z0, y) if L else None, q_at(z1, y) if R else None, x)
        return P + Q - PQ

    return G


def distance(walls: tuple[str, ...] | list[str]) -> Fn:
    """Product of normalised wall distances; zero exactly on the given walls, O(1) inside."""
    ws = set(walls)

    def D(x: Tensor, y: Tensor) -> Tensor:
        d = torch.ones_like(x)
        if {"left", "right"} <= ws:
            d = d * 4.0 * x * (1.0 - x)
        elif "left" in ws:
            d = d * x
        elif "right" in ws:
            d = d * (1.0 - x)
        if {"bottom", "top"} <= ws:
            d = d * 4.0 * y * (1.0 - y)
        elif "bottom" in ws:
            d = d * y
        elif "top" in ws:
            d = d * (1.0 - y)
        return d

    return D


CORNERS = {
    ("left", "bottom"): (0.0, 0.0),
    ("right", "bottom"): (1.0, 0.0),
    ("left", "top"): (0.0, 1.0),
    ("right", "top"): (1.0, 1.0),
}


def check_corners(data: dict[str, Fn], what: str, tol: float = 1e-9) -> None:
    """Hard constraints need single-valued data where two constrained walls meet (e.g. a uniform lid is not)."""
    for (a, b), (cx, cy) in CORNERS.items():
        if a in data and b in data:
            X, Y = torch.tensor([cx], dtype=torch.float64), torch.tensor([cy], dtype=torch.float64)
            va, vb = float(data[a](X, Y)[0]), float(data[b](X, Y)[0])
            if abs(va - vb) > tol * max(1.0, abs(va), abs(vb)):
                raise ValueError(
                    f"hard_bc: {what} data are discontinuous at the {a}/{b} corner ({va:g} vs {vb:g}); an exact "
                    "constraint is impossible. Use formulation.hard_bc=false or a regularised profile (e.g. cosh)."
                )


# --- the problem ---------------------------------------------------------------------------------------------------
@dataclass
class WallBC:
    u: Fn
    v: Fn
    alpha: float
    beta: float
    g: Fn


def _grad(f: Tensor, X: Tensor) -> Tensor:
    return torch.autograd.grad(f, X, grad_outputs=torch.ones_like(f), create_graph=True)[0]


@dataclass
class Problem2D:
    name: str
    params: dict[str, float]
    walls: dict[str, WallBC]
    form: dict[str, Any]
    forcing: tuple[Fn, Fn, Fn] | None = None
    truth: dict[str, Any] = field(default_factory=dict)  # x, y (1D), fields (ny, nx), optional nu_top, ref info
    input_names: tuple[str, ...] = ("x", "y")
    lo: tuple[float, ...] = (0.0, 0.0)
    hi: tuple[float, ...] = (1.0, 1.0)
    has_time: bool = False

    def __post_init__(self) -> None:
        f = self.form
        self.kind = f["kind"]
        self.hard = bool(f["hard_bc"])
        self.gauge = f["pressure_gauge"]
        self.scales = {k: 1.0 for k in RESIDUALS}
        if f["normalise"]["kind"] == "manual":
            self.scales.update({k: float(v) for k, v in f["normalise"]["scales"].items()})
        self.Re, self.Ri = float(self.params["Re"]), float(self.params.get("Ri", 0.0))
        self.Pe = self.Re * float(self.params["Pr"])
        self.th_dir = [w for w in WALLS if self.walls[w].beta == 0.0]
        if self.hard:
            if self.kind == "primitive":
                ud = {w: self.walls[w].u for w in WALLS}
                vd = {w: self.walls[w].v for w in WALLS}
                check_corners(ud, "u")
                check_corners(vd, "v")
                self.Gu, self.Gv, self.Dvel = transfinite(ud), transfinite(vd), distance(WALLS)
            else:
                self.Dvel = distance(WALLS)
            if self.th_dir:
                td = {w: self._theta_dirichlet(w) for w in self.th_dir}
                check_corners(td, "theta")
                self.Gt, self.Dt = transfinite(td), distance(self.th_dir)
        if self.gauge["kind"] == "point":
            # float64 so the gauge point is exact (a float32 0.3 is 0.30000001); .to(X) casts to the network dtype
            self.X0 = torch.tensor([[float(self.gauge["x"]), float(self.gauge["y"])]], dtype=torch.float64)

    def _theta_dirichlet(self, w: str) -> Fn:
        bc = self.walls[w]
        return lambda x, y, bc=bc: bc.g(x, y) / bc.alpha

    @property
    def out_dim(self) -> int:
        return 4 if self.kind == "primitive" else 3

    @property
    def p_index(self) -> int:
        return 2 if self.kind == "primitive" else 1

    # --- network outputs -> fields ---------------------------------------------------------------------------
    def transform(self, model: torch.nn.Module, X: Tensor) -> dict[str, Tensor]:
        out = model(X)
        x, y = X[:, 0], X[:, 1]
        F: dict[str, Tensor] = {}
        if self.kind == "primitive":
            u, v, p, th = out.unbind(-1)
            if self.hard:
                D = self.Dvel(x, y)
                u, v = self.Gu(x, y) + D * u, self.Gv(x, y) + D * v
            F["u"], F["v"] = u, v
        else:
            psi, p, th = out.unbind(-1)
            F["psi"] = self.Dvel(x, y) * psi if self.hard else psi
        if self.hard and self.th_dir:
            th = self.Gt(x, y) + self.Dt(x, y) * th
        if self.gauge["kind"] == "point" and self.gauge["mode"] == "hard":
            p = p - model(self.X0.to(X))[0, self.p_index] + float(self.gauge["value"])
        F["p"], F["theta"] = p, th
        return F

    def fields(self, model: torch.nn.Module, X: Tensor, order: int = 2) -> dict[str, Tensor]:
        """order 0: u, v, p, theta; 1: + first derivatives of u, v, theta; 2: + second derivatives and grad p."""
        X = X.detach().clone().requires_grad_(True)
        F = self.transform(model, X)
        d: dict[str, Tensor] = {"p": F["p"], "theta": F["theta"], "x": X[:, 0], "y": X[:, 1]}
        if self.kind == "primitive":
            d["u"], d["v"] = F["u"], F["v"]
            if order >= 1:
                for k in ("u", "v"):
                    g = _grad(d[k], X)
                    d[k + "_x"], d[k + "_y"] = g[:, 0], g[:, 1]
        else:
            d["psi"] = F["psi"]
            g = _grad(F["psi"], X)
            d["u"], d["v"] = g[:, 1], -g[:, 0]
            if order >= 1:
                gu, gv = _grad(d["u"], X), _grad(d["v"], X)
                d["u_x"], d["u_y"], d["v_x"], d["v_y"] = gu[:, 0], gu[:, 1], gv[:, 0], gv[:, 1]
        if order >= 1:
            g = _grad(d["theta"], X)
            d["theta_x"], d["theta_y"] = g[:, 0], g[:, 1]
        if order >= 2:
            for k in ("u", "v", "theta"):
                d[k + "_xx"] = _grad(d[k + "_x"], X)[:, 0]
                d[k + "_yy"] = _grad(d[k + "_y"], X)[:, 1]
            g = _grad(d["p"], X)
            d["p_x"], d["p_y"] = g[:, 0], g[:, 1]
        return d

    # --- residuals ----------------------------------------------------------------------------------------------
    def residuals(self, d: dict[str, Tensor]) -> dict[str, Tensor]:
        """Unnormalised primitive-variable residuals (identical definition for both formulations)."""
        u, v, th = d["u"], d["v"], d["theta"]
        if self.forcing is not None:
            fx, fy, Q = (f(d["x"], d["y"]) for f in self.forcing)
        else:
            fx = fy = Q = torch.zeros_like(u)
        return {
            "mom_x": u * d["u_x"] + v * d["u_y"] + d["p_x"] - (d["u_xx"] + d["u_yy"]) / self.Re - fx,
            "mom_y": u * d["v_x"] + v * d["v_y"] + d["p_y"] - (d["v_xx"] + d["v_yy"]) / self.Re - self.Ri * th - fy,
            "cont": d["u_x"] + d["v_y"],
            "energy": u * d["theta_x"] + v * d["theta_y"] - (d["theta_xx"] + d["theta_yy"]) / self.Pe - Q,
        }

    def wall_residuals(self, model: torch.nn.Module, walls: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        """Per wall: u - u_w, v - v_w, alpha theta + beta dtheta/dn_out - g (all, regardless of hard constraints)."""
        out = {}
        for w, Xw in walls.items():
            d = self.fields(model, Xw, order=1)
            bc = self.walls[w]
            x, y = d["x"], d["y"]
            nx, ny = NORMAL[w]
            dn = nx * d["theta_x"] + ny * d["theta_y"]
            out[w] = {
                "u": d["u"] - bc.u(x, y),
                "v": d["v"] - bc.v(x, y),
                "theta": bc.alpha * d["theta"] + bc.beta * dn - bc.g(x, y),
            }
        return out

    def _imposed_softly(self, w: str, comp: str) -> bool:
        if not self.hard:
            return True
        if comp == "theta":
            return w not in self.th_dir
        if self.kind == "primitive":
            return False
        tangential = "u" if w in ("bottom", "top") else "v"
        return comp == tangential  # psi = 0 on walls makes the NORMAL velocity exactly zero

    # --- losses (Trainer interface) --------------------------------------------------------------------------
    def residual_vectors(self, model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Per-point residual of every TRAINED loss component; loss_k = mean(r_k^2) (normalised PDE residuals)."""
        d = self.fields(model, batch["pde"], order=2)
        r = self.residuals(d)
        out: dict[str, Tensor] = {}
        for k in RESIDUALS:
            if k == "cont" and self.kind == "streamfunction":
                continue  # identically zero (up to round-off) by construction
            out["pde_" + k] = r[k] / self.scales[k]
        wr = self.wall_residuals(model, batch["walls"])
        for c in ("u", "v", "theta"):
            parts = [wr[w][c] for w in WALLS if self._imposed_softly(w, c)]
            if parts:
                out["bc_" + c] = torch.cat(parts)
        g = self.gauge
        if g["kind"] == "point" and g["mode"] == "soft":
            out["gauge"] = self.transform(model, self.X0.to(batch["pde"]))["p"] - float(g["value"])
        elif g["kind"] == "mean":
            out["gauge"] = (d["p"].mean() - float(g["value"])).reshape(1)
        return out

    def loss_components(self, model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Tensor]:
        return {k: (v**2).mean() for k, v in self.residual_vectors(model, batch).items()}

    def weighted_total(self, comp: dict[str, Tensor], weights: dict[str, float]) -> Tensor:
        tot = sum(weights["pde"] * v for k, v in comp.items() if k.startswith("pde_"))
        tot = tot + sum(weights["bc"] * v for k, v in comp.items() if k.startswith("bc_"))
        if "gauge" in comp:
            tot = tot + weights.get("gauge", 1.0) * comp["gauge"]
        return tot

    def sample(
        self, n_pde: int, n_ic: int, n_bc: int, method: str, seed: int, device: Any, dtype: Any
    ) -> dict[str, Any]:
        """Interior points (n_pde) and n_bc points on each wall; n_ic is unused (steady)."""

        def unit(n: int, k: int, s: int) -> np.ndarray:
            if method == "sobol":
                return sobol_unit(n, k, s)
            if method == "lhs":  # Latin hypercube (Step 4.4)
                from scipy.stats import qmc

                return qmc.LatinHypercube(d=k, seed=s).random(n)
            return np.random.default_rng(s).uniform(size=(n, k))

        mk = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)  # noqa: E731
        walls = {}
        for j, w in enumerate(WALLS):
            s = unit(n_bc, 1, seed + 2 + j)[:, 0]
            walls[w] = mk(wall_points(w, s))
        return {"pde": mk(unit(n_pde, 2, seed)), "walls": walls}

    # --- normalisation ------------------------------------------------------------------------------------------
    def fit_normalisation(self, model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, float]:
        """init_rms: scale_k = RMS of residual k for the given (initial) network on the batch, then frozen."""
        if self.form["normalise"]["kind"] != "init_rms":
            return dict(self.scales)
        r = self.residuals(self.fields(model, batch["pde"], order=2))
        for k in RESIDUALS:
            rms = float(torch.sqrt((r[k].detach() ** 2).mean()))
            self.scales[k] = rms if rms > 1e-12 else 1.0
        return dict(self.scales)


def wall_points(w: str, s: np.ndarray) -> np.ndarray:
    z, o = np.zeros_like(s), np.ones_like(s)
    return {
        "left": np.stack([z, s], 1),
        "right": np.stack([o, s], 1),
        "bottom": np.stack([s, z], 1),
        "top": np.stack([s, o], 1),
    }[w]


# --- builders ------------------------------------------------------------------------------------------------------
def build_mms(pcfg: dict[str, Any], form: dict[str, Any], n_grid: int) -> Problem2D:
    base = "problems2d.C_mms"
    par = {k: float(v) for k, v in pcfg["parameters"].items()}
    ex = pcfg["exact"]
    psi, p, th = (parse_expr(ex[k], f"{base}.exact.{k}") for k in ("psi", "p", "theta"))
    m = mms.manufacture_ns(psi, p, par["Re"], theta=th, Ri=par.get("Ri", 0.0), Pe=par["Re"] * par["Pr"])
    sub = lambda e: e.subs(mms.t, 0)  # noqa: E731  (steady: expressions have no t)
    fn = {k: torch_fn(sub(getattr(m, k))) for k in ("u", "v", "p", "fx", "fy", "Q")}
    fn["theta"], fn["psi"] = torch_fn(th), torch_fn(psi)
    walls = {}
    for w in WALLS:
        a, b = (float(q) for q in pcfg["thermal_bc"][w])
        nx, ny = NORMAL[w]
        g = a * th + b * (nx * sp.diff(th, mms.x) + ny * sp.diff(th, mms.y))
        walls[w] = WallBC(fn["u"], fn["v"], a, b, torch_fn(g))
    if form["hard_bc"] and form["kind"] == "streamfunction":
        s = torch.linspace(0, 1, 101, dtype=torch.float64)
        worst = max(float(fn["psi"](*torch.as_tensor(wall_points(w, s.numpy())).unbind(1)).abs().max()) for w in WALLS)
        if worst > 1e-10:
            raise ValueError(
                f"hard_bc with the streamfunction formulation imposes psi = 0 on all walls, but the exact psi of "
                f"{base} reaches {worst:.3g} there; use hard_bc=false or a psi that vanishes on the walls"
            )
    xs = np.linspace(0.0, 1.0, n_grid)
    X, Y = np.meshgrid(xs, xs)
    tX, tY = torch.as_tensor(X), torch.as_tensor(Y)
    truth = {
        "x": xs,
        "y": xs,
        "fields": {k: fn[k](tX, tY).numpy() for k in FIELDS},
        "kind": "exact (manufactured)",
        "exact_fn": fn,  # torch callables (u, v, p, theta, psi, fx, fy, Q); not serialised
    }
    return Problem2D("C_mms", par, walls, form, (fn["fx"], fn["fy"], fn["Q"]), truth)


def cavity_conditions(pcfg: dict[str, Any]) -> dict[str, Any]:
    if pcfg["conditions"] == "data":
        from aerosynapse.thermal.data import spec as DS

        return DS.load_spec(pcfg.get("data_config") or None, "smoke")["cases"]["caseC"]["conditions"]
    return pcfg["conditions"]


def build_cavity(pcfg: dict[str, Any], form: dict[str, Any], reference: dict[str, Any] | None = None) -> Problem2D:
    par = {k: float(v) for k, v in pcfg["parameters"].items()}
    cond = cavity_conditions(pcfg)
    walls = {}
    for w in WALLS:
        vel = cond["velocity"].get(w, {"profile": "zero"})
        along = (lambda x, y: x) if w in ("bottom", "top") else (lambda x, y: y)
        moving = lambda x, y, vel=vel, along=along: profile_torch(vel, along(x, y))  # noqa: E731
        zero = lambda x, y: torch.zeros_like(x)  # noqa: E731
        u_w, v_w = (moving, zero) if w in ("bottom", "top") else (zero, moving)
        t = cond["thermal"][w]
        if t["type"] == "dirichlet":
            a, b, gval = 1.0, 0.0, float(t.get("value", 0.0))
        elif t["type"] == "neumann":
            a, b, gval = 0.0, 1.0, float(t.get("value", 0.0))
        else:
            Bi = par[t["Bi"]] if isinstance(t["Bi"], str) else float(t["Bi"])
            a, b, gval = Bi, 1.0, Bi * float(t.get("theta_inf", 0.0))
        walls[w] = WallBC(u_w, v_w, a, b, lambda x, y, gval=gval: torch.full_like(x, gval))
    if reference is None:
        from aerosynapse.thermal.pinn.ref2d import cavity_reference

        rc = pcfg["reference"]
        reference = cavity_reference(par, cond, int(rc["grid"]), int(rc["refine"]), float(rc["tol"]))
    truth = {
        "x": reference["x"],
        "y": reference["y"],
        "fields": {k: reference[k] for k in FIELDS},
        "nu_top": reference["nu_top"],
        "ref_diff": reference.get("coarse_fine_diff", {}),
        "kind": f"steady Newton FV {reference.get('n_fine', '?')}^2 (central, bc_order 2)",
    }
    return Problem2D("C_cavity", par, walls, form, None, truth)


def build_problem2d(
    name: str, spec: dict[str, Any], form: dict[str, Any] | None = None, reference: dict[str, Any] | None = None
) -> Problem2D:
    form = form or spec["formulation"]
    pcfg = spec["problems2d"][name]
    if name == "C_mms":
        return build_mms(pcfg, form, int(spec["formulation_study"]["evaluation"]["n_grid"]))
    if name == "C_cavity":
        return build_cavity(pcfg, form, reference)
    raise ValueError(f"unknown 2D problem {name!r}; available: C_mms, C_cavity")


# --- evaluation ----------------------------------------------------------------------------------------------------
def _batched(
    problem: Problem2D, model: torch.nn.Module, X: Tensor, order: int, keys: tuple[str, ...], chunk: int = 4096
):
    out: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for i in range(0, X.shape[0], chunk):
        d = problem.fields(model, X[i : i + chunk], order=order)
        for k in keys:
            out[k].append(d[k].detach().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def evaluate2d(problem: Problem2D, model: torch.nn.Module, n_wall: int = 200, full: bool = True) -> dict[str, Any]:
    """Errors vs the truth on its grid. Pressure is compared mean-free (gauge-invariant)."""
    p0 = next(model.parameters())
    tr = problem.truth
    X, Y = np.meshgrid(tr["x"], tr["y"])
    pts = torch.as_tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=p0.dtype, device=p0.device)
    keys = FIELDS + (("u_x", "v_y", "theta_y") if full else ())
    pred = _batched(problem, model, pts, 1 if full else 0, keys)
    res: dict[str, Any] = {}
    for k in FIELDS:
        ref = tr["fields"][k]
        pk = pred[k].reshape(ref.shape)
        if k == "p":
            ref, pk = ref - ref.mean(), pk - pk.mean()
        e = pk - ref
        res[f"rel_l2_{k}"] = float(np.linalg.norm(e) / max(np.linalg.norm(ref), 1e-300))
        res[f"linf_{k}"] = float(np.abs(e).max())
    res["rel_l2_max"] = max(res["rel_l2_u"], res["rel_l2_v"], res["rel_l2_theta"])
    if not full:
        return res
    res["max_div"] = float(np.abs(pred["u_x"] + pred["v_y"]).max())
    s = np.linspace(0.0, 1.0, n_wall)
    walls = {w: torch.as_tensor(wall_points(w, s), dtype=p0.dtype, device=p0.device) for w in WALLS}
    wr = problem.wall_residuals(model, walls)
    mx = lambda parts: max((float(t.detach().abs().max()) for t in parts), default=float("nan"))  # noqa: E731
    normal = {"left": "u", "right": "u", "bottom": "v", "top": "v"}
    res["bc_max_normal"] = mx([wr[w][normal[w]] for w in WALLS])
    res["bc_max_tangential"] = mx([wr[w]["v" if normal[w] == "u" else "u"] for w in WALLS])
    res["bc_max_theta_dirichlet"] = mx([wr[w]["theta"] for w in WALLS if w in problem.th_dir])
    res["bc_max_theta_other"] = mx([wr[w]["theta"] for w in WALLS if w not in problem.th_dir])
    if "nu_top" in tr:
        top = torch.as_tensor(wall_points("top", np.asarray(tr["x"])), dtype=p0.dtype, device=p0.device)
        nu = float(problem.fields(model, top, order=1)["theta_y"].detach().mean())
        res["nu_top"], res["nu_top_ref"] = nu, float(tr["nu_top"])
        res["nu_top_rel_err"] = abs(nu - tr["nu_top"]) / max(abs(tr["nu_top"]), 1e-300)
    return res


def validation_loss(problem: Problem2D, model: torch.nn.Module, n_val: int, n_wall: int, seed: int) -> dict[str, float]:
    """Truth-free and formulation-independent: mean squares of the UNNORMALISED primitive residuals (incl.
    continuity) on an independent point set + mean squares of ALL wall conditions (incl. hard-constrained ones)."""
    p0 = next(model.parameters())
    b = problem.sample(n_val, 0, n_wall, "sobol", seed, p0.device, p0.dtype)
    out: dict[str, float] = {}
    for i in range(0, n_val, 2048):
        r = problem.residuals(problem.fields(model, b["pde"][i : i + 2048], order=2))
        for k in RESIDUALS:
            out["val_" + k] = out.get("val_" + k, 0.0) + float((r[k].detach() ** 2).sum()) / n_val
    pde = sum(out["val_" + k] for k in RESIDUALS)
    wr = problem.wall_residuals(model, b["walls"])
    bc = 0.0
    for c in ("u", "v", "theta"):
        out["val_bc_" + c] = float(torch.cat([wr[w][c].detach() for w in wr]).pow(2).mean())
        bc += out["val_bc_" + c]
    out.update({"val_pde": pde, "val_bc": bc, "val_loss": pde + bc})
    return out
