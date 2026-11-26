"""Phase 8 backend: everything the Streamlit app (src/app/main.py) and the CLI (src/app/cli.py) compute, as plain
functions without any user-interface code, so both front-ends and the tests run the same code.

No network is trained here. PINN predictions come from SAVED checkpoints:
  A_steady, B_slab   the Step 6.2 deep ensembles (results/p6_2_uq_default/members, 8 members, parametric in Pe / Bi)
  D_graetz           the Step 4.7 'full-exact_psi' channel networks (results/p4_7_ablation_smoke, 3 seeds, Re, Pr, Bi)
The heated cavity has no usable PINN (Step 6.7: 121 % error), so it is solver-only.
Solvers are the verified Phase 1 codes: src/numerics/fd1d.py (1D), src/pinn/channel2d.py (steady channel, Newton +
QUICK energy) and src/numerics/ns_projection.py (cavity, MAC projection to steady state).

Every case and every condition is user-definable through the parameters of these functions; the defaults are the
project's default cases. Validity messages (validity()) are part of every result.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
UQ_RUN = ROOT / "results" / "p6_2_uq_default"
GRAETZ_RUN = {"run": "results/p4_7_ablation_smoke", "configuration": "full-exact_psi"}


# ============================================================================ cases
@dataclass(frozen=True)
class Param:
    name: str
    default: float
    lo: float  # widest value the UI accepts
    hi: float
    train: tuple[float, float] | None = None  # PINN training range (None: no PINN for this parameter)
    log: bool = True
    help: str = ""


@dataclass(frozen=True)
class Case:
    key: str
    title: str
    dim: int
    transient: bool
    equation: str
    params: tuple[Param, ...]
    pinn: str | None  # where the PINN comes from, or None
    pinn_note: str = ""
    conditions: dict = field(default_factory=dict)  # user-definable conditions with their defaults

    def defaults(self) -> dict[str, float]:
        return {p.name: p.default for p in self.params}


CASES: dict[str, Case] = {
    "A_steady": Case(
        "A_steady",
        "Steady convection-diffusion (1D)",
        1,
        False,
        r"\theta' = \mathrm{Pe}^{-1}\,\theta'' + q,\quad \theta(0)=\theta_L,\ \theta(1)=\theta_R",
        (Param("Pe", 10.0, 0.1, 1000.0, (1.0, 20.0), help="Péclet number: convection / diffusion"),),
        "Step 6.2 deep ensemble (8 members), parametric in Pe on [1, 20]",
        "trained with theta_L = 0, theta_R = 1, q = 0",
        {"theta_left": 0.0, "theta_right": 1.0, "source": 0.0},
    ),
    "B_slab": Case(
        "B_slab",
        "Heated slab cooled at its surface (1D, transient)",
        1,
        True,
        r"\theta_t = \theta_{xx} - m^2\theta + Q,\ \theta_x(0)=0,\ \theta_x(1) + \mathrm{Bi}\,\theta(1) = 0,"
        r"\ \theta(x,0)=\theta_0",
        (Param("Bi", 1.0, 0.01, 100.0, (0.3, 3.0), help="Biot number hL/k: surface cooling / conduction"),),
        "Step 6.2 deep ensemble (8 members), parametric in Bi on [0.3, 3]",
        "trained with Q = 1, m = 1, theta0 = 1, t in [0, 0.5]",
        {"Q": 1.0, "m": 1.0, "theta0": 1.0, "T": 0.5},
    ),
    "D_graetz": Case(
        "D_graetz",
        "Graetz channel: developing flow and heat transfer (2D)",
        2,
        False,
        r"\text{steady Navier-Stokes + energy in an 8 x 1 channel; walls cooled with Biot condition}",
        (
            Param("Re", 30.0, 1.0, 2000.0, (10.0, 63.0957), help="Reynolds number (channel height, mean velocity)"),
            Param("Pr", 1.0, 0.1, 50.0, (0.7, 1.6212), help="Prandtl number"),
            Param("Bi", 5.0, 0.01, 100.0, (0.5, 19.9054), help="wall Biot number"),
        ),
        "Step 4.7 'full-exact_psi' networks (3 seeds), parametric in (Re, Pr, Bi)",
        "smoke-budget networks: mean Nusselt number error 4.7 % (Step 6.7)",
        {"cells_per_unit": 16},
    ),
    "cavity": Case(
        "cavity",
        "Heated lid-driven cavity (2D, solver only)",
        2,
        False,
        r"\text{Navier-Stokes + energy, lid } u=1,\ \theta_{lid}=1,\ \theta_{bottom}=0",
        (
            Param("Re", 100.0, 1.0, 20000.0, None, help="Reynolds number (lid speed, cavity size)"),
            Param("Pr", 0.71, 0.1, 50.0, None, help="Prandtl number"),
        ),
        None,
        "no usable PINN: the parametric cavity PINN has 121 % error (Step 6.7)",
        {"N": 32},
    ),
}


def case(key: str) -> Case:
    if key not in CASES:
        raise KeyError(f"unknown case {key!r}; choose from {sorted(CASES)}")
    return CASES[key]


# ============================================================================ validity guard
def _metrics(rel: str) -> dict | None:
    f = ROOT / rel
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def validity(key: str, params: dict, grid: dict | None = None) -> list[dict]:
    """Messages {level: ok | warn | stop, text} about the request: PINN training range, verified regime of the
    solver, grid resolution. 'stop' = the PINN answer must not be used as a prediction (shown as extrapolation)."""
    c, out = case(key), []
    grid = grid or {}
    for p in c.params:
        v = float(params.get(p.name, p.default))
        if c.pinn and p.train:
            lo, hi = p.train
            if lo <= v <= hi:
                out.append(
                    {"level": "ok", "text": f"{p.name} = {v:g} is inside the PINN training range [{lo:g}, {hi:g}]."}
                )
            else:
                f = v / hi if v > hi else lo / v
                out.append(
                    {
                        "level": "stop",
                        "text": f"EXTRAPOLATION: {p.name} = {v:g} is outside the PINN training range [{lo:g}, {hi:g}] "
                        f"(factor {f:.3g} beyond). Step 6.3: errors grow by orders of magnitude outside the range; "
                        "use the solver, or read the ensemble spread as a warning sign, not as an error bar.",
                    }
                )
    if key == "cavity":
        Re = float(params.get("Re", 100.0))
        m = _metrics("results/p6_3_extrapolation_default/metrics.json")
        crit = (m or {}).get("cavity", {}).get("critical_Re", {})
        finest = max((int(k) for k, v in crit.items() if v), default=None)
        if Re <= 1000:
            out.append(
                {"level": "ok", "text": f"Re = {Re:g}: inside the range verified against Ghia et al. (Re <= 1000)."}
            )
        elif finest is not None and Re >= float(crit[str(finest)]):
            out.append(
                {
                    "level": "stop",
                    "text": f"Re = {Re:g} is above the critical Reynolds number of the steady 2D solution "
                    f"(Re_c = {float(crit[str(finest)]):.0f} on {finest}² cells, Step 6.3): the flow is not steady and "
                    "not laminar-verified. Any result here is an explicit extrapolation study, not a valid prediction.",
                }
            )
        else:
            out.append(
                {
                    "level": "warn",
                    "text": f"Re = {Re:g} is above the verified range (Re <= 1000); thin boundary layers need finer "
                    "grids. Treat as unverified.",
                }
            )
        N = int(grid.get("N", c.conditions["N"]))
        h_cell = 1.0 / N
        if Re * h_cell > 20:
            out.append(
                {"level": "warn", "text": f"cell Reynolds number {Re * h_cell:.3g} on {N}² cells: refine the grid."}
            )
    if key == "A_steady":
        Pe = float(params.get("Pe", 10.0))
        N = int(grid.get("N", 40))
        if grid.get("scheme", "central") == "central" and Pe / N > 2:
            out.append(
                {
                    "level": "warn",
                    "text": f"cell Péclet number Pe/N = {Pe / N:.3g} > 2: central differences oscillate; refine the "
                    "grid or use upwind / QUICK.",
                }
            )
    if key == "D_graetz":
        Re = float(params.get("Re", 30.0))
        if Re > 2000:
            out.append({"level": "stop", "text": "Re > 2000: channel flow is no longer laminar; not modelled."})
    return out


# ============================================================================ solvers (verified Phase 1 codes)
def solve_steady(
    Pe: float,
    N: int = 40,
    scheme: str = "central",
    theta_left: float = 0.0,
    theta_right: float = 1.0,
    source: float = 0.0,
) -> dict:
    """1D steady convection-diffusion theta' = theta''/Pe + q on [0, 1] with Dirichlet ends (fd1d), and the exact
    solution where one exists (q = 0)."""
    from aerosynapse.thermal.numerics import fd1d
    from aerosynapse.thermal.physics import analytic as an

    t0 = time.perf_counter()
    f = fd1d.ConvDiff1D(
        int(N),
        a=1.0,
        D=1.0 / float(Pe),
        source=(lambda X, t: float(source) + 0 * X) if source else None,
        left=fd1d.dirichlet(float(theta_left)),
        right=fd1d.dirichlet(float(theta_right)),
        scheme=scheme,
    )
    u = f.nodal(f.steady(), 0.0)
    wall = time.perf_counter() - t0
    xf = np.linspace(0, 1, 801)
    out = {"x": f.x, "theta": u, "wall_s": wall, "x_fine": xf, "N": int(N), "scheme": scheme}
    if not source:
        ex = lambda x: theta_left + (theta_right - theta_left) * an.steady_convdiff(x, float(Pe))  # noqa: E731
        out["exact"] = ex(xf)
        out["err_inf"] = float(np.max(np.abs(u - ex(f.x))))
    return out


def solve_slab(
    Bi: float,
    Q: float = 1.0,
    m: float = 1.0,
    theta0: float = 1.0,
    T: float = 0.5,
    N: int = 40,
    n_out: int = 6,
    dt: float | None = None,
) -> dict:
    """1D transient slab theta_t = theta_xx - m^2 theta + Q, symmetry at x = 0, Biot cooling at x = 1 (fd1d +
    Crank-Nicolson with Rannacher start-up, as src/inverse/body.py: the first two steps are four implicit-Euler
    half-steps, which damp the stiff modes of the uniform initial state meeting the Robin boundary), with the exact
    eigen-series solution at the output times."""
    from aerosynapse.thermal.numerics import fd1d
    from aerosynapse.thermal.numerics.time_integrators import integrate
    from aerosynapse.thermal.physics import analytic as an

    t0 = time.perf_counter()
    p = fd1d.ConvDiff1D(
        int(N),
        a=0.0,
        D=1.0,
        s=float(m) ** 2,
        source=lambda X, t: float(Q) + 0 * X,
        left=fd1d.neumann(0.0),
        right=fd1d.robin(float(Bi)),
        scheme="central",
    )
    ts = np.linspace(0, float(T), int(n_out))
    # Crank-Nicolson is second order in time: dt proportional to h keeps the time error O(h^2) like the space error
    dt = float(dt) if dt else 0.2 / int(N)
    y = float(theta0) * np.ones(p.idx.size)
    rows, tc = [p.nodal(y, 0.0)], 0.0
    dt = min(dt, float(ts[1]) / 3)  # at least three steps before the first output, so the start-up fits
    y = integrate(p, y, 0.0, 2.0 * dt, 0.5 * dt, "implicit_euler")
    tc = 2.0 * dt
    for tt in ts[1:]:
        y = integrate(p, y, tc, float(tt), dt, "crank_nicolson")
        tc = float(tt)
        rows.append(p.nodal(y, tc))
    wall = time.perf_counter() - t0
    th = np.stack(rows)
    ex = np.stack([an.slab_transient(p.x, float(tt), float(Bi), float(Q), float(m), float(theta0)) for tt in ts])
    return {
        "x": p.x,
        "t": ts,
        "theta": th,
        "exact": ex,
        "err_inf": float(np.max(np.abs(th - ex)[1:])),
        "wall_s": wall,
        "N": int(N),
        "dt": dt,
    }


def graetz_design() -> dict:
    """The Graetz channel as the PINN was trained on it (geometry, fixed conditions, solver settings)."""
    ad = graetz_adapter(load_networks=False)
    return {"pcfg": ad["pcfg"], "fixed": ad["fixed"], "Lx": ad["Lx"]}


def solve_graetz(Re: float, Pr: float, Bi: float, cells_per_unit: int = 16) -> dict:
    """Steady channel flow + energy (src/pinn/channel2d.solve_channel: Newton flow, QUICK energy), mean Nusselt
    number on the bottom wall = - mean dtheta/dy."""
    from aerosynapse.thermal.pinn import channel2d as CH

    d = graetz_design()
    r = CH.solve_channel(
        {**d["fixed"], "Re": float(Re), "Pr": float(Pr), "Bi": float(Bi)},
        d["pcfg"],
        int(cells_per_unit),
        1e-10,
        "quick",
    )
    return {
        "x": r["x"],
        "y": r["y"],
        "theta": r["theta"],
        "u": r["u"],
        "v": r["v"],
        "Nu": -float(r["q_bottom"]),
        "wall_s": float(r["wall_s"]),
        "newton_iterations": int(r["newton_iterations"]),
        "kind": r["kind"],
        "max_div": float(r["max_div"]),
        "cells_per_unit": int(cells_per_unit),
    }


def solve_cavity(Re: float, Pr: float = 0.71, N: int = 32, tol: float = 1e-10, scheme: str = "quick") -> dict:
    """Heated lid-driven cavity, steady: flow by steady Newton on the MAC grid (exact Jacobian, continuation in Re
    from rest if needed; Phase 1.7), then the energy equation on the converged flow (lid theta = 1, bottom
    theta = 0, adiabatic sides; QUICK, second-order walls). Lid u = 1 (uniform, as in the Ghia comparison)."""
    from aerosynapse.thermal.numerics import fvm2d as fv
    from aerosynapse.thermal.numerics import ns_projection as ns
    from aerosynapse.thermal.numerics.ns_steady import SteadyNewtonNS

    t0 = time.perf_counter()
    n = int(N)
    s = ns.MACNavierStokes2D(fv.Grid2D(n, n), float(Re), {"top": ns.wall(u=1.0)})
    nw = SteadyNewtonNS(s)
    info = nw.solve(nw.pack(s.u, s.v, None, None), tol=float(tol), max_iter=40)
    ramp: list[float] = []
    if not info["converged"]:
        x = nw.pack(np.zeros_like(s.u), np.zeros_like(s.v), None, None)
        for r in (Re / 8, Re / 4, Re / 2, Re):
            s.Re = float(r)
            info = nw.solve(x, tol=float(tol), max_iter=40)
            x = nw.x
            ramp.append(float(r))
    nw.store()
    E = fv.FVEnergy2D(
        s.g,
        1.0 / (float(Re) * float(Pr)),
        fv.FaceVelocity.from_arrays(s.g, s.u, s.v),
        bcs={
            "top": fv.dirichlet2(1.0),
            "bottom": fv.dirichlet2(0.0),
            "left": fv.neumann2(0.0),
            "right": fv.neumann2(0.0),
        },
        scheme=scheme,
        bc_order=2,
    )
    th = E.steady().reshape(n, n)
    U, V = s.cell_velocity()
    psi = s.streamfunction()
    return {
        "x": s.g.xc.copy(),
        "y": s.g.yc.copy(),
        "theta": th,
        "u": U,
        "v": V,
        "psi": psi,
        "vorticity": s.vorticity(),
        "psi_min": float(np.min(psi)),
        "converged": bool(info["converged"]),
        "newton_residual": float(info["residual_history"][-1]),
        "continuation": ramp,
        "Nu_lid": -float(np.mean(E.wall_gradient(th.ravel(), 0.0, "top"))),
        "max_div": float(s.max_div()),
        "wall_s": time.perf_counter() - t0,
        "N": n,
    }


def solve(key: str, params: dict, opts: dict | None = None) -> dict:
    """The verified solver of a case with the user's parameters and options."""
    o = {**case(key).conditions, **(opts or {})}
    if key == "A_steady":
        return solve_steady(
            params["Pe"], o.get("N", 40), o.get("scheme", "central"), o["theta_left"], o["theta_right"], o["source"]
        )
    if key == "B_slab":
        return solve_slab(params["Bi"], o["Q"], o["m"], o["theta0"], o["T"], o.get("N", 40), o.get("n_out", 6))
    if key == "D_graetz":
        return solve_graetz(params["Re"], params["Pr"], params["Bi"], o["cells_per_unit"])
    return solve_cavity(params["Re"], params.get("Pr", 0.71), o["N"])


# ============================================================================ PINN ensembles (saved checkpoints)
_CACHE: dict = {}


def pinn_available(key: str) -> tuple[bool, str]:
    """Whether the saved networks of a case exist on this machine (checkpoints are git-ignored)."""
    c = case(key)
    if not c.pinn:
        return False, c.pinn_note
    if key in ("A_steady", "B_slab"):
        met = _metrics("results/p6_2_uq_default/metrics.json")
        if not met:
            return False, "results/p6_2_uq_default is missing"
        dirs = [Path(m["dir"]) for m in met["members"] if m["problem"] == key and m["kind"] == "ensemble"]
        miss = [d for d in dirs if not (d / "checkpoint.pt").exists()]
        return (not miss and bool(dirs)), (f"{len(miss)} checkpoints missing" if miss else f"{len(dirs)} members")
    run = ROOT / GRAETZ_RUN["run"] / "runs"
    ok = all((run / f"{GRAETZ_RUN['configuration']}_s{s}" / "checkpoint.pt").exists() for s in range(3))
    return ok, ("3 members" if ok else "checkpoints of results/p4_7_ablation_smoke missing")


def ensemble_1d(key: str) -> list:
    """[(record, problem, network)] of the Step 6.2 ensemble members of a 1D case (loaded once per process)."""
    if key not in _CACHE:
        from aerosynapse.thermal.experiments.p6_3_extrapolation import load_members

        _, mem = load_members(UQ_RUN, key)
        _CACHE[key] = [(m, p, n) for m, p, n in mem if m["kind"] == "ensemble"]
    return _CACHE[key]


def graetz_adapter(load_networks: bool = True) -> dict:
    """The Step 4.7 channel networks (adapter of Step 6.4) or only the design they were trained on."""
    k = "D_graetz" if load_networks else "D_graetz_design"
    if k not in _CACHE:
        if load_networks:
            from aerosynapse.thermal.experiments.p6_4_sensitivity import Graetz

            ad = Graetz({"surrogate": GRAETZ_RUN})
            _CACHE[k] = {"members": ad.members, "pcfg": ad.pcfg, "fixed": ad.fixed, "Lx": ad.Lx, "box": ad.box}
        else:
            from aerosynapse.thermal.experiments.p4_5b_channel import case_design
            from aerosynapse.thermal.pinn import spec as SP

            run = ROOT / GRAETZ_RUN["run"]
            base = SP.load_spec([str(run / "config.yaml")], "smoke")
            ab = base["ablation_study"]
            des = case_design(base, ab["case"])
            pcfg = SP.deep_merge(base["problems2d"][ab["case"]], (ab.get("problems") or {}).get(ab["case"]) or {})
            _CACHE[k] = {
                "pcfg": pcfg,
                "fixed": dict(des["fixed"]),
                "Lx": float(pcfg["geometry"]["Lx"]),
                "box": des["train_box"],
            }
    return _CACHE[k]


def pinn_1d(key: str, value: float, x: np.ndarray, t: np.ndarray | None = None, use: list[str] | None = None) -> dict:
    """Ensemble prediction (mean, sd, every member) of a 1D case at one parameter value, with the query time."""
    from aerosynapse.thermal.pinn.metrics import predict_grid

    mem = [m for m in ensemble_1d(key) if not use or m[0]["name"] in use]
    if not mem:
        raise ValueError(f"none of the members {use} exists")
    t0 = time.perf_counter()
    U = np.stack([predict_grid(n, p, x, t, float(value))[0] for _, p, n in mem])
    wall = time.perf_counter() - t0
    return {
        "x": x,
        "t": t,
        "members": U,
        "mean": U.mean(0),
        "sd": U.std(0, ddof=1),
        "wall_s": wall,
        "n_members": len(mem),
        "names": [m["name"] for m, _, _ in mem],
    }


def pinn_graetz(Re: float, Pr: float, Bi: float, nx: int = 128, ny: int = 16, use: list[str] | None = None) -> dict:
    """Temperature field of the channel networks on a cell-centred grid, and their mean Nusselt number."""
    import torch

    ad = graetz_adapter(True)
    Lx = ad["Lx"]
    x = (np.arange(nx) + 0.5) * Lx / nx
    y = (np.arange(ny) + 0.5) / ny
    X, Y = np.meshgrid(x, y)
    xy = np.stack([X.ravel(), Y.ravel()], 1)
    vals = {"Re": float(Re), "Pr": float(Pr), "Bi": float(Bi)}
    t0 = time.perf_counter()
    fields, nus = [], []
    for name, prob, net in ad["members"]:
        if use and name not in use:
            continue
        p0 = next(net.parameters())
        Xin = prob.inputs(xy, vals, p0)
        th = prob.fields(net, Xin, order=0)["theta"].detach().cpu().numpy().reshape(ny, nx)
        fields.append(th)
        xw = (np.arange(256) + 0.5) / 256 * Lx
        W = prob.inputs(np.stack([xw, np.zeros_like(xw)], 1), vals, p0)
        d = prob.fields(net, W, order=1)
        nus.append(-float(d["theta_y"].detach().cpu().numpy().mean()))
    wall = time.perf_counter() - t0
    F = np.stack(fields)
    _ = torch  # torch is imported for the networks' tensors
    return {
        "x": x,
        "y": y,
        "members": F,
        "mean": F.mean(0),
        "sd": F.std(0, ddof=1),
        "Nu_members": nus,
        "Nu": float(np.mean(nus)),
        "Nu_sd": float(np.std(nus, ddof=1)),
        "wall_s": wall,
        "n_members": len(F),
    }


def predict(key: str, params: dict, opts: dict | None = None) -> dict:
    """PINN prediction of a case at the user's parameters (on the grid of the case's default output)."""
    c = case(key)
    ok, why = pinn_available(key)
    if not ok:
        raise RuntimeError(f"no PINN for {c.title}: {why}")
    o = {**c.conditions, **(opts or {})}
    use = o.get("members")
    if key == "A_steady":
        return pinn_1d(key, params["Pe"], np.linspace(0, 1, 401), use=use)
    if key == "B_slab":
        t = np.linspace(0, float(o["T"]), int(o.get("n_out", 6)))
        return pinn_1d(key, params["Bi"], np.linspace(0, 1, 201), t, use=use)
    return pinn_graetz(params["Re"], params["Pr"], params["Bi"], use=use)


def member_names(key: str) -> list[str]:
    """Names of the saved networks of a case (the checkpoints the user can include or leave out)."""
    if key in ("A_steady", "B_slab"):
        met = _metrics("results/p6_2_uq_default/metrics.json") or {"members": []}
        return [m["name"] for m in met["members"] if m["problem"] == key and m["kind"] == "ensemble"]
    if key == "D_graetz":
        return [f"seed {s}" for s in range(3)]
    return []


def sweep_1d(key: str, values: np.ndarray, use: list[str] | None = None) -> dict:
    """Ensemble error against the exact solution, ensemble spread and member range across parameter values, under
    the conditions the networks were trained with (how Step 6.3 detects extrapolation)."""
    from aerosynapse.thermal.pinn.metrics import predict_grid

    mem = [m for m in ensemble_1d(key) if not use or m[0]["name"] in use]
    prob = mem[0][1]
    x = np.linspace(0.0, 1.0, 201)
    t = np.linspace(0.0, float(prob.T), 6)[1:] if prob.has_time else None
    out: dict[str, list] = {"value": [], "rel_l2": [], "rms_sd": [], "member_min": [], "member_max": []}
    for v in values:
        ref = prob.reference(x, t, float(v))
        U = np.stack([predict_grid(n, p, x, t, float(v))[0] for _, p, n in mem])
        mu, sd = U.mean(0), U.std(0, ddof=1)
        nrm = max(float(np.linalg.norm(ref)), 1e-300)
        me = [float(np.linalg.norm(u - ref) / nrm) for u in U]
        out["value"].append(float(v))
        out["rel_l2"].append(float(np.linalg.norm(mu - ref) / nrm))
        out["rms_sd"].append(float(np.sqrt(np.mean(sd**2))))
        out["member_min"].append(min(me))
        out["member_max"].append(max(me))
    return {k: np.asarray(v) for k, v in out.items()}


# ============================================================================ comparison
def compare(key: str, params: dict, opts: dict | None = None) -> dict:
    """Solver and PINN at the same parameters: errors of the PINN against the solver, wall times, speed-up."""
    s = solve(key, params, opts)
    p = predict(key, params, opts)
    if key == "A_steady":
        ref = np.interp(p["x"], s["x"], s["theta"])
        err = p["mean"] - ref
    elif key == "B_slab":
        ref = np.stack([np.interp(p["x"], s["x"], row) for row in s["theta"]])
        err = p["mean"] - ref
    else:
        from scipy.interpolate import RegularGridInterpolator

        f = RegularGridInterpolator((s["y"], s["x"]), s["theta"], bounds_error=False, fill_value=None)
        Y, X = np.meshgrid(p["y"], p["x"], indexing="ij")
        ref = f(np.stack([Y.ravel(), X.ravel()], 1)).reshape(Y.shape)
        err = p["mean"] - ref
    nrm = max(float(np.linalg.norm(ref)), 1e-300)
    out = {
        "solver": s,
        "pinn": p,
        "reference_on_pinn_grid": ref,
        "error": err,
        "rel_l2": float(np.linalg.norm(err) / nrm),
        "max_abs": float(np.max(np.abs(err))),
        "solver_s": float(s["wall_s"]),
        "pinn_s": float(p["wall_s"]),
        "speedup": float(s["wall_s"] / max(p["wall_s"], 1e-12)),
        "validity": validity(key, params, opts),
    }
    if key == "D_graetz":
        out["Nu_solver"], out["Nu_pinn"] = s["Nu"], p["Nu"]
        out["Nu_rel_err"] = abs(p["Nu"] - s["Nu"]) / abs(s["Nu"])
    return out


# ============================================================================ grid convergence (verification)
def convergence(
    key: str, params: dict, Ns: list[int] | None = None, schemes: list[str] | None = None, opts: dict | None = None
) -> dict:
    """Errors against the exact solution on a sequence of grids and the observed order (least-squares slope of
    log error vs log h over the finest half), for the 1D cases."""
    if key not in ("A_steady", "B_slab"):
        raise ValueError("grid convergence against an exact solution is available for the 1D cases")
    Ns = list(Ns or [10, 20, 40, 80, 160, 320])
    o = {**case(key).conditions, **(opts or {})}
    out = {"N": Ns, "h": [1.0 / n for n in Ns], "schemes": {}}
    for sch in schemes or (["central", "upwind1"] if key == "A_steady" else ["central"]):
        errs = []
        for n in Ns:
            if key == "A_steady":
                r = solve_steady(params["Pe"], n, sch, o["theta_left"], o["theta_right"], 0.0)
            else:
                r = solve_slab(params["Bi"], o["Q"], o["m"], o["theta0"], o["T"], n, 3)
            errs.append(r["err_inf"])
        h = np.array(out["h"])
        k = max(2, len(Ns) // 2)
        order = float(np.polyfit(np.log(h[-k:]), np.log(np.array(errs)[-k:]), 1)[0])
        out["schemes"][sch] = {"err_inf": errs, "observed_order": order}
    return out


# ============================================================================ inverse identification
def inverse(
    q_vol: float = 1.0e5,
    noise: float = 0.02,
    n_sensors: int = 8,
    unknown: tuple[str, ...] = ("alpha", "k", "h"),
    truth: dict | None = None,
    init: dict | None = None,
    seed: int = 0,
    data_file: str | None = None,
    body: dict | None = None,
) -> dict:
    """Least-squares identification (Phase 5, analytic forward model, Levenberg-Marquardt in log-parameters) of the
    unknowns from synthetic sensor data (or the user's CSV: columns x [m], t [s], T, optional sigma), with Gauss-Newton
    95 % intervals and the identifiability rank."""
    from aerosynapse.thermal.inverse import body as B
    from aerosynapse.thermal.inverse import classical as C
    from aerosynapse.thermal.inverse import data as D
    from aerosynapse.thermal.inverse import spec as IS

    over = {
        "body": {"q_vol": float(q_vol), **(body or {})},
        "data": {"level": float(noise), "seed": int(seed)},
        "measurement": {"sensors": {"n": int(n_sensors)}},
    }
    if truth:
        over["parameters"] = {k: float(v) for k, v in truth.items()}
    spec = IS.load_spec(overrides=over)
    un = list(unknown)
    data = D.read_file(data_file, spec, float(noise)) if data_file else D.synthetic(spec, float(noise), int(seed))
    guess = {**spec["initial_guess"], **(init or {})}
    t0 = time.perf_counter()
    tr = {k: float(spec["parameters"][k]) for k in B.PARAMS}
    # identifiability BEFORE fitting (Step 5.1): a rank-deficient Fisher matrix at the initial guess means some
    # combination of the unknowns is not determined by the data (no source: only Bi = hL/k); fitting would drift
    pre = C.covariance_at(spec, data, un, guess)
    base = {
        "n_unknown": len(un),
        "source": data["source"],
        "data": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in data.items() if k != "T_clean"},
    }
    if int(pre["rank"]) < len(un):
        return base | {
            "identifiable": False,
            "rank": int(pre["rank"]),
            "null_directions": [list(map(float, v)) for v in pre["null_directions"]],
            "rows": [],
            "message": f"not identifiable: the Fisher information has rank {int(pre['rank'])} < {len(un)} unknowns "
            "(null direction in log-parameters above). Fix one of the confounded parameters, add a known heat "
            "source, or measure something that separates them (Steps 5.1, 5.6).",
            "wall_s": time.perf_counter() - t0,
        }
    try:
        est = D.least_squares_estimate(spec, data, un, guess)
        cov = C.covariance_at(spec, data, un, est["estimates"])
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as err:
        return base | {
            "identifiable": True,
            "rank": int(pre["rank"]),
            "rows": [],
            "null_directions": [],
            "message": f"fit failed: {type(err).__name__}: {err}",
            "wall_s": time.perf_counter() - t0,
        }
    wall = time.perf_counter() - t0
    rows = []
    for k in un:
        e, sd = float(est["estimates"][k]), float(cov["sd_log"][k])
        lo, hi = (e * np.exp(-1.96 * sd), e * np.exp(1.96 * sd)) if np.isfinite(sd) else (0.0, np.inf)
        rows.append(
            {
                "parameter": k,
                "estimate": e,
                "ci95": [float(lo), float(hi)],
                "sd_log": sd,
                "truth": tr[k] if not data_file else None,
                "error_pct": 100 * (e / tr[k] - 1) if not data_file else None,
            }
        )
    xs, ix = np.unique(data["r"], return_inverse=True)
    ts, it = np.unique(data["t"], return_inverse=True)
    nt = max(B.truncation_terms(float(np.min(ts)) * est["estimates"]["alpha"] / float(spec["body"]["L"]) ** 2), 20)
    fitted = B.temperatures(est["estimates"], IS.body_of(spec), xs, ts, "analytic", nt)
    Bi = est["estimates"]["h"] * float(spec["body"]["L"]) / est["estimates"]["k"]
    return base | {
        "rows": rows,
        "Bi": Bi,
        "chi2_per_reading": est["chi2_per_reading"],
        "rank": int(cov["rank"]),
        "null_directions": [list(map(float, v)) for v in cov["null_directions"]],
        "sensor_x": xs,
        "times": ts,
        "fitted": fitted,
        "wall_s": wall,
        "identifiable": int(cov["rank"]) == len(un),
        "message": "converged" if est["success"] else f"optimiser: {est['message']}",
    }


# ============================================================================ training monitor (saved logs)
def training_runs(key: str) -> dict[str, Path]:
    """{member name: log.csv} of the saved networks of a case."""
    if key in ("A_steady", "B_slab"):
        met = _metrics("results/p6_2_uq_default/metrics.json") or {"members": []}
        return {
            m["name"]: Path(m["dir"]) / "log.csv"
            for m in met["members"]
            if m["problem"] == key and m["kind"] == "ensemble" and (Path(m["dir"]) / "log.csv").exists()
        }
    if key == "D_graetz":
        run = ROOT / GRAETZ_RUN["run"] / "runs"
        return {
            f"seed {s}": run / f"{GRAETZ_RUN['configuration']}_s{s}" / "log.csv"
            for s in range(3)
            if (run / f"{GRAETZ_RUN['configuration']}_s{s}" / "log.csv").exists()
        }
    return {}


def read_log(path: Path) -> dict[str, np.ndarray]:
    """Columns of a training log (numeric where possible; empty cells -> nan)."""
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, np.ndarray] = {}
    for k in rows[0] if rows else []:
        vals = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) if v not in ("", None) else np.nan for v in vals])
        except ValueError:
            out[k] = np.array(vals)
    return out


# ============================================================================ export
def to_csv(columns: dict[str, np.ndarray]) -> bytes:
    """Equal-length 1D columns as CSV bytes."""
    keys = list(columns)
    n = len(np.asarray(columns[keys[0]]))
    buf = io.StringIO()
    buf.write(",".join(keys) + "\n")
    arrs = [np.asarray(columns[k]).ravel() for k in keys]
    for i in range(n):
        buf.write(",".join(f"{a[i]:.10g}" if np.issubdtype(a.dtype, np.number) else str(a[i]) for a in arrs) + "\n")
    return buf.getvalue().encode("utf-8")


def jsonable(o):
    """Results as JSON-serialisable data (arrays -> lists)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    return o


def field_columns(key: str, r: dict, which: str = "theta") -> dict[str, np.ndarray]:
    """A solver or PINN result as flat columns for CSV export."""
    if "y" in r and np.ndim(r.get(which, r.get("mean"))) == 2 and key in ("D_graetz", "cavity"):
        F = r.get(which, r.get("mean"))
        Y, X = np.meshgrid(r["y"], r["x"], indexing="ij")
        cols = {"x": X.ravel(), "y": Y.ravel(), which: np.asarray(F).ravel()}
        if "sd" in r:
            cols["sd"] = np.asarray(r["sd"]).ravel()
        return cols
    if key == "B_slab":
        F = r.get(which, r.get("mean"))
        T, X = np.meshgrid(r["t"], r["x"], indexing="ij")
        cols = {"t": T.ravel(), "x": X.ravel(), which: np.asarray(F).ravel()}
        if "sd" in r:
            cols["sd"] = np.asarray(r["sd"]).ravel()
        return cols
    cols = {"x": r["x"], which: r.get(which, r.get("mean"))}
    if "sd" in r:
        cols["sd"] = r["sd"]
    return cols
