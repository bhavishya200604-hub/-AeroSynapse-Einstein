r"""Skin heating of the airframe with the merged parametric PINN project.

A skin panel of thickness :math:`L`, heated on its outer face by the boundary layer (heat-transfer
coefficient :math:`h`, adiabatic-wall temperature :math:`T_{aw}`) and insulated inside, is exactly the
project's Case B slab with :math:`m=0, Q=0`:

.. math:: \theta_t=\theta_{xx},\quad \theta_x(0,t)=0,\quad \theta_x(1,t)+Bi\,\theta(1,t)=0,\quad \theta(x,0)=1,

with :math:`\theta=(T-T_{aw})/(T_0-T_{aw})`, :math:`x` from the inner face, Fourier time :math:`t=\alpha\tau/L^2` and
:math:`Bi=hL/k`. One parametric PINN (inputs :math:`x, t, \ln Bi`) is trained over the Biot numbers of the
airframe's panels with the project's trainer (Adam then L-BFGS), and checked against the project's analytic
eigen-series and its finite-difference solver, both verified in that project. Outside the trained range
(:math:`t` beyond the horizon or Bi outside the box) the answer comes from the analytic series, and the
result says so: the project's own finding is that PINNs do not extrapolate.

Uses: the inner-face temperature lag and the through-thickness temperature difference, which give the
thermal stress :math:`\sigma=E\alpha\,\Delta T/(1-\nu)` in each zone's selected material, checked against its
allowable at temperature.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aerosynapse.thermal.numerics import fd1d
from aerosynapse.thermal.numerics.time_integrators import integrate
from aerosynapse.thermal.physics import analytic as an
from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn.architectures import build_network
from aerosynapse.thermal.pinn.metrics import evaluate_model, predict_grid
from aerosynapse.thermal.pinn.trainer import Trainer

BI_RANGE = (0.2, 20.0)
HORIZON = 2.0  # Fourier time covered by the network


def skin_spec(profile: str = "smoke", epochs: int = 4000, lbfgs: int = 300) -> dict[str, Any]:
    spec = SP.load_spec(profile=profile)
    spec["problems"]["B_slab"].update({"parametric": True, "parameter": {"name": "Bi", "lo": BI_RANGE[0], "hi": BI_RANGE[1], "n": 5, "scale": "log"}, "T": HORIZON,
                                       "conditions": {"Q": 0.0, "m": 0.0, "theta0": 1.0}})
    spec["network"].update({"width": 48, "depth": 4})
    spec["training"]["adam"]["epochs"] = epochs
    spec["training"]["lbfgs"]["iterations"] = lbfgs
    spec["sampling"].update({"n_pde": 1500, "n_ic": 300, "n_bc": 300})
    spec["device"] = "cpu"
    return spec


def train(out_dir: str | Path, profile: str = "smoke", epochs: int = 4000, lbfgs: int = 300) -> dict[str, Any]:
    spec = skin_spec(profile, epochs, lbfgs)
    prob = P.build_problem("B_slab", spec["problems"]["B_slab"])
    torch.manual_seed(spec["seed"])
    net = build_network(spec["network"], prob.lo, prob.hi).to(dtype=torch.float64)
    tr = Trainer(net, prob, spec, out_dir, tag="skin")
    t0 = time.perf_counter()
    tr.run()
    wall = time.perf_counter() - t0
    torch.save({"state": net.state_dict(), "network": spec["network"], "lo": prob.lo, "hi": prob.hi, "bi_range": BI_RANGE, "horizon": HORIZON}, Path(out_dir) / "skin_pinn.pt")
    checks = {}
    for bi in (0.3, 1.0, 3.0, 10.0, 18.0):
        ev = evaluate_model(net, prob, bi, 101, 21)
        checks[f"Bi={bi}"] = {"rel_l2_vs_analytic": float(ev["rel_l2"]), "linf": float(ev["linf"])}
    return {"wall_s": wall, "checks": checks, "net": net, "problem": prob}


def load(path: str | Path) -> tuple[torch.nn.Module, P.Problem]:
    ck = torch.load(path, weights_only=False)
    spec = skin_spec()
    prob = P.build_problem("B_slab", spec["problems"]["B_slab"])
    net = build_network(ck["network"], ck["lo"], ck["hi"]).to(dtype=torch.float64)
    net.load_state_dict(ck["state"])
    return net, prob


def theta_pinn(net: torch.nn.Module, prob: P.Problem, Bi: float, fo: np.ndarray, x: np.ndarray) -> np.ndarray:
    u, _ = predict_grid(net, prob, x, fo, Bi)
    return np.asarray(u)


def theta_exact(Bi: float, fo: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.array([an.slab_transient(x, max(t, 1e-3), Bi) for t in fo])


def theta_fd(Bi: float, fo: np.ndarray, x: np.ndarray, N: int = 80, dt: float = 1e-3) -> np.ndarray:
    """The project's finite-difference solver (central differences, Crank-Nicolson) on the same slab."""
    f = fd1d.ConvDiff1D(N, a=0.0, D=1.0, left=fd1d.neumann(0.0), right=fd1d.robin(Bi, 0.0), scheme="central")
    y, t, out = np.ones(len(f.idx)), 0.0, []
    for tt in fo:
        if tt > t:
            y = integrate(f, y, t, float(tt), dt, "crank_nicolson")
            t = float(tt)
        out.append(np.interp(x, f.x, f.nodal(y, t)))
    return np.array(out)


def h_coefficient(T_aw: float, q_at: float, T_wall: float) -> float:
    return float(q_at / max(T_aw - T_wall, 1e-6))


def panel(net: torch.nn.Module | None, prob: P.Problem | None, mat: Any, thickness: float, h: float, T0: float, T_aw: float, duration: float, n_t: int = 40) -> dict[str, Any]:
    """Temperature history of a panel: outer and inner face, through-thickness difference, thermal stress."""
    Bi = h * thickness / mat.k
    alpha = mat.diffusivity()
    tau = np.linspace(0, duration, n_t)
    fo = alpha * tau / thickness**2
    x = np.array([0.0, 1.0])
    inside = net is not None and BI_RANGE[0] <= Bi <= BI_RANGE[1]
    th = np.zeros((n_t, 2))
    src = []
    for i, f in enumerate(fo):
        if f <= 1e-9:
            th[i] = 1.0
            src.append("initial")
        elif inside and f <= HORIZON:
            th[i] = theta_pinn(net, prob, Bi, np.array([f]), x)[0]
            src.append("pinn")
        else:
            th[i] = an.slab_transient(x, max(f, 1e-3), Bi) if Bi > 1e-6 else np.ones(2)
            src.append("analytic")
    T = T_aw + (T0 - T_aw) * th
    dT = np.abs(T[:, 1] - T[:, 0])
    nu = 0.3
    sigma = mat.modulus(float(T.max())) * mat.alpha * dT / (1 - nu)
    allow = mat.strength(float(T.max()))
    return {"Bi": float(Bi), "fourier_end": float(fo[-1]), "time_s": tau.tolist(), "T_outer_K": T[:, 1].tolist(), "T_inner_K": T[:, 0].tolist(),
            "dT_max_K": float(dT.max()), "thermal_stress_MPa": float(sigma.max() / 1e6), "allowable_MPa": float(allow / 1e6), "stress_ratio": float(sigma.max() / allow),
            "T_max_K": float(T.max()), "source": {s: src.count(s) for s in set(src)}}
