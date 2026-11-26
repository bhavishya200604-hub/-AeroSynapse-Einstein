r"""Unsteady Navier-Stokes check of any body, using the repository's verified MAC projection solver.

This solves the full 2D equations in time, so it sees what steady models cannot: separation,
vortex shedding and its frequency, and the size of the unsteady forces. It shares nothing with
NeuralFoil or the boundary-layer solver.

Limits: 2D laminar flow on a uniform grid with a staircase body, so Reynolds numbers stay in the
hundreds. It tests the direction of a design change at low Re. It says nothing about turbulent
flow, and 2D wakes are stronger and more coherent than real 3D ones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerosynapse.numerics.ns2d import FluidBC, NSSolver
from aerosynapse.shapeopt.geometry import polygon_mask, rotate

FloatArr = NDArray[np.float64]


@dataclass
class NSCase:
    name: str
    polys: list[FloatArr]  # Selig or closed contours, chord about 1, nose near x = 0
    Re: float
    alpha: float = 0.0
    h: float = 1.0 / 32.0
    lx: float = 10.0
    ly: float = 5.0
    x_body: float = 2.5
    t_end: float = 40.0
    t_avg: float = 15.0
    pivot: tuple[float, float] = (0.25, 0.0)


def run_case(c: NSCase) -> dict[str, Any]:
    """Run one case and return mean forces, fluctuation, shedding frequency and a vorticity frame."""
    t0 = time.time()
    nx, ny = round(c.lx / c.h), round(c.ly / c.h)
    polys = [rotate(p, c.alpha, c.pivot) + np.array([c.x_body, c.ly / 2]) for p in c.polys]
    xc, yc = (np.arange(nx) + 0.5) * c.h, (np.arange(ny) + 0.5) * c.h
    solid = polygon_mask(polys, xc, yc)
    far = FluidBC("inlet", 1.0)
    bcs = {"left": far, "right": FluidBC("outlet"), "bottom": FluidBC("slip"), "top": FluidBC("slip")}
    s = NSSolver(nx, ny, c.lx, c.ly, 1.0 / c.Re, bcs, solid=solid)
    s.U[:, 1:-1] = 1.0
    # A brief disturbance behind the body. A symmetric body in a symmetric grid otherwise stays on
    # the steady solution for ever, even above the shedding threshold, because the numerics never
    # break the symmetry: a cylinder at Re = 100 then gives C_D 1.21 and no shedding. A stable flow
    # simply lets this die away.
    xv = (np.arange(nx + 2) - 0.5) * c.h
    yv = np.arange(ny + 1) * c.h
    s.V += 0.15 * np.exp(-(((xv[:, None] - c.x_body - 1.5) ** 2) + (yv[None, :] - c.ly / 2) ** 2) / 0.4)
    t, k = 0.0, 0
    T, FX, FY = [], [], []
    while t < c.t_end:
        dt = min(s.stable_dt(0.4), 0.02)
        _, _, f = s.step(dt)
        t += dt
        k += 1
        if k % 4 == 0:
            T.append(t)
            FX.append(f[0])
            FY.append(f[1])
        if k % 500 == 0 and not np.isfinite(s.U).all():
            raise FloatingPointError(f"{c.name}: solution diverged at t = {t:.2f}")
    T_a, FX_a, FY_a = np.array(T), np.array(FX), np.array(FY)
    m = T_a > c.t_end - c.t_avg
    cd, cl = 2 * FX_a[m], 2 * FY_a[m]
    # uniform resample before the FFT, because the time step varies
    tu = np.linspace(T_a[m][0], T_a[m][-1], 2048)
    sig = np.interp(tu, T_a[m], cl) - cl.mean()
    win = np.hanning(len(sig))
    P = np.abs(np.fft.rfft(sig * win))
    fr = np.fft.rfftfreq(len(sig), tu[1] - tu[0])
    st = float(fr[1:][np.argmax(P[1:])])
    uc, vc = s.U, s.V
    u_cc, v_cc = 0.5 * (uc[:-1, 1:-1] + uc[1:, 1:-1]), 0.5 * (vc[1:-1, :-1] + vc[1:-1, 1:])
    om = np.gradient(v_cc, c.h, axis=0) - np.gradient(u_cc, c.h, axis=1)
    om[solid] = 0.0
    return {
        "name": c.name,
        "Re": c.Re,
        "alpha": c.alpha,
        "h": c.h,
        "cd_mean": float(cd.mean()),
        "cl_mean": float(cl.mean()),
        "cl_rms": float(cl.std()),
        "cd_rms": float(cd.std()),
        "strouhal": st,
        "unsteady": bool(cl.std() > 0.01),
        "steps": k,
        "wall_s": time.time() - t0,
        "divergence_max": float(np.abs(s.divergence()).max()),
        "vorticity": om[:: 2, :: 2].astype(np.float32).tolist(),
        "solid": solid[:: 2, :: 2].tolist(),
        "t": T_a[-1200:].tolist(),
        "cl_t": (2 * FY_a[-1200:]).tolist(),
    }
