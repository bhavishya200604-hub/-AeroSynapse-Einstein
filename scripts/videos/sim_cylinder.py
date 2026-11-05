"""Simulate laminar vortex shedding behind a cylinder (DFG 2D-2 geometry, Re = 100) for the showcase video.

Re = 100 is OUTSIDE the validated range of the solver (validated at Re = 20); the regime guard is passed with
extrapolate=True and the video labels the clip as an unvalidated extrapolation (laminar periodic wake, not turbulence).
Saves vorticity / speed snapshots to media/_cache/cylinder.npz."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aerosynapse.numerics.ns2d import FluidBC, NSSolver  # noqa: E402
from aerosynapse.physics.regime import classify  # noqa: E402


def main(ny: int = 62, t_start: float = 3.0, t_end: float = 8.0, n_frames: int = 240) -> None:
    regime = classify("cylinder", 100.0)  # the guard refuses this case; we only visualise, with the verdict on screen
    label = f"regime guard: '{regime}' (outside validated range) - unvalidated visualisation"
    H, Lx, D, Um, nu = 0.41, 2.2, 0.1, 1.5, 1e-3
    h = H / ny
    nx = int(round(Lx / h))
    xc, yc = (np.arange(nx) + 0.5) * h, (np.arange(ny) + 0.5) * h
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    solid = (X - 0.2) ** 2 + (Y - 0.2) ** 2 < (D / 2) ** 2
    prof = lambda y: 4 * Um * y * (H - y) / H**2  # noqa: E731
    s = NSSolver(
        nx,
        ny,
        nx * h,
        H,
        nu,
        {"left": FluidBC("inlet", prof), "right": FluidBC("outlet"), "bottom": FluidBC("wall"), "top": FluidBC("wall")},
        solid=solid,
    )
    times = np.linspace(t_start, t_end, n_frames)
    vort, speed, tt = [], [], []
    t, k, i, t0 = 0.0, 0, 0, time.perf_counter()
    while i < n_frames:
        dt = s.stable_dt()
        s.step(dt)
        t += dt
        k += 1
        if t >= times[i]:
            uc = 0.5 * (s.U[:-1, 1:-1] + s.U[1:, 1:-1])
            vc = 0.5 * (s.V[1:-1, :-1] + s.V[1:-1, 1:])
            om = np.gradient(vc, h, axis=0) - np.gradient(uc, h, axis=1)
            om[solid] = np.nan
            sp = np.hypot(uc, vc)
            sp[solid] = np.nan
            vort.append(om.astype(np.float32))
            speed.append(sp.astype(np.float32))
            tt.append(t)
            i += 1
    np.savez_compressed(
        ROOT / "media" / "_cache" / "cylinder.npz",
        vort=np.array(vort),
        speed=np.array(speed),
        t=np.array(tt),
        h=h,
        label=label,
        steps=k,
        wall=time.perf_counter() - t0,
    )
    print(label, "steps", k, "wall", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
