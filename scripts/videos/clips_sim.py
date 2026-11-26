"""Simulation clips computed live by the platform's solvers (no stored data except the cylinder cache)."""

from __future__ import annotations

import numpy as np
from _base import CACHE, CLIPS

from aerosynapse.viz import neon as N

FPS = 30


def heat2d() -> str:
    from aerosynapse.numerics.heat import solve_heat_2d
    from aerosynapse.physics.library import heat_2d

    fig = N.plt.figure(figsize=(N.W, N.H), dpi=N.DPI)
    N.header(
        fig,
        "Heat diffusion in 2D",
        "T_t = α ∇²T  ·  Crank–Nicolson finite differences  ·  depth-shaded surface",
        "computed live: aerosynapse.numerics.heat.solve_heat_2d (n = 48)",
    )
    ax = fig.add_axes([0.05, 0.06, 0.9, 0.84], projection="3d")
    times = np.linspace(0.005, 1.2, 150)

    def frame(i: int) -> None:
        ax.cla()
        N.style_3d(ax)
        p = heat_2d(alpha=0.05, t_final=float(times[i]))
        p.initial_condition = lambda x, y: (
            np.sin(np.pi * x) * np.sin(np.pi * y) + 0.6 * np.sin(3 * np.pi * x) * np.sin(2 * np.pi * y)
        )
        X, Y, T = solve_heat_2d(p, 48, nt=max(4, int(times[i] / 0.01)))
        ax.plot_surface(
            X,
            Y,
            T,
            facecolors=N.shaded(T, N.AURORA, -1.2, 1.5),
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )
        ax.contourf(X, Y, T, zdir="z", offset=-1.4, cmap=N.AURORA, levels=24, vmin=-1.2, vmax=1.5, alpha=0.8)
        ax.set_zlim(-1.4, 1.6)
        ax.view_init(28, -60 + 0.8 * i)
        ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("T")
        ax.set_title(f"t = {times[i]:.3f}", color=N.GOLD)

    return str(N.write(fig, frame, len(times), CLIPS / "10_heat2d.mp4", FPS))


def convdiff() -> str:
    from aerosynapse.numerics.convdiff import solve_convdiff_fv
    from aerosynapse.physics.library import convdiff_1d

    fig, axs = N.figure(2, 2)
    fig.subplots_adjust(top=0.84, bottom=0.08, hspace=0.35, wspace=0.15)
    N.header(
        fig,
        "Transient convection–diffusion across Péclet numbers",
        "u_t + u_x = Pe⁻¹ u_xx  ·  finite-volume (van Leer / Crank–Nicolson) vs exact solution",
        "computed live: aerosynapse.numerics.convdiff.solve_convdiff_fv (256 cells); exact: method of images",
    )
    pes = [1.0, 10.0, 100.0, 1000.0]
    times = np.linspace(0.0, 1.0, 120)
    arts = []
    for ax, pe, c in zip(axs.ravel(), pes, (N.CYAN, N.LIME, N.ORANGE, N.MAGENTA), strict=True):
        ax.set_xlim(0, 1), ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Pe = {pe:g}", color=c)
        g = N.glow(ax, np.zeros(2), np.zeros(2), c, 2.2, "FV solver")
        e = ax.plot([], [], "--", color="white", lw=1, alpha=0.8, label="exact")[0]
        f = ax.fill_between([0, 1], 0, 0, color=c, alpha=0.15)
        ax.legend(loc="upper right", fontsize=8)
        arts.append([ax, pe, g, e, f, c])

    def frame(i: int) -> None:
        t = float(times[i])
        for a in arts:
            ax, pe, g, e, f, c = a
            p = convdiff_1d(pe=pe, t_final=1.0, sigma0=0.05)
            if t == 0:
                x = (np.arange(256) + 0.5) / 256
                u = p.initial_condition(x)
            elif pe <= 10:
                r = solve_convdiff_fv(
                    0,
                    1,
                    256,
                    1.0,
                    1 / pe,
                    p.initial_condition,
                    t,
                    scheme="central",
                    integrator="theta",
                    dt=t / max(4, int(t / 0.004)),
                )
                x, u = r.x, r.u
            else:
                r = solve_convdiff_fv(0, 1, 256, 1.0, 1 / pe, p.initial_condition, t, scheme="vanleer", cfl=0.5)
                x, u = r.x, r.u
            N.set_glow(g, x, u)
            e.set_data(x, p.exact(x, t))
            a[4].remove()
            a[4] = ax.fill_between(x, 0, u, color=c, alpha=0.15)
        fig.texts[0].set_text(f"Transient convection–diffusion across Péclet numbers   t = {t:.2f}")

    return str(N.write(fig, frame, len(times), CLIPS / "11_convdiff.mp4", FPS))


def burgers_wave() -> str:
    from aerosynapse.numerics.burgers import solve_burgers
    from aerosynapse.numerics.wave import solve_wave

    fig, (a1, a2) = N.figure(1, 2)
    fig.subplots_adjust(top=0.82, bottom=0.12, wspace=0.2)
    N.header(
        fig,
        "Nonlinear steepening and wave propagation",
        "left: viscous Burgers (ν = 0.01) shock formation  ·  right: wave equation, two modes, leapfrog",
        "computed live: aerosynapse.numerics.burgers (MUSCL + Rusanov, 512 cells), numerics.wave (leapfrog, Courant 0.8)",
    )
    u0 = lambda x: np.sin(x) + 0.5  # noqa: E731
    w0 = lambda x: np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)  # noqa: E731
    times = np.linspace(0.0, 2.5, 150)
    a1.set_xlim(0, 2 * np.pi), a1.set_ylim(-0.7, 1.7), a1.set_xlabel("x"), a1.set_ylabel("u")
    a2.set_xlim(0, 1), a2.set_ylim(-1.7, 1.7), a2.set_xlabel("x"), a2.set_ylabel("u")
    g1 = N.glow(a1, np.zeros(2), np.zeros(2), N.MAGENTA, 2.4)
    g2 = N.glow(a2, np.zeros(2), np.zeros(2), N.CYAN, 2.4)
    fills = [a1.fill_between([0, 1], 0, 0), a2.fill_between([0, 1], 0, 0)]

    def frame(i: int) -> None:
        t = float(times[i])
        if t == 0:
            x1 = (np.arange(512) + 0.5) / 512 * 2 * np.pi
            u1 = u0(x1)
            x2 = np.linspace(0, 1, 201)
            u2 = w0(x2)
        else:
            r = solve_burgers(0, 2 * np.pi, 512, 0.01, u0, t, cfl=0.4)
            x1, u1 = r.x, r.u
            x2, u2, _ = solve_wave(1.0, 200, 1.0, w0, t)
        N.set_glow(g1, x1, u1)
        N.set_glow(g2, x2, u2)
        for k in range(2):
            fills[k].remove()
        fills[0] = a1.fill_between(x1, -0.7, u1, color=N.MAGENTA, alpha=0.12)
        fills[1] = a2.fill_between(x2, 0, u2, color=N.CYAN, alpha=0.12)
        a1.set_title(f"Burgers  t = {t:.2f}", color=N.MAGENTA)
        a2.set_title(f"Wave  t = {t:.2f}", color=N.CYAN)

    return str(N.write(fig, frame, len(times), CLIPS / "12_burgers_wave.mp4", FPS))


def cavity() -> str:
    from aerosynapse.numerics.ns2d import FluidBC, NSSolver

    n = 64
    s = NSSolver(
        n,
        n,
        1.0,
        1.0,
        1 / 400,
        {"left": FluidBC("wall"), "right": FluidBC("wall"), "bottom": FluidBC("wall"), "top": FluidBC("wall", 1.0)},
    )
    snaps, t, i = [], 0.0, 0
    targets = np.linspace(0.05, 25.0, 150)
    while i < len(targets):
        dt = s.stable_dt()
        s.step(dt)
        t += dt
        if t >= targets[i]:
            uc = 0.5 * (s.U[:-1, 1:-1] + s.U[1:, 1:-1])
            vc = 0.5 * (s.V[1:-1, :-1] + s.V[1:-1, 1:])
            snaps.append((uc.copy(), vc.copy(), t))
            i += 1
    fig, ax = N.figure()
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.3, right=0.72)
    N.header(
        fig,
        "Lid-driven cavity, Re = 400 (validated vs Ghia et al. 1982)",
        "incompressible Navier–Stokes · MAC projection · speed (depth gradient) + streamlines",
        "computed live: aerosynapse.numerics.ns2d (64² cells); validation RMS in results/phase6_fluids_aero",
    )
    xs = (np.arange(n) + 0.5) / n

    def frame(k: int) -> None:
        ax.cla()
        uc, vc, tt = snaps[k]
        sp = np.hypot(uc, vc)
        ax.imshow(sp.T, origin="lower", extent=(0, 1, 0, 1), cmap=N.AURORA, vmin=0, vmax=1.0, interpolation="bicubic")
        ax.streamplot(xs, xs, uc.T, vc.T, color=N.CYAN, density=1.4, linewidth=0.6, arrowsize=0.6)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.set_title(f"t = {tt:.2f}   (lid moves →)", color=N.GOLD)
        ax.grid(False)

    return str(N.write(fig, frame, len(snaps), CLIPS / "13_cavity.mp4", FPS))


def cylinder() -> str:
    d = np.load(CACHE / "cylinder.npz", allow_pickle=True)
    vort, speed, tt, h = d["vort"], d["speed"], d["t"], float(d["h"])
    nx, ny = vort.shape[1:]
    ext = (0, nx * h, 0, ny * h)
    fig, (a1, a2) = N.figure(2, 1)
    fig.subplots_adjust(top=0.8, bottom=0.07, hspace=0.3, left=0.05, right=0.97)
    N.header(
        fig,
        "Vortex street behind a cylinder (Re = 100) — UNVALIDATED VISUALISATION",
        f"laminar 2D periodic wake, NOT turbulence · {str(d['label'])}",
        "computed live: aerosynapse.numerics.ns2d (staircase immersed cylinder); validated only at Re = 20 (DFG 2D-1)",
    )
    lim = float(np.nanpercentile(np.abs(vort), 99))
    im1 = a1.imshow(
        vort[0].T, origin="lower", extent=ext, cmap=N.NEON_DIVERGING, vmin=-lim, vmax=lim, interpolation="bicubic"
    )
    im2 = a2.imshow(
        speed[0].T,
        origin="lower",
        extent=ext,
        cmap=N.ICE,
        vmin=0,
        vmax=float(np.nanmax(speed)),
        interpolation="bicubic",
    )
    for a, ttl in ((a1, "vorticity ω (diverging gradient)"), (a2, "speed |u| (depth gradient)")):
        a.set_title(ttl, color=N.TEXT, fontsize=11), a.grid(False), a.set_aspect("equal")

    def frame(k: int) -> None:
        im1.set_data(vort[k].T)
        im2.set_data(speed[k].T)
        a1.set_title(f"vorticity ω (diverging gradient)   t = {tt[k]:.2f}", color=N.TEXT, fontsize=11)

    return str(N.write(fig, frame, len(tt), CLIPS / "14_cylinder.mp4", FPS))


ALL = [heat2d, convdiff, burgers_wave, cavity]
