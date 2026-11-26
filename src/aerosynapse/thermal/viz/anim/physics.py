"""Physics scenes: analytic solutions and fresh solver runs (no network training).

1D (analytic, src/physics/analytic.py): an advected, diffusing pulse at three Péclet numbers; a convectively cooled
slab at three Biot numbers; the steady boundary layer as Pe grows, with the finite-difference schemes beside the exact
solution. 2D (the verified MAC Navier-Stokes + energy solver, src/numerics/ns_projection.py, started from rest):
heated lid-driven cavity, natural convection in a differentially heated cavity, thermal entry of a Graetz channel,
flow past a heated square cylinder. Massless tracer particles are advected through the computed velocity field
(second-order Runge-Kutta, bilinear interpolation of the cell-centred velocity).

Every compute_* function returns arrays only and is cached on disk by its parameters (core.cached).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aerosynapse.thermal.viz import style as S
from aerosynapse.thermal.viz.anim.core import Scene, cached


# ============================================================================ 1D, analytic
def compute_pulse(p: dict) -> dict:
    from aerosynapse.thermal.physics import analytic as an

    x = np.linspace(0, 1, 600)
    t = np.linspace(0, p["T"], p["frames"])
    th = np.stack([an.gaussian_pulse(x[None, :], t[:, None], pe, p["sigma0"], p["x0"], 1.0) for pe in p["Pe"]])
    return {"x": x, "t": t, "theta": th, "Pe": np.asarray(p["Pe"], float)}


def scene_pulse(p: dict, cache: Path) -> Scene:
    d, _ = cached(cache, "pulse", p, lambda: compute_pulse(p))
    x, t, th, pes = d["x"], d["t"], d["theta"], d["Pe"]

    def make():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.0, 3.6))
        lines = [ax.plot([], [], color=S.PALETTE[k], label=f"Pe = {pe:g}")[0] for k, pe in enumerate(pes)]
        ax.plot(x, th[0, 0], ":", color=S.MUTED, lw=1, label="initial pulse")
        ax.set(xlim=(0, 1), ylim=(-0.05, 1.08), xlabel=S.label("x"), ylabel=S.label("theta"))
        ax.legend(loc="upper right")
        title = ax.set_title("")

        def update(i):
            for k, ln in enumerate(lines):
                ln.set_data(x, th[k, i])
            title.set_text(f"Advection and diffusion of a heat pulse   t = {t[i]:.3f}")

        return fig, update

    return Scene(
        "pulse",
        "Heat pulse: advection versus diffusion",
        "A Gaussian temperature pulse carried by a uniform flow: at low Péclet number it spreads by diffusion, at "
        "high Péclet number it is carried almost unchanged (exact solution).",
        len(t),
        p["fps"],
        make,
        "analytic solution (src/physics/analytic.py, problem A_pulse)",
    )


def compute_slab(p: dict) -> dict:
    from aerosynapse.thermal.physics import analytic as an

    x = np.linspace(0, 1, 300)
    t = np.concatenate([[0.0], np.geomspace(1e-4, p["T"], p["frames"] - 1)])
    th = np.stack(
        [
            np.stack([an.slab_transient(x, float(tt), bi, p["Q"], p["m"], p["theta0"], n_terms=400) for tt in t])
            for bi in p["Bi"]
        ]
    )
    return {"x": x, "t": t, "theta": th, "Bi": np.asarray(p["Bi"], float)}


def scene_slab(p: dict, cache: Path) -> Scene:
    d, _ = cached(cache, "slab", p, lambda: compute_slab(p))
    x, t, th, bis = d["x"], d["t"], d["theta"], d["Bi"]

    def make():
        import matplotlib.pyplot as plt

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
        prof = [a1.plot([], [], color=S.PALETTE[k], label=f"Bi = {b:g}")[0] for k, b in enumerate(bis)]
        a1.set(
            xlim=(0, 1),
            ylim=(min(0, th.min()) - 0.02, th.max() * 1.05),
            xlabel="depth from the centre $x$",
            ylabel=S.label("theta"),
        )
        a1.axvline(1, color=S.AXIS, lw=0.8)
        a1.text(
            0.985, 0.04, "cooled\nsurface", ha="right", va="bottom", transform=a1.transAxes, color=S.INK2, fontsize=8
        )
        a1.legend(loc="lower left")
        trace = [a2.plot([], [], color=S.PALETTE[k])[0] for k in range(len(bis))]
        dots = [a2.plot([], [], "o", color=S.PALETTE[k], ms=5)[0] for k in range(len(bis))]
        a2.set(
            xscale="log",
            xlim=(t[1], t[-1]),
            ylim=a1.get_ylim(),
            xlabel=S.label("t") + " (Fourier)",
            ylabel=r"surface temperature $\theta(1, t)$",
        )
        title = fig.suptitle("")

        def update(i):
            for k in range(len(bis)):
                prof[k].set_data(x, th[k, i])
                trace[k].set_data(t[1 : i + 1], th[k, 1 : i + 1, -1])
                dots[k].set_data([t[max(i, 1)]], [th[k, max(i, 1), -1]])
            title.set_text(f"Heated slab cooled at its surface   t = {t[i]:.4f}")

        return fig, update

    return Scene(
        "slab",
        "Convectively cooled slab",
        "Temperature profiles of a slab with internal heating and a convectively cooled surface: the Biot number "
        "sets how much of the temperature drop sits inside the slab and how much at its surface (exact series).",
        len(t),
        p["fps"],
        make,
        "analytic eigen-series (src/physics/analytic.py, problem B_slab)",
    )


def compute_boundary_layer(p: dict) -> dict:
    from aerosynapse.thermal.numerics import fd1d
    from aerosynapse.thermal.physics import analytic as an

    pes = np.geomspace(p["Pe_min"], p["Pe_max"], p["frames"])
    x = np.linspace(0, 1, 800)
    exact = np.stack([an.steady_convdiff(x, pe) for pe in pes])
    out = {"Pe": pes, "x": x, "exact": exact}
    for sch in p["schemes"]:
        rows, xn = [], None
        for pe in pes:
            f = fd1d.ConvDiff1D(
                p["N"], scheme=sch, a=1.0, D=1.0 / pe, left=fd1d.dirichlet(0.0), right=fd1d.dirichlet(1.0)
            )
            rows.append(f.nodal(f.steady(), 0.0))
            xn = f.x
        out[f"fd_{sch}"] = np.stack(rows)
        out["x_fd"] = xn
    return out


def scene_boundary_layer(p: dict, cache: Path) -> Scene:
    d, _ = cached(cache, "boundary_layer", p, lambda: compute_boundary_layer(p))
    pes, x, ex, xf = d["Pe"], d["x"], d["exact"], d["x_fd"]
    names = {"central": "central differences", "upwind1": "first-order upwind", "quick": "QUICK"}

    def make():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        (le,) = ax.plot([], [], color=S.REF, lw=1.6, label="exact")
        lf = [
            ax.plot([], [], "o-", ms=3.2, lw=0.8, color=S.PALETTE[k], label=f"{names.get(s, s)} ({p['N']} cells)")[0]
            for k, s in enumerate(p["schemes"])
        ]
        allv = np.concatenate([np.ravel(d[f"fd_{s}"]) for s in p["schemes"]] + [np.ravel(ex)])
        lo, hi = min(-0.1, float(allv.min())), max(1.1, float(allv.max()))  # every overshoot stays visible
        pad = 0.06 * (hi - lo)
        ax.set(xlim=(0, 1), ylim=(lo - pad, hi + 3 * pad), xlabel=S.label("x"), ylabel=S.label("theta"))
        ax.legend(loc="upper left")
        title = ax.set_title("")

        def update(i):
            le.set_data(x, ex[i])
            for k, s in enumerate(p["schemes"]):
                lf[k].set_data(xf, d[f"fd_{s}"][i])
            title.set_text(
                f"Steady convection-diffusion   Pe = {pes[i]:.3g}   cell Péclet number = {pes[i] / p['N']:.2g}"
            )

        return fig, update

    return Scene(
        "boundary_layer",
        "Thermal boundary layer and discretisation schemes",
        "As the Péclet number grows the temperature change is squeezed into a thin layer at the outflow. Once the "
        "cell Péclet number exceeds 2, central differences oscillate and first-order upwinding smears the layer.",
        len(pes),
        p["fps"],
        make,
        "analytic solution and src/numerics/fd1d.py",
    )


# ============================================================================ 2D, Navier-Stokes + energy solver
def _advect(px, py, xc, yc, U, V, dt, Lx, Ly, solid=None):
    """One RK2 step of tracer particles in the cell-centred velocity (bilinear), clipped to the domain."""
    from scipy.interpolate import RegularGridInterpolator

    # fill_value=None: linear extrapolation in the half cell between the outer cell centres and the boundary, so
    # tracers keep moving through inlets and outlets (a zero fill there would trap them at the domain edge)
    iu = RegularGridInterpolator((yc, xc), U, bounds_error=False, fill_value=None)
    iv = RegularGridInterpolator((yc, xc), V, bounds_error=False, fill_value=None)

    def vel(x, y):
        q = np.stack([y, x], 1)
        return iu(q), iv(q)

    u1, v1 = vel(px, py)
    xm, ym = px + 0.5 * dt * u1, py + 0.5 * dt * v1
    u2, v2 = vel(xm, ym)
    return np.clip(px + dt * u2, 0, Lx), np.clip(py + dt * v2, 0, Ly)


def _run_ns(
    s, t_end: float, n_frames: int, n_particles: int, rng_seed: int, Lx: float, Ly: float, seed_box=None, inject=None
) -> dict:
    """Advance a MAC solver from its current state to t_end, storing n_frames snapshots of theta, vorticity, speed and
    the tracer positions (tracers advected with the snapshot velocity between snapshots, in sub-steps)."""
    g = s.g
    rng = np.random.default_rng(rng_seed)
    x0, x1, y0, y1 = seed_box or (0.02 * Lx, 0.98 * Lx, 0.02 * Ly, 0.98 * Ly)
    px, py = rng.uniform(x0, x1, n_particles), rng.uniform(y0, y1, n_particles)
    times = np.linspace(0, t_end, n_frames)
    TH, W, SP, PX, PY = [], [], [], [], []
    for k, tt in enumerate(times):
        if tt > s.t:
            s.run(tt)
        U, V = s.cell_velocity()
        if k > 0:
            dt_f = times[k] - times[k - 1]
            nsub = max(1, int(np.ceil(dt_f / 0.05)))
            for _ in range(nsub):
                px, py = _advect(px, py, g.xc, g.yc, U, V, dt_f / nsub, Lx, Ly)
            if inject is not None:
                px, py = inject(px, py, rng)
        TH.append(s.theta.copy() if s.theta is not None else np.zeros((g.ny, g.nx)))
        W.append(s.vorticity())
        SP.append(np.hypot(U, V))
        PX.append(px.copy())
        PY.append(py.copy())
    return {
        "t": times,
        "theta": np.array(TH),
        "vort": np.array(W),
        "speed": np.array(SP),
        "px": np.array(PX),
        "py": np.array(PY),
        "xc": g.xc,
        "yc": g.yc,
    }


def compute_cavity(p: dict) -> dict:
    from aerosynapse.thermal.numerics import fvm2d as fv
    from aerosynapse.thermal.numerics import ns_projection as ns

    n = p["N"]
    en = ns.EnergySpec(
        Pe=p["Re"] * p["Pr"],
        bcs={
            "top": fv.dirichlet2(1.0),
            "bottom": fv.dirichlet2(0.0),
            "left": fv.neumann2(0.0),
            "right": fv.neumann2(0.0),
        },
    )
    s = ns.MACNavierStokes2D(fv.Grid2D(n, n), p["Re"], {"top": ns.wall(u=1.0)}, energy=en)
    s.set_state(np.zeros_like(s.u), np.zeros_like(s.v), theta=np.zeros((n, n)))
    return _run_ns(s, p["T"], p["frames"], p["particles"], 0, 1.0, 1.0)


def compute_natural(p: dict) -> dict:
    """Differentially heated square cavity from rest: Boussinesq buoyancy Ri theta e_y with free-fall scaling
    (Ri = 1, Re = sqrt(Ra / Pr), Pe = Re Pr); hot left wall theta = +1/2, cold right -1/2, adiabatic top / bottom."""
    from aerosynapse.thermal.numerics import fvm2d as fv
    from aerosynapse.thermal.numerics import ns_projection as ns

    n = p["N"]
    re = float(np.sqrt(p["Ra"] / p["Pr"]))
    en = ns.EnergySpec(
        Pe=re * p["Pr"],
        bcs={
            "left": fv.dirichlet2(0.5),
            "right": fv.dirichlet2(-0.5),
            "top": fv.neumann2(0.0),
            "bottom": fv.neumann2(0.0),
        },
    )
    s = ns.MACNavierStokes2D(fv.Grid2D(n, n), re, {}, energy=en, Ri=1.0)
    s.set_state(np.zeros_like(s.u), np.zeros_like(s.v), theta=np.zeros((n, n)))
    return _run_ns(s, p["T"], p["frames"], p["particles"], 1, 1.0, 1.0)


def compute_graetz(p: dict) -> dict:
    """Thermal entry of a fully developed channel flow (Graetz problem), walls suddenly heated to theta = 1."""
    from aerosynapse.thermal.numerics import fvm2d as fv
    from aerosynapse.thermal.numerics import ns_projection as ns

    L, nx, ny = p["L"], p["nx"], p["ny"]
    g = fv.Grid2D(nx, ny, L, 1.0)
    en = ns.EnergySpec(
        Pe=p["Re"] * p["Pr"],
        bcs={
            "left": fv.dirichlet2(0.0),
            "right": fv.neumann2(0.0),
            "bottom": fv.dirichlet2(1.0),
            "top": fv.dirichlet2(1.0),
        },
    )
    s = ns.MACNavierStokes2D(
        g, p["Re"], {"left": ns.wall(u=lambda s_, t: 6 * s_ * (1 - s_)), "right": ns.outflow()}, energy=en
    )
    u0 = 6 * g.yc[:, None] * (1 - g.yc[:, None]) * np.ones((1, nx + 1))
    s.set_state(u0, np.zeros_like(s.v), theta=np.zeros((ny, nx)))

    def inject(px, py, rng):  # particles leaving at the outlet re-enter at the inlet
        out = px >= L - 1e-9
        px = np.where(out, rng.uniform(0.0, 0.1, px.size), px)
        py = np.where(out, rng.uniform(0.02, 0.98, px.size), py)
        return px, py

    d = _run_ns(s, p["T"], p["frames"], p["particles"], 2, L, 1.0, seed_box=(0.0, L, 0.02, 0.98), inject=inject)
    th = d["theta"]
    w = np.maximum(6 * d["yc"] * (1 - d["yc"]), 0)
    d["theta_bulk"] = (th * w[None, :, None]).sum(1) / w.sum()
    return d


def compute_obstacle(p: dict) -> dict:
    from aerosynapse.thermal.numerics import fvm2d as fv
    from aerosynapse.thermal.numerics import ns_projection as ns

    L, nx, ny = p["L"], p["nx"], p["ny"]
    g = fv.Grid2D(nx, ny, L, 1.0)
    X, Y = g.Xc, g.Yc
    c, h = p["centre"], p["size"] / 2
    solid = (np.abs(X - c[0]) < h) & (np.abs(Y - c[1]) < h)
    en = ns.EnergySpec(
        Pe=p["Re"] * p["Pr"],
        solid_theta=1.0,
        # bounded advection of temperature in the wake (cell Peclet numbers ~10): a TVD limiter, set in the config
        scheme=p.get("energy_scheme", "tvd_vanleer"),
        bcs={
            "left": fv.dirichlet2(0.0),
            "right": fv.neumann2(0.0),
            "bottom": fv.neumann2(0.0),
            "top": fv.neumann2(0.0),
        },
    )
    s = ns.MACNavierStokes2D(
        g,
        p["Re"],
        # convective outflow, as for the verified unsteady Schaefer-Turek 2D-2 case (Step 1.7): vortices leave the
        # domain with backflow, which the zero-gradient outflow does not tolerate
        {"left": ns.wall(u=lambda s_, t: 6 * s_ * (1 - s_)), "right": ns.outflow(convective=True)},
        solid=solid,
        energy=en,
    )
    # a symmetric start never breaks symmetry numerically: a small random cross-flow perturbation (amplitude
    # `perturb`) triggers the wake instability, as in any physical experiment; it decays if the flow is stable
    v0 = p.get("perturb", 0.0) * np.random.default_rng(4).standard_normal(s.v.shape)
    s.set_state(np.zeros_like(s.u), v0, theta=np.zeros((ny, nx)))
    s.run(p["spinup"])  # let the wake develop before recording
    s.t = 0.0

    def inject(px, py, rng):
        out = px >= L - 1e-9
        inside = (np.abs(px - c[0]) < h) & (np.abs(py - c[1]) < h)
        bad = out | inside
        px = np.where(bad, rng.uniform(0.0, 0.1, px.size), px)
        py = np.where(bad, rng.uniform(0.05, 0.95, px.size), py)
        return px, py

    d = _run_ns(s, p["T"], p["frames"], p["particles"], 3, L, 1.0, seed_box=(0.0, L, 0.05, 0.95), inject=inject)
    d["solid"] = solid
    return d


def scene_cavity(p: dict, cache: Path) -> Scene:
    d, _ = cached(cache, "cavity", p, lambda: compute_cavity(p))
    return _side_by_side(
        "cavity",
        f"Heated lid-driven cavity, Re = {p['Re']:g}",
        "The lid starts moving at t = 0 and drives a primary vortex; the hot lid's heat is carried down "
        "along the walls (left: temperature with tracer particles; right: vorticity).",
        d,
        p,
    )


def scene_natural(p: dict, cache: Path) -> Scene:
    d, _ = cached(cache, "natural", p, lambda: compute_natural(p))
    return _side_by_side(
        "natural_convection",
        f"Natural convection, Ra = {p['Ra']:.0e}".replace("e+0", "e"),
        "A square cavity with a hot left and a cold right wall, fluid initially at rest: buoyant boundary "
        "layers rise and sink along the walls and set up one convection roll (left: temperature with "
        "tracers; right: vorticity).",
        d,
        p,
    )


def _side_by_side(name, title, caption, d, p) -> Scene:
    t = d["t"]

    def make():
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.2))
        lo, hi = float(d["theta"].min()), float(d["theta"].max())
        im1 = a1.imshow(
            d["theta"][0],
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="inferno",
            vmin=lo,
            vmax=hi,
            interpolation="bilinear",
        )
        S.colorbar(fig, im1, a1, S.label("theta"), shrink=0.85)
        sc = a1.scatter(d["px"][0], d["py"][0], s=3, c="white", alpha=0.8, lw=0)
        lim = float(np.percentile(np.abs(d["vort"]), 99)) or 1.0
        im2 = a2.imshow(
            d["vort"][0],
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap=S.DIV_CMAP,
            norm=TwoSlopeNorm(0, -lim, lim),
            interpolation="bilinear",
        )
        S.colorbar(fig, im2, a2, r"vorticity $\omega$", shrink=0.85)
        for ax, lab in ((a1, "temperature and tracers"), (a2, "vorticity")):
            ax.set(aspect="equal", xlabel=S.label("x"), ylabel=S.label("y"), title=lab)
            ax.grid(False)
        sup = fig.suptitle("")

        def update(i):
            im1.set_data(d["theta"][i])
            sc.set_offsets(np.c_[d["px"][i], d["py"][i]])
            im2.set_data(d["vort"][i])
            sup.set_text(f"{title}   t = {t[i]:.2f}")

        return fig, update

    return Scene(
        name,
        title,
        caption,
        len(t),
        p["fps"],
        make,
        "MAC Navier-Stokes + energy solver (src/numerics/ns_projection.py), run from rest; no network",
        meta={k: v for k, v in p.items() if k not in ("frames", "fps")},
    )


def scene_graetz(p: dict, cache: Path) -> Scene:
    d, _ = cached(cache, "graetz", p, lambda: compute_graetz(p))
    t, xc = d["t"], d["xc"]

    def make():
        import matplotlib.pyplot as plt

        fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.0, 4.4), gridspec_kw={"height_ratios": [1.15, 1]})
        im = a1.imshow(
            d["theta"][0],
            origin="lower",
            extent=(0, p["L"], 0, 1),
            cmap="inferno",
            vmin=0,
            vmax=1,
            interpolation="bilinear",
            aspect="auto",
        )
        S.colorbar(fig, im, a1, S.label("theta"))
        sc = a1.scatter(d["px"][0], d["py"][0], s=2.5, c="white", alpha=0.8, lw=0)
        a1.set(xlabel=S.label("x"), ylabel=S.label("y"), title="temperature and tracer particles")
        a1.grid(False)
        (ln,) = a2.plot([], [], color=S.PINN)
        a2.set(xlim=(0, p["L"]), ylim=(0, 1.02), xlabel=S.label("x"), ylabel=r"bulk temperature $\theta_b$")
        sup = fig.suptitle("")

        def update(i):
            im.set_data(d["theta"][i])
            sc.set_offsets(np.c_[d["px"][i], d["py"][i]])
            ln.set_data(xc, d["theta_bulk"][i])
            sup.set_text(f"Graetz channel: thermal entry, Pe = {p['Re'] * p['Pr']:g}   t = {t[i]:.2f}")

        return fig, update

    return Scene(
        "graetz",
        "Graetz channel: thermal entry",
        "Fully developed flow between two plates whose walls are suddenly heated: a thermal boundary layer "
        "grows from the walls while the cold fluid keeps entering, and the bulk temperature rises towards the "
        "wall temperature downstream.",
        len(t),
        p["fps"],
        make,
        "MAC Navier-Stokes + energy solver (src/numerics/ns_projection.py); no network",
        meta={k: v for k, v in p.items() if k not in ("frames", "fps")},
    )


def scene_obstacle(p: dict, cache: Path) -> Scene:
    d, _ = cached(cache, "obstacle", p, lambda: compute_obstacle(p))
    t, xc, yc = d["t"], d["xc"], d["yc"]
    solid = d["solid"].astype(float)

    def make():
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.0, 4.6))
        im1 = a1.imshow(
            d["theta"][0],
            origin="lower",
            extent=(0, p["L"], 0, 1),
            cmap="inferno",
            vmin=0,
            vmax=1,
            interpolation="bilinear",
        )
        S.colorbar(fig, im1, a1, S.label("theta"), shrink=0.9)
        sc = a1.scatter(d["px"][0], d["py"][0], s=2, c="white", alpha=0.75, lw=0)
        lim = float(np.percentile(np.abs(d["vort"]), 99)) or 1.0
        im2 = a2.imshow(
            d["vort"][0],
            origin="lower",
            extent=(0, p["L"], 0, 1),
            cmap=S.DIV_CMAP,
            norm=TwoSlopeNorm(0, -lim, lim),
            interpolation="bilinear",
        )
        S.colorbar(fig, im2, a2, r"vorticity $\omega$", shrink=0.9)
        for ax, lab in ((a1, "temperature and tracers"), (a2, "vorticity")):
            ax.contourf(xc, yc, solid, levels=[0.5, 1.5], colors=["#4a4a4a"], zorder=3)
            ax.set(aspect="equal", xlabel=S.label("x"), ylabel=S.label("y"), title=lab)
            ax.grid(False)
        sup = fig.suptitle("")

        def update(i):
            im1.set_data(d["theta"][i])
            sc.set_offsets(np.c_[d["px"][i], d["py"][i]])
            im2.set_data(d["vort"][i])
            sup.set_text(f"Flow past a heated square cylinder, Re = {p['Re']:g}   t = {t[i]:.2f}")

        return fig, update

    return Scene(
        "obstacle",
        "Flow past a heated square cylinder",
        "Channel flow around a hot square obstacle: the wake rolls up into alternating vortices that carry the "
        "heat downstream (top: temperature with tracers; bottom: vorticity). Re is based on the channel height and "
        f"the mean inflow velocity; based on the obstacle size and the peak inflow velocity it is "
        f"{p['Re'] * p['size'] * 1.5:g}. Illustration only, not a verified benchmark: at {p['nx'] / p['L']:g} cells "
        f"per unit length the cell Reynolds number is about {p['Re'] * 1.5 * p['L'] / p['nx']:.2g}, and the "
        "grid-scale wiggles in front of the obstacle are this resolution limit of the central scheme.",
        len(t),
        p["fps"],
        make,
        "MAC Navier-Stokes + energy solver with an immersed solid (src/numerics/ns_projection.py); no network",
        meta={k: v for k, v in p.items() if k not in ("frames", "fps")},
    )


SCENES = {
    "pulse": scene_pulse,
    "slab": scene_slab,
    "boundary_layer": scene_boundary_layer,
    "cavity": scene_cavity,
    "natural_convection": scene_natural,
    "graetz": scene_graetz,
    "obstacle": scene_obstacle,
}
