"""Flow pictures in the style of the thermal project's solver figures: a plane through a 3D flow run, drawn as two
panels on white paper. Top: the temperature rise (inferno: black free stream, bright where the air is compressed
and heated, at stagnation points, shocks and in the wake) with massless tracer particles carried by the in-plane
velocity; bottom: the vorticity normal to the plane (diverging, centred on zero). The solid's outline is the
design's exact section (from the kernel's distance function), not the solver's staircase.

* ``section_frames(d, plane)`` resamples the plane onto a uniform grid (cubic) and returns the fields;
* ``figure(d, plane)`` writes the still picture;
* ``animate(d, plane)`` writes an MP4 of the tracers streaming through the flow.

Planes: ``("y", y0)`` a cut at constant span (the wing section at ``y0``, or the centre plane at 0) and ``("z", z0)``
a cut seen from above.

Temperature: the solver is inviscid-compressible, so T is p / rho (ideal gas) and the rise is shown as
``(T - T_inf) / (T_0 - T_inf)`` with the stagnation temperature ``T_0 = T_inf (1 + 0.2 M_inf^2)``: 0 in the free
stream and 1 at a stagnation point.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from aerosynapse.sim.common import plt


def _design(d: Path) -> dict[str, Any] | None:
    try:
        return dict(json.loads((d / "case.json").read_text(encoding="utf-8"))["design"])
    except Exception:
        return None


def wing_station(design: dict[str, Any], eta: float = 0.55) -> tuple[float, float, float]:
    """Span position, leading-edge x and chord of the main wing at a fraction of its semi-span."""
    from aerosynapse.structures.wing import main_wing

    f = main_wing(design)
    t = f.get("translate", [0, 0, 0])
    span, cr, taper = float(f["span"]), float(f["root_chord"]), float(f.get("taper", 0.5))
    sweep, dih = np.radians(float(f.get("sweep", 30))), np.radians(float(f.get("dihedral", 0)))
    y = float(t[1]) + eta * span * np.cos(dih)
    x_le = float(t[0]) + eta * span * np.tan(sweep)
    return y, x_le, cr * (1 - (1 - taper) * eta)


def section_frames(d: str | Path, plane: tuple[str, float], window: tuple[float, float, float, float] | None = None, n: int = 560) -> dict[str, Any]:
    """The plane on a uniform grid: temperature rise, Mach number, in-plane velocity (units of the free-stream
    speed), normal vorticity (units of U/L), the solid mask from the exact geometry, and the axes."""
    from scipy.interpolate import RegularGridInterpolator

    from aerosynapse.cad import kernel
    from aerosynapse.sim.flow3d import Volume

    d = Path(d)
    V = Volume(d)
    axis, at = plane
    if axis == "y":
        j = int(np.clip(np.searchsorted(V.y, at), 1, len(V.y) - 1))
        w1 = float(np.clip((at - V.y[j - 1]) / max(V.y[j] - V.y[j - 1], 1e-12), 0, 1))

        def cut(a):  # x, z plane, linear between the two nearest y layers
            return (1 - w1) * a[:, j - 1, :] + w1 * a[:, j, :]
        ha, va, comps = V.x, V.z, (V.u, V.w)
    else:
        k = int(np.clip(np.searchsorted(V.z, at), 1, len(V.z) - 1))
        w1 = float(np.clip((at - V.z[k - 1]) / max(V.z[k] - V.z[k - 1], 1e-12), 0, 1))

        def cut(a):
            return (1 - w1) * a[:, :, k - 1] + w1 * a[:, :, k]
        ha, va, comps = V.x, V.y, (V.u, V.v)
    lo, hi = V.body_lo, V.body_hi
    if window is None:
        L = V.L
        if axis == "y":
            window = (lo[0] - 0.25 * L, hi[0] + 0.9 * L, lo[2] - 0.35 * L, hi[2] + 0.35 * L)
        else:
            window = (lo[0] - 0.25 * L, hi[0] + 0.9 * L, lo[1] - 0.25 * L, hi[1] + 0.25 * L)
    x0, x1, y0, y1 = window
    ny = max(60, int(n * (y1 - y0) / (x1 - x0)))
    X = np.linspace(x0, x1, n)
    Y = np.linspace(y0, y1, ny)
    XX, YY = np.meshgrid(X, Y)
    pts = np.c_[YY.ravel(), XX.ravel()]
    iy0, iy1 = max(0, int(np.searchsorted(va, y0)) - 3), min(len(va), int(np.searchsorted(va, y1)) + 3)
    ix0, ix1 = max(0, int(np.searchsorted(ha, x0)) - 3), min(len(ha), int(np.searchsorted(ha, x1)) + 3)
    solid_cells = cut(V.solid.astype(float)) > 0.5
    from scipy.ndimage import distance_transform_edt

    near = distance_transform_edt(solid_cells, return_indices=True)[1] if solid_cells.any() else None

    def sample(a, fill=None):
        """Cubic over the solver's cells near the window (smooth fields, smooth vorticity). The solver's solid
        cells (a staircase, thicker than the exact body) first take the value of the nearest fluid cell, so the flow
        reaches the exact wall with its slip velocity and the cubic does not ring there."""
        c = cut(a)
        if near is not None:
            c = c[near[0], near[1]] if fill is None else np.where(solid_cells, fill, c)
        c = c[ix0:ix1, iy0:iy1].T
        f = RegularGridInterpolator((va[iy0:iy1], ha[ix0:ix1]), c, method="cubic", bounds_error=False, fill_value=None)
        return f(pts).reshape(ny, n)

    t_ratio = (V.p / np.maximum(V.rho, 1e-12)) / (V.p_inf / V.rho_inf)
    t0 = 1 + 0.2 * V.minf**2
    temp = (sample(t_ratio) - 1) / max(t0 - 1, 1e-9)
    mach = sample(np.nan_to_num(V.mach, nan=V.minf))
    U = V.Uinf
    u1, u2 = sample(comps[0]) / U, sample(comps[1]) / U
    # the exact solid from the design's distance function (the solver's cells are a staircase)
    design = _design(d)
    if design is not None:
        P3 = np.zeros((XX.size, 3))
        P3[:, 0] = XX.ravel()
        P3[:, 1 if axis == "y" else 2] = at
        P3[:, 2 if axis == "y" else 1] = YY.ravel()
        solid = kernel.distance(design, P3).reshape(ny, n) < 0
    else:
        solid = sample(V.solid.astype(float), 1.0) > 0.5
    L = V.L
    dx, dy = (X[1] - X[0]) / L, (Y[1] - Y[0]) / L
    vort = np.gradient(u1, dy, axis=0) - np.gradient(u2, dx, axis=1)  # normal component, units U/L
    for a in (temp, mach, u1, u2, vort):
        a[solid] = np.nan
    u1 = np.where(solid, 0.0, u1)
    u2 = np.where(solid, 0.0, u2)
    return {"x": X, "y": Y, "temp": temp, "mach": mach, "u1": u1, "u2": u2, "vort": vort, "solid": solid, "L": L,
            "minf": V.minf, "axis": axis, "at": at}


def _title(d: Path, plane: tuple[str, float], label: str | None) -> str:
    info: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        info = json.loads((d / "run_info.json").read_text(encoding="utf-8")).get("flight", {})
    m, a = info.get("mach"), info.get("alpha")
    cond = (f", Mach {float(m):.2f}" if m is not None else "") + (f", $\\alpha$ = {float(a):g}$^\\circ$" if a is not None else "")
    return (label or ("Flow round the aircraft, centre plane" if plane == ("y", 0.0) else "Flow round the wing section")) + cond


def _panels(F: dict[str, Any], title: str):
    from matplotlib.colors import TwoSlopeNorm

    from aerosynapse.thermal.viz import style as TS

    pl = plt()
    X, Y = F["x"], F["y"]
    ext = (X[0], X[-1], Y[0], Y[-1])
    aspect = (Y[-1] - Y[0]) / (X[-1] - X[0])
    fig, (a1, a2) = pl.subplots(2, 1, figsize=(9.0, max(3.6, 2 * 9.0 * aspect * 0.86 + 1.2)))
    hi = float(np.clip(np.nanpercentile(F["temp"], 99.7), 0.15, 1.0))
    im1 = a1.imshow(F["temp"], origin="lower", extent=ext, cmap="inferno", vmin=0.0, vmax=hi, interpolation="bicubic")
    TS.colorbar(fig, im1, a1, r"$(T - T_\infty)/(T_0 - T_\infty)$", shrink=0.9)
    lim = float(np.nanpercentile(np.abs(F["vort"]), 99)) or 1.0
    im2 = a2.imshow(F["vort"], origin="lower", extent=ext, cmap=TS.DIV_CMAP, norm=TwoSlopeNorm(0, -lim, lim), interpolation="bicubic")
    TS.colorbar(fig, im2, a2, r"vorticity $\omega L / U$", shrink=0.9)
    hl, vl = ("x [m]", "z [m]") if F["axis"] == "y" else ("x [m]", "y [m]")
    for ax, lab in ((a1, "temperature and tracers"), (a2, "vorticity")):
        ax.contourf(X, Y, F["solid"].astype(float), levels=[0.5, 1.5], colors=["#4a4a4a"], zorder=3)
        ax.set(aspect="equal", xlabel=hl, ylabel=vl, title=lab, xlim=(X[0], X[-1]), ylim=(Y[0], Y[-1]))
        ax.grid(False)
    sup = fig.suptitle(title)
    return pl, fig, a1, im1, im2, sup


class Tracers:
    """Massless particles advected by the in-plane velocity (second-order Runge-Kutta, bilinear interpolation).
    They start evenly spread over the window; those that leave it or touch the solid start again at the upstream
    edge, spread over the distance the free stream covers in one step, so the stream stays even."""

    def __init__(self, F: dict[str, Any], n: int = 2400, seed: int = 3):
        from scipy.interpolate import RegularGridInterpolator

        self.F = F
        X, Y = F["x"], F["y"]
        self.iu = RegularGridInterpolator((Y, X), F["u1"], bounds_error=False, fill_value=1.0)
        self.iv = RegularGridInterpolator((Y, X), F["u2"], bounds_error=False, fill_value=0.0)
        self.isolid = RegularGridInterpolator((Y, X), F["solid"].astype(float), bounds_error=False, fill_value=0.0)
        self.rng = np.random.default_rng(seed)
        # for each solid grid point, the nearest fluid point: particles that touch the body slide along it
        from scipy.ndimage import distance_transform_edt

        solid = np.asarray(F["solid"], bool)
        _, (self.ny_, self.nx_) = distance_transform_edt(solid, return_indices=True) if solid.any() else (None, np.indices(solid.shape))
        self.px = self.rng.uniform(X[0], X[-1], n)
        self.py = self.rng.uniform(Y[0], Y[-1], n)
        for _ in range(6):  # none inside the solid
            inside = self.isolid(np.c_[self.py, self.px]) > 0.5
            if not inside.any():
                break
            self.px[inside] = self.rng.uniform(X[0], X[-1], int(inside.sum()))
            self.py[inside] = self.rng.uniform(Y[0], Y[-1], int(inside.sum()))

    def _respawn(self, bad: np.ndarray, reach: float) -> None:
        X, Y = self.F["x"], self.F["y"]
        for _ in range(4):
            k = int(bad.sum())
            if not k:
                return
            self.px[bad] = self.rng.uniform(X[0], X[0] + reach, k)
            self.py[bad] = self.rng.uniform(Y[0], Y[-1], k)
            bad = self.isolid(np.c_[self.py, self.px]) > 0.5

    def step(self, dt: float) -> None:  # dt in units of L / U, velocities in units of U
        L = self.F["L"]

        def vel(x, y):
            q = np.c_[y, x]
            return self.iu(q) * L, self.iv(q) * L
        u1, v1 = vel(self.px, self.py)
        xm, ym = self.px + 0.5 * dt * u1, self.py + 0.5 * dt * v1
        u2, v2 = vel(xm, ym)
        self.px, self.py = self.px + dt * u2, self.py + dt * v2
        X, Y = self.F["x"], self.F["y"]
        hit = self.isolid(np.c_[self.py, self.px]) > 0.5
        if hit.any():  # onto the nearest fluid point: the particle slides round the body and on into the wake
            i = np.clip(np.rint((self.py[hit] - Y[0]) / (Y[1] - Y[0])).astype(int), 0, len(Y) - 1)
            j = np.clip(np.rint((self.px[hit] - X[0]) / (X[1] - X[0])).astype(int), 0, len(X) - 1)
            self.py[hit] = Y[self.ny_[i, j]]
            self.px[hit] = X[self.nx_[i, j]]
        bad = (self.px > X[-1]) | (self.px < X[0]) | (self.py < Y[0]) | (self.py > Y[-1]) | (self.isolid(np.c_[self.py, self.px]) > 0.5)
        self._respawn(bad, reach=dt * L)


def _scatter(ax, tr: Tracers):
    return ax.scatter(tr.px, tr.py, s=1.4, c="white", alpha=0.8, lw=0, zorder=4)


def figure(d: str | Path, plane: tuple[str, float], name: str, label: str | None = None, window: tuple[float, float, float, float] | None = None,
           F: dict[str, Any] | None = None) -> str:
    d = Path(d)
    F = F or section_frames(d, plane, window)
    pl, fig, a1, *_ = _panels(F, _title(d, plane, label))
    tr = Tracers(F)
    for _ in range(10):  # a short run so the particles show the streamlines' bunching at the body
        tr.step(0.004)
    _scatter(a1, tr)
    f = d / f"{name}.png"
    fig.savefig(f, dpi=150)
    pl.close(fig)
    return f.name


def animate(d: str | Path, plane: tuple[str, float], name: str, label: str | None = None, seconds: float = 8.0, fps: int = 25,
            window: tuple[float, float, float, float] | None = None, F: dict[str, Any] | None = None) -> str | None:
    """An MP4 of the tracers streaming through the plane: the free stream crosses the window 1.6 times.

    The fields are drawn once and kept as a bitmap; each frame puts back that bitmap and draws only the particles
    and the clock (blitting), so a frame costs milliseconds instead of a full redraw."""
    import imageio_ffmpeg

    d = Path(d)
    F = F or section_frames(d, plane, window)
    pl, fig, a1, im1, im2, sup = _panels(F, _title(d, plane, label))
    fig.set_dpi(110)
    fig.canvas.draw()
    fig.set_layout_engine("none")  # keep the layout found by that draw for every frame
    tr = Tracers(F)
    sc = _scatter(a1, tr)
    clock = a1.text(0.99, 0.03, "", transform=a1.transAxes, ha="right", va="bottom", color="white", fontsize=9, zorder=5)
    sc.set_animated(True)
    clock.set_animated(True)
    canvas = fig.canvas
    canvas.draw()
    bg = canvas.copy_from_bbox(fig.bbox)
    w, h = canvas.get_width_height()
    W, H = w - w % 2, h - h % 2  # even sizes for yuv420p
    frames = int(seconds * fps)
    dt = 1.6 * (F["x"][-1] - F["x"][0]) / F["L"] / frames
    f = d / f"{name}.mp4"
    try:
        wr = imageio_ffmpeg.write_frames(str(f), (W, H), fps=fps, codec="libx264", pix_fmt_out="yuv420p", quality=8, macro_block_size=1)
        wr.send(None)
        for i in range(frames):
            tr.step(dt)
            sc.set_offsets(np.c_[tr.px, tr.py])
            clock.set_text(f"$t\\,U/L$ = {(i + 1) * dt:.3f}")
            canvas.restore_region(bg)
            a1.draw_artist(sc)
            a1.draw_artist(clock)
            buf = np.asarray(canvas.buffer_rgba())[:H, :W, :3]
            wr.send(np.ascontiguousarray(buf))
        wr.close()
    except Exception as e:
        (d / f"{name}_error.txt").write_text(str(e), encoding="utf-8")
        pl.close(fig)
        return None
    pl.close(fig)
    return f.name


def run(d: str | Path, movies: bool = True, eta: float = 0.55) -> list[str]:
    """The standard flow pictures of a 3D run: the wing section at ``eta`` of the semi-span and the centre plane."""
    d = Path(d)
    out = []
    design = _design(d)
    planes: list[tuple[str, tuple[str, float], str, tuple[float, float, float, float] | None]] = []
    if design is not None:
        with contextlib.suppress(Exception):
            y, xle, c = wing_station(design, eta)
            planes.append(("7_flow_wing_section", ("y", y), f"Flow round the wing section at {100 * eta:.0f} % span", (xle - 1.0 * c, xle + 3.0 * c, -0.8 * c, 0.8 * c)))
    planes.append(("8_flow_centre_plane", ("y", 0.0), "Flow round the aircraft, centre plane", None))
    for name, plane, label, window in planes:
        F = section_frames(d, plane, window)
        out.append(figure(d, plane, name, label, window, F=F))
        if movies:
            m = animate(d, plane, name, label, window=window, F=F)
            if m:
                out.append(m)
    return out
