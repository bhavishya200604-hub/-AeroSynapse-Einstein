r"""Full-3D flow analysis around the whole aircraft: run the flow solver on a full model (no symmetry plane) with the
whole volume saved, then extract what the flow does everywhere around the object.

* **Fields** in every cell: Mach number, pressure coefficient, vorticity and the Q-criterion
  :math:`Q=\tfrac12(\lVert\Omega\rVert^2-\lVert S\rVert^2)` (positive in vortex cores).
* **Streamlines** from a seed rake upstream that covers the whole aircraft plus rings around the wing tips, integrated
  with second-order Runge-Kutta through the trilinear velocity field until they leave the box or touch the body.
* **Isosurfaces** (marching cubes): vortex cores at a Q level from the field's own distribution, and the sonic surface
  (M = 1) when the flow is transonic or supersonic.
* **Figures**: convergence history (residual and forces), surface pressure maps (top and bottom), section pressure
  distributions on the wing, and a 3D render (streamlines, vortex cores, Cp on the body).
* **viz3d.json** for Studio's 3D view: streamlines with Mach along them and the isosurface meshes (OBJ).

Cases (flight states): cruise, transonic, supersonic, high_alpha, sideslip (beta 5 deg, needs the full model) and gust
(time-accurate, a vertical sine gust). Qualities draft / standard / fine set the cell size (length/30, /48, /70) and steps.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from aerosynapse.sim.common import NEON, STYLE, finish, plt, save

G = 1.4
CASES: dict[str, dict[str, float]] = {
    "cruise": {"mach": 0.85, "alpha": 2.0, "beta": 0.0, "altitude": 11000.0},
    "transonic": {"mach": 0.95, "alpha": 6.0, "beta": 0.0, "altitude": 5000.0},
    "supersonic": {"mach": 1.6, "alpha": 2.0, "beta": 0.0, "altitude": 12000.0},
    "high_alpha": {"mach": 0.3, "alpha": 25.0, "beta": 0.0, "altitude": 6000.0},
    "sideslip": {"mach": 0.6, "alpha": 4.0, "beta": 5.0, "altitude": 6000.0},
    "gust": {"mach": 0.7, "alpha": 3.0, "beta": 0.0, "altitude": 3000.0, "gust_amplitude": 0.05, "gust_wavelength": 20.0},
}
QUALITY: dict[str, tuple[float, int]] = {"live": (18.0, 260), "draft": (30.0, 900), "standard": (48.0, 2500), "fine": (70.0, 3500)}


# ------------------------------------------------------------------ running
def run_case(design: dict[str, Any], case: str, quality: str, out: str | Path) -> dict[str, Any]:
    from aerosynapse.cad import kernel
    from aerosynapse.cfd import runner
    from aerosynapse.workflow.stages import reference

    out = Path(out)
    L = kernel.info(design, h=0.1)["length"]
    ref = reference(design, L)
    cpl, steps = QUALITY[quality]
    c = CASES[case]
    fl = runner.Flight(mach=c["mach"], alpha=c["alpha"], beta=c["beta"], altitude=c["altitude"], gust_amplitude=c.get("gust_amplitude", 0.0),
                       gust_wavelength=c.get("gust_wavelength", 30.0))
    steady = case != "gust"
    if steady:  # stop once the mean forces settle, with room to get there (see setup.STEADY_STEPS)
        from aerosynapse.sim.setup import STEADY_STEPS

        steps = max(steps, STEADY_STEPS.get(quality, steps))
    num = runner.Numerics(cells_per_length=cpl, steps=steps, max_cells=2.4e6, residual_drop=1e-5, save_volume=True, full_model=True,
                          unsteady=not steady, force_band=0.005 if steady and quality != "live" else 0.0)
    t0 = time.time()
    r = runner.run(design, fl, ref, out, num)
    r["case"], r["quality"], r["wall_seconds"] = case, quality, time.time() - t0
    (out / "run_info.json").write_text(json.dumps({"case": case, "quality": quality, "flight": c, "reference": ref}, indent=1), encoding="utf-8")
    return r


# ------------------------------------------------------------------ the volume
class Volume:
    def __init__(self, d: Path):
        h = json.loads((d / "volume.json").read_text(encoding="utf-8"))
        self.x, self.y, self.z = (np.asarray(h[k], float) for k in ("x", "y", "z"))
        nx, ny, nz = len(self.x), len(self.y), len(self.z)
        names = list(h.get("vars") or ["rho", "u", "v", "w", "p", "solid"])
        a = np.fromfile(d / "volume.bin", np.float32).reshape(len(names), nz, ny, nx).transpose(0, 3, 2, 1)  # var, x, y, z
        self.rho, self.u, self.v, self.w, self.p = (a[i].astype(np.float64) for i in range(5))
        self.solid = a[5] > 0.5
        # statistics of a time-accurate run: tke / U^2, p_rms / q, u_rms / U, |vorticity|
        self.stats = {n: a[i].astype(np.float64) for i, n in enumerate(names) if i >= 6}
        self.samples = int(h.get("averaged_samples", 0))
        f = ~self.solid[0]
        self.rho_inf = float(self.rho[0][f].mean())
        self.U = np.array([self.u[0][f].mean(), self.v[0][f].mean(), self.w[0][f].mean()])
        self.p_inf = float(self.p[0][f].mean())
        self.Uinf = float(np.linalg.norm(self.U))
        with np.errstate(divide="ignore", invalid="ignore"):
            self.mach = np.sqrt(self.u**2 + self.v**2 + self.w**2) / np.sqrt(G * np.maximum(self.p, 1e-12) / np.maximum(self.rho, 1e-12))
        self.mach[self.solid] = np.nan
        self.cp = (self.p - self.p_inf) / (0.5 * self.rho_inf * self.Uinf**2)
        self.cp[self.solid] = np.nan
        self.minf = float(np.nanmean(self.mach[0]))
        idx = np.argwhere(self.solid)
        if len(idx):
            lo, hi = idx.min(0), idx.max(0)
            self.body_lo = np.array([self.x[lo[0]], self.y[lo[1]], self.z[lo[2]]])
            self.body_hi = np.array([self.x[hi[0]], self.y[hi[1]], self.z[hi[2]]])
        else:
            self.body_lo, self.body_hi = np.array([self.x[0], self.y[0], self.z[0]]), np.array([self.x[-1], self.y[-1], self.z[-1]])
        self.L = float(self.body_hi[0] - self.body_lo[0])

    def crop(self, up: float = 0.3, down: float = 1.0, side: float = 0.35) -> tuple[slice, slice, slice]:
        lo = self.body_lo - np.array([up, side, side]) * self.L
        hi = self.body_hi + np.array([down, side, side]) * self.L
        sl = []
        for c, a, b in zip((self.x, self.y, self.z), lo, hi, strict=True):
            i0, i1 = int(np.searchsorted(c, a)), int(np.searchsorted(c, b))
            sl.append(slice(max(0, i0 - 1), min(len(c), i1 + 1)))
        return tuple(sl)  # type: ignore[return-value]

    def q_criterion(self, sl: tuple[slice, slice, slice]) -> np.ndarray:
        x, y, z = self.x[sl[0]], self.y[sl[1]], self.z[sl[2]]
        u, v, w = self.u[sl], self.v[sl], self.w[sl]
        ux, uy, uz = np.gradient(u, x, y, z)
        vx, vy, vz = np.gradient(v, x, y, z)
        wx, wy, wz = np.gradient(w, x, y, z)
        q = -0.5 * (ux**2 + vy**2 + wz**2) - (uy * vx + uz * wx + vz * wy)
        q *= (self.L / max(self.Uinf, 1e-9)) ** 2
        q[self.solid[sl]] = 0.0
        return q

    def interpolators(self, sl: tuple[slice, slice, slice]) -> Any:
        from scipy.interpolate import RegularGridInterpolator as Interp

        g = (self.x[sl[0]], self.y[sl[1]], self.z[sl[2]])
        vel = np.stack([self.u[sl], self.v[sl], self.w[sl]], -1)
        m = np.nan_to_num(self.mach[sl], nan=0.0)
        return (Interp(g, vel, bounds_error=False, fill_value=np.nan), Interp(g, self.solid[sl].astype(float), bounds_error=False, fill_value=1.0),
                Interp(g, m, bounds_error=False, fill_value=np.nan))


def streamlines(V: Volume, sl: tuple[slice, slice, slice], n_y: int = 26, n_z: int = 13) -> list[dict[str, Any]]:
    vel, sol, mach = V.interpolators(sl)
    x0 = V.body_lo[0] - 0.12 * V.L
    span = max(V.body_hi[1] - V.body_lo[1], 0.2 * V.L)
    yc, zc = 0.5 * (V.body_lo[1] + V.body_hi[1]), 0.5 * (V.body_lo[2] + V.body_hi[2])
    hz = max(V.body_hi[2] - V.body_lo[2], 0.08 * V.L)
    Y, Z = np.meshgrid(np.linspace(yc - 0.62 * span, yc + 0.62 * span, n_y), np.linspace(zc - 0.9 * hz, zc + 0.9 * hz, n_z), indexing="ij")
    seeds = [np.c_[np.full(Y.size, x0), Y.ravel(), Z.ravel()]]
    # rings around both wing tips catch the tip vortices
    tips = np.argwhere(V.solid)
    if len(tips):
        for sgn in (-1, 1):
            j = tips[:, 1]
            k = int(np.argmax(sgn * V.y[j]))
            ty, tz = V.y[j[k]], V.z[tips[k, 2]]
            for a in np.linspace(0, 2 * np.pi, 14, endpoint=False):
                for r in (0.03, 0.07):
                    seeds.append(np.array([[x0, ty - sgn * 0.02 * span + r * span * np.cos(a), tz + r * span * np.sin(a)]]))
    P = np.vstack(seeds)
    cellmin = float(min(np.diff(V.x[sl[0]]).min(), np.diff(V.y[sl[1]]).min(), np.diff(V.z[sl[2]]).min()))
    ds = max(0.6 * cellmin, V.L / 400)
    n = int(2.6 * V.L / ds)
    alive = np.ones(len(P), bool)
    paths = [[p.copy()] for p in P]
    m0 = mach(P)
    ms = [[float(m0[i])] for i in range(len(P))]
    for _ in range(n):
        idx = np.nonzero(alive)[0]
        if not len(idx):
            break
        p = P[idx]
        k1 = vel(p)
        sp = np.linalg.norm(k1, axis=1)
        ok = np.isfinite(sp) & (sp > 1e-6 * V.Uinf)
        h = np.where(ok, ds / np.where(ok, sp, 1.0), 0.0)[:, None]
        k2 = vel(p + 0.5 * h * np.nan_to_num(k1))
        q = p + h * np.nan_to_num(k2)
        stop = ~ok | ~np.isfinite(q).all(1) | (sol(q) > 0.5) | ~np.isfinite(vel(q)).all(1)
        mq = mach(q)
        alive[idx[stop]] = False
        go = idx[~stop]
        P[go] = q[~stop]
        for t, i in zip(np.nonzero(~stop)[0], go, strict=True):
            paths[i].append(q[t].copy())
            ms[i].append(float(mq[t]))
    out = []
    for pth, m in zip(paths, ms, strict=True):
        if len(pth) >= 6:
            a = np.asarray(pth)[::2]
            out.append({"pts": np.round(a, 4).tolist(), "s": np.round(np.asarray(m)[::2], 4).tolist()})
    return out


def isosurface(field: np.ndarray, level: float, coords: tuple[np.ndarray, np.ndarray, np.ndarray], path: Path) -> int:
    from skimage.measure import marching_cubes

    if not (np.nanmin(field) < level < np.nanmax(field)):
        return 0
    verts, faces, _, _ = marching_cubes(np.nan_to_num(field, nan=float(np.nanmin(field))), level)
    xyz = np.c_[[np.interp(verts[:, a], np.arange(len(coords[a])), coords[a]) for a in range(3)]].T
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(f"v {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in xyz))
        f.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces))
    return int(len(faces))


# ------------------------------------------------------------------ figures
def figures(d: Path, V: Volume | None, out: Path, ref: dict[str, Any]) -> list[str]:
    from aerosynapse.cfd import runner

    pl = plt()
    figs = []
    try:
        h = runner.history(d)
        fig, ax = pl.subplots(1, 2, figsize=(11, 3.3))
        ax[0].semilogy(h["step"], h["residual"], color=NEON[0])
        ax[0].set(xlabel="step", ylabel="density residual", title="Convergence of the residual")
        ax[1].plot(h["step"], h["CL"], label="CL")
        ax[1].plot(h["step"], 10 * h["CD"], label="10 x CD")
        if "Cm" in h:
            ax[1].plot(h["step"], h["Cm"], label="Cm")
        ax[1].set(xlabel="step", title="Forces and moment through the run")
        ax[1].legend(fontsize=7.5)
        figs.append(save(fig, out, "1_convergence"))
    except Exception:
        pass
    try:
        s = np.genfromtxt(d / "surface.csv", delimiter=",", names=True)
        fig, ax = pl.subplots(1, 2, figsize=(11, 4.2))
        for k, (sel, ttl) in enumerate(((s["nz"] > 0.2, "upper surface"), (s["nz"] < -0.2, "lower surface"))):
            sc = ax[k].scatter(s["x"][sel], s["y"][sel], c=s["cp"][sel], cmap="turbo_r", s=5, vmin=np.percentile(s["cp"], 2), vmax=np.percentile(s["cp"], 98))
            ax[k].set(aspect="equal", xlabel="x, m", ylabel="y, m", title=f"Pressure coefficient, {ttl}")
            fig.colorbar(sc, ax=ax[k], shrink=0.8, label="Cp")
        figs.append(save(fig, out, "2_surface_pressure"))
        ys = np.abs(s["y"])
        ymax = ys.max()
        fig, ax = pl.subplots(figsize=(9, 3.4))
        for k, eta in enumerate((0.35, 0.55, 0.75)):
            band = np.abs(ys - eta * ymax) < 0.04 * ymax
            if band.sum() > 10:
                xb = s["x"][band]
                ax.scatter((xb - xb.min()) / max(np.ptp(xb), 1e-9), s["cp"][band], s=8, color=NEON[k], label=f"{int(100 * eta)} % semispan")
        ax.invert_yaxis()
        ax.set(xlabel="x / local chord", ylabel="Cp", title="Wing pressure distributions (3D flow)")
        ax.legend(fontsize=7.5)
        figs.append(save(fig, out, "3_wing_sections"))
    except Exception:
        pass
    if V is not None:
        try:
            sl = V.crop(0.2, 0.6, 0.3)
            jm = int(np.argmin(np.abs(V.y[sl[1]] - 0.5 * (V.body_lo[1] + V.body_hi[1]))))
            fig, ax = pl.subplots(1, 2, figsize=(12, 3.6))
            m = V.mach[sl][:, jm, :].T
            im = ax[0].pcolormesh(V.x[sl[0]], V.z[sl[2]], m, cmap="turbo", shading="auto")
            fig.colorbar(im, ax=ax[0], label="Mach")
            ax[0].set(aspect="equal", xlabel="x, m", ylabel="z, m", title="Mach number on the centre plane")
            kz = int(np.argmin(np.abs(V.z[sl[2]] - 0.5 * (V.body_lo[2] + V.body_hi[2]))))
            im = ax[1].pcolormesh(V.x[sl[0]], V.y[sl[1]], V.cp[sl][:, :, kz].T, cmap="RdBu_r", shading="auto", vmin=-1, vmax=1)
            fig.colorbar(im, ax=ax[1], label="Cp")
            ax[1].set(aspect="equal", xlabel="x, m", ylabel="y, m", title="Pressure coefficient through the wing plane")
            figs.append(save(fig, out, "4_field_slices"))
        except Exception:
            pass
    return figs


def body_mesh(d: Path) -> Path | None:
    """The aircraft's own surface (from the design the solver ran) for 3D pictures and Studio."""
    from aerosynapse.cad import kernel

    f = d / "body.obj"
    if not f.exists():
        try:
            design = json.loads((d / "case.json").read_text(encoding="utf-8"))["design"]
            kernel.mesh(design, f, h=0.06)
        except Exception:
            return None
    return f


def render3d(d: Path, out: Path, lines: list[dict[str, Any]], meshes: list[Path], name: str = "5_flow_3d.png", view: str = "iso") -> str | None:
    """A 3D picture: the aircraft coloured by the solver's Cp, streamlines coloured by Mach, vortex cores and sonic surface."""
    try:
        import pyvista as pv
        from scipy.spatial import cKDTree

        pv.OFF_SCREEN = True
        s = np.genfromtxt(d / "surface.csv", delimiter=",", names=True)
        bm = body_mesh(d)
        p = pv.Plotter(off_screen=True, window_size=(1500, 860))
        plain = STYLE != "dark"
        if plain:
            p.set_background("white")
        else:
            p.set_background("#050b18", top="#18345f")
        ink = "black" if plain else "white"
        lo, hi = np.percentile(s["cp"], 2), np.percentile(s["cp"], 98)
        if bm is not None:
            body = pv.read(str(bm))
            pts = np.c_[s["x"], s["y"], s["z"]]
            q = body.points.copy()
            half = np.all(s["y"] >= -1e-3)
            if half:
                q[:, 1] = np.abs(q[:, 1])  # a half-model run: mirror the surface data
            _, i = cKDTree(pts).query(q)
            body["Cp"] = s["cp"][i]
            p.add_mesh(body, scalars="Cp", cmap="RdBu" if plain else "turbo_r", clim=(lo, hi), smooth_shading=True, specular=0.15 if plain else 0.4,
                       scalar_bar_args={"title": "Cp", "color": ink, "position_x": 0.82, "position_y": 0.1, "width": 0.06, "height": 0.6, "vertical": True})
            c = np.array(body.center)
            Lb = float(np.ptp(body.points[:, 0]))
        else:
            c, Lb = np.array([0.0, 0.0, 0.0]), 10.0
        for ln in lines[::2]:
            a = np.asarray(ln["pts"])
            if len(a) < 2:
                continue
            poly = pv.lines_from_points(a)
            poly["Mach"] = np.asarray(ln["s"])[: len(a)]
            if plain:
                p.add_mesh(poly, color="#4d4d4d", line_width=0.8, opacity=0.55)
            else:
                p.add_mesh(poly, scalars="Mach", cmap="cool", line_width=1.2, opacity=0.75, show_scalar_bar=False)
        for m in meshes:
            if m.exists() and m.stat().st_size > 100:
                col = ("#7a5c99" if "vortex" in m.name else "#c9a227") if plain else ("#f472b6" if "vortex" in m.name else "#fbbf24")
                p.add_mesh(pv.read(str(m)), color=col, opacity=0.4 if plain else 0.5, smooth_shading=True)
        if view == "iso":
            p.camera.position = tuple(c + np.array([-0.85, -1.05, 0.62]) * Lb)
        elif view == "top":
            p.camera.position = tuple(c + np.array([0.0, 0.0, 1.6]) * Lb)
        else:
            p.camera.position = tuple(c + np.array([0.0, -1.7, 0.1]) * Lb)
        p.camera.focal_point = tuple(c)
        p.camera.up = (0.0, 0.0, 1.0)
        f = out / name
        p.screenshot(str(f))
        p.close()
        return f.name
    except Exception as e:
        (out / "render3d_error.txt").write_text(str(e), encoding="utf-8")
        return None


# ------------------------------------------------------------------ post-processing
def wall_stats(d: Path) -> dict[str, float]:
    """Separated wall-area fraction and hottest wall temperature from the surface samples."""
    f = d / "surface.csv"
    if not f.exists():
        return {}
    s = np.genfromtxt(f, delimiter=",", names=True)
    if s.size == 0:
        return {}
    a = np.atleast_1d(s["area"])
    return {"separated_fraction": float((a * (np.atleast_1d(s["separated"]) > 0.5)).sum() / max(a.sum(), 1e-12)), "T_wall_max_K": float(np.nanmax(s["T_wall_K"]))}


def force_settling(d: Path) -> float | None:
    """Relative change of CL over the last tenth of the run: the force-convergence measure of a steady case."""
    f = d / "history.csv"
    if not f.exists():
        return None
    h = np.genfromtxt(f, delimiter=",", names=True)
    if h.size < 10 or "CL" not in (h.dtype.names or ()):
        return None
    cl = np.atleast_1d(h["CL"])
    tail = cl[int(0.9 * len(cl)):]
    return float((tail.max() - tail.min()) / max(abs(cl[-1]), 1e-6))


def postprocess(d: str | Path) -> dict[str, Any]:
    d = Path(d)
    info = json.loads((d / "run_info.json").read_text()) if (d / "run_info.json").exists() else {}
    solver = d / "solver_result.json"  # the flow solver's own result; result.json becomes this analysis' result
    if not solver.exists():
        shutil.copyfile(d / "result.json", solver)
    res_run = json.loads(solver.read_text())
    if "T_wall_max_K" not in res_run:  # an older run whose solver result was overwritten: estimate from the surface samples
        res_run.update(wall_stats(d), wall_stats_source="surface samples (all wall cells)")
    lines: list[str] = []
    res: dict[str, Any] = {"case": info.get("case", ""), "flight": info.get("flight", {}), "coefficients": res_run.get("coefficients"), "cells": res_run.get("cells"),
                           "separated_fraction": res_run.get("separated_fraction"), "T_wall_max_K": res_run.get("T_wall_max_K"),
                           "seconds": res_run.get("seconds"), "converged": res_run.get("converged"), "half_model": res_run.get("half_model")}
    V = Volume(d) if (d / "volume.bin").exists() else None
    viz: dict[str, Any] = {"title": f"3D flow: {info.get('case', '')}", "lines": [], "meshes": [], "volume": V is not None}
    if V is not None:
        sl = V.crop()
        coords = (V.x[sl[0]], V.y[sl[1]], V.z[sl[2]])
        t0 = time.time()
        st = streamlines(V, sl)
        viz["lines"] = st
        viz["line_scalar"] = "Mach"
        mm = np.concatenate([np.asarray(s_["s"]) for s_ in st]) if st else np.array([0.0, 1.0])
        viz["line_range"] = [float(np.nanpercentile(mm, 2)), float(np.nanpercentile(mm, 98))]
        q = V.q_criterion(sl)
        near = V.crop(0.05, 0.5, 0.15)
        qn = V.q_criterion(near)
        pos = qn[qn > 0]
        qlev = float(max(2.0, np.percentile(pos, 99.0))) if len(pos) else 10.0
        nq = isosurface(q, qlev, coords, d / "iso_vortex.obj")
        if nq:
            viz["meshes"].append({"obj": "iso_vortex.obj", "color": [0.96, 0.45, 0.71, 0.5] if STYLE == "dark" else [0.62, 0.52, 0.72, 0.32], "label": f"vortex cores, Q = {qlev:.1f} (U/L)^2"})
        mfield = np.where(np.isnan(V.mach[sl]), V.minf, V.mach[sl])
        ns = isosurface(mfield, 1.0, coords, d / "iso_sonic.obj") if np.nanmax(mfield) > 1.0 and np.nanmin(mfield) < 1.0 else 0
        if ns:
            viz["meshes"].append({"obj": "iso_sonic.obj", "color": [0.98, 0.75, 0.14, 0.4] if STYLE == "dark" else [0.85, 0.72, 0.38, 0.3], "label": "sonic surface, M = 1"})
        res.update({"streamlines": len(st), "vortex_core_faces": nq, "sonic_surface_faces": ns, "q_level": qlev, "mach_max": float(np.nanmax(V.mach[sl])),
                    "cp_min_field": float(np.nanmin(V.cp[sl])), "freestream_mach": V.minf, "post_seconds": time.time() - t0})
        lines += [f"Full-3D flow around the whole aircraft: {res['cells']} cells, {'half' if res['half_model'] else 'full'} model, {res['seconds']:.0f} s",
                  f"freestream Mach {V.minf:.3f}; highest Mach in the field {res['mach_max']:.3f}; lowest Cp {res['cp_min_field']:.3f}",
                  f"{len(st)} streamlines from a rake upstream and rings around the wing tips",
                  f"vortex cores (Q = {qlev:.1f} (U/L)^2): {nq} faces; sonic surface: "
                  + ("none (subsonic flow)" if not ns else f"{ns} faces" + (" (a small supersonic pocket over the body)" if res["mach_max"] < 1.1 and V.minf < 1 else ""))]
    rms = res_run.get("rms") or {}
    if res_run.get("samples"):  # time-accurate run: forces are averages over the statistics window
        cm = res_run.get("coefficients") or {}
        lines.append(f"time-accurate run: forces averaged over {int(res_run['samples'])} samples; CL {cm.get('CL', 0):.5f} +/- {rms.get('CL', 0):.5f} r.m.s., "
                     f"CD {cm.get('CD', 0):.5f} +/- {rms.get('CD', 0):.5f}")
        res["force_rms"] = rms
    else:
        settle = force_settling(d)
        if settle is not None:
            res["CL_change_last_10pct"] = settle
            lines.append(f"forces settled: CL changed by {100 * settle:.3f} % over the last 10 % of the steps")
    sq = info.get("sequencing") or {}
    if sq.get("sequenced"):
        lines.append(f"grid sequencing: coarse pass {sq.get('coarse_cells', 0):.0f} cells, {sq.get('coarse_steps', 0):.0f} steps, {sq.get('coarse_seconds', 0):.0f} s; "
                     f"full grid {sq.get('fine_steps', 0):.0f} steps, {sq.get('fine_seconds', 0):.0f} s (stopped on {sq.get('stop_reason')})")
        res["sequencing"] = sq
    elif res_run.get("stop_reason") == "forces":
        lines.append(f"stopped after {res_run.get('steps', 0):.0f} steps: the forces had settled")
    walls = int(res_run.get("wall_cells", 0) or 0) + int(res_run.get("thin_surface_cells", 0) or 0)
    res["wall_cells"] = walls
    if 0 < walls < 120:  # the workflow grid study: 139 wall cells (h 0.45 m) gave CL within 10 % of the finest grid; the live grid has about 50
        res["grid_warning"] = True
        lines.append(f"WARNING: coarse grid: only {walls} cells touch the aircraft, so the wings are thinner than a cell. The flow pattern is indicative; "
                     "forces are not reliable here (drag can even come out negative). Use standard or fine quality for numbers.")
    c = res_run.get("coefficients") or {}
    lines.append("forces: " + ", ".join(f"{k} {v:.5f}" for k, v in c.items() if k in ("CL", "CD", "CY", "Cl", "Cm", "Cn", "CD_friction")))
    lines.append(f"converged to the residual target: {res_run.get('converged')}; separated wall area {100 * res_run.get('separated_fraction', 0):.1f} %, hottest wall {res_run.get('T_wall_max_K', 0):.0f} K"
                 + (f" (from the {res_run['wall_stats_source']})" if res_run.get("wall_stats_source") else ""))
    figs = figures(d, V, d, info.get("reference", {}))
    for nm, vw in (("5_flow_3d.png", "iso"), ("6_flow_3d_top.png", "top")):
        r3 = render3d(d, d, viz["lines"], [d / m["obj"] for m in viz["meshes"]], nm, vw)
        if r3:
            figs.append(r3)
    S = info.get("setup") or {}
    mode = str(S.get("flow_pictures", "stills and movies"))
    if V is not None and mode != "none":  # the two-panel plane cuts: temperature and tracers over vorticity
        from aerosynapse.sim import flowviz

        t1 = time.time()
        try:
            fv = flowviz.run(d, movies=mode == "stills and movies", eta=float(S.get("flow_section_span", 55.0)) / 100)
            figs += [f for f in fv if f.endswith(".png")]
            res["flow_movies"] = [f for f in fv if f.endswith(".mp4")]
            lines.append(f"flow pictures: {', '.join(fv)} ({time.time() - t1:.0f} s)")
        except Exception as e:  # pictures never fail the analysis
            lines.append(f"flow pictures skipped: {e}")
    res["figures"] = figs
    (d / "viz3d.json").write_text(json.dumps(viz), encoding="utf-8")
    return finish(d, f"3D flow analysis: {info.get('case', '')}", lines, res)


def run_setup(design: dict[str, Any], setup: dict[str, Any] | None, out: str | Path) -> dict[str, Any]:
    """One 3D flow run with every input from a setup (see ``aerosynapse.sim.setup``, kind ``cfd``)."""
    from aerosynapse.cad import kernel
    from aerosynapse.cfd import runner
    from aerosynapse.sim import setup as SU
    from aerosynapse.workflow.stages import reference

    S = SU.merged("cfd", setup)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    L = kernel.info(design, h=0.1)["length"]
    ref = reference(design, L)
    cpl, steps = (float(S["cells_per_length"]), int(S["steps"])) if S["quality"] == "custom" else SU.QUALITY[str(S["quality"])]
    fl = runner.Flight(mach=float(S["mach"]), alpha=float(S["alpha"]), beta=float(S["beta"]), altitude=float(S["altitude"]), delta_T=float(S["delta_T"]),
                       gust_amplitude=float(S["gust_amplitude"]) / 100, gust_wavelength=float(S["gust_wavelength"]),
                       turbulence_intensity=float(S["turbulence_intensity"]) / 100, turbulence_length=float(S["turbulence_length"]))
    dom = [float(S[k]) for k in ("domain_up", "domain_down", "domain_side", "domain_vert")]
    unsteady = {"on": True, "off": False}.get(str(S["unsteady"]), fl.gust_amplitude > 0 or fl.turbulence_intensity > 0)
    num = runner.Numerics(cells_per_length=cpl, steps=steps, max_cells=float(S["max_cells"]), stretch=float(S["stretch"]), cfl=float(S["cfl"]),
                          unsteady=unsteady, average_from=int(steps * float(S["average_from"]) / 100) if unsteady else -1, sgs=str(S["sgs"]),
                          wall_model=bool(S["wall_model"]), viscous=bool(S["viscous"]), residual_drop=float(S["residual_drop"]),
                          save_volume=bool(S["save_volume"]), domain=dom if all(x > 0 for x in dom) else [], full_model=bool(S["full_model"]))
    extra_model = {"thin_surfaces": bool(S["thin_surfaces"])}
    if S["low_mach_fix"] != "auto":
        extra_model["low_mach_fix"] = S["low_mach_fix"] == "on"
    probes = []
    for k in ("probe_1", "probe_2", "probe_3"):
        txt = str(S.get(k) or "").strip()
        if txt:
            v = [float(t) for t in txt.replace(";", ",").split(",")]
            if len(v) != 3:
                raise ValueError(f"{k}: give x, y, z in metres")
            probes.append({"name": k.replace("_", " "), "at": v})
    num.force_band = float(S.get("force_band", 0.0)) / 100 if not unsteady else 0.0
    if num.force_band > 0 and S["quality"] != "custom":  # a steady run stopped on its forces: enough steps to settle
        num.steps = max(num.steps, SU.STEADY_STEPS.get(str(S["quality"]), num.steps))
    t0 = time.time()
    seq: dict[str, Any] = {"start": str(S.get("start", "free stream"))}
    if seq["start"] == "previous run" and str(S.get("start_folder") or "").strip():
        num.init_from = str(S["start_folder"]).strip()
    elif seq["start"] == "coarse pass first" and cpl >= 24:
        # grid sequencing: the same flow on a grid 0.55 times as fine (about 6 times fewer cells, larger time steps)
        # settles the large-scale field cheaply; the full grid then only has to refine it
        cnum = runner.Numerics(**{**num.__dict__, "cells_per_length": cpl * 0.55, "steps": min(800, max(300, steps // 3)), "save_volume": True,
                                  "unsteady": False, "average_from": -1, "force_band": 0.01, "init_from": ""})
        cflight = runner.Flight(**{**fl.__dict__, "gust_amplitude": 0.0, "turbulence_intensity": 0.0})
        rc = runner.run(design, cflight, ref, out / "coarse_pass", cnum, extra_model=extra_model)
        seq.update(coarse_seconds=rc.get("wall_seconds"), coarse_steps=rc.get("steps"), coarse_cells=rc.get("cells"))
        num.init_from = str(out / "coarse_pass")
    r = runner.run(design, fl, ref, out, num, extra_model=extra_model, probes=probes or None)
    r["wall_seconds"] = time.time() - t0
    seq.update(fine_seconds=r.get("seconds"), fine_steps=r.get("steps"), stop_reason=r.get("stop_reason"), sequenced=r.get("sequenced"))
    (out / "run_info.json").write_text(json.dumps({"case": str(S["preset"]), "quality": str(S["quality"]), "flight": {k: S[k] for k in
                                       ("mach", "alpha", "beta", "altitude", "delta_T", "turbulence_intensity", "turbulence_length", "gust_amplitude", "gust_wavelength")},
                                       "reference": ref, "setup": S, "sequencing": seq}, indent=1), encoding="utf-8")
    return r


def run(design: dict[str, Any], out: str | Path, case: str = "cruise", quality: str = "standard", setup: dict[str, Any] | None = None) -> dict[str, Any]:
    if setup is not None:
        run_setup(design, setup, out)
    else:
        run_case(design, case, quality, out)
    return postprocess(out)
