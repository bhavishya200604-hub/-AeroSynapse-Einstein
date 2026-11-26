"""Step 1.6: 2D incompressible Navier–Stokes on a MAC grid (projection method) — verification + demos.

(a) Taylor–Green vortex (exact, unforced): spatial order of u and consistent pressure; divergence
(b) coupled NS + energy + Boussinesq MMS (Ri=0.5, Pr=0.71): spatial orders of u, v, p, theta
(c) temporal order of euler / ssprk2 / ssprk3 (coupled MMS, fixed grid, reference dt=2e-4)
(d) plane Poiseuille channel (inflow/outflow BCs): velocity error and dp/dx vs exact -8/Re
(e) obstacle mask vs true wall (horizontal and vertical surfaces): max difference
(f) pressure-Poisson solvers: SuperLU (factorised once) vs multigrid vs MG-PCG, cost per solve
(g) lid-driven cavity with passive heat transfer (uniform lid and cosh-regularised lid): steady state,
    divergence, energy balance, centreline profiles stored for the Step-1.7 benchmark comparison
(h) channel with heated square / staircase-circle obstacle (demo; no benchmark claims)

Run:  python experiments/p1_6_ns.py --profile smoke|full
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from aerosynapse.thermal.numerics import fvm2d as fv
from aerosynapse.thermal.numerics import ns_projection as ns
from aerosynapse.thermal.numerics.convergence import fit_order, observed_orders
from aerosynapse.thermal.physics import mms
from aerosynapse.thermal.utils.experiment import Experiment, cli_profile
from aerosynapse.thermal.viz.style import DIV_CMAP, PALETTE, SEQ_CMAP, apply_style, save_fig

PROFILES = {
    "smoke": {
        "N_tg": [16, 32, 64, 128],
        "N_mms": [16, 32, 64],
        "N_time": 24,
        "N_pois": [8, 16, 32],
        "N_solver": [64, 128],
        "cavity": [(100.0, 64)],
        "cavity_tol": 1e-4,
        "cavity_tmax": 60.0,
        "obstacle": {"Re": 20.0, "nx": 160, "ny": 32, "T": 8.0},
    },
    "full": {
        "N_tg": [16, 32, 64, 128, 256],
        "N_mms": [16, 32, 64, 128],
        "N_time": 32,
        "N_pois": [8, 16, 32, 64, 128],
        "N_solver": [64, 128, 256, 512],
        "cavity": [(100.0, 128), (400.0, 128), (1000.0, 128), (1000.0, 256)],
        "cavity_tol": 1e-6,
        "cavity_tmax": 200.0,
        "obstacle": {"Re": 100.0, "nx": 320, "ny": 64, "T": 40.0},
    },
}


def orders(Ns: list[int], errs: list[float]) -> dict:
    h = 1 / np.array(Ns, float)
    return {
        "err": errs,
        "pairwise": observed_orders(h, errs).tolist(),
        "order_last2": fit_order(h[-2:], errs[-2:])[0],
    }


def mms_bcs(ms: mms.ManufacturedNS) -> dict[str, ns.VelBC]:
    uf, vf = ms.u_fn, ms.v_fn
    return {
        "left": ns.wall(u=lambda s, t: uf(0 * s, s, t), v=lambda s, t: vf(0 * s, s, t)),
        "right": ns.wall(u=lambda s, t: uf(1 + 0 * s, s, t), v=lambda s, t: vf(1 + 0 * s, s, t)),
        "bottom": ns.wall(u=lambda s, t: uf(s, 0 * s, t), v=lambda s, t: vf(s, 0 * s, t)),
        "top": ns.wall(u=lambda s, t: uf(s, 1 + 0 * s, t), v=lambda s, t: vf(s, 1 + 0 * s, t)),
    }


def pressure_error(s: ns.MACNavierStokes2D, p_fn) -> float:
    p, pe = s.pressure(), p_fn(s.g.Xc, s.g.Yc, s.t + 0 * s.g.Xc)
    return float(np.max(np.abs((p - p.mean()) - (pe - pe.mean()))))


def tg_study(Ns: list[int], ex: Experiment) -> dict:
    Re, T = 100.0, 0.5
    tg = mms.taylor_green(Re)
    eu, ep, divs, wall = [], [], [], []
    for n in Ns:
        s = ns.MACNavierStokes2D(fv.Grid2D(n, n), Re, mms_bcs(tg))
        s.set_state(tg.u_fn(s.Xu, s.Yu, 0 * s.Xu), tg.v_fn(s.Xv, s.Yv, 0 * s.Xv), project=True)
        dmax = [0.0]
        t0 = time.perf_counter()
        info = s.run(T, callback=lambda so, h, dmax=dmax: dmax.__setitem__(0, max(dmax[0], so.max_div())))
        wall.append(time.perf_counter() - t0)
        eu.append(float(np.max(np.abs(s.u - tg.u_fn(s.Xu, s.Yu, T + 0 * s.Xu)))))
        ep.append(pressure_error(s, tg.p_fn))
        divs.append(dmax[0])
        ex.log(study="taylor_green", N=n, err_u=eu[-1], err_p=ep[-1], max_div=dmax[0], steps=info["steps"])
    return {"N": Ns, "u": orders(Ns, eu), "p": orders(Ns, ep), "max_div_over_run": divs, "wall_s": wall}


def make_mms(n: int, ms: mms.ManufacturedNS, rk: str = "ssprk3") -> ns.MACNavierStokes2D:
    g = fv.Grid2D(n, n)
    en = ns.EnergySpec(
        Pe=float(ms.Pe),
        source=ms.Q_fn,
        bcs=fv.bcs_from_mms(ms.theta_ms, {"left": (1, 0), "right": (2, 1), "bottom": (0, 1), "top": (1, 0)}),
    )
    s = ns.MACNavierStokes2D(
        g, float(ms.Re), mms_bcs(ms), force=(ms.fx_fn, ms.fy_fn), energy=en, Ri=float(ms.Ri), rk=rk
    )
    # projected initial state: the sampled MMS field has max|D u| = O(h^2) (9e-3 at 24^2); starting from it
    # produced a spurious O(dt) error in the first smoke run (RK3 pairwise 2.96 -> 1.13)
    s.set_state(
        ms.u_fn(s.Xu, s.Yu, 0 * s.Xu),
        ms.v_fn(s.Xv, s.Yv, 0 * s.Xv),
        ms.theta_fn(g.Xc, g.Yc, 0 * g.Xc),
        project=True,
    )
    return s


def mms_study(Ns: list[int], ex: Experiment) -> dict:
    ms = mms.mms_ns_default(Re=50.0, Ri=0.5, Pr=0.71)
    T = 0.5
    e = {"u": [], "v": [], "p": [], "theta": []}
    divs = []
    for n in Ns:
        s = make_mms(n, ms)
        info = s.run(T)
        g = s.g
        e["u"].append(float(np.max(np.abs(s.u - ms.u_fn(s.Xu, s.Yu, T + 0 * s.Xu)))))
        e["v"].append(float(np.max(np.abs(s.v - ms.v_fn(s.Xv, s.Yv, T + 0 * s.Xv)))))
        e["p"].append(pressure_error(s, ms.p_fn))
        e["theta"].append(float(np.max(np.abs(s.theta - ms.theta_fn(g.Xc, g.Yc, T + 0 * g.Xc)))))
        divs.append(info["max_div"])
        ex.log(study="mms_ns", N=n, **{f"err_{k}": v[-1] for k, v in e.items()}, max_div=divs[-1])
    return {"N": Ns, **{k: orders(Ns, v) for k, v in e.items()}, "final_max_div": divs}


def time_study(n: int, ex: Experiment) -> dict:
    ms = mms.mms_ns_default(Re=50.0, Ri=0.5, Pr=0.71)
    T = 0.5
    ref = make_mms(n, ms)
    ref.run(T, dt=2e-4)
    dts = [0.01, 0.005, 0.0025, 0.00125]
    out = {"N": n, "dt": dts}
    for rk in ("euler", "ssprk2", "ssprk3"):
        errs = []
        for dt in dts:
            s = make_mms(n, ms, rk)
            s.run(T, dt=dt)
            errs.append(
                float(
                    max(
                        np.abs(s.u - ref.u).max(),
                        np.abs(s.v - ref.v).max(),
                        np.abs(s.theta - ref.theta).max(),
                    )
                )
            )
            ex.log(study="time_ns", rk=rk, dt=dt, err=errs[-1])
        out[rk] = {
            "err": errs,
            "pairwise": observed_orders(dts, errs).tolist(),
            "order_fit": fit_order(dts, errs)[0],
            "formal": ns.RK_ORDER[rk],
        }
    return out


def channel(nx: int, ny: int, k: int = 0, Re: float = 20.0) -> ns.MACNavierStokes2D:
    y0 = k / ny
    solid = None
    if k:
        solid = np.zeros((ny + k, nx), bool)
        solid[:k, :] = True
    g = fv.Grid2D(nx, ny + k, 2.0, 1.0 + y0)
    prof = lambda s, t: np.where(s > y0, 4 * (s - y0) * (1 + y0 - s), 0.0)  # noqa: E731
    s = ns.MACNavierStokes2D(g, Re, {"left": ns.wall(u=prof), "right": ns.outflow()}, solid=solid)
    s.set_state(np.zeros_like(s.u), np.zeros_like(s.v))
    return s


def poiseuille_study(Ns: list[int], ex: Experiment) -> dict:
    Re = 20.0
    eu, edp, steps = [], [], []
    for n in Ns:
        s = channel(2 * n, n, 0, Re)
        info = s.run(60.0, steady_tol=1e-9)
        eu.append(float(np.max(np.abs(s.u - 4 * s.Yu * (1 - s.Yu)))))
        p = s.pressure()
        dpdx = float(np.mean(np.diff(p[:, n // 2 : 3 * n // 2], axis=1)) / s.g.dx)
        edp.append(abs(dpdx + 8 / Re))
        steps.append(info["steps"])
        ex.log(study="poiseuille", N=n, err_u=eu[-1], dpdx=dpdx, steps=info["steps"], max_div=info["max_div"])
    return {"N": Ns, "u": orders(Ns, eu), "dpdx": orders(Ns, edp), "steps": steps, "Re": Re}


def mask_study() -> dict:
    a, b = channel(32, 16, 0), channel(32, 16, 4)
    for s in (a, b):
        s.run(1.0, dt=2e-3)
    n, k = 16, 3
    lid = lambda s, t: np.where(s < 1 + 1e-12, np.sin(np.pi * s) ** 2, 0.0)  # noqa: E731
    c = ns.MACNavierStokes2D(fv.Grid2D(n, n), 100.0, {"top": ns.wall(u=lid)})
    sol = np.zeros((n, n + k), bool)
    sol[:, n:] = True
    d = ns.MACNavierStokes2D(fv.Grid2D(n + k, n, 1 + k / n, 1.0), 100.0, {"top": ns.wall(u=lid)}, solid=sol)
    for s in (c, d):
        s.set_state(np.zeros_like(s.u), np.zeros_like(s.v))
        s.run(0.4, dt=4e-3)
    return {
        "horizontal_max_du": float(np.abs(a.u - b.u[4:]).max()),
        "horizontal_max_dp": float(np.abs(a.pressure() - b.pressure()[4:]).max()),
        "vertical_max_du": float(np.abs(c.u - d.u[:, : n + 1]).max()),
        "vertical_max_dv": float(np.abs(c.v - d.v[:, :n]).max()),
    }


def solver_study(Ns: list[int], ex: Experiment) -> dict:
    out = {}
    rng = np.random.default_rng(0)
    for n in Ns:
        rec = {}
        for m in ("splu", "mg", "mgcg"):
            t0 = time.perf_counter()
            s = ns.MACNavierStokes2D(fv.Grid2D(n, n), 100.0, {"top": ns.wall(u=1.0)}, poisson=m)
            setup = time.perf_counter() - t0
            u, v = rng.standard_normal(s.u.shape), rng.standard_normal(s.v.shape)
            t0 = time.perf_counter()
            reps = 3
            for _ in range(reps):
                uu, vv, _ = s.project(u.copy(), v.copy(), 0.0)
            per = (time.perf_counter() - t0) / reps
            rec[m] = {
                "setup_s": setup,
                "per_projection_s": per,
                "max_div_after": float(np.abs(s.divergence(uu, vv)).max()),
            }
            ex.log(
                study="poisson_solver",
                N=n,
                method=m,
                setup_s=setup,
                per_projection_s=per,
                max_div=rec[m]["max_div_after"],
            )
        out[f"N{n}"] = rec
    return out


def cosh_lid(C: float = 50.0):
    return lambda s, t: 1.0 - np.cosh(C * (s - 0.5)) / np.cosh(0.5 * C)


def cavity_study(cases: list[tuple[float, int]], tol: float, tmax: float, ex: Experiment) -> tuple[dict, dict]:
    out, fields = {}, {}
    Pr = 0.71
    for Re, n in cases:
        for lid_name, lid in (("uniform", 1.0), ("cosh50", cosh_lid(50.0))):
            en = ns.EnergySpec(
                Pe=Re * Pr,
                bcs={
                    "top": fv.dirichlet2(1.0),
                    "bottom": fv.dirichlet2(0.0),
                    "left": fv.neumann2(0.0),
                    "right": fv.neumann2(0.0),
                },
            )
            s = ns.MACNavierStokes2D(fv.Grid2D(n, n), Re, {"top": ns.wall(u=lid)}, energy=en)
            s.set_state(np.zeros_like(s.u), np.zeros_like(s.v), theta=np.zeros((n, n)))
            t0 = time.perf_counter()
            dmax = [0.0]

            def cb(
                so: ns.MACNavierStokes2D,
                h: float,
                dmax: list = dmax,
                Re: float = Re,
                n: int = n,
                lid_name: str = lid_name,
            ) -> None:
                if so.nsteps % 200 == 0:
                    dmax[0] = max(dmax[0], so.max_div())
                    ex.log(study="cavity", Re=Re, N=n, lid=lid_name, t=so.t, dt=h, ke=so.kinetic_energy())

            info = s.run(tmax, steady_tol=tol, callback=cb)
            wall = time.perf_counter() - t0
            psi = s.streamfunction()
            j, i = np.unravel_index(np.argmin(psi), psi.shape)
            q = s.energy.boundary_heat_flow(s.theta.ravel(), s.t)
            Pe = Re * Pr
            key = f"Re{Re:g}_N{n}_{lid_name}"
            out[key] = {
                "t_final": s.t,
                "steps": info["steps"],
                "steady_change_rate": info["last_change_rate"],
                "reached_steady_tol": info["last_change_rate"] < tol,
                "max_div_sampled": dmax[0],
                "final_max_div": info["max_div"],
                "dt_min": info["dt_min"],
                "dt_max": info["dt_max"],
                "wall_s": wall,
                "psi_min": float(psi[j, i]),
                "psi_min_xy": [float(s.g.xf[i]), float(s.g.yf[j])],
                "Nu_top_in": q["top"] * Pe,
                "Nu_bottom_out": -q["bottom"] * Pe,
                "heat_balance_rel": (q["top"] + q["bottom"] + q["left"] + q["right"]) / abs(q["top"]),
                "centreline_u_at_x0.5": {"y": s.g.yc.tolist(), "u": s.u[:, n // 2].tolist()},
                "centreline_v_at_y0.5": {"x": s.g.xc.tolist(), "v": s.v[n // 2, :].tolist()},
            }
            fields[key] = s
    return out, fields


def obstacle_demo(cfg: dict, ex: Experiment) -> tuple[dict, dict]:
    Re, nx, ny, T = cfg["Re"], cfg["nx"], cfg["ny"], cfg["T"]
    g = fv.Grid2D(nx, ny, 5.0, 1.0)
    X, Y = g.Xc, g.Yc
    shapes = {
        "square": (X > 1.0) & (X < 1.25) & (Y > 0.375) & (Y < 0.625),
        "circle_staircase": (X - 1.125) ** 2 + (Y - 0.5) ** 2 < 0.125**2,
    }
    out, fields = {}, {}
    for name, solid in shapes.items():
        en = ns.EnergySpec(
            Pe=Re * 0.71,
            solid_theta=1.0,
            bcs={
                "left": fv.dirichlet2(0.0),
                "right": fv.neumann2(0.0),
                "bottom": fv.neumann2(0.0),
                "top": fv.neumann2(0.0),
            },
        )
        s = ns.MACNavierStokes2D(
            g,
            Re,
            {"left": ns.wall(u=lambda s_, t: 4 * s_ * (1 - s_)), "right": ns.outflow()},
            solid=solid,
            energy=en,
        )
        s.set_state(np.zeros_like(s.u), np.zeros_like(s.v), theta=np.zeros((ny, nx)))
        t0 = time.perf_counter()
        info = s.run(T)
        out[name] = {
            "solid_cells": int(solid.sum()),
            "steps": info["steps"],
            "wall_s": time.perf_counter() - t0,
            "final_max_div": info["max_div"],
            "max_abs_normal_velocity_on_solid_faces": float(
                max(np.abs(s.u[s.solid_u & ~s._mirror_u]).max(), np.abs(s.v[s.solid_v & ~s._mirror_v]).max())
            ),
            "net_boundary_outflow": s.net_boundary_flux(),
            "mean_outlet_theta": float(np.mean(s.theta[:, -1])),
        }
        fields[name] = s
    return out, fields


def figures(res: dict, cav: dict, obs: dict, ex: Experiment, tag: str) -> list[str]:
    paths = []
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    h = 1 / np.array(res["tg"]["N"], float)
    axs[0].loglog(h, res["tg"]["u"]["err"], "o-", label=f"TG u (p={res['tg']['u']['order_last2']:.2f})")
    axs[0].loglog(h, res["tg"]["p"]["err"], "s-", label=f"TG p (p={res['tg']['p']['order_last2']:.2f})")
    hm = 1 / np.array(res["mms"]["N"], float)
    for k in ("u", "p", "theta"):
        axs[0].loglog(hm, res["mms"][k]["err"], "^--", ms=4, label=f"MMS {k} (p={res['mms'][k]['order_last2']:.2f})")
    axs[0].loglog(h, 0.5 * h**2, "k:", lw=0.8, label="$O(h^2)$")
    axs[0].set(xlabel="$h$", ylabel=r"$\|e\|_\infty$", title="NS spatial convergence")
    axs[0].legend(fontsize=6)
    tr = res["time"]
    for rk in ("euler", "ssprk2", "ssprk3"):
        axs[1].loglog(tr["dt"], tr[rk]["err"], "o-", label=f"{rk} (p={tr[rk]['order_fit']:.2f})")
    axs[1].set(xlabel=r"$\Delta t$", ylabel="max error vs ref", title=f"Temporal order, {tr['N']}$^2$ MMS")
    axs[1].legend(fontsize=7)
    for m in ("splu", "mg", "mgcg"):
        Ns = [int(k[1:]) for k in res["solver"]]
        axs[2].loglog(Ns, [res["solver"][f"N{n}"][m]["per_projection_s"] for n in Ns], "o-", label=m)
    axs[2].set(xlabel="$N$ (N$^2$ cells)", ylabel="s / projection", title="Pressure solver cost (this CPU)")
    axs[2].legend()
    paths += save_fig(fig, f"p1_6_ns_convergence_{tag}", ex.dir)

    keys = list(cav)[:2]
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.4), gridspec_kw={"width_ratios": [1.25, 1.25, 1, 1]})
    for ax, key in zip(axs[:2], keys, strict=True):
        s = cav[key]
        im = ax.contourf(s.g.xc, s.g.yc, s.theta, 20, cmap=SEQ_CMAP)
        psi = s.streamfunction()
        ax.contour(s.g.xf, s.g.yf, psi, 14, colors="w", linewidths=0.5)
        ax.set(title=key.replace("_", " "), xlabel="$x^*$", ylabel="$y^*$", aspect="equal")
        fig.colorbar(im, ax=ax, label=r"$\theta$", shrink=0.8)
    # centrelines in two panels with their own axes (u against y, v against x); the two lids differ only near the
    # lid, so they are told apart by line style as well as colour
    for i, key in enumerate(keys):
        s = cav[key]
        lid = key.split("_")[-1]
        ls = "-" if i == 0 else "--"
        # MAC grid: u lives on vertical faces, so u[:, nx / 2] is exactly x = 0.5; v[ny / 2, :] is exactly y = 0.5
        axs[2].plot(s.u[:, s.g.nx // 2], s.g.yc, ls, color=PALETTE[i], label=lid)
        axs[3].plot(s.g.xc, s.v[s.g.ny // 2, :], ls, color=PALETTE[i], label=lid)
    axs[2].set(xlabel="$u(0.5, y^*)$", ylabel="$y^*$", title="vertical centreline")
    axs[3].set(xlabel="$x^*$", ylabel="$v(x^*, 0.5)$", title="horizontal centreline")
    for ax in axs[2:]:
        ax.legend(title="lid", loc="best")
    paths += save_fig(fig, f"p1_6_cavity_{tag}", ex.dir)

    fig, axs = plt.subplots(2, 2, figsize=(11, 4.2))
    for row, (name, s) in enumerate(obs.items()):
        w = s.vorticity()
        # corners surrounded by solid cells carry mirror-ghost values, not physical vorticity: mask them
        sp_ = np.pad(s.solid, 1, constant_values=False)
        inside = sp_[:-1, :-1] & sp_[1:, :-1] & sp_[:-1, 1:] & sp_[1:, 1:]
        w = np.where(inside, np.nan, w)
        lim = np.percentile(np.abs(w), 99)
        im = axs[row, 0].pcolormesh(s.g.xf, s.g.yf, w, cmap=DIV_CMAP, vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=axs[row, 0], label=r"$\omega$")
        im = axs[row, 1].pcolormesh(s.g.xf, s.g.yf, s.theta, cmap=SEQ_CMAP, vmin=0, vmax=1)
        fig.colorbar(im, ax=axs[row, 1], label=r"$\theta$")
        for ax in axs[row]:
            ax.set(aspect="equal", ylabel="$y^*$", title=f"{name}, Re={res['obstacle_cfg']['Re']:g}, t*={s.t:g}")
    axs[1, 0].set_xlabel("$x^*$")
    axs[1, 1].set_xlabel("$x^*$")
    paths += save_fig(fig, f"p1_6_obstacle_{tag}", ex.dir)
    return [str(p) for p in paths]


def main() -> None:
    profile = cli_profile()
    prof = PROFILES[profile]
    apply_style()
    with Experiment(f"p1_6_ns_{profile}", {"seed": 0, "profile": profile, **prof}) as ex:
        res: dict = {
            "tg": tg_study(prof["N_tg"], ex),
            "mms": mms_study(prof["N_mms"], ex),
            "time": time_study(prof["N_time"], ex),
            "poiseuille": poiseuille_study(prof["N_pois"], ex),
            "mask_equivalence": mask_study(),
            "solver": solver_study(prof["N_solver"], ex),
            "obstacle_cfg": prof["obstacle"],
        }
        res["cavity"], cav = cavity_study(prof["cavity"], prof["cavity_tol"], prof["cavity_tmax"], ex)
        res["obstacle"], obs = obstacle_demo(prof["obstacle"], ex)
        ex.metrics.update(res)
        ex.metrics["figures"] = figures(res, cav, obs, ex, profile)
    r = res
    print(
        "TG  u",
        [f"{e:.2e}" for e in r["tg"]["u"]["err"]],
        np.round(r["tg"]["u"]["pairwise"], 3),
        "p",
        np.round(r["tg"]["p"]["pairwise"], 3),
        "maxdiv",
        max(r["tg"]["max_div_over_run"]),
    )
    for k in ("u", "v", "p", "theta"):
        print(f"MMS {k:5s}", [f"{e:.2e}" for e in r["mms"][k]["err"]], np.round(r["mms"][k]["pairwise"], 3))
    for rk in ("euler", "ssprk2", "ssprk3"):
        print(f"time {rk:7s} fit={r['time'][rk]['order_fit']:.3f} pair={np.round(r['time'][rk]['pairwise'], 3)}")
    print(
        "Poiseuille u",
        np.round(r["poiseuille"]["u"]["pairwise"], 3),
        "dpdx",
        np.round(r["poiseuille"]["dpdx"]["pairwise"], 3),
        [f"{e:.2e}" for e in r["poiseuille"]["dpdx"]["err"]],
    )
    print("mask", r["mask_equivalence"])
    for k, v in r["solver"].items():
        print(
            "solver",
            k,
            {m: (round(d["per_projection_s"], 4), float(f"{d['max_div_after']:.1e}")) for m, d in v.items()},
        )
    for k, v in r["cavity"].items():
        print(
            "cavity",
            k,
            {a: (round(b, 5) if isinstance(b, float) else b) for a, b in v.items() if not isinstance(b, dict)},
        )
    for k, v in r["obstacle"].items():
        print("obstacle", k, v)


if __name__ == "__main__":
    main()
