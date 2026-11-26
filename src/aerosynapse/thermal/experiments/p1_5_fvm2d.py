"""Step 1.5: 2D finite-volume energy solver verification and solver performance.

(a) steady MMS spatial order (upwind1 / central / quick), mixed D/N/R BCs, cellular flow, + energy balance
(b) transient MMS spatial order for TVD limiters (SSP-RK3)
(c) temporal order: implicit Euler, CN, BDF2, Douglas–Gunn ADI vs a Radau reference on the same grid
(d) 2D Robin conduction vs exact product solution theta = S(x,t) S(y,t) (1D slab eigen-series)
(e) multigrid: V-cycle convergence factor / iterations vs N, vs Jacobi-CG and MG-preconditioned CG
(f) implicit time-stepping cost per step: CN+SuperLU, CN+MGCG, ADI; transient energy balance (CN)

Run:  python experiments/p1_5_fvm2d.py --profile smoke|full
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
import sympy as sp

from aerosynapse.thermal.numerics import fvm2d as fv
from aerosynapse.thermal.numerics.adi import integrate_adi
from aerosynapse.thermal.numerics.convergence import fit_order, observed_orders
from aerosynapse.thermal.numerics.linalg import cg, jacobi_preconditioner
from aerosynapse.thermal.numerics.mol import solve_mol
from aerosynapse.thermal.numerics.multigrid import MultigridHelmholtz, laplacian_fv
from aerosynapse.thermal.numerics.time_integrators import integrate
from aerosynapse.thermal.physics import analytic as an
from aerosynapse.thermal.physics import mms
from aerosynapse.thermal.utils.experiment import Experiment, cli_profile
from aerosynapse.thermal.viz.style import DIV_CMAP, SEQ_CMAP, apply_style, save_fig

PROFILES = {
    "smoke": {
        "N_space": [16, 32, 64, 128],
        "N_tvd": [16, 32, 64],
        "N_time": 24,
        "N_prod": [16, 32, 64],
        "N_mg": [32, 64, 128, 256],
        "N_perf": [64, 128, 256],
    },
    "full": {
        "N_space": [16, 32, 64, 128, 256, 512],
        "N_tvd": [16, 32, 64, 128, 256],
        "N_time": 48,
        "N_prod": [16, 32, 64, 128, 256],
        "N_mg": [32, 64, 128, 256, 512, 1024],
        "N_perf": [64, 128, 256, 512, 1024],
    },
}
SPEC = {"left": (1.0, 0.0), "right": (2.0, 1.0), "bottom": (0.0, 1.0), "top": (3.0, 1.0)}
X, Y, T = mms.x, mms.y, mms.t
PSI = sp.sin(sp.pi * X) * sp.sin(sp.pi * Y) / sp.pi
PSIF = sp.lambdify((X, Y, T), PSI, "numpy")


def mms_problem(n: int, scheme: str, steady: bool, Pe: float) -> tuple[fv.FVEnergy2D, mms.ManufacturedSolution]:
    if steady:
        th = sp.cos(sp.pi * X / 2 + sp.Rational(1, 5)) * sp.sin(sp.pi * Y + sp.Rational(1, 7)) + X * Y**2 / 2
        u, v = mms.velocity_from_streamfunction(PSI)
        ms = mms.manufacture(th, [u, v], 1 / sp.nsimplify(Pe), 0)
    else:
        ms = mms.mms_2d_default(Pe=Pe)
    g = fv.Grid2D(n, n)
    vel = fv.FaceVelocity.from_streamfunction(g, lambda x, y, t: PSIF(x, y, t))
    return fv.FVEnergy2D(g, 1 / Pe, vel, source=ms.forcing, bcs=fv.bcs_from_mms(ms, SPEC), scheme=scheme), ms


def steady_study(Ns: list[int], ex: Experiment) -> dict:
    out = {}
    for sch in ("upwind1", "central", "quick"):
        errs, bal, div = [], [], []
        for n in Ns:
            p, ms = mms_problem(n, sch, True, 50.0)
            y = p.steady()
            errs.append(float(np.max(np.abs(y.reshape(n, n) - ms.exact(p.g.Xc, p.g.Yc, 0 * p.g.Xc)))))
            eb = p.energy_balance(y, 0.0)
            bal.append(abs(eb["residual"]) / eb["scale"])
            U, V = p.vel.at(0.0)
            div.append(float(np.max(np.abs(fv.discrete_divergence(p.g, U, V)))))
            ex.log(
                study="steady2d",
                scheme=sch,
                N=n,
                err_inf=errs[-1],
                balance_rel=bal[-1],
                storage_rel=abs(eb["storage_rate"]) / eb["scale"],
            )
        h = 1 / np.array(Ns, float)
        out[sch] = {
            "N": Ns,
            "err_inf": errs,
            "pairwise": observed_orders(h, errs).tolist(),
            "order_fit_last3": fit_order(h[-3:], errs[-3:])[0],
            "energy_balance_rel_max": max(bal),
            "max_div_h_u": max(div),
        }
    return out


def tvd_study(Ns: list[int], ex: Experiment) -> dict:
    out, Pe, Tf = {}, 50.0, 0.25
    for sch in ("central", "tvd_minmod", "tvd_vanleer", "tvd_superbee"):
        errs = []
        for n in Ns:
            p, ms = mms_problem(n, sch, False, Pe)
            h = 1 / n
            dt = min(0.2 * h, 0.1 * h * h * Pe)
            y0 = ms.exact(p.g.Xc, p.g.Yc, 0 * p.g.Xc).ravel()
            y = integrate(p, y0, 0.0, Tf, dt, "ssprk3")
            errs.append(float(np.max(np.abs(y.reshape(n, n) - ms.exact(p.g.Xc, p.g.Yc, Tf + 0 * p.g.Xc)))))
            ex.log(study="tvd2d", scheme=sch, N=n, err_inf=errs[-1])
        h = 1 / np.array(Ns, float)
        out[sch] = {
            "N": Ns,
            "err_inf": errs,
            "pairwise": observed_orders(h, errs).tolist(),
            "order_fit": fit_order(h, errs)[0],
        }
    return out


def temporal_study(n: int, ex: Experiment) -> dict:
    p, ms = mms_problem(n, "central", False, 20.0)
    y0 = ms.exact(p.g.Xc, p.g.Yc, 0 * p.g.Xc).ravel()
    Tf = 1.0
    ref = solve_mol(p, y0, 0.0, Tf, "Radau", 1e-12, 1e-14).y[:, -1]
    dts = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    out = {}
    runners = {
        m: (lambda dt, m=m: integrate(p, y0, 0.0, Tf, dt, m)) for m in ("implicit_euler", "crank_nicolson", "bdf2")
    }
    runners["adi_douglas_gunn"] = lambda dt: integrate_adi(p, y0, 0.0, Tf, dt)
    for m, run in runners.items():
        errs = [float(np.max(np.abs(run(dt) - ref))) for dt in dts]
        for dt, e in zip(dts, errs, strict=True):
            ex.log(study="time2d", method=m, dt=dt, err_inf=e)
        out[m] = {
            "dt": dts,
            "err_inf": errs,
            "pairwise": observed_orders(dts, errs).tolist(),
            "order_fit": fit_order(dts, errs)[0],
        }
    out["N"] = n
    return out


def product_problem(n: int, Bi: float) -> fv.FVEnergy2D:
    bcs = {"left": fv.neumann2(0.0), "bottom": fv.neumann2(0.0), "right": fv.robin2(Bi), "top": fv.robin2(Bi)}
    return fv.FVEnergy2D(fv.Grid2D(n, n), D=1.0, bcs=bcs, scheme="central")


def product_study(Ns: list[int], ex: Experiment) -> tuple[dict, dict]:
    """2D conduction, symmetry at x=0,y=0, Robin (Bi) at x=1,y=1, theta0=1: theta = S(x,t) S(y,t)."""
    out, fields = {}, {}
    Tf = 0.1
    for Bi in (1.0, 10.0):
        e_cn, e_adi = [], []
        for n in Ns:
            p = product_problem(n, Bi)
            y0 = np.ones(p.g.n)
            # fixed small dt: the IC theta=1 is incompatible with the Robin BC and CN is not L-stable, so a
            # coarse dt leaves undamped stiff modes (seen with dt = 4/n^2: erratic orders 2.7/3.5)
            dt = 1e-4
            exact = np.outer(an.slab_transient(p.g.yc, Tf, Bi), an.slab_transient(p.g.xc, Tf, Bi))
            ycn = integrate(p, y0, 0.0, Tf, dt, "crank_nicolson")
            yad = integrate_adi(p, y0, 0.0, Tf, dt)
            e_cn.append(float(np.max(np.abs(ycn.reshape(n, n) - exact))))
            e_adi.append(float(np.max(np.abs(yad.reshape(n, n) - exact))))
            ex.log(study="product2d", Bi=Bi, N=n, dt=dt, err_cn=e_cn[-1], err_adi=e_adi[-1])
            fields[(Bi, n)] = (p.g, ycn.reshape(n, n), exact)
        h = 1 / np.array(Ns, float)
        out[f"Bi{Bi:g}"] = {
            "N": Ns,
            "err_cn": e_cn,
            "err_adi": e_adi,
            "order_cn": fit_order(h[-3:], e_cn[-3:])[0],
            "order_adi": fit_order(h[-3:], e_adi[-3:])[0],
        }
    return out, fields


MG_BCS = {
    "dirichlet": {k: fv.dirichlet2(0.0) for k in fv.SIDES},
    "mixed": {
        "left": fv.dirichlet2(0.0),
        "right": fv.robin2(5.0),
        "bottom": fv.neumann2(0.0),
        "top": fv.robin2(0.5),
    },
    "neumann_singular": {k: fv.neumann2(0.0) for k in fv.SIDES},
}


def mg_study(Ns: list[int], ex: Experiment) -> tuple[dict, dict]:
    out, hist = {}, {}
    for name, bcs in MG_BCS.items():
        rec: dict = {
            "N": Ns,
            "mg_iters": [],
            "mg_rho": [],
            "mg_time_s": [],
            "mgcg_iters": [],
            "mgcg_time_s": [],
            "jacobi_cg_iters": [],
            "splu_time_s": [],
        }
        for n in Ns:
            g = fv.Grid2D(n, n)
            f = (np.cos(np.pi * g.Xc) * np.cos(2 * np.pi * g.Yc) + g.Xc * g.Yc).ravel()
            t0 = time.perf_counter()
            mg = MultigridHelmholtz(g, 0.0, 1.0, bcs)
            x, h = mg.solve(f, rtol=1e-10)
            tm = time.perf_counter() - t0
            fp = f - f.mean() if mg.singular else f
            t0 = time.perf_counter()
            _, hc = cg(mg.M, fp, M=mg.precondition, rtol=1e-10)
            tc = time.perf_counter() - t0
            if n <= 256:
                _, hj = cg(mg.M, fp, M=jacobi_preconditioner(mg.M), rtol=1e-10, maxiter=20000)
                nj = len(hj) - 1
            else:
                nj = None
            if not mg.singular and n <= 512:
                t0 = time.perf_counter()
                xs = sps.linalg.spsolve(mg.M.tocsc(), f)
                ts = time.perf_counter() - t0
                assert np.max(np.abs(xs - x)) < 1e-8 * max(1, np.max(np.abs(xs)))
            else:
                ts = None
            rho = (h[-1] / h[0]) ** (1 / (len(h) - 1))
            for k, v in (
                ("mg_iters", len(h) - 1),
                ("mg_rho", rho),
                ("mg_time_s", tm),
                ("mgcg_iters", len(hc) - 1),
                ("mgcg_time_s", tc),
                ("jacobi_cg_iters", nj),
                ("splu_time_s", ts),
            ):
                rec[k].append(v)
            ex.log(
                study="multigrid",
                bcs=name,
                N=n,
                mg_iters=len(h) - 1,
                mg_rho=rho,
                mgcg_iters=len(hc) - 1,
                jacobi_cg_iters=nj,
                mg_time_s=tm,
                splu_time_s=ts,
            )
            hist[(name, n)] = h
        out[name] = rec
    return out, hist


def perf_study(Ns: list[int], ex: Experiment) -> dict:
    """Heat equation with Robin cooling on all sides + uniform source; 10 CN/ADI steps per grid."""
    out: dict = {}
    Bi, Q, dt, nsteps = 2.0, 1.0, 1e-3, 10
    for n in Ns:
        bcs = {k: fv.robin2(Bi) for k in fv.SIDES}
        p = fv.FVEnergy2D(fv.Grid2D(n, n), D=1.0, bcs=bcs, source=lambda x, y, t: Q + 0 * x, scheme="central")
        _ = p.A
        y0 = np.zeros(p.g.n)
        L = laplacian_fv(p.g, bcs)

        def mg_factory(M: sps.spmatrix, p: fv.FVEnergy2D = p, L: sps.csr_matrix = L, bcs: dict = bcs):
            # M = I - k D L  ->  identify k from an off-diagonal entry, build matching multigrid
            r, c = L.nonzero()
            j = np.flatnonzero(r != c)[0]
            k = -M[r[j], c[j]] / (p.D * L[r[j], c[j]])
            mg = MultigridHelmholtz(p.g, 1.0, k * p.D, bcs)
            assert abs(mg.M - M).max() < 1e-10
            return lambda rhs: cg(mg.M, rhs, M=mg.precondition, rtol=1e-12)[0]

        rec = {}
        res = {}
        _ = p.split_operators()  # operator assembly (probing) excluded from all timings
        for name, run in (
            (
                "cn_splu",
                lambda k, p=p, y0=y0: integrate(p, y0, 0.0, k * dt, dt, "crank_nicolson", solver="splu"),
            ),
            (
                "cn_mgcg",
                lambda k, p=p, y0=y0, f=mg_factory: integrate(p, y0, 0.0, k * dt, dt, "crank_nicolson", solver=f),
            ),
            ("adi", lambda k, p=p, y0=y0: integrate_adi(p, y0, 0.0, k * dt, dt)),
        ):
            if name == "cn_splu" and n > 512:
                continue
            t0 = time.perf_counter()
            run(1)
            t1 = time.perf_counter() - t0
            t0 = time.perf_counter()
            res[name] = run(nsteps + 1)
            t_all = time.perf_counter() - t0
            per_step = (t_all - t1) / nsteps
            rec[f"{name}_ms_per_step"] = 1e3 * per_step
            rec[f"{name}_setup_ms"] = 1e3 * max(t1 - per_step, 0.0)
        ref = res.get("cn_splu", res["cn_mgcg"])
        rec["max_diff_cn_mgcg_vs_ref"] = float(np.max(np.abs(res["cn_mgcg"] - ref)))
        rec["max_diff_adi_vs_cn"] = float(np.max(np.abs(res["adi"] - ref)))
        # transient energy balance for CN: sum(theta^{N}) - sum(theta^0) == dt/2 sum_n (F^n + F^{n+1})
        storage = (np.sum(res["cn_mgcg"]) - np.sum(y0)) * p.g.vol
        rec["energy_storage"] = float(storage)
        out[f"N{n}"] = rec
        ex.log(study="perf", N=n, **rec)
    # CN energy identity on the smallest grid with the exact (direct) solver
    n = Ns[0]
    bcs = {k: fv.robin2(Bi) for k in fv.SIDES}
    p = fv.FVEnergy2D(fv.Grid2D(n, n), D=1.0, bcs=bcs, source=lambda x, y, t: Q + 0 * x, scheme="central")
    ys = [np.zeros(p.g.n)]
    integrate(p, ys[0], 0.0, 0.1, 1e-3, "crank_nicolson", callback=lambda k, t, y: ys.append(y.copy()))
    rate = [float(np.sum(p.rhs(y, 0.0)) * p.g.vol) for y in ys]  # = boundary inflow + source (telescoped)
    inflow = [sum(p.boundary_heat_flow(y, 0.0).values()) + Q * 1.0 for y in ys]
    stored = (np.sum(ys[-1]) - np.sum(ys[0])) * p.g.vol
    trap = 1e-3 * (0.5 * inflow[0] + sum(inflow[1:-1]) + 0.5 * inflow[-1])
    out["cn_energy_identity"] = {
        "N": n,
        "stored": float(stored),
        "trapezoid_inflow": float(trap),
        "abs_residual": float(abs(stored - trap)),
        "rate_vs_flux_max_diff": float(np.max(np.abs(np.array(rate) - np.array(inflow)))),
    }
    return out


def figures(res: dict, fields: dict, hist: dict, ex: Experiment, prof: dict, tag: str) -> list[str]:
    paths = []
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    for sch, r in res["steady"].items():
        axs[0].loglog(1 / np.array(r["N"]), r["err_inf"], "o-", label=f"{sch} (p={r['order_fit_last3']:.2f})")
    for sch, r in res["tvd"].items():
        axs[0].loglog(1 / np.array(r["N"]), r["err_inf"], "s--", ms=3, label=f"{sch}, transient")
    axs[0].set(xlabel="$h$", ylabel=r"$\|e\|_\infty$", title="2D MMS, Pe=50, mixed D/N/R")
    axs[0].legend(fontsize=6)
    for m, r in res["time"].items():
        if isinstance(r, dict):
            axs[1].loglog(r["dt"], r["err_inf"], "o-", label=f"{m} (p={r['order_fit']:.2f})")
    axs[1].set(
        xlabel=r"$\Delta t$",
        ylabel=r"$\|e\|_\infty$ vs Radau",
        title=f"Temporal order, {res['time']['N']}$^2$",
    )
    axs[1].legend(fontsize=6)
    for name in MG_BCS:
        for n in prof["N_mg"][-2:]:
            axs[2].semilogy(hist[(name, n)], "o-", ms=3, label=f"{name} N={n}")
    axs[2].set(xlabel="V-cycle", ylabel=r"$\|r\|/\|f\|$", title="Multigrid convergence")
    axs[2].legend(fontsize=6)
    paths += save_fig(fig, f"p1_5_fvm2d_convergence_{tag}", ex.dir)

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    n = prof["N_prod"][-1]
    g, num, exact = fields[(10.0, n)]
    im = axs[0].contourf(g.xc, g.yc, num, 20, cmap=SEQ_CMAP)
    fig.colorbar(im, ax=axs[0], label=r"$\theta$")
    axs[0].set(xlabel="$x^*$", ylabel="$y^*$", title=f"2D conduction, Bi=10, $t_F$=0.1, {n}$^2$", aspect="equal")
    err = num - exact
    lim = np.max(np.abs(err))
    im = axs[1].pcolormesh(g.xf, g.yf, err, cmap=DIV_CMAP, vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=axs[1], label=r"$\theta_h - S(x)S(y)$")
    axs[1].set(xlabel="$x^*$", title="Error vs product solution", aspect="equal")
    # cellular-flow demo: heated bottom, cooled top (Robin), Pe = 200
    nd_ = 128
    gd = fv.Grid2D(nd_, nd_)
    vel = fv.FaceVelocity.from_streamfunction(gd, lambda x, y, t: PSIF(x, y, t))
    pd = fv.FVEnergy2D(
        gd,
        1 / 200.0,
        vel,
        scheme="central",
        bcs={
            "bottom": fv.dirichlet2(1.0),
            "top": fv.robin2(5.0),
            "left": fv.neumann2(0.0),
            "right": fv.neumann2(0.0),
        },
    )
    th = pd.steady().reshape(nd_, nd_)
    im = axs[2].contourf(gd.xc, gd.yc, th, 20, cmap=SEQ_CMAP)
    Xs, Ys = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    axs[2].streamplot(
        Xs,
        Ys,
        np.sin(np.pi * Xs) * np.cos(np.pi * Ys),
        -np.cos(np.pi * Xs) * np.sin(np.pi * Ys),
        color="w",
        linewidth=0.5,
        density=0.8,
    )
    fig.colorbar(im, ax=axs[2], label=r"$\theta$")
    axs[2].set(xlabel="$x^*$", title="Cellular flow, Pe=200 (demo)", aspect="equal")
    paths += save_fig(fig, f"p1_5_fvm2d_fields_{tag}", ex.dir)
    nu_bottom = float(np.mean(pd.wall_diffusive_flux(th.ravel(), 0.0, "bottom")))
    ex.metrics["demo_cellular_Pe200_mean_bottom_flux"] = nu_bottom
    ex.metrics["demo_cellular_energy_balance"] = pd.energy_balance(th.ravel(), 0.0)
    return [str(p) for p in paths]


def main() -> None:
    profile = cli_profile()
    prof = PROFILES[profile]
    apply_style()
    with Experiment(f"p1_5_fvm2d_{profile}", {"seed": 0, "profile": profile, **prof}) as ex:
        res: dict = {
            "steady": steady_study(prof["N_space"], ex),
            "tvd": tvd_study(prof["N_tvd"], ex),
            "time": temporal_study(prof["N_time"], ex),
        }
        res["product"], fields = product_study(prof["N_prod"], ex)
        res["multigrid"], hist = mg_study(prof["N_mg"], ex)
        res["perf"] = perf_study(prof["N_perf"], ex)
        ex.metrics.update(res)
        ex.metrics["figures"] = figures(res, fields, hist, ex, prof, profile)
    for k, v in res["steady"].items():
        print(
            f"steady {k:10s} fit={v['order_fit_last3']:.3f} errs={[f'{e:.2e}' for e in v['err_inf']]} "
            f"bal={v['energy_balance_rel_max']:.1e} div={v['max_div_h_u']:.1e}"
        )
    for k, v in res["tvd"].items():
        print(f"tvd    {k:14s} fit={v['order_fit']:.3f} pair={np.round(v['pairwise'], 3)}")
    for k, v in res["time"].items():
        if isinstance(v, dict):
            print(f"time   {k:18s} fit={v['order_fit']:.3f} pair={np.round(v['pairwise'], 3)}")
    for k, v in res["product"].items():
        print(
            f"prod   {k} cn={[f'{e:.2e}' for e in v['err_cn']]} p={v['order_cn']:.3f} "
            f"adi={[f'{e:.2e}' for e in v['err_adi']]} p={v['order_adi']:.3f}"
        )
    for k, v in res["multigrid"].items():
        print(
            f"mg     {k:16s} its={v['mg_iters']} rho={np.round(v['mg_rho'], 3).tolist()} "
            f"mgcg={v['mgcg_iters']} jac={v['jacobi_cg_iters']}"
        )
    for k, v in res["perf"].items():
        print(f"perf   {k}", {a: float(f"{b:.3g}") for a, b in v.items() if isinstance(b, float)})


if __name__ == "__main__":
    main()
