"""Step 1.2: 1D FD verification — spatial order of every convection scheme, temporal order of every
integrator, all BC types, Case B (Robin + source + fin) against the eigen-series solution.

Run:  python experiments/p1_2_fd1d.py --profile smoke|full
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from aerosynapse.thermal.numerics import fd1d
from aerosynapse.thermal.numerics.convergence import fit_order, observed_orders
from aerosynapse.thermal.numerics.time_integrators import EXPLICIT, FORMAL_ORDER, IMPLICIT, integrate
from aerosynapse.thermal.physics import analytic as an
from aerosynapse.thermal.physics import mms
from aerosynapse.thermal.utils.experiment import Experiment, cli_profile
from aerosynapse.thermal.viz.style import apply_style, save_fig

PROFILES = {
    "smoke": {"N_space": [16, 32, 64, 128, 256], "N_tvd": [20, 40, 80, 160], "N_caseB": [10, 20, 40, 80]},
    "full": {
        "N_space": [16, 32, 64, 128, 256, 512, 1024, 2048],
        "N_tvd": [20, 40, 80, 160, 320, 640],
        "N_caseB": [10, 20, 40, 80, 160, 320],
    },
}
FORMAL_SPACE = {"upwind1": 1, "central": 2, "upwind2": 2, "quick": 2}


def bc_from_mms(ms: mms.ManufacturedSolution, kind: str, side: str, t_const: bool = True) -> fd1d.BC:
    """Build a BC for side 'L'/'R' whose data is taken from the manufactured solution."""
    xb, nrm = (0.0, (-1,)) if side == "L" else (1.0, (1,))
    a, b = {"D": (1.0, 0.0), "N": (0.0, 1.0), "R": (2.0, 1.0)}[kind]
    g = ms.robin_data(nrm, a, b)
    if t_const:
        return fd1d.BC(a, b, float(g(xb, 0.0)))
    return fd1d.BC(a, b, lambda t, g=g, xb=xb: float(g(xb, t)))


def spatial_study(Ns: list[int], ex: Experiment) -> dict:
    """Steady MMS, direct sparse solve: isolates the spatial error."""
    x = mms.x
    th = sp.sin(sp.pi * x + sp.Rational(1, 3)) + sp.cos(2 * x) / 2
    out = {}
    cases = [("DR", 20.0), ("DN", 20.0), ("RD", 20.0), ("RR", 20.0), ("NR", 2.0), ("NR", 20.0)]
    for bcs, Pe in cases:
        ms = mms.manufacture(th, [1.0], 1 / sp.nsimplify(Pe), 0)
        Q = lambda X, t, ms=ms: ms.forcing(X, 0 * X)  # noqa: E731
        for sch in fd1d.LINEAR_SCHEMES:
            errs = []
            for N in Ns:
                p = fd1d.ConvDiff1D(
                    N,
                    a=1.0,
                    D=1 / Pe,
                    source=Q,
                    left=bc_from_mms(ms, bcs[0], "L"),
                    right=bc_from_mms(ms, bcs[1], "R"),
                    scheme=sch,
                )
                e = float(np.max(np.abs(p.nodal(p.steady(), 0.0) - ms.exact(p.x, 0 * p.x))))
                errs.append(e)
                ex.log(study="space_steady", bcs=bcs, Pe=Pe, scheme=sch, N=N, err_inf=e)
            h = 1 / np.array(Ns, float)
            out[f"{bcs}_Pe{Pe:g}_{sch}"] = {
                "err_inf": errs,
                "pairwise": observed_orders(h, errs).tolist(),
                "order_fit_last3": fit_order(h[-3:], errs[-3:])[0],
                "formal": FORMAL_SPACE[sch],
            }
    return out


def tvd_study(Ns: list[int], ex: Experiment) -> dict:
    """Transient MMS with SSP-RK3 and small dt (spatial error dominates) for every scheme."""
    Pe, T = 20.0, 0.5
    ms = mms.mms_1d_default(Pe=Pe, a=1.0)
    out = {}
    for sch in fd1d.SCHEMES:
        errs = []
        for N in Ns:
            p = fd1d.ConvDiff1D(
                N,
                a=1.0,
                D=1 / Pe,
                source=ms.forcing,
                scheme=sch,
                left=bc_from_mms(ms, "D", "L", False),
                right=bc_from_mms(ms, "R", "R", False),
            )
            h = p.h
            dt = min(0.1 * h, 0.2 * h * h * Pe)
            y = integrate(p, ms.exact(p.x_active, 0 * p.x_active), 0.0, T, dt, "ssprk3")
            e = float(np.max(np.abs(p.nodal(y, T) - ms.exact(p.x, T + 0 * p.x))))
            errs.append(e)
            ex.log(study="space_transient", scheme=sch, N=N, dt=dt, err_inf=e)
        h = 1 / np.array(Ns, float)
        out[sch] = {
            "err_inf": errs,
            "pairwise": observed_orders(h, errs).tolist(),
            "order_fit_last3": fit_order(h[-3:], errs[-3:])[0],
        }
    return out


def temporal_study(ex: Experiment) -> dict:
    """Fixed grid (N=40, central, D/R BCs, time-dependent data); reference = RK4 with dt=1e-4."""
    Pe, T, N = 20.0, 1.0, 40
    ms = mms.mms_1d_default(Pe=Pe, a=1.0)
    p = fd1d.ConvDiff1D(
        N,
        a=1.0,
        D=1 / Pe,
        source=ms.forcing,
        scheme="central",
        left=bc_from_mms(ms, "D", "L", False),
        right=bc_from_mms(ms, "R", "R", False),
    )
    y0 = ms.exact(p.x_active, 0 * p.x_active)
    ref = integrate(p, y0, 0.0, T, 1e-4, "rk4")
    dts = {
        "euler": [2e-3, 1e-3, 5e-4, 2.5e-4],
        "ssprk3": [6e-3, 3e-3, 1.5e-3, 7.5e-4],
        "rk4": [6e-3, 3e-3, 1.5e-3, 7.5e-4],
    }
    out = {}
    for m in EXPLICIT + IMPLICIT:
        dl = dts.get(m, [0.1, 0.05, 0.025, 0.0125, 0.00625])
        errs = []
        for dt in dl:
            e = float(np.max(np.abs(integrate(p, y0, 0.0, T, dt, m) - ref)))
            errs.append(e)
            ex.log(study="time", method=m, dt=dt, err_inf=e)
        out[m] = {
            "dt": dl,
            "err_inf": errs,
            "pairwise": observed_orders(dl, errs).tolist(),
            "order_fit": fit_order(dl, errs)[0],
            "formal": FORMAL_ORDER[m],
        }
    # Thomas vs SuperLU on the same tridiagonal CN system
    yt = integrate(p, y0, 0.0, T, 0.01, "crank_nicolson", solver="thomas")
    ys = integrate(p, y0, 0.0, T, 0.01, "crank_nicolson", solver="splu")
    out["thomas_vs_splu_maxdiff"] = float(np.max(np.abs(yt - ys)))
    return out


def caseB_study(Ns: list[int], ex: Experiment) -> dict:
    """Case B: theta_t = theta_xx - m^2 theta + Q, symmetry at 0, Robin at 1, vs eigen-series."""
    out = {}
    for Bi, Q, m in [(0.1, 1.0, 0.0), (1.0, 0.0, 0.0), (10.0, 1.0, 1.0), (100.0, 1.0, 2.0)]:
        T, errs = 0.2, []
        for N in Ns:
            p = fd1d.ConvDiff1D(
                N,
                a=0.0,
                D=1.0,
                s=m * m,
                source=lambda X, t, Q=Q: Q + 0 * X,
                left=fd1d.neumann(0.0),
                right=fd1d.robin(Bi),
                scheme="central",
            )
            # dt small enough that the CN temporal error is far below the spatial one
            y = integrate(p, np.ones(p.idx.size), 0.0, T, 0.1 / N**2, "crank_nicolson")
            e = float(np.max(np.abs(p.nodal(y, T) - an.slab_transient(p.x, T, Bi, Q, m))))
            errs.append(e)
            ex.log(study="caseB", Bi=Bi, Q=Q, m=m, N=N, err_inf=e)
        h = 1 / np.array(Ns, float)
        out[f"Bi{Bi:g}_Q{Q:g}_m{m:g}"] = {
            "err_inf": errs,
            "pairwise": observed_orders(h, errs).tolist(),
            "order_fit_last3": fit_order(h[-3:], errs[-3:])[0],
        }
    return out


def figures(res: dict, Ns: list[int], ex: Experiment) -> list[str]:
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    h = 1 / np.array(Ns, float)
    for sch in fd1d.LINEAR_SCHEMES:
        axs[0].loglog(h, res["space"][f"DR_Pe20_{sch}"]["err_inf"], "o-", label=sch)
    axs[0].loglog(h, 2 * h, "k:", lw=0.8, label="$O(h)$")
    axs[0].loglog(h, 2 * h**2, "k--", lw=0.8, label="$O(h^2)$")
    axs[0].set(xlabel="$h$", ylabel=r"$\|e\|_\infty$", title="Steady MMS, Pe=20, D/Robin")
    axs[0].legend()
    for m in EXPLICIT + IMPLICIT:
        r = res["time"][m]
        axs[1].loglog(r["dt"], r["err_inf"], "o-", label=f"{m} (p={r['order_fit']:.2f})")
    axs[1].set(xlabel=r"$\Delta t$", ylabel=r"$\|e\|_\infty$ vs RK4 ref", title="Temporal order, N=40")
    axs[1].legend(fontsize=6)
    ht = 1 / np.array(PROF["N_tvd"], float)
    for sch in fd1d.SCHEMES:
        axs[2].loglog(ht, res["tvd"][sch]["err_inf"], "o-", label=sch, ms=3)
    axs[2].set(xlabel="$h$", ylabel=r"$\|e\|_\infty$", title="Transient MMS, SSP-RK3")
    axs[2].legend(fontsize=6)
    name = f"p1_2_fd1d_convergence_{PROF['profile']}"
    return [str(p) for p in save_fig(fig, name, ex.dir)]


PROF: dict = {}


def main() -> None:
    profile = cli_profile()
    PROF.update(PROFILES[profile], profile=profile)
    cfg = {"seed": 0, **PROF}
    apply_style()
    with Experiment(f"p1_2_fd1d_{profile}", cfg) as ex:
        res = {
            "space": spatial_study(PROF["N_space"], ex),
            "tvd": tvd_study(PROF["N_tvd"], ex),
            "time": temporal_study(ex),
            "caseB": caseB_study(PROF["N_caseB"], ex),
        }
        ex.metrics.update(res)
        ex.metrics["figures"] = figures(res, PROF["N_space"], ex)
    for k, v in res["space"].items():
        print(f"space {k:28s} fit={v['order_fit_last3']:.3f} last={v['err_inf'][-1]:.2e}")
    for k, v in res["tvd"].items():
        print(f"tvd   {k:28s} fit={v['order_fit_last3']:.3f} last={v['err_inf'][-1]:.2e}")
    for k, v in res["time"].items():
        print(f"time  {k:28s} {v if isinstance(v, float) else round(v['order_fit'], 3)}")
    for k, v in res["caseB"].items():
        print(f"caseB {k:28s} fit={v['order_fit_last3']:.3f} last={v['err_inf'][-1]:.2e}")


if __name__ == "__main__":
    main()
