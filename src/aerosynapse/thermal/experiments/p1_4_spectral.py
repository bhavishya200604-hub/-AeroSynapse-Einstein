"""Step 1.4: Chebyshev spectral collocation (exponential convergence, Robin via row replacement) and an
independent SciPy method-of-lines (BDF/Radau) cross-check of the FD solver and time integrators.

Run:  python experiments/p1_4_spectral.py --profile smoke|full
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from aerosynapse.thermal.numerics import fd1d
from aerosynapse.thermal.numerics import spectral as spc
from aerosynapse.thermal.numerics.mol import solve_mol
from aerosynapse.thermal.numerics.time_integrators import integrate
from aerosynapse.thermal.physics import analytic as an
from aerosynapse.thermal.physics import mms
from aerosynapse.thermal.utils.experiment import Experiment, cli_profile
from aerosynapse.thermal.viz.style import apply_style, save_fig

PROFILES = {
    "smoke": {
        "N_diff": list(range(4, 65, 4)),
        "N_steady": [8, 12, 16, 24, 32, 48, 64, 96, 128],
        "N_pulse": [32, 64, 128, 256],
        "N_fd_caseB": 160,
    },
    "full": {
        "N_diff": list(range(4, 129, 4)),
        "N_steady": [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512],
        "N_pulse": [32, 64, 128, 256, 512],
        "N_fd_caseB": 640,
    },
}


def exp_rate(Ns: list[int], errs: list[float]) -> float | None:
    """Fit log e = c - kappa N on the pre-plateau branch: points with e < 1e-1 up to and including the
    first point below 1e-11 (later points sit on the round-off plateau). Returns kappa (None if < 2 pts)."""
    N, e = np.array(Ns, float), np.array(errs)
    below = np.flatnonzero(e < 1e-11)
    stop = below[0] if below.size else len(e) - 1
    m = (np.arange(len(e)) <= stop) & (e < 1e-1)
    if m.sum() < 2:
        return None
    return float(-np.polyfit(N[m], np.log(e[m]), 1)[0])


def diff_study(Ns: list[int]) -> dict:
    e1, e2 = [], []
    for N in Ns:
        D1, x = spc.cheb_mapped(N, -1.0, 1.0)
        f = np.exp(x) * np.sin(5 * x)
        df = np.exp(x) * (np.sin(5 * x) + 5 * np.cos(5 * x))
        d2f = np.exp(x) * (10 * np.cos(5 * x) - 24 * np.sin(5 * x))
        e1.append(float(np.max(np.abs(D1 @ f - df))))
        e2.append(float(np.max(np.abs(D1 @ (D1 @ f) - d2f))))
    return {"N": Ns, "err_D1": e1, "err_D2": e2, "rate_D1": exp_rate(Ns, e1), "rate_D2": exp_rate(Ns, e2)}


def steady_cases() -> dict:
    xs = mms.x
    th = sp.sin(sp.pi * xs + sp.Rational(1, 3)) + sp.cos(2 * xs) / 2
    ms = mms.manufacture(th, [1.0], sp.Rational(1, 20), 0)
    gL, gR = ms.robin_data((-1,), 2.0, 1.0), ms.robin_data((1,), 2.0, 1.0)
    cases = {}
    for Pe in (10.0, 100.0, 1000.0):
        cases[f"A_Pe{Pe:g}"] = (
            dict(a=1.0, D=1 / Pe, left=fd1d.dirichlet(0.0), right=fd1d.dirichlet(1.0)),
            lambda x, Pe=Pe: an.steady_convdiff(x, Pe),
        )
    cases["B_Bi3_Q1_m2"] = (
        dict(
            a=0.0,
            D=1.0,
            s=4.0,
            source=lambda x, t: 1.0 + 0 * x,
            left=fd1d.neumann(0.0),
            right=fd1d.robin(3.0),
        ),
        lambda x: an.slab_steady(x, 3.0, 1.0, 2.0),
    )
    cases["MMS_RobinRobin_Pe20"] = (
        dict(
            a=1.0,
            D=0.05,
            source=lambda x, t: ms.forcing(x, 0 * x),
            left=fd1d.BC(2.0, 1.0, float(gL(0.0, 0.0))),
            right=fd1d.BC(2.0, 1.0, float(gR(1.0, 0.0))),
        ),
        lambda x: ms.exact(x, 0 * x),
    )
    return cases


def steady_study(Ns: list[int], ex: Experiment) -> dict:
    out = {}
    xf = np.linspace(0, 1, 2001)
    for name, (kw, exact) in steady_cases().items():
        es, ef = [], []
        for N in Ns:
            p = spc.ChebConvDiff1D(N, **kw)
            th = p.nodal(p.steady(), 0.0)
            # error measured on a fine uniform grid via barycentric interpolation (not only at nodes)
            es.append(float(np.max(np.abs(spc.interpolate(p.x, th, xf) - exact(xf)))))
            q = fd1d.ConvDiff1D(N, scheme="central", **kw)
            ef.append(float(np.max(np.abs(q.nodal(q.steady(), 0.0) - exact(q.x)))))
            ex.log(study="steady", case=name, N=N, err_spectral=es[-1], err_fd_central=ef[-1])
        out[name] = {
            "N": Ns,
            "err_spectral": es,
            "err_fd_central": ef,
            "exp_rate_spectral": exp_rate(Ns, es),
            "N_to_reach_1e-8_spectral": next((n for n, e in zip(Ns, es, strict=True) if e < 1e-8), None),
            "N_to_reach_1e-8_fd": next((n for n, e in zip(Ns, ef, strict=True) if e < 1e-8), None),
        }
    return out


def caseB_transient(N_fd: int, ex: Experiment) -> dict:
    out = {}
    for Bi in (0.1, 1.0, 10.0, 100.0):
        Q, m = 1.0, 1.0
        kw = dict(
            a=0.0,
            D=1.0,
            s=m * m,
            source=lambda x, t, Q=Q: Q + 0 * x,
            left=fd1d.neumann(0.0),
            right=fd1d.robin(Bi),
        )
        ts = np.array([0.01, 0.1, 0.5])
        ps = spc.ChebConvDiff1D(24, **kw)
        t0 = time.perf_counter()
        sol = solve_mol(ps, np.ones(ps.idx.size), 0.0, ts[-1], "Radau", 1e-12, 1e-14, t_eval=ts)
        t_spec = time.perf_counter() - t0
        e_spec = [
            float(np.max(np.abs(ps.nodal(sol.y[:, k], t) - an.slab_transient(ps.x, t, Bi, Q, m))))
            for k, t in enumerate(ts)
        ]
        pf = fd1d.ConvDiff1D(N_fd, scheme="central", **kw)
        y0 = np.ones(pf.idx.size)
        rad = solve_mol(pf, y0, 0.0, 0.1, "Radau", 1e-11, 1e-13).y[:, -1]
        bdf = solve_mol(pf, y0, 0.0, 0.1, "BDF", 1e-9, 1e-12).y[:, -1]
        cn = integrate(pf, y0, 0.0, 0.1, 1e-4, "crank_nicolson")
        ser = an.slab_transient(pf.x, 0.1, Bi, Q, m)
        rec = {
            "cheb24_radau_err_at_t": dict(zip(map(str, ts), e_spec, strict=True)),
            "cheb24_radau_wall_s": t_spec,
            "fd_radau_vs_series_t0.1": float(np.max(np.abs(pf.nodal(rad, 0.1) - ser))),
            "fd_bdf_vs_series_t0.1": float(np.max(np.abs(pf.nodal(bdf, 0.1) - ser))),
            "fd_cn_dt1e-4_vs_series_t0.1": float(np.max(np.abs(pf.nodal(cn, 0.1) - ser))),
            "fd_cn_vs_fd_radau_t0.1": float(np.max(np.abs(cn - rad))),
            "N_fd": N_fd,
        }
        out[f"Bi{Bi:g}"] = rec
        ex.log(study="caseB_transient", Bi=Bi, **{k: v for k, v in rec.items() if isinstance(v, float)})
    return out


def pulse_study(Ns: list[int], ex: Experiment) -> dict:
    """Case A transient Gaussian, Pe = 100, x in [0, 2], exact time-dependent Dirichlet data, T = 1."""
    Pe, s0, xc, T = 100.0, 0.05, 0.25, 1.0
    g = lambda xb: lambda t: float(an.gaussian_pulse(xb, t, Pe, s0, xc))  # noqa: E731
    kw = dict(a=1.0, D=1 / Pe, left=fd1d.BC(1, 0, g(0.0)), right=fd1d.BC(1, 0, g(2.0)), x0=0.0, x1=2.0)
    xf = np.linspace(0, 2, 4001)
    es, ef = [], []
    for N in Ns:
        p = spc.ChebConvDiff1D(N, **kw)
        y = solve_mol(p, an.gaussian_pulse(p.x_active, 0.0, Pe, s0, xc), 0.0, T, "Radau", 1e-11, 1e-13).y[:, -1]
        es.append(float(np.max(np.abs(spc.interpolate(p.x, p.nodal(y, T), xf) - an.gaussian_pulse(xf, T, Pe, s0, xc)))))
        q = fd1d.ConvDiff1D(N, scheme="central", **kw)
        yq = solve_mol(q, an.gaussian_pulse(q.x_active, 0.0, Pe, s0, xc), 0.0, T, "Radau", 1e-11, 1e-13).y[:, -1]
        ef.append(float(np.max(np.abs(q.nodal(yq, T) - an.gaussian_pulse(q.x, T, Pe, s0, xc)))))
        ex.log(study="pulse", N=N, err_spectral=es[-1], err_fd_central=ef[-1])
    return {"N": Ns, "err_spectral": es, "err_fd_central": ef, "Pe": Pe}


def figures(res: dict, ex: Experiment, tag: str) -> list[str]:
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    d = res["diff"]
    axs[0].semilogy(d["N"], d["err_D1"], "o-", label=r"$D_N f$")
    axs[0].semilogy(d["N"], d["err_D2"], "s-", label=r"$D_N^2 f$")
    axs[0].set(xlabel="$N$", ylabel="max error", title=r"Chebyshev derivative of $e^x\sin 5x$")
    axs[0].legend()
    for name, r in res["steady"].items():
        (ln,) = axs[1].semilogy(r["N"], r["err_spectral"], "o-", ms=3, label=name)
        axs[1].semilogy(r["N"], r["err_fd_central"], "--", color=ln.get_color(), lw=0.8)
    axs[1].set(xlabel="$N$", ylabel=r"$\|e\|_\infty$", title="Steady BVPs (solid: Chebyshev, dashed: FD)")
    axs[1].legend(fontsize=6)
    pr = res["pulse"]
    axs[2].loglog(pr["N"], pr["err_spectral"], "o-", label="Chebyshev + Radau")
    axs[2].loglog(pr["N"], pr["err_fd_central"], "s-", label="FD central + Radau")
    axs[2].set(xlabel="$N$", ylabel=r"$\|e\|_\infty$", title="Gaussian pulse, Pe=100, $t^*$=1")
    axs[2].legend()
    return [str(p) for p in save_fig(fig, f"p1_4_spectral_{tag}", ex.dir)]


def main() -> None:
    profile = cli_profile()
    prof = PROFILES[profile]
    apply_style()
    with Experiment(f"p1_4_spectral_{profile}", {"seed": 0, "profile": profile, **prof}) as ex:
        res = {
            "diff": diff_study(prof["N_diff"]),
            "steady": steady_study(prof["N_steady"], ex),
            "caseB_transient": caseB_transient(prof["N_fd_caseB"], ex),
            "pulse": pulse_study(prof["N_pulse"], ex),
        }
        ex.metrics.update(res)
        ex.metrics["figures"] = figures(res, ex, profile)
    print("diff rates", res["diff"]["rate_D1"], res["diff"]["rate_D2"], min(res["diff"]["err_D1"]))
    for k, v in res["steady"].items():
        print(
            f"{k:22s} rate={v['exp_rate_spectral']} N1e-8 spec={v['N_to_reach_1e-8_spectral']} "
            f"fd={v['N_to_reach_1e-8_fd']} spec_err={[f'{e:.1e}' for e in v['err_spectral']]}"
        )
    for k, v in res["caseB_transient"].items():
        print(k, {a: (b if not isinstance(b, float) else float(f"{b:.3g}")) for a, b in v.items()})
    print("pulse spec", [f"{e:.2e}" for e in res["pulse"]["err_spectral"]])
    print("pulse fd  ", [f"{e:.2e}" for e in res["pulse"]["err_fd_central"]])


if __name__ == "__main__":
    main()
