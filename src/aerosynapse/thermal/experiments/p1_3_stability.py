"""Step 1.3: stability & accuracy analysis.

(a) von Neumann limits (symbolic + numeric) for every scheme x explicit integrator
(b) empirical stability limits from the actual solver vs von Neumann
(c) cell-Peclet study: central differencing wiggles for Pe_h > 2 (steady boundary layer, Pe=100)
(d) numerical diffusion: upwind pulse vs the modified-equation prediction nu_num = a h/2 (1 - C)
(e) Gibbs/dispersive oscillations for a top-hat; TVD schemes: no new extrema, TV non-increasing

Run:  python experiments/p1_3_stability.py --profile smoke|full
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from aerosynapse.thermal.numerics import fd1d
from aerosynapse.thermal.numerics import stability as st
from aerosynapse.thermal.numerics.time_integrators import integrate
from aerosynapse.thermal.physics import analytic as an
from aerosynapse.thermal.utils.experiment import Experiment, cli_profile
from aerosynapse.thermal.viz.style import DIV_CMAP, apply_style, save_fig

PROFILES = {
    "smoke": {"N_gibbs": 400, "N_pulse": 600, "bisect_iters": 18, "N_domain": [100, 400]},
    "full": {"N_gibbs": 1600, "N_pulse": 2400, "bisect_iters": 30, "N_domain": [100, 400, 1600]},
}
EXPL = ("euler", "ssprk3", "rk4")


def bisect(stable: Callable[[float], bool], lo: float, hi: float, iters: int) -> float:
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if stable(mid) else (lo, mid)
    return lo


def vn_limits(iters: int) -> dict:
    out: dict = {"max_C_r0": {}, "max_r_C0": {}}
    for sch in fd1d.LINEAR_SCHEMES:
        for m in EXPL:
            out["max_C_r0"][f"{sch}/{m}"] = st.max_stable_courant(sch, m)
    for m in EXPL:
        out["max_r_C0"][m] = bisect(
            lambda rv, m=m: st.max_amplification("central", m, 0.0, rv) <= 1 + 1e-10, 0.0, 5.0, 50
        )
    G = {
        f"{s}/{m}": str(st.abs2_simplified(st.amplification_factor(s, m).subs(st.r, 0)))
        for s, m in [("upwind1", "euler"), ("central", "euler")]
    }
    out["abs_G2_pure_advection"] = G
    out["abs_G2_ftcs_diffusion"] = str(st.abs2_simplified(st.amplification_factor("central", "euler").subs(st.C, 0)))
    out["modified_equation_space"] = {
        s: {k: str(v) for k, v in st.modified_equation_terms(s).items()} for s in fd1d.LINEAR_SCHEMES
    }
    out["numerical_diffusion_euler"] = {s: str(st.numerical_diffusion_euler(s)) for s in fd1d.LINEAR_SCHEMES}
    return out


def _grows(p: fd1d.ConvDiff1D, method: str, dt: float, nsteps: int) -> bool:
    """True if max_n ||y^n||_inf exceeds 1e3 ||y^0||_inf at ANY step (running max, not final state:
    with an outflow boundary a growing packet can be swept out, leaving a small final state)."""
    y0 = np.random.default_rng(0).standard_normal(p.idx.size)
    peak = [0.0]

    def cb(n: int, t: float, y: np.ndarray) -> None:
        v = np.max(np.abs(y))
        peak[0] = max(peak[0], v if np.isfinite(v) else np.inf)

    with np.errstate(all="ignore"):
        integrate(p, y0, 0.0, nsteps * dt, dt, method, callback=cb)
    return peak[0] > 1e3 * np.max(np.abs(y0))


def empirical_limits(iters: int, Ns: list[int]) -> dict:
    """Bisection on the solver itself (Dirichlet BCs, random IC, growth factor 1e3 = unstable).

    Repeated for several N: on a finite domain with an outflow boundary, convective instabilities of
    non-normal operators can leave the domain before amplifying, so the empirical limit is expected to
    approach the von Neumann limit only as the domain (in cells) grows.
    """
    out: dict = {}
    rv = 0.25
    for N in Ns:
        h = 1 / N
        ns = max(3000, 4 * N)
        out.setdefault("ftcs_diffusion_r", {"von_neumann": 0.5})[f"N{N}"] = bisect(
            lambda r_, N=N, h=h, ns=ns: not _grows(fd1d.ConvDiff1D(N, a=0, D=1.0), "euler", r_ * h * h, ns),
            0.3,
            0.7,
            iters,
        )
        for sch, m, hi in [
            ("upwind1", "euler", 2.5),
            ("central", "rk4", 4.0),
            ("central", "ssprk3", 3.0),
            ("quick", "rk4", 3.0),
            ("upwind2", "rk4", 3.0),
        ]:
            d = out.setdefault(f"advection_{sch}_{m}_C", {"von_neumann": st.max_stable_courant(sch, m)})
            d[f"N{N}"] = bisect(
                lambda Cv, sch=sch, m=m, N=N, h=h: (
                    not _grows(fd1d.ConvDiff1D(N, a=1.0, D=0.0, scheme=sch), m, Cv * h, 4 * N)
                ),
                0.05,
                hi,
                iters,
            )

        def stable_cd(Cv: float, N: int = N, h: float = h, ns: int = ns) -> bool:
            dt = Cv * h
            return not _grows(fd1d.ConvDiff1D(N, a=1.0, D=rv * h * h / dt, scheme="central"), "euler", dt, ns)

        out.setdefault("ftcs_convdiff_r0.25_C", {"von_neumann": float(np.sqrt(2 * rv))})[f"N{N}"] = bisect(
            stable_cd, 0.05, 1.5, iters
        )
    return out


def cell_peclet_study(ex: Experiment) -> tuple[dict, dict]:
    Pe = 100.0
    out, prof = {}, {}
    for N in [10, 20, 25, 40, 50, 100, 200]:
        for sch in fd1d.LINEAR_SCHEMES:
            p = fd1d.ConvDiff1D(N, a=1.0, D=1 / Pe, left=fd1d.dirichlet(0.0), right=fd1d.dirichlet(1.0), scheme=sch)
            th = p.nodal(p.steady(), 0.0)
            ex_ = an.steady_convdiff(p.x, Pe)
            rec = {
                "Pe_h": Pe / N,
                "min": float(th.min()),
                "max": float(th.max()),
                "n_osc": st.count_oscillations(th),
                "err_inf": float(np.max(np.abs(th - ex_))),
            }
            out[f"N{N}_{sch}"] = rec
            ex.log(study="cell_peclet", N=N, scheme=sch, **rec)
            prof[(N, sch)] = (p.x, th)
    return out, prof


def pulse_study(N: int) -> tuple[dict, dict]:
    """Pure advection (D=0) of a Gaussian with upwind1; compare to modified-equation prediction."""
    x0, x1, s0, T = 0.0, 3.0, 0.08, 1.5
    out, prof = {}, {}
    for m, Cv in [("euler", 0.2), ("euler", 0.5), ("euler", 0.9), ("rk4", 0.5)]:
        p = fd1d.ConvDiff1D(N, a=1.0, D=0.0, scheme="upwind1", x0=x0, x1=x1)
        h = p.h
        y = integrate(p, an.gaussian_pulse(p.x_active, 0.0, 1e300, s0, 0.5), 0.0, T, Cv * h, m)
        th = p.nodal(y, T)
        nu = h / 2 * (1 - Cv) if m == "euler" else h / 2
        pred = an.gaussian_pulse(p.x, T, 1 / nu, s0, 0.5)
        exact = an.gaussian_pulse(p.x, T, 1e300, s0, 0.5)
        key = f"upwind1_{m}_C{Cv}"
        out[key] = {
            "nu_num_predicted": nu,
            "peak_numeric": float(th.max()),
            "peak_modified_eq": float(pred.max()),
            "peak_exact": float(exact.max()),
            "maxdiff_vs_modified_eq": float(np.max(np.abs(th - pred))),
            "maxdiff_vs_exact": float(np.max(np.abs(th - exact))),
        }
        prof[key] = (p.x, th, pred, exact)
    return out, prof


def gibbs_study(N: int) -> tuple[dict, dict]:
    x0, x1, T, Cv = 0.0, 2.5, 1.0, 0.4
    runs = [
        ("upwind1", "ssprk3"),
        ("central", "rk4"),
        ("central", "crank_nicolson"),
        ("upwind2", "ssprk3"),
        ("quick", "ssprk3"),
        ("tvd_minmod", "ssprk3"),
        ("tvd_vanleer", "ssprk3"),
        ("tvd_superbee", "ssprk3"),
        ("tvd_vanalbada", "ssprk3"),
    ]
    out, prof = {}, {}
    for sch, m in runs:
        p = fd1d.ConvDiff1D(N, a=1.0, D=0.0, scheme=sch, x0=x0, x1=x1)
        ic = ((p.x_active > 0.25) & (p.x_active < 0.75)).astype(float)
        tv = [st.total_variation(p.nodal(ic, 0.0))]
        y = integrate(
            p,
            ic,
            0.0,
            T,
            Cv * p.h,
            m,
            callback=lambda n, t, yy, p=p, tv=tv: tv.append(st.total_variation(p.nodal(yy, t))),
        )
        th = p.nodal(y, T)
        exact = ((p.x > 1.25) & (p.x < 1.75)).astype(float)
        out[f"{sch}/{m}"] = {
            "overshoot": float(th.max() - 1),
            "undershoot": float(-th.min()),
            "TV_final_over_initial": tv[-1] / tv[0],
            "max_TV_increase": float(np.max(np.diff(tv))),
            "L1_error": float(np.sum(np.abs(th - exact)) * p.h),
        }
        prof[f"{sch}/{m}"] = (p.x, th, exact)
    return out, prof


def figures(ex: Experiment, cp_prof: dict, pulse_prof: dict, gibbs_prof: dict, tag: str) -> list[str]:
    paths = []
    # stability maps
    Cs, rs = np.linspace(0, 2.0, 161), np.linspace(0, 1.0, 101)
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for ax, (s, m) in zip(
        axs,
        [("central", "euler"), ("upwind1", "euler"), ("central", "rk4"), ("quick", "ssprk3")],
        strict=True,
    ):
        G = st.stability_map(s, m, Cs, rs)
        im = ax.contourf(Cs, rs, np.log10(G), levels=np.linspace(-1.5, 1.5, 25), cmap=DIV_CMAP, extend="both")
        ax.contour(Cs, rs, G, levels=[1 + 1e-10], colors="k", linewidths=1.2)
        ax.set(xlabel=r"$C = a\Delta t/h$", title=f"{s} + {m}")
    cl = Cs[Cs**2 / 2 <= 1.0]
    axs[0].plot(cl, cl**2 / 2, "w--", lw=1, label=r"$C^2=2r$")
    axs[0].set_ylim(0, 1)
    axs[0].legend(loc="lower right")
    axs[0].set_ylabel(r"$r = D\Delta t/h^2$")
    fig.colorbar(im, ax=axs, label=r"$\log_{10}\max_\xi|G|$ (black: $|G|=1$)")
    paths += save_fig(fig, f"p1_3_stability_maps_{tag}", ex.dir)

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    xf = np.linspace(0, 1, 1000)
    for N in [10, 25, 50]:
        x, th = cp_prof[(N, "central")]
        axs[0].plot(x, th, "o-", ms=3, label=f"central N={N} (Pe$_h$={100 / N:g})")
    x, th = cp_prof[(25, "upwind1")]
    axs[0].plot(x, th, "s--", ms=3, label="upwind1 N=25")
    axs[0].plot(xf, an.steady_convdiff(xf, 100.0), "k-", lw=1, label="exact")
    axs[0].set(xlabel="$x^*$", ylabel=r"$\theta$", title="Cell Péclet, Pe=100", xlim=(0.6, 1.0))
    axs[0].legend(fontsize=6)
    for key, (x, th, pred, _) in pulse_prof.items():
        if "euler" in key:
            axs[1].plot(x, th, label=key.replace("upwind1_", ""))
            axs[1].plot(x, pred, "k:", lw=0.8)
    exact = next(iter(pulse_prof.values()))[3]
    axs[1].plot(x, exact, "k-", lw=1, label="exact (no diffusion)")
    axs[1].set(
        xlabel="$x^*$",
        ylabel=r"$\theta$",
        xlim=(1.4, 2.6),
        title="Upwind numerical diffusion (dotted: modified eq.)",
    )
    axs[1].legend(fontsize=6)
    for key in ["upwind1/ssprk3", "central/rk4", "quick/ssprk3", "tvd_vanleer/ssprk3", "tvd_superbee/ssprk3"]:
        x, th, exact = gibbs_prof[key]
        axs[2].plot(x, th, label=key)
    axs[2].plot(x, exact, "k-", lw=1)
    axs[2].set(xlabel="$x^*$", ylabel=r"$\theta$", xlim=(1.0, 2.0), title="Top-hat, C=0.4, t*=1")
    axs[2].legend(fontsize=6)
    paths += save_fig(fig, f"p1_3_peclet_diffusion_gibbs_{tag}", ex.dir)
    return [str(p) for p in paths]


def main() -> None:
    profile = cli_profile()
    prof = PROFILES[profile]
    apply_style()
    with Experiment(f"p1_3_stability_{profile}", {"seed": 0, "profile": profile, **prof}) as ex:
        ex.metrics["von_neumann"] = vn_limits(prof["bisect_iters"])
        ex.metrics["empirical_vs_von_neumann"] = empirical_limits(prof["bisect_iters"], prof["N_domain"])
        ex.metrics["cell_peclet"], cp_prof = cell_peclet_study(ex)
        ex.metrics["numerical_diffusion"], pulse_prof = pulse_study(prof["N_pulse"])
        ex.metrics["gibbs"], gibbs_prof = gibbs_study(prof["N_gibbs"])
        ex.metrics["figures"] = figures(ex, cp_prof, pulse_prof, gibbs_prof, profile)
    m = ex.metrics
    print("max C (r=0):", {k: round(v, 4) for k, v in m["von_neumann"]["max_C_r0"].items()})
    print("max r (C=0):", {k: round(v, 4) for k, v in m["von_neumann"]["max_r_C0"].items()})
    for k, v in m["empirical_vs_von_neumann"].items():
        print(f"{k:34s}", {a: round(b, 4) for a, b in v.items()})
    for k, v in m["numerical_diffusion"].items():
        print(k, {a: round(b, 5) for a, b in v.items()})
    for k, v in m["gibbs"].items():
        print(f"{k:24s}", {a: float(f"{b:.3g}") for a, b in v.items()})
    for N in [10, 25, 50, 100]:
        print(
            N,
            {
                s: (m["cell_peclet"][f"N{N}_{s}"]["n_osc"], round(m["cell_peclet"][f"N{N}_{s}"]["min"], 4))
                for s in fd1d.LINEAR_SCHEMES
            },
        )
    _ = sp  # sympy used indirectly via stability module


if __name__ == "__main__":
    main()
