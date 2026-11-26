"""Study scenes: the project's verification, identifiability, validity and cost results as animations, plus one more
flow (Rayleigh-Benard convection). Every scene either runs the project's own models (finite differences, the Phase 5
forward model, the MAC solver) or replays a saved metrics file; no network is trained.

  grid_convergence   Phase 1 verification: refining the grid, the FD solution converges to the exact one at order 2
                     (central) and 1 (upwind); observed orders computed from the errors shown
  identifiability    Phase 5: least-squares misfit over (k, h) as the known heat source grows from 0; without a source
                     only Bi = hL/k is identifiable (a valley), with it the 95 % region closes around the truth
  extrapolation      Step 6.3 (saved): PINN ensemble error, spread and residual swept across the parameter range and out
                     of it (results/p6_3_extrapolation_default)
  break_even         Step 6.7 (saved): cumulative cost of PINN (training + queries) vs the solver against the number of
                     queries (results/p6_7_cost_default)
  rayleigh_benard    convection cells in a layer heated from below (MAC solver + energy, Boussinesq, free-fall scaling)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from aerosynapse.thermal.viz import style as S
from aerosynapse.thermal.viz.anim.core import Scene, _code_version, cached

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
# ============================================================================ verification: grid convergence
def compute_grid_convergence(p: dict) -> dict:
    from aerosynapse.thermal.numerics import fd1d
    from aerosynapse.thermal.physics import analytic as an

    Ns = np.unique(np.round(np.geomspace(p["N_min"], p["N_max"], p["levels"])).astype(int))
    x = np.linspace(0, 1, 1000)
    out = {"N": Ns, "x": x, "exact": an.steady_convdiff(x, p["Pe"])}
    for sch in p["schemes"]:
        err, sol = [], []
        for n in Ns:
            f = fd1d.ConvDiff1D(
                int(n), scheme=sch, a=1.0, D=1.0 / p["Pe"], left=fd1d.dirichlet(0.0), right=fd1d.dirichlet(1.0)
            )
            u = f.nodal(f.steady(), 0.0)
            err.append(float(np.max(np.abs(u - an.steady_convdiff(f.x, p["Pe"])))))
            sol.append(np.interp(x, f.x, u))  # drawn as the piecewise-linear nodal solution
        out[f"err_{sch}"] = np.array(err)
        out[f"sol_{sch}"] = np.stack(sol)
        # observed order from the finest levels: least-squares slope of log error against log h
        h = 1.0 / Ns.astype(float)
        k = max(3, len(Ns) // 2)
        out[f"order_{sch}"] = np.polyfit(np.log(h[-k:]), np.log(np.array(err)[-k:]), 1)[0]
    return out


def scene_grid_convergence(p: dict, cache: Path) -> Scene:
    d, _ = cached(cache, "grid_convergence", p, lambda: compute_grid_convergence(p))
    Ns, x, ex = d["N"], d["x"], d["exact"]
    hold = int(p["hold"])
    names = {"central": "central differences", "upwind1": "first-order upwind"}
    n_frames = len(Ns) * hold + 2 * hold

    def make():
        import matplotlib.pyplot as plt

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
        a1.plot(x, ex, color=S.REF, lw=1.8, label="exact")
        lines = [
            a1.plot([], [], "-", lw=1.2, color=S.PALETTE[k], label=names.get(s, s))[0]
            for k, s in enumerate(p["schemes"])
        ]
        a1.set(xlim=(0.5, 1), ylim=(-0.1, 1.1), xlabel=S.label("x"), ylabel=S.label("theta"))
        a1.legend(loc="upper left")
        t1 = a1.set_title("")
        h = 1.0 / Ns.astype(float)
        pts = [a2.loglog([], [], "o-", ms=4, color=S.PALETTE[k])[0] for k in range(len(p["schemes"]))]
        allerr = np.concatenate([d[f"err_{s}"] for s in p["schemes"]])
        a2.set(
            xlim=(h.min() / 1.5, h.max() * 1.5),
            ylim=(allerr.min() / 3, allerr.max() * 3),
            xlabel="grid spacing $h$",
            ylabel="max nodal error",
        )
        for order, ls in ((1, ":"), (2, "--")):  # reference slopes through the coarse end of the data
            h0, e0 = h[2], allerr.max() / 2
            a2.loglog(h, e0 * (h / h0) ** order, ls, color=S.MUTED, lw=0.9)
            a2.annotate(f"order {order}", (h[-1], e0 * (h[-1] / h0) ** order), color=S.MUTED, fontsize=8, va="top")
        a2.invert_xaxis()
        a2.set_title("error against grid spacing")
        res = a2.text(
            0.97,
            0.96,
            "",
            transform=a2.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color=S.INK,
            bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": S.AXIS, "lw": 0.8},
        )
        res.set_visible(False)
        fig.suptitle(f"Verification: grid refinement of steady convection-diffusion, Pe = {p['Pe']:g}")

        def update(i):
            j = min(i // hold, len(Ns) - 1)
            for k, s in enumerate(p["schemes"]):
                lines[k].set_data(x, d[f"sol_{s}"][j])
                pts[k].set_data(h[: j + 1], d[f"err_{s}"][: j + 1])
            t1.set_text(f"{Ns[j]} cells")
            done = i >= len(Ns) * hold  # final hold: the observed orders, fitted from the errors shown
            res.set_text(
                "observed order\n" + "\n".join(f"{names.get(s, s)}: {d[f'order_{s}']:.2f}" for s in p["schemes"])
            )
            res.set_visible(done)

        return fig, update

    return Scene(
        "grid_convergence",
        "Verification: the solver converges at its design order",
        "Before any network is trained the numerical solver is verified: refining the grid, the finite-difference "
        "solution approaches the exact one and the error falls as h^2 (central differences) and h (first-order "
        "upwind). The observed orders are fitted from the errors shown.",
        n_frames,
        p["fps"],
        make,
        "src/numerics/fd1d.py against the exact solution (src/physics/analytic.py)",
        category="physics",
        meta={k: v for k, v in p.items() if k not in ("frames", "fps")},
    )


# ============================================================================ inverse: identifiability
def compute_identifiability(p: dict) -> dict:
    from aerosynapse.thermal.inverse import body as B
    from aerosynapse.thermal.inverse import data as D
    from aerosynapse.thermal.inverse import spec as IS

    base = IS.load_spec()
    truth = {k: float(base["parameters"][k]) for k in B.PARAMS}
    q_max = float(base["body"]["q_vol"]) * p["q_factor"]
    qs = np.concatenate([[0.0], np.geomspace(q_max / 1e3, q_max, p["frames"] - 1)])
    lk = np.log(truth["k"]) + np.linspace(-np.log(p["span"]), np.log(p["span"]), p["grid"])
    lh = np.log(truth["h"]) + np.linspace(-np.log(p["span"]), np.log(p["span"]), p["grid"])
    chi = np.empty((len(qs), len(lh), len(lk)))
    for m, q in enumerate(qs):
        spec = copy.deepcopy(base)
        spec["body"]["q_vol"] = float(q)
        data = D.synthetic(spec, p["level"], 0)
        nt = IS.n_terms(spec)
        for i, h_ in enumerate(lh):
            for j, k_ in enumerate(lk):
                par = {**truth, "k": float(np.exp(k_)), "h": float(np.exp(h_))}
                r = (D.model_at(par, spec, data["r"], data["t"], nt) - data["T"]) / data["sigma"]
                chi[m, i, j] = float(r @ r)
    return {"q": qs, "lk": lk, "lh": lh, "chi2": chi, "k_true": truth["k"], "h_true": truth["h"], "q_base": q_max}


def _crlb_extent(qs: np.ndarray, level: float, chi2_95: float = 5.99) -> np.ndarray:
    """Extent max / min of k over the LINEARISED 95 % joint region of (ln k, ln h) at the truth, alpha known:
    exp(2 sqrt(5.99 Var(ln k))) with Var from the inverse Fisher matrix of the clean model (Gauss-Newton / Cramer-Rao,
    as in Phase 5). inf where the Fisher matrix is singular (no source: only Bi = hL/k is identifiable)."""
    from aerosynapse.thermal.inverse import body as B
    from aerosynapse.thermal.inverse import data as D
    from aerosynapse.thermal.inverse import spec as IS

    base = IS.load_spec()
    truth = {k: float(base["parameters"][k]) for k in B.PARAMS}
    out = []
    for q in qs:
        spec = copy.deepcopy(base)
        spec["body"]["q_vol"] = float(q)
        data = D.synthetic(spec, level, 0)
        nt = IS.n_terms(spec)
        cols = []
        for name in ("k", "h"):
            step = 1e-4
            up = {**truth, name: truth[name] * np.exp(step)}
            dn = {**truth, name: truth[name] * np.exp(-step)}
            f = D.model_at(up, spec, data["r"], data["t"], nt) - D.model_at(dn, spec, data["r"], data["t"], nt)
            cols.append(f / (2 * step) / data["sigma"])
        J = np.stack(cols, 1)
        F = J.T @ J
        s = np.linalg.svd(F, compute_uv=False)
        if s[-1] <= 1e-10 * s[0]:
            out.append(np.inf)
            continue
        var = np.linalg.inv(F)[0, 0]
        out.append(float(np.exp(2 * np.sqrt(chi2_95 * var))))
    return np.array(out)


def scene_identifiability(p: dict, cache: Path) -> Scene:
    d, _ = cached(cache, "identifiability", p, lambda: compute_identifiability(p))
    qs, K, H = d["q"], np.exp(d["lk"]), np.exp(d["lh"])
    dchi = d["chi2"] - d["chi2"].min(axis=(1, 2), keepdims=True)
    width = _crlb_extent(qs, p["level"])
    hold = int(p.get("hold", 10))
    n_frames = len(qs) + hold

    def make():
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
        lev = np.geomspace(1, max(10.0, float(dchi.max())), 14)
        state = {"cs": None, "c95": None}
        a1.set(
            xscale="log",
            yscale="log",
            xlabel="conductivity $k$ [W/(m K)]",
            ylabel="heat transfer coeff. $h$ [W/(m² K)]",
        )
        a1.plot(K, d["h_true"] / d["k_true"] * K, "--", color=S.MUTED, lw=0.9, label="constant Bi = hL/k")
        a1.plot([d["k_true"]], [d["h_true"]], "*", ms=13, color=S.PINN, mec="white", zorder=5, label="truth")
        a1.legend(loc="lower right", fontsize=8)
        a1.set(xlim=(K[0], K[-1]), ylim=(H[0], H[-1]))
        from matplotlib.ticker import NullFormatter

        for axis in (a1.xaxis, a1.yaxis):
            axis.set_minor_formatter(NullFormatter())
        a1.grid(False)
        t1 = a1.set_title("")
        (wl,) = a2.semilogx([], [], "-", color=S.SOLVER, lw=1.6)
        (wp,) = a2.semilogx([], [], "o", color=S.SOLVER, ms=5)
        ok = np.isfinite(width)
        a2.set(
            xlim=(qs[1] / 1.5, qs[-1] * 1.5),
            ylim=(1, float(width[ok].max()) * 1.3),
            yscale="log",
            xlabel="known heat source $q_{vol}$ [W/m³]",
            ylabel="k range in the 95 % region (max / min)",
        )
        a2.axvline(d["q_base"] / p["q_factor"], color=S.MUTED, lw=0.8, ls=":")
        a2.annotate(
            "default experiment",
            (d["q_base"] / p["q_factor"], float(width[ok].max())),
            fontsize=8,
            color=S.MUTED,
            rotation=90,
            va="top",
            ha="right",
        )
        t2 = a2.set_title("")
        fig.suptitle("Identifiability of k and h: least-squares misfit with 2 % noise (α known)")

        def update(i):
            m = min(i, len(qs) - 1)
            for key in ("cs", "c95"):
                if state[key] is not None:
                    state[key].remove()
            state["cs"] = a1.contourf(
                K, H, np.maximum(dchi[m], 1), levels=lev, norm=LogNorm(), cmap="viridis_r", alpha=0.9, zorder=1
            )
            state["c95"] = a1.contour(K, H, dchi[m], levels=[5.99], colors=[S.PINN], linewidths=1.6, zorder=3)
            q = qs[m]
            t1.set_text("no heat source: k, h confounded" if q == 0 else f"$q_{{vol}}$ = {q:.3g} W/m³")
            sel = ok & (np.arange(len(qs)) <= m) & (qs > 0)
            wl.set_data(qs[sel], width[sel])
            wp.set_data(qs[sel][-1:], width[sel][-1:])
            t2.set_text(
                f"linearised 95 % region: k within a factor {width[m]:.3g}"
                if np.isfinite(width[m])
                else "Fisher matrix singular: k and h not identifiable"
            )

        return fig, update

    return Scene(
        "identifiability",
        "Identifiability: why a heat source makes k and h recoverable",
        "Least-squares misfit (shaded, Δχ² on a log scale) of noisy slab temperatures over conductivity k and heat "
        "transfer coefficient h; the orange curve is the 95 % joint region. Without a heat source only the Biot number "
        "hL/k is identifiable: the region is a long valley along constant Bi. A known heat source closes it around "
        "the truth.",
        n_frames,
        p["fps"],
        make,
        "Phase 5 forward model and default experiment (src/inverse, configs/inverse/defaults.yaml); no network",
        category="learning",
        meta={k: v for k, v in p.items() if k not in ("frames", "fps")},
    )


# ============================================================================ validity: extrapolation (saved)
def scene_extrapolation(p: dict, cache: Path) -> Scene:
    met = json.loads((ROOT / "results/p6_3_extrapolation_default/metrics.json").read_text(encoding="utf-8"))
    cases = [(c, met["parametric"][c]) for c in p["cases"]]
    n = int(p["frames"])
    hold = int(p.get("hold", 15))

    def make():
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, len(cases), figsize=(5.0 * len(cases), 4.3), squeeze=False)
        arts = []
        for ax, (name, c) in zip(axs[0], cases, strict=True):
            v = c["values"]
            x = np.array(v["value"])
            lo, hi = c["range"]
            ax.axvspan(lo, hi, color=S.DATA, alpha=0.12, lw=0, label="training range")
            series = [
                (np.array(v["rel_l2"]), S.PINN, "-", "ensemble error (rel. L2)"),
                (np.array(v["rms_sd"]), S.SOLVER, "--", "ensemble spread"),
                (np.array(v["residual"]), S.MUTED, ":", "PDE residual"),
            ]
            ys = np.concatenate([s[0] for s in series] + [np.array(v["member_rel_l2_max"])])
            band = ax.fill_between(x[:0], x[:0], x[:0], color=S.PINN, alpha=0.15, lw=0)
            ls = [ax.plot([], [], st, color=col, lw=1.6, label=lab)[0] for _, col, st, lab in series]
            cur = ax.axvline(x[0], color=S.INK, lw=0.8)
            ax.set(
                xscale="log",
                yscale="log",
                xlim=(x.min(), x.max()),
                ylim=(ys[ys > 0].min() / 2, ys.max() * 2),
                xlabel=c["param_name"],
                ylabel="error / indicator",
            )
            title = ax.set_title("")
            arts.append((ax, x, v, series, ls, cur, title, lo, hi, [band], name))
        h, lab = axs[0][0].get_legend_handles_labels()
        fig.legend(h, lab, loc="outside lower center", ncol=4, fontsize=8.5)
        fig.suptitle("Where the surrogate stops being valid: sweeping the parameter out of the training range")

        def update(i):
            f = min(i / max(n - hold - 1, 1), 1.0)
            for ax, x, v, series, ls, cur, title, lo, hi, band, name in arts:
                xv = float(np.exp(np.log(x[0]) + f * (np.log(x[-1]) - np.log(x[0]))))
                sel = x <= xv * (1 + 1e-9)
                for (y, *_), ln in zip(series, ls, strict=True):
                    ln.set_data(x[sel], y[sel])
                band[0].remove()
                band[0] = ax.fill_between(
                    x[sel],
                    np.array(v["member_rel_l2_min"])[sel],
                    np.array(v["member_rel_l2_max"])[sel],
                    color=S.PINN,
                    alpha=0.15,
                    lw=0,
                )
                cur.set_xdata([xv, xv])
                j = int(np.argmin(np.abs(np.log(x) - np.log(xv))))
                where = "inside the training range" if lo <= x[j] <= hi else "EXTRAPOLATION"
                title.set_text(
                    f"{CASE_LABEL.get(name, name)}: {where}\n"
                    f"error {100 * v['rel_l2'][j]:.3g} %, spread {100 * v['rms_sd'][j]:.3g} %"
                )

        return fig, update

    return Scene(
        "extrapolation",
        "Validity limits: error and warning signals out of range",
        "A trained PINN ensemble queried across and beyond its training range (green). Inside, the error is small; "
        "outside it grows, and the ensemble spread and the PDE residual grow with it, so the model can flag its own "
        "extrapolation. Shaded: range of the individual members.",
        n,
        p["fps"],
        make,
        "results/p6_3_extrapolation_default (Step 6.3); saved, no training",
        category="learning",
        meta={k: v for k, v in p.items() if k not in ("frames", "fps")},
    )


# ============================================================================ cost: break-even (saved)
CASE_LABEL = {
    "A_steady": "steady convection-diffusion (1D)",
    "B_slab": "heated slab (1D)",
    "D_graetz": "Graetz channel (2D)",
    "C_cavity": "heated cavity (2D)",
}


def scene_break_even(p: dict, cache: Path) -> Scene:
    met = json.loads((ROOT / "results/p6_7_cost_default/metrics.json").read_text(encoding="utf-8"))
    an = met["analysis"]
    cases = [c for c in p["cases"] if c in an]
    nq = np.geomspace(1, p["max_queries"], 400)
    n = int(p["frames"])
    hold = int(p.get("hold", 20))

    def make():
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(10.0, 6.4))
        arts = []
        for ax, c in zip(axs.ravel(), cases, strict=False):
            a = an[c]
            solver = nq * a["solver_s"]
            pinn = a["offline_ensemble_s"] + nq * a["pinn_query_ensemble_s"]
            ax.set(
                xscale="log",
                yscale="log",
                xlim=(1, p["max_queries"]),
                ylim=(min(solver[0], pinn[0]) / 3, max(solver[-1], pinn[-1]) * 3),
            )
            (ls,) = ax.plot([], [], color=S.SOLVER, lw=1.8, label=f"solver ({a['solver_s']:.3g} s per query)")
            (lp,) = ax.plot(
                [],
                [],
                color=S.PINN,
                lw=1.8,
                label=(
                    f"PINN ensemble (train {a['offline_ensemble_s']:.0f} s"
                    f" + {a['pinn_query_ensemble_s']:.2g} s per query)"
                ),
            )
            (mk,) = ax.plot([], [], "o", ms=8, mfc="white", mec=S.INK, mew=1.5, zorder=5)
            ax.legend(loc="upper left", fontsize=7.5)
            ax.set_title(
                f"{CASE_LABEL.get(c, c)}: PINN error {100 * a['pinn_error']:.3g} %,"
                f" solver {100 * a['solver_err']:.2g} %",
                fontsize=9,
            )
            txt = ax.text(0.98, 0.04, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=S.INK)
            arts.append((a, solver, pinn, ls, lp, mk, txt))
        for ax in axs[1]:
            ax.set_xlabel("number of queries")
        for ax in axs[:, 0]:
            ax.set_ylabel("cumulative wall time [s]")
        sup = fig.suptitle("")

        def update(i):
            f = min(i / max(n - hold - 1, 1), 1.0)
            m = max(2, int(round(f * (len(nq) - 1))) + 1)
            for a, solver, pinn, ls, lp, mk, txt in arts:
                ls.set_data(nq[:m], solver[:m])
                lp.set_data(nq[:m], pinn[:m])
                be = a["break_even_ensemble"]
                valid = a["pinn_error"] <= p["valid_error"]
                mk.set_data([], [])
                txt.set_text("")
                if not valid:  # a cheaper but wrong answer is not a break-even
                    txt.set_text(
                        f"no break-even: PINN error {100 * a['pinn_error']:.3g} % > {100 * p['valid_error']:g} %"
                    )
                elif be is None or not np.isfinite(be):
                    txt.set_text("no break-even: the solver is cheaper per query")
                elif be <= nq[m - 1]:
                    mk.set_data([be], [be * a["solver_s"]])
                    txt.set_text(f"break-even after {be:,.0f} queries")
            sup.set_text(f"When does training pay off?   {nq[m - 1]:,.0f} queries")

        return fig, update

    return Scene(
        "break_even",
        "Cost-benefit: the break-even number of queries",
        "Cumulative wall time of answering N queries with the numerical solver against training a PINN ensemble once "
        "and then querying it. The PINN pays off only after the break-even number of queries, and only where it is "
        "accurate: its error is shown next to the solver's, and no break-even is claimed where the PINN error exceeds "
        f"{100 * p['valid_error']:g} % (the cavity PINN failed).",
        n,
        p["fps"],
        make,
        "results/p6_7_cost_default (Step 6.7, measured on this machine); saved, no training",
        category="learning",
        meta={k: v for k, v in p.items() if k not in ("frames", "fps")},
    )


# ============================================================================ flow: Rayleigh-Benard convection
def compute_rayleigh_benard(p: dict) -> dict:
    """Layer heated from below (theta = +1/2 bottom, -1/2 top, adiabatic sides, no-slip walls) from the conduction
    profile plus a small seeded perturbation: Boussinesq buoyancy Ri theta e_y with free-fall scaling (Ri = 1,
    Re = sqrt(Ra / Pr), Pe = Re Pr)."""
    from aerosynapse.thermal.numerics import fvm2d as fv
    from aerosynapse.thermal.numerics import ns_projection as ns
    from aerosynapse.thermal.viz.anim.physics import _run_ns

    A, ny = p["aspect"], p["ny"]
    nx = int(round(A * ny))
    re = float(np.sqrt(p["Ra"] / p["Pr"]))
    en = ns.EnergySpec(
        Pe=re * p["Pr"],
        scheme=p.get("energy_scheme", "tvd_vanleer"),
        bcs={
            "bottom": fv.dirichlet2(0.5),
            "top": fv.dirichlet2(-0.5),
            "left": fv.neumann2(0.0),
            "right": fv.neumann2(0.0),
        },
    )
    g = fv.Grid2D(nx, ny, A, 1.0)
    s = ns.MACNavierStokes2D(g, re, {}, energy=en, Ri=1.0)
    th0 = 0.5 - g.Yc + p["perturb"] * np.random.default_rng(5).standard_normal((ny, nx))
    s.set_state(np.zeros_like(s.u), np.zeros_like(s.v), theta=th0)
    if p["spinup"] > 0:
        s.run(p["spinup"])
    s.t = 0.0
    return _run_ns(s, p["T"], p["frames"], p["particles"], 6, A, 1.0)


def scene_rayleigh_benard(p: dict, cache: Path) -> Scene:
    from aerosynapse.thermal.viz.anim import physics

    # the run also depends on the shared solver driver in physics.py: its source is part of the cache key
    key = {**p, "_driver": _code_version(physics._run_ns)}
    d, _ = cached(cache, "rayleigh_benard", key, lambda: compute_rayleigh_benard(p))
    t, A = d["t"], p["aspect"]

    def make():
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 6.4))
        im1 = a1.imshow(
            d["theta"][0],
            origin="lower",
            extent=(0, A, 0, 1),
            cmap="inferno",
            vmin=-0.5,
            vmax=0.5,
            interpolation="bilinear",
        )
        S.colorbar(fig, im1, a1, S.label("theta"), shrink=0.9)
        sc = a1.scatter(d["px"][0], d["py"][0], s=2.5, c="white", alpha=0.8, lw=0)
        lim = float(np.percentile(np.abs(d["vort"]), 99)) or 1.0
        im2 = a2.imshow(
            d["vort"][0],
            origin="lower",
            extent=(0, A, 0, 1),
            cmap=S.DIV_CMAP,
            norm=TwoSlopeNorm(0, -lim, lim),
            interpolation="bilinear",
        )
        S.colorbar(fig, im2, a2, r"vorticity $\omega$", shrink=0.9)
        for ax, lab in ((a1, "temperature and tracers (hot bottom, cold top)"), (a2, "vorticity")):
            ax.set(aspect="equal", xlabel=S.label("x"), ylabel=S.label("y"), title=lab)
            ax.grid(False)
        sup = fig.suptitle("")

        def update(i):
            im1.set_data(d["theta"][i])
            sc.set_offsets(np.c_[d["px"][i], d["py"][i]])
            im2.set_data(d["vort"][i])
            sup.set_text(f"Rayleigh-Bénard convection, Ra = {p['Ra']:.3g}, Pr = {p['Pr']:g}   t = {t[i]:.1f}")

        return fig, update

    return Scene(
        "rayleigh_benard",
        "Rayleigh-Bénard convection",
        "A fluid layer heated from below: above the critical Rayleigh number the conduction state is unstable and "
        "rising hot plumes and sinking cold plumes organise into convection rolls. Started from the conduction profile "
        "with a small seeded perturbation. Illustration from the verified solver, not a benchmark comparison.",
        len(t),
        p["fps"],
        make,
        "MAC Navier-Stokes + energy solver with Boussinesq buoyancy (src/numerics/ns_projection.py); no network",
        meta={k: v for k, v in p.items() if k not in ("frames", "fps")},
    )


SCENES = {
    "grid_convergence": scene_grid_convergence,
    "identifiability": scene_identifiability,
    "extrapolation": scene_extrapolation,
    "break_even": scene_break_even,
    "rayleigh_benard": scene_rayleigh_benard,
}
