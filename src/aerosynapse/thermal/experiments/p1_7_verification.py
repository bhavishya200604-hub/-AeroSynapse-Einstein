"""Step 1.7: verification summary and Phase-1 gate table.

(1) Steady NS + energy + Boussinesq MMS solved exactly (Newton, no time error) on N = 16..128 (256 full):
    asymptotic spatial orders of u, v, p, theta; divergence; GCI of a functional (mean wall heat flux).
(2) Plane Poiseuille with inflow/outflow (Newton), N up to 128: u error and dp/dx orders.
(3) Gate table: every scheme/integrator order from the Step 1.2/1.5/1.6 metrics files (+ (1),(2)) against the
    formal order with tolerance +-0.15 (finest-pair observed order), divergence and energy-balance residuals,
    and the benchmark deviations / GCI from p1_7_benchmarks. Writes docs/tables/phase1_gate.md.

Run:  python experiments/p1_7_verification.py --profile smoke|full   (after p1_7_benchmarks.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sympy as sp

from aerosynapse.thermal.numerics import fvm2d as fv
from aerosynapse.thermal.numerics import ns_projection as ns
from aerosynapse.thermal.numerics.convergence import gci, observed_orders
from aerosynapse.thermal.numerics.ns_steady import SteadyNewtonNS
from aerosynapse.thermal.physics import mms
from aerosynapse.thermal.utils.experiment import RESULTS_DIR, ROOT, Experiment, cli_profile

PROFILES = {
    "smoke": {"N_mms": [16, 32, 64, 128], "N_pois": [16, 32, 64, 128]},
    "full": {"N_mms": [16, 32, 64, 128, 256], "N_pois": [16, 32, 64, 128, 256]},
}
TOL = 0.15


def steady_mms() -> mms.ManufacturedNS:
    x, y = mms.x, mms.y
    psi = sp.sin(sp.pi * x) * sp.sin(sp.pi * y) / sp.pi + 2 * x**2 * (1 - x) * y * (1 - y) ** 2
    p = sp.cos(sp.pi * x) * sp.sin(sp.pi * y / 2 + sp.Rational(1, 3)) + x * y / 3
    th = sp.cos(sp.pi * x / 2 + sp.Rational(1, 5)) * sp.sin(sp.pi * y + sp.Rational(1, 7)) + x * y**2 / 2
    return mms.manufacture_ns(psi, p, 50.0, theta=th, Ri=0.5, Pe=50.0 * 0.71)


def mms_study(Ns: list[int], ex: Experiment) -> dict:
    ms = steady_mms()
    uf, vf = ms.u_fn, ms.v_fn
    bcs = {
        "left": ns.wall(u=lambda s, t: uf(0 * s, s, t), v=lambda s, t: vf(0 * s, s, t)),
        "right": ns.wall(u=lambda s, t: uf(1 + 0 * s, s, t), v=lambda s, t: vf(1 + 0 * s, s, t)),
        "bottom": ns.wall(u=lambda s, t: uf(s, 0 * s, t), v=lambda s, t: vf(s, 0 * s, t)),
        "top": ns.wall(u=lambda s, t: uf(s, 1 + 0 * s, t), v=lambda s, t: vf(s, 1 + 0 * s, t)),
    }
    e = {"u": [], "v": [], "p": [], "theta": []}
    qwall, divs, its = [], [], []
    for n in Ns:
        g = fv.Grid2D(n, n)
        en = ns.EnergySpec(
            Pe=float(ms.Pe),
            source=ms.Q_fn,
            bcs=fv.bcs_from_mms(ms.theta_ms, {"left": (1, 0), "right": (2, 1), "bottom": (0, 1), "top": (1, 0)}),
            bc_order=2,  # second-order wall closure (D30) so the wall-gradient functional is O(h^2)
        )
        s = ns.MACNavierStokes2D(g, float(ms.Re), bcs, force=(ms.fx_fn, ms.fy_fn), energy=en, Ri=float(ms.Ri))
        N = SteadyNewtonNS(s)
        z = 0 * g.Xc
        info = N.solve(
            N.pack(uf(s.Xu, s.Yu, 0 * s.Xu), vf(s.Xv, s.Yv, 0 * s.Xv), None, ms.theta_fn(g.Xc, g.Yc, z)),
            tol=1e-10,
        )
        p = N.store()
        e["u"].append(float(np.abs(s.u - uf(s.Xu, s.Yu, 0 * s.Xu)).max()))
        e["v"].append(float(np.abs(s.v - vf(s.Xv, s.Yv, 0 * s.Xv)).max()))
        pe = ms.p_fn(g.Xc, g.Yc, z)
        e["p"].append(float(np.abs((p - p.mean()) - (pe - pe.mean())).max()))
        e["theta"].append(float(np.abs(s.theta - ms.theta_fn(g.Xc, g.Yc, z)).max()))
        qwall.append(float(np.mean(s.energy.wall_gradient(s.theta.ravel(), 0.0, "left"))))
        divs.append(s.max_div())
        its.append(info["iterations"])
        ex.log(
            study="steady_mms",
            N=n,
            **{f"err_{k}": v[-1] for k, v in e.items()},
            max_div=divs[-1],
            newton_its=its[-1],
        )
    h = 1 / np.array(Ns, float)
    out = {"N": Ns, "newton_iterations": its, "max_div": divs}
    for k, v in e.items():
        out[k] = {"err": v, "pairwise": observed_orders(h, v).tolist()}
    exact_q = float(sp.N(sp.integrate(sp.diff(ms.theta, mms.x).subs(mms.x, 0), (mms.y, 0, 1))))
    out["wall_gradient_left_mean"] = {
        "values": qwall,
        "exact": exact_q,
        "gci": gci(qwall[-1], qwall[-2], qwall[-3], h[-1], h[-2], h[-3]),
        "err": [abs(q - exact_q) for q in qwall],
    }
    out["wall_gradient_left_mean"]["pairwise"] = observed_orders(h, out["wall_gradient_left_mean"]["err"]).tolist()
    return out


def poiseuille_study(Ns: list[int], ex: Experiment) -> dict:
    Re = 20.0
    eu, edp = [], []
    for n in Ns:
        s = ns.MACNavierStokes2D(
            fv.Grid2D(2 * n, n, 2.0, 1.0),
            Re,
            {"left": ns.wall(u=lambda s_, t: 4 * s_ * (1 - s_)), "right": ns.outflow()},
        )
        N = SteadyNewtonNS(s)
        N.solve(N.pack(s.u, s.v, None, None), tol=1e-11)
        p = N.store()
        eu.append(float(np.abs(s.u - 4 * s.Yu * (1 - s.Yu)).max()))
        edp.append(abs(float(np.mean(np.diff(p[:, n // 2 : 3 * n // 2], axis=1)) / s.g.dx) + 8 / Re))
        ex.log(study="poiseuille_newton", N=n, err_u=eu[-1], err_dpdx=edp[-1])
    h = 1 / np.array(Ns, float)
    return {
        "N": Ns,
        "u": {"err": eu, "pairwise": observed_orders(h, eu).tolist()},
        "dpdx": {"err": edp, "pairwise": observed_orders(h, edp).tolist()},
    }


def closure_study(Ns: list[int], ex: Experiment) -> dict:
    """FV energy MMS (cellular flow, D/R/N/R and all-Dirichlet BCs): theta and mean wall-gradient orders for
    both wall closures. Expected: theta O(h^2) for both; wall gradient O(h) for bc_order=1, O(h^2) for 2."""
    x, y, t = mms.x, mms.y, mms.t
    psi = sp.sin(sp.pi * x) * sp.sin(sp.pi * y) / sp.pi
    u, v = mms.velocity_from_streamfunction(psi)
    th = sp.cos(sp.pi * x / 2 + sp.Rational(1, 5)) * sp.sin(sp.pi * y + sp.Rational(1, 7)) + x * y**2 / 2
    ms = mms.manufacture(th, [u, v], sp.Rational(1, 50), 0)
    psif = sp.lambdify((x, y, t), psi, "numpy")
    exact_q = float(sp.N(sp.integrate(sp.diff(th, x).subs(x, 0), (y, 0, 1))))
    out: dict = {}
    for order in (1, 2):
        for name, spec in (
            ("DRNR", {"left": (1, 0), "right": (2, 1), "bottom": (0, 1), "top": (3, 1)}),
            ("allD", {k: (1, 0) for k in fv.SIDES}),
        ):
            et, eq = [], []
            for n in Ns:
                g = fv.Grid2D(n, n)
                p = fv.FVEnergy2D(
                    g,
                    1 / 50,
                    fv.FaceVelocity.from_streamfunction(g, lambda X, Y, T: psif(X, Y, T)),
                    source=lambda X, Y, T: ms.forcing(X, Y, 0 * X),
                    bcs=fv.bcs_from_mms(ms, spec),
                    bc_order=order,
                )
                yv = p.steady()
                et.append(float(np.abs(yv.reshape(n, n) - ms.exact(g.Xc, g.Yc, 0 * g.Xc)).max()))
                eq.append(abs(float(np.mean(p.wall_gradient(yv, 0.0, "left"))) - exact_q))
                ex.log(study="closure", bc_order=order, bcs=name, N=n, err_theta=et[-1], err_wallgrad=eq[-1])
            h = 1 / np.array(Ns, float)
            out[f"bc{order}_{name}"] = {
                "theta": {"err": et, "pairwise": observed_orders(h, et).tolist()},
                "wall_grad": {"err": eq, "pairwise": observed_orders(h, eq).tolist()},
            }
    return out


def load(exp: str) -> tuple[dict | None, str]:
    for prof in ("full", "smoke"):
        f = RESULTS_DIR / f"{exp}_{prof}" / "metrics.json"
        if f.exists():
            return json.loads(f.read_text()), f"{exp}_{prof}"
    return None, exp


def gate_rows(ver: dict) -> list[dict]:
    rows = []

    def add(src: str, item: str, formal: float, pairs: list[float], fit: float | None = None, note: str = "") -> None:
        obs = float(pairs[-1])
        rows.append(
            {
                "source": src,
                "item": item,
                "formal": formal,
                "observed_finest_pair": obs,
                "fit": fit,
                "pass": abs(obs - formal) <= TOL,
                "note": note,
            }
        )

    m, src = load("p1_2_fd1d")
    if m:
        for key, v in m["space"].items():
            if key.startswith("NR_Pe20"):
                continue  # ill-conditioned configuration (D12), reported separately
            add(src, f"1D steady {key}", v["formal"], v["pairwise"], v["order_fit_last3"])
        for sch in ("upwind1", "central", "upwind2", "quick"):
            v = m["tvd"][sch]
            add(
                src,
                f"1D transient MMS {sch}",
                1 if sch == "upwind1" else 2,
                v["pairwise"],
                v["order_fit_last3"],
            )
        for k, v in m["time"].items():
            if isinstance(v, dict):
                add(src, f"1D time {k}", v["formal"], v["pairwise"], v["order_fit"])
        for k, v in m["caseB"].items():
            add(src, f"1D Case B vs series {k}", 2, v["pairwise"], v["order_fit_last3"])
    m, src = load("p1_5_fvm2d")
    if m:
        for sch, f in (("upwind1", 1), ("central", 2), ("quick", 2)):
            v = m["steady"][sch]
            add(src, f"2D FV steady MMS {sch}", f, v["pairwise"], v["order_fit_last3"])
        for k, f in (("implicit_euler", 1), ("crank_nicolson", 2), ("bdf2", 2), ("adi_douglas_gunn", 2)):
            v = m["time"][k]
            add(src, f"2D FV time {k}", f, v["pairwise"], v["order_fit"])
        for bi, v in m["product"].items():
            add(
                src,
                f"2D FV product solution {bi} (CN)",
                2,
                observed_orders(1 / np.array(v["N"]), v["err_cn"]),
            )
            add(
                src,
                f"2D FV product solution {bi} (ADI)",
                2,
                observed_orders(1 / np.array(v["N"]), v["err_adi"]),
            )
    m, src = load("p1_6_ns")
    if m:
        add(src, "NS Taylor-Green u (transient)", 2, m["tg"]["u"]["pairwise"])
        add(src, "NS Taylor-Green p", 2, m["tg"]["p"]["pairwise"])
        for k in ("u", "v", "p", "theta"):
            add(
                src,
                f"NS+energy transient MMS {k}",
                2,
                m["mms"][k]["pairwise"],
                note="pre-asymptotic at N<=64; see steady MMS" if k in ("u", "v") else "",
            )
        for rk in ("euler", "ssprk2", "ssprk3"):
            v = m["time"][rk]
            add(src, f"NS time {rk}", v["formal"], v["pairwise"], v["order_fit"])
    src = "p1_7_verification"
    for k in ("u", "v", "p", "theta"):
        add(src, f"NS+energy STEADY MMS (Newton) {k}", 2, ver["mms"][k]["pairwise"])
    add(
        src,
        "NS steady MMS mean wall gradient (2nd-order one-sided)",
        2,
        ver["mms"]["wall_gradient_left_mean"]["pairwise"],
    )
    for key, v in ver["closure"].items():
        order = int(key[2])
        add(src, f"2D FV {key} theta", 2, v["theta"]["pairwise"])
        add(
            src,
            f"2D FV {key} mean wall gradient",
            order,
            v["wall_grad"]["pairwise"],
            note="formal O(h) with half-cell closure (Step 1.7 finding)" if order == 1 else "second-order closure",
        )
    add(src, "Poiseuille u (Newton)", 2, ver["poiseuille"]["u"]["pairwise"])
    add(src, "Poiseuille dp/dx (Newton)", 2, ver["poiseuille"]["dpdx"]["pairwise"])
    return rows


def write_table(rows: list[dict], extra: dict, path: Path) -> None:
    lines = [
        "# Phase 1 gate table (auto-generated by experiments/p1_7_verification.py — do not edit)",
        "",
        f"Criterion: |observed order (finest grid pair) − formal| ≤ {TOL}. Sources: results/<source>/metrics.json.",
        "",
        "| # | Item | Formal | Observed (finest pair) | LSQ fit | Status | Source | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        fit = f"{r['fit']:.3f}" if r["fit"] is not None else "—"
        lines.append(
            f"| {i} | {r['item']} | {r['formal']:g} | {r['observed_finest_pair']:.3f} | {fit} | "
            f"{'PASS' if r['pass'] else '**FAIL**'} | {r['source']} | {r['note']} |"
        )
    npass = sum(r["pass"] for r in rows)
    lines += ["", f"**{npass}/{len(rows)} order checks pass.**", "", "## Conservation", ""]
    for k, v in extra["conservation"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Benchmarks (deviation from VERIFIED / VERIFIED-SECONDARY / DERIVED references)", ""]
    for k, v in extra["benchmarks"].items():
        lines.append(f"- {k}: {v}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark_summary() -> dict:
    m, src = load("p1_7_benchmarks")
    out: dict = {}
    if not m:
        return {"p1_7_benchmarks": "not run"}
    for key, r in m["cavity"].items():
        if key.startswith("Re"):
            g = r["psi_min"]["gci"]
            out[f"Cavity {key} vs Ghia (finest grid) max|Δu|, max|Δv|"] = (
                f"{r['ghia_finest']['u']['max_abs_dev']:.4f}, {r['ghia_finest']['v']['max_abs_dev']:.4f}; "
                f"Richardson {r['ghia_richardson']['u']['max_abs_dev']:.4f}, {r['ghia_richardson']['v']['max_abs_dev']:.4f}; "
                f"ψ_min = {r['psi_min']['values'][-1]:.6f} (p = {g.get('p', float('nan')):.2f}, "
                f"GCI_fine = {100 * g.get('GCI_fine21', float('nan')):.3f}%, {g.get('type')}) [{src}]"
            )
    for key, r in m["schafer_turek_2d1"].items():
        if key.startswith("k"):
            out[f"Schäfer–Turek 2D-1 {r['cells_per_D']} cells/D (staircase)"] = (
                f"C_D {r['C_D']:.4f} ({100 * r['rel_err_C_D']:+.2f}%), C_L {r['C_L']:.5f} ({100 * r['rel_err_C_L']:+.1f}%), "
                f"Δp {r['dp']:.5f} ({100 * r['rel_err_dp']:+.2f}%) [{src}]"
            )
    r = m["schafer_turek_2d2"]
    if "St" in r:
        out[f"Schäfer–Turek 2D-2 {10 * r['k']} cells/D (staircase)"] = (
            f"St {r['St']:.4f} ({100 * r['rel_err_St']:+.2f}%), C_D,max {r['C_D_max']:.4f} ({100 * r['rel_err_C_D_max']:+.2f}%), "
            f"C_L,max {r['C_L_max']:.4f} ({100 * r['rel_err_C_L_max']:+.2f}%) [{src}]"
        )
    for kind in ("cwt", "chf"):
        r = m["graetz"][kind]
        out[f"Graetz {kind.upper()} fully developed Nu (Pe_H = {m['graetz']['Pe_H']:g})"] = (
            f"{r['Nu'][-1]:.5f} vs derived {r['reference']:.5f} ({100 * r['rel_err_finest']:+.3f}%), "
            f"p = {r['gci'].get('p', float('nan')):.2f}, GCI = {100 * r['gci'].get('GCI_fine21', float('nan')):.4f}% [{src}]"
        )
    for key, r in m["de_vahl_davis"].items():
        if key.startswith("Ra") and "_N" not in key:
            out[f"de Vahl Davis {key} mean Nu"] = (
                f"{r['Nu'][-1]:.4f}: {100 * r['rel_err_finest_vs_dVD']:+.2f}% vs dVD, "
                f"{100 * r['rel_err_finest_vs_Hortmann']:+.2f}% vs Hortmann; p = {r['gci'].get('p', float('nan')):.2f}, "
                f"GCI = {100 * r['gci'].get('GCI_fine21', float('nan')):.3f}% [{src}]"
            )
    return out


def conservation_summary(ver: dict) -> dict:
    out = {"steady NS MMS max|∇_h·u|": f"{max(ver['mms']['max_div']):.1e} (p1_7_verification)"}
    m, src = load("p1_6_ns")
    if m:
        out["TG max|∇_h·u| over run"] = f"{max(m['tg']['max_div_over_run']):.1e} ({src})"
    m, src = load("p1_5_fvm2d")
    if m:
        out["2D FV steady energy balance (relative)"] = (
            f"{max(v['energy_balance_rel_max'] for v in m['steady'].values()):.1e} ({src})"
        )
        out["2D FV CN transient energy identity"] = f"{m['perf']['cn_energy_identity']['abs_residual']:.1e} ({src})"
    m, src = load("p1_7_benchmarks")
    if m:
        hb = [abs(v["heat_balance_rel"]) for k, v in m["de_vahl_davis"].items() if "_N" in k and isinstance(v, dict)]
        dv = [v["max_div"] for k, v in m["de_vahl_davis"].items() if "_N" in k and isinstance(v, dict)]
        out["de Vahl Davis hot/cold heat balance (relative, Newton steady)"] = f"{max(hb):.1e} ({src})"
        out["de Vahl Davis max|∇_h·u|"] = f"{max(dv):.1e} ({src})"
    return out


def main() -> None:
    profile = cli_profile()
    prof = PROFILES[profile]
    with Experiment(f"p1_7_verification_{profile}", {"seed": 0, "profile": profile, **prof}) as ex:
        ver = {
            "mms": mms_study(prof["N_mms"], ex),
            "poiseuille": poiseuille_study(prof["N_pois"], ex),
            "closure": closure_study(prof["N_mms"], ex),
        }
        rows = gate_rows(ver)
        extra = {"conservation": conservation_summary(ver), "benchmarks": benchmark_summary()}
        ex.metrics.update(ver)
        ex.metrics["gate_rows"] = rows
        ex.metrics["gate_pass"] = int(sum(r["pass"] for r in rows))
        ex.metrics["gate_total"] = len(rows)
        ex.metrics.update(extra)
        table = ROOT / "docs" / "tables" / "phase1_gate.md"
        write_table(rows, extra, table)
        ex.metrics["table"] = str(table)
    for k in ("u", "v", "p", "theta"):
        print(
            f"steady MMS {k}: {[f'{e:.2e}' for e in ver['mms'][k]['err']]} pairs {np.round(ver['mms'][k]['pairwise'], 3)}"
        )
    print(
        "wall grad:",
        np.round(ver["mms"]["wall_gradient_left_mean"]["pairwise"], 3),
        "Poiseuille u",
        np.round(ver["poiseuille"]["u"]["pairwise"], 3),
        "dpdx",
        np.round(ver["poiseuille"]["dpdx"]["pairwise"], 3),
    )
    print(f"GATE: {ex.metrics['gate_pass']}/{ex.metrics['gate_total']} pass")
    for r in rows:
        if not r["pass"]:
            print("  FAIL:", r["item"], r["formal"], round(r["observed_finest_pair"], 3), r["source"])


if __name__ == "__main__":
    main()
