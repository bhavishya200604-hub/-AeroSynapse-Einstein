"""Step 5.1: identifiability of (alpha, k, h) BEFORE any training — sensitivity Jacobian, Fisher information,
SVD / condition number, Cramer-Rao lower bounds — for the dimensional 1D body (slab / cylinder / sphere) of configs/inverse/defaults.yaml.

Studies (studies.* in the config, all user-definable):
  scenarios    named variations of the base problem (default: with_source, no_source, surface_only), every noise level
  sweeps       one setting varied at a time from the base problem at identifiability.reference_level
  fdm_check    Jacobian / bounds with the Phase 1 FDM forward model vs the analytic eigen-series (verification)
  monte_carlo  least-squares fits (analytic model) of noisy synthetic data: sample std of the estimates vs the bound

    python experiments/p5_1_identifiability.py [--config my.yaml] [--set key=value] [--tag name]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import least_squares

from aerosynapse.thermal.inverse import body as SL
from aerosynapse.thermal.inverse import design as DS
from aerosynapse.thermal.inverse import identifiability as ID
from aerosynapse.thermal.inverse import spec as IS
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.viz import inverse_figs as VF
from aerosynapse.thermal.viz.style import save_fig

P = list(SL.PARAMS)
PCOL = {"alpha": "#0072B2", "k": "#D55E00", "h": "#009E73"}
SWEEP_PATH = {
    "h": ("parameters", "h"),
    "alpha": ("parameters", "alpha"),
    "k": ("parameters", "k"),
    "q_vol": ("body", "q_vol"),
    "n_sensors": ("measurement", "sensors", "n"),
    "layout": ("measurement", "sensors", "layout"),
    "t1": ("measurement", "times", "t1"),
}


def sweep_override(spec: dict, key: str, value) -> dict:
    """Override of one sweep value. A design part read from the data file (from_data) is switched off for the part the
    sweep varies: sensor sweeps use the configured layout / n; the t1 sweep keeps the file's t0 and number of times."""
    over = nested(SWEEP_PATH[key], value)
    m = spec["measurement"]
    if key in ("n_sensors", "layout") and m["sensors"].get("from_data"):
        over = IS.deep_merge(over, {"measurement": {"sensors": {"from_data": False}}})
    if key == "t1" and m["times"].get("from_data"):
        lin = IS.explicit_design(spec, keep_time_grid=False)["measurement"]["times"]
        over = IS.deep_merge(over, {"measurement": {"times": {k: v for k, v in lin.items() if k != "t1"}}})
    return over


def nested(path: tuple, value) -> dict:
    d = value
    for k in reversed(path):
        d = {k: d}
    return d


def setup(spec: dict, model: str | None = None):
    """(forward function of a parameter dict, true parameters, sensor positions, times)."""
    sl, xs, ts = IS.body_of(spec), IS.sensor_positions(spec), IS.measurement_times(spec)
    p = {k: float(spec["parameters"][k]) for k in P}
    n, f = IS.n_terms(spec), spec["forward"]
    m = model or f["model"]
    fdm = IS.fdm_settings(spec)
    return (lambda pp: SL.temperatures(pp, sl, xs, ts, m, n, fdm)), p, xs, ts


def analyse_spec(spec: dict, levels: list[float], model: str | None = None) -> dict:
    fn, p, xs, ts = setup(spec, model)
    T = fn(p)
    J = ID.jacobian(fn, p, P, float(spec["jacobian"]["rel_step"]))
    out = {
        "groups": IS.body_of(spec).groups(p),
        "n_sensors": len(xs),
        "sensors_frac": (xs / float(spec["body"]["L"])).tolist(),
        "n_times": len(ts),
        "n_obs": int(T.size),
        "T_range": [float(T.min()), float(T.max())],
        "levels": {},
    }
    for lv in levels:
        sig = IS.noise_sigma(spec, T, lv)
        r = ID.analyse(J, sig, P, float(spec["identifiability"]["rank_tol"]))
        r.pop("fisher")
        r["sigma_K_mean"] = float(np.mean(sig))
        out["levels"][str(lv)] = r
    out["_J"], out["_T"], out["_ts"], out["_xs"] = J, T, ts, xs
    return out


def verdict(c: float, thr: list[float]) -> str:
    if not np.isfinite(c):
        return "not identifiable (structural)"
    return (
        "well identifiable" if c < thr[0] else "weakly identifiable" if c < thr[1] else "practically not identifiable"
    )


def monte_carlo(spec: dict, level: float, R: int, seed: int) -> dict:
    """R least-squares fits of noisy synthetic data (analytic model, log-parameters, start 20% off in every
    parameter); sample std of log-estimates vs the bound."""
    fn, p, _, _ = setup(spec, "analytic")
    T = fn(p)
    sig = IS.noise_sigma(spec, T, level)
    rng = np.random.default_rng(seed)
    x_true = np.log([p[k] for k in P])
    est, fails, nfev = [], 0, []
    calls = [0]
    for _ in range(R):
        data = T + sig * rng.standard_normal(T.shape)

        def res(x, data=data):
            calls[0] += 1
            return ((fn(dict(zip(P, np.exp(x), strict=True))) - data) / sig).ravel()

        calls[0] = 0

        sol = least_squares(res, x_true + np.log(1.2), method="lm", xtol=1e-12, ftol=1e-12)
        if not sol.success:
            fails += 1
            continue
        est.append(sol.x)
        nfev.append(calls[0])  # real model calls incl. the finite-difference Jacobian (SciPy's nfev excludes them)
    E = np.asarray(est)
    std = E.std(0, ddof=1)
    return {
        "replicates": R,
        "failed": fails,
        "level": level,
        "sample_std_log": dict(zip(P, std.tolist(), strict=True)),
        "std_rel_uncertainty": 1.0 / np.sqrt(2.0 * (len(E) - 1)),  # relative 1-sigma error of a sample std
        "bias_log": dict(zip(P, (E.mean(0) - x_true).tolist(), strict=True)),
        "mean_model_calls": float(np.mean(nfev)),
        "_E": E,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument(
        "--table-only", action="store_true", help="rebuild the markdown table from the saved metrics of --tag"
    )
    a = ap.parse_args()
    cli = IS.parse_set(a.sets)
    spec = IS.load_spec(a.config or None, cli)
    st, idf = spec["studies"], spec["identifiability"]
    levels = [float(v) for v in spec["measurement"]["noise"]["levels"]]
    # your data: the bounds at YOUR noise level (data.level, e.g. auto from the sigma column) instead of the reference
    ref = float(spec["data"]["level"]) if spec["data"]["source"] == "file" else float(idf["reference_level"])
    thr = [float(v) for v in idf.get("thresholds", [10.0, 50.0])]
    VF.style()
    table = (
        ROOT / "results" / f"p5_1_identifiability_{a.tag}" / "identifiability.md"
        if (a.config or a.sets or a.tag != "default")
        else ROOT / "docs" / "tables" / "phase5_1_identifiability.md"
    )
    if a.table_only:
        import json

        from aerosynapse.thermal.experiments.p5_3_classical import _restore

        met = _restore(
            json.loads(
                (ROOT / "results" / f"p5_1_identifiability_{a.tag}" / "metrics.json").read_text(encoding="utf-8")
            )
        )
        keys = ("scenarios", "sweeps", "fdm_check", "monte_carlo", "design")
        report(
            {k: met.get(k) for k in keys}, spec, float(met["reference_level"]), met["thresholds_percent"], table, a.tag
        )
        print(f"wrote {table}")
        return
    with Experiment(f"p5_1_identifiability_{a.tag}", spec) as ex:
        res: dict = {"scenarios": {}, "sweeps": {}}
        for name, over in (st.get("scenarios") or {"base": {}}).items():
            t0 = time.perf_counter()
            r = analyse_spec(
                IS.load_spec(a.config or None, IS.deep_merge(cli, over or {})), sorted(set(levels + [ref]))
            )
            res["scenarios"][name] = r
            for lv, x in r["levels"].items():
                ex.log(
                    study="scenario",
                    name=name,
                    level=float(lv),
                    rank=x["rank"],
                    cond=x["condition_number"],
                    **{f"crlb_{k}_pct": v for k, v in x["crlb_percent"].items()},
                )
            x = r["levels"][str(ref)]
            print(
                f"[{name}] rank {x['rank']} cond {x['condition_number']:.3g} CRLB@{ref:g}: "
                + ", ".join(f"{k} {v:.3g}%" for k, v in x["crlb_percent"].items())
                + f"  ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
        for key, values in (st.get("sweeps") or {}).items():
            rows = []
            for v in values:
                r = analyse_spec(
                    IS.load_spec(a.config or None, IS.deep_merge(cli, sweep_override(spec, key, v))), [ref]
                )
                x = r["levels"][str(ref)]
                rows.append(
                    {
                        "value": v,
                        "groups": r["groups"],
                        "n_obs": r["n_obs"],
                        "rank": x["rank"],
                        "cond": x["condition_number"],
                        "crlb_percent": x["crlb_percent"],
                        "correlation": x["correlation"],
                        "collinearity_index": x["collinearity_index"],
                    }
                )
                ex.log(
                    study="sweep",
                    name=key,
                    value=str(v),
                    rank=x["rank"],
                    cond=x["condition_number"],
                    **{f"crlb_{k}_pct": c for k, c in x["crlb_percent"].items()},
                )
            res["sweeps"][key] = rows
            print(
                f"[sweep {key}] "
                + "; ".join(f"{r['value']}: " + "/".join(f"{r['crlb_percent'][k]:.3g}" for k in P) for r in rows),
                flush=True,
            )
        if st.get("fdm_check"):
            t0 = time.perf_counter()
            base = res["scenarios"].get("with_source") or next(iter(res["scenarios"].values()))
            rf = analyse_spec(spec, [ref], "fdm")
            Ja, Jf = base["_J"], rf["_J"]
            res["fdm_check"] = {
                "max_abs_T_diff_K": float(np.abs(base["_T"] - rf["_T"]).max()),
                "jacobian_rel_col_diff": dict(
                    zip(P, (np.linalg.norm(Jf - Ja, axis=0) / np.linalg.norm(Ja, axis=0)).tolist(), strict=True)
                ),
                "crlb_percent_fdm": rf["levels"][str(ref)]["crlb_percent"],
                "crlb_percent_analytic": base["levels"][str(ref)]["crlb_percent"],
                "fdm": IS.fdm_settings(spec),
                "wall_s": time.perf_counter() - t0,
            }
            print(f"[fdm check] {res['fdm_check']['jacobian_rel_col_diff']}", flush=True)
        mc = st.get("monte_carlo") or {}
        if int(mc.get("replicates", 0)) > 0:
            t0 = time.perf_counter()
            res["monte_carlo"] = monte_carlo(spec, float(mc["level"]), int(mc["replicates"]), int(spec["seed"]))
            res["monte_carlo"]["crlb_log"] = analyse_spec(spec, [float(mc["level"])])["levels"][
                str(float(mc["level"]))
            ]["crlb_log"]
            res["monte_carlo"]["wall_s"] = time.perf_counter() - t0
            print(
                f"[monte carlo] std {res['monte_carlo']['sample_std_log']} vs CRLB {res['monte_carlo']['crlb_log']}",
                flush=True,
            )
        do = spec.get("design_optimisation") or {}
        if do.get("enabled"):
            t0 = time.perf_counter()
            dsg = DS.Designer(spec)
            opt = dsg.optimise(int(do["starts"]), int(do["seed"]))
            for k in ("base", "optimal"):
                opt[k].pop("fisher")
            print(
                f"[design] {opt['optimal_design']} CRLB % {opt['optimal']['crlb_percent']} (base {opt['base']['crlb_percent']}) "
                f"{opt['n_evaluations']} evaluations",
                flush=True,
            )
            if int(do.get("monte_carlo", 0)) > 0:
                lv = float((st.get("monte_carlo") or {}).get("level", ref))
                so = dsg.spec_of(opt["optimal_design"])
                m = monte_carlo(so, lv, int(do["monte_carlo"]), int(spec["seed"]))
                m["crlb_log"] = analyse_spec(so, [lv])["levels"][str(lv)]["crlb_log"]
                opt["monte_carlo"] = m
                print(f"[design monte carlo] std {m['sample_std_log']} vs CRLB {m['crlb_log']}", flush=True)
            opt["wall_s"] = time.perf_counter() - t0
            res["design"] = opt
            res["_design_spec"] = dsg.spec_of(opt["optimal_design"])
        figs = figures(res, spec, ref, ex, a.tag)
        clean = {
            "scenarios": {
                n: {k: v for k, v in r.items() if not k.startswith("_")} for n, r in res["scenarios"].items()
            },
            "sweeps": res["sweeps"],
            "fdm_check": res.get("fdm_check"),
            "monte_carlo": {k: v for k, v in (res.get("monte_carlo") or {}).items() if not k.startswith("_")} or None,
            "design": _clean_design(res.get("design")),
        }
        ex.metrics.update(
            {**clean, "reference_level": ref, "thresholds_percent": thr, "figures": figs, "git": git_hash()}
        )
        report(clean, spec, ref, thr, table, a.tag)
        ex.metrics["table"] = str(table)
    print(f"wrote {table}")


def _clean_design(d: dict | None) -> dict | None:
    if not d:
        return None
    out = {k: v for k, v in d.items() if k != "monte_carlo"}
    if d.get("monte_carlo"):
        out["monte_carlo"] = {k: v for k, v in d["monte_carlo"].items() if not k.startswith("_")}
    return out


def report(res: dict, spec: dict, ref: float, thr: list[float], path, tag: str) -> None:
    s, p = spec["body"], spec["parameters"]
    # your data: the parameters are nominal values and the design is read from the data file
    pw = "true" if spec["data"].get("compare_to_parameters", True) else "nominal"
    m = spec["measurement"]
    sens = (
        f"from your data file ({len(IS.sensor_positions(spec))} positions)"
        if m["sensors"].get("from_data") and m["sensors"].get("positions") is None
        else str(m["sensors"])
    )
    tims = (
        f"from your data file ({len(IS.measurement_times(spec))} times)"
        if m["times"].get("from_data")
        else str(m["times"])
    )
    L = [
        f"# Step 5.1 identifiability of (alpha, k, h) — `{tag}` (auto-generated by experiments/p5_1_identifiability.py)",
        "",
        f"Body {s.get('shape', 'slab')}, L = {s['L']:g} m, T0 = {s['T0']:g}, T_inf = {s['T_inf']:g}, q_vol = {s['q_vol']:g} W/m³; {pw} alpha = {p['alpha']:g} m²/s, "
        f"k = {p['k']:g} W/(m K), h = {p['h']:g} W/(m² K). Measurements: sensors {sens}, times {tims}, "
        f"noise model {spec['measurement']['noise']['model']} (σ relative to the temperature scale T0 − T_inf). Forward model {spec['forward']['model']}. "
        f"CRLB = Cramér–Rao lower bound on the standard deviation of an unbiased estimate (1σ, % of the parameter, first order). "
        f"Verdict per parameter at the reference level {ref:g}: < {thr[0]:g}% well, < {thr[1]:g}% weakly, otherwise practically not identifiable; "
        "inf = structurally not identifiable (singular Fisher information).",
        "",
        "## Scenarios",
        "",
        "| scenario | Bi | Q | sensors x/L | obs | rank | condition no. | collinearity γ | null direction (log α, log k, log h) | corr(α,k) / corr(α,h) / corr(k,h) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n, r in res["scenarios"].items():
        x = r["levels"][str(ref)]
        c = x["correlation"]
        L.append(
            f"| {n} | {r['groups']['Bi']:.4g} | {r['groups']['Q']:.4g} | {', '.join(f'{v:.3g}' for v in r['sensors_frac'])} | {r['n_obs']} | {x['rank']}/3 | "
            f"{x['condition_number']:.3g} | {x['collinearity_index']:.3g} | {'; '.join('(' + ', '.join(f'{v:.3f}' for v in d) + ')' for d in x['null_directions']) or '—'} | "
            + (f"{c[0][1]:.3f} / {c[0][2]:.3f} / {c[1][2]:.3f}" if c else "—")
            + " |"
        )
    L += [
        "",
        "CRLB (1σ, % of the parameter) per noise level:",
        "",
        "| scenario | noise level | mean σ [K] | α | k | h | verdict at reference level |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, r in res["scenarios"].items():
        for lv, x in r["levels"].items():
            cp = x["crlb_percent"]
            ver = "; ".join(f"{k}: {verdict(cp[k], thr)}" for k in P) if float(lv) == ref else ""
            L.append(
                f"| {n} | {float(lv):g} | {x['sigma_K_mean']:.3g} | {_pct(cp['alpha'])} | {_pct(cp['k'])} | {_pct(cp['h'])} | {ver} |"
            )
    L += ["", f"## Sweeps (one setting varied from the base problem, noise level {ref:g})", ""]
    for key, rows in res["sweeps"].items():
        L += [
            f"### {key}",
            "",
            "| value | Bi | Q | obs | rank | condition no. | CRLB α % | CRLB k % | CRLB h % | corr(k,h) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            c = r["correlation"]
            L.append(
                f"| {r['value']} | {r['groups']['Bi']:.4g} | {r['groups']['Q']:.4g} | {r['n_obs']} | {r['rank']}/3 | {r['cond']:.3g} | "
                f"{_pct(r['crlb_percent']['alpha'])} | {_pct(r['crlb_percent']['k'])} | {_pct(r['crlb_percent']['h'])} | {f'{c[1][2]:.3f}' if c else '—'} |"
            )
        L.append("")
    fc = res.get("fdm_check")
    if fc:
        L += [
            "## Verification: FDM forward model vs analytic eigen-series (base problem)",
            "",
            f"Max |T_fdm − T_analytic| = {fc['max_abs_T_diff_K']:.2e} K (FDM {fc['fdm']}); relative Jacobian column differences "
            + ", ".join(f"{k} {v:.2e}" for k, v in fc["jacobian_rel_col_diff"].items())
            + "; CRLB % FDM vs analytic: "
            + ", ".join(f"{k} {fc['crlb_percent_fdm'][k]:.4g} vs {fc['crlb_percent_analytic'][k]:.4g}" for k in P)
            + ".",
            "",
        ]
    mc = res.get("monte_carlo")
    if mc:
        u = mc["std_rel_uncertainty"]
        L += [
            "## Verification: Monte-Carlo least squares vs the Cramér–Rao bound",
            "",
            f"{mc['replicates']} noisy data sets (level {mc['level']:g}), Levenberg–Marquardt on log-parameters from a start 20% off in every parameter; "
            f"{mc['failed']} failed fits; mean {mc['mean_model_calls']:.0f} model evaluations per fit (incl. finite-difference Jacobians). A sample std of {mc['replicates'] - mc['failed']} estimates has a "
            f"relative 1σ uncertainty of {100 * u:.1f}%.",
            "",
            "| parameter | sample std of log-estimate | CRLB | ratio | bias of log-estimate |",
            "|---|---|---|---|---|",
        ]
        for i, k in enumerate(P):
            sd, cr = mc["sample_std_log"][k], mc["crlb_log"][i]
            L.append(f"| {k} | {sd:.4g} | {cr:.4g} | {sd / cr:.3f} | {mc['bias_log'][k]:+.2e} |")
        L.append("")
    d = res.get("design")
    if d:
        b, o = d["base"], d["optimal"]
        L += [
            "## Optimal experiment design",
            "",
            f"Criterion {d['criterion']}, variables {d['variables']}, bounds {spec['design_optimisation']['bounds']}; same number of readings as the base "
            f"design; {len(d['starts'])} starts (best objective {min(s['f'] for s in d['starts']):.4f}, all: "
            + ", ".join(f"{s['f']:.4f}" for s in d["starts"])
            + f"); {d['n_evaluations']} design evaluations, {d['wall_s']:.0f} s.",
            "",
            "| design | sensors r/L | record length t1 [s] | time spacing | CRLB α % | CRLB k % | CRLB h % | condition no. | corr(α,k) | corr(k,h) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, dd, r in (("base", d["base_design"], b), ("optimised", d["optimal_design"], o)):
            c = r["correlation"]
            L.append(
                f"| {name} | {', '.join(f'{v:.3f}' for v in dd.get('positions', []))} | {dd.get('t1', float('nan')):.0f} | {dd.get('spacing', 1.0):.2f} | "
                + " | ".join(f"{r['crlb_percent'][k]:.3g}" for k in P)
                + f" | {r['condition_number']:.3g} | "
                + (f"{c[0][1]:.3f} | {c[1][2]:.3f}" if c else "— | —")
                + " |"
            )
        L += [
            "",
            "Improvement (base ÷ optimised bound): "
            + ", ".join(f"{k} {b['crlb_percent'][k] / o['crlb_percent'][k]:.2f}x" for k in P)
            + ".",
        ]
        m = d.get("monte_carlo")
        if m:
            u = m["std_rel_uncertainty"]
            L += [
                "",
                f"Verification at the optimised design: {m['replicates']} least-squares fits (level {m['level']:g}, start 20% off), {m['failed']} failed; "
                f"sample std / CRLB "
                + ", ".join(f"{k} {m['sample_std_log'][k] / m['crlb_log'][i]:.3f}" for i, k in enumerate(P))
                + f" (std of a sample std: {100 * u:.1f}%).",
            ]
        L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _pct(v: float) -> str:
    return "inf" if not np.isfinite(v) else f"{v:.3g}"


def _cov(r: dict) -> np.ndarray:
    c = np.asarray(r["crlb_log"], float)
    return np.asarray(r["correlation"], float) * np.outer(c, c)


def figures(res: dict, spec: dict, ref: float, ex, tag: str) -> list[str]:
    """Sensitivities, bound sweeps, confidence ellipses (CRLB + Monte-Carlo estimates), optimised design."""
    paths = []
    sc = res["scenarios"]
    L = float(spec["body"]["L"])
    panels = []
    for n in [x for x in ("with_source", "no_source") if x in sc] or list(sc)[:2]:
        r = sc[n]
        J = r["_J"].reshape(len(r["_ts"]), len(r["_xs"]), len(P))
        panels.append(
            {
                "title": f"{n.replace('_', ' ')}  (Bi {r['groups']['Bi']:.3g}, Q {r['groups']['Q']:.3g})",
                "t": r["_ts"],
                "J": J,
                "x_frac": r["_xs"] / L,
                "sigma_K": r["levels"][str(ref)]["sigma_K_mean"],
            }
        )
    paths += save_fig(VF.sensitivities(panels, P), f"p5_1_sensitivities_{tag}", ex.dir)
    meta = {  # x label, axis scale, panel title, base value of the swept setting
        "h": ("h [W/(m² K)]", True, "heat transfer coefficient", spec["parameters"]["h"]),
        "q_vol": ("q_vol [W/m³]", "symlog", "heat source", spec["body"]["q_vol"]),
        "n_sensors": ("number of sensors", True, "sensor count", spec["measurement"]["sensors"]["n"]),
        "t1": ("t1 [s]", True, "record length", spec["measurement"]["times"]["t1"]),
        "layout": ("", False, "sensor layout", spec["measurement"]["sensors"]["layout"]),
        "alpha": ("α [m²/s]", True, "diffusivity", spec["parameters"]["alpha"]),
        "k": ("k [W/(m K)]", True, "conductivity", spec["parameters"]["k"]),
    }
    sweeps = {}
    for key, rows in res["sweeps"].items():
        xl, lg, ttl, bv = meta.get(key, (key, False, key, None))
        sweeps[key] = {
            "x": [r["value"] for r in rows],
            "xlabel": xl,
            "log_x": lg,
            "title": ttl,
            "base": bv,
            "crlb": {k: [r["crlb_percent"][k] for r in rows] for k in P},
        }
    if sweeps:
        thr = [float(v) for v in spec["identifiability"].get("thresholds", [10.0, 50.0])]
        paths += save_fig(VF.crlb_sweeps(sweeps, P, thr, ref), f"p5_1_crlb_sweeps_{tag}", ex.dir)
    mc, d = res.get("monte_carlo"), res.get("design")
    if mc and d and d.get("monte_carlo"):
        x_true = np.log([float(spec["parameters"][k]) for k in P])
        lv = mc["level"]
        rb = analyse_spec(spec, [lv])["levels"][str(lv)]
        ro = analyse_spec(res["_design_spec"], [lv])["levels"][str(lv)]
        designs = [
            {"name": "base design", "color": VF.BASE, "cov": _cov(rb), "samples": mc["_E"] - x_true},
            {"name": "optimised design", "color": VF.EMPH, "cov": _cov(ro), "samples": d["monte_carlo"]["_E"] - x_true},
        ]
        paths += save_fig(VF.confidence_ellipses(designs, P), f"p5_1_ellipses_{tag}", ex.dir)
    if d:
        so = res["_design_spec"]
        base = {
            "positions": (res["scenarios"].get("with_source", next(iter(sc.values())))["_xs"] / L).tolist(),
            "times": IS.measurement_times(spec).tolist(),
            "crlb": d["base"]["crlb_percent"],
        }
        opt = {
            "positions": (IS.sensor_positions(so) / L).tolist(),
            "times": IS.measurement_times(so).tolist(),
            "crlb": d["optimal"]["crlb_percent"],
        }
        paths += save_fig(VF.design_comparison(base, opt, P, L), f"p5_1_design_{tag}", ex.dir)
    return [str(p) for p in paths]


if __name__ == "__main__":
    main()
