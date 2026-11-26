"""Step 5.3: classical baseline — weighted least squares (SciPy) with the FDM forward solver, for (alpha, k, h).

Per case (studies.classical.cases; base = the configured problem, optimised_design = the Step 5.1 optimal design,
no_source = q_vol 0 so that k and h are confounded):
  fit          FDM least squares from the initial guess: estimates, 95 % intervals (Gauss-Newton), convergence path
  reference    the same fit with the analytic eigen-series model (discretisation bias of the FDM estimator)
  multistart   classical.multistart.n starts log-uniform in [guess / spread, guess x spread]: convergence basin
  monte_carlo  classical.monte_carlo.replicates FDM fits of independent noise realisations (identifiable cases only):
               bias and sample spread vs the Cramer-Rao bound, coverage of the 95 % intervals
and, if studies.classical.compare_pinn names a Step 5.2 run, that run's PINN errors on the same data.

    python experiments/p5_3_classical.py [--config my.yaml] [--set key=value] [--tag name] [--figures-only]
    (your measured data: --set data.source=file --set data.file=... --set data.compare_to_parameters=false)
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from aerosynapse.thermal.experiments import p5_2_pinn_inverse as P52
from aerosynapse.thermal.experiments.p5_1_identifiability import analyse_spec
from aerosynapse.thermal.inverse import body as B
from aerosynapse.thermal.inverse import classical as C
from aerosynapse.thermal.inverse import data as D
from aerosynapse.thermal.inverse import identifiability as ID
from aerosynapse.thermal.inverse import spec as IS
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, _jsonable, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz import inverse_figs as VF
from aerosynapse.thermal.viz.style import save_fig

MOD = "aerosynapse.thermal.experiments.p5_3_classical"


def case_spec(files, cli: dict, case: str) -> dict:
    """base = the configured problem (your data file, if data.source = file); no_source = the same with q_vol 0;
    otherwise a Step 5.2 case (a body shape name or optimised_design). With your data, every case other than base is
    a SYNTHETIC what-if at your nominal `parameters` (your file belongs to your configuration only)."""
    if case == "base":
        return IS.load_spec(files or None, cli)
    base = IS.load_spec(files or None, cli)
    what_if = (
        {"data": {"source": "synthetic", "compare_to_parameters": True}} if base["data"]["source"] == "file" else {}
    )
    if case == "no_source":
        return IS.load_spec(files or None, IS.deep_merge(cli, IS.deep_merge(what_if, {"body": {"q_vol": 0.0}})))
    return P52.case_spec(files, IS.deep_merge(cli, what_if), case)


def bound(spec: dict, level: float) -> dict:
    """Cramer-Rao analysis of the UNKNOWN parameters only (the known ones are fixed) on the spec's design."""
    a = analyse_spec(spec, [level])
    unknown = list(spec["unknown"])
    J = a["_J"][:, [list(B.PARAMS).index(k) for k in unknown]]
    sig = IS.noise_sigma(spec, a["_T"], level)
    r = ID.analyse(J, sig, unknown, float(spec["identifiability"]["rank_tol"]))
    r.pop("fisher")
    return r


def mc_batch(spec: dict, seeds: list[int], level: float) -> list[dict]:
    """FDM least-squares fits of independent synthetic noise realisations (run in its own process)."""
    unknown = list(spec["unknown"])
    guess = {k: float(spec["initial_guess"][k]) for k in unknown}
    out = []
    for sd in seeds:
        r = C.safe_fit(spec, D.synthetic(spec, level, sd), unknown, guess)
        keep = ("estimates", "sd_log", "success", "n_model_calls", "chi2_per_reading")
        out.append({k: r[k] for k in keep} | {"seed": sd})
    return out


def multistart_task(spec: dict, n: int, spread: float, seed: int) -> list[dict]:
    unknown = list(spec["unknown"])
    guess = {k: float(spec["initial_guess"][k]) for k in unknown}
    return C.multistart(spec, D.load(spec), unknown, guess, n, spread, seed)


def rel(est: dict, spec: dict) -> dict:
    return {k: (est[k] - float(spec["parameters"][k])) / float(spec["parameters"][k]) for k in spec["unknown"]}


def bi_log_sd(fit: dict, unknown: list[str]) -> float | None:
    """sd of ln Bi = ln h - ln k at the estimate (Gauss-Newton; finite even when k and h are confounded)."""
    if not {"h", "k"} <= set(unknown):
        return None
    return C.log_sd(fit, unknown, {"h": 1.0, "k": -1.0})


def run_case(files, cli: dict, case: str, ex) -> dict:
    t0 = time.perf_counter()
    s = case_spec(files, cli, case)
    unknown = list(s["unknown"])
    guess = {k: float(s["initial_guess"][k]) for k in unknown}
    d = D.load(s)
    f = C.fit(s, d, unknown, guess)
    ref = C.fit(s, d, unknown, guess, "analytic")
    cr = bound(s, float(s["data"]["level"])) if s["data"]["source"] == "synthetic" else None
    body = IS.body_of(s)
    g, gt = body.groups(f["estimates"]), body.groups({k: float(s["parameters"][k]) for k in B.PARAMS})
    # discretisation error of the estimator's forward model on this design (at the truth for synthetic data, else at
    # the estimate): FDM with the estimator's mesh and step counts vs the eigen-series
    p_chk = {k: float(s["parameters"][k]) for k in B.PARAMS} if s["data"]["source"] == "synthetic" else f["estimates"]
    t1 = time.perf_counter()
    T_fdm = D.model_at(p_chk, s, d["r"], d["t"], IS.n_terms(s), "fdm", C.forward_settings(s, guess))
    t_fdm = time.perf_counter() - t1
    T_ser = D.model_at(p_chk, s, d["r"], d["t"], IS.n_terms(s))
    out = {
        "fdm_vs_series_max_K": float(np.abs(T_fdm - T_ser).max()),
        "fdm_eval_s": t_fdm,
        "spec": s,
        "fit": f,
        "reference": {k: ref[k] for k in ("estimates", "sd_log", "chi2_per_reading", "n_model_calls", "wall_s")},
        "crlb_percent": cr["crlb_percent"] if cr else None,
        "rank": cr["rank"] if cr else None,
        "n_readings": len(d["T"]),
        "sigma_mean_K": float(np.mean(d["sigma"])),
        "design": {"sensors_frac": (np.unique(d["r"]) / float(s["body"]["L"])).tolist(), "t_max": float(d["t"].max())},
        "q_vol": float(s["body"]["q_vol"]),
        "Bi": g["Bi"],
        "Bi_true": gt["Bi"],
        "Bi_sd_log": bi_log_sd(f, unknown),
        "Q": g["Q"],
        "Q_true": gt["Q"],
    }
    if s["data"].get("compare_to_parameters", True):
        out["rel_err"] = rel(f["estimates"], s)
        out["ref_rel_err"] = rel(ref["estimates"], s)
    for m, r in (("fdm", f), ("analytic", ref)):
        ex.log(study="fit", case=case, model=m, chi2=r["chi2_per_reading"], **{k: r["estimates"][k] for k in unknown})
    est = ", ".join(f"{k} {f['estimates'][k]:.4g} (sd ln {f['sd_log'][k]:.3g})" for k in unknown)
    ana = ", ".join(f"{k} {ref['estimates'][k]:.4g}" for k in unknown)
    print(
        f"[{case}] FDM LS {est}; chi2/reading {f['chi2_per_reading']:.4f}, {f['n_model_calls']} calls, "
        f"{f['wall_s']:.0f} s; analytic {ana}; Bi {g['Bi']:.4g} / {gt['Bi']:.4g} ({time.perf_counter() - t0:.0f} s)",
        flush=True,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--figures-only", action="store_true", help="redraw the figures and the table from saved metrics")
    ap.add_argument(
        "--resummarise",
        action="store_true",
        help="with --figures-only: recompute the multi-start summary from the saved fits",
    )
    a = ap.parse_args()
    cli = IS.parse_set(a.sets)
    base = IS.load_spec(a.config or None, cli)
    cl, st = base["classical"], base["studies"]["classical"]
    custom = bool(a.config or a.sets or a.tag != "default")
    odir = ROOT / "results" / f"p5_3_classical_{a.tag}"
    table = odir / "classical.md" if custom else ROOT / "docs" / "tables" / "phase5_3_classical.md"
    VF.style()
    if a.figures_only:
        met = _restore(json.loads((odir / "metrics.json").read_text(encoding="utf-8")))
        if a.resummarise:  # recompute the multi-start summary from the saved per-start fits (no re-fitting)
            tol = float(cl["multistart"].get("chi2_tol", 0.1))
            for c, d in met["cases"].items():
                m = d["multistart"]
                keys = ("start", "estimates", "chi2_per_reading", "n_model_calls", "success", "error")
                cols = (
                    m["starts"],
                    m["estimates"],
                    m["chi2_per_reading"],
                    m["n_model_calls"],
                    m["success"],
                    m["errors"],
                )
                r = [dict(zip(keys, v, strict=True)) for v in zip(*cols, strict=True)]
                d["multistart"] = ms_summary(r, d["unknown"], d["n_readings"], tol)
            met["multistart_resummarised"] = {"git": git_hash(), "chi2_tol_total": tol}
            (odir / "metrics.json").write_text(json.dumps(_jsonable(met), indent=2), encoding="utf-8")
        figures(met["cases"], base, met["pinn"], a.tag, odir)
        report(met["cases"], base, met["pinn"], table, a.tag)
        print(f"wrote {table}")
        return
    with Experiment(f"p5_3_classical_{a.tag}", base) as ex:
        cases = {c: run_case(a.config, cli, c, ex) for c in st["cases"]}
        ms = cl["multistart"]
        mjobs = [
            (
                MOD,
                "multistart_task",
                dict(spec=d["spec"], n=int(ms["n"]), spread=float(ms["spread"]), seed=int(ms["seed"])),
            )
            for d in cases.values()
        ]
        mout = run_tasks(
            mjobs,
            base["execution"],
            lambda i, r, sec: print(f"[multistart {list(cases)[i]}] {len(r)} starts, {sec:.0f} s", flush=True),
        )
        for c, r in zip(cases, mout, strict=True):
            cases[c]["multistart"] = ms_summary(
                r, list(cases[c]["spec"]["unknown"]), cases[c]["n_readings"], float(ms.get("chi2_tol", 0.1))
            )
            for x in r:
                ex.log(study="multistart", case=c, chi2=x["chi2_per_reading"], **x["estimates"])
        mc = cl["monte_carlo"]
        R, bsz, lv = int(mc["replicates"]), int(mc["batch"]), float(mc["level"])
        mcases = [
            c
            for c, d in cases.items()
            if R > 0 and d["rank"] == len(d["spec"]["unknown"]) and d["spec"]["data"]["source"] == "synthetic"
        ]
        jobs, labels = [], []
        for c in mcases:
            seeds = list(range(1000, 1000 + R))
            for i in range(0, R, bsz):
                jobs.append((MOD, "mc_batch", dict(spec=cases[c]["spec"], seeds=seeds[i : i + bsz], level=lv)))
                labels.append(c)
        outs = run_tasks(
            jobs,
            base["execution"],
            lambda i, r, sec: print(f"[monte carlo {labels[i]}] batch {i + 1}/{len(jobs)}, {sec:.0f} s", flush=True),
        )
        for c in mcases:
            fits = [x for lab, o in zip(labels, outs, strict=True) if lab == c for x in o]
            m = cases[c]["monte_carlo"] = mc_summary(fits, cases[c]["spec"], lv)
            for x in fits:
                ex.log(study="monte_carlo", case=c, seed=x["seed"], **x["estimates"])
            ratio = ", ".join(f"{k} {m['std_log'][k] / m['crlb_log'][k]:.3f}" for k in m["std_log"])
            print(f"[monte carlo {c}] sample std / CRLB {ratio}", flush=True)
        pinn = load_pinn(st.get("compare_pinn"), base) if base["data"].get("compare_to_parameters", True) else None
        clean = {}
        for c, d in cases.items():
            clean[c] = {k: v for k, v in d.items() if k not in ("spec", "fit")}
            clean[c]["fit"] = {k: v for k, v in d["fit"].items() if k != "path"}
            clean[c]["path"] = d["fit"]["path"]
            clean[c]["unknown"] = list(d["spec"]["unknown"])
        ex.metrics.update({"cases": clean, "pinn": pinn, "git": git_hash()})
        ex.metrics["figures"] = figures(clean, base, pinn, a.tag, ex.dir)
        report(clean, base, pinn, table, a.tag)
        ex.metrics["table"] = str(table.relative_to(ROOT))
    print(f"wrote {table}")


def ms_summary(r: list[dict], unknown: list[str], n_readings: int, chi2_tol: float = 0.1) -> dict:
    """Convergence basin of a multi-start: which starts reach the best fit (TOTAL chi2 within chi2_tol of the best:
    0.1 = at most ~0.3 sd along any direction, i.e. the same minimum up to the solver tolerance), which stop elsewhere
    (a plateau or another local minimum), which fail numerically (safe_fit)."""
    ok = [x for x in r if x["estimates"] is not None]
    chi = np.array([x["chi2_per_reading"] for x in r], float)
    best = float(chi.min())
    at_best = (chi - best) * n_readings < chi2_tol
    kh = {"h", "k"} <= set(unknown)
    return {
        "starts": [x["start"] for x in r],
        "estimates": [x["estimates"] for x in r],
        "chi2_per_reading": chi.tolist(),
        "at_best": at_best.tolist(),
        "n_model_calls": [x["n_model_calls"] for x in r],
        "success": [x["success"] for x in r],
        "errors": [x.get("error") for x in r],
        "best_chi2": best,
        "n_at_best": int(at_best.sum()),
        "chi2_tol_total": chi2_tol,
        "n_failed": len(r) - len(ok),
        "spread_log": {k: float(np.ptp(np.log([x["estimates"][k] for x in ok]))) for k in unknown},
        "spread_log_Bi": (
            float(
                np.ptp(
                    [np.log(x["estimates"]["h"] / x["estimates"]["k"]) for x, b in zip(r, at_best, strict=True) if b]
                )
            )
            if kh
            else None
        ),  # over the starts at the best fit
        "at_best_spread_log": {
            k: float(np.ptp(np.log([x["estimates"][k] for x, b in zip(r, at_best, strict=True) if b]))) for k in unknown
        },
    }


def mc_summary(fits: list[dict], spec: dict, level: float) -> dict:
    unknown = list(spec["unknown"])
    ok = [x for x in fits if x["success"]]
    X = np.array([[np.log(x["estimates"][k] / float(spec["parameters"][k])) for k in unknown] for x in ok])
    cr = bound(spec, level)
    c = np.asarray(cr["crlb_log"], float)
    n = len(ok)
    cover = {
        k: float(np.mean([abs(X[i, j]) <= 1.96 * ok[i]["sd_log"][k] for i in range(n)])) for j, k in enumerate(unknown)
    }
    return {
        "replicates": len(fits),
        "failed": len(fits) - n,
        "level": level,
        "bias_log": dict(zip(unknown, X.mean(0).tolist(), strict=True)),
        "bias_se_log": dict(zip(unknown, (X.std(0, ddof=1) / np.sqrt(n)).tolist(), strict=True)),
        "std_log": dict(zip(unknown, X.std(0, ddof=1).tolist(), strict=True)),
        "crlb_log": dict(zip(unknown, c.tolist(), strict=True)),
        "std_rel_uncertainty": float(1 / np.sqrt(2 * (n - 1))),
        "coverage95": cover,
        "mean_model_calls": float(np.mean([x["n_model_calls"] for x in ok])),
        "mean_chi2_per_reading": float(np.mean([x["chi2_per_reading"] for x in ok])),
        "samples_log": X.tolist(),
        "crlb_cov_log": (np.asarray(cr["correlation"], float) * np.outer(c, c)).tolist(),
    }


def load_pinn(run: str | None, spec: dict) -> dict | None:
    """Per-variant PINN errors in % (one value per seed) of a Step 5.2 run on its default case, or None."""
    if not run or not (ROOT / "results" / run / "metrics.json").exists():
        return None
    met = json.loads((ROOT / "results" / run / "metrics.json").read_text(encoding="utf-8"))
    case0 = met["records"][0]["shape"]
    out = {"run": run, "variants": {}}
    for vn in dict.fromkeys(r["variant"] for r in met["records"] if r["shape"] == case0):
        rs = [r for r in met["records"] if r["shape"] == case0 and r["variant"] == vn and "rel_err" in r]
        out["variants"][vn] = {k: [100 * r["rel_err"][k] for r in rs] for k in spec["unknown"]}
    return out


def _restore(o):
    """Saved metrics write non-finite floats as strings ('inf'); turn them back into floats."""
    if isinstance(o, dict):
        return {k: _restore(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_restore(v) for v in o]
    return float(o) if o in ("inf", "-inf", "nan") else o


def _f(v: float | None, fmt: str = ".3g") -> str:
    return "inf" if v is None or not np.isfinite(v) else format(v, fmt)


def report(cases: dict, spec: dict, pinn: dict | None, path, tag: str) -> None:
    cl = spec["classical"]
    p = spec["parameters"]
    L = [
        f"# Step 5.3 classical baseline — `{tag}` (auto-generated by experiments/p5_3_classical.py)",
        "",
        f"Weighted least squares in ln p (scipy least_squares `{cl['method']}`, {cl['jac']} Jacobian, relative step "
        f"{cl['diff_step']:g}) with the FDM forward solver (N {cl['fdm']['N']}, Fourier step "
        f"{cl['fdm']['dt_fourier']:g}, step counts fixed from the initial guess); 95 % intervals exp(ln p ± 1.96 sd) "
        f"from the Gauss–Newton covariance. Initial guess {spec['initial_guess']}; "
        + (
            f"true {p}; data synthetic (level {spec['data']['level']:g}, seed {spec['data']['seed']}). Errors (estimate − true)/true in %."
            if spec["data"].get("compare_to_parameters", True)
            else f"nominal {p}; YOUR data file {spec['data'].get('file')} (noise level {spec['data']['level']:.3g}); estimates in physical units "
            "(other cases: synthetic what-ifs at the nominal values)."
        ),
        "",
    ]
    for c, d in cases.items():
        unk, f = d["unknown"], d["fit"]
        L += [
            f"## {c}: {d['n_readings']} readings, mean σ {d['sigma_mean_K']:.3f} K, q_vol {d['q_vol']:g} W/m³, "
            f"identifiability rank {d['rank'] if d['rank'] is not None else d['fit']['rank']}/{len(unk)}",
            "",
            f"Estimator forward model: max |FDM − series| {d['fdm_vs_series_max_K']:.4f} K on this design, "
            f"{d['fdm_eval_s']:.2f} s per evaluation.",
            "",
            "| estimator | "
            + " | ".join(f"{k} error % [95 % interval]" for k in unk)
            + " | χ²/reading | model calls | time s |",
            "|---|" + "---|" * (len(unk) + 3),
        ]
        if d["crlb_percent"]:
            crlb = " | ".join(f"± {_f(d['crlb_percent'][k])}" for k in unk)
            L.append(f"| Cramér–Rao bound (1σ) | {crlb} | — | — | — |")
        tail = f" | {f['chi2_per_reading']:.4f} | {f['n_model_calls']} | {f['wall_s']:.1f} |"
        if "rel_err" in d:
            ci = {k: [100 * (v / float(p[k]) - 1) for v in f["ci95"][k]] for k in unk}
            cells = [f"{100 * d['rel_err'][k]:+.2f} [{_f(ci[k][0], '+.1f')}, {_f(ci[k][1], '+.1f')}]" for k in unk]
            L.append("| least squares, FDM | " + " | ".join(cells) + tail)
            rf = d["reference"]
            L.append(
                "| least squares, analytic model | "
                + " | ".join(f"{100 * d['ref_rel_err'][k]:+.2f}" for k in unk)
                + f" | {rf['chi2_per_reading']:.4f} | {rf['n_model_calls']} | {rf['wall_s']:.1f} |"
            )
        else:
            cells = [f"{f['estimates'][k]:.4g} [{f['ci95'][k][0]:.4g}, {f['ci95'][k][1]:.4g}]" for k in unk]
            L.append("| least squares, FDM (estimate [95 % interval]) | " + " | ".join(cells) + tail)
        if pinn and c == next(iter(cases)):
            for vn, e in pinn["variants"].items():
                cells = [f"{np.mean(e[k]):+.1f} ± {np.std(e[k], ddof=1):.1f}" for k in unk]
                L.append(f"| PINN {vn} (Step 5.2, {len(e[unk[0]])} seeds) | " + " | ".join(cells) + " | — | — | — |")
        bsd = d["Bi_sd_log"]
        corr = np.round(np.asarray(f["correlation"]), 4).tolist() if f["correlation"] else "singular"
        tw = "true" if "rel_err" in d else "nominal"  # your data: the configured parameters are nominal values only
        L += [
            "",
            f"- Biot number Bi = hL/k: {d['Bi']:.4g} ({tw} {d['Bi_true']:.4g}"
            + (f", error {100 * (d['Bi'] / d['Bi_true'] - 1):+.2f} %" if tw == "true" else "")
            + (f", Gauss–Newton 1σ {100 * bsd:.2g} %" if bsd is not None else "")
            + f"); source group Q = qL²/(kΔT): {d['Q']:.4g} ({tw} {d['Q_true']:.4g}).",
            f"- Correlation of the estimates {corr}; condition number of the weighted Jacobian "
            f"{_f(f['condition_number'])}, rank {f['rank']}/{len(unk)} at the estimate (central differences; the solver's forward-difference Jacobian: "
            f"condition number {_f(f.get('condition_number_solver_jac'))})"
            + (
                f" (null direction in ln p {np.round(f['null_directions'][0], 3).tolist()})"
                if f["null_directions"]
                else ""
            )
            + f"; solver: {f['message']}",
        ]
        m = d["multistart"]
        others = [
            (e, c)
            for e, c, b in zip(m["estimates"], m["chi2_per_reading"], m.get("at_best", []), strict=False)
            if e is not None and not b
        ]
        L.append(
            f"- Multi-start ({len(m['chi2_per_reading'])} starts, spread ×{cl['multistart']['spread']:g}): "
            f"{m['n_at_best']} reach the best χ²/reading {m['best_chi2']:.4f} (total χ² within "
            f"{m.get('chi2_tol_total', 0.1):g} of the best), "
            f"{len(others)} stop elsewhere, {m.get('n_failed', 0)} fail numerically; range of the ln-estimates at the "
            "best fit "
            + ", ".join(f"{k} {v:.2g}" for k, v in (m.get("at_best_spread_log") or m["spread_log"]).items())
            + (f", ln Bi {m['spread_log_Bi']:.2g}" if m["spread_log_Bi"] is not None else "")
            + "."
            + "".join(
                f" Stopped at {', '.join(f'{k} {e[k]:.3g}' for k in unk)} (χ²/reading {c:.3g})." for e, c in others
            )
        )
        mc = d.get("monte_carlo")
        if mc:
            L.append(
                f"- Monte Carlo ({mc['replicates']} FDM fits, {mc['failed']} failed, level {mc['level']:g}): bias of "
                "ln p "
                + ", ".join(f"{k} {mc['bias_log'][k]:+.4f} ± {mc['bias_se_log'][k]:.4f}" for k in unk)
                + "; sample std / CRLB "
                + ", ".join(f"{k} {mc['std_log'][k] / mc['crlb_log'][k]:.3f}" for k in unk)
                + f" (± {100 * mc['std_rel_uncertainty']:.0f} % sampling uncertainty); 95 % interval coverage "
                + ", ".join(f"{k} {100 * mc['coverage95'][k]:.0f} %" for k in unk)
                + f"; {mc['mean_model_calls']:.0f} model calls per fit."
            )
        L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def best_so_far(path: list[dict]) -> list[dict]:
    """The incumbent of a solver path: at each model evaluation, the parameters of the lowest cost so far (rejected
    trust-region trial points and finite-difference evaluations are not iterates)."""
    out, best = [], None
    for x in path:
        if best is None or x["cost"] < best["cost"]:
            best = x
        out.append(best)
    return out


def figures(cases: dict, spec: dict, pinn: dict | None, tag: str, odir) -> list[str]:
    VF.style()
    paths = []
    c0 = next(iter(cases))
    known_truth = "rel_err" in cases[c0]
    # your data (no truth): every ratio is taken to the final least-squares estimate of the base case
    truth = {k: float(spec["parameters"][k]) for k in B.PARAMS} | ({} if known_truth else cases[c0]["fit"]["estimates"])
    ref = "true" if known_truth else "est"
    unk = cases[c0]["unknown"]
    same = {c: d for c, d in cases.items() if d["unknown"] == unk}
    hist = {c.replace("_", " "): [best_so_far(d["path"])] for c, d in same.items()}
    fig = VF.convergence(hist, unk, truth, None, xlabel="model evaluation (best fit so far)", ref=ref)
    paths += save_fig(fig, f"p5_3_convergence_{tag}", odir)
    rows = []  # every estimator on the SAME data (the first case), against that case's bounds
    d0 = cases[c0]
    if "rel_err" in d0:
        for name, est, sd in (
            ("least\nsquares\nFDM", d0["fit"]["estimates"], d0["fit"]["sd_log"]),
            ("least\nsquares\nanalytic", d0["reference"]["estimates"], d0["reference"]["sd_log"]),
        ):
            rows.append(
                {
                    "name": name,
                    "errors": {k: [100 * (est[k] / truth[k] - 1)] for k in unk},
                    "ci": {k: [100 * (est[k] * np.exp(s * 1.96 * sd[k]) / truth[k] - 1) for s in (-1, 1)] for k in unk},
                }
            )
    if pinn:
        rows += [
            {"name": "PINN\n" + vn.replace("_", " ").replace(" ", "\n", 1), "errors": e}
            for vn, e in pinn["variants"].items()
        ]
    if rows:
        paths += save_fig(VF.error_dots(rows, unk, cases[c0]["crlb_percent"]), f"p5_3_errors_{tag}", odir)
    if {"k", "h"} <= set(unk):
        ms = {c.replace("_", " "): d["multistart"] for c, d in same.items()}
        paths += save_fig(VF.multistart_kh(ms, truth, ref), f"p5_3_multistart_{tag}", odir)
    des = [
        {
            "name": c.replace("_", " "),
            "color": VF.BASE if c == c0 else VF.EMPH,
            "cov": d["monte_carlo"]["crlb_cov_log"],
            "samples": d["monte_carlo"]["samples_log"],
        }
        for c, d in same.items()
        if d.get("monte_carlo")
    ]
    if des:
        paths += save_fig(VF.confidence_ellipses(des, unk), f"p5_3_monte_carlo_{tag}", odir)
    return [str(p) for p in paths]


if __name__ == "__main__":
    main()
