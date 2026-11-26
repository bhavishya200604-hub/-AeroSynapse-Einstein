"""Step 5.5: parameter uncertainty quantification.

For every case of studies.uq.cases (overrides of the base problem):
  data-seed analysis  one realisation: least squares + Gauss-Newton (Laplace) intervals, parametric and residual
                      bootstrap (uq.bootstrap.replicates), Bayesian posterior by adaptive Metropolis (uq.mcmc); the FDM
                      Gauss-Newton fit as the estimator reference; samples kept for the figures
  calibration         studies.uq.datasets independent noise draws (seeds 1000...): every method on each, coverage of
                      the uq.levels intervals for alpha, k, h and the Biot number (ln Bi = ln h - ln k), interval
                      widths, and the bias / RMSE of the point estimates (least squares, bootstrap bias-corrected,
                      posterior mean and median)
PINN deep ensemble (studies.uq.pinn): one Step 5.2 variant over several seeds on the base data; members found in a
saved Step 5.2 run are reused (bit-identical, R5.4m), the others trained; ensemble mean +- t sd vs the truth.

    python experiments/p5_5_uq.py [--config my.yaml] [--set key=value] [--tag name] [--figures-only]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import t as student_t

from aerosynapse.thermal.experiments import p5_3_classical as P53
from aerosynapse.thermal.experiments import p5_4_robustness as P54
from aerosynapse.thermal.inverse import body as B
from aerosynapse.thermal.inverse import classical as C
from aerosynapse.thermal.inverse import data as D
from aerosynapse.thermal.inverse import spec as IS
from aerosynapse.thermal.inverse import uq as U
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz import inverse_figs as VF
from aerosynapse.thermal.viz.style import save_fig

MOD = "aerosynapse.thermal.experiments.p5_5_uq"
METHODS = ("gauss_newton", "bootstrap_parametric", "bootstrap_residual", "posterior")
POINTS = ("least_squares", "bias_corrected", "posterior_mean", "posterior_median")


def quantities(unknown: list[str]) -> list[str]:
    return list(unknown) + (["Bi"] if {"h", "k"} <= set(unknown) else [])


def with_bi(X: np.ndarray, unknown: list[str]) -> np.ndarray:
    """Append ln(h/k) (= ln Bi up to the constant ln L) to rows of ln p."""
    X = np.atleast_2d(X)
    if {"h", "k"} <= set(unknown):
        X = np.column_stack([X, X[:, unknown.index("h")] - X[:, unknown.index("k")]])
    return X


def truth_log(spec: dict) -> np.ndarray:
    unknown = list(spec["unknown"])
    return with_bi(np.log([[float(spec["parameters"][k]) for k in unknown]]), unknown)[0]


def analyse(spec: dict, data: dict, seed: int, n_boot: int, keep: bool) -> dict:
    """Every UQ method on one data set (ln p; quantities include ln Bi)."""
    unknown = list(spec["unknown"])
    q = quantities(unknown)
    cfg = spec["uq"]
    model = cfg["model"]
    t0 = time.perf_counter()
    f = C.fit(spec, data, unknown, {k: float(spec["initial_guess"][k]) for k in unknown}, model)
    x_hat = with_bi(np.log([[f["estimates"][k] for k in unknown]]), unknown)[0]
    sd = [f["sd_log"][k] for k in unknown] + ([C.log_sd(f, unknown, {"h": 1.0, "k": -1.0})] if "Bi" in q else [])
    boots = {
        kind: U.bootstrap(spec, data, unknown, f, n_boot, kind, 7 * seed + i + 1, model)
        for i, kind in enumerate(cfg["bootstrap"]["kinds"])
    }
    mc = cfg["mcmc"]
    post = U.posterior(
        spec, data, unknown, f, mc["steps"], mc["burn"], mc["adapt_every"], mc["prior_factor"], 7 * seed + 5, model
    )
    samples = {f"bootstrap_{k}": with_bi(b["samples_log"], unknown) for k, b in boots.items()}
    samples["posterior"] = with_bi(post["samples_log"], unknown)
    intervals = {}
    for lv in cfg["levels"]:
        z = float(U.norm.ppf(0.5 + lv / 2))
        iv = {
            "gauss_newton": {qq: [float(x_hat[j] - z * sd[j]), float(x_hat[j] + z * sd[j])] for j, qq in enumerate(q)}
        }
        for m, S in samples.items():
            iv[m] = U.percentile_interval(S, q, lv)
        intervals[str(lv)] = iv
    points = {
        "least_squares": x_hat.tolist(),
        "posterior_mean": samples["posterior"].mean(0).tolist(),
        "posterior_median": np.median(samples["posterior"], 0).tolist(),
    }
    if "bootstrap_parametric" in samples:
        points["bias_corrected"] = (2 * x_hat - samples["bootstrap_parametric"].mean(0)).tolist()
    out = {
        "seed": seed,
        "fit": {k: f[k] for k in ("estimates", "sd_log", "chi2_per_reading", "rank", "n_model_calls")},
        "sd_gn": sd,
        "intervals": intervals,
        "points": points,
        "sd": {m: S.std(0, ddof=1).tolist() for m, S in samples.items()} | {"gauss_newton": sd},
        "mcmc": {
            k: post[k] for k in ("acceptance", "tau", "ess", "evaluations", "at_prior_edge", "bounds_log", "wall_s")
        },
        "bootstrap": {k: {"failed": b["failed"], "wall_s": b["wall_s"]} for k, b in boots.items()},
        "wall_s": time.perf_counter() - t0,
    }
    if keep:
        th = max(1, len(samples["posterior"]) // 3000)
        out["samples"] = {m: S[:: th if m == "posterior" else 1].tolist() for m, S in samples.items()}
        out["gn_cov_log"] = np.linalg.pinv(np.asarray(f["fisher"], float)).tolist()
    return out


def seed_task(spec: dict) -> dict:
    """The data-seed realisation with every method (samples kept) and the FDM Gauss-Newton fit as reference."""
    d = D.load(spec)
    unknown = list(spec["unknown"])
    r = analyse(spec, d, int(spec["data"]["seed"]), int(spec["uq"]["bootstrap"]["replicates"]), True)
    ff = C.fit(spec, d, unknown, {k: float(spec["initial_guess"][k]) for k in unknown}, "fdm")
    r["fdm_fit"] = {"estimates": ff["estimates"], "sd_log": ff["sd_log"], "wall_s": ff["wall_s"]}
    r["n_readings"] = len(d["T"])
    r["sigma_mean_K"] = float(np.mean(d["sigma"]))
    return r


def dataset_task(spec: dict, seeds: list[int], n_boot: int) -> list[dict]:
    return [analyse(spec, D.synthetic(spec, float(spec["data"]["level"]), sd), sd, n_boot, False) for sd in seeds]


def calibration(results: list[dict], spec: dict) -> dict:
    """Coverage of every method / level / quantity over the datasets; median 95 % width; point-estimate bias, RMSE."""
    q = quantities(list(spec["unknown"]))
    xt = truth_log(spec)
    n = len(results)
    cov = {}
    for m in METHODS:
        if m not in results[0]["intervals"][next(iter(results[0]["intervals"]))]:
            continue
        cov[m] = {}
        for lv in results[0]["intervals"]:
            hit = np.array(
                [
                    [r["intervals"][lv][m][qq][0] <= xt[j] <= r["intervals"][lv][m][qq][1] for j, qq in enumerate(q)]
                    for r in results
                ]
            )
            cov[m][lv] = dict(zip(q, hit.mean(0).tolist(), strict=True))
    width = (
        {
            m: dict(
                zip(
                    q,
                    np.median(
                        [
                            [r["intervals"]["0.95"][m][qq][1] - r["intervals"]["0.95"][m][qq][0] for qq in q]
                            for r in results
                        ],
                        0,
                    ).tolist(),
                    strict=True,
                )
            )
            for m in cov
        }
        if "0.95" in results[0]["intervals"]
        else {}
    )
    pts = {}
    for p in POINTS:
        if p not in results[0]["points"]:
            continue
        E = np.array([r["points"][p] for r in results]) - xt
        pts[p] = {
            "bias_log": dict(zip(q, E.mean(0).tolist(), strict=True)),
            "rmse_log": dict(zip(q, np.sqrt((E**2).mean(0)).tolist(), strict=True)),
            "bias_se_log": dict(zip(q, (E.std(0, ddof=1) / np.sqrt(n)).tolist(), strict=True)),
        }
    ess = np.array([r["mcmc"]["ess"] for r in results])
    return {
        "datasets": n,
        "coverage": cov,
        "coverage_halfwidth95": {
            lv: float(1.96 * np.sqrt(float(lv) * (1 - float(lv)) / n)) for lv in results[0]["intervals"]
        },
        "width95_log": width,
        "points": pts,
        "mcmc": {
            "ess_min": float(ess.min()),
            "ess_median": float(np.median(ess)),
            "acceptance_median": float(np.median([r["mcmc"]["acceptance"] for r in results])),
            "at_prior_edge_mean": {
                k: float(np.mean([r["mcmc"]["at_prior_edge"][k] for r in results])) for k in spec["unknown"]
            },
        },
        "bootstrap_failed": {
            k: int(sum(r["bootstrap"][k]["failed"] for r in results)) for k in results[0]["bootstrap"]
        },
        "wall_s_median": float(np.median([r["wall_s"] for r in results])),
    }


def pinn_member(est: dict, spec: dict, data: dict, ref_log: np.ndarray | None = None) -> dict:
    """Derived quantities of a PINN estimate on the same data: ln-ratios (with Bi) to the truth, or to ref_log (your
    data: the least-squares estimate), and the misfit of the exact model at the PINN's estimates."""
    unknown = list(spec["unknown"])
    ref = truth_log(spec) if ref_log is None else np.asarray(ref_log)
    x = with_bi(np.log([[float(est[k]) for k in unknown]]), unknown)[0] - ref
    nt = max(B.truncation_terms(float(np.min(data["t"])) * float(est["alpha"]) / float(spec["body"]["L"]) ** 2), 20)
    T = D.model_at(est, spec, data["r"], data["t"], nt)
    return {"estimates": est, "log_ratio": x.tolist(), "model_rmse_K": float(np.sqrt(np.mean((T - data["T"]) ** 2)))}


def same_problem(saved_config, spec: dict) -> bool:
    """True if a saved run solved the same problem (body, parameters, unknowns, guess, data, measurement, PINN
    settings and variants); settings added since then are compared at their default values."""
    import yaml

    saved = yaml.safe_load(saved_config.read_text(encoding="utf-8"))
    dflt = IS.load_spec()
    for key in ("body", "parameters", "unknown", "initial_guess", "data", "measurement", "pinn"):
        a, b = saved.get(key), spec.get(key)
        if isinstance(b, dict):
            a = IS.deep_merge(dflt.get(key) or {}, a or {})
            a, b = ({k: v for k, v in x.items() if k != "level_from_file"} for x in (a, b))
        if json.dumps(a, sort_keys=True, default=str) != json.dumps(b, sort_keys=True, default=str):
            return False
    va = (saved.get("studies") or {}).get("pinn_inverse", {}).get("variants")
    return json.dumps(va, sort_keys=True) == json.dumps(spec["studies"]["pinn_inverse"]["variants"], sort_keys=True)


def ensemble(members: list[dict], q: list[str]) -> dict:
    X = np.array([m["log_ratio"] for m in members])
    M = len(X)
    mean, sd = X.mean(0), X.std(0, ddof=1) if M > 1 else np.full(X.shape[1], np.nan)
    tq = float(student_t.ppf(0.975, M - 1)) if M > 1 else float("nan")
    return {
        "members": M,
        "mean_log_ratio": dict(zip(q, mean.tolist(), strict=True)),
        "sd_log": dict(zip(q, sd.tolist(), strict=True)),
        "t_quantile": tq,
        "interval95_log_ratio": {
            qq: [float(mean[j] - tq * sd[j]), float(mean[j] + tq * sd[j])] for j, qq in enumerate(q)
        },
        "covers_truth": {qq: bool(mean[j] - tq * sd[j] <= 0 <= mean[j] + tq * sd[j]) for j, qq in enumerate(q)},
        "z": {qq: float(mean[j] / sd[j]) if sd[j] > 0 else float("inf") for j, qq in enumerate(q)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--figures-only", action="store_true", help="redraw the figures and the table from saved metrics")
    a = ap.parse_args()
    cli = IS.parse_set(a.sets)
    base = IS.load_spec(a.config or None, cli)
    su = base["studies"]["uq"]
    custom = bool(a.config or a.sets or a.tag != "default")
    odir = ROOT / "results" / f"p5_5_uq_{a.tag}"
    table = odir / "uq.md" if custom else ROOT / "docs" / "tables" / "phase5_5_uq.md"
    VF.style()
    if a.figures_only:
        met = P53._restore(json.loads((odir / "metrics.json").read_text(encoding="utf-8")))
        figures(met, base, a.tag, odir)
        report(met, base, table, a.tag)
        print(f"wrote {table}")
        return
    with Experiment(f"p5_5_uq_{a.tag}", base) as ex:
        specs = {c: IS.load_spec(a.config or None, IS.deep_merge(cli, ov or {})) for c, ov in su["cases"].items()}
        jobs, tags = [], []
        rp = su.get("pinn")
        pinn_spec = specs[next(iter(specs))]
        reused = {}
        if rp:
            variant = next(v for v in base["studies"]["pinn_inverse"]["variants"] if v["name"] == rp["variant"])
            src = ROOT / "results" / str(rp.get("reuse")) / "metrics.json"
            if rp.get("reuse") and src.exists() and same_problem(src.parent / "config.yaml", pinn_spec):
                m52 = json.loads(src.read_text(encoding="utf-8"))
                case0 = m52["records"][0]["shape"]
                for r in m52["records"]:
                    if r["shape"] == case0 and r["variant"] == rp["variant"] and r["seed"] in rp["seeds"]:
                        reused[int(r["seed"])] = r
            for sd in rp["seeds"]:
                if int(sd) not in reused:
                    jobs.append((P54.MOD, "pinn_task", dict(spec=pinn_spec, variant=variant, seed=int(sd))))
                    tags.append(("pinn", int(sd)))
        for c, s in specs.items():
            jobs.append((MOD, "seed_task", dict(spec=s)))
            tags.append(("seed", c))
        R, bsz = int(su["datasets"]), int(su["batch"])
        for c, s in specs.items():
            seeds = list(range(1000, 1000 + R))
            for i in range(0, R, bsz):
                jobs.append(
                    (
                        MOD,
                        "dataset_task",
                        dict(spec=s, seeds=seeds[i : i + bsz], n_boot=int(su["bootstrap_replicates"])),
                    )
                )
                tags.append(("cal", c))
        print(f"{len(jobs)} tasks ({len(reused)} PINN members reused)", flush=True)
        outs = run_tasks(
            jobs,
            base["execution"],
            lambda i, r, sec: print(f"[{i + 1}/{len(jobs)}] {tags[i][0]} {tags[i][1]}: {sec:.0f} s", flush=True),
        )
        seed_res, cal_res, pinn_new = {}, {c: [] for c in specs}, {}
        for (kind, key), r in zip(tags, outs, strict=True):
            if kind == "seed":
                seed_res[key] = r
            elif kind == "cal":
                cal_res[key] += r
            else:
                pinn_new[key] = r
        cases = {}
        for c, s in specs.items():
            known = bool(s["data"].get("compare_to_parameters", True))
            cases[c] = {
                "overrides": su["cases"][c],
                "quantities": quantities(list(s["unknown"])),
                "known_truth": known,
                "data_source": s["data"]["source"],
                "L": float(s["body"]["L"]),
                # reference of every ratio: the truth, or (your data) the least-squares estimate of this case
                "truth_log": (truth_log(s) if known else np.asarray(seed_res[c]["points"]["least_squares"])).tolist(),
                "seed": seed_res[c],
                "calibration": calibration(cal_res[c], s) if cal_res[c] else None,
            }
            for r in cal_res[c]:
                ex.log(
                    case=c,
                    seed=r["seed"],
                    **{
                        f"{p}_{qq}": v
                        for p in r["points"]
                        for qq, v in zip(cases[c]["quantities"], r["points"][p], strict=True)
                    },
                )
        pinn = None
        if rp:
            d0 = D.load(pinn_spec)
            members = []
            ref0 = np.asarray(cases[next(iter(specs))]["truth_log"])
            for sd in rp["seeds"]:
                est = reused[int(sd)]["estimates"] if int(sd) in reused else pinn_new[int(sd)]["estimates"]
                mm = pinn_member(est, pinn_spec, d0, ref0) | {"seed": int(sd), "reused": int(sd) in reused}
                mm["data_rmse_K"] = (reused[int(sd)] if int(sd) in reused else pinn_new[int(sd)])["data_rmse_K"]
                members.append(mm)
            q0 = quantities(list(pinn_spec["unknown"]))
            pinn = {
                "variant": rp["variant"],
                "case": next(iter(specs)),
                "members": members,
                "ensemble": ensemble(members, q0),
                "reuse": rp.get("reuse"),
                "new_wall_s": {str(k): v["wall_s"] for k, v in pinn_new.items()},
            }
        ex.metrics.update({"cases": cases, "pinn": pinn, "git": git_hash()})
        ex.metrics["figures"] = figures(ex.metrics, base, a.tag, ex.dir)
        report(ex.metrics, base, table, a.tag)
        ex.metrics["table"] = str(table.relative_to(ROOT))
    print(f"wrote {table}")


LABEL = {
    "gauss_newton": "Gauss–Newton (Laplace)",
    "bootstrap_parametric": "parametric bootstrap",
    "bootstrap_residual": "residual bootstrap",
    "posterior": "Bayesian posterior (MCMC)",
}
PLABEL = {
    "least_squares": "least squares",
    "bias_corrected": "bootstrap bias-corrected",
    "posterior_mean": "posterior mean",
    "posterior_median": "posterior median",
}


def _pct(lr: float) -> str:
    """ln-ratio -> signed % error (inf-safe)."""
    if not np.isfinite(lr):
        return "inf" if lr > 0 else "-100"
    v = 100 * (np.exp(lr) - 1)
    return f"{v:+.3g}" if abs(v) < 1e4 else f"{v:+.2e}"


def _cell(v: float, j: int, known: bool, xt: np.ndarray, q: list[str], L: float) -> str:
    """A ln-value as % error vs the truth, or (your data) as a value in physical units (Bi = hL/k)."""
    if known:
        return _pct(v - xt[j])
    return f"{np.exp(v) * (L if q[j] == 'Bi' else 1.0):.4g}"


def report(met: dict, spec: dict, path, tag: str) -> None:
    uq = spec["uq"]
    L = [
        f"# Step 5.5 parameter uncertainty — `{tag}` (auto-generated by experiments/p5_5_uq.py)",
        "",
        f"Least squares in ln p ({uq['model']} model; the FDM fit is shown for reference); Gauss–Newton (Laplace) intervals; "
        f"parametric and residual bootstrap ({uq['bootstrap']['replicates']} refits on the data-seed realisation, "
        f"{spec['studies']['uq']['bootstrap_replicates']} per calibration dataset), percentile intervals; Bayesian posterior "
        f"(adaptive Metropolis, Gaussian likelihood with the known σ, prior log-uniform on [guess/{uq['mcmc']['prior_factor']:g}, "
        f"guess×{uq['mcmc']['prior_factor']:g}]), equal-tailed credible intervals. Bi = hL/k. Errors and interval ends as % of the true value.",
        "",
    ]
    for c, d in met["cases"].items():
        q, sr, xt = d["quantities"], d["seed"], np.asarray(d["truth_log"])
        known = d.get("known_truth", True)
        what = "data-seed realisation" if d.get("data_source", "synthetic") == "synthetic" else "YOUR data file"
        L += [
            f"## {c} (overrides {d['overrides']}; {sr['n_readings']} readings, σ {sr['sigma_mean_K']:.3f} K)",
            "",
            f"{what[0].upper() + what[1:]} — 95 % intervals ("
            + ("% of the truth" if known else "values: α m²/s, k W/(m K), h W/(m² K), Bi")
            + "):",
            "",
            "| method | " + " | ".join(q) + " |",
            "|---|" + "---|" * len(q),
        ]
        for m in METHODS:
            if m in sr["intervals"]["0.95"]:
                iv = sr["intervals"]["0.95"][m]
                L.append(
                    f"| {LABEL[m]} | "
                    + " | ".join(
                        f"[{_cell(iv[qq][0], j, known, xt, q, d.get('L', 1.0))}, {_cell(iv[qq][1], j, known, xt, q, d.get('L', 1.0))}]"
                        for j, qq in enumerate(q)
                    )
                    + " |"
                )
        for p in POINTS:
            if p in sr["points"]:
                L.append(
                    f"| point: {PLABEL[p]} | "
                    + " | ".join(_cell(sr["points"][p][j], j, known, xt, q, d.get("L", 1.0)) for j in range(len(q)))
                    + " |"
                )
        ff = sr["fdm_fit"]
        L.append(
            "| FDM least squares (reference) | "
            + " | ".join(
                _cell(np.log(ff["estimates"][k]), j, known, xt, q, d.get("L", 1.0))
                for j, k in enumerate(spec["unknown"])
            )
            + (" | — |" if "Bi" in q else " |")
        )
        mc = sr["mcmc"]
        L += [
            "",
            f"MCMC: {mc['evaluations']} evaluations, acceptance {mc['acceptance']:.2f}, integrated autocorrelation time "
            + ", ".join(f"{t:.0f}" for t in mc["tau"])
            + ", ESS "
            + ", ".join(f"{e:.0f}" for e in mc["ess"])
            + f", fraction at the prior edge {', '.join(f'{k} {v:.3f}' for k, v in mc['at_prior_edge'].items())}; {mc['wall_s']:.0f} s. "
            + "Bootstrap failures: "
            + ", ".join(f"{k} {v['failed']}" for k, v in sr["bootstrap"].items())
            + ".",
            "",
        ]
        cal = d["calibration"]
        if cal:
            levels = list(next(iter(cal["coverage"].values())))
            L += [
                f"Calibration over {cal['datasets']} independent {'SYNTHETIC datasets at your nominal parameters (your file is not used here)' if d.get('data_source') == 'file' else 'datasets'} — empirical coverage % at nominal "
                + ", ".join(f"{100 * float(lv):g}" for lv in levels)
                + " % (binomial ±"
                + ", ".join(f"{100 * cal['coverage_halfwidth95'][lv]:.0f}" for lv in levels)
                + " %):",
                "",
                "| method | " + " | ".join(q) + " | median 95 % width (ln) |",
                "|---|" + "---|" * (len(q) + 1),
            ]
            for m, cv in cal["coverage"].items():
                L.append(
                    f"| {LABEL[m]} | "
                    + " | ".join(" / ".join(f"{100 * cv[lv][qq]:.0f}" for lv in levels) for qq in q)
                    + " | "
                    + ", ".join(f"{cal['width95_log'][m][qq]:.3g}" for qq in q)
                    + " |"
                )
            L += [
                "",
                "| point estimate | " + " | ".join(f"{qq}: bias ± s.e. / RMSE (ln)" for qq in q) + " |",
                "|---|" + "---|" * len(q),
            ]
            for p, st in cal["points"].items():
                L.append(
                    f"| {PLABEL[p]} | "
                    + " | ".join(
                        f"{st['bias_log'][qq]:+.4f} ± {st['bias_se_log'][qq]:.4f} / {st['rmse_log'][qq]:.4f}"
                        for qq in q
                    )
                    + " |"
                )
            m2 = cal["mcmc"]
            L += [
                "",
                f"MCMC over the datasets: ESS min {m2['ess_min']:.0f}, median {m2['ess_median']:.0f}; acceptance median {m2['acceptance_median']:.2f}; "
                f"fraction at the prior edge {', '.join(f'{k} {v:.3f}' for k, v in m2['at_prior_edge_mean'].items())}; bootstrap failures "
                + ", ".join(f"{k} {v}" for k, v in cal["bootstrap_failed"].items())
                + f"; median {cal['wall_s_median']:.0f} s per dataset.",
                "",
            ]
    p = met.get("pinn")
    if p:
        e = p["ensemble"]
        q = list(e["mean_log_ratio"])
        L += [
            f"## PINN deep ensemble (`{p['variant']}`, {e['members']} seeds, case {p['case']}; members reused from {p['reuse']}: "
            + ", ".join(str(m["seed"]) for m in p["members"] if m["reused"])
            + ")",
            "",
            "| member | "
            + " | ".join(f"{qq} error %" for qq in q)
            + " | data RMSE K | exact model at its estimates, RMSE K |",
            "|---|" + "---|" * (len(q) + 2),
        ]
        for m in p["members"]:
            L.append(
                f"| seed {m['seed']} | "
                + " | ".join(_pct(v) for v in m["log_ratio"])
                + f" | {m['data_rmse_K']:.3f} | {m['model_rmse_K']:.3f} |"
            )
        L.append("| ensemble mean | " + " | ".join(_pct(e["mean_log_ratio"][qq]) for qq in q) + " | — | — |")
        L.append(
            f"| ensemble 95 % interval (mean ± {e['t_quantile']:.2f} sd) | "
            + " | ".join(
                f"[{_pct(e['interval95_log_ratio'][qq][0])}, {_pct(e['interval95_log_ratio'][qq][1])}]" for qq in q
            )
            + " | — | — |"
        )
        L.append(
            f"| covers the {'truth' if met['cases'][p['case']].get('known_truth', True) else 'least-squares estimate'}? (z = mean / sd) | "
            + " | ".join(f"{'yes' if e['covers_truth'][qq] else 'no'} ({e['z'][qq]:.3g})" for qq in q)
            + " | — | — |"
        )
        L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(met: dict, spec: dict, tag: str, odir) -> list[str]:
    VF.style()
    cases = met["cases"]
    known = all(d.get("known_truth", True) for d in cases.values())
    ref_word = "truth" if known else "least-squares estimate"  # your data: ratios to the estimate
    paths = []
    rel = {}
    for c, d in cases.items():
        if d["calibration"]:
            cv = d["calibration"]["coverage"]
            rel[c.replace("_", " ")] = {
                m: {qq: [cv[m][lv][qq] for lv in cv[m]] for qq in d["quantities"]}
                | {"levels": [float(lv) for lv in cv[m]]}
                for m in cv
            }
    if rel:
        q = next(iter(cases.values()))["quantities"]
        paths += save_fig(VF.reliability(rel, q, {m: LABEL[m] for m in METHODS}), f"p5_5_reliability_{tag}", odir)
    iv = {}
    for c, d in cases.items():
        xt = np.asarray(d["truth_log"])
        s = d["seed"]["intervals"]
        iv[c.replace("_", " ")] = {
            m: {
                "95": {
                    qq: [s["0.95"][m][qq][0] - xt[j], s["0.95"][m][qq][1] - xt[j]]
                    for j, qq in enumerate(d["quantities"])
                },
                "68": {
                    qq: [s["0.68"][m][qq][0] - xt[j], s["0.68"][m][qq][1] - xt[j]]
                    for j, qq in enumerate(d["quantities"])
                }
                if "0.68" in s
                else None,
            }
            for m in s["0.95"]
        } | {"_point": {qq: d["seed"]["points"]["least_squares"][j] - xt[j] for j, qq in enumerate(d["quantities"])}}
    paths += save_fig(
        VF.uq_intervals(iv, next(iter(cases.values()))["quantities"], {m: LABEL[m] for m in METHODS}, ref_word),
        f"p5_5_intervals_{tag}",
        odir,
    )
    pairs = {}
    for c, d in cases.items():
        sm = d["seed"].get("samples")
        if sm:
            xt = np.asarray(d["truth_log"])[: len(spec["unknown"])]
            pairs[c.replace("_", " ")] = {
                "posterior": np.asarray(sm["posterior"])[:, : len(xt)] - xt,
                "bootstrap": np.asarray(sm.get("bootstrap_parametric", sm["posterior"]))[:, : len(xt)] - xt,
                "gn_cov": np.asarray(d["seed"]["gn_cov_log"]),
                "x_hat": np.asarray(d["seed"]["points"]["least_squares"])[: len(xt)] - xt,
            }
    if pairs:
        paths += save_fig(
            VF.posterior_pairs(pairs, list(spec["unknown"]), "true" if known else "estimate"),
            f"p5_5_posterior_{tag}",
            odir,
        )
    p = met.get("pinn")
    if p:
        c0 = cases[p["case"]]
        xt = np.asarray(c0["truth_log"])
        ls = {
            qq: [
                c0["seed"]["intervals"]["0.95"]["posterior"][qq][0] - xt[j],
                c0["seed"]["intervals"]["0.95"]["posterior"][qq][1] - xt[j],
            ]
            for j, qq in enumerate(c0["quantities"])
        }
        paths += save_fig(VF.pinn_ensemble(p, ls, ref_word), f"p5_5_pinn_ensemble_{tag}", odir)
    return [str(x) for x in paths]


if __name__ == "__main__":
    main()
