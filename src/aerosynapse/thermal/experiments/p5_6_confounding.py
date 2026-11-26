"""Step 5.6: confounding cases — when k and h are separable, when they are not, and what helps; reported honestly.

Cases (studies.confounding), each analysed with the Cramer-Rao bound (rank, condition number, collinearity index),
least squares (multi-start, best chi2), the PROFILE LIKELIHOOD of every unknown (95 % intervals that may be open on one
or both sides) and, for full-rank cases, a Monte Carlo of least-squares fits:
  configured   the configured problem — YOUR data file if data.source = file (profiles on your data)
  q_vol sweep  source strength 0 ... 1e6: without a source only Bi = hL/k is identifiable
  h sweep      Biot number from the lumped limit to the fixed-surface-temperature limit
  known        prior knowledge: one of k, h fixed (unknown subsets) at no / weak source
  priors       Bayesian: a log-normal prior on one parameter (a handbook value, deliberately off by offset_sd sd)
  pinn         the Step 5.2 PINN with fewer unknowns on the configured data: confounding vs the PINN's own limits
Every case other than the configured one is a synthetic what-if at the configured (nominal) parameters.

    python experiments/p5_6_confounding.py [--config my.yaml] [--set key=value] [--tag name] [--figures-only]
"""

from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np

from aerosynapse.thermal.experiments import p5_3_classical as P53
from aerosynapse.thermal.experiments import p5_4_robustness as P54
from aerosynapse.thermal.inverse import body as B
from aerosynapse.thermal.inverse import classical as C
from aerosynapse.thermal.inverse import data as D
from aerosynapse.thermal.inverse import profile as PR
from aerosynapse.thermal.inverse import spec as IS
from aerosynapse.thermal.inverse import uq as U
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz import inverse_figs as VF
from aerosynapse.thermal.viz.style import save_fig

MOD = "aerosynapse.thermal.experiments.p5_6_confounding"


def _what_if(spec: dict, level: float) -> dict:
    s = copy.deepcopy(spec)
    s["data"] |= {"source": "synthetic", "compare_to_parameters": True, "level": float(level)}
    return s


def build_cases(base: dict, cfg: dict) -> list[dict]:
    """The case list: [{group, name, label, spec, priors}]; specs are complete (validated base + edits)."""
    lv = float(cfg["level"])
    out = [
        {
            "group": "configured",
            "name": "configured",
            "label": "configured problem",
            "spec": copy.deepcopy(base),
            "priors": None,
        }
    ]
    if base["data"]["source"] == "synthetic":
        out[0]["spec"]["data"]["level"] = lv
    nominal = _what_if(base, lv)
    for v in cfg["sweeps"].get("q_vol") or []:
        s = copy.deepcopy(nominal)
        s["body"]["q_vol"] = float(v)
        out.append({"group": "q_vol", "name": f"q_vol={v:g}", "label": f"{v:g}", "spec": s, "priors": None})
    for v in cfg["sweeps"].get("h") or []:
        s = copy.deepcopy(nominal)
        f = float(v) / float(s["parameters"]["h"])
        s["parameters"]["h"] = float(v)
        s["initial_guess"]["h"] = float(s["initial_guess"]["h"]) * f  # the guess keeps its ratio to the truth
        out.append({"group": "h", "name": f"h={v:g}", "label": f"{v:g}", "spec": s, "priors": None})
    for q in cfg.get("known_at_q_vol") or []:
        for name, unk in (cfg.get("known") or {}).items():
            s = copy.deepcopy(nominal)
            s["body"]["q_vol"] = float(q)
            s["unknown"] = list(unk)
            out.append(
                {
                    "group": "known",
                    "name": f"{name}, q_vol={q:g}",
                    "label": f"{name.replace('_', ' ')}, q {q:g}",
                    "spec": s,
                    "priors": None,
                }
            )
    pr = cfg.get("priors")
    if pr:
        par = pr["parameter"]
        for sd in pr["sd"]:
            s = copy.deepcopy(nominal)
            s["body"]["q_vol"] = float(pr["q_vol"])
            mean = float(np.log(float(s["parameters"][par])) + float(pr["offset_sd"]) * float(sd))
            out.append(
                {
                    "group": "priors",
                    "name": f"prior {par} sd={sd:g}",
                    "label": f"{100 * sd:g} %",
                    "spec": s,
                    "priors": {par: [mean, float(sd)]},
                }
            )
    return out


def best_fit(spec: dict, data: dict, unknown: list[str], starts: int, seed: int) -> dict:
    """Least squares from the guess and starts - 1 Sobol starts (x4 spread); the best chi2, refitted with covariance."""
    guess = {k: float(spec["initial_guess"][k]) for k in unknown}
    fits = C.multistart(spec, data, unknown, guess, max(int(starts), 1), 4.0, seed, "analytic")
    ok = [f for f in fits if f["estimates"] is not None]
    best = min(ok, key=lambda f: f["chi2_per_reading"])
    f = C.fit(spec, data, unknown, {k: best["estimates"][k] for k in unknown}, "analytic")
    f["starts_at_best"] = int(
        sum(abs(x["chi2_per_reading"] - best["chi2_per_reading"]) * len(data["T"]) < 0.1 for x in ok)
    )
    f["starts"] = len(fits)
    return f


def case_task(spec: dict, cfg: dict, priors: dict | None) -> dict:
    """Bound, best least-squares fit, profile likelihood of every unknown, (priors) posterior."""
    t0 = time.perf_counter()
    unknown = list(spec["unknown"])
    d = D.load(spec)
    lv = float(spec["data"]["level"])
    b = P53.bound(spec, lv)
    f = best_fit(spec, d, unknown, int(cfg["starts"]), 0)
    pc = cfg["profile"]
    prof = {
        k: PR.profile(spec, d, unknown, f, k, float(pc["span"]), int(pc["fine"]), int(pc["coarse"])) for k in unknown
    }
    for p in prof.values():
        p.pop("nuisance")
    out = {
        "unknown": unknown,
        "n_readings": len(d["T"]),
        "sigma_mean_K": float(np.mean(d["sigma"])),
        "crlb_log": dict(zip(unknown, b["crlb_log"], strict=True)),
        "rank": b["rank"],
        "condition_number": b["condition_number"],
        "collinearity_index": b["collinearity_index"],
        "null_directions": b["null_directions"],
        "fit": {k: f[k] for k in ("estimates", "sd_log", "chi2_per_reading", "rank", "starts", "starts_at_best")},
        "bi_sd_log": C.log_sd(f, unknown, {"h": 1.0, "k": -1.0}) if {"h", "k"} <= set(unknown) else None,
        "profiles": prof,
        "parameters": {k: float(spec["parameters"][k]) for k in B.PARAMS},
        "groups": IS.body_of(spec).groups({k: float(spec["parameters"][k]) for k in B.PARAMS}),
        "groups_estimate": IS.body_of(spec).groups(f["estimates"]),
    }
    if priors:
        # chain start: the point of the likelihood ridge where the prior parameters sit at their prior means (the other
        # unknowns re-fitted) -- on a confounded valley the least-squares point can lie anywhere, far outside the prior
        fixed = {k: float(np.exp(v[0])) for k, v in priors.items() if k in unknown}
        rest = [k for k in unknown if k not in fixed]
        s0 = copy.deepcopy(spec)
        s0["parameters"].update(fixed)
        start = dict(fixed)
        if rest:
            g = C.fit(s0, d, rest, {k: f["estimates"][k] for k in rest}, "analytic", with_cov=False)
            start |= {k: g["estimates"][k] for k in rest}
        f0 = C.covariance_at(spec, d, unknown, start)
        mc = spec["uq"]["mcmc"]
        post = U.posterior(
            spec,
            d,
            unknown,
            f0,
            mc["steps"],
            mc["burn"],
            mc["adapt_every"],
            mc["prior_factor"],
            11,
            "analytic",
            {k: tuple(v) for k, v in priors.items()},
        )
        S = post["samples_log"]
        out["posterior"] = {
            "priors": priors,
            "median_log": dict(zip(unknown, np.median(S, 0).tolist(), strict=True)),
            "ci95_log": {k: np.quantile(S[:, j], [0.025, 0.975]).tolist() for j, k in enumerate(unknown)},
            "sd_log": dict(zip(unknown, S.std(0, ddof=1).tolist(), strict=True)),
            "acceptance": post["acceptance"],
            "ess": post["ess"],
        }
    out["wall_s"] = time.perf_counter() - t0
    return out


def mc_task(spec: dict, seeds: list[int], level: float) -> list[dict]:
    """Least-squares fits (from the guess, analytic model) of independent synthetic noise draws."""
    unknown = list(spec["unknown"])
    guess = {k: float(spec["initial_guess"][k]) for k in unknown}
    out = []
    for sd in seeds:
        r = C.safe_fit(spec, D.synthetic(spec, level, sd), unknown, guess, "analytic")
        out.append(
            {k: r.get(k) for k in ("estimates", "sd_log", "success", "chi2_per_reading", "n_model_calls")}
            | {"seed": sd}
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--figures-only", action="store_true", help="redraw the figures and the table from saved metrics")
    a = ap.parse_args()
    base = IS.load_spec(a.config or None, IS.parse_set(a.sets))
    cfg = base["studies"]["confounding"]
    custom = bool(a.config or a.sets or a.tag != "default")
    odir = ROOT / "results" / f"p5_6_confounding_{a.tag}"
    table = odir / "confounding.md" if custom else ROOT / "docs" / "tables" / "phase5_6_confounding.md"
    VF.style()
    if a.figures_only:
        met = P53._restore(json.loads((odir / "metrics.json").read_text(encoding="utf-8")))
        figures(met, a.tag, odir)
        report(met, base, table, a.tag)
        print(f"wrote {table}")
        return
    with Experiment(f"p5_6_confounding_{a.tag}", base) as ex:
        cases = build_cases(base, cfg)
        jobs, tags = [], []
        rp = cfg.get("pinn")
        if rp:
            variant = next(v for v in base["studies"]["pinn_inverse"]["variants"] if v["name"] == rp["variant"])
            for unk in rp["unknown_sets"]:
                for sd in rp["seeds"]:
                    s = copy.deepcopy(cases[0]["spec"])
                    s["unknown"] = list(unk)
                    jobs.append((P54.MOD, "pinn_task", dict(spec=s, variant=variant, seed=int(sd))))
                    tags.append(("pinn", (tuple(unk), int(sd))))
        for i, c in enumerate(cases):
            jobs.append((MOD, "case_task", dict(spec=c["spec"], cfg=cfg, priors=c["priors"])))
            tags.append(("case", i))
        print(f"{len(cases)} cases, {len(jobs)} tasks", flush=True)
        outs = run_tasks(
            jobs,
            base["execution"],
            lambda i, r, sec: print(
                f"[{i + 1}/{len(jobs)}] {tags[i][0]} {tags[i][1] if tags[i][0] == 'pinn' else cases[tags[i][1]]['name']}: {sec:.0f} s",
                flush=True,
            ),
        )
        res, pinn = {}, []
        for (kind, key), r in zip(tags, outs, strict=True):
            if kind == "case":
                res[key] = r
            else:
                pinn.append(r | {"unknown": list(key[0])})
        # Monte Carlo for the synthetic cases with a full-rank Fisher matrix (second pass: needs the ranks)
        R, bsz = int(cfg["replicates"]), int(cfg["batch"])
        jobs, labels = [], []
        for i, c in enumerate(cases):
            if (
                R > 0
                and c["spec"]["data"]["source"] == "synthetic"
                and c["priors"] is None
                and res[i]["rank"] == len(res[i]["unknown"])
            ):
                seeds = list(range(1000, 1000 + R))
                for j in range(0, R, bsz):
                    jobs.append(
                        (
                            MOD,
                            "mc_task",
                            dict(spec=c["spec"], seeds=seeds[j : j + bsz], level=float(c["spec"]["data"]["level"])),
                        )
                    )
                    labels.append(i)
        outs = run_tasks(
            jobs,
            base["execution"],
            lambda i, r, sec: print(f"[mc {i + 1}/{len(jobs)}] {cases[labels[i]]['name']}: {sec:.0f} s", flush=True),
        )
        mcs: dict = {}
        for lab, o in zip(labels, outs, strict=True):
            mcs.setdefault(lab, []).extend(o)
        rows = []
        for i, c in enumerate(cases):
            r = (
                res[i]
                | {k: c[k] for k in ("group", "name", "label")}
                | {
                    "data_source": c["spec"]["data"]["source"],
                    "known_truth": bool(c["spec"]["data"].get("compare_to_parameters", True)),
                    "level": float(c["spec"]["data"]["level"]),
                }
            )
            if i in mcs:
                cr = r["crlb_log"]
                r["mc"] = P54.mc_stats(mcs[i], c["spec"], cr, r["n_readings"], 10.0)
            rows.append(r)
            ex.log(case=c["name"], rank=r["rank"], **{f"est_{k}": v for k, v in r["fit"]["estimates"].items()})
        ex.metrics.update({"rows": rows, "pinn": pinn, "git": git_hash()})
        ex.metrics["figures"] = figures(ex.metrics, a.tag, ex.dir)
        report(ex.metrics, base, table, a.tag)
        ex.metrics["table"] = str(table.relative_to(ROOT))
    print(f"wrote {table}")


def _ci_cell(r: dict, k: str) -> str:
    """Profile 95 % interval: % of the truth, or (no truth: your data) values; 'open' sides marked."""
    lo, hi = r["profiles"][k]["ci95_log"]
    if r["known_truth"]:
        xt = np.log(r["parameters"][k])
        f = lambda v, s: ("−∞" if s < 0 else "+∞") if v is None else f"{100 * (np.exp(v - xt) - 1):+.3g}"  # noqa: E731
        return f"[{f(lo, -1)}, {f(hi, 1)}] %"
    f = lambda v, s: ("0" if s < 0 else "∞") if v is None else f"{np.exp(v):.4g}"  # noqa: E731
    return f"[{f(lo, -1)}, {f(hi, 1)}]"


def _err_cell(r: dict, k: str) -> str:
    if r["profiles"][k]["ci95_log"] == [None, None]:
        return "not identifiable"  # any value along the flat valley fits equally well: no estimate to report
    e = r["fit"]["estimates"][k]
    return f"{100 * (e / r['parameters'][k] - 1):+.3g} %" if r["known_truth"] else f"{e:.4g}"


def _f(v, fmt=".3g"):
    return "∞" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)


GROUP_TITLE = {
    "configured": "Configured problem",
    "q_vol": "Source strength q_vol [W/m³] (synthetic what-ifs)",
    "h": "Heat-transfer coefficient h [W/(m² K)] — Biot number (synthetic what-ifs)",
    "known": "Prior knowledge: one parameter known (synthetic what-ifs)",
    "priors": "Bayesian prior on one parameter, mean off by offset_sd × sd (synthetic what-ifs)",
}


def report(met: dict, spec: dict, path, tag: str) -> None:
    cfg = spec["studies"]["confounding"]
    L = [
        f"# Step 5.6 confounding cases — `{tag}` (auto-generated by experiments/p5_6_confounding.py)",
        "",
        "Per case: Cramér–Rao bound (1σ, %), rank / condition number / collinearity index of the sensitivities, the best "
        f"least-squares fit of {cfg['starts']} starts (error %), and the PROFILE-LIKELIHOOD 95 % interval of each unknown "
        f"(Δχ² = 3.84 over [p/{cfg['profile']['span']:g}, p×{cfg['profile']['span']:g}]; ±∞ = the data cannot bound that side); "
        f"Monte Carlo ({cfg['replicates']} fits, analytic model) for full-rank synthetic cases. Q = q L²/(k ΔT), Bi = hL/k.",
        "",
    ]
    rows = met["rows"]
    for g in dict.fromkeys(r["group"] for r in rows):
        sel = [r for r in rows if r["group"] == g]
        if g == "priors":
            continue
        L += [
            f"## {GROUP_TITLE.get(g, g)}",
            "",
            "| case | Bi | Q | rank | cond. | collin. | unknown: CRLB % / LS error / profile 95 % | MC std/CRLB, median abs err % | Bi 1σ % |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sel:
            cells = "<br>".join(
                f"{k}: {_f(100 * r['crlb_log'][k])} / {_err_cell(r, k)} / {_ci_cell(r, k)}" for k in r["unknown"]
            )
            mc = r.get("mc") or {}
            mcs = (
                "<br>".join(f"{k}: {mc['std_over_crlb'][k]:.2f}, {mc['median_abs_pct'][k]:.3g}" for k in r["unknown"])
                if mc.get("std_over_crlb")
                else "—"
            )
            L.append(
                f"| {r['label']}{' (your data; Bi, Q of the estimate; CRLB at the nominal parameters)' if r['data_source'] == 'file' else ''} | "
                f"{(r['groups_estimate'] if r['data_source'] == 'file' else r['groups'])['Bi']:.3g} | "
                f"{(r['groups_estimate'] if r['data_source'] == 'file' else r['groups'])['Q']:.3g} | {r['rank']}/{len(r['unknown'])} | "
                f"{_f(r['condition_number'])} | {_f(r['collinearity_index'])} | {cells} | {mcs} | {'—' if r['bi_sd_log'] is None else _f(100 * r['bi_sd_log'])} |"
            )
        L.append("")
    pri = [r for r in rows if r["group"] == "priors"]
    if pri:
        par = cfg["priors"]["parameter"]
        L += [
            f"## {GROUP_TITLE['priors']} ({par}, q_vol {cfg['priors']['q_vol']:g}, offset {cfg['priors']['offset_sd']:g} sd)",
            "",
            "| prior sd | "
            + " | ".join(f"{k}: posterior median error % [95 %]" for k in pri[0]["unknown"])
            + " | acceptance | ESS min |",
            "|---|" + "---|" * len(pri[0]["unknown"]) + "---|---|",
        ]
        for r in pri:
            p = r["posterior"]
            cells = []
            for k in r["unknown"]:
                xt = np.log(r["parameters"][k])
                lo, hi = p["ci95_log"][k]
                cells.append(
                    f"{100 * (np.exp(p['median_log'][k] - xt) - 1):+.3g} [{100 * (np.exp(lo - xt) - 1):+.3g}, {100 * (np.exp(hi - xt) - 1):+.3g}]"
                )
            L.append(f"| {r['label']} | " + " | ".join(cells) + f" | {p['acceptance']:.2f} | {min(p['ess']):.0f} |")
        L.append("")
    if met.get("pinn"):
        L += [
            f"## PINN with fewer unknowns (Step 5.2 variant `{cfg['pinn']['variant']}`, configured data)",
            "",
            "| unknown | seed | error % of each unknown | Bi / true | PINN data RMSE K | exact model at its estimates, RMSE K | time s |",
            "|---|---|---|---|---|---|---|",
        ]
        for p in met["pinn"]:
            L.append(
                f"| {', '.join(p['unknown'])} | {p['seed']} | "
                + ", ".join(f"{k} {100 * v:+.3g}" for k, v in p["rel_err"].items())
                + f" | {p['Bi_ratio']:.3f} | {p['data_rmse_K']:.3f} | {p['model_rmse_K']:.3f} | {p['wall_s']:.0f} |"
            )
        L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(met: dict, tag: str, odir) -> list[str]:
    VF.style()
    rows = met["rows"]
    paths = []
    show = [
        r
        for r in rows
        if r["group"] in ("configured", "q_vol") or (r["group"] == "h" and r["label"] in ("3", "100000"))
    ]
    if show:
        # every profile relative to its own estimate (a confounded valley has no meaningful point estimate)
        prof = [
            {
                "label": "configured"
                if r["group"] == "configured"
                else f"{'q_vol' if r['group'] == 'q_vol' else 'h'} = {r['label']}",
                "unknown": r["unknown"],
                "profiles": {
                    k: {
                        "x": (np.asarray(r["profiles"][k]["grid_log"]) - np.log(r["fit"]["estimates"][k])).tolist(),
                        "d": r["profiles"][k]["delta_chi2"],
                    }
                    for k in r["unknown"]
                },
            }
            for r in show
        ]
        paths += save_fig(VF.profiles(prof, "est"), f"p5_6_profiles_{tag}", odir)
    sweeps = {}
    for g, xkey in (("q_vol", "Q"), ("h", "Bi")):
        sel = [r for r in rows if r["group"] == g]
        if sel:
            sweeps[g] = {
                "x": [r["groups"][xkey] for r in sel],
                "xlabel": "source group Q = q L²/(k ΔT)" if g == "q_vol" else "Biot number Bi = hL/k",
                "crlb": {k: [100 * r["crlb_log"][k] for r in sel] for k in sel[0]["unknown"]},
                "profile": {k: [_halfwidth(r, k) for r in sel] for k in sel[0]["unknown"]},
                "bi": [100 * r["bi_sd_log"] if r["bi_sd_log"] is not None else np.nan for r in sel],
            }
    if sweeps:
        paths += save_fig(VF.identifiability_map(sweeps), f"p5_6_map_{tag}", odir)
    pri = [r for r in rows if r["group"] == "priors"]
    known = []  # per source strength: all three unknown (reference), then each known-parameter case
    for r in (r for r in rows if r["group"] == "known"):
        q = r["name"].split("q_vol=")[-1]
        ref = next((x for x in rows if x["group"] == "q_vol" and x["name"] == f"q_vol={q}"), None)
        if ref is not None and ref["name"] not in {x["name"] for x in known}:
            known.append(ref | {"label": f"all unknown, q {q}"})
        known.append(r)
    if pri or known:
        paths += save_fig(VF.prior_knowledge(known, pri), f"p5_6_prior_knowledge_{tag}", odir)
    return [str(p) for p in paths]


def _halfwidth(r: dict, k: str) -> float:
    """Half-width of the profile 95 % interval in % (exp of the ln half-width, minus 1); inf if open on a side."""
    lo, hi = r["profiles"][k]["ci95_log"]
    if lo is None or hi is None:
        return float("inf")
    return 100 * (np.exp((hi - lo) / 2) - 1)


if __name__ == "__main__":
    main()
