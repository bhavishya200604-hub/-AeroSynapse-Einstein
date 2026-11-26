"""Step 5.4: noise robustness (sigma = 0, 1, 2, 5, 10 %), sensor number and sensor placement.

Estimator: the Step 5.3 least squares. Per setting (studies.robustness):
  bound        Cramer-Rao bound of the unknown parameters on that design and noise level (analytic Jacobian)
  single fit   the data-seed realisation, FDM model (the estimator) and analytic model (equivalence check)
  monte carlo  `replicates` fits of independent noise draws (seeds 1000...) with `mc_model` (analytic by default:
               equal to the FDM estimator to 2e-5 in ln p, R5.3c; re-checked per setting and against Step 5.3's FDM
               Monte Carlo on the same draws): bias, sample std / CRLB, 95 % interval coverage, median |error|,
               stalled fits (chi2/reading far above the truth's) and numerical failures
sigma = 0: the analytic fit of analytic data is an inverse crime (exact by construction); the FDM fit gives the
estimator's model-error floor. Residual weights then come from zero_noise_weight_level (no intervals, no bound).
PINN (optional): one Step 5.2 variant on the base design at a few noise levels, with the exact-model consistency check.

    python experiments/p5_4_robustness.py [--config my.yaml] [--set key=value] [--tag name] [--figures-only]
"""

from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np

from aerosynapse.thermal.experiments import p5_2_pinn_inverse as P52
from aerosynapse.thermal.experiments import p5_3_classical as P53
from aerosynapse.thermal.inverse import body as B
from aerosynapse.thermal.inverse import classical as C
from aerosynapse.thermal.inverse import data as D
from aerosynapse.thermal.inverse import spec as IS
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz import inverse_figs as VF
from aerosynapse.thermal.viz.style import save_fig

MOD = "aerosynapse.thermal.experiments.p5_4_robustness"
KEEP = ("estimates", "sd_log", "success", "chi2_per_reading", "n_model_calls", "rank", "message", "error", "wall_s")


def data_of(spec: dict, level: float, seed: int, weight_level: float) -> dict:
    """Synthetic data; for sigma = 0 the residual weights come from weight_level (the data stay noise-free)."""
    d = D.synthetic(spec, level, seed)
    d["noise_sigma_mean_K"] = float(np.mean(d["sigma"]))
    if level == 0:
        d["sigma"] = np.broadcast_to(IS.noise_sigma(spec, d["T_clean"], weight_level), d["T"].shape).copy()
    return d


def _guess(spec: dict) -> dict:
    return {k: float(spec["initial_guess"][k]) for k in spec["unknown"]}


def setting_task(spec: dict, level: float, weight_level: float) -> dict:
    """Single-realisation fits (data seed): FDM estimator and analytic model on the same data."""
    unknown = list(spec["unknown"])
    d = data_of(spec, level, int(spec["data"]["seed"]), weight_level)
    out = {"n_readings": len(d["T"]), "sigma_mean_K": d["noise_sigma_mean_K"]}
    for m in ("fdm", "analytic"):
        r = C.safe_fit(spec, d, unknown, _guess(spec), m)
        out[m] = {k: r.get(k) for k in KEEP}
    ef, ea = out["fdm"]["estimates"], out["analytic"]["estimates"]
    out["fdm_vs_analytic_dlog"] = (
        float(max(abs(np.log(ef[k] / ea[k])) for k in unknown)) if ef is not None and ea is not None else None
    )
    return out


def mc_task(spec: dict, seeds: list[int], level: float, model: str) -> list[dict]:
    unknown = list(spec["unknown"])
    out = []
    for sd in seeds:
        r = C.safe_fit(spec, D.synthetic(spec, level, sd), unknown, _guess(spec), model)
        out.append(
            {k: r.get(k) for k in ("estimates", "sd_log", "success", "chi2_per_reading", "n_model_calls")}
            | {"seed": sd}
        )
    return out


def pinn_task(spec: dict, variant: dict, seed: int) -> dict:
    """One Step 5.2 PINN training + the exact model evaluated at its estimates on the same data (consistency)."""
    r = P52.run_one(spec, variant, seed)
    d = D.load(spec)
    est = r["estimates"]
    nt = max(B.truncation_terms(float(np.min(d["t"])) * est["alpha"] / float(spec["body"]["L"]) ** 2), 20)
    T_model = D.model_at(est, spec, d["r"], d["t"], nt)
    body = IS.body_of(spec)
    g, gt = body.groups(est), body.groups({k: float(spec["parameters"][k]) for k in B.PARAMS})
    keep = ("estimates", "data_rmse_K", "wall_s", "losses", "residual")
    return {k: r[k] for k in keep if k in r} | {
        "rel_err": P53.rel(est, spec),
        "model_rmse_K": float(np.sqrt(np.mean((T_model - d["T"]) ** 2))),
        "Bi_ratio": g["Bi"] / gt["Bi"],
        "Q_ratio": g["Q"] / gt["Q"] if gt["Q"] else None,
        "level": float(spec["data"]["level"]),
        "seed": seed,
    }


def mc_stats(fits: list[dict], spec: dict, crlb_log: dict | None, n_readings: int, stall_sigmas: float) -> dict:
    """Monte-Carlo summary. Stalled = chi2/reading > 1 + stall_sigmas sqrt(2 / readings) (the true parameters give
    1 +- sqrt(2 / readings)); stalled and failed fits are counted and excluded from the statistics."""
    unknown = list(spec["unknown"])
    thr = 1.0 + stall_sigmas * np.sqrt(2.0 / n_readings)
    failed = [x for x in fits if x["estimates"] is None]
    stalled = [x for x in fits if x["estimates"] is not None and x["chi2_per_reading"] > thr]
    ok = [x for x in fits if x["estimates"] is not None and x["chi2_per_reading"] <= thr]
    out = {"replicates": len(fits), "failed": len(failed), "stalled": len(stalled), "stall_threshold": float(thr)}
    if len(ok) < 2:
        return out
    X = np.array([[np.log(x["estimates"][k] / float(spec["parameters"][k])) for k in unknown] for x in ok])
    n = len(ok)
    std = X.std(0, ddof=1)
    A = 100 * np.abs(np.exp(X) - 1)
    sd = np.array([[float(x["sd_log"][k]) for k in unknown] for x in ok])
    out |= {
        "used": n,
        "bias_log": dict(zip(unknown, X.mean(0).tolist(), strict=True)),
        "bias_se_log": dict(zip(unknown, (std / np.sqrt(n)).tolist(), strict=True)),
        "std_log": dict(zip(unknown, std.tolist(), strict=True)),
        "rmse_log": dict(zip(unknown, np.sqrt((X**2).mean(0)).tolist(), strict=True)),
        "median_abs_pct": dict(zip(unknown, np.median(A, 0).tolist(), strict=True)),
        "p95_abs_pct": dict(zip(unknown, np.percentile(A, 95, axis=0).tolist(), strict=True)),
        "coverage95": dict(zip(unknown, np.mean(np.abs(X) <= 1.96 * sd, 0).tolist(), strict=True)),
        "std_rel_uncertainty": float(1 / np.sqrt(2 * (n - 1))),
        "coverage_halfwidth95": float(1.96 * np.sqrt(0.95 * 0.05 / n)),
    }
    if crlb_log:
        out["std_over_crlb"] = {k: out["std_log"][k] / crlb_log[k] for k in unknown}
        out["bias_over_crlb"] = {k: out["bias_log"][k] / crlb_log[k] for k in unknown}
    return out


def build_settings(files, cli: dict, rb: dict) -> list[dict]:
    """Every (study, label, design, level, spec) of the study, with a key that identifies identical problems. The study
    is synthetic by nature: with your own configuration it runs around your body shape, nominal `parameters` and
    measurement design (also one read from your data file) — your measured values are not used here."""
    out = []
    designs = {}
    syn = {"data": {"source": "synthetic", "compare_to_parameters": True}}
    cli = IS.deep_merge(cli, syn)
    for dsg in rb["noise"]["designs"]:
        designs[dsg] = P53.case_spec(files, cli, dsg)
    for dsg, ds in designs.items():
        for lv in rb["noise"]["levels"]:
            s = copy.deepcopy(ds)
            s["data"]["level"] = float(lv)
            out.append({"study": "noise", "label": f"{100 * lv:g} %", "design": dsg, "level": float(lv), "spec": s})
    se = rb["sensors"]
    for n in se["n"]:
        ov = {"measurement": {"sensors": {"n": int(n), "layout": se["layout"], "positions": None, "from_data": False}}}
        s = IS.load_spec(files or None, IS.deep_merge(cli, ov | {"data": {"level": float(se["level"])}}))
        out.append({"study": "sensors", "label": str(n), "design": "base", "level": float(se["level"]), "spec": s})
    pl = rb["placement"]
    for name, ov in (pl["layouts"] or {}).items():
        sens = {"positions": None, "from_data": False} | dict(ov or {})
        s = IS.load_spec(
            files or None, IS.deep_merge(cli, {"measurement": {"sensors": sens}, "data": {"level": float(pl["level"])}})
        )
        out.append({"study": "placement", "label": name, "design": "base", "level": float(pl["level"]), "spec": s})
    for st in out:
        s = st["spec"]
        st["key"] = (
            tuple(np.round(IS.sensor_positions(s), 12)),
            tuple(np.round(IS.measurement_times(s), 9)),
            st["level"],
            float(s["body"]["q_vol"]),
        )
        st["positions_frac"] = (IS.sensor_positions(s) / float(s["body"]["L"])).tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--figures-only", action="store_true", help="redraw the figures and the table from saved metrics")
    a = ap.parse_args()
    cli = IS.parse_set(a.sets)
    base = IS.load_spec(a.config or None, cli)
    rb = base["studies"]["robustness"]
    custom = bool(a.config or a.sets or a.tag != "default")
    odir = ROOT / "results" / f"p5_4_robustness_{a.tag}"
    table = odir / "robustness.md" if custom else ROOT / "docs" / "tables" / "phase5_4_robustness.md"
    VF.style()
    if a.figures_only:
        met = P53._restore(json.loads((odir / "metrics.json").read_text(encoding="utf-8")))
        figures(met, base, a.tag, odir)
        report(met, base, table, a.tag)
        print(f"wrote {table}")
        return
    with Experiment(f"p5_4_robustness_{a.tag}", base) as ex:
        t0 = time.perf_counter()
        settings = build_settings(a.config, cli, rb)
        uniq: dict = {}
        for st in settings:
            uniq.setdefault(st["key"], st)
        wl = float(rb["zero_noise_weight_level"])
        for st in uniq.values():
            b = P53.bound(st["spec"], st["level"] if st["level"] > 0 else wl)
            st["rank"] = b["rank"]
            st["crlb_log"] = dict(zip(st["spec"]["unknown"], b["crlb_log"], strict=True)) if st["level"] > 0 else None
        print(
            f"{len(settings)} settings ({len(uniq)} distinct), bounds in {time.perf_counter() - t0:.0f} s", flush=True
        )
        jobs, tags = [], []
        rp = rb.get("pinn")
        if rp:
            variant = next(v for v in base["studies"]["pinn_inverse"]["variants"] if v["name"] == rp["variant"])
            for lv in rp["levels"]:
                for sd in rp["seeds"]:
                    s = copy.deepcopy(base)  # level set after validation: 0 (noise-free) is allowed here only
                    s["data"] |= {"level": float(lv), "source": "synthetic", "compare_to_parameters": True}
                    jobs.append((MOD, "pinn_task", dict(spec=s, variant=variant, seed=int(sd))))
                    tags.append(("pinn", (float(lv), int(sd))))
        R, bsz, model = int(rb["replicates"]), int(rb["batch"]), rb["mc_model"]
        for key, st in uniq.items():
            jobs.append((MOD, "setting_task", dict(spec=st["spec"], level=st["level"], weight_level=wl)))
            tags.append(("setting", key))
            if R > 0 and st["level"] > 0 and st["rank"] == len(st["spec"]["unknown"]):
                seeds = list(range(1000, 1000 + R))
                for i in range(0, R, bsz):
                    jobs.append(
                        (
                            MOD,
                            "mc_task",
                            dict(spec=st["spec"], seeds=seeds[i : i + bsz], level=st["level"], model=model),
                        )
                    )
                    tags.append(("mc", key))
        names = {key: f"{st['study']} {st['label']} ({st['design']})" for key, st in uniq.items()}

        def done(i, r, sec):
            kind, key = tags[i]
            what = f"PINN level {key[0]:g} seed {key[1]}" if kind == "pinn" else f"{kind} {names[key]}"
            print(f"[{i + 1}/{len(jobs)}] {what}: {sec:.0f} s", flush=True)

        outs = run_tasks(jobs, base["execution"], done)
        res = {key: {"setting": None, "mc": []} for key in uniq}
        pinn = []
        for (kind, key), r in zip(tags, outs, strict=True):
            if kind == "pinn":
                pinn.append(r)
            elif kind == "setting":
                res[key]["setting"] = r
            else:
                res[key]["mc"] += r
        rows = []
        for st in settings:
            u = uniq[st["key"]]
            rr = res[st["key"]]
            sg = rr["setting"]
            row = {k: st[k] for k in ("study", "label", "design", "level", "positions_frac")}
            row |= {
                "rank": u["rank"],
                "crlb_log": u["crlb_log"],
                "n_readings": sg["n_readings"],
                "sigma_mean_K": sg["sigma_mean_K"],
                "fit_fdm": sg["fdm"],
                "fit_analytic": sg["analytic"],
                "fdm_vs_analytic_dlog": sg["fdm_vs_analytic_dlog"],
                "rel_err_fdm": P53.rel(sg["fdm"]["estimates"], st["spec"]) if sg["fdm"]["estimates"] else None,
                "unknown": list(st["spec"]["unknown"]),
                "t_max": float(IS.measurement_times(st["spec"]).max()),
            }
            if rr["mc"]:
                row["mc"] = mc_stats(rr["mc"], st["spec"], u["crlb_log"], sg["n_readings"], float(rb["stall_sigmas"]))
            rows.append(row)
            for x in rr["mc"] if st is u else []:
                if x["estimates"]:
                    ex.log(
                        study=st["study"],
                        label=st["label"],
                        design=st["design"],
                        seed=x["seed"],
                        chi2=x["chi2_per_reading"],
                        **x["estimates"],
                    )
        check = fdm_mc_check(res, uniq, base)
        ex.metrics.update({"rows": rows, "pinn": pinn, "fdm_mc_check": check, "git": git_hash(), "mc_model": model})
        ex.metrics["figures"] = figures(ex.metrics, base, a.tag, ex.dir)
        report(ex.metrics, base, table, a.tag)
        ex.metrics["table"] = str(table.relative_to(ROOT))
    print(f"wrote {table}")


def fdm_mc_check(res: dict, uniq: dict, spec: dict) -> list[dict]:
    """Analytic Monte-Carlo fits vs the Step 5.3 FDM Monte-Carlo fits on the SAME noise draws (seeds 1000...)."""
    p53 = ROOT / "results" / "p5_3_classical_default" / "metrics.json"
    if not p53.exists():
        return []
    m53 = P53._restore(json.loads(p53.read_text(encoding="utf-8")))
    out = []
    for key, st in uniq.items():
        if st["study"] != "noise" or st["design"] not in m53["cases"] or not res[key]["mc"]:
            continue
        mc = m53["cases"][st["design"]].get("monte_carlo")
        if not mc or abs(mc["level"] - st["level"]) > 1e-12:
            continue
        if mc.get("failed"):
            continue  # 5.3 samples then no longer align with the seeds
        fits = sorted(res[key]["mc"], key=lambda x: x["seed"])
        m = min(len(fits), len(mc["samples_log"]))
        if m < 2 or any(x["estimates"] is None for x in fits[:m]):
            continue
        S = np.asarray(mc["samples_log"])[:m]  # FDM fits of seeds 1000... in order
        unknown = list(st["spec"]["unknown"])
        A = np.array(
            [[np.log(x["estimates"][k] / float(st["spec"]["parameters"][k])) for k in unknown] for x in fits[:m]]
        )
        out.append(
            {
                "design": st["design"],
                "level": st["level"],
                "draws": m,
                "max_abs_dlog": float(np.abs(A - S).max()),
                "std_fdm": dict(zip(unknown, S.std(0, ddof=1).tolist(), strict=True)),
                "std_analytic": dict(zip(unknown, A.std(0, ddof=1).tolist(), strict=True)),
            }
        )
    return out


def _f(v, fmt=".3g"):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)


def report(met: dict, spec: dict, path, tag: str) -> None:
    rows, rb = met["rows"], spec["studies"]["robustness"]
    unk = rows[0]["unknown"]
    L = [
        f"# Step 5.4 robustness — `{tag}` (auto-generated by experiments/p5_4_robustness.py)",
        "",
        *(
            [
                f"Your configuration: every setting below is SYNTHETIC around your body ({spec['body'].get('shape', 'slab')}), your "
                "nominal parameters (the 'truth' of the errors below) and your measurement design; your measured values are not used.",
                "",
            ]
            if spec["data"]["source"] == "file"
            else []
        ),
        f"Least squares as in Step 5.3. Per setting: Cramér–Rao bound (1σ, %), Monte Carlo of {rb['replicates']} fits "
        f"({met['mc_model']} model; independent noise draws): sample std (%), median |error| (%), sample std / CRLB, 95 % "
        f"interval coverage, bias / CRLB, stalled fits (χ²/reading > 1 + {rb['stall_sigmas']:g}·√(2/readings)) and numerical "
        "failures; plus the single fit of the data-seed realisation with the FDM estimator (error %). Noise level = σ / RMS "
        "of the clean θ signal (Phase 2 gaussian model); σ = 0: noise-free data, weights from the "
        f"{100 * rb['zero_noise_weight_level']:g} % level, FDM fit only (the analytic fit of analytic data is exact by "
        "construction).",
        "",
    ]
    for study, title in (
        ("noise", "Noise level"),
        ("sensors", "Number of sensors (uniform layout, 60 times each)"),
        ("placement", "Sensor placement"),
    ):
        sel = [r for r in rows if r["study"] == study]
        if not sel:
            continue
        L += [f"## {title}", ""]
        hdr = (
            "| setting | σ K | "
            + " | ".join(f"{k}: CRLB / std / median abs err %" for k in unk)
            + " | stalled / failed | FDM single fit error % |"
        )
        L += [hdr, "|---|---|" + "---|" * len(unk) + "---|---|"]
        for r in sel:
            mc = r.get("mc") or {}
            cr = r["crlb_log"]
            cells = []
            for k in unk:
                c = _f(100 * cr[k]) if cr else "—"
                s = _f(100 * mc["std_log"][k]) if "std_log" in mc else "—"
                md = _f(mc["median_abs_pct"][k]) if "median_abs_pct" in mc else "—"
                cells.append(f"{c} / {s} / {md}")
            sf = f"{mc.get('stalled', '—')} / {mc.get('failed', '—')}" if mc else "—"
            fe = " / ".join(f"{100 * r['rel_err_fdm'][k]:+.3g}" for k in unk) if r["rel_err_fdm"] else "failed"
            name = (
                f"{r['label']} ({r['design'].replace('_', ' ')})" if study == "noise" else r["label"].replace("_", " ")
            )
            L.append(f"| {name} | {r['sigma_mean_K']:.3f} | " + " | ".join(cells) + f" | {sf} | {fe} |")
        L += [
            "",
            "| setting | readings | rank | std / CRLB | 95 % coverage | bias / CRLB | FDM − analytic max abs Δ ln p |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in sel:
            mc = r.get("mc") or {}
            ratio = " / ".join(_f(mc["std_over_crlb"][k]) for k in unk) if "std_over_crlb" in mc else "—"
            cov = " / ".join(f"{100 * mc['coverage95'][k]:.0f}" for k in unk) if "coverage95" in mc else "—"
            bias = " / ".join(f"{mc['bias_over_crlb'][k]:+.2f}" for k in unk) if "bias_over_crlb" in mc else "—"
            name = (
                f"{r['label']} ({r['design'].replace('_', ' ')})" if study == "noise" else r["label"].replace("_", " ")
            )
            L.append(
                f"| {name} | {r['n_readings']} | {r['rank']}/{len(unk)} | {ratio} | {cov} | {bias} | {_f(r['fdm_vs_analytic_dlog'], '.1e')} |"
            )
        if study == "placement":
            L += [
                "",
                "Positions (fractions of L): "
                + "; ".join(f"{r['label']} {np.round(r['positions_frac'], 3).tolist()}" for r in sel),
            ]
        L.append("")
    if met.get("fdm_mc_check"):
        L += [
            "Analytic Monte-Carlo fits vs the Step 5.3 FDM Monte Carlo on the same noise draws: "
            + "; ".join(
                f"{c['design']} at {100 * c['level']:g} %: {c['draws']} draws, max abs Δ ln p {c['max_abs_dlog']:.1e}"
                for c in met["fdm_mc_check"]
            )
            + ".",
            "",
        ]
    if met.get("pinn"):
        L += [
            "## PINN (Step 5.2 variant `" + spec["studies"]["robustness"]["pinn"]["variant"] + "`, base design)",
            "",
            "| noise level | seed | "
            + " | ".join(f"{k} error %" for k in unk)
            + " | Bi / true | Q / true | PINN data RMSE K | exact model at its estimates, RMSE K | time s |",
            "|---|---|" + "---|" * len(unk) + "---|---|---|---|---|",
        ]
        for p in sorted(met["pinn"], key=lambda x: (x["level"], x["seed"])):
            L.append(
                f"| {100 * p['level']:g} % | {p['seed']} | "
                + " | ".join(f"{100 * p['rel_err'][k]:+.1f}" for k in unk)
                + f" | {p['Bi_ratio']:.3f} | {_f(p['Q_ratio'])} | {p['data_rmse_K']:.3f} | {p['model_rmse_K']:.3f} | {p['wall_s']:.0f} |"
            )
        L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(met: dict, spec: dict, tag: str, odir) -> list[str]:
    VF.style()
    rows = met["rows"]
    unk = rows[0]["unknown"]
    paths = []
    designs = {}
    for r in (r for r in rows if r["study"] == "noise"):
        d = designs.setdefault(
            r["design"].replace("_", " "),
            {"levels": [], "crlb": {k: [] for k in unk}, "std": {k: [] for k in unk}, "floor": None},
        )
        if r["level"] == 0:
            d["floor"] = {k: abs(100 * r["rel_err_fdm"][k]) for k in unk} if r["rel_err_fdm"] else None
            continue
        d["levels"].append(100 * r["level"])
        for k in unk:
            d["crlb"][k].append(100 * r["crlb_log"][k])
            d["std"][k].append(100 * r["mc"]["std_log"][k] if r.get("mc", {}).get("std_log") else np.nan)
    pinn = None
    if met.get("pinn"):
        pinn = {
            "levels": [100 * p["level"] for p in met["pinn"]],
            "err": {k: [abs(100 * p["rel_err"][k]) for p in met["pinn"]] for k in unk},
        }
    if designs:
        paths += save_fig(VF.noise_robustness(designs, pinn, unk), f"p5_4_noise_{tag}", odir)
    sens = [r for r in rows if r["study"] == "sensors"]
    plc = [r for r in rows if r["study"] == "placement"]
    if sens or plc:

        def pack(rs):
            return [
                {
                    "label": r["label"].replace("_", " "),
                    "crlb": {k: 100 * r["crlb_log"][k] for k in unk} if r["crlb_log"] else None,
                    "std": {k: 100 * r["mc"]["std_log"][k] for k in unk} if r.get("mc", {}).get("std_log") else None,
                    "rank": r["rank"],
                }
                for r in rs
            ]

        paths += save_fig(VF.sensor_studies(pack(sens), pack(plc), unk), f"p5_4_sensors_{tag}", odir)
    secs = []
    for title, sel in (
        *[
            (f"noise, {d.replace('_', ' ')}", [r for r in rows if r["study"] == "noise" and r["design"] == d])
            for d in dict.fromkeys(r["design"] for r in rows if r["study"] == "noise")
        ],
        ("sensors", sens),
        ("placement", plc),
    ):
        sel = [r for r in sel if r.get("mc", {}).get("std_over_crlb")]
        if sel:
            secs.append(
                {
                    "title": title,
                    "labels": [r["label"].replace("_", " ") for r in sel],
                    "ratio": {k: [r["mc"]["std_over_crlb"][k] for r in sel] for k in unk},
                    "coverage": {k: [100 * r["mc"]["coverage95"][k] for r in sel] for k in unk},
                    "ratio_band": float(np.mean([r["mc"]["std_rel_uncertainty"] for r in sel])),
                    "coverage_band": float(100 * np.mean([r["mc"]["coverage_halfwidth95"] for r in sel])),
                }
            )
    if secs:
        paths += save_fig(VF.calibration(secs, unk), f"p5_4_calibration_{tag}", odir)
    return [str(p) for p in paths]


if __name__ == "__main__":
    main()
