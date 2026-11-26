"""Step 5.2: PINN inverse identification of (alpha, k, h) — log-parameters, joint vs two-stage training, data vs physics
loss weighting — on the 1D body of configs/inverse/defaults.yaml (slab by default; cylinder / sphere selectable).

For every variant (studies.pinn_inverse.variants) and seed the PINN is trained on the same measurements (data.*:
synthetic noisy data from the forward model, or YOUR file). Reference values on the same data: the classical weighted
least-squares estimate (Levenberg-Marquardt, analytic model, same initial guess) and, for synthetic data, the
Cramer-Rao bound of Step 5.1 for that design and noise level.

    python experiments/p5_2_pinn_inverse.py [--config my.yaml] [--set key=value] [--tag name]
    e.g. your measured data:  --set data.source=file --set data.file=examples/inverse/slab_measured_example.csv
         a sphere, k known:   --set body.shape=sphere --set unknown=[alpha,h]
"""

from __future__ import annotations

import argparse

import numpy as np

from aerosynapse.thermal.experiments.p5_1_identifiability import analyse_spec
from aerosynapse.thermal.inverse import body as B
from aerosynapse.thermal.inverse import data as D
from aerosynapse.thermal.inverse import design as DS
from aerosynapse.thermal.inverse import pinn as PI
from aerosynapse.thermal.inverse import spec as IS
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz import inverse_figs as VF
from aerosynapse.thermal.viz.style import save_fig

MOD = "aerosynapse.thermal.experiments.p5_2_pinn_inverse"


def case_spec(files, cli: dict, case: str | None) -> dict:
    """The spec of a study case: None = the configured problem; a shape name = the same problem on that body shape;
    'optimised_design' = the configured problem measured with the Step 5.1 optimal design (recomputed, deterministic).
    With your data (data.source = file) every case other than the configured one is a SYNTHETIC what-if at your nominal
    `parameters`: your file belongs to your configuration only."""
    if case and IS.load_spec(files or None, cli)["data"]["source"] == "file":
        cli = IS.deep_merge(cli, {"data": {"source": "synthetic", "compare_to_parameters": True}})
    if case == "optimised_design":
        base = IS.load_spec(files or None, cli)
        opt = DS.Designer(base).optimise(
            int(base["design_optimisation"]["starts"]), int(base["design_optimisation"]["seed"])
        )
        return DS.Designer(base).spec_of(opt["optimal_design"])
    return IS.load_spec(files or None, IS.deep_merge(cli, {"body": {"shape": case}} if case else {}))


def run_one(spec: dict, variant: dict, seed: int) -> dict:
    """One PINN training (executed in its own process)."""
    data = D.load(spec)
    unknown = list(spec["unknown"])
    init = {k: float(spec["initial_guess"][k]) for k in unknown}
    known = {k: float(spec["parameters"][k]) for k in B.PARAMS}
    r = PI.train(
        IS.pinn_settings(spec, variant),
        IS.body_of(spec),
        data,
        unknown,
        init,
        known,
        variant.get("mode", "joint"),
        seed,
    )
    r["sigma_mean_K"] = float(np.mean(data["sigma"]))
    return r


def rel_err(est: dict, spec: dict) -> dict:
    return {k: (est[k] - float(spec["parameters"][k])) / float(spec["parameters"][k]) for k in spec["unknown"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--table-only", action="store_true", help="rebuild the table from the saved metrics of --tag")
    a = ap.parse_args()
    if a.table_only:
        import json

        import yaml

        rdir = ROOT / "results" / f"p5_2_pinn_inverse_{a.tag}"
        met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
        saved = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
        out = (
            ROOT / "docs" / "tables" / "phase5_2_pinn_inverse.md"
            if a.tag == "default" and not (a.config or a.sets)
            else rdir / "pinn_inverse.md"
        )
        report(met["records"], met["cases"], saved, met["compare"], out, a.tag)

        class _Ex:  # figures() needs only .dir
            dir = rdir

        figures(met["records"], met["cases"], saved, a.tag, _Ex)
        print(f"wrote {out} and the figures")
        return
    cli = IS.parse_set(a.sets)
    base = case_spec(a.config, cli, None)
    pi = base["studies"]["pinn_inverse"]
    compare = bool(base["data"].get("compare_to_parameters", True))
    shapes = [base["body"]["shape"]] + [s for s in (pi.get("shapes") or []) if s != base["body"]["shape"]]
    if pi.get("optimised_design"):
        shapes.append("optimised_design")
    with Experiment(f"p5_2_pinn_inverse_{a.tag}", base) as ex:
        cases = {}

        def prepare(sh: str) -> dict:
            s = case_spec(a.config, cli, sh if sh != base["body"]["shape"] else None)
            data = D.load(s)
            init = {k: float(s["initial_guess"][k]) for k in s["unknown"]}
            ls = D.least_squares_estimate(s, data, list(s["unknown"]), init)
            crlb = None
            if s["data"]["source"] == "synthetic":
                lv = float(s["data"]["level"])
                r = analyse_spec(s, [lv])["levels"][str(lv)]
                crlb = r["crlb_percent"]
            cases[sh] = {
                "spec": s,
                "least_squares": ls,
                "crlb_percent": crlb,
                "n_readings": len(data["T"]),
                "sigma_mean_K": float(np.mean(data["sigma"])),
                "data_source": data["source"],
                "groups": IS.body_of(s).groups({k: float(s["parameters"][k]) for k in B.PARAMS}),
            }
            print(
                f"[{sh}] least squares {ls['estimates']} chi2/reading {ls['chi2_per_reading']:.3f}; CRLB % {crlb}",
                flush=True,
            )
            return s

        def show(i: int, r: dict, sec: float) -> None:
            sh, v, seed = labels[i]
            e = rel_err(r["estimates"], cases[sh]["spec"]) if compare else {}
            print(
                f"[{sh} {v['name']:14s} s{seed}] "
                + ", ".join(
                    f"{k} {r['estimates'][k]:.4g}" + (f" ({100 * e[k]:+.2f}%)" if e else "")
                    for k in cases[sh]["spec"]["unknown"]
                )
                + f" | data RMSE {r['data_rmse_K']:.3f} K (σ {r['sigma_mean_K']:.3f}) | {r['wall_s']:.0f}s",
                flush=True,
            )

        def launch(sh: str, variants: list[dict]) -> list:
            s = prepare(sh)
            jobs, labels[:] = [], []
            for v in variants:
                for seed in pi["seeds"]:
                    jobs.append((MOD, "run_one", dict(spec=s, variant=v, seed=seed)))
                    labels.append((sh, v, seed))
            return list(zip(list(labels), run_tasks(jobs, base["execution"], show), strict=True))

        labels: list = []
        done = launch(shapes[0], pi["variants"])
        chosen = pi.get("shape_variant", "auto")
        if len(shapes) > 1:
            if chosen == "auto":  # smallest mean |error| / CRLB (or |error| without a bound) on the default shape
                c0 = cases[shapes[0]]

                def score(vn: str) -> float:
                    rs = [r for (_, v, _), r in done if v["name"] == vn]
                    return float(
                        np.mean(
                            [
                                abs(rel_err(r["estimates"], c0["spec"])[k])
                                / ((c0["crlb_percent"] or {}).get(k, 100.0) / 100)
                                for r in rs
                                for k in base["unknown"]
                            ]
                        )
                    )

                chosen = min((v["name"] for v in pi["variants"]), key=score) if compare else pi["variants"][0]["name"]
            vsel = next(v for v in pi["variants"] if v["name"] == chosen)
            print(f"[shapes] variant {chosen}", flush=True)
            for sh in shapes[1:]:
                done += launch(sh, [vsel])
        records = []
        for (sh, v, seed), r in done:
            rec = {
                "shape": sh,
                "variant": v["name"],
                "mode": v.get("mode", "joint"),
                "seed": seed,
                **{
                    k: r[k]
                    for k in (
                        "estimates",
                        "losses",
                        "data_rmse_K",
                        "sigma_mean_K",
                        "w_data_final",
                        "wall_s",
                        "t_ref",
                        "history",
                    )
                },
            }
            if compare:
                rec["rel_err"] = rel_err(r["estimates"], cases[sh]["spec"])
            records.append(rec)
            ex.log(
                shape=sh,
                variant=v["name"],
                seed=seed,
                data_rmse_K=r["data_rmse_K"],
                wall_s=r["wall_s"],
                **{f"est_{k}": r["estimates"][k] for k in B.PARAMS},
                **{f"err_{k}": rec.get("rel_err", {}).get(k) for k in B.PARAMS},
            )
        summary = {sh: {k: v for k, v in c.items() if k != "spec"} for sh, c in cases.items()}
        for sh, c in summary.items():
            if compare:
                c["least_squares"]["rel_err"] = rel_err(c["least_squares"]["estimates"], cases[sh]["spec"])
        ex.metrics.update(
            {"cases": summary, "records": records, "compare": compare, "shape_variant": chosen, "git": git_hash()}
        )
        ex.metrics["figures"] = figures(records, summary, base, a.tag, ex)
        table = (
            ex.dir / "pinn_inverse.md"
            if (a.config or a.sets or a.tag != "default")
            else ROOT / "docs" / "tables" / "phase5_2_pinn_inverse.md"
        )
        report(records, summary, base, compare, table, a.tag)
        ex.metrics["table"] = str(table)
    print(f"wrote {table}")


def _ms(v: list[float]) -> str:
    v = np.asarray(v, float)
    return f"{v.mean():+.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v[0]:+.2f}"


def report(records: list[dict], cases: dict, spec: dict, compare: bool, path, tag: str) -> None:
    unk = list(spec["unknown"])
    pn = spec["pinn"]
    L = [
        f"# Step 5.2 PINN inverse identification — `{tag}` (auto-generated by experiments/p5_2_pinn_inverse.py)",
        "",
        f"Unknown {unk} (known: {[k for k in B.PARAMS if k not in unk]}); initial guess {spec['initial_guess']}; {'true' if compare else 'nominal'} {spec['parameters']}. "
        f"Data: {spec['data']['source']} (level {spec['data']['level']:g}). PINN {pn['network']}, collocation {pn['collocation']}, Adam "
        f"{pn['adam']['epochs']} (lr {pn['adam']['lr']:g}, log-parameters {pn['adam']['lr_params']:g}) + L-BFGS {pn['lbfgs']['iterations']}; "
        "θ = 1 + t̂ N(r̂², t̂) (exact initial condition and centre symmetry). Errors = (estimate − true) / true in %, mean ± std over seeds; "
        "abs err / CRLB = mean absolute error in units of the Cramér–Rao bound of Step 5.1 for the same design and noise (≈ 1 is the best an "
        "unbiased estimator can do on average; ~0.8 expected for |N(0, 1)|).",
        "",
    ]
    for sh, c in cases.items():
        rs = [r for r in records if r["shape"] == sh]
        g = c["groups"]
        L += [
            f"## {sh} ({'' if compare else 'nominal '}Bi {g['Bi']:.4g}, Q {g['Q']:.4g}, L²/α {g['t_scale']:.4g} s); {c['n_readings']} readings, mean σ {c['sigma_mean_K']:.3f} K; data: {c['data_source']}",
            "",
        ]
        hdr = (
            "| estimator | seeds | "
            + " | ".join(f"{k} error %" if compare else f"{k} (estimate)" for k in unk)
            + (" | " + " | ".join(f"{k} abs err / CRLB" for k in unk) if c["crlb_percent"] else "")
            + " | data RMSE / σ | time s |"
        )
        L += [hdr, "|" + "---|" * (2 + len(unk) * (2 if c["crlb_percent"] else 1) + 2)]
        if c["crlb_percent"]:
            L.append(
                "| Cramér–Rao bound (1σ) | — | "
                + " | ".join(f"± {c['crlb_percent'][k]:.3g}" for k in unk)
                + " | "
                + " | ".join("1" for _ in unk)
                + " | — | — |"
            )
        ls = c["least_squares"]
        if not compare:  # your data: the least-squares estimate on the same data, for comparison
            L.append(
                "| least squares (analytic model) | — | "
                + " | ".join(f"{ls['estimates'][k]:.4g}" for k in unk)
                + f" | {np.sqrt(ls['chi2_per_reading']):.3f} | — |"
            )
        if compare:
            L.append(
                "| least squares (analytic model) | — | "
                + " | ".join(f"{100 * ls['rel_err'][k]:+.2f}" for k in unk)
                + (
                    " | " + " | ".join(f"{abs(100 * ls['rel_err'][k]) / c['crlb_percent'][k]:.2f}" for k in unk)
                    if c["crlb_percent"]
                    else ""
                )
                + f" | {np.sqrt(ls['chi2_per_reading']):.3f} | — |"
            )
        for vn in dict.fromkeys(r["variant"] for r in rs):
            vr = [r for r in rs if r["variant"] == vn]
            cells = []
            if compare:
                cells += [_ms([100 * r["rel_err"][k] for r in vr]) for k in unk]
                if c["crlb_percent"]:
                    cells += [
                        f"{np.mean([abs(100 * r['rel_err'][k]) for r in vr]) / c['crlb_percent'][k]:.2f}" for k in unk
                    ]
            else:
                cells += [f"{np.mean([r['estimates'][k] for r in vr]):.4g}" for k in unk]
            L.append(
                f"| PINN {vn} | {len(vr)} | "
                + " | ".join(cells)
                + f" | {np.mean([r['data_rmse_K'] / r['sigma_mean_K'] for r in vr]):.3f} | {np.mean([r['wall_s'] for r in vr]):.0f} |"
            )
        L.append("")
    if not compare:
        L.append("Errors not reported: data.compare_to_parameters = false (estimates shown instead).")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(records: list[dict], cases: dict, spec: dict, tag: str, ex) -> list[str]:
    """Parameter convergence (small multiples: parameters x variants, every seed) and per-seed errors against the
    +-1 / +-2 Cramer-Rao bands, for the default case (plus the least-squares estimate as a reference row)."""
    VF.style()
    paths = []
    unk = list(spec["unknown"])
    sh0 = next(iter(cases))
    truth = {k: float(spec["parameters"][k]) for k in B.PARAMS}
    rs0 = [r for r in records if r["shape"] == sh0]
    hist = {vn: [r["history"] for r in rs0 if r["variant"] == vn] for vn in dict.fromkeys(r["variant"] for r in rs0)}
    crlb = cases[sh0]["crlb_percent"]
    paths += save_fig(VF.convergence(hist, unk, truth, crlb), f"p5_2_convergence_{tag}", ex.dir)
    if all("rel_err" in r for r in records):
        rows = [
            {"name": vn, "errors": {k: [100 * r["rel_err"][k] for r in rs0 if r["variant"] == vn] for k in unk}}
            for vn in hist
        ]
        ls = cases[sh0]["least_squares"]
        rows.append({"name": "least_squares", "errors": {k: [100 * ls["rel_err"][k]] for k in unk}})
        paths += save_fig(VF.error_dots(rows, unk, crlb), f"p5_2_errors_{tag}", ex.dir)
    return [str(p) for p in paths]


if __name__ == "__main__":
    main()
