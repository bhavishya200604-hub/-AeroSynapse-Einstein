"""Step 4.1: architecture comparison for the 1D PINN (plain MLP, Fourier features with tuned sigma, modified MLP,
residual blocks, random weight factorisation, SIREN) on user-defined (problem, parameter) cases.

    python experiments/p4_1_architectures.py --profile smoke
    python experiments/p4_1_architectures.py --config my.yaml --tag mine
    python experiments/p4_1_architectures.py --set 'arch_study.seeds=[0,1,2]' --set 'arch_study.sigma_sweep.enabled=false'
    python experiments/p4_1_architectures.py --describe

Everything is read from the layered PINN spec (configs/pinn/defaults.yaml: `network`, `arch_study`, training, ...).
Fourier sigma is chosen by the TRUTH-FREE validation loss (PDE+IC+BC loss on an independent collocation set); the analytic
error is only reported. Outputs: results/p4_1_architectures_<tag>/, docs/tables/phase4_1_architectures.md, figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from aerosynapse.thermal.experiments.p3_pinn1d import train_one
from aerosynapse.thermal.experiments.p4_report import arch_table
from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz.style import apply_style, save_fig

COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]


def run_case(spec: dict, net: dict, seed: int, case: dict, out_dir: Path, tag: str) -> dict:
    s = SP.deep_merge(spec, {})
    s["network"], s["seed"] = net, seed
    prob = P.build_problem(case["problem"], spec["problems"][case["problem"]], case["value"])
    out = train_one(s, prob, [case["value"]], out_dir, tag, case=case)
    rec, _ = out["records"][0]
    rec = {k: v for k, v in rec.items() if k not in ("grad_norms",)}
    rec["seed"] = seed
    rec["history"] = out["history"]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--describe", action="store_true")
    ap.add_argument(
        "--table-only",
        action="store_true",
        help="rebuild the table and the figures from the saved run of --tag (no training)",
    )
    a = ap.parse_args()
    if a.table_only:
        rdir = ROOT / "results" / f"p4_1_architectures_{a.tag or a.profile}"
        met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
        saved = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
        out = (
            rdir / "architectures.md"
            if (a.config or a.sets or (a.tag or a.profile) != saved["profile"])
            else ROOT / "docs" / "tables" / "phase4_1_architectures.md"
        )
        arch_table(met["runs"], met["sigma_sweep"], saved, out, a.tag or a.profile)
        print(f"wrote {out}")
        # figures from the same saved run (training histories = each run's log.csv); nothing is trained
        from aerosynapse.thermal.utils import replay as RP

        hists = {}
        for r in met["runs"]:
            log = rdir / "runs" / f"{r['problem']}_{r['value']:g}_{r['label']}_s{r['seed']}" / "log.csv"
            if log.exists():
                hists[(r["case_index"], r["label"], r["seed"])] = RP.history(log)
        apply_style()
        print(
            "\n".join(figures(met["runs"], hists, met["sigma_sweep"], saved, a.tag or a.profile, RP.target(rdir.name)))
        )
        return
    spec = SP.load_spec(a.config or None, a.profile, SP.parse_set(a.sets))
    st = spec["arch_study"]
    if a.describe:
        print(SP.describe(spec))
        print("arch_study:", yaml.safe_dump(st, sort_keys=False))
        return
    tag = a.tag or a.profile
    custom = bool(a.config or a.sets)
    apply_style()
    print(SP.describe(spec), flush=True)
    SP.check_user_files(spec, st["cases"])
    MOD = "aerosynapse.thermal.experiments.p4_1_architectures"
    with Experiment(f"p4_1_architectures_{tag}", spec) as ex:
        exe = spec["execution"]
        sweep = None
        sigma_star = None
        cache: dict = {}
        if st["sigma_sweep"].get("enabled", False):
            sw = st["sigma_sweep"]
            seed0 = st["seeds"][0]
            sigmas = [float(sg) for sg in sw["values"]]
            tasks = [
                (
                    MOD,
                    "run_case",
                    dict(
                        spec=spec,
                        net=SP.deep_merge(spec["network"], {"arch": "fourier", "fourier": {"sigma": sg}}),
                        seed=seed0,
                        case=sw["case"],
                        out_dir=ex.dir / "runs" / f"sigma_{sg:g}",
                        tag=tag,
                    ),
                )
                for sg in sigmas
            ]

            def sweep_done(i: int, r: dict, sec: float) -> None:
                print(
                    f"[sigma {sigmas[i]:g}] val_loss={r['val_loss']:.3e} relL2={r['rel_l2']:.3e} ({r['status']})",
                    flush=True,
                )

            runs = []
            for sg, r in zip(sigmas, run_tasks(tasks, exe, sweep_done), strict=True):
                cache[(sw["case"]["problem"], sw["case"]["value"], f"sigma={sg:g}", seed0)] = r
                runs.append({"sigma": sg, "val_loss": r["val_loss"], "rel_l2": r["rel_l2"], "status": r["status"]})
            sigma_star = min(runs, key=lambda q: q["val_loss"])["sigma"]
            sweep = {"case": sw["case"], "seed": seed0, "runs": runs, "selected_sigma": sigma_star}
        jobs: list[tuple] = []  # (case index, label, seed, cached record or None, task or None)
        for ci, case in enumerate(st["cases"]):
            for entry in st["architectures"]:
                label, net = SP.network_variant(spec, entry)
                tuned = net["arch"] == "fourier" and sigma_star is not None and isinstance(entry, str)
                if tuned and st["sigma_sweep"].get("use_tuned", True):
                    net = SP.deep_merge(net, {"fourier": {"sigma": float(sigma_star)}})
                    label = "fourier"
                for seed in st["seeds"]:
                    key = (
                        case["problem"],
                        case["value"],
                        f"sigma={sigma_star:g}" if sigma_star is not None else "",
                        seed,
                    )
                    if tuned and key in cache and not case.get("data"):
                        jobs.append((ci, label, seed, dict(cache[key]), None))
                        continue
                    out_dir = ex.dir / "runs" / f"{case['problem']}_{case['value']:g}_{label}_s{seed}"
                    task = (MOD, "run_case", dict(spec=spec, net=net, seed=seed, case=case, out_dir=out_dir, tag=tag))
                    jobs.append((ci, label, seed, None, task))
        pending = [j for j in jobs if j[4] is not None]

        def show(r: dict, label: str, seed: int) -> None:
            print(
                f"[{r['problem']} {r['value']:g} | {label:13s} s{seed}] relL2={r['rel_l2']:.3e} Linf={r['linf']:.2e} "
                f"val={r['val_loss']:.2e} params={r['n_params']} {r['wall_s']:.0f}s {r['status']}",
                flush=True,
            )

        results = iter(run_tasks([j[4] for j in pending], exe, lambda i, r, s: show(r, pending[i][1], pending[i][2])))
        recs: list[dict] = []
        histories: dict = {}
        for ci, label, seed, cached, task in jobs:
            r = cached if cached is not None else next(results)
            if cached is not None:
                show(r, label, seed)
            r["label"] = label
            r["case_index"] = ci
            histories[(ci, label, seed)] = r.pop("history")
            recs.append(r)
            ex.log(
                problem=r["problem"],
                value=r["value"],
                arch=label,
                seed=seed,
                rel_l2=r["rel_l2"],
                linf=r["linf"],
                rel_h1=r["rel_h1"],
                val_loss=r["val_loss"],
                n_params=r["n_params"],
                wall_s=r["wall_s"],
                status=r["status"],
            )
        ex.metrics.update({"runs": recs, "sigma_sweep": sweep, "git": git_hash(), "torch": torch.__version__})
        ex.metrics["n_runs"] = len(recs)
        ex.metrics["figures"] = figures(recs, histories, sweep, spec, tag, ex)
        table = (ex.dir / "architectures.md") if custom else (ROOT / "docs" / "tables" / "phase4_1_architectures.md")
        arch_table(recs, sweep, spec, table, tag)
        ex.metrics["table"] = str(table)
    print(f"wrote {table}")


def figures(recs: list[dict], histories: dict, sweep: dict | None, spec: dict, tag: str, ex: Experiment) -> list[str]:
    paths = []
    cases = list(dict.fromkeys((r["problem"], r["value"]) for r in recs))
    labels = list(dict.fromkeys(r["label"] for r in recs))
    fig, axs = plt.subplots(1, len(cases), figsize=(4.4 * len(cases), 3.5), squeeze=False)
    for ax, (prob, val) in zip(axs[0], cases, strict=True):
        means = []
        for lab in labels:
            v = [r["rel_l2"] for r in recs if r["problem"] == prob and r["value"] == val and r["label"] == lab]
            means.append(np.mean(v) if v else np.nan)
        ax.bar(range(len(labels)), np.maximum(means, 1e-16), color=COLORS[: len(labels)])
        ax.axhline(spec["evaluation"]["target_rel_l2"], color="k", ls=":", lw=0.8)
        ax.set(yscale="log", xticks=range(len(labels)), ylabel="relative $L_2$ error", title=f"{prob}, value {val:g}")
        ax.set_xticklabels(labels, rotation=50, fontsize=7)
    paths += save_fig(fig, f"p4_1_error_by_architecture_{tag}", ex.dir)
    fig, axs = plt.subplots(1, len(cases), figsize=(4.4 * len(cases), 3.3), squeeze=False)
    for ax, ci in zip(axs[0], range(len(cases)), strict=True):
        for i, lab in enumerate(labels):
            h = next((v for (c, lb, s), v in histories.items() if c == ci and lb == lab), None)
            if h:
                ax.semilogy([max(r["total"], 1e-30) for r in h], color=COLORS[i % 7], label=lab)
        ax.set(xlabel="log row (Adam then L-BFGS)", ylabel="training loss", title=f"{cases[ci][0]}, {cases[ci][1]:g}")
        ax.legend(fontsize=6)
    paths += save_fig(fig, f"p4_1_loss_curves_{tag}", ex.dir)
    if sweep:
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        s = [q["sigma"] for q in sweep["runs"]]
        ax.loglog(s, [q["val_loss"] for q in sweep["runs"]], "o-", label="validation loss (truth-free)")
        ax2 = ax.twinx()
        ax2.loglog(s, [q["rel_l2"] for q in sweep["runs"]], "s--", color="#D55E00", label="rel $L_2$ (reported only)")
        ax2.grid(False)
        ax.axvline(sweep["selected_sigma"], color="k", ls=":", lw=0.8)
        ax.set(xlabel="Fourier $\\sigma$", ylabel="validation loss", title="Fourier $\\sigma$ sweep")
        ax2.set_ylabel("relative $L_2$ error")
        paths += save_fig(fig, f"p4_1_sigma_sweep_{tag}", ex.dir)
    return [str(p) for p in paths]


if __name__ == "__main__":
    main()
