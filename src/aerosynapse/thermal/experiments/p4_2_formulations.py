"""Step 4.2: 2D formulations - primitive (u,v,p,theta) vs stream function (psi,p,theta), hard-constraint BCs, pressure
gauge and residual normalisation - on a manufactured solution (exact truth) and the heated lid-driven cavity (truth:
steady Newton FV reference on two grids).

    python experiments/p4_2_formulations.py --profile smoke
    python experiments/p4_2_formulations.py --config my.yaml --tag mine
    python experiments/p4_2_formulations.py --set 'formulation_study.cases=[C_cavity]' --set problems2d.C_cavity.parameters.Re=400
    python experiments/p4_2_formulations.py --describe
    python experiments/p4_2_formulations.py --table-only --tag smoke

Everything comes from the layered PINN spec (configs/pinn/defaults.yaml: formulation, problems2d, formulation_study,
network, sampling, training, loss.weights). Outputs: results/p4_2_formulations_<tag>/, docs/tables/phase4_2_formulations.md
(default settings only; custom settings write the table into the results directory), figures.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from aerosynapse.thermal.experiments.p4_report import formulation_table
from aerosynapse.thermal.pinn import problems2d as P2
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn import userdata as U
from aerosynapse.thermal.pinn.architectures import build_network
from aerosynapse.thermal.pinn.networks import resolve_device
from aerosynapse.thermal.pinn.ref2d import cavity_reference
from aerosynapse.thermal.pinn.trainer import Trainer
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz.style import DIV_CMAP, SEQ_CMAP, apply_style, save_fig

COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000", "#999999"]


def train_variant(
    spec: dict, case: str | dict, label: str, form: dict, seed: int, reference: dict | None, out_dir: Path, tag: str
) -> tuple[dict, dict]:
    """One training run (executed in its own process by default, D57). ``case``: name or {problem, data}."""
    entry = case if isinstance(case, dict) else {"problem": case}
    case = entry["problem"]
    s = SP.deep_merge(spec, {})
    s["seed"] = seed
    dev = resolve_device(spec["device"])
    dtype = torch.float64 if spec["dtype"] == "float64" else torch.float32
    prob = P2.build_problem2d(case, s, form, reference)
    torch.manual_seed(seed)
    net = build_network(s["network"], list(prob.lo), list(prob.hi), prob.out_dim).to(device=dev, dtype=dtype)
    ev_cfg = s["formulation_study"]["evaluation"]
    terms, evals = U.attach(s, entry, prob, dev, dtype, seed)
    tr = Trainer(net, prob, s, out_dir, eval_fn=lambda m: P2.evaluate2d(prob, m, full=False), tag=tag, data_terms=terms)
    scales = prob.fit_normalisation(net, tr.batch(0))
    t0 = time.perf_counter()
    res = tr.run()
    wall = time.perf_counter() - t0
    ev = P2.evaluate2d(prob, net, int(ev_cfg["n_wall"]), full=True)
    val = P2.validation_loss(prob, net, int(ev_cfg["n_val"]), int(ev_cfg["n_wall"]), 7_777_777 + seed)
    last = tr.history[-1] if tr.history else {}
    rec = {
        "case": case,
        "variant": label,
        "seed": seed,
        "kind": form["kind"],
        "hard_bc": form["hard_bc"],
        "gauge": "none"
        if form["pressure_gauge"]["kind"] == "none"
        else f"{form['pressure_gauge']['kind']}-{form['pressure_gauge']['mode']}",
        "normalise": form["normalise"]["kind"],
        "scales": scales,
        "status": res["status"],
        "wall_s": wall,
        "n_params": net.n_params(),
        "final_losses": {k: v for k, v in last.items() if k == "total" or k.startswith(("pde_", "bc_", "gauge"))},
        **ev,
        **val,
        "user_data": U.evaluate_all(prob, net, evals),
    }
    tr_ = {k: v for k, v in prob.truth.items() if k != "exact_fn"}  # torch callables are not picklable
    Xg, Yg = np.meshgrid(tr_["x"], tr_["y"])
    pts = torch.as_tensor(np.stack([Xg.ravel(), Yg.ravel()], 1), dtype=dtype, device=dev)
    pred = P2._batched(prob, net, pts, 0, P2.FIELDS)
    fields = {k: pred[k].reshape(Xg.shape) for k in P2.FIELDS}
    np.savez(out_dir / "prediction.npz", x=tr_["x"], y=tr_["y"], **fields)
    return rec, {"history": tr.history, "pred": fields, "truth": tr_}


def redraw(met: dict, saved: dict, rdir: Path, tag: str) -> list[str]:
    """Phase 7: the figures of a saved run, without training: predictions from each run's prediction.npz, histories
    from its log.csv, the truth rebuilt (exact manufactured solution) or loaded (reference_C_cavity.npz)."""
    from aerosynapse.thermal.utils import replay as RP

    apply_style()
    truths = {}
    for case in dict.fromkeys(r["case"] for r in met["runs"]):
        reference = None
        if case == "C_cavity":
            with np.load(rdir / "reference_C_cavity.npz") as z:
                reference = {k: z[k] for k in z.files} | met["references"]["C_cavity"]
        entry = next(e for e in saved["formulation_study"]["variants"])
        _, form = SP.formulation_variant(saved, entry)
        prob = P2.build_problem2d(case, saved, form, reference)
        truths[case] = {k: v for k, v in prob.truth.items() if k != "exact_fn"}
    extras = {}
    for r in met["runs"]:
        d = rdir / "runs" / f"{r['case']}_{r['variant']}_s{r['seed']}"
        with np.load(d / "prediction.npz") as z:
            pred = {k: z[k] for k in P2.FIELDS}
        extras[(r["case"], r["variant"], r["seed"])] = {
            "history": RP.history(d / "log.csv"),
            "pred": pred,
            "truth": truths[r["case"]],
        }
    return figures(met["runs"], extras, saved, tag, RP.target(rdir.name))


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
    tag = a.tag or a.profile
    custom = bool(a.config or a.sets)
    if a.table_only:
        rdir = ROOT / "results" / f"p4_2_formulations_{tag}"
        met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
        saved = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
        # the docs table belongs to the default study only (tag == profile name); any other tag writes into its results dir
        out = (
            rdir / "formulations.md"
            if (custom or tag != saved["profile"])
            else ROOT / "docs" / "tables" / "phase4_2_formulations.md"
        )
        formulation_table(met["runs"], met["references"], saved, out, tag)
        print(f"wrote {out}")
        for f in redraw(met, saved, rdir, tag):
            print(f)
        return
    spec = SP.load_spec(a.config or None, a.profile, SP.parse_set(a.sets))
    fs = spec["formulation_study"]
    if a.describe:
        print(SP.describe(spec))
        print(yaml.safe_dump({k: spec[k] for k in ("formulation", "problems2d", "formulation_study")}, sort_keys=False))
        return
    SP.check_user_files(spec, fs["cases"])
    apply_style()
    print(SP.describe(spec), flush=True)
    with Experiment(f"p4_2_formulations_{tag}", spec) as ex:
        refs: dict = {}
        ref_info: dict = {}
        for case in [SP.case_name(c) for c in fs["cases"]]:
            if case == "C_cavity":
                pc = spec["problems2d"]["C_cavity"]
                rc = pc["reference"]
                t0 = time.perf_counter()
                par = {k: float(v) for k, v in pc["parameters"].items()}
                refs[case] = cavity_reference(
                    par, P2.cavity_conditions(pc), int(rc["grid"]), int(rc["refine"]), float(rc["tol"])
                )
                r = refs[case]
                np.savez(ex.dir / "reference_C_cavity.npz", **{k: r[k] for k in ("x", "y", *P2.FIELDS)})
                ref_info[case] = {
                    "kind": "steady Newton FV (central energy convection, bc_order 2)",
                    "n_coarse": r["n_coarse"],
                    "n_fine": r["n_fine"],
                    "coarse_fine_diff": r["coarse_fine_diff"],
                    "newton": r["newton"],
                    "nu_top": r["nu_top"],
                    "nu_bottom": r["nu_bottom"],
                    "wall_s": time.perf_counter() - t0,
                }
                print(
                    f"[reference C_cavity] {r['n_coarse']}/{r['n_fine']} coarse-fine diff {r['coarse_fine_diff']}",
                    flush=True,
                )
            else:
                ref_info[case] = {"kind": "exact manufactured solution", "n_grid": fs["evaluation"]["n_grid"]}
        jobs = []
        for entry_case in fs["cases"]:
            case = SP.case_name(entry_case)
            for entry in fs["variants"]:
                label, form = SP.formulation_variant(spec, entry)
                for seed in fs["seeds"]:
                    kw = dict(
                        spec=spec,
                        case=entry_case,
                        label=label,
                        form=form,
                        seed=seed,
                        reference=refs.get(case),
                        out_dir=ex.dir / "runs" / f"{case}_{label}_s{seed}",
                        tag=tag,
                    )
                    jobs.append((case, label, seed, ("aerosynapse.thermal.experiments.p4_2_formulations", "train_variant", kw)))

        def show(i: int, out: tuple, sec: float) -> None:
            rec = out[0]
            print(
                f"[{rec['case']} | {rec['variant']:20s} s{rec['seed']}] worst(u,v,th)={rec['rel_l2_max']:.2e} "
                f"u={rec['rel_l2_u']:.2e} v={rec['rel_l2_v']:.2e} th={rec['rel_l2_theta']:.2e} p={rec['rel_l2_p']:.2e} "
                f"div={rec['max_div']:.1e} val={rec['val_loss']:.2e} {rec['wall_s']:.0f}s {rec['status']}",
                flush=True,
            )

        outs = run_tasks([j[3] for j in jobs], spec["execution"], show)
        recs: list[dict] = []
        extras: dict = {}
        for (case, label, seed, _), (rec, extra) in zip(jobs, outs, strict=True):
            recs.append(rec)
            extras[(case, label, seed)] = extra
            keys = ("case", "variant", "seed", "rel_l2_max", "rel_l2_p", "max_div", "val_loss", "wall_s", "status")
            ex.log(**{k: rec[k] for k in keys})
        ex.metrics.update(
            {"runs": recs, "references": ref_info, "git": git_hash(), "torch": torch.__version__, "n_runs": len(recs)}
        )
        ex.metrics["figures"] = figures(recs, extras, spec, tag, ex)
        table = ex.dir / "formulations.md" if custom else ROOT / "docs" / "tables" / "phase4_2_formulations.md"
        formulation_table(recs, ref_info, spec, table, tag)
        ex.metrics["table"] = str(table)
    print(f"wrote {table}")


def figures(recs: list[dict], extras: dict, spec: dict, tag: str, ex: Experiment) -> list[str]:
    paths = []
    cases = list(dict.fromkeys(r["case"] for r in recs))
    labels = list(dict.fromkeys(r["variant"] for r in recs))
    fig, axs = plt.subplots(1, len(cases), figsize=(5.0 * len(cases), 3.6), squeeze=False)
    w = 0.38
    for ax, case in zip(axs[0], cases, strict=True):
        for j, (key, name) in enumerate((("rel_l2_max", "worst of u, v, θ"), ("rel_l2_p", "p (mean-free)"))):
            m = [
                np.mean([r[key] for r in recs if r["case"] == case and r["variant"] == lb] or [np.nan]) for lb in labels
            ]
            ax.bar(np.arange(len(labels)) + (j - 0.5) * w, np.maximum(m, 1e-16), w, label=name, color=COLORS[j])
        ax.axhline(spec["evaluation"]["target_rel_l2"], color="k", ls=":", lw=0.8)
        ax.set(yscale="log", xticks=range(len(labels)), ylabel="relative $L_2$ error", title=case)
        ax.set_xticklabels(labels, rotation=50, fontsize=7)
        ax.legend(fontsize=7)
    paths += save_fig(fig, f"p4_2_error_by_formulation_{tag}", ex.dir)
    fig, axs = plt.subplots(1, len(cases), figsize=(5.0 * len(cases), 3.4), squeeze=False)
    for ax, case in zip(axs[0], cases, strict=True):
        for i, lb in enumerate(labels):
            h = next((v["history"] for (c, l_, s), v in extras.items() if c == case and l_ == lb), None)
            if h:
                ax.semilogy([max(r["total"], 1e-30) for r in h], color=COLORS[i % len(COLORS)], label=lb)
        ax.set(xlabel="log row (Adam then L-BFGS)", ylabel="training loss (not comparable across variants)", title=case)
        ax.legend(fontsize=6)
    paths += save_fig(fig, f"p4_2_loss_curves_{tag}", ex.dir)
    for case in cases:
        sel = [r for r in recs if r["case"] == case]
        best = min(sel, key=lambda r: r["rel_l2_max"])
        e = extras[(case, best["variant"], best["seed"])]
        tr, pr = e["truth"], e["pred"]
        fig, axs = plt.subplots(3, 4, figsize=(13, 9))
        for j, k in enumerate(P2.FIELDS):
            ref = tr["fields"][k] - (tr["fields"][k].mean() if k == "p" else 0.0)
            pk = pr[k] - (pr[k].mean() if k == "p" else 0.0)
            ext = (0, 1, 0, 1)
            for i, (arr, ttl, cm) in enumerate(
                ((ref, "truth", SEQ_CMAP), (pk, f"PINN {best['variant']}", SEQ_CMAP), (pk - ref, "error", DIV_CMAP))
            ):
                lim = np.abs(arr).max() if cm == DIV_CMAP else None
                im = axs[i, j].imshow(arr, origin="lower", extent=ext, cmap=cm, vmin=-lim if lim else None, vmax=lim)
                axs[i, j].set(title=f"{k}: {ttl}" + (" (mean-free)" if k == "p" else ""), xticks=[0, 1], yticks=[0, 1])
                fig.colorbar(im, ax=axs[i, j], fraction=0.046)
        fig.suptitle(
            f"{case}: best variant {best['variant']} (worst-field rel L2 {best['rel_l2_max']:.2e}) vs {tr['kind']}"
        )
        paths += save_fig(fig, f"p4_2_fields_{case}_{tag}", ex.dir)
    return [str(p) for p in paths]


if __name__ == "__main__":
    main()
