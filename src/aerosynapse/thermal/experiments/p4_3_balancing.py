"""Step 4.3: loss-balancing suite - fixed, learning-rate annealing, GradNorm, NTK-trace, SoftAdapt, ReLoBRaLo,
uncertainty weighting and self-adaptive weights - on 1D (analytic truth) and 2D (exact MMS / Newton FV truth) cases.

    python experiments/p4_3_balancing.py --profile smoke
    python experiments/p4_3_balancing.py --config my.yaml --tag mine
    python experiments/p4_3_balancing.py --set 'balance_study.methods=[fixed,relobralo]' --set loss.balancing.relobralo.temperature=0.01
    python experiments/p4_3_balancing.py --describe
    python experiments/p4_3_balancing.py --table-only --tag smoke

Everything comes from the layered PINN spec (`loss.balancing` method blocks, `balance_study`, `problems`, `problems2d`,
`formulation`, network/sampling/training). Every method is compared by the same truth-free validation metric (1D: the
fixed-weight loss on an independent point set; 2D: the formulation-independent validation loss of Step 4.2) and by the
error against the truth. Outputs: results/p4_3_balancing_<tag>/, docs/tables/phase4_3_balancing.md (default settings
only), figures.
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

from aerosynapse.thermal.experiments.p4_report import balance_table
from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import problems2d as P2
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn import userdata as U
from aerosynapse.thermal.pinn.architectures import build_network
from aerosynapse.thermal.pinn.balancing import group_of
from aerosynapse.thermal.pinn.metrics import evaluate_model
from aerosynapse.thermal.pinn.networks import resolve_device
from aerosynapse.thermal.pinn.ref2d import cavity_reference
from aerosynapse.thermal.pinn.trainer import Trainer
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz.style import apply_style, save_fig

COLORS = ["#000000", "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#999999", "#F0E442"]


def case_label(c: dict) -> str:
    return f"{c['problem']} {c['value']:g}" if "value" in c else c["problem"]


def grad_spread(tr: Trainer, batch: dict, eff: dict[str, float]) -> tuple[float, float]:
    """max/min over components of ||grad_theta L_k|| (unweighted) and of ||grad_theta g_k w_k L_k|| (weighted)."""
    _, comp = tr.fixed_losses(batch)
    params = [p for p in tr.model.parameters() if p.requires_grad]
    n = {}
    for k, v in comp.items():
        g = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
        n[k] = float(torch.sqrt(sum((x**2).sum() for x in g if x is not None)))
    pos = {k: v for k, v in n.items() if v > 0}
    if len(pos) < 2:
        return float("nan"), float("nan")
    w = {k: pos[k] * eff[k] for k in pos}
    return max(pos.values()) / min(pos.values()), max(w.values()) / min(w.values())


def train_case(spec: dict, case: dict, label: str, bal: dict, seed: int, refs: dict, out_dir: Path, tag: str):
    s = SP.deep_merge(spec, {"loss": {"balancing": bal}})
    s["seed"] = seed
    dev = resolve_device(s["device"])
    dtype = torch.float64 if s["dtype"] == "float64" else torch.float32
    two_d = case["problem"] in SP.PROBLEMS2D
    if two_d:
        form = SP.deep_merge(s["formulation"], case.get("formulation", {}) or {})
        prob = P2.build_problem2d(case["problem"], s, form, refs.get(case["problem"]))
        out_dim, lo, hi = prob.out_dim, list(prob.lo), list(prob.hi)

        def eval_fn(m):
            return P2.evaluate2d(prob, m, full=False)
    else:
        prob = P.build_problem(case["problem"], s["problems"][case["problem"]], case["value"])
        out_dim, lo, hi = 1, prob.lo, prob.hi

        def eval_fn(m):
            return {k: v for k, v in evaluate_model(m, prob, case["value"], 201, 11).items() if not k.startswith("_")}

    torch.manual_seed(seed)
    net = build_network(s["network"], lo, hi, out_dim).to(device=dev, dtype=dtype)
    terms, evals = U.attach(s, case, prob, dev, dtype, seed)
    tr = Trainer(net, prob, s, out_dir, eval_fn=eval_fn, tag=tag, data_terms=terms)
    if two_d:
        prob.fit_normalisation(net, tr.batch(0))
    t0 = time.perf_counter()
    res = tr.run()
    wall = time.perf_counter() - t0
    rec: dict = {
        "case": case_label(case),
        "problem": case["problem"],
        "dim": 2 if two_d else 1,
        "method": label,
        "method_kind": bal["method"],
        "seed": seed,
        "status": res["status"],
        "wall_s": wall,
        "n_params": net.n_params(),
    }
    if two_d:
        ev = P2.evaluate2d(prob, net, int(s["formulation_study"]["evaluation"]["n_wall"]), full=True)
        val = P2.validation_loss(prob, net, int(s["formulation_study"]["evaluation"]["n_val"]), 64, 7_777_777 + seed)
        rec.update(
            {**ev, **val, "err": ev["rel_l2_max"], "formulation": form["kind"] + ("+hard" if form["hard_bc"] else "")}
        )
    else:
        ev = {
            k: v
            for k, v in evaluate_model(net, prob, case["value"], s["evaluation"]["nx"], s["evaluation"]["nt"]).items()
            if not k.startswith("_")
        }
        sp_ = s["sampling"]
        vb = prob.sample(sp_["n_pde"], sp_["n_ic"], sp_["n_bc"], "random", 7_777_777 + seed, dev, dtype)
        v_total, _ = tr.fixed_losses(vb, include_data=False)  # physics-only, fixed weights
        rec.update({**ev, "val_loss": float(v_total.detach()), "err": ev["rel_l2"]})
    names = (
        [
            k
            for k in tr.history[-1]
            if not k.startswith("w_") and k.split("_")[0] in ("pde", "ic", "bc", "gauge", "data")
        ]
        if tr.history
        else []
    )
    if tr.balancer is not None:
        eff = tr.balancer.effective()
        rec.update({"n_updates": tr.balancer.n_updates, "n_clamped": tr.balancer.n_clamped})
    else:
        eff = {k: tr.weights[group_of(k)] for k in names}
        rec.update({"n_updates": 0, "n_clamped": 0})
    rec["final_weights"] = eff
    rec["user_data"] = U.evaluate_all(prob, net, evals)
    rec["weight_spread"] = max(eff.values()) / max(min(eff.values()), 1e-300) if eff else float("nan")
    last = tr.batch(max(s["training"]["adam"]["epochs"] - 1, 0))
    rec["grad_spread_unweighted"], rec["grad_spread_weighted"] = grad_spread(tr, last, eff)
    return rec, tr.history


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
        rdir = ROOT / "results" / f"p4_3_balancing_{tag}"
        met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
        saved = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
        # the docs table belongs to the default study only (tag == profile name); any other tag writes into its results dir
        out = (
            rdir / "balancing.md"
            if (custom or tag != saved["profile"])
            else ROOT / "docs" / "tables" / "phase4_3_balancing.md"
        )
        balance_table(met["runs"], met["references"], saved, out, tag)
        print(f"wrote {out}")
        # figures from the same saved run (training histories = each run's log.csv); nothing is trained
        from aerosynapse.thermal.utils import replay as RP

        hists = {}
        for r in met["runs"]:
            log = rdir / "runs" / f"{r['case'].replace(' ', '_')}_{r['method']}_s{r['seed']}" / "log.csv"
            if log.exists():
                hists[(r["case"], r["method"], r["seed"])] = RP.history(log)
        apply_style()
        for f in figures(met["runs"], hists, saved, tag, RP.target(rdir.name)):
            print(f)
        return
    spec = SP.load_spec(a.config or None, a.profile, SP.parse_set(a.sets))
    SP.check_balance_study_runnable(spec)
    SP.check_user_files(spec, spec["balance_study"]["cases"])
    st = spec["balance_study"]
    if a.describe:
        print(SP.describe(spec))
        print(yaml.safe_dump({"loss": spec["loss"], "balance_study": st}, sort_keys=False))
        return
    apply_style()
    print(SP.describe(spec), flush=True)
    with Experiment(f"p4_3_balancing_{tag}", spec) as ex:
        refs, ref_info = {}, {}
        for c in st["cases"]:
            if c["problem"] == "C_cavity" and "C_cavity" not in refs:
                pc = spec["problems2d"]["C_cavity"]
                rc = pc["reference"]
                par = {k: float(v) for k, v in pc["parameters"].items()}
                refs["C_cavity"] = r = cavity_reference(
                    par, P2.cavity_conditions(pc), int(rc["grid"]), int(rc["refine"]), float(rc["tol"])
                )
                ref_info["C_cavity"] = {
                    "n_coarse": r["n_coarse"],
                    "n_fine": r["n_fine"],
                    "coarse_fine_diff": r["coarse_fine_diff"],
                    "newton": r["newton"],
                }
                print(f"[reference C_cavity] {r['n_coarse']}/{r['n_fine']} diff {r['coarse_fine_diff']}", flush=True)
        jobs = []
        for c in st["cases"]:
            for entry in st["methods"]:
                label, bal = SP.balancing_variant(spec, entry)
                for seed in st["seeds"]:
                    out_dir = ex.dir / "runs" / f"{case_label(c).replace(' ', '_')}_{label}_s{seed}"
                    kw = dict(spec=spec, case=c, label=label, bal=bal, seed=seed, refs=refs, out_dir=out_dir, tag=tag)
                    jobs.append((label, seed, ("aerosynapse.thermal.experiments.p4_3_balancing", "train_case", kw)))

        def show(i: int, out: tuple, sec: float) -> None:
            rec = out[0]
            print(
                f"[{rec['case']:13s} | {rec['method']:14s} s{rec['seed']}] err={rec['err']:.3e} val={rec['val_loss']:.2e} "
                f"wspread={rec['weight_spread']:.1e} gspread {rec['grad_spread_unweighted']:.1e}->"
                f"{rec['grad_spread_weighted']:.1e} clamped={rec['n_clamped']} {rec['wall_s']:.0f}s {rec['status']}",
                flush=True,
            )

        outs = run_tasks([j[2] for j in jobs], spec["execution"], show)
        recs, hists = [], {}
        for (label, seed, _), (rec, hist) in zip(jobs, outs, strict=True):
            recs.append(rec)
            hists[(rec["case"], label, seed)] = hist
            keys = ("case", "method", "seed", "err", "val_loss", "weight_spread", "wall_s", "status")
            ex.log(**{k: rec[k] for k in keys})
        ex.metrics.update(
            {"runs": recs, "references": ref_info, "git": git_hash(), "torch": torch.__version__, "n_runs": len(recs)}
        )
        ex.metrics["figures"] = figures(recs, hists, spec, tag, ex)
        table = ex.dir / "balancing.md" if custom else ROOT / "docs" / "tables" / "phase4_3_balancing.md"
        balance_table(recs, ref_info, spec, table, tag)
        ex.metrics["table"] = str(table)
    print(f"wrote {table}")


def figures(recs: list[dict], hists: dict, spec: dict, tag: str, ex: Experiment) -> list[str]:
    paths = []
    cases = list(dict.fromkeys(r["case"] for r in recs))
    methods = list(dict.fromkeys(r["method"] for r in recs))
    fig, axs = plt.subplots(1, len(cases), figsize=(3.6 * len(cases), 3.6), squeeze=False)
    for ax, c in zip(axs[0], cases, strict=True):
        m = [np.mean([r["err"] for r in recs if r["case"] == c and r["method"] == mt] or [np.nan]) for mt in methods]
        ax.bar(range(len(methods)), np.maximum(m, 1e-16), color=[COLORS[i % len(COLORS)] for i in range(len(methods))])
        ax.axhline(spec["evaluation"]["target_rel_l2"], color="k", ls=":", lw=0.8)
        ax.set(yscale="log", xticks=range(len(methods)), ylabel="relative $L_2$ error (worst field in 2D)", title=c)
        ax.set_xticklabels(methods, rotation=60, fontsize=7)
    paths += save_fig(fig, f"p4_3_error_by_method_{tag}", ex.dir)
    # error during training (evaluation rows) per case
    fig, axs = plt.subplots(1, len(cases), figsize=(3.6 * len(cases), 3.4), squeeze=False)
    for ax, c in zip(axs[0], cases, strict=True):
        for i, mt in enumerate(methods):
            h = next((v for (cc, ll, _), v in hists.items() if cc == c and ll == mt), None)
            if not h:
                continue
            key = "rel_l2_max" if any("rel_l2_max" in r for r in h) else "rel_l2"
            pts = [(j, r[key]) for j, r in enumerate(h) if key in r]
            if pts:
                ax.semilogy(*zip(*pts, strict=True), "-o", ms=2, color=COLORS[i % len(COLORS)], label=mt)
        ax.set(xlabel="log row (Adam then L-BFGS)", ylabel="error during training", title=c)
        ax.legend(fontsize=5)
    paths += save_fig(fig, f"p4_3_error_history_{tag}", ex.dir)
    # weight trajectories for every adaptive method, one figure per case
    for c in cases:
        adaptive = [
            mt
            for mt in methods
            if any(
                k.startswith("w_")
                for (cc, ll, _), h in hists.items()
                if cc == c and ll == mt
                for k in (h[0] if h else {})
            )
        ]
        if not adaptive:
            continue
        nc = min(4, len(adaptive))
        nr = -(-len(adaptive) // nc)
        fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.8 * nr), squeeze=False)
        for ax in axs.ravel()[len(adaptive) :]:
            ax.axis("off")
        for ax, mt in zip(axs.ravel(), adaptive, strict=False):
            h = next(v for (cc, ll, _), v in hists.items() if cc == c and ll == mt)
            keys = [k for k in h[0] if k.startswith("w_")]
            for j, k in enumerate(keys):
                ax.semilogy([max(r[k], 1e-30) for r in h], color=COLORS[(j + 1) % len(COLORS)], label=k[2:])
            ax.set(title=mt, xlabel="log row", ylabel="effective weight")
            ax.legend(fontsize=5)
        fig.suptitle(f"{c}: adaptive weights (frozen during L-BFGS)")
        paths += save_fig(fig, f"p4_3_weights_{c.replace(' ', '_')}_{tag}", ex.dir)
    return [str(p) for p in paths]


if __name__ == "__main__":
    main()
