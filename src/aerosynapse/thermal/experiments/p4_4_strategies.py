"""Step 4.4: sampling and training strategies - Sobol / random / LHS collocation, random resampling, residual-based
adaptive sampling (RAD, RAR-D), causal temporal weighting, time-marching, curriculum in Pe/Re, and optimiser
strategies (Adam schedule, gradient clipping, with/without L-BFGS) - on 1D and 2D cases.

    python experiments/p4_4_strategies.py --profile smoke
    python experiments/p4_4_strategies.py --config my.yaml --tag mine
    python experiments/p4_4_strategies.py --set 'strategy_study.strategies=[{name: sobol}, {name: rad, sampling: {adaptive: {method: rad}}}]'
    python experiments/p4_4_strategies.py --describe
    python experiments/p4_4_strategies.py --table-only --tag smoke

A strategy is {name, <overrides of sampling / training / loss / network>}; every run executes in its own process (D57)
and may use user data (`data:` in a case). Strategies that do not apply to a case (causal / time-marching on steady
problems, curriculum without a Pe/Re parameter) are recorded as n/a, not run.
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

from aerosynapse.thermal.experiments.p4_report import strategy_table
from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import problems2d as P2
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn import strategies as ST
from aerosynapse.thermal.pinn import userdata as U
from aerosynapse.thermal.pinn.architectures import build_network
from aerosynapse.thermal.pinn.metrics import evaluate_model
from aerosynapse.thermal.pinn.networks import resolve_device
from aerosynapse.thermal.pinn.ref2d import cavity_reference
from aerosynapse.thermal.pinn.trainer import Trainer
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz.style import apply_style, save_fig

COLORS = ["#000000", "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#999999", "#F0E442", "#8B4513"]
OVERRIDABLE = ("sampling", "training", "loss", "network")


def case_label(c: dict) -> str:
    return f"{c['problem']} {c['value']:g}" if "value" in c else c["problem"]


def strategy_spec(spec: dict, strat: dict) -> dict:
    return SP.deep_merge(spec, {k: v for k, v in strat.items() if k in OVERRIDABLE})


def applicable(spec: dict, case: dict, strat: dict) -> tuple[bool, str]:
    s = strategy_spec(spec, strat)
    two_d = case["problem"] in SP.PROBLEMS2D
    transient = not two_d and case["problem"] != "A_steady"
    tr = s["training"]
    if (tr["causal"]["enabled"] or tr["time_marching"]["enabled"]) and not transient:
        return False, "causal weighting / time-marching need a transient problem"
    if tr["curriculum"]["enabled"] and curriculum_param(s, case) is None:
        return False, "no Pe/Re parameter for the curriculum (set training.curriculum.parameter)"
    if tr["time_marching"]["enabled"] and case.get("data"):
        roles = [
            s["user_data"]["sources"][n].get("role", "evaluate")
            for n in ([case["data"]] if isinstance(case["data"], str) else case["data"])
        ]
        if any(r in ("train", "both") for r in roles):
            return False, "time-marching with user TRAINING data is not supported (use role: evaluate)"
    return True, ""


def curriculum_param(s: dict, case: dict) -> str | None:
    want = s["training"]["curriculum"]["parameter"]
    if case["problem"] in SP.PROBLEMS2D:
        names = list(SP.PROBLEMS2D[case["problem"]])
        name = "Re" if want == "auto" else want
        return name if name in names else None
    pname = SP.PROBLEMS[case["problem"]]
    if want == "auto":
        return pname if pname == "Pe" else None
    return pname if want == pname else None


def _device(s: dict) -> tuple:
    return resolve_device(s["device"]), torch.float64 if s["dtype"] == "float64" else torch.float32


def build(s: dict, case: dict, ref: dict | None, value: float | None = None):
    """Problem for a case; ``value`` overrides the case parameter (curriculum stages)."""
    if case["problem"] in SP.PROBLEMS2D:
        form = SP.deep_merge(s["formulation"], case.get("formulation", {}) or {})
        if value is not None:
            pname = curriculum_param(s, case)
            s = SP.deep_merge(s, {"problems2d": {case["problem"]: {"parameters": {pname: float(value)}}}})
        return P2.build_problem2d(case["problem"], s, form, ref), form
    v = case["value"] if value is None else value
    return P.build_problem(case["problem"], s["problems"][case["problem"]], v), None


def evaluate_case(s: dict, case: dict, prob, model, seed: int) -> dict:
    dev, dtype = _device(s)
    if hasattr(prob, "fields"):
        ev_cfg = s["formulation_study"]["evaluation"]
        ev = P2.evaluate2d(prob, model, int(ev_cfg["n_wall"]), full=True)
        val = P2.validation_loss(prob, model, int(ev_cfg["n_val"]), 64, 7_777_777 + seed)
        return {**ev, **val, "err": ev["rel_l2_max"]}
    ev = {
        k: v
        for k, v in evaluate_model(model, prob, case["value"], s["evaluation"]["nx"], s["evaluation"]["nt"]).items()
        if not k.startswith("_")
    }
    sp_ = s["sampling"]
    vb = prob.sample(sp_["n_pde"], sp_["n_ic"], sp_["n_bc"], "random", 7_777_777 + seed, dev, dtype)
    comp = P.component_losses(model, prob, vb)
    return {**ev, "val_loss": float(P.weighted_total(comp, s["loss"]["weights"]).detach()), "err": ev["rel_l2"]}


def train_strategy(spec: dict, case: dict, label: str, strat: dict, seed: int, refs: dict, out_dir: Path, tag: str):
    """One run (own process by default). Returns (record, history)."""
    s = strategy_spec(spec, strat)
    s["seed"] = seed
    dev, dtype = _device(s)
    ref = refs.get(case["problem"])
    tr_cfg = s["training"]
    t0 = time.perf_counter()
    info: dict = {}
    history: list = []
    base, form = build(s, case, ref)
    if tr_cfg["time_marching"]["enabled"]:
        K = int(tr_cfg["time_marching"]["windows"])
        ti = base.input_names.index("t")
        edges = list(np.linspace(base.lo[ti], base.hi[ti], K + 1))
        epochs = ST.split_budget(tr_cfg["adam"]["epochs"], K)
        lbfgs = ST.split_budget(tr_cfg["lbfgs"]["iterations"], K)
        sp_ = s["sampling"]
        models, prev, status = [], None, "ok"
        for k in range(K):
            wp = ST.WindowProblem(base, edges[k], edges[k + 1], prev)
            sk = SP.deep_merge(
                s,
                {
                    "training": {"adam": {"epochs": epochs[k]}, "lbfgs": {"iterations": lbfgs[k]}},
                    "sampling": {"n_pde": -(-sp_["n_pde"] // K), "n_bc": -(-sp_["n_bc"] // K)},
                },
            )
            torch.manual_seed(seed * 100 + k)
            net = build_network(sk["network"], wp.lo, wp.hi).to(device=dev, dtype=dtype)
            trn = Trainer(net, wp, sk, out_dir / f"window_{k}", tag=tag)
            res = trn.run()
            history += [{**r, "window": k} for r in trn.history]
            status = res["status"] if res["status"] != "ok" else status
            models.append(net)
            prev = net
        model = ST.CompositeModel(models, edges, ti)
        prob = base
        info = {"windows": K, "status": status}
        _, evals = U.attach(s, case, base, dev, dtype, seed)
    elif tr_cfg["curriculum"]["enabled"]:
        cc = tr_cfg["curriculum"]
        pname = curriculum_param(s, case)
        target = float(base.params[pname]) if hasattr(base, "params") else float(case["value"])
        values = ST.curriculum_values(target, float(cc["start_fraction"]), int(cc["stages"]))
        epochs = ST.split_budget(tr_cfg["adam"]["epochs"], len(values))
        torch.manual_seed(seed)
        out_dim = base.out_dim if hasattr(base, "out_dim") else 1
        model = build_network(s["network"], list(base.lo), list(base.hi), out_dim).to(device=dev, dtype=dtype)
        status = "ok"
        for k, v in enumerate(values):
            last = k == len(values) - 1
            prob_k, _ = build(s, case, ref, value=v) if not last else (base, form)
            sk = SP.deep_merge(
                s,
                {
                    "training": {
                        "adam": {"epochs": epochs[k]},
                        "lbfgs": {"iterations": tr_cfg["lbfgs"]["iterations"] if last else 0},
                    }
                },
            )
            terms, evals = U.attach(s, case, prob_k, dev, dtype, seed) if last else ([], [])
            trn = Trainer(model, prob_k, sk, out_dir / f"stage_{k}", tag=tag, data_terms=terms)
            if hasattr(prob_k, "fit_normalisation"):
                prob_k.fit_normalisation(model, trn.batch(0))
            res = trn.run()
            history += [{**r, "stage": k, pname: v} for r in trn.history]
            status = res["status"] if res["status"] != "ok" else status
        prob = base
        info = {"stages": values, "parameter": pname, "status": status}
    else:
        torch.manual_seed(seed)
        out_dim = base.out_dim if hasattr(base, "out_dim") else 1
        model = build_network(s["network"], list(base.lo), list(base.hi), out_dim).to(device=dev, dtype=dtype)
        terms, evals = U.attach(s, case, base, dev, dtype, seed)
        trn = Trainer(model, base, s, out_dir, tag=tag, data_terms=terms)
        if hasattr(base, "fit_normalisation"):
            base.fit_normalisation(model, trn.batch(0))
        res = trn.run()
        history = trn.history
        prob = base
        info = {"status": res["status"]}
        if trn.sampler is not None:
            info.update(
                {
                    "n_pde_final": ST._size(trn.sampler.points),
                    "sampler_updates": trn.sampler.n_updates,
                    "k": trn.sampler.k,
                    "c": trn.sampler.c,
                }
            )
        if trn.causal is not None:
            info.update(
                {
                    "causal_eps_final": trn.causal.eps,
                    "causal_min_w": trn.causal.last_min_w,
                    "causal_eps_steps": trn.causal.eps_history,
                }
            )
    wall = time.perf_counter() - t0
    rec = {
        "case": case_label(case),
        "problem": case["problem"],
        "dim": 2 if case["problem"] in SP.PROBLEMS2D else 1,
        "strategy": label,
        "seed": seed,
        "wall_s": wall,
        "n_params": sum(p.numel() for p in model.parameters()),
        **info,
        **evaluate_case(s, case, prob, model, seed),
        "user_data": U.evaluate_all(prob, model, evals),
    }
    if form is not None:
        rec["formulation"] = form["kind"] + ("+hard" if form["hard_bc"] else "")
    return rec, history


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
        rdir = ROOT / "results" / f"p4_4_strategies_{tag}"
        met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
        saved = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
        out = (
            rdir / "strategies.md"
            if (custom or tag != saved["profile"])
            else ROOT / "docs" / "tables" / "phase4_4_strategies.md"
        )
        strategy_table(met["runs"], met["skipped"], met["references"], saved, out, tag)
        print(f"wrote {out}")
        # figures from the same saved run (training histories = each run's log.csv); nothing is trained
        from aerosynapse.thermal.utils import replay as RP

        hists = {}
        for r in met["runs"]:
            log = rdir / "runs" / f"{r['case'].replace(' ', '_')}_{r['strategy']}_s{r['seed']}" / "log.csv"
            if log.exists():
                hists[(r["case"], r["strategy"], r["seed"])] = RP.history(log)
        apply_style()
        for f in figures(met["runs"], hists, saved, tag, RP.target(rdir.name)):
            print(f)
        return
    spec = SP.load_spec(a.config or None, a.profile, SP.parse_set(a.sets))
    st = spec["strategy_study"]
    SP.check_user_files(spec, st["cases"])
    SP.check_strategy_study_runnable(spec)
    if a.describe:
        print(SP.describe(spec))
        print(
            yaml.safe_dump(
                {"strategy_study": st, "sampling": spec["sampling"], "training": spec["training"]}, sort_keys=False
            )
        )
        return
    apply_style()
    print(SP.describe(spec), flush=True)
    with Experiment(f"p4_4_strategies_{tag}", spec) as ex:
        refs, ref_info = {}, {}
        if any(c["problem"] == "C_cavity" for c in st["cases"]):
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
        jobs, skipped = [], []
        for c in st["cases"]:
            for strat in st["strategies"]:
                label = strat["name"]
                ok, why = applicable(spec, c, strat)
                if not ok:
                    skipped.append({"case": case_label(c), "strategy": label, "reason": why})
                    continue
                for seed in st["seeds"]:
                    out_dir = ex.dir / "runs" / f"{case_label(c).replace(' ', '_')}_{label}_s{seed}"
                    kw = dict(
                        spec=spec, case=c, label=label, strat=strat, seed=seed, refs=refs, out_dir=out_dir, tag=tag
                    )
                    jobs.append(("aerosynapse.thermal.experiments.p4_4_strategies", "train_strategy", kw))
        for sk in skipped:
            print(f"[{sk['case']:13s} | {sk['strategy']:16s}] n/a: {sk['reason']}", flush=True)

        def show(i: int, out: tuple, sec: float) -> None:
            r = out[0]
            extra = ""
            if "n_pde_final" in r:
                extra = f" pts={r['n_pde_final']}"
            if "causal_eps_final" in r:
                extra += f" eps={r['causal_eps_final']:g}"
            print(
                f"[{r['case']:13s} | {r['strategy']:16s} s{r['seed']}] err={r['err']:.3e} val={r['val_loss']:.2e}{extra} {r['wall_s']:.0f}s {r['status']}",
                flush=True,
            )

        outs = run_tasks(jobs, spec["execution"], show)
        recs, hists = [], {}
        for rec, hist in outs:
            recs.append(rec)
            hists[(rec["case"], rec["strategy"], rec["seed"])] = hist
            ex.log(**{k: rec[k] for k in ("case", "strategy", "seed", "err", "val_loss", "wall_s", "status")})
        ex.metrics.update(
            {
                "runs": recs,
                "skipped": skipped,
                "references": ref_info,
                "git": git_hash(),
                "torch": torch.__version__,
                "n_runs": len(recs),
            }
        )
        ex.metrics["figures"] = figures(recs, hists, spec, tag, ex)
        table = ex.dir / "strategies.md" if custom else ROOT / "docs" / "tables" / "phase4_4_strategies.md"
        strategy_table(recs, skipped, ref_info, spec, table, tag)
        ex.metrics["table"] = str(table)
    print(f"wrote {table}")


def figures(recs: list[dict], hists: dict, spec: dict, tag: str, ex: Experiment) -> list[str]:
    paths = []
    cases = list(dict.fromkeys(r["case"] for r in recs))
    strats = list(dict.fromkeys(r["strategy"] for r in recs))
    fig, axs = plt.subplots(1, len(cases), figsize=(3.8 * len(cases), 3.7), squeeze=False)
    for ax, c in zip(axs[0], cases, strict=True):
        m = [np.mean([r["err"] for r in recs if r["case"] == c and r["strategy"] == s] or [np.nan]) for s in strats]
        ax.bar(
            range(len(strats)),
            np.where(np.isfinite(m), np.maximum(m, 1e-16), 0.0),
            color=[COLORS[i % len(COLORS)] for i in range(len(strats))],
        )
        ax.axhline(spec["evaluation"]["target_rel_l2"], color="k", ls=":", lw=0.8)
        ax.set(yscale="log", xticks=range(len(strats)), ylabel="relative $L_2$ error (worst field in 2D)", title=c)
        ax.set_xticklabels(strats, rotation=65, fontsize=6)
    paths += save_fig(fig, f"p4_4_error_by_strategy_{tag}", ex.dir)
    fig, axs = plt.subplots(1, len(cases), figsize=(3.8 * len(cases), 3.4), squeeze=False)
    for ax, c in zip(axs[0], cases, strict=True):
        for i, s in enumerate(strats):
            h = hists.get((c, s, recs[0]["seed"]))
            if not h:
                continue
            ys = [float(r["total"]) for r in h if r.get("total") not in (None, "")]
            ax.semilogy(np.maximum(ys, 1e-30), color=COLORS[i % len(COLORS)], label=s, lw=0.8)
        ax.set(xlabel="log row", ylabel="training loss (objectives differ)", title=c)
        ax.legend(fontsize=5)
    paths += save_fig(fig, f"p4_4_loss_curves_{tag}", ex.dir)
    return [str(p) for p in paths]


if __name__ == "__main__":
    main()
