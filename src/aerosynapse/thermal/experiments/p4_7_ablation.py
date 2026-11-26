"""Step 4.7: ablation of the Phase 4 recipe — leave-one-out and add-one-in, >= 3 seeds, mean +- std, training time
vs accuracy.

The recipe is a set of named components (ablation_study.components in configs/pinn/defaults.yaml), each a set of
overrides applied when the component is ON (default: exact_psi, grad_clip, ntk, rad, siren, data). Designs:

  leave_one_out   'full' = every component ON, then 'full-<c>' = all but c
  add_one_in      'base' = every component OFF (the settings of the config file), then 'base+<c>' = only c

Every configuration is trained with every seed (network initialisation and collocation; the snapshot data are fixed).
Training uses the Step 4.6 run functions (one network over the parameter box scored at the 4.5b interp / extrap test
points, or one test point scored on held-out cells); the 'data' component switches 4.6 snapshots on (hybrid mode).
Each configuration is validated as a whole (load_spec) and its problem and network are built once before any training.

    python experiments/p4_7_ablation.py --profile smoke [--config my.yaml] [--set key=value] [--tag name]
    python experiments/p4_7_ablation.py --describe        # the configurations and their components
    python experiments/p4_7_ablation.py --table-only --tag smoke
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from aerosynapse.thermal.experiments import p4_6_hybrid as HY
from aerosynapse.thermal.experiments.p4_5b_channel import case_design
from aerosynapse.thermal.experiments.p4_report import ablation_aggregate, ablation_table
from aerosynapse.thermal.pinn import channel2d as CH
from aerosynapse.thermal.pinn import hybrid as H
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz.style import apply_style, save_fig

HY_MOD = "aerosynapse.thermal.experiments.p4_6_hybrid"
REF_MOD = "aerosynapse.thermal.experiments.p4_5b_channel"
DESIGN_COLORS = {"leave_one_out": "#0072B2", "add_one_in": "#D55E00"}


def configurations(ab: dict) -> list[dict]:
    names = list(ab["components"])
    out = []
    if "leave_one_out" in ab["designs"]:
        out.append({"name": "full", "design": "leave_one_out", "on": names, "changed": None})
        out += [
            {"name": f"full-{c}", "design": "leave_one_out", "on": [n for n in names if n != c], "changed": c}
            for c in names
        ]
    if "add_one_in" in ab["designs"]:
        out.append({"name": "base", "design": "add_one_in", "on": [], "changed": None})
        out += [{"name": f"base+{c}", "design": "add_one_in", "on": [c], "changed": c} for c in names]
    only = ab.get("only")  # optional subset of configuration names (e.g. one configuration at a larger budget)
    if only:
        unknown = set(only) - {c["name"] for c in out}
        if unknown:
            raise ValueError(f"ablation_study.only: unknown configurations {sorted(unknown)}")
        out = [c for c in out if c["name"] in set(only)]
    return out


def config_spec(files: list | None, profile: str, cli: dict, ab: dict, on: list[str]) -> dict:
    """Full, validated spec of one configuration; hybrid_study is set so that the 4.6 run functions use exactly these
    settings (its own formulation / common / problems / data replaced by the ablation's)."""
    over = SP.deep_merge(cli, ab.get("common") or {})
    data = None
    for c in on:
        comp = dict(ab["components"][c])
        data = comp.pop("data", data)
        over = SP.deep_merge(over, comp)
    s = SP.load_spec(files or None, profile, over)
    hs = s["hybrid_study"]
    hs.update(
        {
            "formulation": {},
            "common": {},
            "problems": ab.get("problems") or {},
            "data": dict(ab["data"]),
            "user_data": {},
        }
    )
    if data:
        hs["parametric"] = {
            **hs["parametric"],
            "n_snapshots": int(data.get("n_snapshots", 1)),
            "n_data": int(data["n_data"]),
        }
    s["ablation_data"] = data
    return s


def preflight(s: dict, case: str, design: dict, kind: str, pt: dict) -> None:
    """Build the configuration's problem and network once (no training): fails before hours of compute."""
    s0 = HY.study_spec(s, 0)
    form = SP.deep_merge(s0["formulation"], s0["hybrid_study"]["formulation"])
    pc = s0["problems2d"][case]
    if kind == "parametric":
        prob = CH.build_channel(case, pc, form, design["dims"], design["fixed"], design["u_lim"])
    else:
        prob = CH.build_channel(case, pc, form, [], pt["values"], 1.0)
    HY._net(s0, prob)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--table-only", action="store_true", help="rebuild table and figure from saved metrics of --tag")
    a = ap.parse_args()
    tag = a.tag or a.profile
    custom = bool(a.config or a.sets)
    if a.table_only:
        rdir = ROOT / "results" / f"p4_7_ablation_{tag}"
        met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
        saved = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
        out = (
            rdir / "ablation.md"
            if (custom or tag != saved["profile"])
            else ROOT / "docs" / "tables" / "phase4_7_ablation.md"
        )
        ablation_table(met["records"], saved, out, tag)
        print(f"wrote {out}")
        from aerosynapse.thermal.utils import replay as RP  # the figure from the same saved records; nothing is trained

        apply_style()
        for f in figures(met["records"], tag, RP.target(rdir.name)):
            print(f)
        return
    cli = SP.parse_set(a.sets)
    base = SP.load_spec(a.config or None, a.profile, cli)
    ab, case = base["ablation_study"], base["ablation_study"]["case"]
    confs = configurations(ab)
    specs = {c["name"]: config_spec(a.config, a.profile, cli, ab, c["on"]) for c in confs}
    design = case_design(base, case)
    if ab["kind"] == "per_point" and ab["point"] >= len(design["points"]):
        raise ValueError(f"ablation_study.point = {ab['point']}: {case} has {len(design['points'])} test points")
    pt = design["points"][ab["point"]]
    datas = [s["ablation_data"] for s in specs.values() if s["ablation_data"]]
    n_snap = max((int(d.get("n_snapshots", 1)) for d in datas), default=0) if ab["kind"] == "parametric" else 0
    tpts = HY.train_design_points(base, case)[:n_snap]
    for c in confs:
        preflight(specs[c["name"]], case, design, ab["kind"], pt)
    if a.describe:
        print(yaml.safe_dump({"ablation_study": ab}, sort_keys=False))
        for c in confs:
            s = specs[c["name"]]
            print(
                f"{c['name']:16s} {c['design']:14s} on={c['on']}  formulation={s['formulation']['kind']}/hard={s['formulation']['hard_bc']} "
                f"arch={s['network']['arch']} balancing={s['loss']['balancing']['method']} adaptive={s['sampling']['adaptive']['method']} "
                f"grad_clip={s['training']['adam']['grad_clip']} data={s['ablation_data']}"
            )
        print(
            f"{len(confs)} configurations x {len(ab['seeds'])} seeds = {len(confs) * len(ab['seeds'])} trainings; preflight ok"
        )
        return
    apply_style()
    pcfg = SP.deep_merge(base["problems2d"][case], (ab.get("problems") or {}).get(case) or {})
    with Experiment(f"p4_7_ablation_{tag}", base) as ex:
        need = [(p, "test") for p in (design["points"] if ab["kind"] == "parametric" else [pt])] + [
            (p, "snapshot") for p in tpts
        ]
        refs_all = run_tasks(
            [(REF_MOD, "reference_task", dict(values=p["values"], pcfg=pcfg)) for p, _ in need],
            base["execution"],
            lambda i, r, sec: print(f"[reference {need[i][1]} p{need[i][0]['index']}] {r['wall_s']:.0f}s", flush=True),
        )
        ref_of = {(role, p["index"]): r for (p, role), r in zip(need, refs_all, strict=True)}
        refs = [ref_of[("test", p["index"])] for p in design["points"]] if ab["kind"] == "parametric" else None
        snaps = [(p, ref_of[("snapshot", p["index"])]) for p in tpts]
        exclude = np.zeros_like(np.asarray(ref_of[("test", pt["index"])]["solid"], bool))
        if ab["kind"] == "per_point" and datas:
            d = ab["data"]
            exclude = H.snapshot_points(
                ref_of[("test", pt["index"])],
                max(int(x["n_data"]) for x in datas),
                d["fields"],
                d["placement"],
                float(d["noise"]),
                int(d["seed"]),
            )["mask"]
        jobs, labels = [], []
        for c in confs:
            s = specs[c["name"]]
            data = s["ablation_data"]
            mode = "hybrid" if data else "data_free"
            for seed in ab["seeds"]:
                out_dir = ex.dir / "runs" / f"{c['name']}_s{seed}"
                if ab["kind"] == "parametric":
                    kw = dict(
                        spec=s,
                        case=case,
                        mode=mode,
                        seed=seed,
                        design=design,
                        refs=refs,
                        snaps=snaps[: int(data["n_snapshots"])] if data else [],
                        out_dir=out_dir,
                        tag=tag,
                    )
                    jobs.append((HY_MOD, "run_parametric", kw))
                else:
                    kw = dict(
                        spec=s,
                        case=case,
                        mode=mode,
                        n_data=int(data["n_data"]) if data else 0,
                        seed=seed,
                        pt=pt,
                        ref=ref_of[("test", pt["index"])],
                        exclude=exclude,
                        out_dir=out_dir,
                        tag=tag,
                    )
                    jobs.append((HY_MOD, "run_point", kw))
                labels.append((c, seed))

        def show(i: int, out, sec: float) -> None:
            c, seed = labels[i]
            rs = out if isinstance(out, list) else [out]
            e = np.mean([r["rel_l2_max"] for r in rs])
            print(
                f"[{c['name']:16s} s{seed}] mean worst {e:.3e} over {len(rs)} point(s), train {rs[0]['train_wall_s']:.0f}s {rs[0]['status']}",
                flush=True,
            )

        outs = run_tasks(jobs, base["execution"], show)
        records = []
        for (c, seed), out in zip(labels, outs, strict=True):
            for r in out if isinstance(out, list) else [out]:
                r.update(
                    {"config": c["name"], "design": c["design"], "components": list(c["on"]), "changed": c["changed"]}
                )
                records.append(r)
                ex.log(
                    **{
                        k: r[k]
                        for k in (
                            "config",
                            "design",
                            "seed",
                            "point",
                            "split",
                            "rel_l2_max",
                            "rel_l2_theta",
                            "q_bottom_rel_err",
                            "train_wall_s",
                            "status",
                        )
                    }
                )
        ex.metrics.update(
            {
                "records": records,
                "configurations": [
                    {
                        **c,
                        "settings": {
                            k: specs[c["name"]][k] for k in ("formulation", "network", "loss", "sampling", "training")
                        },
                        "data": specs[c["name"]]["ablation_data"],
                    }
                    for c in confs
                ],
                "design": {
                    "points": design["points"],
                    "snapshots": tpts,
                    "dims": [[x.name, x.lo, x.hi, x.scale] for x in design["dims"]],
                },
                "git": git_hash(),
                "torch": torch.__version__,
            }
        )
        ex.metrics["figures"] = figures(records, tag, ex)
        table = (
            ex.dir / "ablation.md"
            if custom or tag != base["profile"]
            else ROOT / "docs" / "tables" / "phase4_7_ablation.md"
        )
        ablation_table(records, base, table, tag)
        ex.metrics["table"] = str(table)
    print(f"wrote {table}")


def figures(records: list[dict], tag: str, ex) -> list[str]:
    """Training time vs accuracy: one marker per configuration (mean +- std over seeds of the run error and time)."""
    agg = ablation_aggregate(records)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, g in agg.items():
        ax.errorbar(
            g["time_mean"],
            g["err_mean"],
            xerr=g["time_std"],
            yerr=g["err_std"],
            fmt="o",
            color=DESIGN_COLORS.get(g["design"], "k"),
            ms=5,
            capsize=2,
        )
        ax.annotate(name, (g["time_mean"], g["err_mean"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    for dsg, col in DESIGN_COLORS.items():
        ax.plot([], [], "o", color=col, label=dsg)
    ax.set_yscale("log")
    ax.set(
        xlabel="training wall time per network [s] (mean ± std over seeds)",
        ylabel="error (mean ± std over seeds)",
        title="Step 4.7 ablation: training time vs accuracy",
    )
    ax.legend(fontsize=7)
    return [str(p) for p in save_fig(fig, f"p4_7_time_vs_accuracy_{tag}", ex.dir)]


if __name__ == "__main__":
    main()
