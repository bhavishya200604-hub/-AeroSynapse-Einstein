"""Step 4.5b: parametric 2D training on Case D - the thermally developing channel (Graetz, Robin walls) and the
channel with a heated square cylinder - with (x, y, log Re, log Pr, log Bi) inputs; one parametric network per case vs
one network per test point; test points from a parameter design (interp inside, extrap outside the training box);
references = Phase 1 steady solver on two grids.

    python experiments/p4_5b_channel.py --profile smoke
    python experiments/p4_5b_channel.py --set 'channel_study.cases=[D_graetz]' --tag graetz
    python experiments/p4_5b_channel.py --describe
    python experiments/p4_5b_channel.py --table-only --tag smoke

Settings: problems2d.D_graetz / D_obstacle (geometry, parameters, inlet, thermal walls, reference) and channel_study
(design, variants, common overrides, seeds, user data per case). Every run in its own process (D57).
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

from aerosynapse.thermal.experiments.p4_report import channel_table
from aerosynapse.thermal.data.design import SPLITS, Dim, check_leakage, design_case
from aerosynapse.thermal.pinn import channel2d as CH
from aerosynapse.thermal.pinn import problems2d as P2
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn import userdata as U
from aerosynapse.thermal.pinn.architectures import build_network
from aerosynapse.thermal.pinn.networks import resolve_device
from aerosynapse.thermal.pinn.trainer import Trainer
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz.style import DIV_CMAP, SEQ_CMAP, apply_style, save_fig

MOD = "aerosynapse.thermal.experiments.p4_5b_channel"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#000000"]


def case_design(spec: dict, case: str) -> dict:
    cs = spec["channel_study"]
    dims, fixed = [], {}
    for k, v in spec["problems2d"][case]["parameters"].items():
        if isinstance(v, dict):
            dims.append(Dim(k, float(v["lo"]), float(v["hi"]), v.get("scale", "log")))
        else:
            fixed[k] = float(v)
    d = cs["design"]
    seed = int(d["seed"]) + list(SP.CHANNEL).index(case)
    des = design_case(case, dims, d["sizes"], seed, float(d["u_lim"]), d["method"], float(d["min_dist_factor"]))
    points = []
    for i, s in enumerate(des.split):
        if SPLITS[s] in ("interp", "extrap"):
            vals = {**fixed, **{dd.name: float(des.params[i, j]) for j, dd in enumerate(dims)}}
            points.append({"case": case, "index": i, "split": SPLITS[s], "values": vals})
    return {
        "case": case,
        "dims": dims,
        "fixed": fixed,
        "u_lim": des.u_lim,
        "points": points,
        "leakage": check_leakage(des),
        "train_box": {dd.name: [float(dd.to_phys(0.0)), float(dd.to_phys(des.u_lim))] for dd in dims},
    }


def reference_task(values: dict, pcfg: dict) -> dict:
    r = pcfg["reference"]
    t0 = time.perf_counter()
    out = CH.channel_reference(values, pcfg, int(r["cells_per_unit"]), int(r["refine"]), float(r["tol"]), r["scheme"])
    out["wall_s"] = time.perf_counter() - t0
    return out


def variant_spec(spec: dict, variant: dict) -> dict:
    s = SP.deep_merge(spec, spec["channel_study"].get("common") or {})
    return SP.deep_merge(s, {k: v for k, v in variant.items() if k in ("sampling", "training", "loss", "network")})


def _dev(s: dict) -> tuple:
    return resolve_device(s["device"]), torch.float64 if s["dtype"] == "float64" else torch.float32


def record(case: str, variant: dict, seed: int, pt: dict, ref: dict, ev: dict) -> dict:
    v = pt["values"]
    unc = ref["coarse_fine_diff"]
    return {
        "case": case,
        "variant": variant["name"],
        "kind": variant["kind"],
        "seed": seed,
        "point": pt["index"],
        "split": pt["split"],
        **{k: float(x) for k, x in v.items()},
        "Pe": float(v["Re"] * v["Pr"]),
        "ref_unc_max": max(unc["u"], unc["v"], unc["theta"]),
        "ref_unc_p": unc["p"],
        **ev,
    }


def train_parametric(
    spec: dict, case: str, variant: dict, seed: int, design: dict, refs: list, out_dir: Path, tag: str
):
    s = variant_spec(spec, variant)
    s["seed"] = seed
    dev, dtype = _dev(s)
    form = SP.deep_merge(s["formulation"], s["channel_study"]["formulation"])
    prob = CH.build_channel(case, s["problems2d"][case], form, design["dims"], design["fixed"], design["u_lim"])
    torch.manual_seed(seed)
    net = build_network(s["network"], list(prob.lo), list(prob.hi), prob.out_dim).to(device=dev, dtype=dtype)
    terms, evals = U.attach(
        s, {"problem": case, "data": (s["channel_study"].get("data") or {}).get(case, [])}, prob, dev, dtype, seed
    )
    tr = Trainer(net, prob, s, out_dir, tag=tag, data_terms=terms)
    prob.fit_normalisation(net, tr.batch(0))
    t0 = time.perf_counter()
    res = tr.run()
    wall = time.perf_counter() - t0
    val = P2.validation_loss(prob, net, int(s["formulation_study"]["evaluation"]["n_val"]), 64, 7_777_777 + seed)
    recs = []
    for pt, ref in zip(design["points"], refs, strict=True):
        r = record(case, variant, seed, pt, ref, CH.evaluate_channel(prob, net, ref, pt["values"]))
        r.update(
            {"train_wall_s": wall, "n_params": net.n_params(), "status": res["status"], "val_loss": val["val_loss"]}
        )
        recs.append(r)
        if pt is design["points"][0]:
            fluid = ~np.asarray(ref["solid"])
            X, Y = np.meshgrid(ref["x"], ref["y"])
            pr = P2._batched(
                prob,
                net,
                prob.inputs(np.stack([X.ravel(), Y.ravel()], 1), pt["values"], next(net.parameters())),
                0,
                P2.FIELDS,
            )
            np.savez(
                out_dir / f"pred_point{pt['index']}.npz", fluid=fluid, **{k: pr[k].reshape(X.shape) for k in P2.FIELDS}
            )
    info = {
        "case": case,
        "variant": variant["name"],
        "seed": seed,
        "wall_s": wall,
        "status": res["status"],
        "n_params": net.n_params(),
        "val": val,
        "user_data": U.evaluate_all(prob, net, evals),
    }
    return recs, info, tr.history


def train_per_point(spec: dict, case: str, variant: dict, seed: int, pt: dict, ref: dict, out_dir: Path, tag: str):
    s = variant_spec(spec, variant)
    s["seed"] = seed
    dev, dtype = _dev(s)
    form = SP.deep_merge(s["formulation"], s["channel_study"]["formulation"])
    prob = CH.build_channel(case, s["problems2d"][case], form, [], pt["values"], 1.0)
    torch.manual_seed(seed)
    net = build_network(s["network"], list(prob.lo), list(prob.hi), prob.out_dim).to(device=dev, dtype=dtype)
    tr = Trainer(net, prob, s, out_dir, tag=tag)
    prob.fit_normalisation(net, tr.batch(0))
    t0 = time.perf_counter()
    res = tr.run()
    wall = time.perf_counter() - t0
    val = P2.validation_loss(prob, net, int(s["formulation_study"]["evaluation"]["n_val"]), 64, 7_777_777 + seed)
    r = record(case, variant, seed, pt, ref, CH.evaluate_channel(prob, net, ref, None))
    r.update({"train_wall_s": wall, "n_params": net.n_params(), "status": res["status"], "val_loss": val["val_loss"]})
    return r, tr.history


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
        rdir = ROOT / "results" / f"p4_5b_channel_{tag}"
        met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
        saved = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
        out = (
            rdir / "channel.md"
            if (custom or tag != saved["profile"])
            else ROOT / "docs" / "tables" / "phase4_5b_channel.md"
        )
        channel_table(met["records"], met["runs"], met["designs"], met["references"], saved, out, tag)
        print(f"wrote {out}")
        # figures from the same saved run; the field figure's reference (first design point of each case) was not
        # stored, so it is re-solved with the verified steady solver (numerics only, deterministic; no training)
        from aerosynapse.thermal.utils import replay as RP

        refs = {
            c: [reference_task(d["points"][0]["values"], saved["problems2d"][c])]
            for c, d in met["designs"].items()
            if d.get("points")
        }
        apply_style()
        for f in figures(met["records"], refs, met["designs"], saved["channel_study"], tag, RP.target(rdir.name)):
            print(f)
        return
    spec = SP.load_spec(a.config or None, a.profile, SP.parse_set(a.sets))
    cs = spec["channel_study"]
    SP.check_user_files(spec, [{"problem": c, "data": srcs} for c, srcs in (cs.get("data") or {}).items()])
    designs = {c: case_design(spec, c) for c in cs["cases"]}
    if a.describe:
        print(yaml.safe_dump({"channel_study": cs, **{c: spec["problems2d"][c] for c in cs["cases"]}}, sort_keys=False))
        for c, d in designs.items():
            print(c, "train box", d["train_box"], "fixed", d["fixed"])
            for p in d["points"]:
                print("  ", p["split"], {k: round(v, 4) for k, v in p["values"].items()})
        return
    apply_style()
    exe = spec["execution"]
    with Experiment(f"p4_5b_channel_{tag}", spec) as ex:
        pts = [p for c in cs["cases"] for p in designs[c]["points"]]
        rtasks = [(MOD, "reference_task", dict(values=p["values"], pcfg=spec["problems2d"][p["case"]])) for p in pts]

        def rdone(i: int, r: dict, sec: float) -> None:
            p, d = pts[i], r["coarse_fine_diff"]
            extra = f" C_D={r['C_D']:.4f}" if "C_D" in r else ""
            print(
                f"[reference {p['case']} {p['split']:6s} "
                + " ".join(f"{k}={v:.3g}" for k, v in p["values"].items())
                + f"] coarse-fine u {d['u']:.1e} v {d['v']:.1e} theta {d['theta']:.1e} p {d['p']:.1e} q {d['q_bottom']:.1e}{extra} {r['wall_s']:.0f}s",
                flush=True,
            )

        refs_all = run_tasks(rtasks, exe, rdone)
        refs = {c: [r for p, r in zip(pts, refs_all, strict=True) if p["case"] == c] for c in cs["cases"]}
        ref_info = [
            {
                "case": p["case"],
                "point": p["index"],
                "split": p["split"],
                "values": p["values"],
                "kind": r["kind"],
                "n_coarse": r["n_coarse"],
                "n_fine": r["n_fine"],
                "coarse_fine_diff": r["coarse_fine_diff"],
                "q_bottom": r["q_bottom"],
                "C_D": r.get("C_D"),
                "newton": r["newton"],
                "wall_s": r["wall_s"],
            }
            for p, r in zip(pts, refs_all, strict=True)
        ]
        jobs, labels = [], []
        for c in cs["cases"]:
            for variant in cs["variants"]:
                for seed in cs["seeds"]:
                    if variant["kind"] == "parametric":
                        kw = dict(
                            spec=spec,
                            case=c,
                            variant=variant,
                            seed=seed,
                            design=designs[c],
                            refs=refs[c],
                            out_dir=ex.dir / "runs" / f"{c}_{variant['name']}_s{seed}",
                            tag=tag,
                        )
                        jobs.append((MOD, "train_parametric", kw))
                        labels.append((c, variant["name"], seed, None))
                    else:
                        for p, r in zip(designs[c]["points"], refs[c], strict=True):
                            if p["split"] not in variant.get("splits", ["interp", "extrap"]):
                                continue
                            kw = dict(
                                spec=spec,
                                case=c,
                                variant=variant,
                                seed=seed,
                                pt=p,
                                ref=r,
                                out_dir=ex.dir / "runs" / f"{c}_{variant['name']}_p{p['index']}_s{seed}",
                                tag=tag,
                            )
                            jobs.append((MOD, "train_per_point", kw))
                            labels.append((c, variant["name"], seed, p["index"]))

        def show(i: int, out: tuple, sec: float) -> None:
            rs = out[0] if isinstance(out[0], list) else [out[0]]
            for r in rs:
                extra = f" C_D_err={r['C_D_rel_err']:.2e}" if "C_D_rel_err" in r else ""
                print(
                    f"[{r['case']} {r['variant']:12s} s{r['seed']} {r['split']:6s} Re={r['Re']:.0f} Pe={r['Pe']:.0f} Bi={r.get('Bi', float('nan')):.3g}] "
                    f"worst={r['rel_l2_max']:.2e} p={r['rel_l2_p']:.2e} q_err={r['q_bottom_rel_err']:.2e}{extra} (ref unc {r['ref_unc_max']:.1e}) "
                    f"train {r['train_wall_s']:.0f}s {r['status']}",
                    flush=True,
                )

        outs = run_tasks(jobs, exe, show)
        records, runs = [], []
        for (c, name, seed, pidx), out in zip(labels, outs, strict=True):
            if pidx is None:
                recs, info, _ = out
                records += recs
                runs.append(info)
            else:
                rec, _ = out
                records.append(rec)
                runs.append(
                    {
                        "case": c,
                        "variant": name,
                        "seed": seed,
                        "point": pidx,
                        "wall_s": rec["train_wall_s"],
                        "status": rec["status"],
                    }
                )
        for r in records:
            ex.log(
                **{
                    k: r[k]
                    for k in (
                        "case",
                        "variant",
                        "seed",
                        "point",
                        "split",
                        "Re",
                        "Pe",
                        "rel_l2_max",
                        "rel_l2_p",
                        "q_bottom_rel_err",
                        "status",
                    )
                }
            )
        dsave = {
            c: {
                "fixed": d["fixed"],
                "u_lim": d["u_lim"],
                "train_box": d["train_box"],
                "leakage": d["leakage"],
                "points": d["points"],
                "dims": [[x.name, x.lo, x.hi, x.scale] for x in d["dims"]],
            }
            for c, d in designs.items()
        }
        ex.metrics.update(
            {
                "records": records,
                "runs": runs,
                "designs": dsave,
                "references": ref_info,
                "git": git_hash(),
                "torch": torch.__version__,
            }
        )
        ex.metrics["figures"] = figures(records, refs, designs, cs, tag, ex)
        table = ex.dir / "channel.md" if custom else ROOT / "docs" / "tables" / "phase4_5b_channel.md"
        channel_table(records, runs, dsave, ref_info, spec, table, tag)
        ex.metrics["table"] = str(table)
    print(f"wrote {table}")


def figures(records: list[dict], refs: dict, designs: dict, cs: dict, tag: str, ex: Experiment) -> list[str]:
    paths = []
    cases = list(dict.fromkeys(r["case"] for r in records))
    fig, axs = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 3.8), squeeze=False)
    for ax, c in zip(axs[0], cases, strict=True):
        for i, v in enumerate(dict.fromkeys(r["variant"] for r in records)):
            for split, mk in (("interp", "o"), ("extrap", "s")):
                rs = [r for r in records if r["case"] == c and r["variant"] == v and r["split"] == split]
                if rs:
                    ax.loglog(
                        [r["Pe"] for r in rs],
                        [r["rel_l2_max"] for r in rs],
                        mk,
                        color=COLORS[i % len(COLORS)],
                        label=f"{v} ({split})",
                    )
        rs = [r for r in records if r["case"] == c]
        ax.loglog([r["Pe"] for r in rs], [r["ref_unc_max"] for r in rs], "k_", ms=12, label="reference coarse-fine")
        ax.set(xlabel="Pe", ylabel="worst-field rel $L_2$", title=c)
        ax.legend(fontsize=6)
    paths += save_fig(fig, f"p4_5b_error_vs_Pe_{tag}", ex.dir)
    for c in cases:
        par = [v["name"] for v in cs["variants"] if v["kind"] == "parametric"]
        if not par:
            continue
        p0 = designs[c]["points"][0]
        f = ex.dir / "runs" / f"{c}_{par[0]}_s{cs['seeds'][0]}" / f"pred_point{p0['index']}.npz"
        if not f.exists():
            continue
        pred, ref = np.load(f), refs[c][0]
        fluid = pred["fluid"]
        L, H = float(ref["x"][-1] + ref["x"][0]), float(ref["y"][-1] + ref["y"][0])
        fig, axs = plt.subplots(3, 2, figsize=(11, 7))
        for col, k in enumerate(("u", "theta")):
            rk = np.where(fluid, np.asarray(ref[k]), np.nan)
            pk = np.where(fluid, pred[k], np.nan)
            for row, (arr, ttl, cm) in enumerate(
                ((rk, "reference", SEQ_CMAP), (pk, f"PINN {par[0]}", SEQ_CMAP), (pk - rk, "error", DIV_CMAP))
            ):
                lim = float(np.nanmax(np.abs(arr))) if row == 2 else None
                im = axs[row, col].imshow(
                    arr,
                    origin="lower",
                    extent=(0, L, 0, H),
                    cmap=cm,
                    vmin=-lim if lim else None,
                    vmax=lim,
                    aspect="auto",
                )
                axs[row, col].set(title=f"{k}: {ttl}")
                fig.colorbar(im, ax=axs[row, col], fraction=0.046)
        fig.suptitle(
            f"{c}: first test point ({p0['split']}) " + " ".join(f"{k}={v:.3g}" for k, v in p0["values"].items())
        )
        paths += save_fig(fig, f"p4_5b_fields_{c}_{tag}", ex.dir)
    return [str(p) for p in paths]


if __name__ == "__main__":
    main()
