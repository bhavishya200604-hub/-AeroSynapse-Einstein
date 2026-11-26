"""Step 4.5a: parametric 2D training on Case C - ONE network with (x, y, log Re, log Pr, log Bi [, Ri]) inputs over the
Phase 2 Case C parameter range, evaluated on the Phase 2 interpolation and extrapolation test points against steady
FV references (Phase 1 solver), compared with one non-parametric network per test point.

    python experiments/p4_5_parametric.py --profile smoke
    python experiments/p4_5_parametric.py --config my.yaml --tag mine
    python experiments/p4_5_parametric.py --set 'param_study.variants=[{name: parametric, kind: parametric}]'
    python experiments/p4_5_parametric.py --describe
    python experiments/p4_5_parametric.py --table-only --tag smoke

Settings: `param_study` in configs/pinn/defaults.yaml (parameters 'data' = the Phase 2 Case C design, or user ranges;
reference grid / scheme; variants; seeds; user data names). Every training run and every reference solve executes in
its own process (D57). Outputs: results/p4_5_parametric_<tag>/, docs/tables/phase4_5_parametric.md (default tag only).
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from aerosynapse.thermal.experiments.p4_report import param_table
from aerosynapse.thermal.data import spec as DS
from aerosynapse.thermal.data.design import SPLITS, Dim, check_leakage, design_case
from aerosynapse.thermal.pinn import param2d as PP
from aerosynapse.thermal.pinn import problems2d as P2
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn import userdata as U
from aerosynapse.thermal.pinn.architectures import build_network
from aerosynapse.thermal.pinn.networks import resolve_device
from aerosynapse.thermal.pinn.ref2d import cavity_reference, solve_steady_decoupled
from aerosynapse.thermal.pinn.trainer import Trainer
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz.style import DIV_CMAP, SEQ_CMAP, apply_style, save_fig

MOD = "aerosynapse.thermal.experiments.p4_5_parametric"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#000000"]


def study_design(spec: dict) -> dict:
    """Swept dims, fixed parameters, the design (Phase 2 Case C exactly when parameters == 'data') and the test points."""
    ps = spec["param_study"]
    if ps["parameters"] == "data":
        dspec = DS.load_spec(spec["problems2d"]["C_cavity"].get("data_config") or None, "smoke")
        names = list(dspec["cases"])
        ccfg = dspec["cases"]["caseC"]
        dims, fixed = DS.case_dims(ccfg, "caseC")
        dz = dspec["design"]
        des = design_case(
            "caseC",
            dims,
            ccfg["sizes"],
            dspec["seed"] + names.index("caseC"),
            dz["unit_train_limit"],
            dz["method"],
            dz["min_dist_factor"],
        )
        source = "Phase 2 Case C design (configs/data: same seed, sizes, splits as data/datasets/*/caseC.h5)"
    else:
        dims, fixed = [], {}
        for k, v in ps["parameters"].items():
            if isinstance(v, dict):
                dims.append(Dim(k, float(v["lo"]), float(v["hi"]), v.get("scale", "log")))
            else:
                fixed[k] = float(v)
        dz = ps["design"]
        des = design_case(
            "param", dims, dz["sizes"], int(dz["seed"]), float(dz["u_lim"]), dz["method"], float(dz["min_dist_factor"])
        )
        source = "user-defined ranges (param_study.parameters / design)"
    points = []
    for i, s in enumerate(des.split):
        if SPLITS[s] in ps["test_splits"]:
            vals = {**fixed, **{d.name: float(des.params[i, j]) for j, d in enumerate(dims)}}
            points.append({"index": i, "split": SPLITS[s], "values": vals, "u": des.u[i].tolist()})
    leak = check_leakage(des)
    box = {d.name: [float(d.to_phys(0.0)), float(d.to_phys(des.u_lim))] for d in dims}
    return {
        "dims": dims,
        "fixed": fixed,
        "u_lim": des.u_lim,
        "points": points,
        "source": source,
        "leakage": leak,
        "train_box": box,
    }


def reference_task(values: dict, cavity_cfg: dict, rc: dict) -> dict:
    """Steady FV reference at one parameter point (own process). Ri = 0: decoupled flow (Newton) + energy (linear
    scheme rc['scheme']); otherwise the coupled Newton reference (central energy convection)."""
    cond = P2.cavity_conditions(cavity_cfg)
    solver = (
        functools.partial(solve_steady_decoupled, scheme=rc["scheme"]) if float(values.get("Ri", 0.0)) == 0.0 else None
    )
    t0 = time.perf_counter()
    r = cavity_reference(values, cond, int(rc["grid"]), int(rc["refine"]), float(rc["tol"]), solver=solver)
    r["kind"] = (
        f"decoupled: steady Newton flow + {rc['scheme']} energy" if solver else "coupled steady Newton (central energy)"
    )
    r["wall_s"] = time.perf_counter() - t0
    return r


def variant_spec(spec: dict, variant: dict) -> dict:
    s = SP.deep_merge(spec, spec["param_study"].get("common") or {})
    return SP.deep_merge(s, {k: v for k, v in variant.items() if k in ("sampling", "training", "loss", "network")})


def _dev(s: dict) -> tuple:
    return resolve_device(s["device"]), torch.float64 if s["dtype"] == "float64" else torch.float32


def point_record(variant: dict, seed: int, pt: dict, ref: dict, ev: dict) -> dict:
    v = pt["values"]
    Pe = v["Re"] * v["Pr"] if "Pr" in v else v.get("Pe", float("nan"))
    unc = ref["coarse_fine_diff"]
    return {
        "variant": variant["name"],
        "kind": variant["kind"],
        "seed": seed,
        "point": pt["index"],
        "split": pt["split"],
        **{k: float(x) for k, x in v.items()},
        "Pe": float(Pe),
        "ref_unc_max": max(unc["u"], unc["v"], unc["theta"]),
        "ref_unc_p": unc["p"],
        **ev,
    }


def train_parametric(spec: dict, variant: dict, seed: int, design: dict, refs: list, out_dir: Path, tag: str):
    """One parametric network for the whole range; evaluated at every test point. Returns (records, info, history)."""
    s = variant_spec(spec, variant)
    s["seed"] = seed
    ps = s["param_study"]
    dev, dtype = _dev(s)
    form = SP.deep_merge(s["formulation"], ps["formulation"])
    prob = PP.build_param_cavity(s["problems2d"]["C_cavity"], form, design["dims"], design["fixed"], design["u_lim"])
    torch.manual_seed(seed)
    net = build_network(s["network"], list(prob.lo), list(prob.hi), prob.out_dim).to(device=dev, dtype=dtype)
    terms, evals = U.attach(s, {"problem": "C_cavity", "data": ps.get("data") or []}, prob, dev, dtype, seed)
    tr = Trainer(net, prob, s, out_dir, tag=tag, data_terms=terms)
    prob.fit_normalisation(net, tr.batch(0))
    t0 = time.perf_counter()
    res = tr.run()
    wall = time.perf_counter() - t0
    ev_cfg = s["formulation_study"]["evaluation"]
    val = P2.validation_loss(prob, net, int(ev_cfg["n_val"]), 64, 7_777_777 + seed)
    recs = []
    for pt, ref in zip(design["points"], refs, strict=True):
        X = prob.grid_inputs(np.asarray(ref["x"]), np.asarray(ref["y"]), pt["values"], next(net.parameters()))
        pr = P2._batched(prob, net, X, 0, P2.FIELDS)
        np.savez(
            out_dir / f"pred_point{pt['index']}.npz",
            **{k: pr[k].reshape(np.asarray(ref["u"]).shape) for k in P2.FIELDS},
        )
        r = point_record(variant, seed, pt, ref, PP.evaluate_point(prob, net, ref, pt["values"], int(ev_cfg["n_wall"])))
        r.update(
            {"train_wall_s": wall, "n_params": net.n_params(), "status": res["status"], "val_loss": val["val_loss"]}
        )
        recs.append(r)
    info = {
        "variant": variant["name"],
        "seed": seed,
        "wall_s": wall,
        "status": res["status"],
        "n_params": net.n_params(),
        "val": val,
        "user_data": U.evaluate_all(prob, net, evals),
        "n_pde": s["sampling"]["n_pde"],
    }
    return recs, info, tr.history


def train_per_point(spec: dict, variant: dict, seed: int, pt: dict, ref: dict, out_dir: Path, tag: str):
    """One NON-parametric network at one test point (baseline). Returns (record, history)."""
    s = variant_spec(spec, variant)
    s["seed"] = seed
    ps = s["param_study"]
    dev, dtype = _dev(s)
    form = SP.deep_merge(s["formulation"], ps["formulation"])
    s = SP.deep_merge(s, {"problems2d": {"C_cavity": {"parameters": pt["values"]}}})
    prob = P2.build_problem2d("C_cavity", s, form, reference=ref)
    torch.manual_seed(seed)
    net = build_network(s["network"], list(prob.lo), list(prob.hi), prob.out_dim).to(device=dev, dtype=dtype)
    tr = Trainer(net, prob, s, out_dir, tag=tag)
    prob.fit_normalisation(net, tr.batch(0))
    t0 = time.perf_counter()
    res = tr.run()
    wall = time.perf_counter() - t0
    ev_cfg = s["formulation_study"]["evaluation"]
    val = P2.validation_loss(prob, net, int(ev_cfg["n_val"]), 64, 7_777_777 + seed)
    r = point_record(variant, seed, pt, ref, PP.evaluate_point(prob, net, ref, None, int(ev_cfg["n_wall"])))
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
        rdir = ROOT / "results" / f"p4_5_parametric_{tag}"
        met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
        saved = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
        out = (
            rdir / "parametric.md"
            if (custom or tag != saved["profile"])
            else ROOT / "docs" / "tables" / "phase4_5_parametric.md"
        )
        param_table(met["records"], met["runs"], met["design"], met["references"], saved, out, tag)
        print(f"wrote {out}")
        # figures from the same saved run: references from reference_point<i>.npz, predictions from the runs' npz
        from aerosynapse.thermal.utils import replay as RP

        refs = []
        for p in met["design"]["points"]:
            f = rdir / f"reference_point{p['index']}.npz"
            if f.exists():
                with np.load(f) as z:
                    refs.append({k: z[k] for k in z.files})
            else:
                refs.append(None)
        apply_style()
        for f in figures(met["records"], refs, met["design"], saved, tag, RP.target(rdir.name)):
            print(f)
        return
    spec = SP.load_spec(a.config or None, a.profile, SP.parse_set(a.sets))
    ps = spec["param_study"]
    SP.check_user_files(spec, [{"problem": "C_cavity", "data": ps.get("data") or []}])
    design = study_design(spec)
    if a.describe:
        print(SP.describe(spec))
        print(yaml.safe_dump({"param_study": ps}, sort_keys=False))
        print("design:", design["source"], "train box", design["train_box"], "fixed", design["fixed"])
        for p in design["points"]:
            print(" ", p["split"], {k: round(v, 4) for k, v in p["values"].items()})
        return
    apply_style()
    print(SP.describe(spec), flush=True)
    print(f"design: {design['source']}; train box {design['train_box']}; leakage {design['leakage']}", flush=True)
    exe = spec["execution"]
    with Experiment(f"p4_5_parametric_{tag}", spec) as ex:
        cav = spec["problems2d"]["C_cavity"]
        rtasks = [
            (MOD, "reference_task", dict(values=p["values"], cavity_cfg=cav, rc=ps["reference"]))
            for p in design["points"]
        ]

        def rdone(i: int, r: dict, sec: float) -> None:
            p = design["points"][i]
            print(
                f"[reference {p['split']:6s} {fmt_vals(p['values'])}] coarse-fine {fmt_unc(r['coarse_fine_diff'])} {r['wall_s']:.0f}s",
                flush=True,
            )

        refs = run_tasks(rtasks, exe, rdone)
        ref_info = [
            {
                "point": p["index"],
                "split": p["split"],
                "values": p["values"],
                "kind": r["kind"],
                "n_coarse": r["n_coarse"],
                "n_fine": r["n_fine"],
                "coarse_fine_diff": r["coarse_fine_diff"],
                "nu_top": r["nu_top"],
                "newton": r["newton"],
                "wall_s": r["wall_s"],
            }
            for p, r in zip(design["points"], refs, strict=True)
        ]
        for p, r in zip(design["points"], refs, strict=True):
            np.savez(ex.dir / f"reference_point{p['index']}.npz", **{k: r[k] for k in ("x", "y", *P2.FIELDS)})
        jobs, labels = [], []
        for variant in ps["variants"]:
            for seed in ps["seeds"]:
                if variant["kind"] == "parametric":
                    kw = dict(
                        spec=spec,
                        variant=variant,
                        seed=seed,
                        design=design,
                        refs=refs,
                        out_dir=ex.dir / "runs" / f"{variant['name']}_s{seed}",
                        tag=tag,
                    )
                    jobs.append((MOD, "train_parametric", kw))
                    labels.append((variant["name"], seed, None))
                else:
                    for p, r in zip(design["points"], refs, strict=True):
                        if p["split"] not in variant.get("splits", ps["test_splits"]):
                            continue
                        kw = dict(
                            spec=spec,
                            variant=variant,
                            seed=seed,
                            pt=p,
                            ref=r,
                            out_dir=ex.dir / "runs" / f"{variant['name']}_p{p['index']}_s{seed}",
                            tag=tag,
                        )
                        jobs.append((MOD, "train_per_point", kw))
                        labels.append((variant["name"], seed, p["index"]))

        def show(i: int, out: tuple, sec: float) -> None:
            rs = out[0] if isinstance(out[0], list) else [out[0]]
            for r in rs:
                print(
                    f"[{r['variant']:16s} s{r['seed']} {r['split']:6s} Re={r['Re']:.0f} Pe={r['Pe']:.0f} Bi={r.get('Bi', float('nan')):.3g}] "
                    f"worst={r['rel_l2_max']:.2e} p={r['rel_l2_p']:.2e} Nu_err={r['nu_top_rel_err']:.2e} (ref unc {r['ref_unc_max']:.1e}) "
                    f"train {r['train_wall_s']:.0f}s {r['status']}",
                    flush=True,
                )

        outs = run_tasks(jobs, exe, show)
        records, runs, hists = [], [], {}
        for (name, seed, pidx), out in zip(labels, outs, strict=True):
            if pidx is None:
                recs, info, hist = out
                records += recs
                runs.append(info)
                hists[(name, seed)] = hist
            else:
                rec, hist = out
                records.append(rec)
                runs.append(
                    {
                        "variant": name,
                        "seed": seed,
                        "point": pidx,
                        "wall_s": rec["train_wall_s"],
                        "status": rec["status"],
                        "n_params": rec["n_params"],
                    }
                )
        for r in records:
            ex.log(
                **{
                    k: r[k]
                    for k in (
                        "variant",
                        "seed",
                        "point",
                        "split",
                        "Re",
                        "Pe",
                        "rel_l2_max",
                        "rel_l2_p",
                        "nu_top_rel_err",
                        "status",
                    )
                }
            )
        dsave = {
            "source": design["source"],
            "fixed": design["fixed"],
            "u_lim": design["u_lim"],
            "train_box": design["train_box"],
            "leakage": design["leakage"],
            "points": design["points"],
            "dims": [[d.name, d.lo, d.hi, d.scale] for d in design["dims"]],
        }
        ex.metrics.update(
            {
                "records": records,
                "runs": runs,
                "design": dsave,
                "references": ref_info,
                "git": git_hash(),
                "torch": torch.__version__,
            }
        )
        ex.metrics["figures"] = figures(records, refs, design, spec, tag, ex)
        table = ex.dir / "parametric.md" if custom else ROOT / "docs" / "tables" / "phase4_5_parametric.md"
        param_table(records, runs, dsave, ref_info, spec, table, tag)
        ex.metrics["table"] = str(table)
    print(f"wrote {table}")


def fmt_vals(v: dict) -> str:
    return " ".join(f"{k}={x:.3g}" for k, x in v.items())


def fmt_unc(d: dict) -> str:
    return " ".join(f"{k} {d[k]:.1e}" for k in ("u", "v", "theta", "p"))


def figures(records: list[dict], refs: list, design: dict, spec: dict, tag: str, ex: Experiment) -> list[str]:
    paths = []
    variants = list(dict.fromkeys(r["variant"] for r in records))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, v in enumerate(variants):
        for split, mk in (("interp", "o"), ("extrap", "s")):
            rs = [r for r in records if r["variant"] == v and r["split"] == split]
            if rs:
                axs[0].loglog(
                    [r["Pe"] for r in rs],
                    [r["rel_l2_max"] for r in rs],
                    mk,
                    color=COLORS[i % len(COLORS)],
                    label=f"{v} ({split})",
                )
                axs[1].plot(
                    [r["nu_top_ref"] for r in rs],
                    [r["nu_top"] for r in rs],
                    mk,
                    color=COLORS[i % len(COLORS)],
                    label=f"{v} ({split})",
                )
    rs = [r for r in records if r["variant"] == variants[0]]
    axs[0].loglog([r["Pe"] for r in rs], [r["ref_unc_max"] for r in rs], "k_", ms=12, label="reference coarse-fine")
    axs[0].axhline(spec["evaluation"]["target_rel_l2"], color="k", ls=":", lw=0.8)
    axs[0].set(xlabel="Pe = Re Pr", ylabel="worst-field rel $L_2$ error", title="error over the test points")
    lim = [min(r["nu_top_ref"] for r in records) * 0.8, max(r["nu_top_ref"] for r in records) * 1.2]
    axs[1].plot(lim, lim, "k:", lw=0.8)
    axs[1].set(xlabel="Nu_top reference", ylabel="Nu_top PINN", title="top-wall Nusselt number")
    for ax in axs:
        ax.legend(fontsize=6)
    paths += save_fig(fig, f"p4_5_error_vs_Pe_{tag}", ex.dir)
    par = [r for r in records if r["kind"] == "parametric"]
    interp = [r for r in par if r["split"] == "interp"]
    if interp:
        worst = max(interp, key=lambda r: r["rel_l2_max"])
        ref = refs[[p["index"] for p in design["points"]].index(worst["point"])]
        pred = np.load(ex.dir / "runs" / f"{worst['variant']}_s{worst['seed']}" / f"pred_point{worst['point']}.npz")
        fig, axs = plt.subplots(2, 3, figsize=(12, 7))
        for row, (k, cm) in enumerate((("theta", SEQ_CMAP), ("u", SEQ_CMAP))):
            for col, (arr, ttl) in enumerate(
                ((ref[k], "reference"), (pred[k], f"PINN {worst['variant']}"), (pred[k] - ref[k], "error"))
            ):
                c = DIV_CMAP if col == 2 else cm
                lim = float(np.abs(arr).max()) if col == 2 else None
                im = axs[row, col].imshow(
                    arr, origin="lower", extent=(0, 1, 0, 1), cmap=c, vmin=-lim if lim else None, vmax=lim
                )
                axs[row, col].set(title=f"{k}: {ttl}", xticks=[0, 1], yticks=[0, 1])
                fig.colorbar(im, ax=axs[row, col], fraction=0.046)
        fig.suptitle(
            f"worst interpolation point: Re {worst['Re']:.0f}, Pe {worst['Pe']:.0f}, Bi {worst.get('Bi', float('nan')):.3g} (worst-field rel L2 {worst['rel_l2_max']:.2e})"
        )
        paths += save_fig(fig, f"p4_5_fields_worst_interp_{tag}", ex.dir)
    return [str(p) for p in paths]


if __name__ == "__main__":
    main()
