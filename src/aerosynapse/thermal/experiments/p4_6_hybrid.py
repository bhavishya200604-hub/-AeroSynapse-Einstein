"""Step 4.6: hybrid data + physics vs data-free vs data-only on the Case D channels; how little data suffices.

CFD snapshot data = the verified solver (Phase 1) on its fine reference grid. Two parts (hybrid_study in
configs/pinn/defaults.yaml, every entry user-definable):

  per_point   one network at ONE test point (default: the first interp point of the 4.5b design) for every mode and
              every data budget n in per_point.n_data. Data cells are drawn nested in n (aerosynapse.thermal.pinn.hybrid), and ALL
              per-point runs of a case are scored on the same held-out cells: the fluid cells not in the largest
              data set (data_free included), so errors are comparable across modes and n.
  parametric  one network over the parameter box per mode; data = snapshots at the first n_snapshots TRAIN design
              points (n_data cells each); scored at the 4.5b interp / extrap test points (other parameter values).

Modes: data_free (physics only), hybrid (physics + data), data_only (physics weights 0). Your own data files can be
added per case through hybrid_study.user_data (user_data sources, D60).

    python experiments/p4_6_hybrid.py --profile smoke [--config my.yaml] [--set key=value] [--tag name]
    python experiments/p4_6_hybrid.py --describe          # resolved settings, test points and snapshot points
    python experiments/p4_6_hybrid.py --table-only --tag smoke
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

from aerosynapse.thermal.experiments.p4_5b_channel import case_design
from aerosynapse.thermal.experiments.p4_report import hybrid_table
from aerosynapse.thermal.data.design import SPLITS, Dim, design_case
from aerosynapse.thermal.pinn import channel2d as CH
from aerosynapse.thermal.pinn import hybrid as H
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn import userdata as U
from aerosynapse.thermal.pinn.architectures import build_network
from aerosynapse.thermal.pinn.networks import resolve_device
from aerosynapse.thermal.pinn.trainer import Trainer
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz.style import apply_style, save_fig

MOD = "aerosynapse.thermal.experiments.p4_6_hybrid"
REF_MOD = "aerosynapse.thermal.experiments.p4_5b_channel"
COLORS = {"data_free": "#000000", "data_only": "#D55E00", "hybrid": "#0072B2"}


def study_spec(spec: dict, seed: int) -> dict:
    hs = spec["hybrid_study"]
    s = SP.deep_merge(spec, hs.get("common") or {})
    s["problems2d"] = SP.deep_merge(s["problems2d"], hs.get("problems") or {})
    s["seed"] = seed
    return s


def train_design_points(spec: dict, case: str) -> list[dict]:
    """The TRAIN points of the 4.5b design of ``case`` (same call as case_design, so the same design)."""
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
    return [
        {
            "case": case,
            "index": i,
            "split": "train",
            "values": {**fixed, **{dd.name: float(des.params[i, j]) for j, dd in enumerate(dims)}},
        }
        for i, s in enumerate(des.split)
        if SPLITS[s] == "train"
    ]


def _net(s: dict, prob) -> torch.nn.Module:
    dev = resolve_device(s["device"])
    dtype = torch.float64 if s["dtype"] == "float64" else torch.float32
    torch.manual_seed(s["seed"])
    return build_network(s["network"], list(prob.lo), list(prob.hi), prob.out_dim).to(device=dev, dtype=dtype)


def _user_terms(s: dict, case: str, prob, net, mode: str) -> tuple[list, list]:
    srcs = (s["hybrid_study"].get("user_data") or {}).get(case, [])
    p0 = next(net.parameters())
    terms, evals = U.attach(s, {"problem": case, "data": srcs}, prob, p0.device, p0.dtype, s["seed"])
    return ([] if mode == "data_free" else terms), evals


def _fit(term, net) -> float:
    return H.fit_error(term, net)


def refit(tag: str) -> None:
    """Recompute data_fit of a finished run from its saved networks (runs/*/checkpoint.pt) and rebuild table and
    figures; used once for the smoke run whose fit metric still divided v by its own (near-zero) norm."""
    rdir = ROOT / "results" / f"p4_6_hybrid_{tag}"
    met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
    spec = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
    hs = spec["hybrid_study"]
    d = hs["data"]
    cache: dict = {}

    def ref(case: str, values: dict) -> dict:
        key = (case, tuple(sorted(values.items())))
        if key not in cache:
            cache[key] = CH.channel_reference(
                values,
                study_spec(spec, 0)["problems2d"][case],
                int(spec["problems2d"][case]["reference"]["cells_per_unit"]),
                int(spec["problems2d"][case]["reference"]["refine"]),
                float(spec["problems2d"][case]["reference"]["tol"]),
                spec["problems2d"][case]["reference"]["scheme"],
            )
        return cache[key]

    done: dict = {}
    for r in met["records"]:
        if r["mode"] == "data_free":
            continue
        key = (r["case"], r["kind"], r["mode"], r["n_data"], r["seed"])
        if key not in done:
            s = study_spec(spec, r["seed"])
            c = r["case"]
            form = SP.deep_merge(s["formulation"], hs["formulation"])
            des = met["designs"][c]
            if r["kind"] == "per_point":
                pt = next(p for p in des["points"] if p["index"] == r["point"])
                prob = CH.build_channel(c, s["problems2d"][c], form, [], pt["values"], 1.0)
                run = f"{c}_point_{r['mode']}_n{r['n_data']}_s{r['seed']}"
                snaps = [
                    (
                        H.snapshot_points(
                            ref(c, pt["values"]),
                            r["n_data"],
                            d["fields"],
                            d["placement"],
                            float(d["noise"]),
                            int(d["seed"]),
                        ),
                        None,
                    )
                ]
            else:
                dims = [Dim(n, lo, hi, sc) for n, lo, hi, sc in des["dims"]]
                prob = CH.build_channel(c, s["problems2d"][c], form, dims, des["fixed"], des["u_lim"])
                run = f"{c}_parametric_{r['mode']}_s{r['seed']}"
                snaps = [
                    (
                        H.snapshot_points(
                            ref(c, p["values"]),
                            r["n_data"],
                            d["fields"],
                            d["placement"],
                            float(d["noise"]),
                            int(d["seed"]) + k,
                        ),
                        p["values"],
                    )
                    for k, p in enumerate(des["snapshots"])
                ]
            net = _net(s, prob)
            st = torch.load(rdir / "runs" / run / "checkpoint.pt", map_location="cpu", weights_only=False)
            net.load_state_dict(st["model"])
            done[key] = H.fit_error(H.snapshot_term("snap", prob, snaps, next(net.parameters())), net)
            print(f"refit {run}: data_fit {r['data_fit']:.3g} -> {done[key]:.3g}", flush=True)
        r["data_fit_v_own_norm"] = r["data_fit"]
        r["data_fit"] = done[key]
    met["refit"] = {
        "git": git_hash(),
        "note": "data_fit recomputed with u, v relative to the velocity-vector norm (fit_error)",
    }
    (rdir / "metrics.json").write_text(json.dumps(met, indent=1), encoding="utf-8")
    custom = tag != spec["profile"]
    out = rdir / "hybrid.md" if custom else ROOT / "docs" / "tables" / "phase4_6_hybrid.md"
    hybrid_table(met["records"], met["references"], spec, out, tag)
    apply_style()

    class _Ex:  # figures() needs only .dir
        dir = rdir

    figures(met["records"], tag, _Ex)
    print(f"wrote {rdir / 'metrics.json'} and {out}")


def _train(s: dict, prob, net, mode: str, terms: list, out_dir: Path, tag: str) -> tuple[dict, float]:
    tr = Trainer(net, prob, s, out_dir, weights=H.mode_weights(s["loss"]["weights"], mode), tag=tag, data_terms=terms)
    prob.fit_normalisation(net, tr.batch(0))
    t0 = time.perf_counter()
    res = tr.run()
    return res, time.perf_counter() - t0


def run_point(spec, case, mode, n_data, seed, pt, ref, exclude, out_dir, tag) -> dict:
    s = study_spec(spec, seed)
    hs = s["hybrid_study"]
    form = SP.deep_merge(s["formulation"], hs["formulation"])
    prob = CH.build_channel(case, s["problems2d"][case], form, [], pt["values"], 1.0)
    net = _net(s, prob)
    terms, evals = _user_terms(s, case, prob, net, mode)
    snap_term = None
    if mode != "data_free":
        d = hs["data"]
        snap = H.snapshot_points(ref, n_data, d["fields"], d["placement"], float(d["noise"]), int(d["seed"]))
        snap_term = H.snapshot_term("snap", prob, [(snap, None)], next(net.parameters()))
        terms = [snap_term, *terms]
    res, wall = _train(s, prob, net, mode, terms, out_dir, tag)
    ev = CH.evaluate_channel(prob, net, ref, None, exclude=exclude)
    v = pt["values"]
    return {
        "case": case,
        "kind": "per_point",
        "mode": mode,
        "n_data": 0 if mode == "data_free" else int(n_data),
        "n_snapshots": 0 if mode == "data_free" else 1,
        "seed": seed,
        "point": pt["index"],
        "split": pt["split"],
        **{k: float(x) for k, x in v.items()},
        "Pe": float(v["Re"] * v["Pr"]),
        "n_eval_cells": int((~np.asarray(ref["solid"]) & ~exclude).sum()),
        "data_fit": _fit(snap_term, net) if snap_term is not None else None,
        "ref_unc_max": max(ref["coarse_fine_diff"][k] for k in ("u", "v", "theta")),
        "train_wall_s": wall,
        "status": res["status"],
        "n_params": net.n_params(),
        "user_data": U.evaluate_all(prob, net, evals),
        **ev,
    }


def run_parametric(spec, case, mode, seed, design, refs, snaps, out_dir, tag) -> list[dict]:
    s = study_spec(spec, seed)
    hs = s["hybrid_study"]
    form = SP.deep_merge(s["formulation"], hs["formulation"])
    prob = CH.build_channel(case, s["problems2d"][case], form, design["dims"], design["fixed"], design["u_lim"])
    net = _net(s, prob)
    terms, evals = _user_terms(s, case, prob, net, mode)
    snap_term = None
    if mode != "data_free":
        d, pa = hs["data"], hs["parametric"]
        sp = [
            (
                H.snapshot_points(
                    r, int(pa["n_data"]), d["fields"], d["placement"], float(d["noise"]), int(d["seed"]) + k
                ),
                p["values"],
            )
            for k, (p, r) in enumerate(snaps)
        ]
        snap_term = H.snapshot_term("snap", prob, sp, next(net.parameters()))
        terms = [snap_term, *terms]
    res, wall = _train(s, prob, net, mode, terms, out_dir, tag)
    fit = _fit(snap_term, net) if snap_term is not None else None
    out = []
    for pt, ref in zip(design["points"], refs, strict=True):
        v = pt["values"]
        out.append(
            {
                "case": case,
                "kind": "parametric",
                "mode": mode,
                "n_data": 0 if mode == "data_free" else int(hs["parametric"]["n_data"]),
                "n_snapshots": 0 if mode == "data_free" else len(snaps),
                "seed": seed,
                "point": pt["index"],
                "split": pt["split"],
                **{k: float(x) for k, x in v.items()},
                "Pe": float(v["Re"] * v["Pr"]),
                "data_fit": fit,
                "ref_unc_max": max(ref["coarse_fine_diff"][k] for k in ("u", "v", "theta")),
                "train_wall_s": wall,
                "status": res["status"],
                "n_params": net.n_params(),
                "user_data": U.evaluate_all(prob, net, evals),
                **CH.evaluate_channel(prob, net, ref, v),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--table-only", action="store_true", help="rebuild table and figures from saved metrics of --tag")
    ap.add_argument("--refit", action="store_true", help="recompute data_fit of --tag from its saved networks")
    a = ap.parse_args()
    if a.refit:
        refit(a.tag or a.profile)
        return
    tag = a.tag or a.profile
    custom = bool(a.config or a.sets)
    if a.table_only:
        rdir = ROOT / "results" / f"p4_6_hybrid_{tag}"
        met = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
        saved = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
        out = (
            rdir / "hybrid.md"
            if (custom or tag != saved["profile"])
            else ROOT / "docs" / "tables" / "phase4_6_hybrid.md"
        )
        hybrid_table(met["records"], met["references"], saved, out, tag)
        print(f"wrote {out}")
        from aerosynapse.thermal.utils import replay as RP  # the figures, as the help promises, from the same saved records

        apply_style()
        for f in figures(met["records"], tag, RP.target(rdir.name)):
            print(f)
        return
    spec = SP.load_spec(a.config or None, a.profile, SP.parse_set(a.sets))
    hs = spec["hybrid_study"]
    s0 = study_spec(spec, 0)
    SP.check_user_files(s0, [{"problem": c, "data": v} for c, v in (hs.get("user_data") or {}).items()])
    designs = {c: case_design(s0, c) for c in hs["cases"]}
    pp, pa = hs["per_point"], hs["parametric"]
    for c, d in designs.items():
        if pp["enabled"] and pp["point"] >= len(d["points"]):
            raise ValueError(f"hybrid_study.per_point.point = {pp['point']}: {c} has {len(d['points'])} test points")
    tpts = {c: train_design_points(s0, c)[: int(pa["n_snapshots"])] for c in hs["cases"]} if pa["enabled"] else {}
    if a.describe:
        print(yaml.safe_dump({"hybrid_study": hs, **{c: s0["problems2d"][c] for c in hs["cases"]}}, sort_keys=False))
        for c in hs["cases"]:
            for p in designs[c]["points"]:
                print(c, p["split"], {k: round(v, 4) for k, v in p["values"].items()})
            for p in tpts.get(c, []):
                print(c, "snapshot (train)", {k: round(v, 4) for k, v in p["values"].items()})
        return
    apply_style()
    exe = spec["execution"]
    with Experiment(f"p4_6_hybrid_{tag}", spec) as ex:
        need = []  # (case, point dict, role)
        for c in hs["cases"]:
            if pa["enabled"]:
                need += [(c, p, "test") for p in designs[c]["points"]]
            elif pp["enabled"]:
                need.append((c, designs[c]["points"][pp["point"]], "test"))
            if pa["enabled"] and any(m != "data_free" for m in hs["modes"]):
                need += [(c, p, "snapshot") for p in tpts[c]]
        rtasks = [(REF_MOD, "reference_task", dict(values=p["values"], pcfg=s0["problems2d"][c])) for c, p, _ in need]

        def rdone(i: int, r: dict, sec: float) -> None:
            c, p, role = need[i]
            d = r["coarse_fine_diff"]
            print(
                f"[reference {c} {role} {p['split']} "
                + " ".join(f"{k}={v:.3g}" for k, v in p["values"].items())
                + f"] coarse-fine u {d['u']:.1e} v {d['v']:.1e} theta {d['theta']:.1e} {r['wall_s']:.0f}s",
                flush=True,
            )

        refs_all = run_tasks(rtasks, exe, rdone)
        ref_of = {(c, role, p["index"]): r for (c, p, role), r in zip(need, refs_all, strict=True)}
        ref_info = [
            {
                "case": c,
                "role": role,
                "point": p["index"],
                "split": p["split"],
                "values": p["values"],
                "kind": r["kind"],
                "n_coarse": r["n_coarse"],
                "n_fine": r["n_fine"],
                "coarse_fine_diff": r["coarse_fine_diff"],
                "q_bottom": r["q_bottom"],
                "C_D": r.get("C_D"),
                "wall_s": r["wall_s"],
            }
            for (c, p, role), r in zip(need, refs_all, strict=True)
        ]
        jobs, labels = [], []
        for c in hs["cases"]:
            for seed in hs["seeds"]:
                if pp["enabled"]:
                    pt = designs[c]["points"][pp["point"]]
                    ref = ref_of[(c, "test", pt["index"])]
                    d = hs["data"]
                    big = H.snapshot_points(
                        ref, max(pp["n_data"]), d["fields"], d["placement"], float(d["noise"]), int(d["seed"])
                    )
                    for mode in hs["modes"]:
                        for n in [0] if mode == "data_free" else pp["n_data"]:
                            kw = dict(
                                spec=spec,
                                case=c,
                                mode=mode,
                                n_data=n,
                                seed=seed,
                                pt=pt,
                                ref=ref,
                                exclude=big["mask"],
                                out_dir=ex.dir / "runs" / f"{c}_point_{mode}_n{n}_s{seed}",
                                tag=tag,
                            )
                            jobs.append((MOD, "run_point", kw))
                            labels.append((c, "per_point", mode, n, seed))
                if pa["enabled"]:
                    refs = [ref_of[(c, "test", p["index"])] for p in designs[c]["points"]]
                    snaps = (
                        [(p, ref_of[(c, "snapshot", p["index"])]) for p in tpts[c]]
                        if (c, "snapshot", tpts[c][0]["index"]) in ref_of
                        else []
                    )
                    for mode in hs["modes"]:
                        kw = dict(
                            spec=spec,
                            case=c,
                            mode=mode,
                            seed=seed,
                            design=designs[c],
                            refs=refs,
                            snaps=snaps,
                            out_dir=ex.dir / "runs" / f"{c}_parametric_{mode}_s{seed}",
                            tag=tag,
                        )
                        jobs.append((MOD, "run_parametric", kw))
                        labels.append((c, "parametric", mode, None, seed))

        def show(i: int, out, sec: float) -> None:
            for r in out if isinstance(out, list) else [out]:
                extra = f" C_D_err={r['C_D_rel_err']:.2e}" if "C_D_rel_err" in r else ""
                fit = f" fit={r['data_fit']:.1e}" if r["data_fit"] is not None else ""
                print(
                    f"[{r['case']} {r['kind']:10s} {r['mode']:9s} n={r['n_data']:<5d} s{r['seed']} p{r['point']} {r['split']:6s}] "
                    f"worst={r['rel_l2_max']:.2e} u={r['rel_l2_u']:.2e} theta={r['rel_l2_theta']:.2e} q_err={r['q_bottom_rel_err']:.2e}"
                    f"{extra}{fit} train {r['train_wall_s']:.0f}s {r['status']}",
                    flush=True,
                )

        outs = run_tasks(jobs, exe, show)
        records = [r for o in outs for r in (o if isinstance(o, list) else [o])]
        for r in records:
            ex.log(
                **{
                    k: r[k]
                    for k in (
                        "case",
                        "kind",
                        "mode",
                        "n_data",
                        "n_snapshots",
                        "seed",
                        "point",
                        "split",
                        "Pe",
                        "rel_l2_max",
                        "rel_l2_u",
                        "rel_l2_theta",
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
                "snapshots": tpts.get(c, []),
                "dims": [[x.name, x.lo, x.hi, x.scale] for x in d["dims"]],
            }
            for c, d in designs.items()
        }
        ex.metrics.update(
            {
                "records": records,
                "references": ref_info,
                "designs": dsave,
                "git": git_hash(),
                "torch": torch.__version__,
            }
        )
        ex.metrics["figures"] = figures(records, tag, ex)
        table = (
            ex.dir / "hybrid.md"
            if custom or tag != spec["profile"]
            else ROOT / "docs" / "tables" / "phase4_6_hybrid.md"
        )
        hybrid_table(records, ref_info, spec, table, tag)
        ex.metrics["table"] = str(table)
    print(f"wrote {table}")


def figures(records: list[dict], tag: str, ex: Experiment) -> list[str]:
    paths = []
    cases = list(dict.fromkeys(r["case"] for r in records))
    pts = [r for r in records if r["kind"] == "per_point"]
    if pts:
        fig, axs = plt.subplots(1, len(cases), figsize=(5.4 * len(cases), 3.8), squeeze=False)
        for ax, c in zip(axs[0], cases, strict=True):
            rs = [r for r in pts if r["case"] == c]
            for mode in ("data_only", "hybrid"):
                m = sorted((r for r in rs if r["mode"] == mode), key=lambda r: r["n_data"])
                if m:
                    ax.loglog(
                        [r["n_data"] for r in m], [r["rel_l2_max"] for r in m], "o-", color=COLORS[mode], label=mode
                    )
            for r in rs:
                if r["mode"] == "data_free":
                    ax.axhline(r["rel_l2_max"], color=COLORS["data_free"], ls="--", label="data_free (physics only)")
            ax.axhline(r["ref_unc_max"], color="0.6", ls=":", label="reference coarse-fine")
            ax.set(
                xlabel="number of data points (snapshot cells)",
                ylabel="worst-field rel $L_2$ (held-out cells)",
                title=f"{c}: one test point",
            )
            ax.legend(fontsize=6)
        paths += save_fig(fig, f"p4_6_error_vs_ndata_{tag}", ex.dir)
    par = [r for r in records if r["kind"] == "parametric"]
    if par:
        fig, axs = plt.subplots(1, len(cases), figsize=(5.4 * len(cases), 3.8), squeeze=False)
        for ax, c in zip(axs[0], cases, strict=True):
            for k, mode in enumerate(dict.fromkeys(r["mode"] for r in par)):
                m = [r for r in par if r["case"] == c and r["mode"] == mode]
                ax.semilogy(
                    [r["point"] + 0.12 * (k - 1) for r in m],
                    [r["rel_l2_max"] for r in m],
                    "o",
                    color=COLORS[mode],
                    label=mode,
                )
            ax.set(xlabel="test point (interp then extrap)", ylabel="worst-field rel $L_2$", title=f"{c}: parametric")
            ax.set_xticks(sorted({r["point"] for r in par if r["case"] == c}))
            ax.legend(fontsize=6)
        paths += save_fig(fig, f"p4_6_parametric_{tag}", ex.dir)
    return [str(p) for p in paths]


if __name__ == "__main__":
    main()
