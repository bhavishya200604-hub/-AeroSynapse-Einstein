"""Step 6B.2b gap run G2 (paper/PREREG_G1_G3.md): is gradient conflict lower in failing PINN runs? (5 seeds)

1D  A_steady Pe 100, variants mlp / fourier (sigma 0.5) / siren, seeds 0-4, trained exactly as in Step 6.5 (its saved
    config reused; seed 0 repeats Step 6.5 and is a reproduction check). Per run: final relative L2 against the
    analytic solution and, at the final weights on the Step 6.5 point set, the cosine and the cancellation ratio
    |g_pde + g_cond| / (|g_pde| + |g_cond|) of the unweighted PDE and boundary-group gradients. Failure = rel L2 > 0.1.
2D  saved networks (no training here): heated cavity, variant parametric (Step 4.5a seed 0 + G2 seeds 1-4; fails)
    vs Graetz full-exact_psi (Step 4.7 seeds 0-2 + G1 seeds 3-4; succeeds). Unweighted PDE-BC cosine at the final
    (and initial) weights on the Step 6.6b fixed batch (seed 21); error = mean worst-field rel L2 over the interpolation
    test points of the run's own records.
Test (pre-registered): per setting a one-sided Welch t-test failures < successes, Holm over the 2 settings, bootstrap
CI of the difference of means, Spearman(cosine, error). SUPPORTED if Holm p < 0.05, mean failure cosine < -0.5 and
mean success cosine > -0.3.

    python -m aerosynapse.thermal.experiments.p6b_g2_conflict [--config my.yaml] [--set key=value] [--tag name]
                                          [--parts 1d,2d] [--reuse] [--analysis-only]
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import yaml

from aerosynapse.thermal.data.spec import coerce_numbers, deep_merge, parse_set
from aerosynapse.thermal.utils import stats as S
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz.style import save_fig

MOD = "aerosynapse.thermal.experiments.p6b_g2_conflict"
DEFAULT = ROOT / "configs" / "pinn" / "gap_g2.yaml"


def load_cfg(files: list[str], sets: list[str]) -> dict:
    cfg = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    for f in files:
        cfg = deep_merge(cfg, coerce_numbers(yaml.safe_load(Path(f).read_text(encoding="utf-8")) or {}))
    return deep_merge(cfg, parse_set(sets))


# --------------------------------------------------------------------------- worker tasks
def oned_task(config: str, case: str, variant: str, seed: int, out_dir: str) -> dict:
    """Train one Step 6.5 network with another seed (same code path as Step 6.5) and measure the conflict."""
    from aerosynapse.thermal.experiments import p6_5_diagnostics as D65
    from aerosynapse.thermal.pinn import problems as P
    from aerosynapse.thermal.pinn import spec as SP

    files = [str(ROOT / config)]
    ds = SP.load_spec(files, "smoke")["diagnostics_study"]
    c = ds["cases"][case]
    spec = D65._spec(files, "smoke", {}, ds["variants"][variant] or {}, seed)
    prob = P.build_problem(c["problem"], spec["problems"][c["problem"]], float(c["value"]))
    t0 = time.perf_counter()
    net, _, _, res, wall = D65.train_with_snapshots(spec, prob, Path(out_dir), ds["snapshot_every"])
    x, t, y = D65._truth_grid(prob, float(c["value"]), int(ds["spectrum"]["nx"]))
    rel = float(np.linalg.norm(D65._predict(net, prob, x, t, float(c["value"])) - y) / max(np.linalg.norm(y), 1e-300))
    ds_seed = copy.deepcopy(ds) | {"seed": int(seed)}
    conf = D65.conflict_task(files, "smoke", {}, ds_seed, case, variant, str(Path(out_dir) / "checkpoint.pt"))
    return {
        "case": case,
        "variant": variant,
        "seed": int(seed),
        "rel_l2": rel,
        "status": res["status"],
        "train_wall_s": wall,
        "init": conf["init"],
        "final": conf["final"],
        "wall_s": time.perf_counter() - t0,
    }


def _errors(run: Path, case: str, select: str) -> dict[int, float]:
    """Mean worst-field rel L2 over the interpolation test points, per seed (the run's own records)."""
    met = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    key = "config" if case == "D_graetz" else "variant"
    out: dict[int, list[float]] = {}
    for r in met["records"]:
        if r.get(key) == select and r.get("split") == "interp":
            out.setdefault(int(r["seed"]), []).append(float(r["rel_l2_max"]))
    return {s: float(np.mean(v)) for s, v in out.items()}


def twod_task(case: str, select: str, run: str, seeds: list[int], points: dict) -> list[dict]:
    """Unweighted PDE-BC conflict of saved networks at the initial and final weights (Step 6.6b batch)."""
    from aerosynapse.thermal.experiments.p6_4_sensitivity import ADAPTERS
    from aerosynapse.thermal.experiments.p6_6b_2d import _conflict

    rdir = ROOT / run
    if case == "D_graetz":
        ad = ADAPTERS[case]({"surrogate": {"run": run, "configuration": select}})
        have = [int(s) for s in ad.base["ablation_study"]["seeds"]]
    else:
        ad = ADAPTERS[case]({"surrogate": {"run": run, "variants": [select], "seeds": seeds}})
        have = list(seeds)
    err = _errors(rdir, case, select)
    out = []
    for i, seed in enumerate(have):
        if seed not in seeds:
            continue
        name, prob, net = ad.members[i]
        p0 = next(net.parameters())
        pts = prob.sample(
            int(points["n_pde"]), 0, int(points["n_bc"]), "sobol", int(points["seed"]), p0.device, p0.dtype
        )
        fin = _conflict(net, prob, pts, None)
        ini = _conflict(ad.initial(i), prob, pts, None)
        out.append(
            {
                "case": case,
                "select": select,
                "run": run,
                "seed": seed,
                "member": name,
                "error": err.get(seed),
                "init": ini["unweighted"],
                "final": fin["unweighted"],
                "groups": fin["groups"],
                "group_cosine_final": fin["cosine"],
            }
        )
    return out


# --------------------------------------------------------------------------- analysis
def analyse(rows1: list[dict], rows2: list[dict], cfg: dict) -> dict:
    an = cfg["analysis"]
    nb, bs = int(an["n_boot"]), int(an["boot_seed"])
    out: dict = {"settings": {}}
    sets = {}
    if rows1:
        thr = float(cfg["oned"]["fail_threshold"])
        sets["1d"] = (
            [r["final"]["cosine"] for r in rows1 if r["rel_l2"] > thr],
            [r["final"]["cosine"] for r in rows1 if r["rel_l2"] <= thr],
            [r["final"]["cosine"] for r in rows1],
            [r["rel_l2"] for r in rows1],
        )
    if rows2:
        fc = cfg["twod"]["fails"]["case"]
        ok = [r for r in rows2 if r["error"] is not None]
        sets["2d"] = (
            [r["final"]["cosine"] for r in rows2 if r["case"] == fc],
            [r["final"]["cosine"] for r in rows2 if r["case"] != fc],
            [r["final"]["cosine"] for r in ok],
            [r["error"] for r in ok],
        )
    for k, (fail, succ, cos, err) in sets.items():
        e = {"n_fail": len(fail), "n_success": len(succ)}
        if len(fail) >= 2 and len(succ) >= 2:
            e["welch"] = S.welch_less(np.array(fail), np.array(succ), nb, bs)
        else:
            e["welch"] = None
            e["note"] = "fewer than 2 runs in a group: no test"
        e["spearman_cos_error"] = S.spearman(np.array(cos), np.array(err)) if len(cos) >= 3 else None
        out["settings"][k] = e
    keys = list(out["settings"])
    p = [(out["settings"][k]["welch"] or {}).get("p", float("nan")) for k in keys]
    for k, pa in zip(keys, S.holm(p), strict=True):
        e = out["settings"][k]
        e["p_holm"] = pa
        w = e["welch"]
        e["supported"] = bool(
            w is not None
            and np.isfinite(pa)
            and pa < float(an["alpha"])
            and w["mean_a"] < float(an["fail_cos_max"])
            and w["mean_b"] > float(an["success_cos_min"])
        )
        e["verdict"] = "SUPPORTED" if e["supported"] else "NOT SUPPORTED"
    return out


# --------------------------------------------------------------------------- figures / report
def figures(rows1: list[dict], rows2: list[dict], thr: float, tag: str, odir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    from aerosynapse.thermal.viz.style import apply_style

    apply_style()
    paths = []
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    # Okabe-Ito: vermillion = the variant / case that fails, blue and bluish green = those that succeed
    cols = {"mlp": "#D55E00", "fourier": "#0072B2", "siren": "#009E73", "C_cavity": "#D55E00", "D_graetz": "#0072B2"}
    for r in rows1:
        axs[0].scatter(r["final"]["cosine"], r["rel_l2"], color=cols.get(r["variant"], "k"), s=28)
    for v in sorted({r["variant"] for r in rows1}):
        axs[0].scatter([], [], color=cols.get(v, "k"), label=v)
    axs[0].axhline(thr, color="0.5", lw=0.8, ls=":")
    axs[0].set(
        xscale="linear",
        yscale="log",
        xlabel="PDE-BC gradient cosine (final)",
        ylabel="final relative L2",
        title="1D A_steady Pe 100",
    )
    for r in rows2:
        if r["error"] is not None:
            axs[1].scatter(r["final"]["cosine"], r["error"], color=cols.get(r["case"], "k"), s=28)
    for c in sorted({r["case"] for r in rows2}):
        axs[1].scatter([], [], color=cols.get(c, "k"), label=c)
    axs[1].set(
        yscale="log",
        xlabel="PDE-BC gradient cosine (final, unweighted)",
        ylabel="mean worst-field rel L2 (interp)",
        title="2D saved networks",
    )
    for ax in axs:
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlim(-1.05, 1.05)
        ax.legend(fontsize=8)
    paths += save_fig(fig, f"p6b_g2_conflict_{tag}", odir, docs=False)
    # 3D: cosine, cancellation ratio and log error per run (1D) — where in this space do failures sit?
    if rows1:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(projection="3d")
        for r in rows1:
            ax.scatter(
                r["final"]["cosine"],
                r["final"]["cancellation"],
                np.log10(r["rel_l2"]),
                color=cols.get(r["variant"], "k"),
                s=30,
            )
        for v in sorted({r["variant"] for r in rows1}):
            ax.scatter([], [], [], color=cols.get(v, "k"), label=v)
        ax.legend(fontsize=8)
        ax.set(xlabel="cosine", ylabel="cancellation", zlabel="log10 rel L2", title="1D runs (final weights)")
        paths += save_fig(fig, f"p6b_g2_conflict3d_{tag}", odir, docs=False)
    plt.close("all")
    return [str(p) for p in paths]


def report(met: dict, path: Path, tag: str) -> None:
    L = [
        f"# Gap run G2 — gradient conflict at 5 seeds — `{tag}` (auto-generated by experiments/p6b_g2_conflict.py)",
        "",
        "Pre-registered in paper/PREREG_G1_G3.md (committed before the run). Verification of training behaviour only.",
        "",
    ]
    if met.get("oned"):
        L += [
            "| 1D variant | seed | final rel L2 | cosine init → final | cancellation init → final | status |",
            "|---|---|---|---|---|---|",
        ]
        for r in met["oned"]:
            L.append(
                f"| {r['variant']} | {r['seed']} | {r['rel_l2']:.3g} | {r['init']['cosine']:+.3f} → {r['final']['cosine']:+.3f} | "
                f"{r['init']['cancellation']:.3f} → {r['final']['cancellation']:.3f} | {r['status']} |"
            )
        rc = met.get("reproduction")
        if rc:
            L += ["", f"Reproduction of Step 6.5 (seed 0): {rc['summary']}"]
        L.append("")
    if met.get("twod"):
        L += [
            "| 2D case | seed | run | error (mean worst rel L2, interp) | cosine init → final | cancellation final |",
            "|---|---|---|---|---|---|",
        ]
        for r in met["twod"]:
            e = "—" if r["error"] is None else f"{r['error']:.3g}"
            L.append(
                f"| {r['case']} {r['select']} | {r['seed']} | {r['run']} | {e} | {r['init']['cosine']:+.3f} → "
                f"{r['final']['cosine']:+.3f} | {r['final']['cancellation']:.3f} |"
            )
        L.append("")
    an = met.get("analysis") or {}
    for k, e in (an.get("settings") or {}).items():
        w = e.get("welch")
        if w:
            L.append(
                f"- **{k}**: failures n={e['n_fail']} mean cosine {w['mean_a']:+.3f}; successes n={e['n_success']} "
                f"mean {w['mean_b']:+.3f}; difference {w['diff']:+.3f} [95 % CI {w['diff_ci95'][0]:+.3f}, "
                f"{w['diff_ci95'][1]:+.3f}]; Welch one-sided p = {w['p']:.3g}, Holm p = {e['p_holm']:.3g}; "
                + (
                    f"Spearman(cosine, error) rho = {e['spearman_cos_error']['rho']:+.2f} (p {e['spearman_cos_error']['p']:.2g}); "
                    if e.get("spearman_cos_error")
                    else ""
                )
                + f"**{e['verdict']}**"
            )
        else:
            L.append(f"- **{k}**: {e.get('note')} — **{e['verdict']}**")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def reproduction(rows1: list[dict], cfg: dict) -> dict | None:
    """Seed 0 of every variant vs the Step 6.5 run (same code path; training is bit-reproducible)."""
    src = ROOT / Path(cfg["oned"]["config"]).parent / "metrics.json"
    if not src.exists():
        return None
    met = json.loads(src.read_text(encoding="utf-8"))
    ref = {r["variant"]: r["rel_l2"][-1] for r in met.get("cases", []) if r["case"] == cfg["oned"]["case"]}
    rc = {r["variant"]: r["final"]["cosine"] for r in met.get("conflict", []) if r["case"] == cfg["oned"]["case"]}
    items = []
    for r in rows1:
        if r["seed"] == 0 and r["variant"] in ref:
            items.append(
                {
                    "variant": r["variant"],
                    "rel_l2": r["rel_l2"],
                    "rel_l2_step65": ref[r["variant"]],
                    "cosine": r["final"]["cosine"],
                    "cosine_step65": rc.get(r["variant"]),
                }
            )
    dev = max(
        (abs(i["rel_l2"] - i["rel_l2_step65"]) / max(abs(i["rel_l2_step65"]), 1e-300) for i in items), default=None
    )
    return {
        "items": items,
        "max_rel_diff_rel_l2": dev,
        "summary": "no seed-0 runs"
        if dev is None
        else f"max relative difference of the final rel L2 {dev:.2e} over {len(items)} variants",
    }


# --------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--parts", default="1d,2d")
    ap.add_argument("--reuse", action="store_true", help="reuse task results cached in <results>/tasks/")
    ap.add_argument("--analysis-only", action="store_true", help="re-analyse the saved rows (no computation)")
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.sets)
    parts = {p.strip() for p in a.parts.split(",")}
    odir = ROOT / "results" / f"p6b_g2_conflict_{a.tag}"
    if a.analysis_only:
        met = json.loads((odir / "metrics.json").read_text(encoding="utf-8"))
        met["analysis"] = analyse(met.get("oned") or [], met.get("twod") or [], cfg)
        met["figures"] = figures(
            met.get("oned") or [], met.get("twod") or [], float(cfg["oned"]["fail_threshold"]), a.tag, odir
        )
        (odir / "metrics.json").write_text(json.dumps(met, indent=2), encoding="utf-8")
        report(met, odir / "g2.md", a.tag)
        print(f"wrote {odir / 'g2.md'}")
        return
    with Experiment(f"p6b_g2_conflict_{a.tag}", cfg | {"seed": 0}) as ex:
        cache = ex.dir / "tasks"
        cache.mkdir(parents=True, exist_ok=True)
        jobs, keys = [], []
        if "1d" in parts:
            o = cfg["oned"]
            for v in o["variants"]:
                for s in o["seeds"]:
                    jobs.append(
                        (
                            MOD,
                            "oned_task",
                            dict(
                                config=o["config"],
                                case=o["case"],
                                variant=v,
                                seed=int(s),
                                out_dir=str(ex.dir / "runs" / f"{o['case']}_{v}_s{s}"),
                            ),
                        )
                    )
                    keys.append(f"1d_{v}_s{s}")
        if "2d" in parts:
            t = cfg["twod"]
            for grp in ("fails", "succeeds"):
                g = t[grp]
                for r in g["runs"]:
                    jobs.append(
                        (
                            MOD,
                            "twod_task",
                            dict(
                                case=g["case"],
                                select=g["select"],
                                run=r["run"],
                                seeds=[int(x) for x in r["seeds"]],
                                points=t["points"],
                            ),
                        )
                    )
                    keys.append(f"2d_{g['case']}_{Path(r['run']).name}")
        cached = {
            i: json.loads((cache / f"{k}.json").read_text(encoding="utf-8"))
            for i, k in enumerate(keys)
            if a.reuse and (cache / f"{k}.json").exists()
        }
        todo = [i for i in range(len(jobs)) if i not in cached]
        print(f"{len(jobs)} tasks ({len(cached)} cached)", flush=True)

        def done(j, r, sec):
            i = todo[j]
            (cache / f"{keys[i]}.json").write_text(json.dumps(r), encoding="utf-8")
            msg = (
                f"rel L2 {r['rel_l2']:.3g}, cosine {r['final']['cosine']:+.3f}"
                if isinstance(r, dict)
                else ", ".join(f"s{x['seed']} cos {x['final']['cosine']:+.3f}" for x in r)
            )
            print(f"[{j + 1}/{len(todo)}] {keys[i]}: {msg} ({sec:.0f} s)", flush=True)

        new = run_tasks([jobs[i] for i in todo], cfg["execution"], done) if todo else []
        res = dict(cached) | dict(zip(todo, new, strict=True))
        outs = [res[i] for i in range(len(jobs))]
        ex.metrics["oned"] = [o for k, o in zip(keys, outs, strict=True) if k.startswith("1d_")]
        ex.metrics["twod"] = [x for k, o in zip(keys, outs, strict=True) if k.startswith("2d_") for x in o]
        ex.metrics["reproduction"] = reproduction(ex.metrics["oned"], cfg)
        ex.metrics["analysis"] = analyse(ex.metrics["oned"], ex.metrics["twod"], cfg)
        ex.metrics["git"] = git_hash()
        ex.metrics["figures"] = figures(
            ex.metrics["oned"], ex.metrics["twod"], float(cfg["oned"]["fail_threshold"]), a.tag, ex.dir
        )
        report(ex.metrics, ex.dir / "g2.md", a.tag)
    print(f"wrote {odir / 'g2.md'}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"done in {time.perf_counter() - t0:.0f} s")
