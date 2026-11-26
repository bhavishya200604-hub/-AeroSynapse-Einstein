"""Step 6B.2b gap run G3 (paper/PREREG_G1_G3.md): is the smoke-budget error floor budget-limited?

(i)  parametric D_graetz full-exact_psi at the full profile (trained by experiments/p4_7_ablation.py, see
     configs/pinn/gap_g3.yaml); metric = mean worst-field rel L2 over the interpolation test points (as G1), compared
     with the Step 4.7 smoke run of the same seed.
(ii) B_slab Bi 1, the Step 6.1 base setting, trained here at the full profile (Step 6.1 code path); metric = rel L2
     vs the analytic solution, compared with the Step 6.1 base run of the same seed.
Test (pre-registered): per design a paired t-test on log(full / smoke), Holm over the designs, bootstrap CI;
BUDGET-LIMITED if Holm p < 0.05 and the geometric-mean ratio < 0.5. Designs whose full runs are not (yet) available
are reported as pending, not tested.

    python -m aerosynapse.thermal.experiments.p6b_g3_budget [--config my.yaml] [--set key=value] [--tag name] [--parts slab]
                                        [--reuse] [--analysis-only]
"""

from __future__ import annotations

import argparse
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

MOD = "aerosynapse.thermal.experiments.p6b_g3_budget"
DEFAULT = ROOT / "configs" / "pinn" / "gap_g3.yaml"


def load_cfg(files: list[str], sets: list[str]) -> dict:
    cfg = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    for f in files:
        cfg = deep_merge(cfg, coerce_numbers(yaml.safe_load(Path(f).read_text(encoding="utf-8")) or {}))
    return deep_merge(cfg, parse_set(sets))


def slab_task(config: str, profile: str, problem: str, value: float, seed: int, out_dir: str) -> dict:
    """One Step 6.1 base training at another profile (same function as Step 6.1)."""
    from aerosynapse.thermal.experiments.p6_1_scaling import train_task

    return train_task([str(ROOT / config)], profile, {}, problem, float(value), {}, int(seed), out_dir) | {
        "seed": int(seed)
    }


# --------------------------------------------------------------------------- collect
def graetz_errors(run: Path, configuration: str) -> dict[int, dict]:
    """Per seed: mean worst-field rel L2 over the interpolation points, and the training wall time."""
    f = run / "metrics.json"
    if not f.exists():
        return {}
    out: dict[int, dict] = {}
    for r in json.loads(f.read_text(encoding="utf-8"))["records"]:
        if r.get("config") == configuration and r.get("split") == "interp":
            e = out.setdefault(
                int(r["seed"]), {"err": [], "train_wall_s": float(r["train_wall_s"]), "status": r["status"]}
            )
            e["err"].append(float(r["rel_l2_max"]))
    return {
        s: {"error": float(np.mean(v["err"])), **{k: v[k] for k in ("train_wall_s", "status")}} for s, v in out.items()
    }


def slab_smoke(run: Path, problem: str) -> dict[int, dict]:
    """Step 6.1 base runs (factor "base": the base setting, trained once per seed and shared by the sweeps)."""
    met = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    return {
        int(r["seed"]): {"error": float(r["rel_l2"]), "train_wall_s": float(r["wall_s"]), "status": r["status"]}
        for r in met["runs"]
        if r["problem"] == problem and r["factor"] == "base"
    }


def _rows(rows: list[dict]) -> dict[int, dict]:
    return {
        int(r["seed"]): {"error": float(r["rel_l2"]), "train_wall_s": float(r["wall_s"]), "status": r["status"]}
        for r in rows
    }


def reproduction(cfg: dict, rerun: list[dict]) -> dict | None:
    """Smoke-budget slab trainings re-run here vs the Step 6.1 base runs of the same seed (bit-reproducible training)."""
    ref = slab_smoke(ROOT / cfg["slab"]["smoke_run"], cfg["slab"]["problem"])
    pairs = [(int(r["seed"]), float(r["rel_l2"]), ref[int(r["seed"])]["error"]) for r in rerun if int(r["seed"]) in ref]
    if not pairs:
        return None
    dev = max(abs(a - b) / b for _, a, b in pairs)
    return {"pairs": pairs, "max_rel_diff": dev, "summary": f"{len(pairs)} seed(s), max relative difference {dev:.2e}"}


def collect(cfg: dict, slab_full: list[dict], slab_smoke_rerun: list[dict] | None = None) -> dict:
    g = cfg["graetz"]
    smoke = graetz_errors(ROOT / g["smoke_run"], g["configuration"])
    full: dict[int, dict] = {}
    for r in g["full_runs"]:
        full |= graetz_errors(ROOT / r, g["configuration"])
    sl = cfg["slab"]
    return {
        "graetz": {"smoke": smoke, "full": full, "seeds": [int(s) for s in g["seeds"]]},
        "slab": {
            # Step 6.1 base runs where they exist; smoke-budget re-runs (this script) for the other seeds
            "smoke": _rows(slab_smoke_rerun or []) | slab_smoke(ROOT / sl["smoke_run"], sl["problem"]),
            "full": _rows(slab_full),
            # the registered test uses the pre-registered seeds; further seeds are an extension (collect_extra)
            "seeds": [int(s) for s in sl.get("registered_seeds", sl["seeds"])],
        },
    }


def collect_extra(cfg: dict, data: dict) -> dict:
    """Designs reported OUTSIDE the registered Holm family: the slab at all its seeds (extension of the registered
    3-seed design) and the exploratory designs of `exploratory:` (post hoc, logged in DECISIONS.md before running)."""
    out = {}
    sl = cfg["slab"]
    if [int(s) for s in sl["seeds"]] != data["slab"]["seeds"]:
        out["slab, all seeds (extension)"] = dict(data["slab"], seeds=[int(s) for s in sl["seeds"]])
    for name, x in (cfg.get("exploratory") or {}).items():
        full: dict[int, dict] = {}
        for r in x["full_runs"]:
            full |= graetz_errors(ROOT / r, x["configuration"])
        out[f"{name} (exploratory)"] = {
            "smoke": graetz_errors(ROOT / x["smoke_run"], x["configuration"]),
            "full": full,
            "seeds": [int(s) for s in x["seeds"]],
        }
    return out


def analyse_extra(extra: dict, cfg: dict) -> dict:
    """Each extra design on its own (no correction across them; labelled as extension / exploratory)."""
    return {k: analyse({k: d}, cfg)[k] for k, d in extra.items()}


def analyse(data: dict, cfg: dict) -> dict:
    an = cfg["analysis"]
    out = {}
    for k, d in data.items():
        seeds = [s for s in d["seeds"] if s in d["smoke"] and s in d["full"]]
        e = {"seeds": seeds, "pending": [s for s in d["seeds"] if s not in d["full"]]}
        if len(seeds) >= 2:
            e["test"] = S.paired_log_ratio(
                np.array([d["full"][s]["error"] for s in seeds]),
                np.array([d["smoke"][s]["error"] for s in seeds]),
                int(an["n_boot"]),
                int(an["boot_seed"]),
            )
        else:
            e["test"] = None
            e["note"] = f"{len(seeds)} paired seed(s): no test"
        out[k] = e
    keys = list(out)
    p = [(out[k]["test"] or {}).get("p_t", float("nan")) for k in keys]
    for k, pa in zip(keys, S.holm(p), strict=True):
        e = out[k]
        e["p_holm"] = pa
        t = e["test"]
        e["budget_limited"] = bool(
            t and np.isfinite(pa) and pa < float(an["alpha"]) and t["gm_ratio"] < float(an["ratio_max"])
        )
        complete = not e["pending"]
        if t is None:
            e["verdict"] = "NO TEST" + ("" if complete else f" (seeds {e['pending']} pending)")
        else:
            e["verdict"] = ("BUDGET-LIMITED" if e["budget_limited"] else "NOT BUDGET-LIMITED at this budget") + (
                "" if complete else f" (INTERIM: seeds {e['pending']} pending)"
            )
    return out


# --------------------------------------------------------------------------- figure / report
def figure(data: dict, tag: str, odir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    from aerosynapse.thermal.viz.style import apply_style

    apply_style()
    fig, axs = plt.subplots(1, len(data), figsize=(4.5 * len(data), 3.8), layout="constrained", squeeze=False)
    for ax, (k, d) in zip(axs[0], data.items(), strict=True):
        for s in d["seeds"]:
            y = [d["smoke"].get(s, {}).get("error"), d["full"].get(s, {}).get("error")]
            xs = [x for x, v in zip((0, 1), y, strict=True) if v is not None]
            ys = [v for v in y if v is not None]
            if not xs:
                continue
            ax.plot(xs, ys, "o-", color="#0072B2", alpha=0.8)
            ax.annotate(
                f"s{s}",
                (xs[-1], ys[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set(
            xticks=[0, 1],
            xticklabels=["smoke", "full"],
            yscale="log",
            xlim=(-0.3, 1.4),
            title=k,
            ylabel="rel L2 vs analytic" if k.startswith("slab") else "mean worst rel L2 (interp)",
        )
    paths = save_fig(fig, f"p6b_g3_budget_{tag}", odir, docs=False)
    plt.close(fig)
    return [str(p) for p in paths]


def report(met: dict, path: Path, tag: str) -> None:
    L = [
        f"# Gap run G3 — budget limitation — `{tag}` (auto-generated by experiments/p6b_g3_budget.py)",
        "",
        "Pre-registered in paper/PREREG_G1_G3.md. Verification only (errors vs reference / analytic solutions).",
        "",
        "| design | seed | smoke error | full error | full / smoke | full training wall [s] |",
        "|---|---|---|---|---|---|",
    ]
    for k, d in met["data"].items():
        for s in d["seeds"]:
            sm, fu = d["smoke"].get(s), d["full"].get(s)
            ratio = f"{fu['error'] / sm['error']:.3g}" if sm and fu else "—"
            L.append(
                f"| {k} | {s} | {sm['error']:.4g} | "
                + (f"{fu['error']:.4g} | {ratio} | {fu['train_wall_s']:.0f} |" if fu else "pending | — | — |")
                if sm
                else f"| {k} | {s} | — | — | — | — |"
            )
    L.append("")
    for k, e in met["analysis"].items():
        t = e["test"]
        if t:
            L.append(
                f"- **{k}** (n = {t['n']}): geometric-mean ratio full / smoke {t['gm_ratio']:.3g} [95 % CI "
                f"{t['gm_ratio_ci95'][0]:.3g}, {t['gm_ratio_ci95'][1]:.3g}]; paired t p = {t['p_t']:.3g}, Holm "
                f"p = {e['p_holm']:.3g}; Wilcoxon p = {t['p_wilcoxon']:.3g} — **{e['verdict']}**"
            )
        else:
            L.append(f"- **{k}**: {e['note']} — **{e['verdict']}**")
    extra = met.get("extra_analysis") or {}
    if extra:
        L += [
            "",
            "Outside the registered Holm family (each tested on its own; extension or exploratory, see DECISIONS.md):",
            "",
            "| design | seed | smoke error | full error | full / smoke | full training wall [s] |",
            "|---|---|---|---|---|---|",
        ]
        for k, d in met["extra_data"].items():
            for sd in d["seeds"]:
                sm, fu = d["smoke"].get(sd), d["full"].get(sd)
                if not sm:
                    L.append(f"| {k} | {sd} | — | — | — | — |")
                    continue
                full = (
                    f"{fu['error']:.4g} | {fu['error'] / sm['error']:.3g} | {fu['train_wall_s']:.0f}"
                    if fu
                    else "pending | — | —"
                )
                L.append(f"| {k} | {sd} | {sm['error']:.4g} | {full} |")
        L.append("")
        for k, e in extra.items():
            t = e["test"]
            if t:
                L.append(
                    f"- **{k}** (n = {t['n']}): GM ratio full / smoke {t['gm_ratio']:.3g} [95 % CI "
                    f"{t['gm_ratio_ci95'][0]:.3g}, {t['gm_ratio_ci95'][1]:.3g}]; paired t p = {t['p_t']:.3g}; "
                    f"Wilcoxon p = {t['p_wilcoxon']:.3g} — {e['verdict']}"
                )
            else:
                L.append(f"- **{k}**: {e['note']} — {e['verdict']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument(
        "--parts",
        default="slab",
        help="comma list: 'slab' trains design (ii) at the full profile, 'slab_smoke' the same seeds at the smoke "
        "profile (references for seeds without a Step 6.1 run); '' only collects and analyses",
    )
    ap.add_argument("--reuse", action="store_true", help="reuse task results cached in <results>/tasks/")
    ap.add_argument("--analysis-only", action="store_true", help="re-collect and re-analyse (no training)")
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.sets)
    odir = ROOT / "results" / f"p6b_g3_budget_{a.tag}"
    if a.analysis_only:
        met = json.loads((odir / "metrics.json").read_text(encoding="utf-8"))
        met["data"] = collect(cfg, met.get("slab_full") or [], met.get("slab_smoke_rerun") or [])
        met["analysis"] = analyse(met["data"], cfg)
        met["extra_data"] = collect_extra(cfg, met["data"])
        met["extra_analysis"] = analyse_extra(met["extra_data"], cfg)
        met["figures"] = figure(met["data"] | met["extra_data"], a.tag, odir)
        (odir / "metrics.json").write_text(json.dumps(met, indent=2), encoding="utf-8")
        report(met, odir / "g3.md", a.tag)
        print(f"wrote {odir / 'g3.md'}")
        return
    with Experiment(f"p6b_g3_budget_{a.tag}", cfg | {"seed": 0}) as ex:
        sl = cfg["slab"]
        cache = ex.dir / "tasks"
        cache.mkdir(parents=True, exist_ok=True)

        def train(profile: str, seeds: list, prefix: str) -> list[dict]:
            keys = [f"{prefix}_s{x}" for x in seeds]
            jobs = [
                (
                    MOD,
                    "slab_task",
                    dict(
                        config=sl["config"],
                        profile=profile,
                        problem=sl["problem"],
                        value=float(sl["value"]),
                        seed=int(x),
                        out_dir=str(
                            ex.dir / "runs" / f"{sl['problem']}_{profile}_s{x}"
                            if prefix != "slab"
                            else ex.dir / "runs" / f"{sl['problem']}_s{x}"
                        ),
                    ),
                )
                for x in seeds
            ]
            cached = {
                i: json.loads((cache / f"{k}.json").read_text(encoding="utf-8"))
                for i, k in enumerate(keys)
                if a.reuse and (cache / f"{k}.json").exists()
            }
            todo = [i for i in range(len(jobs)) if i not in cached]
            print(f"{len(jobs)} {prefix} trainings at the {profile} profile ({len(cached)} cached)", flush=True)

            def done(j, r, sec):
                (cache / f"{keys[todo[j]]}.json").write_text(json.dumps(r), encoding="utf-8")
                print(f"[{j + 1}/{len(todo)}] {keys[todo[j]]}: rel L2 {r['rel_l2']:.3g} ({sec:.0f} s)", flush=True)

            new = run_tasks([jobs[i] for i in todo], cfg["execution"], done) if todo else []
            res = dict(cached) | dict(zip(todo, new, strict=True))
            return [res[i] for i in range(len(jobs))]

        parts = {x.strip() for x in a.parts.split(",") if x.strip()}
        rows = train(sl["profile"], sl["seeds"], "slab") if "slab" in parts else []
        smoke_rows = train("smoke", sl["seeds"], "slab_smoke") if "slab_smoke" in parts else []
        ex.metrics["slab_smoke_rerun"] = smoke_rows
        ex.metrics["reproduction"] = reproduction(cfg, smoke_rows)
        ex.metrics["slab_full"] = rows
        ex.metrics["data"] = collect(cfg, rows, smoke_rows)
        ex.metrics["analysis"] = analyse(ex.metrics["data"], cfg)
        ex.metrics["extra_data"] = collect_extra(cfg, ex.metrics["data"])
        ex.metrics["extra_analysis"] = analyse_extra(ex.metrics["extra_data"], cfg)
        ex.metrics["git"] = git_hash()
        ex.metrics["figures"] = figure(ex.metrics["data"] | ex.metrics["extra_data"], a.tag, ex.dir)
        report(ex.metrics, ex.dir / "g3.md", a.tag)
    print(f"wrote {odir / 'g3.md'}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"done in {time.perf_counter() - t0:.0f} s")
