"""Phase 9 Step 9.1: one command that regenerates every table and figure of the project from its saved results.

    python tasks.py reproduce                  # level 'saved' (default): tables, figures, papers' numbers / tables /
                                               #   figures, catalogue, gallery, claims audit  (~1 h on the reference PC)
    python tasks.py reproduce --full           # + the full-profile Phase 1 solver reruns           (+ ~3 h)
    python tasks.py reproduce --animations     # + every animation, gallery and showreel            (+ ~0.5 h)
    python tasks.py reproduce --retrain        # prints the from-scratch runs (training included) and their recorded
                                               #   wall times; nothing is started (the user stopped all training, D135)
    python tasks.py reproduce --dry-run        # the steps and their expected runtimes only

Expected runtimes are measured, not estimated: they are the wall times of the last complete runs on the reference
machine (results/p7_3_rerender_default, results/p7_animate_default, the catalogue's recorded run times; hardware in
docs/RESULTS_REGISTRY.md R6.7a). Writes results/p9_reproduce_<tag>/metrics.json (per step: exit code, wall time).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash

PY = sys.executable


def _measured(rel: str) -> dict | None:
    f = ROOT / rel
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def rerender_seconds(full: bool) -> float | None:
    m = _measured("results/p7_3_rerender_default/metrics.json")
    if not m or "runs" not in m:
        return None
    return sum(r["wall_s"] for r in m["runs"] if r.get("full", False) == full)


def steps(a) -> list[dict]:
    anim = _measured("results/p7_animate_default/metrics.json")
    s = [
        {
            "name": "tables and figures from saved results (+ Phase 1 / 2 solver reruns, smoke profile)",
            "cmd": [PY, "-m", "aerosynapse.thermal.experiments.p7_3_rerender", "--skip-full", "--tag", f"reproduce_{a.tag}"],
            "expected_s": rerender_seconds(False),
        },
    ]
    for p in ("paper1", "paper2"):
        for mod in ("make_numbers", "make_tables", "make_figures"):
            s.append({"name": f"{p}: {mod}", "cmd": [PY, "-m", f"paper.{p}.{mod}"], "expected_s": None})
        s.append({"name": f"{p}: lint", "cmd": [PY, "-m", "aerosynapse.thermal.paper_tools.lint_paper", f"paper/{p}"], "expected_s": None})
    s += [
        {"name": "figure catalogue", "cmd": [PY, "-m", "aerosynapse.thermal.viz.catalogue"], "expected_s": None},
        {"name": "figure gallery", "cmd": [PY, "-m", "aerosynapse.thermal.viz.gallery"], "expected_s": None},
        {
            "name": "claims audit",
            "cmd": [PY, "-m", "aerosynapse.thermal.experiments.p9_claims_audit", "--tag", f"reproduce_{a.tag}"],
            "expected_s": None,
        },
    ]
    if a.full:
        s.insert(
            1,
            {
                "name": "full-profile Phase 1 solver reruns",
                "cmd": [
                    PY,
                    "-m",
                    "aerosynapse.thermal.experiments.p7_3_rerender",
                    "--only",
                    "--profile full",
                    "--tag",
                    f"reproduce_full_{a.tag}",
                ],
                "expected_s": rerender_seconds(True),
            },
        )
    if a.animations:
        s.append(
            {
                "name": "animations, gallery and showreel",
                "cmd": [PY, "-m", "aerosynapse.thermal.experiments.p7_animate"],
                "expected_s": sum(e["render_s"] + e.get("prepare_s", 0) for e in anim["animations"]) if anim else None,
            }
        )
    return s


def retrain_plan() -> list[dict]:
    """Every distinct script of the figure catalogue rerun from scratch, with its recorded wall time."""
    figs = json.loads((ROOT / "docs/figures/catalogue.json").read_text(encoding="utf-8"))
    seen: dict[str, dict] = {}
    for f in figs:
        cmd = f["rerender"].split(" --table-only")[0].split(" --figures-only")[0].split(" --compile-only")[0]
        e = seen.setdefault(cmd, {"cmd": cmd, "recorded_s": f.get("run_wall_s") or 0.0})
        e["recorded_s"] = max(e["recorded_s"], f.get("run_wall_s") or 0.0)
    return sorted(seen.values(), key=lambda e: -e["recorded_s"])


def hms(s: float | None) -> str:
    if s is None:
        return "not yet measured"
    return f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min" if s >= 60 else f"{s:.0f} s"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--animations", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--doc-only", action="store_true", help="rewrite docs/REPRODUCE.md from the saved run")
    a = ap.parse_args()
    if a.doc_only:
        write_doc(_measured("results/p9_reproduce_default/metrics.json"))
        return
    if a.retrain:
        plan = retrain_plan()
        for e in plan:
            print(f"{hms(e['recorded_s']):>16}  {e['cmd']}")
        print(
            f"{hms(sum(e['recorded_s'] for e in plan)):>16}  in total (recorded on the reference machine, sequential)"
        )
        print("Nothing started: retraining is not part of reproduce (D135). Run single commands above if needed.")
        return
    plan = steps(a)
    known = [p["expected_s"] for p in plan if p["expected_s"]]
    print(f"{len(plan)} steps; expected at least {hms(sum(known))} (measured steps only)")
    for p in plan:
        print(f"  {hms(p['expected_s']):>16}  {p['name']}")
    if a.dry_run:
        return
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    t0 = time.time()
    with Experiment(f"p9_reproduce_{a.tag}", {"seed": 0, "full": a.full, "animations": a.animations}) as ex:
        runs = []
        for p in plan:
            t1 = time.time()
            r = subprocess.run(p["cmd"], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
            runs.append(
                {
                    "name": p["name"],
                    "cmd": " ".join(p["cmd"][1:]),
                    "exit": r.returncode,
                    "wall_s": round(time.time() - t1, 1),
                    "expected_s": p["expected_s"],
                    "tail": (r.stdout + r.stderr).strip().splitlines()[-5:],
                }
            )
            print(f"[{'ok' if r.returncode == 0 else 'FAIL'}] {runs[-1]['wall_s']:8.1f} s  {p['name']}", flush=True)
            ex.metrics["runs"] = runs
        ex.metrics.update(
            {
                "runs": runs,
                "wall_s": round(time.time() - t0, 1),
                "failed": [x["name"] for x in runs if x["exit"] != 0],
                "git": git_hash(),
            }
        )
    if a.tag == "default":
        write_doc(ex.metrics)
    if ex.metrics["failed"]:
        raise SystemExit(f"failed steps: {ex.metrics['failed']}")


def write_doc(m: dict) -> None:
    """docs/REPRODUCE.md: the levels and their measured runtimes (this run, the full-profile reruns and the
    animations as last measured, the recorded from-scratch times), all read from results files."""
    full = rerender_seconds(True)
    anim = _measured("results/p7_animate_default/metrics.json")
    anim_s = sum(e["render_s"] + e.get("prepare_s", 0) for e in anim["animations"]) if anim else None
    plan = retrain_plan()
    L = [
        "# Reproducing the project",
        "",
        "Every table and figure regenerates from the saved results with one command; nothing is retrained. "
        "Runtimes below are measured on the reference machine (docs/RESULTS_REGISTRY.md R6.7a) and read from "
        "results files (auto-generated by experiments/p9_reproduce.py).",
        "",
        "| command | what it regenerates | measured runtime | source |",
        "|---|---|---|---|",
        f"| `python tasks.py reproduce` | tables, figures, papers' numbers / tables / figures, catalogue, gallery, "
        f"claims audit ({len(m['runs'])} steps) | {hms(m['wall_s'])} | results/p9_reproduce_default |",
        f"| `python tasks.py reproduce --full` | + full-profile Phase 1 solver reruns | + {hms(full)} | "
        "results/p7_3_rerender_default |",
        f"| `python tasks.py reproduce --animations` | + 17 animations, gallery, showreel | + {hms(anim_s)} | "
        "results/p7_animate_default |",
        f"| `python tasks.py reproduce --retrain` | lists the from-scratch runs (training included); starts nothing |"
        f" {hms(sum(e['recorded_s'] for e in plan))} recorded in total | docs/figures/catalogue.json |",
        "",
        "## Steps of the last run",
        "",
        "| step | exit | runtime |",
        "|---|---|---|",
    ]
    L += [f"| {r['name']} | {r['exit']} | {hms(r['wall_s'])} |" for r in m["runs"]]
    L += [
        "",
        "## From a clean clone",
        "",
        "A clone has no git-ignored artefacts (network checkpoints, cached solver arrays, data/datasets). "
        "`python -m aerosynapse.thermal.experiments.p9_clean_clone` clones the repository, runs the CI commands and `reproduce` there, "
        "and classifies every failing step (results/p9_clean_clone_default). Steps that need the artefacts say so; "
        "regenerate them with the experiment scripts (retraining) or copy them from the original machine.",
        "",
    ]
    (ROOT / "docs" / "REPRODUCE.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
