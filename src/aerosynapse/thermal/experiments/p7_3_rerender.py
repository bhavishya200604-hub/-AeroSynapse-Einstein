"""Phase 7 Step 7.3: re-render every figure of the catalogue (docs/figures/catalogue.json) in the project style, and
check that the regenerated numbers equal the committed ones.

    python -m aerosynapse.thermal.experiments.p7_3_rerender                 # everything (saved-results re-renders, then solver reruns)
    python -m aerosynapse.thermal.experiments.p7_3_rerender --skip-full     # without the full-profile Phase 1 reruns (about 3 h)
    python -m aerosynapse.thermal.experiments.p7_3_rerender --list          # the commands and why each is allowed

No network is trained (D135):
  - re-renderable figures use the scripts' --figures-only / --table-only / --analysis-only modes (saved results);
  - Phase 1 figures rerun the verification scripts (finite differences, spectral, finite volume, Navier-Stokes;
    numerical solvers only, deterministic);
  - Phase 2 figures are re-drawn read-only from the existing datasets (--figures-only, nothing written to data/);
  - the Phase 4 gate runs with --compile-only (its G3 benchmark would train networks).
Writes results/p7_3_rerender_<tag>/metrics.json (per command: exit code, wall time, figures written) and
docs/tables/phase7_3_rerender.md, including a numerical diff of every regenerated metrics.json against git HEAD.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash

CAT = ROOT / "docs" / "figures" / "catalogue.json"
FIG = ROOT / "docs" / "figures"
# keys whose values legitimately change between runs (time, provenance), not results
VOLATILE = (
    "wall",
    "time",
    "_s",
    "seconds",
    "git",
    "hash",
    "date",
    "_meta",
    "path",
    "dir",
    "file",
    "machine",
    "python",
    "numpy",
    "torch",
    "host",
    "cpu",
    "memory",
    "latency",
    "speed",
    "rate",
    "contention",
    "elapsed",
)


def commands() -> list[dict]:
    """One entry per distinct command of the catalogue, adapted so that none trains a network."""
    figs = json.loads(CAT.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for f in figs:
        cmd = f["rerender"]
        why = "saved results (re-render flag)" if f["status"].startswith("re-render") else "solver rerun (numerics)"
        if cmd.startswith("python -m aerosynapse.thermal.experiments.p2_data"):
            cmd = cmd if "--figures-only" in cmd else cmd + " --figures-only"
            why = "read-only re-draw from data/datasets (nothing written there)"
        if cmd.startswith("python -m aerosynapse.thermal.experiments.p5_1_identifiability"):
            # its figures need the Jacobians and Monte Carlo samples, which the run does not store: rerun the
            # identifiability analysis (seeded numerics: least-squares fits, no network)
            cmd, why = cmd.replace(" --table-only", ""), "numerics rerun (seeded identifiability analysis)"
        if f["name"] == "p4_5b_residuals_vs_corner":
            # drawn by the script's --residuals mode (inference on the saved networks), with the tags it was run on
            import yaml

            rc = yaml.safe_load((ROOT / "results/p4_5b_residuals/config.yaml").read_text(encoding="utf-8"))
            cmd = f"{cmd} --residuals --tags {' '.join(rc['tags'])} --n {rc['n']}"
            why = "evaluation of saved networks (residuals; no training)"
        if cmd.startswith("python -m aerosynapse.thermal.experiments.p4_gate"):
            cmd = cmd if "--compile-only" in cmd else cmd + " --compile-only"
            why = "compile of saved runs (the G3 benchmark would train: excluded)"
        e = out.setdefault(
            cmd,
            {"cmd": cmd, "why": why, "figures": [], "full": "--profile full" in cmd, "run_wall_s": f.get("run_wall_s")},
        )
        e["figures"].append(f["name"])
    # order: saved-results re-renders, then smoke reruns, then full-profile reruns
    return sorted(out.values(), key=lambda e: (e["full"], not e["why"].startswith("saved"), e["cmd"]))


def flatten(o, prefix="") -> dict:
    if isinstance(o, dict):
        r = {}
        for k, v in o.items():
            r.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return r
    if isinstance(o, list):
        r = {}
        for i, v in enumerate(o):
            r.update(flatten(v, f"{prefix}[{i}]"))
        return r
    return {prefix: o}


def metrics_diff(rel: str) -> dict:
    """Numerical comparison of a regenerated metrics.json with the committed one (volatile keys excluded)."""
    try:
        old = json.loads(
            subprocess.run(
                ["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True
            ).stdout
        )
    except subprocess.CalledProcessError:
        return {"file": rel, "status": "new (not in git)"}
    new = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    a, b = flatten(old), flatten(new)
    keys = [k for k in a.keys() & b.keys() if not any(v in k.lower() for v in VOLATILE)]
    worst, n_diff, examples = 0.0, 0, []
    for k in keys:
        x, y = a[k], b[k]
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, int | float)
            or not isinstance(y, int | float)
        ):
            if x != y and not (isinstance(x, str) and isinstance(y, str)):
                n_diff += 1
                examples.append((k, x, y))
            continue
        d = abs(x - y) / max(abs(x), abs(y), 1e-300)
        if d > 1e-9:
            n_diff += 1
            worst = max(worst, d)
            if len(examples) < 5:
                examples.append((k, x, y))
    return {
        "file": rel,
        "compared": len(keys),
        "differing": n_diff,
        "max_rel_diff": worst,
        "only_old": len(a.keys() - b.keys()),
        "only_new": len(b.keys() - a.keys()),
        "examples": [[k, str(x)[:60], str(y)[:60]] for k, x, y in examples[:5]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="substrings of commands to run")
    ap.add_argument("--timeout", type=float, default=4 * 3600)
    a = ap.parse_args()
    cmds = [c for c in commands() if not (a.skip_full and c["full"])]
    if a.only:
        cmds = [c for c in cmds if any(s in c["cmd"] for s in a.only)]
    if a.list:
        for c in cmds:
            print(f"{len(c['figures']):3d} figs | {c['why']:58s} | {c['cmd']}")
        return
    logs = ROOT / "results" / f"p7_3_rerender_{a.tag}" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    t_start = time.time()
    with Experiment(f"p7_3_rerender_{a.tag}", {"seed": 0, "skip_full": a.skip_full}) as ex:
        runs = []
        for i, c in enumerate(cmds):
            t0 = time.time()
            log = logs / f"{i:02d}.log"
            argv = [sys.executable] + c["cmd"].split()[1:]
            try:
                r = subprocess.run(
                    argv,
                    cwd=ROOT,
                    env=env,
                    stdout=open(log, "w", encoding="utf-8"),
                    stderr=subprocess.STDOUT,
                    timeout=a.timeout,
                )
                code = r.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
            wall = time.time() - t0
            fresh = [
                n for n in c["figures"] if (FIG / f"{n}.png").exists() and (FIG / f"{n}.png").stat().st_mtime >= t0
            ]
            runs.append(
                {
                    **c,
                    "exit": code,
                    "wall_s": round(wall, 1),
                    "log": str(log.relative_to(ROOT)),
                    "rerendered": fresh,
                    "missing": [n for n in c["figures"] if n not in fresh],
                }
            )
            print(
                f"[{i + 1}/{len(cmds)}] exit {code} {wall:7.1f} s {len(fresh)}/{len(c['figures'])} figs  {c['cmd']}",
                flush=True,
            )
            ex.metrics["runs"] = runs  # partial progress is kept if the run is interrupted
        changed = subprocess.run(
            ["git", "diff", "--name-only", "--", "results"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
        ).stdout.split()
        diffs = [metrics_diff(f) for f in changed if f.endswith("metrics.json")]
        figs = json.loads(CAT.read_text(encoding="utf-8"))
        fresh_all = [
            f["name"]
            for f in figs
            if (FIG / f"{f['name']}.png").exists() and (FIG / f"{f['name']}.png").stat().st_mtime >= t_start
        ]
        ex.metrics.update(
            {
                "runs": runs,
                "figures_total": len(figs),
                "figures_rerendered": len(fresh_all),
                "figures_not_rerendered": sorted(set(f["name"] for f in figs) - set(fresh_all)),
                "failed_commands": [r["cmd"] for r in runs if r["exit"] != 0],
                "metrics_diffs": diffs,
                "wall_s": round(time.time() - t_start, 1),
                "git": git_hash(),
            }
        )
    print(f"re-rendered {len(fresh_all)}/{len(figs)} figures; failed commands {ex.metrics['failed_commands']}")


if __name__ == "__main__":
    main()
