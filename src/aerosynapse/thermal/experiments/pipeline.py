"""The automated end-to-end run of the project (D149): one command from data to a deployed Space.

    python tasks.py pipeline                         # quick profile (minutes): every stage at a small budget
    python tasks.py pipeline --full                  # full profile (hours): the recorded studies at full budget
    python tasks.py pipeline --deploy USER/SPACE     # ... and upload the Space at the end (needs `hf auth login`)
    python tasks.py pipeline --only fetch tests      # chosen stages; --list shows them

Stages run in order and stop at the first failure (--keep-going runs the rest). Every stage's command, exit code, wall
time and log tail are recorded in results/pipeline_<profile>/metrics.json and docs/tables/pipeline_<profile>.md.
The stages call the project's own scripts; nothing is re-implemented here.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash

PY = sys.executable


def stages(full: bool, deploy: str | None) -> list[tuple[str, str, list[str]]]:
    """(name, what, command) in order."""
    q = [] if full else ["--quick"]
    out = [
        ("fetch", "NOAA soil data (checked against the recorded SHA-256)", [PY, "-m", "aerosynapse.thermal.experiments.p10_fetch_real"]),
        (
            "verify",
            "verification of the solvers (Phase 1 scripts, MMS / convergence order)",
            [PY, "tasks.py", "verify1"] + (["--full"] if full else []),
        ),
        (
            "real_data",
            "real-data identification study (conduction model, convergence pipeline, hold-out)",
            [PY, "-m", "aerosynapse.thermal.experiments.p10_1_real_soil"] + (["--tag", "pipeline"] if not full else []) + q,
        ),
        (
            "real_pinn",
            "PINN trained on real soil temperatures vs the classical estimator",
            [PY, "-m", "aerosynapse.thermal.experiments.p10_5_real_pinn"] + q,
        ),
        ("lint", "ruff", [PY, "-m", "ruff", "check", "src", "tests", "experiments", "tasks.py", "deploy"]),
        ("tests", "the test suite (slow tests deselected)", [PY, "-m", "pytest", "-p", "no:cacheprovider"]),
        (
            "space",
            "the public Space bundle run end to end (NOAA fetch, server in public mode, smoke checks)",
            [PY, "-m", "aerosynapse.thermal.experiments.p10_3_space", "--tag", "pipeline"],
        ),
    ]
    if deploy:
        out.append(
            (
                "deploy",
                f"upload the Space to huggingface.co/spaces/{deploy}",
                [PY, "deploy/hf_space/build.py", "--push", deploy],
            )
        )
    return out


def run(selected: list[tuple[str, str, list[str]]], keep_going: bool, logdir) -> list[dict]:
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    rows = []
    for name, what, cmd in selected:
        t0 = time.time()
        print(f"[{name}] {what}\n  + {' '.join(cmd)}", flush=True)
        log = logdir / f"{len(rows):02d}_{name}.log"
        with open(log, "w", encoding="utf-8") as fh:
            code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        text = log.read_text(encoding="utf-8", errors="replace").splitlines()
        rows.append(
            {
                "stage": name,
                "what": what,
                "command": " ".join(cmd[1:]),
                "exit": code,
                "wall_s": round(time.time() - t0, 1),
                "log": str(log.relative_to(ROOT)),
                "tail": text[-6:],
            }
        )
        print(f"  -> exit {code} in {rows[-1]['wall_s']:.0f} s", flush=True)
        if code != 0 and not keep_going:
            print(f"stopped at {name}; log: {log}", flush=True)
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--deploy", metavar="USER/SPACE", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    all_stages = stages(a.full, a.deploy)
    if a.list:
        for name, what, cmd in all_stages:
            print(f"{name:10s} {what}\n           {' '.join(cmd[1:])}")
        return
    selected = [s for s in all_stages if not a.only or s[0] in a.only]
    profile = "full" if a.full else "quick"
    with Experiment(f"pipeline_{profile}", {"full": a.full, "deploy": a.deploy, "only": a.only}) as ex:
        logdir = ex.dir / "logs"
        logdir.mkdir(exist_ok=True)
        rows = run(selected, a.keep_going, logdir)
        ok = len(rows) == len(selected) and all(r["exit"] == 0 for r in rows)
        ex.metrics.update({"stages": rows, "pass": ok, "git": git_hash()})
    lines = [
        f"# Automated pipeline ({profile} profile)",
        "",
        f"`python tasks.py pipeline{' --full' if a.full else ''}` (git {ex.metrics['git']}): "
        f"{'PASS' if ok else 'FAIL'}.",
        "",
        "| stage | what | exit | wall [s] |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r['stage']} | {r['what']} | {r['exit']} | {r['wall_s']:.0f} |" for r in rows]
    (ROOT / "docs" / "tables" / f"pipeline_{profile}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"pipeline {profile}: {'PASS' if ok else 'FAIL'} ({sum(r['exit'] == 0 for r in rows)}/{len(selected)} stages)"
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
