"""Phase 9 gate: does `reproduce` work from a clean clone?

    python -m aerosynapse.thermal.experiments.p9_clean_clone [--dest E:/PINNs_clean_clone_check] [--skip-tests] [--tag default]

Clones the committed repository (git clone of HEAD, no untracked or git-ignored files) into --dest and runs there,
with this environment's Python and packages:
  1. the CI commands: ruff check, and pytest without the slow tests;
  2. python tasks.py reproduce (level 'saved').
A clone has none of the git-ignored artefacts (network checkpoints, cached solver arrays, data/datasets), so the
steps that need them must fail with a missing-file error; every failure is classified from its log as
'needs git-ignored artefact' or 'defect'. The gate passes when CI is green and no step fails for another reason.
Writes results/p9_clean_clone_<tag>/metrics.json (in THIS repository).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash

MARK = ".clean_clone_check"  # only folders carrying this marker are ever deleted
MISSING = re.compile(
    r"(FileNotFoundError|No such file|does not exist|has no metrics\.json|no datasets in|checkpoint\.pt|"
    r"\.npz|data[\\/]datasets|not found: run|run the experiment first|missing)",
    re.I,
)


def run(cmd: list[str], cwd: Path, env: dict, log: Path, timeout: float) -> tuple[int | str, float]:
    t0 = time.time()
    try:
        with open(log, "w", encoding="utf-8") as fh:
            r = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
        code: int | str = r.returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    return code, round(time.time() - t0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(ROOT.parent / "PINNs_clean_clone_check"))
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--timeout", type=float, default=6 * 3600)
    a = ap.parse_args()
    dest = Path(a.dest)
    if dest.exists():
        if not (dest / MARK).exists():
            raise SystemExit(f"{dest} exists and is not a clean-clone check folder: choose another --dest")
        shutil.rmtree(dest, onerror=lambda f, p, e: (os.chmod(p, 0o700), f(p)))
    head = git_hash()
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(ROOT), str(dest)], check=True)
    (dest / MARK).write_text("created by experiments/p9_clean_clone.py; safe to delete\n", encoding="utf-8")
    logs = ROOT / "results" / f"p9_clean_clone_{a.tag}" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(dest), "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    with Experiment(f"p9_clean_clone_{a.tag}", {"seed": 0, "dest": str(dest), "head": head}) as ex:
        steps = {}
        steps["ruff"] = run(
            [sys.executable, "-m", "ruff", "check", "src", "tests", "experiments", "tasks.py"],
            dest,
            env,
            logs / "ruff.log",
            600,
        )
        if not a.skip_tests:
            steps["pytest"] = run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    "-W",
                    "ignore",
                    "-o",
                    "addopts=-m 'not slow'",
                ],
                dest,
                env,
                logs / "pytest.log",
                3 * 3600,
            )
        steps["reproduce"] = run(
            [sys.executable, "tasks.py", "reproduce", "--tag", "clone"], dest, env, logs / "reproduce.log", a.timeout
        )
        tail = (logs / "pytest.log").read_text(encoding="utf-8").strip().splitlines()[-1:] if not a.skip_tests else []
        # classify every failed step of the re-render inside the clone from its own log
        rr = dest / "results" / "p7_3_rerender_reproduce_clone" / "metrics.json"
        classified = []
        if rr.exists():
            for r in json.loads(rr.read_text(encoding="utf-8")).get("runs", []):
                if r["exit"] == 0:
                    continue
                text = (dest / r["log"]).read_text(encoding="utf-8", errors="replace")
                hit = MISSING.search(text)
                last = [x for x in text.strip().splitlines() if x.strip()][-1:] or [""]
                classified.append(
                    {
                        "cmd": r["cmd"],
                        "exit": r["exit"],
                        "kind": "needs git-ignored artefact" if hit else "defect",
                        "evidence": (hit.group(0) if hit else "") + " | " + last[0][:200],
                    }
                )
        rp = dest / "results" / "p9_reproduce_clone" / "metrics.json"
        other = []
        if rp.exists():
            other = [
                r
                for r in json.loads(rp.read_text(encoding="utf-8")).get("runs", [])
                if r["exit"] != 0 and not r["name"].startswith("tables and figures")
            ]
        ci_ok = steps["ruff"][0] == 0 and (a.skip_tests or steps["pytest"][0] == 0)
        defects = [c for c in classified if c["kind"] == "defect"] + other
        ex.metrics.update(
            {
                "head": head,
                "dest": str(dest),
                "steps": {k: {"exit": v[0], "wall_s": v[1]} for k, v in steps.items()},
                "pytest_summary": tail,
                "rerender_failures": classified,
                "other_failed_steps": [{"name": r["name"], "tail": r.get("tail")} for r in other],
                "ci_green": ci_ok,
                "defects": len(defects),
                "pass": ci_ok and not defects and rr.exists(),
                "git": git_hash(),
            }
        )
    print(json.dumps({k: ex.metrics[k] for k in ("steps", "pytest_summary", "ci_green", "defects", "pass")}, indent=1))
    for c in classified:
        print(f"  {c['kind']:28s} {c['cmd']}  [{c['evidence'][:90]}]")


if __name__ == "__main__":
    main()
