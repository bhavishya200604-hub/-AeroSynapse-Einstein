"""Run every Phase 5 step on a user configuration (default: the cylinder example) — a demonstration and end-to-end
check that your data and body shape work in all steps. Reduced study sizes where a step is synthetic by nature.

    python experiments/run_user_demo.py [--config ...] [--tag cylinder_demo] [--after PID]

Each step writes results/<step>_<tag>/ (metrics.json, a markdown table, figures); the log of each run is
results/user_demo_<tag>/<step>.log, and results/user_demo_<tag>/summary.json lists exit codes and times.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from aerosynapse.thermal.utils.experiment import ROOT

STEPS = [
    ("p5_1_identifiability", []),
    ("p5_3_classical", []),
    ("p5_4_robustness", ["--set", "studies.robustness.replicates=50"]),
    ("p5_5_uq", ["--set", "studies.uq.datasets=20"]),
    ("p5_6_confounding", ["--set", "studies.confounding.replicates=30"]),
    ("p5_2_pinn_inverse", []),
]


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--config",
        action="append",
        default=None,
        help="config files (default: user_case.yaml + examples/user_cylinder.yaml)",
    )
    ap.add_argument("--tag", default="cylinder_demo")
    ap.add_argument("--after", type=int, default=None, help="wait for this process to finish first")
    ap.add_argument(
        "--steps", default=None, help="comma-separated subset of the steps (e.g. p5_5_uq,p5_2_pinn_inverse)"
    )
    a = ap.parse_args()
    cfgs = a.config or ["configs/inverse/user_case.yaml", "configs/inverse/examples/user_cylinder.yaml"]
    if a.after:
        while _alive(a.after):
            time.sleep(30)
    out = ROOT / "results" / f"user_demo_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    old = out / "summary.json"
    summary = json.loads(old.read_text(encoding="utf-8")) if a.steps and old.exists() else []
    chosen = set(a.steps.split(",")) if a.steps else None
    for step, extra in STEPS:
        if chosen is not None and step not in chosen:
            continue
        summary = [x for x in summary if x["step"] != step]
        cmd = (
            [sys.executable, "-m", f"aerosynapse.thermal.experiments.{step}"]
            + [x for c in cfgs for x in ("--config", c)]
            + ["--tag", a.tag]
            + extra
        )
        t0 = time.perf_counter()
        with open(out / f"{step}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.run(
                cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"}
            ).returncode
        summary.append(
            {"step": step, "command": " ".join(cmd[1:]), "exit_code": rc, "seconds": round(time.perf_counter() - t0, 1)}
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
