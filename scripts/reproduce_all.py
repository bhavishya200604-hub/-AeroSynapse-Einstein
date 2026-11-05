"""Reproduce every reported result: run all phase studies, then figures, the paper tables and the claims audit.

python scripts/reproduce_all.py --profile quick      (measured on the build machine: see docs/REPRODUCIBILITY.md)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    "run_phase1.py",
    "run_phase2.py",
    "run_phase3.py",
    "postprocess_phase3.py",
    "run_phase3_extras.py",
    "run_phase4.py",
    "run_phase4_genmap.py",
    "run_phase5.py",
    "run_phase6.py",
    "run_phase7.py",
    "make_figures.py",
    "make_report.py",
    "make_reproducibility.py",
    "make_architecture.py",
    "run_quality_gate.py",
    "audit_claims.py",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="quick", choices=["quick", "standard", "full"])
    ap.add_argument("--only", nargs="*", default=None, help="subset of step script names")
    a = ap.parse_args()
    times = {}
    for s in a.only or STEPS:
        t0 = time.perf_counter()
        args = [sys.executable, str(ROOT / "scripts" / s)] + (
            ["--profile", a.profile] if s.startswith("run_phase") else []
        )
        print(f"==> {s}", flush=True)
        rc = subprocess.call(args, cwd=ROOT)
        times[s] = time.perf_counter() - t0
        if rc != 0:
            print(f"FAILED: {s} (exit {rc})")
            return rc
    for k, v in times.items():
        print(f"{k:22s} {v:8.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
