"""Run the long checks one after another once the fighter study has finished (one CPU-heavy job at a time).

    python scripts/pipeline_after_study.py

Steps, each logged to results/pipeline/<step>.log and skipped if its marker exists: CFD verification, 3D flow
of the reference and the best design, skin heating with the PINN, surrogate from the data store.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "pipeline"
STEPS = [
    ("cfd_verify", ["cfd_verify.py", "--profile", "standard"]),
    ("thermal_skin", ["run_thermal_skin.py", "--profile", "standard"]),
    ("surrogate", ["train_surrogate.py", "--profile", "standard"]),
    ("fighter_cfd", ["fighter_cfd.py", "--profile", "standard"]),
]


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    study = ROOT / "results" / "fighter" / "study.json"
    while not study.exists():
        time.sleep(60)
    for name, cmd in STEPS:
        done = LOG / f"{name}.done"
        if done.exists():
            continue
        t0 = time.time()
        with open(LOG / f"{name}.log", "w", encoding="utf-8") as fh:
            code = subprocess.call([sys.executable, *cmd], cwd=ROOT / "scripts", stdout=fh, stderr=subprocess.STDOUT)
        (LOG / "status.txt").open("a", encoding="utf-8").write(f"{name}: exit {code} after {time.time() - t0:.0f} s\n")
        if code == 0:
            done.write_text("ok", encoding="utf-8")


if __name__ == "__main__":
    main()
