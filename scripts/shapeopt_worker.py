"""One shape-optimisation job in its own OS process: reads a job file, writes a result file.

Long jobs run this way instead of through ProcessPoolExecutor. On Windows behind a venv launcher
the pool failed twice in three launches (a WinError 6 at start-up, then a manager thread that
spun forever and never returned finished results). A separate process per job has no shared
handles, and a result on disk survives whatever happens to the parent.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from _common import ROOT  # noqa: E402,F401  (puts src on the path)


def main() -> None:
    kind, job_path, out_path = sys.argv[1:4]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    if kind == "cma":
        from aerosynapse.shapeopt.search import run_cma

        res = run_cma(job)
    elif kind == "de":
        from aerosynapse.shapeopt.search import run_de

        res = run_de(job)
    elif kind == "fighter":
        from aerosynapse.fighter.optimise import run as run_fighter

        res = run_fighter(job)
    elif kind == "aircraft":
        from aerosynapse.aircraft.optimise import run

        res = run(job)
    elif kind == "ns":
        import numpy as np

        from aerosynapse.shapeopt.nsverify import NSCase, run_case

        job["polys"] = [np.array(p) for p in job["polys"]]
        job["pivot"] = tuple(job.get("pivot", (0.25, 0.0)))
        res = run_case(NSCase(**job))
    else:
        raise SystemExit(f"unknown job kind {kind}")
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f)
    os.replace(tmp, out_path)  # a result file is either complete or absent


if __name__ == "__main__":
    main()
