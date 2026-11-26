"""Rerun named Navier-Stokes cases from their saved job files and rebuild ns.json.

    python scripts/shapeopt_rerun_ns.py --shape naca2412 cylinder_Re100 airfoil_base_a8
"""

from __future__ import annotations

import argparse
import json

from _common import RESULTS
from _jobs import run_jobs


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--shape", default="naca2412")
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("cases", nargs="+")
    n = a.parse_args()
    out = RESULTS / "shapeopt" / n.shape.replace(":", "_")
    work = out / "jobs" / "ns_standard"
    jobs = []
    for c in n.cases:
        (work / f"{c}.result.json").unlink(missing_ok=True)
        jobs.append((c, "ns", json.loads((work / f"{c}.job.json").read_text(encoding="utf-8"))))
    run_jobs(jobs, work, max_workers=n.workers)
    done = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(work.glob("*.result.json"))]
    (out / "ns.json").write_text(json.dumps(done), encoding="utf-8")
    for r in done:
        print(f"{r['name']:28s} Cd {r['cd_mean']:.4f} Cl {r['cl_mean']:.4f} Cl' {r['cl_rms']:.4f} St {r['strouhal']:.3f}")


if __name__ == "__main__":
    main()
