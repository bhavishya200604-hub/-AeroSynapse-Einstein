"""Phase 9 Step 9.2: test-suite coverage of the physics and numerics code (target >= 80 %).

    python -m aerosynapse.thermal.experiments.p9_coverage [--tag default] [--from-json cache/coverage.json]

Runs the fast test suite (pytest without the slow tests, as CI does) under coverage for src/physics and src/numerics,
or reads the JSON report of such a run (--from-json). Writes results/p9_coverage_<tag>/metrics.json: line coverage per
package and per file, the test outcome line, and whether the target is met.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash

PACKAGES = ("src/physics", "src/numerics")
TARGET = 80.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--from-json", default=None)
    ap.add_argument("--log", default=None, help="with --from-json: the pytest log of that run (outcome line)")
    a = ap.parse_args()
    summary = ""
    if a.log:  # the outcome line of the run that wrote the JSON report
        summary = next(
            (x for x in reversed((ROOT / a.log).read_text(encoding="utf-8").splitlines()) if " passed" in x), ""
        )
    report = ROOT / (a.from_json or "cache/coverage_p9.json")
    if not a.from_json:
        cmd = [
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
            *[f"--cov={p}" for p in PACKAGES],
            f"--cov-report=json:{report}",
        ]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        summary = (r.stdout.strip().splitlines() or [""])[-1]
        if r.returncode != 0:
            raise SystemExit(f"tests failed: {summary}")
    cov = json.loads(report.read_text(encoding="utf-8"))
    per_pkg = {p: {"statements": 0, "covered": 0} for p in PACKAGES}
    files = {}
    for f, d in cov["files"].items():
        rel = f.replace("\\", "/")
        pkg = next((p for p in PACKAGES if rel.startswith(p)), None)
        if not pkg:
            continue
        s = d["summary"]
        per_pkg[pkg]["statements"] += s["num_statements"]
        per_pkg[pkg]["covered"] += s["covered_lines"]
        files[rel] = round(s["percent_covered"], 1)
    for p in per_pkg.values():
        p["percent"] = round(100 * p["covered"] / max(p["statements"], 1), 1)
    st = sum(p["statements"] for p in per_pkg.values())
    cv = sum(p["covered"] for p in per_pkg.values())
    total = round(100 * cv / max(st, 1), 1)
    with Experiment(f"p9_coverage_{a.tag}", {"seed": 0, "packages": list(PACKAGES), "target": TARGET}) as ex:
        ex.metrics.update(
            {
                "packages": per_pkg,
                "total_percent": total,
                "files": files,
                "tests": summary,
                "below_target_files": {k: v for k, v in files.items() if v < TARGET},
                "pass": total >= TARGET and all(p["percent"] >= TARGET for p in per_pkg.values()),
                "git": git_hash(),
            }
        )
    print(json.dumps({k: ex.metrics[k] for k in ("packages", "total_percent", "tests", "pass")}, indent=1))


if __name__ == "__main__":
    main()
