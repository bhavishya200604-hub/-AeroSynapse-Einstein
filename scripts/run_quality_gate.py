"""Quality gate: full test suite with coverage, ruff, mypy -> results/quality_gate/metrics.json."""

from __future__ import annotations

import json
import subprocess
import sys

from _common import RESULTS, ROOT

from aerosynapse.experiments import ExperimentRecord, ExperimentRun


def sh(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main() -> None:
    rec = ExperimentRecord(
        "quality_gate",
        "Do tests, lint and type-check pass, and is coverage on target?",
        "all tests pass; ruff and mypy clean; core coverage >= 90 %, overall >= 80 %",
        "n/a (software quality)",
        0,
        reproducibility="python scripts/run_quality_gate.py",
        evaluation_metrics=["tests passed/failed", "coverage", "ruff", "mypy"],
    )
    with ExperimentRun(rec, RESULTS) as run:
        cov_json = RESULTS / "quality_gate" / "coverage.json"
        junit = RESULTS / "quality_gate" / "junit.xml"
        rc, out = sh(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "--cov=aerosynapse",
                f"--cov-report=json:{cov_json}",
                f"--junitxml={junit}",
            ]
        )
        import xml.etree.ElementTree as ET

        ts = ET.parse(junit).getroot()
        ts = ts if ts.tag == "testsuite" else ts.find("testsuite")
        assert ts is not None
        n_tests, n_fail = int(ts.get("tests", 0)), int(ts.get("failures", 0)) + int(ts.get("errors", 0))
        n_skip = int(ts.get("skipped", 0))
        cov = json.loads(cov_json.read_text())
        pk: dict[str, list[int]] = {}
        for fn, v in cov["files"].items():
            g = fn.split("aerosynapse/")[-1].split("/")[0] if "/" in fn.split("aerosynapse/")[-1] else "top-level"
            s = pk.setdefault(g, [0, 0])
            s[0] += v["summary"]["covered_lines"]
            s[1] += v["summary"]["num_statements"]
        rr, _ = sh(["ruff", "check", "src", "tests", "scripts"])
        rm, mo = sh([sys.executable, "-m", "mypy"])
        run.record.results.update(
            {
                "pytest_exit_code": rc,
                "tests_total": n_tests,
                "tests_passed": n_tests - n_fail - n_skip,
                "tests_failed": n_fail,
                "tests_skipped": n_skip,
                "coverage_total_percent": cov["totals"]["percent_covered"],
                "coverage_by_package_percent": {k: 100 * a / b for k, (a, b) in pk.items() if b},
                "ruff_clean": rr == 0,
                "mypy_clean": rm == 0,
                "mypy_summary": mo.strip().splitlines()[-1] if mo else "",
            }
        )
    print(json.dumps(run.record.results, indent=1))


if __name__ == "__main__":
    main()
