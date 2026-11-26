"""Phase 9: record GitHub Actions CI runs (needs the GitHub CLI `gh`, logged in).

    python -m aerosynapse.thermal.experiments.p9_github_ci [--runs 39121670729 39122532924] [--tag default]

Per run: commit, conclusion, duration, and the pytest / ruff summary lines and failed test ids read from its log.
Writes results/p9_github_ci_<tag>/metrics.json. Without --runs, the latest 2 runs on main are recorded.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess

from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout


def record(run_id: str) -> dict:
    info = json.loads(
        gh("run", "view", run_id, "--json", "headSha,conclusion,status,createdAt,updatedAt,displayTitle,url")
    )
    log = gh("run", "view", run_id, "--log")
    lines = [re.sub(r"^.*?Z ", "", x) for x in log.splitlines()]
    summary = [x.strip() for x in lines if re.search(r"\d+ (passed|failed)", x) and " in " in x]
    failed = sorted({m.group(1) for x in lines if (m := re.match(r"FAILED (\S+)", x.strip()))})
    return {
        "run_id": run_id,
        "commit": info["headSha"][:7],
        "title": info["displayTitle"][:120],
        "conclusion": info["conclusion"],
        "created": info["createdAt"],
        "updated": info["updatedAt"],
        "pytest_summary": summary[-1] if summary else None,
        "ruff_clean": any("All checks passed!" in x for x in lines),
        "failed_tests": failed,
        "url": info["url"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--tag", default="default")
    a = ap.parse_args()
    runs = (
        a.runs or [r["databaseId"] for r in json.loads(gh("run", "list", "--limit", "2", "--json", "databaseId"))][::-1]
    )
    with Experiment(f"p9_github_ci_{a.tag}", {"seed": 0, "runs": [str(r) for r in runs]}) as ex:
        ex.metrics["runs"] = [record(str(r)) for r in runs]
        ex.metrics["git"] = git_hash()
    for r in ex.metrics["runs"]:
        print(r["run_id"], r["commit"], r["conclusion"], r["pytest_summary"], len(r["failed_tests"]), "failed tests")


if __name__ == "__main__":
    main()
