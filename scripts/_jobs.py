"""Run independent jobs as separate processes, a few at a time, skipping any already finished."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def run_jobs(jobs: list[tuple[str, str, dict[str, Any]]], work: Path, max_workers: int = 4) -> list[dict[str, Any]]:
    """``jobs`` holds (name, kind, payload). Returns the results in order; raises if any job failed.

    Each job leaves ``work/<name>.result.json``. Rerunning skips those, so a crash costs only the
    jobs that were still running.
    """
    work.mkdir(parents=True, exist_ok=True)
    todo = []
    for name, kind, payload in jobs:
        out = work / f"{name}.result.json"
        if out.exists():
            print(f"[jobs] {name}: already finished, reusing", flush=True)
            continue
        jf = work / f"{name}.job.json"
        jf.write_text(json.dumps(payload), encoding="utf-8")
        todo.append((name, [sys.executable, str(HERE / "shapeopt_worker.py"), kind, str(jf), str(out)]))
    running: list[tuple[str, subprocess.Popen[bytes]]] = []
    failed: list[str] = []
    while todo or running:
        while todo and len(running) < max_workers:
            name, cmd = todo.pop(0)
            running.append((name, subprocess.Popen(cmd, cwd=HERE)))
        time.sleep(2.0)
        for item in list(running):
            code = item[1].poll()
            if code is not None:
                running.remove(item)
                print(f"[jobs] {item[0]}: exit {code}", flush=True)
                if code:
                    failed.append(item[0])
    if failed:
        raise SystemExit(f"jobs failed: {failed}. Rerun to retry only these.")
    return [json.loads((work / f"{n}.result.json").read_text(encoding="utf-8")) for n, _, _ in jobs]
