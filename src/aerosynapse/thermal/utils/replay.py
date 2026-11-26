"""Phase 7 Step 7.3: the inputs of a script's figures, rebuilt from what its run saved (metrics.json, config.yaml and
each training run's log.csv), so figures can be re-drawn in the project style without training anything."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
def run_dir(exp_id: str) -> Path:
    d = ROOT / "results" / exp_id
    if not (d / "metrics.json").exists():
        raise SystemExit(f"{d} has no metrics.json: run the experiment first")
    return d


def metrics(exp_id: str) -> dict:
    return json.loads((run_dir(exp_id) / "metrics.json").read_text(encoding="utf-8"))


def config(exp_id: str) -> dict:
    return yaml.safe_load((run_dir(exp_id) / "config.yaml").read_text(encoding="utf-8"))


def history(log: Path) -> list[dict]:
    """A training history as the trainer kept it in memory: one dict per logged row, numbers as floats, the phase as
    text. An empty cell means the trainer did not record that key in that row (e.g. evaluation columns between
    evaluations), so it is left out of the row, as in memory."""
    with open(log, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            if k == "phase":
                d[k] = v
            elif v in ("", None):
                continue
            else:
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
        out.append(d)
    return out


def target(exp_id: str) -> SimpleNamespace:
    """Stand-in for the Experiment object: figures only use its .dir."""
    return SimpleNamespace(dir=run_dir(exp_id))
