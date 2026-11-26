"""Experiment bookkeeping: results/<exp_id>/{config.yaml, metrics.json, log.csv, git_hash.txt}.

Every number quoted in docs must come from a metrics.json written through :class:`Experiment`.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import shutil
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
RESULTS_DIR = ROOT / "results"


def set_seed(seed: int) -> None:
    """Seed python, numpy (and torch if importable) for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is an optional dependency until Phase 3
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _git_exe() -> str:
    exe = shutil.which("git")
    if exe:
        return exe
    for cand in (r"C:\Program Files\Git\cmd\git.exe", "/usr/bin/git"):
        if Path(cand).exists():
            return cand
    return "git"


def git_hash() -> str:
    """Return the current commit hash (with '-dirty' if tracked files changed) or 'nogit'."""
    try:
        git = _git_exe()
        h = subprocess.run(
            [git, "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            [git, "status", "--porcelain", "--untracked-files=no", "--", "src", "experiments"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def machine_name() -> str:
    """Processor description without WMI: on Windows, Python 3.12's platform.processor() / machine() query WMI, which
    can time out under load (WinError 258, seen at the end of the Step 5.3 smoke run) and must not cost a run its
    metrics."""
    env = os.environ.get("PROCESSOR_IDENTIFIER") or os.environ.get("PROCESSOR_ARCHITECTURE")
    if env:
        return env
    try:
        return platform.processor() or platform.machine()
    except Exception:  # noqa: BLE001
        return "unknown"


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating | np.integer | np.bool_):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


def cli_profile(default: str = "smoke") -> str:
    """Parse ``--profile smoke|full`` from the command line of an experiment script."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["smoke", "full"], default=default)
    return str(ap.parse_args().profile)


def load_config(path: str | Path, profile: str | None = None) -> dict[str, Any]:
    """Load a YAML config; if it has `profiles: {smoke: ..., full: ...}` merge the chosen one."""
    with open(path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}
    profiles = cfg.pop("profiles", None)
    if profiles is not None:
        prof = profile or cfg.get("profile", "smoke")
        cfg.update(profiles[prof])
        cfg["profile"] = prof
    return cfg


class Experiment:
    """Context object that owns one results folder.

    Usage::

        with Experiment("p1_fd1d_space", cfg) as ex:
            ex.log(N=32, err=1e-3)
            ex.metrics["order"] = 2.0
    """

    def __init__(self, exp_id: str, config: dict[str, Any], root: Path | None = None):
        self.exp_id = exp_id
        self.config = config
        self.dir = (root or RESULTS_DIR) / exp_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metrics: dict[str, Any] = {}
        self._rows: list[dict[str, Any]] = []
        self._t0 = 0.0

    def __enter__(self) -> Experiment:
        self._t0 = time.perf_counter()
        set_seed(int(self.config.get("seed", 0)))
        with open(self.dir / "config.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(_jsonable(self.config), fh, sort_keys=False)
        (self.dir / "git_hash.txt").write_text(git_hash() + "\n", encoding="utf-8")
        return self

    def log(self, **row: Any) -> None:
        self._rows.append(_jsonable(row))

    def fig_path(self, name: str) -> Path:
        return self.dir / name

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.metrics["_meta"] = {
            "exp_id": self.exp_id,
            "wall_time_s": round(time.perf_counter() - self._t0, 3),
            "git_hash": git_hash(),
            "python": platform.python_version(),
            "machine": machine_name(),
            "numpy": np.__version__,
            "status": "error" if exc_type else "ok",
        }
        with open(self.dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(_jsonable(self.metrics), fh, indent=2)
        if self._rows:
            keys: list[str] = []
            for r in self._rows:
                keys += [k for k in r if k not in keys]
            with open(self.dir / "log.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(self._rows)
