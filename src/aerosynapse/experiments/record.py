"""Experiment record and run manager.

Every run writes ``results/<experiment_id>/`` containing ``record.json`` (all spec Section 13
fields), ``config.yaml``, ``metrics.json``, ``history.csv`` and optional checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aerosynapse._version import __version__
from aerosynapse.device import hardware_report
from aerosynapse.exceptions import ConfigurationError


def git_commit(cwd: Path | None = None) -> str:
    """Return the current git commit hash (``"unknown"`` outside a repository)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return out.stdout.strip() or "unknown"


def software_versions() -> dict[str, str | None]:
    """Versions of the key scientific packages."""
    hw = hardware_report()
    return {
        "aerosynapse": __version__,
        "python": hw.get("python"),
        "numpy": hw.get("numpy"),
        "scipy": hw.get("scipy"),
        "torch": hw.get("torch"),
    }


REQUIRED_FIELDS = (
    "experiment_id",
    "research_question",
    "hypothesis",
    "mathematical_formulation",
    "model_configuration",
    "solver_configuration",
    "dataset_description",
    "seed",
    "hardware",
    "training_configuration",
    "optimization_configuration",
    "evaluation_metrics",
    "results",
    "limitations",
    "reproducibility",
)


@dataclass
class ExperimentRecord:
    """All fields required by spec Section 13."""

    experiment_id: str
    research_question: str
    hypothesis: str
    mathematical_formulation: str
    seed: int
    model_configuration: dict[str, Any] = field(default_factory=dict)
    solver_configuration: dict[str, Any] = field(default_factory=dict)
    dataset_description: str = ""
    hardware: dict[str, Any] = field(default_factory=hardware_report)
    training_configuration: dict[str, Any] = field(default_factory=dict)
    optimization_configuration: dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    reproducibility: str = ""
    git_commit: str = field(default_factory=git_commit)
    software_versions: dict[str, str | None] = field(default_factory=software_versions)
    data_provenance: str = ""
    created_unix: float = field(default_factory=time.time)
    status: str = "created"

    def validate(self) -> None:
        """Raise ConfigurationError if a mandatory field is empty."""
        if not self.experiment_id or any(c in self.experiment_id for c in "/\\ "):
            raise ConfigurationError("experiment_id must be non-empty with no spaces or slashes")
        for name in ("research_question", "hypothesis", "mathematical_formulation"):
            if not getattr(self, name):
                raise ConfigurationError(f"experiment record field {name!r} must be set before running")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentRun:
    """Context manager handling the results directory, history, metrics and checkpoints.

    The hypothesis is written to disk *before* the run executes (``record.json`` with status
    ``running``) so it cannot be edited after the results are known.
    """

    def __init__(self, record: ExperimentRecord, root: str | Path = "results", config: Any = None) -> None:
        record.validate()
        self.record = record
        self.dir = Path(root) / record.experiment_id
        self.config = config
        self._history: list[dict[str, Any]] = []
        self._t0 = 0.0

    def __enter__(self) -> ExperimentRun:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.record.status = "running"
        self._write_record()
        if self.config is not None:
            cfg = self.config.model_dump() if hasattr(self.config, "model_dump") else self.config
            (self.dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        self._t0 = time.perf_counter()
        return self

    def log(self, **row: Any) -> None:
        """Append one row to the optimisation / training history."""
        self._history.append(row)

    def checkpoint_path(self, name: str = "checkpoint.pt") -> Path:
        return self.dir / name

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        wall = time.perf_counter() - self._t0
        self.record.results.setdefault("wall_clock_s", wall)
        self.record.status = "failed" if exc_type else "completed"
        if exc_type:
            self.record.results["error"] = f"{exc_type.__name__}: {exc}"
        self._write_record()
        (self.dir / "metrics.json").write_text(
            json.dumps(self.record.results, indent=2, default=_json_default), encoding="utf-8"
        )
        if self._history:
            keys = list(dict.fromkeys(k for row in self._history for k in row))
            with (self.dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(self._history)
        (self.dir / "git_hash.txt").write_text(self.record.git_commit + "\n", encoding="utf-8")

    def _write_record(self) -> None:
        (self.dir / "record.json").write_text(
            json.dumps(self.record.to_dict(), indent=2, default=_json_default), encoding="utf-8"
        )


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file (data provenance)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
