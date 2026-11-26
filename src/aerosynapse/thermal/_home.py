"""Home folder of the thermal PINN project: its configs, data, results, docs and caches.

Set AEROSYNAPSE_THERMAL_HOME to move it, for example to a larger drive.
"""

from __future__ import annotations

import os
from pathlib import Path

HOME = Path(os.environ.get("AEROSYNAPSE_THERMAL_HOME") or Path(__file__).resolve().parents[3] / "projects" / "thermal_pinn")
CODE = Path(__file__).resolve().parent  # the package itself (src/ in the original repository)
THERMAL_TESTS = Path(__file__).resolve().parents[3] / "tests" / "thermal"  # tests/unit in the original repository


def resolve(path: str | Path) -> Path:
    """A path the original project wrote relative to its root: try it as given, then under HOME."""
    p = Path(path)
    return p if p.exists() or p.is_absolute() else HOME / p
