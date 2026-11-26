"""Data slots: what the software knows, where it came from, and what is still missing.

Each slot has a schema. The built-in reference values are public, and their quality is stated.
Slots marked empty are places to upload data in a later phase: wind tunnel polars, CFD results,
engine decks, weights and material allowables. Uploads are checked against the schema, kept in
``data_uploads/`` and then used to calibrate the models.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

UPLOADS = Path("data_uploads")


@dataclass(frozen=True)
class Slot:
    key: str
    title: str
    columns: tuple[str, ...]
    purpose: str


SLOTS = (
    Slot("polar", "Measured or CFD drag polar", ("mach", "cl", "cd"), "Calibrates the drag model: scales computed drag to match these points."),
    Slot("cfd_surface", "CFD surface results", ("x", "y", "z", "cp"), "Compares surface pressure with the 3D panel method."),
    Slot("engine_deck", "Engine deck", ("altitude_m", "mach", "thrust_N", "tsfc_kg_Ns"), "Replaces the cycle model with real engine data."),
    Slot("weights", "Weight statement", ("item", "mass_kg"), "Sets the empty-mass calibration."),
    Slot("materials", "Material allowables", ("name", "rho", "E", "sigma", "alpha", "T_max"), "Replaces round-number material properties."),
    Slot("flight_test", "Flight or tunnel temperatures", ("mach", "altitude_m", "location_x", "T_K"), "Checks the skin temperature model."),
)


def _path(key: str) -> Path:
    return UPLOADS / f"{key}.csv"


def parse(key: str, text: str) -> list[dict[str, Any]]:
    """Validate an upload. Raises ValueError with a message that says how to fix the file."""
    slot = next((s for s in SLOTS if s.key == key), None)
    if slot is None:
        raise ValueError(f"unknown data slot {key!r}")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError("the file has no data rows")
    missing = [c for c in slot.columns if c not in rows[0]]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}. Expected: {', '.join(slot.columns)}")
    out = []
    for i, r in enumerate(rows, start=2):
        rec: dict[str, Any] = {}
        for c in slot.columns:
            v = (r[c] or "").strip()
            try:
                rec[c] = float(v)
            except ValueError:
                if c in ("item", "name"):
                    rec[c] = v
                else:
                    raise ValueError(f"line {i}: column {c} must be a number, found {v!r}") from None
        out.append(rec)
    return out


def store(key: str, text: str, root: Path | None = None) -> int:
    rows = parse(key, text)
    p = (root or Path(".")) / _path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return len(rows)


def load(key: str, root: Path | None = None) -> list[dict[str, Any]] | None:
    p = (root or Path(".")) / _path(key)
    return parse(key, p.read_text(encoding="utf-8")) if p.exists() else None


def status(root: Path | None = None) -> list[dict[str, Any]]:
    out = []
    for s in SLOTS:
        rows = load(s.key, root)
        out.append({"slot": s.title, "columns": ", ".join(s.columns), "rows": len(rows) if rows else 0, "state": "loaded" if rows else "empty, upload later", "purpose": s.purpose})
    return out


def drag_scale(polar: list[dict[str, Any]], predict: Callable[[float, float], float]) -> float:
    """Least-squares factor that makes the model match an uploaded polar."""
    cd = np.array([r["cd"] for r in polar])
    pr = np.array([predict(r["mach"], r["cl"]) for r in polar])
    return float((cd @ pr) / max(float(pr @ pr), 1e-12))


REFERENCE_PATH = Path(__file__).resolve().parents[1] / "data" / "aircraft_reference.json"


def reference() -> dict[str, Any]:
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
