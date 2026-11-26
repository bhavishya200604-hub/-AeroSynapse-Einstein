"""A design project on disk: requirements, stage folders, design revisions with checksums, and a log."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROJECTS = ROOT / "projects"

STAGES: list[tuple[str, str]] = [
    ("concept", "Basic design"),
    ("design", "Design optimisation"),
    ("wing", "Wing optimisation"),
    ("body", "Body optimisation"),
    ("structure", "Structure and design codes"),
    ("thermal", "Skin heating with the PINN"),
    ("simulation", "3D flow simulation"),
    ("export", "Export and documentation"),
]
KEYS = [k for k, _ in STAGES]

DEFAULT_REQUIREMENTS: dict[str, Any] = {
    "family": "conventional",  # fighter design-space family of the baseline (when no input body is given)
    "code": "mil_fighter",  # structural design code, see aerosynapse.structures.codes.CODES
    "W0_kg": 12000.0,  # take-off mass used when the design has no fighter variables
    "v_d_eas_ms": 400.0,  # design dive speed, equivalent airspeed
    "cruise": {"mach": 0.85, "alpha": 2.0, "altitude": 11000.0},
    "manoeuvre": {"mach": 0.6, "alpha": 10.0, "altitude": 6000.0},
    "supersonic": {"mach": 1.6, "alpha": 2.0, "altitude": 12000.0},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


class Project:
    def __init__(self, path: str | Path):
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = PROJECTS / str(path)
        self.dir = p.resolve()
        self.file = self.dir / "project.json"
        if not self.file.exists():
            raise FileNotFoundError(f"no project at {self.dir} (create one with: aerosynapse workflow new NAME)")
        self.meta: dict[str, Any] = json.loads(self.file.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ creation
    @classmethod
    def create(cls, name: str, source: str | Path | None = None, family: str | None = None, code: str = "mil_fighter", W0_kg: float | None = None,
               where: Path | None = None, overwrite: bool = False) -> Project:
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
            raise ValueError("a project name uses letters, digits, '-' and '_' only")
        d = (where or PROJECTS) / name
        if (d / "project.json").exists() and not overwrite:
            raise FileExistsError(f"project {name} exists at {d}")
        d.mkdir(parents=True, exist_ok=True)
        req = json.loads(json.dumps(DEFAULT_REQUIREMENTS))
        req["code"] = code
        origin = "family baseline"
        if source:
            design = json.loads(Path(source).read_text(encoding="utf-8"))
            if "features" not in design:
                raise ValueError(f"{source} is not a design (no 'features' list)")
            (d / "input").mkdir(exist_ok=True)
            (d / "input" / "design.json").write_text(json.dumps(design, indent=1), encoding="utf-8")
            origin = f"input body {Path(source).name}"
            if family is None and "variables" in design:
                family = "freeform"  # the family whose space contains every option
        req["family"] = family or req["family"]
        if W0_kg:
            req["W0_kg"] = float(W0_kg)
        meta = {"name": name, "created": now(), "origin": origin, "requirements": req, "stages": {k: {"status": "pending"} for k in KEYS}, "revisions": [],
                "software": "AeroSynapse design workflow"}
        (d / "project.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return cls(d)

    # ------------------------------------------------------------------ state
    @property
    def name(self) -> str:
        return str(self.meta["name"])

    def save(self) -> None:
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.meta, indent=1), encoding="utf-8")
        tmp.replace(self.file)

    def stage_dir(self, key: str) -> Path:
        d = self.dir / f"{KEYS.index(key) + 1:02d}_{key}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def log(self, msg: str) -> None:
        line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.dir / "workflow.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def set_stage(self, key: str, **info: Any) -> None:
        self.meta["stages"].setdefault(key, {}).update(info)
        self.save()

    def status(self, key: str) -> str:
        return str(self.meta["stages"].get(key, {}).get("status", "pending"))

    # ------------------------------------------------------------------ designs
    def design_before(self, key: str) -> tuple[dict[str, Any], str]:
        """The newest design produced by a completed stage before ``key`` (or the input / nothing)."""
        for k in reversed(KEYS[: KEYS.index(key)]):
            f = self.dir / f"{KEYS.index(k) + 1:02d}_{k}" / "design.json"
            if self.status(k) == "done" and f.exists():
                return json.loads(f.read_text(encoding="utf-8")), k
        inp = self.dir / "input" / "design.json"
        if inp.exists():
            return json.loads(inp.read_text(encoding="utf-8")), "input"
        raise RuntimeError("run the concept stage first")

    def latest_design(self) -> tuple[dict[str, Any], str]:
        return self.design_before("export") if self.status("export") != "done" else self.design_before("export")

    def save_design(self, key: str, design: dict[str, Any], note: str) -> Path:
        design = {**design, "name": design.get("name", self.name)}
        design["revision"] = len(self.meta["revisions"]) + 1
        design["stage"] = key
        f = self.stage_dir(key) / "design.json"
        f.write_text(json.dumps(design, indent=1), encoding="utf-8")
        self.meta["revisions"].append({"rev": design["revision"], "stage": key, "file": str(f.relative_to(self.dir)), "sha256": sha256(f), "time": now(), "note": note})
        self.save()
        return f

    def result(self, key: str) -> dict[str, Any]:
        f = self.dir / f"{KEYS.index(key) + 1:02d}_{key}" / "result.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}

    def write_result(self, key: str, res: dict[str, Any]) -> Path:
        """Strict JSON (NaN and infinity become null) so that every reader, Studio's included, can parse it."""
        f = self.stage_dir(key) / "result.json"
        f.write_text(json.dumps(_finite(json.loads(json.dumps(res, default=_jsonable))), indent=1, allow_nan=False), encoding="utf-8")
        return f


def _finite(o: Any) -> Any:
    if isinstance(o, float):
        return o if o == o and abs(o) != float("inf") else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


def _jsonable(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)
