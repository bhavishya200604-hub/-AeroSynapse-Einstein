"""Set up, run and read ``aeroflow`` cases.

    res = run(design, Flight(mach=0.9, alpha=4, altitude=6000), out="results/cfd/x", cells_per_length=48)
    res["coefficients"]["CD"], surface(res["dir"]), plane(res["dir"], "plane_symmetry")

A case is plain JSON, so a run can be repeated with ``bin/aeroflow case.json``. Each run keeps its case,
history, result, surface data (Cp, friction, wall temperature, separation) and field planes.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerosynapse.cad import kernel


@dataclass
class Flight:
    mach: float = 0.8
    alpha: float = 2.0  # deg
    beta: float = 0.0  # sideslip, deg
    altitude: float = 9000.0  # m
    delta_T: float = 0.0  # K, hot or cold day
    gust_amplitude: float = 0.0  # vertical gust speed / flight speed
    gust_wavelength: float = 30.0  # m
    turbulence_intensity: float = 0.0  # u_rms / flight speed at the inflow
    turbulence_length: float = 3.0  # m


@dataclass
class Numerics:
    cells_per_length: float = 48
    stretch: float = 1.15
    max_cells: float = 1.6e6
    cfl: float = 1.2
    steps: int = 3000
    unsteady: bool = False
    average_from: int = -1  # step from which forces and fields are averaged (unsteady runs)
    sgs: str = "wale"
    wall_model: bool = True
    viscous: bool = True
    residual_drop: float = 1e-4
    save_volume: bool = False
    domain: list[float] = field(default_factory=list)
    full_model: bool = False  # flow all around the body (no symmetry plane): needed for sideslip, shown in full 3D
    init_from: str = ""  # grid sequencing: a folder with an earlier (coarser) run's volume.bin to start from
    force_band: float = 0.0  # > 0: a steady run stops once CL and CD stay within this fraction over 40 samples


def case(design: dict[str, Any], flight: Flight, ref: dict[str, Any], num: Numerics, out: str | Path, extra_model: dict[str, Any] | None = None,
         probes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    unsteady = num.unsteady or flight.gust_amplitude > 0 or flight.turbulence_intensity > 0
    grid: dict[str, Any] = {"cells_per_length": num.cells_per_length, "stretch": num.stretch, "max_cells": num.max_cells}
    if num.domain:
        grid["domain"] = num.domain
    if abs(flight.beta) > 0 or num.full_model:
        grid["half"] = False
    spec: dict[str, Any] = {
        "case": "body",
        "design": design,
        "flow": asdict(flight),
        "grid": grid,
        "model": {"viscous": num.viscous, "sgs": num.sgs, "wall_model": num.wall_model, "cfl": num.cfl if not unsteady else min(num.cfl, 0.9)},
        "reference": ref,
        "run": {
            "steps": num.steps,
            "unsteady": unsteady,
            "average_from": num.average_from if num.average_from >= 0 else (num.steps // 2 if unsteady else -1),
            "residual_drop": num.residual_drop,
            "output": str(Path(out).resolve()),
            "log_every": 10,
            "save_volume": num.save_volume,
        },
    }
    if extra_model:
        spec["model"].update(extra_model)
    if probes:
        spec["run"]["probes"] = probes
    if num.init_from:
        spec["run"]["init_from"] = str(Path(num.init_from).resolve())
    if num.force_band > 0:
        spec["run"]["force_band"] = num.force_band
    return spec


def run(design: dict[str, Any], flight: Flight, ref: dict[str, Any], out: str | Path, num: Numerics | None = None, timeout: float = 6 * 3600,
        extra_model: dict[str, Any] | None = None, probes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run one case; returns result.json plus 'dir'. Raises if the solver fails."""
    num = num or Numerics()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    c = case(design, flight, ref, num, out, extra_model, probes)
    (out / "case.json").write_text(json.dumps(c, indent=1), encoding="utf-8")
    t0 = time.time()
    r = subprocess.run([str(kernel.exe("aeroflow")), str(out / "case.json")], capture_output=True, text=True, timeout=timeout)
    (out / "run.log").write_text(r.stderr, encoding="utf-8")
    if r.returncode != 0 or not (out / "result.json").exists():
        raise RuntimeError(f"aeroflow failed ({r.returncode}): {r.stderr[-800:]}")
    res: dict[str, Any] = json.loads((out / "result.json").read_text(encoding="utf-8"))
    res["dir"] = str(out)
    res["wall_seconds"] = time.time() - t0
    return res


def history(d: str | Path) -> dict[str, NDArray[np.float64]]:
    with open(Path(d) / "history.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].startswith("#")]
    head = rows[0]
    data = np.array([r for r in rows[1:] if len(r) == len(head) and all(c.strip() for c in r)], float)  # a stopped run can leave a partial last line
    return {h: data[:, i] for i, h in enumerate(head)}


def surface(d: str | Path) -> dict[str, NDArray[np.float64]]:
    """Wall data per wall cell: x y z nx ny nz cp cp_rms cf T_wall_K yplus separated area."""
    with open(Path(d) / "surface.csv", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    data = np.array(rows[1:], float).reshape(-1, len(rows[0]))
    return {h: data[:, i] for i, h in enumerate(rows[0])}


def plane(d: str | Path, name: str) -> dict[str, Any]:
    """A field plane: coordinates 'coord_u', 'coord_v' (along 'u_axis', 'v_axis'), and each variable as a 2D array (v, u)."""
    h = json.loads((Path(d) / f"{name}.json").read_text(encoding="utf-8"))
    a = np.fromfile(Path(d) / f"{name}.bin", np.float32).reshape(len(h["vars"]), len(h["v"]), len(h["u"]))
    out: dict[str, Any] = {"coord_u": np.array(h["u"]), "coord_v": np.array(h["v"]), "axis": h["axis"], "at": h["at"], "u_axis": h["u_axis"], "v_axis": h["v_axis"]}
    for i, n in enumerate(h["vars"]):
        out[n] = a[i]
    return out


def run_special(case_json: dict[str, Any], out: str | Path, timeout: float = 6 * 3600) -> dict[str, Any]:
    """Run a built-in verification case (sod, tgv) or any prepared case dictionary."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    case_json = json.loads(json.dumps(case_json))
    case_json.setdefault("run", {})["output"] = str(out.resolve())
    (out / "case.json").write_text(json.dumps(case_json, indent=1), encoding="utf-8")
    r = subprocess.run([str(kernel.exe("aeroflow")), str(out / "case.json")], capture_output=True, text=True, timeout=timeout)
    (out / "run.log").write_text(r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"aeroflow failed: {r.stderr[-800:]}")
    res: dict[str, Any] = json.loads((out / "result.json").read_text(encoding="utf-8"))
    res["dir"] = str(out)
    return res
