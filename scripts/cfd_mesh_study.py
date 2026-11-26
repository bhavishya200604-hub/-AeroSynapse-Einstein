"""R-3: mesh independence of the 3D CFD forces, and agreement of half and full models (KI-68).

The same flight (Falcon cruise: Mach 0.85, 2 deg, 11 km) on a sequence of grids with a constant refinement ratio,
for the full aircraft and for the half model with a symmetry plane. Every steady run stops once its window-mean
forces settle (0.5 % band, up to ``--max-steps``). Each run is a folder of its own; a finished run is never
repeated, so the study resumes after a crash or a reboot.

Grid convergence follows Celik et al. (2008), "Procedure for estimation and reporting of uncertainty due to
discretization in CFD applications", J. Fluids Eng. 130(7), 078001: apparent order p from three grids (fixed-point
iteration for a non-constant ratio), Richardson-extrapolated value, and the fine-grid convergence index
GCI = 1.25 |e21| / (r21^p - 1). Oscillatory convergence (e32/e21 < 0) is reported as such, without an order.
The grid size is the core spacing h = L / cells_per_length (the stretching ratio and domain are the same on every
grid, so the core spacing sets the resolution near the aircraft).

    python scripts/cfd_mesh_study.py --probe 68                 # memory and time per step of one grid, 60 steps
    python scripts/cfd_mesh_study.py                            # the study (resumable)
    python scripts/cfd_mesh_study.py --analyse-only

Outputs in ``results/cfd_mesh_study/<tag>/``: one folder per run, ``study.json``, ``gci.json``,
``mesh_study.png`` and ``summary.txt``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from aerosynapse.sim import flow3d
from aerosynapse.sim.common import NEON, plt

ROOT = Path(__file__).resolve().parents[1]
MIN_FREE_GB = 10.0


class PeakMemory:
    """Polls the solver processes started by this one and keeps their peak working set and private bytes."""

    def __init__(self, every: float = 0.5):
        self.every, self.peak_ws, self.peak_private, self._stop = every, 0, 0, threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        me = psutil.Process()
        while not self._stop.is_set():
            try:
                for c in me.children(recursive=True):
                    if c.name().lower().startswith("aeroflow"):
                        m = c.memory_info()
                        self.peak_ws = max(self.peak_ws, m.rss)
                        self.peak_private = max(self.peak_private, getattr(m, "private", m.vms))
            except psutil.Error:
                pass
            self._stop.wait(self.every)

    def __enter__(self) -> PeakMemory:
        self._t.start()
        return self

    def __exit__(self, *a: Any) -> None:
        self._stop.set()
        self._t.join()


def setup(cpl: float, full: bool, steps: int) -> dict[str, Any]:
    return {"preset": "cruise", "quality": "custom", "cells_per_length": cpl, "steps": steps, "full_model": full, "unsteady": "off",
            "start": "free stream", "force_band": 0.5, "save_volume": True, "flow_pictures": "none", "max_cells": 2.5e6}


def run_one(design: dict[str, Any], d: Path, cpl: float, full: bool, steps: int) -> dict[str, Any]:
    rec_f = d / "mesh_record.json"
    if rec_f.exists():
        return json.loads(rec_f.read_text(encoding="utf-8"))
    free = shutil.disk_usage(d.parent).free / 1e9
    if free < MIN_FREE_GB:
        raise SystemExit(f"only {free:.1f} GB free on the results drive (need {MIN_FREE_GB}); stopping before {d.name}")
    t0 = time.time()
    with PeakMemory() as pm:
        r = flow3d.run_setup(design, setup(cpl, full, steps), d)
    c = r.get("coefficients") or {}
    rec = {"name": d.name, "model": "full" if full else "half", "cells_per_length": cpl, "cells": r.get("cells"), "fluid_cells": r.get("fluid_cells"),
           "wall_cells": r.get("wall_cells"), "thin_surface_cells": r.get("thin_surface_cells"), "steps": r.get("steps"), "stop_reason": r.get("stop_reason"),
           "forces_from": r.get("forces_from", "last step"), "CL": c.get("CL"), "CD": c.get("CD"), "CD_friction": c.get("CD_friction"), "Cm": c.get("Cm"),
           "solver_seconds": r.get("seconds"), "wall_seconds": time.time() - t0, "peak_working_set_MB": pm.peak_ws / 2**20,
           "peak_private_MB": pm.peak_private / 2**20, "grid": r.get("grid"), "half_model": r.get("half_model")}
    rec["seconds_per_step"] = rec["solver_seconds"] / max(rec["steps"] or 1, 1)
    rec["us_per_cell_step"] = 1e6 * rec["seconds_per_step"] / max(rec["cells"] or 1, 1)
    rec_f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def gci3(h: list[float], f: list[float]) -> dict[str, Any]:
    """Celik et al. (2008) for grids h1 < h2 < h3 (fine to coarse) and values f1, f2, f3."""
    h1, h2, h3 = h
    f1, f2, f3 = f
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    out: dict[str, Any] = {"h": h, "f": f, "r21": r21, "r32": r32, "e21": e21, "e32": e32}
    if abs(e21) < 1e-14 * max(abs(f1), 1e-30):
        out.update(kind="converged (no change)", p=None, f_ext=f1, gci_fine=0.0)
        return out
    ratio = e32 / e21
    if ratio < 0:
        out.update(kind="oscillatory", p=None, f_ext=None, gci_fine=None,
                   band=[min(f), max(f)], note="oscillatory convergence: no order; the spread of the three grids bounds the error")
        return out
    if ratio <= 1.0:
        # Celik's R = e21 / e32 >= 1: the change grows on refinement, the grids are not in the asymptotic range.
        # The order formula's absolute value would still return a (meaningless) positive p, so none is given.
        out.update(kind="divergent", p=None, f_ext=None, gci_fine=None, R=float(1.0 / ratio),
                   band=[min(f), max(f)], note="the change grows on refinement (R = e21/e32 >= 1): not in the asymptotic range, no order or GCI")
        return out
    s = np.sign(ratio)
    p = abs(np.log(abs(ratio))) / np.log(r21)
    for _ in range(200):  # fixed point for unequal ratios
        q = np.log((r21**p - s) / (r32**p - s))
        p_new = abs(np.log(abs(ratio)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    f_ext = (r21**p * f1 - f2) / (r21**p - 1)
    gci = 1.25 * abs(e21 / f1) / (r21**p - 1)
    out.update(kind="monotonic", p=float(p), f_ext=float(f_ext), gci_fine=float(gci), rel_err_fine=float(abs((f1 - f_ext) / f_ext)))
    return out


def analyse(out: Path, recs: list[dict[str, Any]], L: float) -> dict[str, Any]:
    res: dict[str, Any] = {"grid_size": "core spacing h = L / cells_per_length", "L": L, "models": {}}
    for model in ("full", "half"):
        rs = sorted([r for r in recs if r["model"] == model and r.get("CL") is not None], key=lambda r: -r["cells_per_length"])  # fine first
        m: dict[str, Any] = {"runs": [{k: r[k] for k in ("cells_per_length", "cells", "wall_cells", "thin_surface_cells", "steps", "stop_reason", "CL", "CD")} for r in rs]}
        if len(rs) >= 3:
            for q in ("CL", "CD"):
                trip = rs[:3]
                m[q] = gci3([L / r["cells_per_length"] for r in trip], [float(r[q]) for r in trip])
        res["models"][model] = m
    # half against full on the same grid
    pairs = []
    for r in recs:
        if r["model"] != "full":
            continue
        h = next((x for x in recs if x["model"] == "half" and x["cells_per_length"] == r["cells_per_length"]), None)
        if h and h.get("CL") is not None and r.get("CL") is not None:
            pairs.append({"cells_per_length": r["cells_per_length"], "CL_full": r["CL"], "CL_half": h["CL"], "CD_full": r["CD"], "CD_half": h["CD"],
                          "CL_ratio": h["CL"] / r["CL"], "CD_ratio": h["CD"] / r["CD"], "wall_full": r["wall_cells"], "wall_half": h["wall_cells"]})
    res["half_vs_full"] = sorted(pairs, key=lambda p: p["cells_per_length"])
    (out / "gci.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    pl = plt()
    fig, axs = pl.subplots(1, 2, figsize=(9.6, 3.6))
    for k, model in enumerate(("full", "half")):
        rs = sorted([r for r in recs if r["model"] == model and r.get("CL") is not None], key=lambda r: r["cells_per_length"])
        if not rs:
            continue
        hs = [L / r["cells_per_length"] for r in rs]
        for ax, q in zip(axs, ("CL", "CD"), strict=True):
            ax.plot(hs, [r[q] for r in rs], "o-", color=NEON[k], lw=1.2, ms=5, label=f"{model} model")
            g = res["models"][model].get(q)
            if g and g.get("f_ext") is not None:
                ax.plot([0], [g["f_ext"]], "*", color=NEON[k], ms=11, label=f"{model}: extrapolated (p = {g['p']:.2f})")
    for ax, q in zip(axs, ("CL", "CD"), strict=True):
        ax.set(xlabel="core grid spacing h [m]", ylabel=q, xlim=(left := 0, None))
        _ = left
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Falcon cruise (Mach 0.85, 2 deg): forces against grid spacing, full and half models")
    fig.savefig(out / "mesh_study.png", dpi=150)
    pl.close(fig)
    return res


def summary(out: Path, recs: list[dict[str, Any]], res: dict[str, Any]) -> None:
    lines = ["CFD mesh study: Falcon cruise, Mach 0.85, alpha 2 deg, 11 km", "=" * 60, "",
             f"{'model':5} {'cpl':>5} {'cells':>9} {'wall':>5} {'thin':>5} {'steps':>6} {'stop':>7} {'CL':>9} {'CD':>9} {'s/step':>7} {'peak MB':>8}"]
    for r in sorted(recs, key=lambda r: (r["model"], r["cells_per_length"])):
        lines.append(f"{r['model']:5} {r['cells_per_length']:5.0f} {r['cells']:9.0f} {r['wall_cells']:5.0f} {r['thin_surface_cells']:5.0f} {r['steps']:6.0f} "
                     f"{str(r['stop_reason']):>7} {r['CL']:9.5f} {r['CD']:9.5f} {r['seconds_per_step']:7.2f} {r['peak_private_MB']:8.0f}")
    lines.append("")
    for model, m in res["models"].items():
        for q in ("CL", "CD"):
            g = m.get(q)
            if not g:
                continue
            if g["kind"] == "monotonic":
                lines.append(f"{model} {q}: monotonic, order p = {g['p']:.2f}, extrapolated {g['f_ext']:.5f}, fine-grid GCI {100 * g['gci_fine']:.1f} %")
            else:
                lines.append(f"{model} {q}: {g['kind']}" + (f" (R = e21/e32 = {g['R']:.2f})" if g.get("R") is not None else "")
                             + (f"; three-grid band {g['band'][0]:.5f} to {g['band'][1]:.5f}" if g.get("band") else ""))
    for p in res["half_vs_full"]:
        lines.append(f"half / full at {p['cells_per_length']:.0f} cells per length: CL {p['CL_ratio']:.3f}, CD {p['CD_ratio']:.3f} "
                     f"(wall cells {p['wall_half']:.0f} half, {p['wall_full']:.0f} full)")
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default=str(ROOT / "results/cfd3d/falcon_cruise_full/case.json"))
    ap.add_argument("--tag", default="falcon_cruise_2026-11-11")
    ap.add_argument("--full", default="24,34,48,68", help="cells per length of the full-model grids")
    ap.add_argument("--half", default="24,34,48,68", help="cells per length of the half-model grids")
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--probe", type=float, default=0.0, help="only measure memory and time per step on this grid (full model, 60 steps)")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--exe", default="", help="aeroflow executable to use instead of bin/aeroflow (a development build)")
    a = ap.parse_args()
    if a.exe:
        from aerosynapse.cad import kernel

        dev = Path(a.exe).resolve()
        build = kernel.build
        kernel.exe = lambda name, _d=dev: _d if name == "aeroflow" else build(name)  # type: ignore[assignment]
    src = json.loads(Path(a.design).read_text(encoding="utf-8"))
    design = src.get("design", src)
    out = ROOT / "results" / "cfd_mesh_study" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if a.probe:
        d = out / f"probe_full_{a.probe:g}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        rec = run_one(design, d, a.probe, True, 60)
        print(json.dumps(rec, indent=1))
        return
    plan = [(float(c), True) for c in a.full.split(",") if c] + [(float(c), False) for c in a.half.split(",") if c]
    plan.sort(key=lambda t: (t[0], not t[1]))  # coarse first, full before half: results arrive early and cheap
    (out / "plan.json").write_text(json.dumps({"plan": plan, "max_steps": a.max_steps, "setup_example": setup(plan[0][0], plan[0][1], a.max_steps),
                                                "design": a.design, "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
    recs: list[dict[str, Any]] = []
    for cpl, full in plan:
        d = out / f"{'full' if full else 'half'}_{cpl:g}"
        if a.analyse_only:
            if (d / "mesh_record.json").exists():
                recs.append(json.loads((d / "mesh_record.json").read_text(encoding="utf-8")))
            continue
        d.mkdir(exist_ok=True)
        print(f"{time.strftime('%H:%M:%S')} start {d.name}", flush=True)
        r = run_one(design, d, cpl, full, a.max_steps)
        recs.append(r)
        print(f"{time.strftime('%H:%M:%S')} done  {d.name}: {r['cells']} cells, {r['steps']} steps ({r['stop_reason']}), CL {r['CL']:.5f}, CD {r['CD']:.5f}, "
              f"{r['wall_seconds']:.0f} s, peak {r['peak_private_MB']:.0f} MB", flush=True)
        L = float(json.loads((d / "run_info.json").read_text(encoding="utf-8"))["reference"]["length"])
        summary(out, recs, analyse(out, recs, L))  # after every run, so partial results are always on disk
    if a.analyse_only and recs:
        d0 = out / recs[0]["name"]
        L = float(json.loads((d0 / "run_info.json").read_text(encoding="utf-8"))["reference"]["length"])
        summary(out, recs, analyse(out, recs, L))


if __name__ == "__main__":
    main()
