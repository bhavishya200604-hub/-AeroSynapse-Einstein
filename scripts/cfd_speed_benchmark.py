"""How much faster are the CFD speed options? The same cruise case on the same grid, run

* ``all_steps`` - every step (the old behaviour);
* ``stop_on_forces`` - stopped once the mean forces of two consecutive 600-step windows agree within the band;
* ``sequenced`` - a coarse pass first, then the full grid from its field, also stopped on the forces.

Writes ``results/cfd_speed/benchmark.json`` and ``benchmark.png`` (CL and CD against wall-clock time). Folders that
already hold a finished run of a variant are reused (``--fresh`` reruns them).

    python scripts/cfd_speed_benchmark.py [--quality draft] [--band 0.5]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from aerosynapse.cfd import runner
from aerosynapse.sim import flow3d
from aerosynapse.sim.common import NEON, plt

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (("all_steps", "free stream", False, "every step"), ("stop_on_forces", "free stream", True, "stop on settled mean forces"),
            ("sequenced", "coarse pass first", True, "coarse pass first, stop on forces"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="draft")
    ap.add_argument("--band", type=float, default=0.5, help="force band, percent")
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--design", default=str(ROOT / "results/cfd3d/falcon_cruise_full/case.json"))
    a = ap.parse_args()
    src = json.loads(Path(a.design).read_text(encoding="utf-8"))
    design = src.get("design", src)
    out = ROOT / "results/cfd_speed"
    out.mkdir(parents=True, exist_ok=True)
    base = {"preset": "cruise", "quality": "custom", "cells_per_length": {"live": 18, "draft": 30, "standard": 48, "fine": 70}[a.quality], "steps": a.steps,
            "save_volume": False, "flow_pictures": "none", "unsteady": "off"}
    runs = {}
    for name, start, stop, _lab in VARIANTS:
        d = out / name
        if a.fresh and d.exists():
            shutil.rmtree(d)
        if not (d / "run_info.json").exists():
            t0 = time.time()
            flow3d.run_setup(design, {**base, "start": start, "force_band": a.band if stop else 0.0}, d)
            (d / "wall_seconds.txt").write_text(f"{time.time() - t0:.1f}", encoding="utf-8")
        r = json.loads((d / "result.json").read_text(encoding="utf-8"))
        info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
        seq = info.get("sequencing") or {}
        wall = float((d / "wall_seconds.txt").read_text()) if (d / "wall_seconds.txt").exists() else (seq.get("coarse_seconds") or 0) + r["seconds"]
        runs[name] = {"wall_seconds": wall, "solver_seconds": r.get("seconds"), "steps": r.get("steps"), "stop_reason": r.get("stop_reason", "steps"),
                      "CL": (r.get("coefficients") or {}).get("CL"), "CD": (r.get("coefficients") or {}).get("CD"), "cells": r.get("cells"),
                      "forces_from": r.get("forces_from", "last step"), "sequencing": seq}
        print(name, json.dumps(runs[name]), flush=True)
    ref = runs["all_steps"]
    for R in runs.values():
        R["speedup"] = ref["wall_seconds"] / max(R["wall_seconds"], 1e-9)
        R["CL_error_percent"] = 100 * (R["CL"] - ref["CL"]) / ref["CL"]
        R["CD_error_percent"] = 100 * (R["CD"] - ref["CD"]) / ref["CD"]
    summary = {"quality": a.quality, "band_percent": a.band, "reference": "all_steps", "runs": runs}
    (out / "benchmark.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    pl = plt()
    fig, axs = pl.subplots(1, 2, figsize=(9.6, 3.5))
    for k, (name, _s, _st, lab) in enumerate(VARIANTS):
        h = runner.history(out / name)
        R = runs[name]
        t_off = R["sequencing"].get("coarse_seconds") or 0.0
        t = t_off + h["step"] / max(R["steps"], 1) * R["solver_seconds"]
        for ax, key in zip(axs, ("CL", "CD"), strict=True):
            ax.plot(t, h[key], color=NEON[k], lw=1.0, label=f"{lab}: {R['wall_seconds']:.0f} s")
            ax.plot([t[-1]], [R[key]], "o", color=NEON[k], ms=5, zorder=4)
    for ax, key in zip(axs, ("CL", "CD"), strict=True):
        v = ref[key]
        ax.axhline(v, color="#111111", lw=0.6, ls=":")
        ax.set(xlabel="wall-clock time [s]", ylabel=key, ylim=(v * 0.85, v * 1.15))
    axs[0].legend(frameon=False, fontsize=8)
    S = runs["stop_on_forces"]
    fig.suptitle(f"Stopping on settled mean forces: {S['speedup']:.1f} times faster, CL {S['CL_error_percent']:+.2f} %, CD {S['CD_error_percent']:+.2f} % "
                 f"({a.quality} grid; dot = reported value)")
    fig.savefig(out / "benchmark.png", dpi=150)
    print(json.dumps({n: {k: R[k] for k in ("speedup", "CL_error_percent", "CD_error_percent", "steps")} for n, R in runs.items()}))


if __name__ == "__main__":
    main()
