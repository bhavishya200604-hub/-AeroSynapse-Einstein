"""Lift of a rectangular wing against its thickness in grid cells: where does the immersed boundary lose lift?

    python scripts/cfd_thickness_study.py --tag baseline
    python scripts/cfd_thickness_study.py --tag fixed --exe path/to/aeroflow_dev.exe

Rectangular wing, aspect ratio 6, NACA section, Mach 0.5, 4 deg, inviscid (no wall model, no subgrid model), half
model, one grid (cells per chord set by --cpc). Only the thickness changes, so the wing passes from a one-cell sheet
(thin-surface treatment) through one to three cells thick (ghost cells that may serve both surfaces) to a resolved
body. The reference is the vortex lattice at the same Mach number (Prandtl-Glauert), corrected for thickness by the
usual small factor 1 + 0.8 t/c on the section lift slope (Abbott and von Doenhoff), reported separately.
Results: results/cfd_lift_study/<tag>/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from _common import RESULTS

from aerosynapse.cad import kernel
from aerosynapse.cfd import runner
from aerosynapse.fighter import vlm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--exe", default="", help="aeroflow executable to use instead of bin/aeroflow (a development build)")
    ap.add_argument("--tc", default="0.02,0.06,0.10,0.15,0.20", help="thickness ratios")
    ap.add_argument("--cpc", type=float, default=12.0, help="cells per chord")
    ap.add_argument("--mach", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    if a.exe:
        dev = Path(a.exe).resolve()
        kernel.exe = lambda name, _d=dev: _d if name == "aeroflow" else kernel.build(name)  # type: ignore[assignment]
    out = RESULTS / "cfd_lift_study" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    ref = {"length": 1.0, "area": 6.0, "chord": 1.0, "span": 6.0, "moment_point": [0.25, 0, 0]}
    rows = []
    for tc in [float(t) for t in a.tc.split(",") if t]:
        wing = {"id": "w", "type": "wing", "span": 3.0, "root_chord": 1.0, "taper": 1.0, "sweep": 0.0, "tc": tc, "profile": "naca",
                "translate": [0, 0, 0], "mirror": True}
        lat = vlm.derivatives([wing], a.mach, 6.0, 1.0, 6.0, 0.25, nc=8, ns=24).CL_alpha * np.radians(a.alpha)
        d = out / f"tc{tc:g}"
        t0 = time.time()
        if (d / "result.json").exists():
            r = json.loads((d / "result.json").read_text(encoding="utf-8"))
        else:
            r = runner.run({"features": [wing]}, runner.Flight(mach=a.mach, alpha=a.alpha, altitude=0.0), ref, d,
                           runner.Numerics(cells_per_length=a.cpc, steps=a.steps, cfl=1.2, residual_drop=1e-5, viscous=False, sgs="none",
                                           wall_model=False, domain=[2, 3, 1.0, 2], force_band=0.005))
        cl = float(r["coefficients"]["CL"])
        row = {"tc": tc, "thickness_cells": tc * a.cpc, "CL": cl, "CD": float(r["coefficients"]["CD"]), "CL_lattice": float(lat),
               "CL_lattice_thick": float(lat * (1 + 0.8 * tc)), "ratio": cl / lat, "cells": r.get("cells"), "wall_cells": r.get("wall_cells"),
               "thin_surface_cells": r.get("thin_surface_cells"), "steps": r.get("steps"), "stop": r.get("stop_reason"), "seconds": time.time() - t0}
        rows.append(row)
        print(f"t/c {tc:5.2f} ({row['thickness_cells']:4.2f} cells): CL {cl:.4f}, lattice {lat:.4f}, ratio {row['ratio']:.3f}; "
              f"wall {row['wall_cells']}, thin {row['thin_surface_cells']}, {row['steps']} steps ({row['stop']}), {row['seconds']:.0f} s", flush=True)
        (out / "summary.json").write_text(json.dumps({"exe": a.exe or "bin/aeroflow", "cpc": a.cpc, "mach": a.mach, "alpha": a.alpha, "rows": rows,
                                                      "date": time.strftime("%Y-%m-%d %H:%M")}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
