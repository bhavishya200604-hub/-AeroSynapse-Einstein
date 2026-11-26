"""Stage 3: density topology optimisation of a body (see aerosynapse.shapeopt.topology)."""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from _common import RESULTS  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from aerosynapse.shapeopt import geometry as G  # noqa: E402
from aerosynapse.shapeopt import topology as T  # noqa: E402

PROFILES = {
    "quick": dict(h=1 / 12, iters=40),
    "standard": dict(h=1 / 16, iters=120),
    "full": dict(h=1 / 24, iters=200),
}
STATES = [
    T.FlowState(100.0, 0.0, name="Re100_a0"),
    T.FlowState(40.0, 10.0, name="Re40_a+10"),
    T.FlowState(40.0, -10.0, name="Re40_a-10"),
]


def contours(phys: np.ndarray, g: T.Grid) -> list[np.ndarray]:
    """Solid outlines at density 0.5, as closed polygons in domain coordinates."""
    cs = plt.contour(g.xc, g.yc, phys.T, levels=[0.5])
    segs = [s for s in cs.allsegs[0] if len(s) > 8]
    plt.close("all")
    return segs


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--shape", default="square")
    a.add_argument("--start", default="shape", choices=["shape", "grey"])
    a.add_argument("--profile", default="standard", choices=list(PROFILES))
    a.add_argument("--chord", type=float, default=0.8, help="scale the input shape to this length")
    n = a.parse_args()
    p = PROFILES[n.profile]
    g = T.Grid(h=p["h"])
    xy = G.load_shape(n.shape)
    xy = xy - xy.mean(0)
    xy = xy * (n.chord / np.ptp(xy[:, 0]))
    cx, cy = 0.5 * (g.box[0] + g.box[1]), 0.5 * (g.box[2] + g.box[3])
    poly = xy + np.array([cx, cy])
    solid0 = G.polygon_mask([poly], g.xc, g.yc).astype(float)
    area_cells = float(solid0.sum())
    Tp = T.Topology(g, STATES, area_cells=area_cells)
    x0 = solid0 if n.start == "shape" else np.where(Tp.mask, area_cells / Tp.mask.sum(), 0.0)
    t0 = time.time()
    tag = f"{n.shape.replace(':', '_')}_{n.start}"
    print(f"[{tag}] {g.nx}x{g.ny} grid, {Tp.ns.N} unknowns, solid area {area_cells * g.h**2:.3f}", flush=True)
    res = T.optimise(Tp, x0, iters=p["iters"], log=lambda s: print(f"[{tag}] {s} t {time.time() - t0:.0f}s", flush=True))
    phys = Tp.field(res.gamma, 32.0)[0]
    # score the unmodified input on the same footing
    base_phys = solid0
    Tb = T.Topology(g, STATES)
    base = [Tb.analyse(base_phys, s, i)[0] for i, s in enumerate(STATES)]
    final = [Tb.analyse((phys > 0.5).astype(float), s, i)[0] for i, s in enumerate(STATES)]
    nb, nh = T.count_bodies(phys)
    if nb == 0 or (phys > 0.5).sum() < 0.8 * area_cells:
        raise SystemExit(f"[{tag}] rejected: the body vanished or lost area ({(phys > 0.5).sum()} of {area_cells:.0f} cells)")
    polys = contours(phys, g)
    out = RESULTS / "shapeopt" / "topology"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{tag}.npz", phys=phys, base=solid0, frames=np.array(res.frames), xc=g.xc, yc=g.yc)
    (out / f"{tag}.json").write_text(
        json.dumps(
            {
                "shape": n.shape,
                "start": n.start,
                "h": g.h,
                "states": [s.name for s in STATES],
                "area": area_cells * g.h**2,
                "area_final": float((phys > 0.5).sum() * g.h**2),
                "cd_base": base,
                "cd_final": final,
                "bodies": nb,
                "holes": nh,
                "history": res.history,
                "polygons": [q.tolist() for q in polys],
                "base_polygon": poly.tolist(),
                "wall_s": time.time() - t0,
            },
            indent=0,
        )
    )
    print(f"[{tag}] done: Cd {np.round(base, 3)} -> {np.round(final, 3)}, bodies {nb}, holes {nh}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
