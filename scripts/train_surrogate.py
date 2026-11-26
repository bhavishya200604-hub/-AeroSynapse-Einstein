"""Build a fighter-performance dataset on disk and train a surrogate from it, streamed in bounded memory.

    python scripts/train_surrogate.py --profile standard [--store D:/aerosynapse_store]

1. Sweep: Sobol samples over the freeform family (every topology option) evaluated with the full performance
   model, in separate processes; each row [37 design variables, 14 outputs] goes to float32 shards on disk.
   Evaluations made during the optimisation (data_store/fighter_evals) are added as further rows.
2. Train: an MLP from the shards (shuffle buffer, Welford normalisation, checkpoint per epoch, held-out shards).
3. Report R^2 and relative error per output on the held-out shards, against a nearest-neighbour baseline.

The data store defaults to <project>/data_store; set AEROSYNAPSE_STORE (or --store) to use another drive.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from _common import RESULTS, ROOT
from scipy.stats import qmc

# physical ranges: a few random designs give degenerate lattices (e.g. maximum lift in the thousands); they are
# counted and left out. Heavy-tailed positive outputs are learned as logarithms.
VALID = {"W0": (5e3, 3e4), "fuel": (500, 2e4), "cruise_drag_N": (1e3, 1e5), "dash_drag_N": (5e3, 3e5), "takeoff_m": (100, 5000), "approach_mps": (30, 200), "CL_max": (0.3, 3.0), "gust_dn": (0.0, 20.0)}
LOG = ("W0", "empty", "fuel", "cruise_drag_N", "dash_drag_N", "takeoff_m")
OUTPUTS = ["W0", "empty", "fuel", "CD0", "cruise_drag_N", "dash_drag_N", "turn_rate_sustained_dps", "Ps_climb", "takeoff_m", "approach_mps", "CL_max", "gust_dn", "extreme_cvar", "feasible"]
PROFILES = {"quick": dict(n=96, epochs=30), "standard": dict(n=1200, epochs=80), "full": dict(n=5000, epochs=150)}


def row(r) -> list[float]:  # noqa: ANN001
    m = r.metrics
    return [r.W0, r.empty, r.fuel, r.CD0, m["cruise_drag_N"], m["dash_drag_N"], m["turn_rate_sustained_dps"], m["Ps_climb"], m["takeoff_m"], m["approach_mps"], m["CL_max"], m["gust_dn"], m["extreme_cvar"], float(r.feasible)]


def worker(start: int, stop: int, n: int, calib: float, store: str) -> None:
    from aerosynapse.fighter import design as D
    from aerosynapse.fighter import performance as P
    from aerosynapse.store.shards import ArrayWriter

    fam = "freeform"
    nv = len(D.variables(fam))
    z = qmc.Sobol(nv, seed=0).random(n)[start:stop]
    w = ArrayWriter(store, f"fighter_sweep/part_{start:06d}", (nv + len(OUTPUTS),), max_bytes=4 * (nv + len(OUTPUTS)) * 100)
    m = P.Mission(calib=calib)
    for zi in z:
        try:
            r = P.evaluate(fam, D.decode(fam, zi), m)
            y = np.array(row(r))
            if np.all(np.isfinite(y)) and r.W0 < 60000:
                w.add(np.concatenate([zi, y]).astype(np.float32))
        except Exception:  # designs the kernel cannot build are skipped
            pass
    w.close()


DROPPED = [0]


def merge(store: str, nv: int) -> str:
    """Gather every part (and the optimiser's evaluations) into one shard set; sizes on disk are bounded per shard."""
    from aerosynapse.store.shards import ArrayReader, ArrayWriter

    out = os.path.join(store, "fighter_rows")
    w = ArrayWriter(store, "fighter_rows", (nv + len(OUTPUTS),), max_bytes=4 * (nv + len(OUTPUTS)) * 200)
    base = os.path.join(store, "fighter_sweep")
    for part in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        rd = ArrayReader(os.path.join(base, part))
        for k in range(len(rd.man.get("shards", []))):
            for r in rd.shard(k):
                r = np.array(r, np.float64)
                y = r[nv:]
                if not all(lo <= y[OUTPUTS.index(n)] <= hi for n, (lo, hi) in VALID.items()):
                    DROPPED[0] += 1
                    continue
                for n in LOG:
                    y[OUTPUTS.index(n)] = np.log(y[OUTPUTS.index(n)])
                w.add(r.astype(np.float32))
    w.close()
    print("[surrogate] merged the sweep rows (the optimiser's own evaluations stay in fighter_evals: they cluster near optima)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="standard", choices=list(PROFILES))
    ap.add_argument("--store", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--part", nargs=4, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.store:
        os.environ["AEROSYNAPSE_STORE"] = a.store
    from aerosynapse.fighter import design as D
    from aerosynapse.store.shards import store_root

    store = str(store_root())
    calib = json.loads((RESULTS / "fighter" / "calibration.json").read_text())["calib"]
    if a.part:
        worker(int(a.part[0]), int(a.part[1]), int(a.part[2]), calib, a.part[3])
        return
    prof = PROFILES[a.profile]
    n = prof["n"]
    nv = len(D.variables("freeform"))
    t0 = time.time()
    chunk = 48
    todo = [(s, min(s + chunk, n)) for s in range(0, n, chunk) if not os.path.exists(os.path.join(store, "fighter_sweep", f"part_{s:06d}", "manifest.json"))]
    running: list[subprocess.Popen[bytes]] = []
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    while todo or running:
        while todo and len(running) < a.workers:
            s, e = todo.pop(0)
            running.append(subprocess.Popen([sys.executable, __file__, "--part", str(s), str(e), str(n), store], env=env, cwd=str(ROOT / "scripts")))
        time.sleep(2)
        running = [p for p in running if p.poll() is None]
    sweep_s = time.time() - t0
    import shutil

    shutil.rmtree(os.path.join(store, "fighter_rows"), ignore_errors=True)
    rows = merge(store, nv)
    from aerosynapse.store import train as T

    rep = T.train(rows, nv, RESULTS / "surrogate", epochs=prof["epochs"])
    # nearest-neighbour baseline on exactly the held-out shards the network never saw
    from aerosynapse.store.shards import ArrayReader

    rd = ArrayReader(rows)
    va = np.concatenate([np.asarray(rd.shard(k)) for k in rep["val_shard_ids"]])
    trn = np.concatenate([np.asarray(rd.shard(k)) for k in rep["train_shard_ids"]])
    lo, hi = trn[:, :nv].min(0), trn[:, :nv].max(0)
    sc = lambda x: (x - lo) / np.maximum(hi - lo, 1e-9)  # noqa: E731
    pred = np.empty((len(va), trn.shape[1] - nv), np.float32)
    for i in range(0, len(va), 256):  # in blocks: the distance matrix stays small
        d = ((sc(va[i : i + 256, None, :nv]) - sc(trn[None, :, :nv])) ** 2).sum(-1)
        pred[i : i + 256] = trn[d.argmin(1), nv:]
    y = va[:, nv:]
    r2_nn = 1 - ((y - pred) ** 2).sum(0) / np.maximum(((y - y.mean(0)) ** 2).sum(0), 1e-12)
    rep.update({"dropped_unphysical_rows": DROPPED[0], "log_outputs": list(LOG), "valid_ranges": VALID, "outputs": OUTPUTS, "sweep_seconds": sweep_s, "store": rows, "nearest_neighbour_r2": r2_nn.tolist(), "store_bytes": sum(sh["bytes"] for sh in rd.man["shards"])})
    (RESULTS / "surrogate" / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    for o, r2, nn in zip(OUTPUTS, rep["val_r2"], r2_nn, strict=True):
        print(f"{o:24s} R2 network {r2:6.3f}   nearest neighbour {nn:6.3f}")


if __name__ == "__main__":
    main()
