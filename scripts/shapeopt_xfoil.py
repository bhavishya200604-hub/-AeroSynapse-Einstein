"""XFOIL judgement of the optimised section: NASA check, named regimes and every hold-out state."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from _common import RESULTS, ROOT

from aerosynapse.shapeopt import aero2d, scenarios, xfoil
from aerosynapse.shapeopt import geometry as G


def nasa_check() -> dict:
    d = json.loads((ROOT / "src" / "aerosynapse" / "data" / "naca0012_ladson.json").read_text(encoding="utf-8"))
    L = np.array(d["rows"])
    p = xfoil.polar(G.load_shape("naca0012"), 6e6, 0.0, 9.0, 0.05, -5.0, 21.0, 0.25, timeout=60)
    att = L[:, 0] <= 16.5
    cl = np.interp(L[:, 0], p["alpha"], p["CL"], left=np.nan, right=np.nan)
    cd = np.interp(L[:, 0], p["alpha"], p["CD"], left=np.nan, right=np.nan)
    ok = att & np.isfinite(cl)
    return {
        "alpha": L[:, 0].tolist(), "exp_cl": L[:, 1].tolist(), "exp_cd": L[:, 2].tolist(), "cl": cl.tolist(), "cd": cd.tolist(),
        "rms_cl": float(np.sqrt(np.mean((cl[ok] - L[ok, 1]) ** 2))), "mean_cd_err": float(np.mean(np.abs(cd[ok] / L[ok, 2] - 1))),
        "cl_max": float(p["CL"].max()), "alpha_cl_max": float(p["alpha"][np.argmax(p["CL"])]), "alpha_last": float(p["alpha"].max()),
    }  # fmt: skip


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--shape", default="naca2412")
    a.add_argument("--workers", type=int, default=3)
    n = a.parse_args()
    out = RESULTS / "shapeopt" / n.shape.replace(":", "_")
    v = json.loads((out / "verify.json").read_text(encoding="utf-8"))
    xy0, xyb = np.array(v["section"]["xy0"]), np.array(v["section"]["xyb"])
    W0, Wb = np.array(v["section"]["W0"]), np.array(v["section"]["Wb"])
    t0 = time.time()
    res: dict = {"nasa": nasa_check()}
    print("NASA check: CL rms", round(res["nasa"]["rms_cl"], 4), "CD error", round(res["nasa"]["mean_cd_err"], 4), "CLmax", round(res["nasa"]["cl_max"], 3), "at", res["nasa"]["alpha_cl_max"], flush=True)

    c = scenarios.concat(scenarios.named(), scenarios.random_extreme(300, 20240))  # the hold-out states
    nf0 = aero2d.solve_at_cl(W0[None], c, "xxxlarge")
    nfb = aero2d.solve_at_cl(Wb[None], c, "xxxlarge")

    def judge(i: int) -> dict:
        M = float(c.mach[i])
        row = {"name": c.names[i], "Re": float(c.Re[i]), "mach": M, "cl": float(c.cl[i]), "ncrit": float(c.ncrit[i]), "xtr": float(c.xtr[i]), "nf_ratio": float(nfb["cd_visc"][0, i] / nf0["cd_visc"][0, i])}
        if M > xfoil.MACH_LIMIT:
            return {**row, "skipped": True}
        beta_cl = float(c.cl[i])
        p0 = xfoil.polar(xy0, float(c.Re[i]), M, float(c.ncrit[i]), float(c.xtr[i]))
        p1 = xfoil.polar(xyb, float(c.Re[i]), M, float(c.ncrit[i]), float(c.xtr[i]))
        a0, a1 = xfoil.at_cl(p0, beta_cl), xfoil.at_cl(p1, beta_cl)
        return {**row, "skipped": False, "cd_base": a0["cd"], "cd_opt": a1["cd"], "cl_max_base": a0["cl_max"], "cl_max_opt": a1["cl_max"], "n0": len(p0["alpha"]), "n1": len(p1["alpha"])}

    done: list[dict] = []
    with ThreadPoolExecutor(max_workers=n.workers) as ex:  # threads only wait on XFOIL, which is its own process
        for k, r in enumerate(ex.map(judge, range(len(c)))):
            done.append(r)
            if k % 40 == 0:
                print(f"{k}/{len(c)} states, {time.time() - t0:.0f}s", flush=True)
    res["states"] = done
    res["polars"] = {}
    for key, Re, nc, xt in (("Re6e6", 6e6, 9.0, 1.0), ("Re3e6_dirty", 3e6, 9.0, 0.03), ("Re2e5", 2e5, 11.0, 1.0)):
        p0, p1 = xfoil.polar(xy0, Re, 0.0, nc, xt, -6, 22, 0.5), xfoil.polar(xyb, Re, 0.0, nc, xt, -6, 22, 0.5)
        res["polars"][key] = {"base": {q: p0[q].tolist() for q in ("alpha", "CL", "CD")}, "opt": {q: p1[q].tolist() for q in ("alpha", "CL", "CD")}}
    res["wall_s"] = time.time() - t0
    (out / "xfoil.json").write_text(json.dumps(res), encoding="utf-8")
    ok = [r for r in done if not r["skipped"] and np.isfinite(r["cd_base"]) and np.isfinite(r["cd_opt"])]
    x = np.array([r["cd_opt"] / r["cd_base"] for r in ok])
    f = np.array([r["nf_ratio"] for r in ok])
    print(f"judged {len(ok)} of {len(done)} states ({sum(r['skipped'] for r in done)} above Mach limit)")
    print(f"XFOIL ratio mean {x.mean():.4f} median {np.median(x):.4f} better {np.mean(x < 1):.3f} worst {x.max():.3f} best {x.min():.3f}")
    print(f"NeuralFoil on same states: mean {f.mean():.4f} better {np.mean(f < 1):.3f}; correlation {np.corrcoef(x, f)[0, 1]:.3f}")
    print("wall", round(time.time() - t0))


if __name__ == "__main__":
    main()
