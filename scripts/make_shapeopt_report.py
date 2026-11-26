"""Build the baseline-versus-optimised comparison page from the results on disk.

Every number on the page is read from search.json, verify.json, ns.json or the topology files, so
the text cannot drift from the data. Run after shapeopt_search.py, shapeopt_topology.py and
shapeopt_verify.py. Output: results/shapeopt/<shape>/report.html
"""

from __future__ import annotations

import argparse
import base64
import html
import json

import numpy as np
from _common import RESULTS

TEMPLATE = (RESULTS.parent / "scripts" / "shapeopt_report_template.html").read_text(encoding="utf-8")


def q8(a: np.ndarray, lim: float) -> str:
    """Quantise a field to one byte per cell for the page."""
    v = np.clip((np.asarray(a, float) / lim + 1) * 127.5, 0, 255).astype(np.uint8)
    return base64.b64encode(v.tobytes()).decode()


def r(x, n=4):
    return np.round(np.asarray(x, float), n).tolist()


def pct(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--shape", default="naca2412")
    n = a.parse_args()
    out = RESULTS / "shapeopt" / n.shape.replace(":", "_")
    v = json.loads((out / "verify.json").read_text(encoding="utf-8"))
    s = json.loads((out / "search.json").read_text(encoding="utf-8"))
    ns_p = out / "ns.json"
    ns = json.loads(ns_p.read_text(encoding="utf-8")) if ns_p.exists() else []
    nsd = {c["name"]: c for c in ns}

    D: dict = {"name": n.shape.upper().replace("NACA", "NACA ")}
    sec = v["section"]
    D["xy0"], D["xyb"] = r(sec["xy0"][::2], 5), r(sec["xyb"][::2], 5)
    D["m0"], D["mb"] = sec["m0"], sec["mb"]
    w = v["wing"]
    D["names"] = w["names"]
    D["wing"] = {
        k: {
            q: (r(w[k][q], 4) if isinstance(w[k][q], list) else w[k][q])
            for q in ("plan", "span", "mac", "drag", "CL", "CDi", "CD", "y", "chord", "cl_section", "gamma", "cl_max_wing", "eta_stall", "v_stall", "wing_mass", "takeoff_mass", "root_bending", "n_gust", "v_flutter", "v_divergence", "t_c")
        }
        for k in ("base", "opt")
    }
    D["v_dive"] = w["v_dive"]
    D["named"] = v["named"]
    rb = v["robust"]
    D["robust"] = {**{k: rb[k] for k in rb if not isinstance(rb[k], list)}, "ratio": r(rb["ratio"], 4), "Re": r(rb["Re"], 0), "mach": r(rb["mach"], 3)}
    D["holdout"] = v["holdout"]
    D["confidence"] = v.get("confidence")
    xp = out / "xfoil.json"
    if xp.exists():
        x = json.loads(xp.read_text(encoding="utf-8"))
        st = x["states"]
        ok = [q for q in st if not q["skipped"] and np.isfinite(q["cd_base"]) and np.isfinite(q["cd_opt"])]
        ra = np.array([q["cd_opt"] / q["cd_base"] for q in ok])
        nf = np.array([q["nf_ratio"] for q in ok])
        far = np.abs(ra - nf) > 0.1
        D["xf"] = {
            "failed_opt": sum(1 for q in st if not q["skipped"] and np.isfinite(q["cd_base"]) and not np.isfinite(q["cd_opt"])),
            "n": len(ok), "total": len(st), "skipped": sum(q["skipped"] for q in st), "failed": len(st) - len(ok) - sum(q["skipped"] for q in st),
            "mean": float(ra.mean()), "median": float(np.median(ra)), "better": float((ra < 1).mean()), "worst": float(ra.max()), "cvar90": float(np.sort(ra)[int(0.9 * len(ra)) :].mean()),
            "nf_mean": float(nf.mean()), "nf_better": float((nf < 1).mean()), "corr": float(np.corrcoef(ra, nf)[0, 1]), "far": int(far.sum()), "mean_near": float(ra[~far].mean()),
            "pts": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(nf, ra, strict=True)],
            "named": [{k: q.get(k) for k in ("name", "Re", "mach", "cl", "skipped", "cd_base", "cd_opt", "nf_ratio")} for q in st[:11]],
            "nasa": {k: x["nasa"][k] for k in ("rms_cl", "mean_cd_err", "alpha_last")},
        }  # fmt: skip
    D["seeds"] = v["seeds"]
    D["polars"] = {k: {d: {q: r(p[d][q], 4) for q in ("alpha", "CL", "CD", "conf")} for d in ("base", "opt")} for k, p in v["polars"].items()}
    D["vii"] = {k: {"alpha": c["alpha"], "base": {q: r(c["base"][q], 5) for q in ("CL", "CD", "xsep_u")}, "opt": {q: r(c["opt"][q], 5) for q in ("CL", "CD", "xsep_u")}, "matched": c["matched"], "clmax_base": c["clmax_base"], "clmax_opt": c["clmax_opt"], "astall_base": c["astall_base"], "astall_opt": c["astall_opt"]} for k, c in v["vii"].items()}
    D["dyn"] = {k: {q: (r(c[q], 3) if isinstance(c[q], list) else c[q]) for q in c} for k, c in v["dynamic_stall"].items()}
    D["hl"] = {"clean": v["high_lift"]["clean_cl_max"], "dcp": v["high_lift"]["dcp_crit"], "options": [{**{k: o[k] for k in ("name", "cl_max", "alpha", "gain", "clearance")}, "layout": o["layout"], "elements": [r(e, 4) for e in o["elements"]]} for o in v["high_lift"]["options"]]}
    tr = []
    for run in s["runs"]:
        if run["trace"]:
            tr.append({"label": f"CMA-ES seed {run['seed']}", "evals": [t["evals"] for t in run["trace"]], "J": r([t["holdout_J"] for t in run["trace"]], 4), "train": r([t["train_J"] for t in run["trace"]], 4)})
    D["trace"] = tr
    D["wall_s"] = s.get("wall_s", 0)

    # topology
    T = {}
    for tag in ("square_shape", "square_grey"):
        jp = RESULTS / "shapeopt" / "topology" / f"{tag}.json"
        if not jp.exists():
            continue
        j = json.loads(jp.read_text(encoding="utf-8"))
        z = np.load(RESULTS / "shapeopt" / "topology" / f"{tag}.npz")
        fr = z["frames"]
        xc, yc = z["xc"], z["yc"]
        ix = np.where((xc > 1.0) & (xc < 3.4))[0]
        iy = np.where((yc > 1.2) & (yc < 2.8))[0]
        pick = [0, 3, 6, 8, 10, len(fr) - 1]
        pick = [min(p, len(fr) - 1) for p in pick]
        T[tag] = {
            "nx": len(ix),
            "ny": len(iy),
            "frames": [base64.b64encode((np.clip(fr[p][np.ix_(ix, iy)], 0, 1) * 255).astype(np.uint8).tobytes()).decode() for p in pick],
            "iters": [min(p * 10, j["history"][-1]["iter"]) for p in pick],
            "base": base64.b64encode((z["base"][np.ix_(ix, iy)] * 255).astype(np.uint8).tobytes()).decode(),
            "cd_base": j["cd_base"],
            "cd_final": j["cd_final"],
            "states": j["states"],
            "bodies": j["bodies"],
            "holes": j["holes"],
            "area": j["area"],
            "area_final": j["area_final"],
            "J": r([h["J"] for h in j["history"]], 3),
            "beta": [h["beta"] for h in j["history"]],
        }
    if len(T) == 2:
        A = np.load(RESULTS / "shapeopt" / "topology" / "square_shape.npz")["phys"] > 0.5
        B = np.load(RESULTS / "shapeopt" / "topology" / "square_grey.npz")["phys"] > 0.5
        T["iou"] = float((A & B).sum() / (A | B).sum())
    D["topo"] = T

    # Navier-Stokes
    D["ns"] = [{k: c[k] for k in ("name", "Re", "alpha", "h", "cd_mean", "cl_mean", "cl_rms", "cd_rms", "strouhal", "unsteady", "wall_s", "divergence_max")} for c in ns]
    fields = {}
    for key in ("airfoil_base_a18", "airfoil_opt_a18", "topo_baseline_Re100", "topo_square_shape_Re100"):
        if key in nsd:
            c = nsd[key]
            F = np.array(c["vorticity"], float)
            fields[key] = {"nx": F.shape[0], "ny": F.shape[1], "w": q8(F, 8.0), "s": base64.b64encode(np.array(c["solid"], np.uint8).tobytes()).decode(), "t": r(c["t"][::4], 2), "cl": r(c["cl_t"][::4], 4)}
    D["fields"] = fields

    page = TEMPLATE.replace("/*__DATA__*/null", json.dumps(D, separators=(",", ":")))
    page = page.replace("__NAME__", html.escape(D["name"]))
    (out / "report.html").write_text(page, encoding="utf-8")
    print("wrote", out / "report.html", round(len(page) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
