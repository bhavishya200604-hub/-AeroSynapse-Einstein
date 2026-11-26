"""Step 6.6b: the Phase 6 analyses of Steps 6.2 (UQ), 6.3 (extrapolation) and 6.5 (training diagnostics) on the saved
2D parametric networks, against the verified 2D FV solver (solutions cached on disk and reused).

Cases (twod_study.cases): the Graetz channel ensemble of Step 4.7 (members = seeds; Robin walls; QoI mean Nu on the
bottom wall) and the heated lid-driven cavity of Step 4.5a (members = its parametric variants; QoI Nu_top) — the
same adapters as Step 6.4 (experiments/p6_4_sensitivity.py), whose reload check proved the networks are rebuilt exactly.
(U) UQ          random log-uniform points in the training box: ensemble mean / spread vs the solver QoI; coverage of
                Gaussian intervals, ECE, CRPS, Spearman(spread, |error|); split-conformal recalibration over random
                calibration / test splits; field level: theta of the members vs the solver on the solver grid
                (pointwise coverage, Spearman) at a few points
(X) extrapolation   rays from the box centre (log space; the 3 axes both ways + random directions) to points at a
                distance d (decades outside the training box); relative QoI error vs d; truth-free indicators: ensemble
                spread (relative), physics residual (RMS of the trained residual groups at that parameter value,
                mean over members — the 2D operators are NONLINEAR, so this is not the residual of the mean) and
                their combination; thresholds = a quantile of the in-range indicator; TPR / FPR / AUC / Spearman;
                a (p1, p2) slice through the box centre extended beyond the box (3D error surfaces)
(D) diagnostics per member, at initialisation (rebuilt with the training seed; RELOAD CHECK = the step-0 losses of
                the training log) and at the final weights: losses per residual group, gradient of every group
                (norms, cosine matrix; PDE vs boundary groups unweighted and with the frozen balancer weights;
                cancellation), residual NTK spectrum (effective rank, condition, traces per group, drift), Hessian
                (Lanczos extremes; full spectrum at the final weights), loss landscape around the final weights
                (filter-normalised directions) and the loss along the straight line init -> final.
Snapshot-data loss terms (Step 4.6 hybrid configurations) are not part of the residual groups (they need the
snapshot sets); the physics and boundary groups are exactly the trained ones.

    python experiments/p6_6b_2d.py [--config my.yaml] [--set key=value] [--tag name] [--parts uq,extrapolation,diagnostics]
                                   [--cases D_graetz,C_cavity] [--figures-only]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn import uq as UQ
from aerosynapse.thermal.pinn import validity as VA
from aerosynapse.thermal.utils.cache import cached
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz import twod_figs as F
from aerosynapse.thermal.viz.style import save_fig

MOD = "aerosynapse.thermal.experiments.p6_6b_2d"
CODE_TAG = "p6_6b-v1"  # part of every cache key: bump when the solver or the post-processing changes
NAMES = ["Re", "Pr", "Bi"]


def adapter(case: str, cfg: dict):
    from aerosynapse.thermal.experiments.p6_4_sensitivity import ADAPTERS

    return ADAPTERS[case]({"surrogate": cfg["surrogate"]})


def solver_kw(case: str, ad, cfg: dict) -> dict:
    s = cfg["solver"]
    if case == "D_graetz":
        return {"pcfg": ad.pcfg, "fixed": ad.fixed, "cells_per_unit": int(s["cells_per_unit"])}
    return {"cond": ad.cond, "fixed": ad.fixed, "grid": int(s["grid"]), "scheme": ad.scheme}


def _solve(case: str, vals: dict, kw: dict) -> dict:
    if case == "D_graetz":
        from aerosynapse.thermal.pinn import channel2d as CH

        r = CH.solve_channel(vals, kw["pcfg"], kw["cells_per_unit"], 1e-10, "quick")
        return {"qoi": -float(r["q_bottom"]), "x": r["x"], "y": r["y"], "theta": r["theta"], "wall_s": r["wall_s"]}
    from aerosynapse.thermal.pinn.ref2d import solve_steady_decoupled

    r = solve_steady_decoupled(vals, kw["cond"], kw["grid"], 1e-10, kw["scheme"])
    return {"qoi": float(r["nu_top"]), "x": r["x"], "y": r["y"], "theta": r["theta"], "wall_s": r["wall_s"]}


# --------------------------------------------------------------------------- worker tasks
def solver_chunk(case: str, P: list, kw: dict, cache_dir: str, fields: bool = False) -> list[dict]:
    """Solver QoI at every row of P (Re, Pr, Bi); each solve cached on disk; a failed solve is recorded, not raised."""
    out = []
    for p in P:
        vals = {**kw["fixed"], **dict(zip(NAMES, map(float, p), strict=True))}
        key = {"tag": CODE_TAG, "case": case, "vals": vals, "kw": {k: v for k, v in kw.items() if k != "fixed"}}
        try:
            r, hit = cached(cache_dir, f"twod_{case}", key, lambda v=vals: _solve(case, v, kw))
            rec = {"qoi": float(r["qoi"]), "ok": bool(np.isfinite(r["qoi"])), "hit": hit}
            if fields:
                rec |= {k: np.asarray(r[k]).tolist() for k in ("x", "y", "theta")}
        except Exception as e:  # noqa: BLE001  (a steady solve that does not converge far outside the box)
            rec = {"qoi": float("nan"), "ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        out.append(rec)
    return out


def _conflict(model, prob, pts: dict, w: dict | None) -> dict:
    """Gradient of every trained residual group L_k = mean r_k^2; norms, cosine matrix; PDE vs boundary groups
    (unweighted and with the balancer weights w): cosine and cancellation |g_a + g_b| / (|g_a| + |g_b|)."""
    from aerosynapse.thermal.pinn import diagnostics as D

    r = prob.residual_vectors(model, pts)
    ps = D.params(model)
    names = list(r)
    G, L = [], {}
    for k in names:
        lk = torch.mean(r[k] ** 2)
        L[k] = float(lk.detach())
        g = torch.autograd.grad(lk, ps, retain_graph=True, allow_unused=True)
        G.append(
            torch.cat([(z if z is not None else torch.zeros_like(p)).reshape(-1) for z, p in zip(g, ps, strict=True)])
        )
    G = torch.stack(G).detach()
    nrm = G.norm(dim=1)
    cos = (G @ G.T) / torch.clamp(nrm[:, None] * nrm[None, :], min=1e-300)

    def pair(wts):
        a = sum(wts.get(k, 1.0) * G[i] for i, k in enumerate(names) if k.startswith("pde"))
        b = sum(wts.get(k, 1.0) * G[i] for i, k in enumerate(names) if not k.startswith("pde"))
        na, nb = float(a.norm()), float(b.norm())
        return {
            "grad_pde": na,
            "grad_bc": nb,
            "cosine": float(a @ b / max(na * nb, 1e-300)),
            "cancellation": float((a + b).norm()) / max(na + nb, 1e-300),
        }

    return {
        "groups": names,
        "loss": L,
        "grad_norm": nrm.tolist(),
        "cosine": cos.tolist(),
        "unweighted": pair({}),
        "weighted": pair(w) if w else None,
    }


def diag_task(case: str, cfg: dict, i: int, dc: dict, cache_dir: str) -> dict:
    """Diagnostics of member i (own process; result cached on disk)."""
    key = {"tag": CODE_TAG, "case": case, "surrogate": cfg["surrogate"], "member": i, "dc": dc}
    return cached(cache_dir, "twod_diag", key, lambda: _diag(case, cfg, i, dc))[0]


def _diag(case: str, cfg: dict, i: int, dc: dict) -> dict:
    import csv

    from aerosynapse.thermal.pinn import diagnostics as D

    t0 = time.perf_counter()
    ad = adapter(case, cfg)
    name, prob, net = ad.members[i]
    s = ad.specs[i]
    n0 = ad.initial(i)
    p0 = next(net.parameters())
    # reload check: the rebuilt initial network on the training batch 0 reproduces the logged step-0 losses
    sm = s["sampling"]
    b0 = prob.sample(sm["n_pde"], sm["n_ic"], sm["n_bc"], sm["method"], int(s["seed"]) * 1_000_003, p0.device, p0.dtype)
    comp = {k: float(v.detach()) for k, v in prob.loss_components(n0, b0).items()}
    with open(ad.log_dirs[i] / "log.csv", encoding="utf-8") as fh:
        row0 = next(csv.DictReader(fh))
    reload = max(abs(comp[k] - float(row0[k])) / max(abs(float(row0[k])), 1e-300) for k in comp if k in row0)
    ck = torch.load(ad.log_dirs[i] / "checkpoint.pt", map_location="cpu", weights_only=False)
    w = (ck.get("balancer") or {}).get("w")
    pp, pj = dc["points"], dc["jacobian_points"]
    pts = prob.sample(int(pp["n_pde"]), 0, int(pp["n_bc"]), "sobol", int(dc["seed"]), p0.device, p0.dtype)
    ptj = prob.sample(int(pj["n_pde"]), 0, int(pj["n_bc"]), "sobol", int(dc["seed"]) + 1, p0.device, p0.dtype)
    out = {
        "case": case,
        "member": name,
        "n_params": int(D.flat(net).numel()),
        "reload_max_rel_diff": float(reload),
        "balancer_w": w,
    }
    Ks = {}
    for tag, model in (("init", n0), ("final", net)):
        e = {"conflict": _conflict(model, prob, pts, w)}
        J, lab = D.jacobian(model, prob, ptj)
        st = D.ntk_stats(J, lab)
        Ks[tag] = st.pop("_K")
        st.pop("pde_eigenvectors")
        e["ntk"] = st
        e["hessian"] = D.hessian_extremes(model, prob, pts)
        out[tag] = e
        print(f"  {case} {name} {tag}: {time.perf_counter() - t0:.0f} s", flush=True)
    out["ntk_drift"] = D.drift(Ks["init"], Ks["final"])
    if dc.get("hessian_full", True):
        out["final"]["hessian_spectrum"] = np.linalg.eigvalsh(D.hessian_full(net, prob, pts)).tolist()
    ls = dc["landscape"]
    rng = np.random.default_rng(int(ls["seed"]))
    d1, d2 = D.filter_normalised(net, rng), D.filter_normalised(net, rng)
    a = np.linspace(-float(ls["span"]), float(ls["span"]), int(ls["n"]))
    out["landscape"] = {"a": a.tolist(), "loss": D.landscape(net, prob, pts, d1, d2, a, a).tolist()}
    sI, LI = D.interpolate(net, prob, pts, D.flat(n0).detach(), D.flat(net).detach(), int(ls.get("n_line", 51)))
    out["interpolation"] = {"s": sI.tolist(), "loss": LI.tolist()}
    out["wall_s"] = time.perf_counter() - t0
    return out


# --------------------------------------------------------------------------- PINN side (main process)
def _point_batch(prob, base: dict, values: dict) -> dict:
    enc = torch.as_tensor(prob.encode(values), dtype=base["pde"].dtype, device=base["pde"].device)

    def put(X):
        X = X.clone()
        X[:, 2:] = enc
        return X

    return {"pde": put(base["pde"]), "walls": {k: put(v) for k, v in base["walls"].items()}}


def pinn_eval(ad, P: np.ndarray, n_wall: int, rp: dict) -> dict:
    """Per member: QoI at every row of P and the physics indicator (sqrt of the sum of mean squared residual groups)."""
    q = np.array([ad.pinn(prob, net, P, n_wall) for _, prob, net in ad.members])
    res = np.empty((len(ad.members), len(P)))
    for m, (_, prob, net) in enumerate(ad.members):
        p0 = next(net.parameters())
        base = prob.sample(int(rp["n_pde"]), 0, int(rp["n_bc"]), "sobol", int(rp["seed"]), p0.device, p0.dtype)
        for j, row in enumerate(P):
            r = prob.residual_vectors(net, _point_batch(prob, base, dict(zip(NAMES, row, strict=True))))
            res[m, j] = VA.physics_indicator({k: v.detach().cpu().numpy() for k, v in r.items()})["total"]
    return {"members": q, "residual": res}


def pinn_field(ad, x, y, values: dict) -> np.ndarray:
    from aerosynapse.thermal.pinn import problems2d as P2

    out = []
    for _, prob, net in ad.members:
        X = prob.grid_inputs(np.asarray(x), np.asarray(y), values, next(net.parameters()))
        out.append(P2._batched(prob, net, X, 0, ("theta",))["theta"].reshape(len(y), len(x)))
    return np.array(out)


# --------------------------------------------------------------------------- designs
def log_box(box: dict) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([np.log10(box[k][0]) for k in NAMES])
    hi = np.array([np.log10(box[k][1]) for k in NAMES])
    return lo, hi


def uq_points(box: dict, n: int, seed: int) -> np.ndarray:
    lo, hi = log_box(box)
    return 10 ** (lo + np.random.default_rng(seed).uniform(size=(n, 3)) * (hi - lo))


def ray_points(box: dict, distances: list[float], n_random: int, seed: int) -> tuple[np.ndarray, list[dict]]:
    """Along rays from the centre (log10 space): the 6 axis directions and n_random random ones. d = 0: halfway to the
    box surface; d > 0: the point whose largest distance outside the box (decades, any parameter) is d."""
    lo, hi = log_box(box)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    dirs = [np.eye(3)[i] * sg for i in range(3) for sg in (1.0, -1.0)]
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        v = rng.standard_normal(3)
        dirs.append(v / np.linalg.norm(v))
    P, meta = [], []
    for k, v in enumerate(dirs):
        a = np.abs(v)
        t_exit = np.min(h[a > 1e-12] / a[a > 1e-12])
        for d in distances:
            t = 0.5 * t_exit if d == 0 else np.min((d + h[a > 1e-12]) / a[a > 1e-12])
            P.append(10 ** (c + t * v))
            meta.append({"ray": k, "axis": k < 6, "d": float(d)})
    return np.array(P), meta


def slice_points(box: dict, pair: list[str], n: int, margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = log_box(box)
    i, j = NAMES.index(pair[0]), NAMES.index(pair[1])
    a = np.linspace(lo[i] - margin, hi[i] + margin, n)
    b = np.linspace(lo[j] - margin, hi[j] + margin, n)
    A, B = np.meshgrid(a, b)
    c = (lo + hi) / 2
    P = np.tile(10**c, (n * n, 1))
    P[:, i], P[:, j] = 10 ** A.ravel(), 10 ** B.ravel()
    return P, a, b


# --------------------------------------------------------------------------- analysis
def analyse_uq(y, M, levels, alpha, n_splits, seed) -> dict:
    ok = np.isfinite(y)
    y, M = y[ok], M[:, ok]
    mu, sd = UQ.moments(M)
    out = {"n": int(ok.sum()), "n_members": int(M.shape[0]), "gaussian": UQ.summary(mu, sd, y, levels)}
    out["rel_err_median"] = float(np.median(np.abs(mu - y) / np.abs(y)))
    out["rel_sd_median"] = float(np.median(sd / np.abs(mu)))
    rng = np.random.default_rng(seed)
    cov, qs = [], []
    for _ in range(n_splits):
        perm = rng.permutation(len(y))
        cal, te = perm[: len(y) // 2], perm[len(y) // 2 :]
        q = UQ.conformal_scale(mu[cal], sd[cal], y[cal], alpha)
        qs.append(q)
        cov.append(float(np.mean(np.abs(y[te] - mu[te]) <= q * np.maximum(sd[te], UQ.SD_FLOOR))))
    out["conformal"] = {
        "alpha": alpha,
        "target": 1 - alpha,
        "coverage_mean": float(np.mean(cov)),
        "coverage_p05_p95": [float(np.quantile(cov, 0.05)), float(np.quantile(cov, 0.95))],
        "q_median": float(np.median(qs)),
        "z_gauss": UQ.z_of(1 - alpha),
    }
    out["points"] = {"y": y.tolist(), "mu": mu.tolist(), "sd": sd.tolist()}
    return out


def analyse_field(th_solver, th_members, levels) -> dict:
    mu, sd = UQ.moments(th_members)
    e = np.abs(mu - th_solver)
    return {
        "rel_l2": float(np.linalg.norm(mu - th_solver) / np.linalg.norm(th_solver)),
        "coverage": {str(k): v for k, v in UQ.coverage(mu.ravel(), sd.ravel(), th_solver.ravel(), levels).items()},
        "spearman_sd_err": VA.spearman(sd.ravel(), e.ravel()),
        "rms_sd": float(np.sqrt(np.mean(sd**2))),
        "rms_err": float(np.sqrt(np.mean(e**2))),
    }


def analyse_extrap(rec: list[dict], tol: float, quantile: float, inrange: dict) -> dict:
    """Indicators: spread (relative), physics residual, combined = max of the two, each over its in-range quantile."""
    ok = [r for r in rec if np.isfinite(r["y"])]
    err = np.array([r["rel_err"] for r in ok])
    out = {"n": len(ok), "n_failed": len(rec) - len(ok), "tol": tol, "quantile": quantile, "indicators": {}}
    thr = {k: float(np.quantile(inrange[k], quantile)) for k in ("spread", "residual")}
    for k in ("spread", "residual", "combined"):
        if k == "combined":
            s = np.array([max(r["spread"] / thr["spread"], r["residual"] / thr["residual"]) for r in ok])
            t = 1.0
        else:
            s, t = np.array([r[k] for r in ok]), thr[k]
        det = VA.detection(s, err, tol, t)
        det["spearman"] = VA.spearman(s, err)
        fpr, tpr = roc(s, err > tol)
        det["roc"] = {"fpr": fpr, "tpr": tpr}
        out["indicators"][k] = det
    ds = sorted({r["d"] for r in ok})
    out["by_distance"] = [
        {
            "d": d,
            "rel_err_median": float(np.median([r["rel_err"] for r in ok if r["d"] == d])),
            "rel_err_max": float(np.max([r["rel_err"] for r in ok if r["d"] == d])),
            "spread_median": float(np.median([r["spread"] for r in ok if r["d"] == d])),
            "residual_median": float(np.median([r["residual"] for r in ok if r["d"] == d])),
            "n": sum(r["d"] == d for r in ok),
        }
        for d in ds
    ]
    return out


def roc(scores, bad) -> tuple[list[float], list[float]]:
    s, b = np.asarray(scores, float), np.asarray(bad, bool)
    if b.all() or (~b).all():
        return [], []
    order = np.argsort(-s)
    tp = np.concatenate([[0], np.cumsum(b[order])]) / b.sum()
    fp = np.concatenate([[0], np.cumsum(~b[order])]) / (~b).sum()
    return fp.tolist(), tp.tolist()


# --------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--parts", default="uq,extrapolation,diagnostics")
    ap.add_argument("--cases", default=None, help="comma-separated subset of twod_study.cases")
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    spec = SP.load_spec(a.config or None, a.profile, SP.parse_set(a.sets))
    ts = spec["twod_study"]
    custom = bool(
        a.config
        or a.sets
        or a.tag != "default"
        or a.profile != "smoke"
        or a.cases
        or a.parts != "uq,extrapolation,diagnostics"
    )
    odir = ROOT / "results" / f"p6_6b_2d_{a.tag}"
    table = odir / "twod.md" if custom else ROOT / "docs" / "tables" / "phase6_6b_2d.md"
    if a.figures_only:
        met = json.loads((odir / "metrics.json").read_text(encoding="utf-8"))
        met["figures"] = figures(met, a.tag, odir, docs=not custom)
        report(met, table, a.tag)
        print(f"wrote {table}")
        return
    parts = {p.strip() for p in a.parts.split(",")}
    cases = [c.strip() for c in a.cases.split(",")] if a.cases else list(ts["cases"])
    cache_dir = str(ROOT / ts["cache_dir"])
    ex_cfg = {"isolate": True, **ts.get("execution", {})}
    with Experiment(f"p6_6b_2d_{a.tag}", spec) as ex:
        met = ex.metrics
        met["cases"] = {}
        jobs, labels = [], []
        designs = {}
        for case in cases:
            cfg = ts["cases"][case]
            t0 = time.perf_counter()
            ad = adapter(case, cfg)
            kw = solver_kw(case, ad, cfg)
            d = {"box": ad.box, "members": [m[0] for m in ad.members], "qoi": ad.qoi}
            blocks = {}
            if "uq" in parts:
                u = ts["uq"]
                blocks["uq"] = uq_points(ad.box, int(u["n_points"]), int(u["seed"]))
                from aerosynapse.thermal.experiments.p6_4_extended import point_values

                d["field_points"] = [point_values(p, ad.box) for p in u["field_points"]]
            if "extrapolation" in parts:
                x = ts["extrapolation"]
                P, meta = ray_points(ad.box, [float(v) for v in x["distances"]], int(x["n_random"]), int(x["seed"]))
                blocks["rays"], d["ray_meta"] = P, meta
                sl = x["slice"]
                Ps, sa, sb = slice_points(ad.box, sl["pair"], int(sl["n"]), float(sl["margin"]))
                blocks["slice"], d["slice_grid"] = Ps, {"pair": sl["pair"], "a": sa.tolist(), "b": sb.tolist()}
            # PINN side now (main process), solver side in workers
            rp = ts["residual_points"]
            for blk, P in blocks.items():
                pe = pinn_eval(ad, P, int(ts["n_wall"]), rp)
                d[blk] = {"P": P.tolist(), "members": pe["members"].tolist(), "residual": pe["residual"].tolist()}
                nch = max(1, int(np.ceil(len(P) / int(ts.get("chunk", 16)))))
                for k, ch in enumerate(np.array_split(np.arange(len(P)), nch)):
                    jobs.append((MOD, "solver_chunk", dict(case=case, P=P[ch].tolist(), kw=kw, cache_dir=cache_dir)))
                    labels.append((case, blk, k, ch.tolist()))
            for k, fp in enumerate(d.get("field_points", [])):
                jobs.append(
                    (
                        MOD,
                        "solver_chunk",
                        dict(case=case, P=[[fp[n] for n in NAMES]], kw=kw, cache_dir=cache_dir, fields=True),
                    )
                )
                labels.append((case, "field", k, [0]))
            if "diagnostics" in parts:
                for i in range(len(ad.members)):
                    jobs.append(
                        (MOD, "diag_task", dict(case=case, cfg=cfg, i=i, dc=ts["diagnostics"], cache_dir=cache_dir))
                    )
                    labels.append((case, "diag", i, None))
            d["pinn_s"] = time.perf_counter() - t0
            designs[case] = (ad, d)
            print(f"{case}: PINN side {d['pinn_s']:.0f} s", flush=True)
        print(f"{len(jobs)} tasks", flush=True)
        outs = run_tasks(
            jobs, ex_cfg, lambda i, r, sec: print(f"[{i + 1}/{len(jobs)}] {labels[i][:3]}: {sec:.0f} s", flush=True)
        )
        for case, (ad, d) in designs.items():
            for blk in ("uq", "rays", "slice"):
                if blk not in d:
                    continue
                y = np.full(len(d[blk]["P"]), np.nan)
                errs = []
                for (c, b, _, idx), o in zip(labels, outs, strict=True):
                    if c == case and b == blk:
                        y[idx] = [r["qoi"] for r in o]
                        errs += [r["error"] for r in o if not r["ok"] and "error" in r]
                d[blk]["y"] = y.tolist()
                d[blk]["solver_errors"] = errs
            fl = [o[0] for (c, b, _, _), o in zip(labels, outs, strict=True) if c == case and b == "field"]
            d["fields"] = []
            for fp, fs in zip(d.get("field_points", []), fl, strict=True):
                if not fs["ok"]:
                    continue
                thm = pinn_field(ad, fs["x"], fs["y"], fp)
                d["fields"].append(
                    {"values": fp, "stats": analyse_field(np.asarray(fs["theta"]), thm, ts["uq"]["levels"])}
                    | {
                        "x": fs["x"],
                        "y": fs["y"],
                        "theta": fs["theta"],
                        "theta_mean": thm.mean(0).tolist(),
                        "theta_sd": thm.std(0, ddof=1).tolist(),
                    }
                )
            d["diagnostics"] = [o for (c, b, _, _), o in zip(labels, outs, strict=True) if c == case and b == "diag"]
            met["cases"][case] = d
        met["analysis"] = analyse(met, ts)
        met["git"] = git_hash()
        save_fields(met, ex.dir)
        met["figures"] = figures(met, a.tag, ex.dir, docs=not custom)
        report(met, table, a.tag)
        met["table"] = str(table.relative_to(ROOT))
    print(f"wrote {table}")


def save_fields(met: dict, odir: Path) -> None:
    """Fields -> <odir>/fields.npz (git-ignored); the metrics keep the statistics."""
    arrs = {}
    for case, d in met["cases"].items():
        for k, f in enumerate(d.get("fields", [])):
            for n in ("x", "y", "theta", "theta_mean", "theta_sd"):
                arrs[f"{case}_{k}_{n}"] = np.asarray(f.pop(n))
    np.savez_compressed(odir / "fields.npz", **arrs)


def analyse(met: dict, ts: dict) -> dict:
    out = {}
    for case, d in met["cases"].items():
        an = {}
        if "uq" in d:
            u = ts["uq"]
            an["uq"] = analyse_uq(
                np.asarray(d["uq"]["y"]),
                np.asarray(d["uq"]["members"]),
                [float(v) for v in u["levels"]],
                float(u["conformal_alpha"]),
                int(u["n_splits"]),
                int(u["seed"]),
            )
        if "rays" in d:
            x = ts["extrapolation"]

            def recs(blk, meta=None, d=d):
                M, y, R = np.asarray(d[blk]["members"]), np.asarray(d[blk]["y"]), np.asarray(d[blk]["residual"])
                mu, sd = M.mean(0), M.std(0, ddof=1)
                return [
                    {
                        "y": float(y[j]),
                        "rel_err": float(abs(mu[j] - y[j]) / abs(y[j])) if np.isfinite(y[j]) else float("nan"),
                        "spread": float(sd[j] / abs(mu[j])),
                        "residual": float(R[:, j].mean()),
                        "d": (meta[j]["d"] if meta else 0.0),
                    }
                    for j in range(len(y))
                ]

            rr = recs("rays", d["ray_meta"])
            inr = [r for r in rr if r["d"] == 0] + (recs("uq") if "uq" in d else [])
            inrange = {k: np.array([r[k] for r in inr]) for k in ("spread", "residual")}
            an["extrapolation"] = analyse_extrap(rr, float(x["tol"]), float(x["quantile"]), inrange)
            an["extrapolation"]["records"] = rr
            sl = recs("slice")
            an["slice"] = {k: [r[k] for r in sl] for k in ("rel_err", "spread", "residual", "y")}
        out[case] = an
    return out


def _f(v, f=".3g") -> str:
    """Number or "—" (None; non-finite values, which the metrics file may hold as strings such as "inf")."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    return format(v, f) if np.isfinite(v) else "—"


def report(met: dict, path: Path, tag: str) -> None:
    L = [
        f"# Step 6.6b: Phase 6 analyses on the 2D networks — `{tag}` (auto-generated by experiments/p6_6b_2d.py)",
        "",
        "Surrogates: the saved parametric 2D PINNs (Graetz channel: Step 4.7 ensemble of seeds; cavity: Step 4.5a "
        "variants). Reference: the verified 2D FV solver at the same parameters (solutions cached on disk).",
        "",
    ]
    for case, d in met["cases"].items():
        an = met["analysis"].get(case, {})
        L += [f"## {case} (QoI {d['qoi']}; members: {', '.join(d['members'])})", ""]
        box = ", ".join(f"{k} [{v[0]:.3g}, {v[1]:.3g}]" for k, v in d["box"].items())
        L.append(f"Training box: {box}.")
        L.append("")
        if "uq" in an:
            u = an["uq"]
            g = u["gaussian"]
            c = u["conformal"]
            L += [
                f"### UQ ({u['n']} random points in the box, {u['n_members']} members)",
                "",
                "| median rel. error | median rel. spread | dispersion (rms z) | ECE | CRPS | Spearman(spread, error) | coverage "
                + " / ".join(g["coverage"])
                + f" | conformal coverage (target {c['target']:.2f}, {c['alpha']} miscoverage) | conformal q (Gaussian z) |",
                "|---|---|---|---|---|---|---|---|---|",
                f"| {u['rel_err_median']:.3g} | {u['rel_sd_median']:.3g} | {g['dispersion']:.3g} | {g['ece']:.3g} | {g['crps']:.3g} | "
                f"{g['rank_corr']:.2f} | "
                + " / ".join(f"{v:.2f}" for v in g["coverage"].values())
                + f" | {c['coverage_mean']:.3f} [{c['coverage_p05_p95'][0]:.2f}, {c['coverage_p05_p95'][1]:.2f}] | "
                f"{_f(c['q_median'])} ({c['z_gauss']:.2f}) |",
                "",
            ]
            if d.get("fields"):
                L += [
                    "Field level (theta on the solver grid):",
                    "",
                    "| point (Re, Pr, Bi) | rel. L2 error | rms error / rms spread | Spearman(spread, |error|) | coverage "
                    + " / ".join(d["fields"][0]["stats"]["coverage"])
                    + " |",
                    "|---|---|---|---|---|",
                ]
                for f in d["fields"]:
                    s, v = f["stats"], f["values"]
                    L.append(
                        f"| ({v['Re']:.3g}, {v['Pr']:.3g}, {v['Bi']:.3g}) | {s['rel_l2']:.3g} | {s['rms_err']:.3g} / {s['rms_sd']:.3g} | "
                        f"{s['spearman_sd_err']:.2f} | " + " / ".join(f"{x:.2f}" for x in s["coverage"].values()) + " |"
                    )
                L.append("")
        if "extrapolation" in an:
            x = an["extrapolation"]
            L += [
                f"### Extrapolation ({x['n']} ray points; {x['n_failed']} solver failures excluded; bad = rel. error > {x['tol']})",
                "",
                "| d (decades outside) | n | median / max rel. error | median rel. spread | median residual |",
                "|---|---|---|---|---|",
            ]
            for b in x["by_distance"]:
                L.append(
                    f"| {b['d']:g} | {b['n']} | {b['rel_err_median']:.3g} / {b['rel_err_max']:.3g} | {b['spread_median']:.3g} | {b['residual_median']:.3g} |"
                )
            L += [
                "",
                f"| indicator | threshold (in-range q{x['quantile']}) | TPR | FPR | AUC | Spearman(indicator, error) |",
                "|---|---|---|---|---|---|",
            ]
            for k, det in x["indicators"].items():
                L.append(
                    f"| {k} | {_f(det['threshold'])} | {_f(det['tpr'], '.2f')} | {_f(det['fpr'], '.2f')} | {_f(det['auc'], '.2f')} | {_f(det['spearman'], '.2f')} |"
                )
            L.append("")
        if d.get("diagnostics"):
            L += [
                "### Training diagnostics (fixed batch; init = rebuilt with the training seed)",
                "",
                "| member | reload check (step-0 losses, max rel. diff) | loss init → final | PDE vs BC cosine (unweighted) init → final | cancellation | weighted cosine (final) | NTK eff. rank init → final | NTK drift | Hessian max / min (final) |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for r in d["diagnostics"]:
                i, f = r["init"], r["final"]
                li, lf = sum(i["conflict"]["loss"].values()), sum(f["conflict"]["loss"].values())
                wc = f["conflict"]["weighted"]
                h = f["hessian"]
                L.append(
                    f"| {r['member']} | {r['reload_max_rel_diff']:.1e} | {li:.3g} → {lf:.3g} | {i['conflict']['unweighted']['cosine']:+.2f} → "
                    f"{f['conflict']['unweighted']['cosine']:+.2f} | {i['conflict']['unweighted']['cancellation']:.3f} → "
                    f"{f['conflict']['unweighted']['cancellation']:.3f} | {_f(wc['cosine'] if wc else None, '+.2f')} | "
                    f"{i['ntk']['effective_rank']:.3g} → {f['ntk']['effective_rank']:.3g} | {r['ntk_drift']:.3g} | "
                    f"{_f(h['max'])} / {_f(h['min'])} |"
                )
            L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(met: dict, tag: str, odir, docs: bool) -> list[str]:
    F.style()
    fz = Path(odir) / "fields.npz"
    fields = dict(np.load(fz)) if fz.exists() else {}
    paths = []
    cases = met["cases"]
    an = met["analysis"]
    if any("uq" in a for a in an.values()):
        paths += save_fig(F.uq(cases, an), f"p6_6b_uq_{tag}", odir, docs=docs)
        if fields:
            paths += save_fig(F.uq_field3d(cases, fields), f"p6_6b_uq_field3d_{tag}", odir, docs=docs)
    if any("extrapolation" in a for a in an.values()):
        paths += save_fig(F.extrapolation(cases, an), f"p6_6b_extrapolation_{tag}", odir, docs=docs)
        paths += save_fig(F.slice3d(cases, an), f"p6_6b_slice3d_{tag}", odir, docs=docs)
    if any(d.get("diagnostics") for d in cases.values()):
        paths += save_fig(F.conflict(cases), f"p6_6b_conflict_{tag}", odir, docs=docs)
        paths += save_fig(F.spectra(cases), f"p6_6b_spectra_{tag}", odir, docs=docs)
        paths += save_fig(F.landscapes3d(cases), f"p6_6b_landscape3d_{tag}", odir, docs=docs)
    return [str(p) for p in paths]


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"done in {time.perf_counter() - t0:.0f} s")
