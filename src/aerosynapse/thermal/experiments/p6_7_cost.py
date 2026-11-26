"""Step 6.7: cost analysis — training time, inference latency, hardware stated, break-even number of queries vs the
solver, for the project's saved PARAMETRIC networks (2D: Graetz channel ensemble of Step 4.7, heated cavity variants of
Step 4.5a; 1D: the Step 6.2 deep ensembles of A_steady and B_slab).

Measured here, one process at a time (execution.workers 1, nothing else running), every task cached in
<results>/tasks/ and reusable with --reuse:
  hardware      CPU, cores, memory, OS, library versions, torch threads, GPU (none)
  latency       PINN inference per query vs batch size (parameter rows per call) and torch threads: the QoI (wall
                means of first derivatives, autograd) and the full field on the solver grid (2D); the field on the
                evaluation grid (1D); median of repeats after an untimed warm-up; extra memory of the largest batch
  retrain       chosen members re-trained ALONE with the same seed and settings: clean training time (the logged times
                were measured with two trainings in parallel), the cost of the solver snapshot data the hybrid Graetz
                surrogate consumed, and a reproducibility check against the saved weights and the logged loss
  solver        2D: re-time the solver at the case grid now (spot check of the stored Step 6.4b ladder); 1D: FD and
                Chebyshev error / time ladders vs the analytic truth (best of repeats)
Re-used (stored data, re-considered): the Step 6.4b grid ladders (solver time, memory and QoI error per grid, one fresh
process per grid), the PINN QoI errors of Steps 6.6b (2D) and 6.2 (1D), the logged training times, and the wall times
of the Step 6.4 Sobol study (a project-level account: which tool was cheaper for the studies actually run).
Analysis: offline cost (data + training, ensemble = all members), per-query cost (PINN ensemble vs solver at the case
grid and at the cheapest grid as accurate as the PINN), break-even queries N* = offline / (t_solver − t_pinn), cost
curves, accuracy-cost trade-off, and a decision map over (number of queries, required accuracy).

    python experiments/p6_7_cost.py [--config my.yaml] [--set key=value] [--tag name] [--cases D_graetz,A_steady]
                                    [--parts hardware,latency,retrain,solver] [--reuse] [--figures-only]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash
from aerosynapse.thermal.utils.isolate import run_tasks
from aerosynapse.thermal.viz import cost_figs as F
from aerosynapse.thermal.viz.style import save_fig

MOD = "aerosynapse.thermal.experiments.p6_7_cost"
NAMES = ["Re", "Pr", "Bi"]
PARTS = "hardware,latency,retrain,solver"


# --------------------------------------------------------------------------- shared helpers
def adapter(case: str, cfg: dict):
    from aerosynapse.thermal.experiments.p6_4_sensitivity import ADAPTERS

    return ADAPTERS[case]({"surrogate": cfg["surrogate"]})


def members_1d(case: str, cfg: dict) -> tuple[list, list, dict]:
    """(problem, net) of every saved ensemble member of the 6.2 run, the member records and the range."""
    from aerosynapse.thermal.experiments import p6_2_uq as U2

    run = ROOT / cfg["run"]
    met = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    import yaml

    saved = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    rng = saved["uq_study"]["problems"][case]["range"]
    recs = [m for m in met["members"] if m["problem"] == case and m["kind"] == "ensemble"]
    out = []
    for m in recs:
        spec = U2.member_spec(None, saved.get("profile", "smoke"), {}, case, rng, m["over"], m["seed"])
        out.append(U2.load_member(spec, case, Path(m["dir"]) / "checkpoint.pt"))
    return out, recs, {"range": rng, "saved": saved}


def _timeit(fn, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"median_s": float(np.median(ts)), "min_s": float(np.min(ts)), "repeats": ts}


def box_points(box: dict, n: int, seed: int) -> np.ndarray:
    lo = np.array([np.log(box[k][0]) for k in NAMES])
    hi = np.array([np.log(box[k][1]) for k in NAMES])
    return np.exp(lo + np.random.default_rng(seed).uniform(size=(n, 3)) * (hi - lo))


def solver_grid_xy(case: str, grid: int, Lx: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    if case == "D_graetz":  # grid = cells per unit height; channel Lx x 1
        nx, ny = int(round(Lx * grid)), int(grid)
        return (np.arange(nx) + 0.5) * Lx / nx, (np.arange(ny) + 0.5) / ny
    return (np.arange(grid) + 0.5) / grid, (np.arange(grid) + 0.5) / grid


# --------------------------------------------------------------------------- tasks (own process each)
def hardware_task() -> dict:
    import scipy

    from aerosynapse.thermal.utils.costmodel import machine

    m = machine()
    return m | {
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER") or platform.machine(),
        "architecture": os.environ.get("PROCESSOR_ARCHITECTURE") or platform.machine(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch_threads_default": torch.get_num_threads(),
        "cuda": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dtype": "float64 (all networks)",
    }


def latency_task(case: str, cfg: dict, lc: dict, threads: int) -> dict:
    """Per-query inference time of every member vs batch size (median of repeats after warm-up)."""
    from aerosynapse.thermal.utils.costmodel import MemorySampler

    if threads > 0:
        torch.set_num_threads(int(threads))
    out = {"case": case, "threads": torch.get_num_threads(), "batches": [], "field": None}
    warm, rep = int(lc["warmup"]), int(lc["repeats"])
    if cfg["dim"] == 2:
        from aerosynapse.thermal.pinn import problems2d as P2

        ad = adapter(case, cfg)
        P = box_points(ad.box, max(int(b) for b in lc["batches"]), int(lc.get("seed", 0)))
        nw = int(lc["n_wall"])
        for B in lc["batches"]:
            per = [
                _timeit(lambda prob=prob, net=net, B=B: ad.pinn(prob, net, P[: int(B)], nw, chunk=int(B)), warm, rep)
                for _, prob, net in ad.members
            ]
            out["batches"].append(
                {
                    "batch": int(B),
                    "member_s": [p["median_s"] for p in per],
                    "per_query_member_s": [p["median_s"] / int(B) for p in per],
                }
            )
        with MemorySampler() as ms:
            for _, prob, net in ad.members:
                ad.pinn(prob, net, P, nw, chunk=len(P))
        out["memory_extra_bytes_largest_batch"] = int(ms.extra)
        x, y = solver_grid_xy(case, int(cfg["solver_grid"]))
        vals = dict(zip(NAMES, P[0], strict=True))

        def field(prob, net):
            X = prob.grid_inputs(x, y, vals, next(net.parameters()))
            P2._batched(prob, net, X, 0, ("u", "v", "theta"))  # autograd needed: u, v = derivatives of psi

        per = [_timeit(lambda prob=prob, net=net: field(prob, net), warm, rep) for _, prob, net in ad.members]
        out["field"] = {"n_points": int(len(x) * len(y)), "member_s": [p["median_s"] for p in per]}
        out["n_wall"] = nw
    else:
        from aerosynapse.thermal.pinn.metrics import predict_grid

        nets, recs, info = members_1d(case, cfg)
        rng = info["range"]
        nx = int(lc["nx_1d"])
        b1 = [int(b) for b in lc.get("batches_1d") or lc["batches"]]  # 1D: one call per query (no batching gain)
        vals = np.exp(np.random.default_rng(0).uniform(np.log(rng[0]), np.log(rng[1]), max(b1)))
        for B in b1:
            per = []
            for prob, net in nets:
                xg = np.linspace(0.0, 1.0, nx)
                tg = (prob.T * np.arange(1, int(lc["nt_1d"]) + 1) / int(lc["nt_1d"])) if prob.has_time else None

                def run(prob=prob, net=net, B=B, xg=xg, tg=tg):
                    for v in vals[: int(B)]:
                        predict_grid(net, prob, xg, tg, float(v))

                per.append(_timeit(run, warm, rep))
            out["batches"].append(
                {
                    "batch": int(B),
                    "member_s": [p["median_s"] for p in per],
                    "per_query_member_s": [p["median_s"] / int(B) for p in per],
                }
            )
        out["field"] = {"n_points": int(nx * (int(lc["nt_1d"]) if nets[0][0].has_time else 1))}
        out["memory_extra_bytes_largest_batch"] = None
    out["n_members"] = len(out["batches"][0]["member_s"])
    return out


def retrain_task(case: str, cfg: dict, i: int, out_dir: str) -> dict:
    """Member i trained again ALONE with its seed and settings: clean training time, data cost, reproducibility."""
    import csv

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)  # the Trainer resumes from an existing checkpoint: always start clean
    out_dir.mkdir(parents=True)
    rec = {"case": case, "member_index": i}
    th = train_threads(case, cfg)
    if th > 0:
        torch.set_num_threads(th)
    rec["threads"] = torch.get_num_threads()
    if cfg["dim"] == 2:
        ad = adapter(case, cfg)
        name, _, saved = ad.members[i]
        s0 = ad.specs[i]
        rec["member"] = name
        if case == "D_graetz":
            from aerosynapse.thermal.experiments import p4_6_hybrid as HY
            from aerosynapse.thermal.experiments.p4_5b_channel import reference_task
            from aerosynapse.thermal.pinn import channel2d as CH
            from aerosynapse.thermal.pinn import hybrid as H

            data = s0.get("ablation_data")
            mode = "hybrid" if data else "data_free"
            tpts = HY.train_design_points(ad.base, ad.case)[: int(data["n_snapshots"])] if data else []
            t0 = time.perf_counter()
            snaps = [(p, reference_task(p["values"], ad.pcfg)) for p in tpts]
            rec["data_s"] = time.perf_counter() - t0
            rec["n_snapshots"] = len(snaps)
            rec["snapshot_s"] = [float(r["wall_s"]) for _, r in snaps]
            hs = s0["hybrid_study"]
            form = SP.deep_merge(s0["formulation"], hs["formulation"])
            prob = CH.build_channel(
                ad.case, s0["problems2d"][ad.case], form, ad.design["dims"], ad.design["fixed"], ad.design["u_lim"]
            )
            net = HY._net(s0, prob)
            terms, _ = HY._user_terms(s0, ad.case, prob, net, mode)
            if mode != "data_free":
                d, pa = hs["data"], hs["parametric"]
                sp = [
                    (
                        H.snapshot_points(
                            r, int(pa["n_data"]), d["fields"], d["placement"], float(d["noise"]), int(d["seed"]) + k
                        ),
                        p["values"],
                    )
                    for k, (p, r) in enumerate(snaps)
                ]
                terms = [H.snapshot_term("snap", prob, sp, next(net.parameters())), *terms]
            res, wall = HY._train(s0, prob, net, mode, terms, out_dir, "p6_7")
        else:
            from aerosynapse.thermal.experiments import p4_5_parametric as P45
            from aerosynapse.thermal.pinn import param2d as PP
            from aerosynapse.thermal.pinn import userdata as U
            from aerosynapse.thermal.pinn.architectures import build_network
            from aerosynapse.thermal.pinn.trainer import Trainer

            s = s0
            ps = s["param_study"]
            dev, dtype = P45._dev(s)
            form = SP.deep_merge(s["formulation"], ps["formulation"])
            prob = PP.build_param_cavity(
                s["problems2d"]["C_cavity"], form, ad.design["dims"], ad.design["fixed"], ad.design["u_lim"]
            )
            torch.manual_seed(int(s["seed"]))
            net = build_network(s["network"], list(prob.lo), list(prob.hi), prob.out_dim).to(device=dev, dtype=dtype)
            terms, _ = U.attach(
                s, {"problem": "C_cavity", "data": ps.get("data") or []}, prob, dev, dtype, int(s["seed"])
            )
            tr = Trainer(net, prob, s, out_dir, tag="p6_7", data_terms=terms)
            prob.fit_normalisation(net, tr.batch(0))
            t0 = time.perf_counter()
            res = tr.run()
            wall = time.perf_counter() - t0
            rec["data_s"], rec["n_snapshots"] = 0.0, 0
        rec["train_s"] = float(wall)
        rec["status"] = res.get("status") if isinstance(res, dict) else None
        new = net.state_dict()
        rec["max_abs_weight_diff"] = float(max((new[k] - saved.state_dict()[k]).abs().max() for k in new))
        logs = {"saved": ad.log_dirs[i] / "log.csv", "new": out_dir / "log.csv"}
    else:
        from aerosynapse.thermal.experiments import p6_2_uq as U2

        nets, recs, info = members_1d(case, cfg)
        m = recs[i]
        rng = info["range"]
        rec["member"] = m["name"]
        mid = float(np.sqrt(rng[0] * rng[1]))
        r = U2.train_task(
            None, info["saved"].get("profile", "smoke"), {}, case, rng, m["over"], m["seed"], str(out_dir), [mid]
        )
        rec["train_s"], rec["data_s"], rec["n_snapshots"], rec["status"] = float(r["wall_s"]), 0.0, 0, r["status"]
        new = torch.load(out_dir / "checkpoint.pt", weights_only=False)["model"]
        old = torch.load(Path(m["dir"]) / "checkpoint.pt", weights_only=False)["model"]
        rec["max_abs_weight_diff"] = float(max((new[k] - old[k]).abs().max() for k in new))
        logs = {"saved": Path(m["dir"]) / "log.csv", "new": out_dir / "log.csv"}
    for k, f in logs.items():
        if f.exists():
            with open(f, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            rec[f"final_total_{k}"] = float(rows[-1]["total"])
            rec[f"logged_wall_{k}"] = float(rows[-1]["wall_s"])
    return rec


def train_threads(case: str, cfg: dict) -> int:
    """torch threads of the ORIGINAL training (execution.threads_per_worker of its saved config; 0 = torch default)."""
    if cfg["dim"] == 2:
        ex = adapter(case, cfg).base.get("execution") or {}
    else:
        import yaml

        saved = yaml.safe_load((ROOT / cfg["run"] / "config.yaml").read_text(encoding="utf-8"))
        ex = (saved.get("uq_study") or {}).get("execution") or saved.get("execution") or {}
    return int(ex.get("threads_per_worker", 0) or 0)


def solver2d_task(case: str, cfg: dict, point: list, repeats: int) -> dict:
    """Spot check: the solver at the case grid, one untimed warm-up and `repeats` timed solves (this machine, now)."""
    ad = adapter(case, cfg)
    vals = {**ad.fixed, **dict(zip(NAMES, map(float, point), strict=True))}
    if case == "D_graetz":
        from aerosynapse.thermal.pinn import channel2d as CH

        def solve():
            return -float(CH.solve_channel(vals, ad.pcfg, int(cfg["solver_grid"]), 1e-10, "quick")["q_bottom"])
    else:
        from aerosynapse.thermal.pinn.ref2d import solve_steady_decoupled

        def solve():
            return float(solve_steady_decoupled(vals, ad.cond, int(cfg["solver_grid"]), 1e-10, ad.scheme)["nu_top"])

    q = solve()
    t = _timeit(solve, 0, int(repeats))
    return {"case": case, "grid": int(cfg["solver_grid"]), "point": vals, "qoi": q} | t


def solver1d_task(case: str, cfg: dict, so: dict) -> dict:
    """FD and Chebyshev ladders (error vs the analytic truth, best-of-repeats time) at the middle of the range."""
    from aerosynapse.thermal.experiments.p6_1_scaling import solver_curves

    _, _, info = members_1d(case, cfg)
    rng = info["range"]
    spec = SP.load_spec(None, info["saved"].get("profile", "smoke"), {})
    mid = float(np.sqrt(rng[0] * rng[1]))
    return {"case": case, "value": mid} | solver_curves(spec, case, mid, so)


# --------------------------------------------------------------------------- analysis
def ladder_2d(lad: dict, case: str) -> list[dict]:
    """Per grid: median solve time and memory over the ladder points, median relative QoI error vs Richardson."""
    g = lad["grid"][case]
    conv = {json.dumps(c["point"], sort_keys=True): c for c in g["convergence"]}
    out = []
    for grid in sorted({r["grid"] for r in g["rows"]}):
        rows = [r for r in g["rows"] if r["grid"] == grid]
        errs = []
        for r in rows:
            c = conv.get(json.dumps(r["point"], sort_keys=True))
            if c and c["gci"].get("f_ext") is not None:
                errs.append(abs(r["qoi"] - c["gci"]["f_ext"]) / abs(c["gci"]["f_ext"]))
        out.append(
            {
                "grid": int(grid),
                "cells": int(rows[0]["cells"]),
                "time_s": float(np.median([r["wall_s"] for r in rows])),
                "mem_bytes": float(np.median([r["mem_bytes"] for r in rows])),
                "rel_err": float(np.median(errs)) if errs else None,
            }
        )
    return out


def break_even(offline: float, t_solver: float, t_pinn: float) -> float | None:
    return offline / (t_solver - t_pinn) if t_solver > t_pinn else None


def analyse(met: dict, cs: dict) -> dict:
    out = {}
    lad = None
    lf = ROOT / cs["ladder"] / "metrics.json"
    if lf.exists():
        lad = json.loads(lf.read_text(encoding="utf-8"))
    for case, d in met["cases"].items():
        cfg = cs["cases"][case]
        a = {"dim": cfg["dim"]}
        # offline cost: logged (parallel) and clean (re-trained alone) training times, data
        logged = [float(v) for v in d["train_logged_s"]]
        a["train_logged_s"] = logged
        rt = d.get("retrain") or []
        a["train_clean_s"] = [r["train_s"] for r in rt]
        a["data_s"] = float(np.mean([r["data_s"] for r in rt])) if rt else 0.0
        M = len(logged)
        # contention = logged / alone for the SAME member; the other members are scaled by the measured factor
        ratios = [logged[r["member_index"]] / r["train_s"] for r in rt]
        a["contention"] = float(np.mean(ratios)) if ratios else None
        f = a["contention"] or 1.0
        clean = {r["member_index"]: r["train_s"] for r in rt}
        a["train_clean_est_s"] = [clean.get(j, logged[j] / f) for j in range(M)]
        a["offline_member_s"] = a["data_s"] + float(np.mean(a["train_clean_est_s"]))
        a["offline_ensemble_s"] = a["data_s"] + float(np.sum(a["train_clean_est_s"]))  # the snapshots are shared
        a["offline_ensemble_logged_s"] = a["data_s"] + float(np.sum(logged))
        a["n_members"] = M
        # inference per query: latency = batch 1, throughput = the BEST measured (batch, threads) setting (recorded)
        lat = {r["threads"]: r for r in d.get("latency") or []}
        if lat:
            t1 = lat[min(lat)]
            bs = t1["batches"]
            combos = [(th, b) for th, r in lat.items() for b in r["batches"]]
            th_l, b_l = min(
                ((th, b) for th, b in combos if b["batch"] == 1), key=lambda z: np.sum(z[1]["per_query_member_s"])
            )
            th_q, b_q = min(combos, key=lambda z: np.sum(z[1]["per_query_member_s"]))
            a["pinn_latency_member_s"] = float(np.median(b_l["per_query_member_s"]))
            a["pinn_latency_ensemble_s"] = float(np.sum(b_l["per_query_member_s"]))
            a["pinn_latency_threads"] = int(th_l)
            a["pinn_query_member_s"] = float(np.median(b_q["per_query_member_s"]))
            a["pinn_query_ensemble_s"] = float(np.sum(b_q["per_query_member_s"]))
            a["pinn_query_setting"] = {"threads": int(th_q), "batch": int(b_q["batch"])}
            a["threads"] = {
                str(k): [float(np.sum(b["per_query_member_s"])) for b in v["batches"]] for k, v in lat.items()
            }
            a["batches"] = [b["batch"] for b in bs]
            if t1.get("field") and t1["field"].get("member_s"):
                a["pinn_field_ensemble_s"] = float(np.sum(t1["field"]["member_s"]))
        a["pinn_error"] = d.get("pinn_error")
        # solver per query
        if cfg["dim"] == 2 and lad:
            L = ladder_2d(lad, case)
            a["ladder"] = L
            g0 = next((r for r in L if r["grid"] == int(cfg["solver_grid"])), None)
            a["solver_grid"] = int(cfg["solver_grid"])
            a["solver_s"] = g0["time_s"] if g0 else None
            a["solver_err"] = g0["rel_err"] if g0 else None
            a["solver_mem_bytes"] = g0["mem_bytes"] if g0 else None
            chk = d.get("solver_check")
            if chk:
                a["solver_check_s"] = chk["median_s"]
            ok = [
                r
                for r in L
                if r["rel_err"] is not None and a["pinn_error"] is not None and r["rel_err"] <= a["pinn_error"]
            ]
            eq = min(ok, key=lambda r: r["time_s"]) if ok else None
            a["equal_accuracy"] = eq
        elif cfg["dim"] == 1 and d.get("solver1d"):
            s1 = d["solver1d"]
            L = [
                {"kind": k, "N": r["N"], "time_s": r["wall_s"], "rel_err": r["rel_l2"]}
                for k in ("fd", "spectral")
                for r in s1[k]
            ]
            a["ladder"] = L
            fd = [r for r in L if r["kind"] == "fd"]
            g0 = next((r for r in fd if r["N"] == int(cfg["solver_N"])), fd[-1])
            a["solver_grid"], a["solver_s"], a["solver_err"] = g0["N"], g0["time_s"], g0["rel_err"]
            ok = [r for r in L if a["pinn_error"] is not None and r["rel_err"] <= a["pinn_error"]]
            a["equal_accuracy"] = min(ok, key=lambda r: r["time_s"]) if ok else None
        tq = a.get("pinn_query_ensemble_s")
        if tq is not None and a.get("solver_s") is not None:
            a["break_even_ensemble"] = break_even(a["offline_ensemble_s"], a["solver_s"], tq)
            a["break_even_member"] = break_even(a["offline_member_s"], a["solver_s"], a["pinn_query_member_s"])
            eq = a.get("equal_accuracy")
            a["break_even_equal_accuracy"] = break_even(a["offline_ensemble_s"], eq["time_s"], tq) if eq else None
            a["speedup_per_query"] = a["solver_s"] / tq
        out[case] = a
    # project account: the Step 6.4 Sobol study, PINN surrogate vs solver
    sf = ROOT / cs.get("sobol_run", "results/p6_4_sensitivity_default") / "metrics.json"
    if sf.exists():
        s = json.loads(sf.read_text(encoding="utf-8"))
        acc = {}
        for case, c in s["cases"].items():
            if case not in out:
                continue
            acc[case] = {
                "solver_runs": c["solver"]["n_runs"],
                "solver_wall_s": c["solver"]["wall_s"],
                "pinn_runs": c["pinn"]["n_runs"],
                "pinn_eval_wall_s": c["pinn"]["wall_s"],
                "pinn_total_s": c["pinn"]["wall_s"] + out[case]["offline_ensemble_logged_s"],
                "pinn_qoi_err_median": c["paired"]["rel_err_median"],
            }
        out["_sobol_account"] = acc
    return out


# --------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--parts", default=PARTS)
    ap.add_argument(
        "--reuse", action="store_true", help="reuse task results cached in <results>/tasks/ by an earlier run"
    )
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    spec = SP.load_spec(a.config or None, a.profile, SP.parse_set(a.sets))
    cs = spec["cost_study"]
    custom = bool(a.config or a.sets or a.tag != "default" or a.profile != "smoke" or a.cases or a.parts != PARTS)
    odir = ROOT / "results" / f"p6_7_cost_{a.tag}"
    table = odir / "cost.md" if custom else ROOT / "docs" / "tables" / "phase6_7_cost.md"
    if a.figures_only:
        met = json.loads((odir / "metrics.json").read_text(encoding="utf-8"))
        met["analysis"] = analyse(met, cs)
        met["figures"] = figures(met, a.tag, odir, docs=not custom)
        report(met, table, a.tag)
        (odir / "metrics.json").write_text(json.dumps(met, indent=1), encoding="utf-8")
        print(f"wrote {table}")
        return
    parts = {p.strip() for p in a.parts.split(",")}
    cases = [c.strip() for c in a.cases.split(",")] if a.cases else list(cs["cases"])
    with Experiment(f"p6_7_cost_{a.tag}", spec) as ex:
        met = ex.metrics
        met["cases"] = {}
        jobs, labels = [], []
        if "hardware" in parts:
            jobs.append((MOD, "hardware_task", {}))
            labels.append(("_", "hardware", 0))
        for case in cases:
            cfg = cs["cases"][case]
            d = {"dim": cfg["dim"]}
            if cfg["dim"] == 2:
                ad = adapter(case, cfg)
                d["members"] = [m[0] for m in ad.members]
                d["box"] = ad.box
                d["train_logged_s"] = [_logged_wall(f / "log.csv") for f in ad.log_dirs]
                ef = ROOT / cfg["error_from"] / "metrics.json"
                if ef.exists():
                    an = json.loads(ef.read_text(encoding="utf-8"))["analysis"].get(case, {})
                    d["pinn_error"] = (an.get("uq") or {}).get("rel_err_median")
                mid = [float(np.sqrt(ad.box[k][0] * ad.box[k][1])) for k in NAMES]
                if "solver" in parts:
                    jobs.append(
                        (
                            MOD,
                            "solver2d_task",
                            dict(case=case, cfg=cfg, point=mid, repeats=int(cs["solver_check"]["repeats"])),
                        )
                    )
                    labels.append((case, "solver_check", 0))
            else:
                _, recs, info = members_1d(case, cfg)
                d["members"] = [m["name"] for m in recs]
                d["range"] = info["range"]
                d["train_logged_s"] = [float(m["training"]["wall_s"]) for m in recs]
                an = json.loads((ROOT / cfg["run"] / "metrics.json").read_text(encoding="utf-8"))["analysis"][case]
                key = max((k for k in an["methods"] if k.startswith("ensemble_M")), key=lambda k: int(k.split("M")[1]))
                d["pinn_error"] = float(an["methods"][key]["rel_l2"])
                d["pinn_error_method"] = key
                if "solver" in parts:
                    jobs.append((MOD, "solver1d_task", dict(case=case, cfg=cfg, so=cs["solvers_1d"])))
                    labels.append((case, "solver1d", 0))
            if "latency" in parts:
                for th in cs["latency"]["threads"]:
                    jobs.append((MOD, "latency_task", dict(case=case, cfg=cfg, lc=cs["latency"], threads=int(th))))
                    labels.append((case, "latency", int(th)))
            if "retrain" in parts:
                for i in cs["retrain"]["members"]:
                    if i < len(d["members"]):
                        jobs.append(
                            (
                                MOD,
                                "retrain_task",
                                dict(case=case, cfg=cfg, i=int(i), out_dir=str(ex.dir / "retrain" / f"{case}_{i}")),
                            )
                        )
                        labels.append((case, "retrain", int(i)))
            met["cases"][case] = d
        cache = ex.dir / "tasks"
        cache.mkdir(parents=True, exist_ok=True)
        keys = [f"{c}_{k}_{i}" for c, k, i in labels]
        done_before = {
            i: json.loads((cache / f"{k}.json").read_text(encoding="utf-8"))
            for i, k in enumerate(keys)
            if a.reuse and (cache / f"{k}.json").exists()
        }
        todo = [i for i in range(len(jobs)) if i not in done_before]
        print(f"{len(jobs)} tasks ({len(done_before)} reused)", flush=True)

        def done(j, r, sec):
            i = todo[j]
            (cache / f"{keys[i]}.json").write_text(json.dumps(r), encoding="utf-8")
            print(f"[{j + 1}/{len(todo)}] {labels[i]}: {sec:.0f} s", flush=True)

        new = run_tasks([jobs[i] for i in todo], {"isolate": True, **cs.get("execution", {})}, done) if todo else []
        res = dict(done_before) | dict(zip(todo, new, strict=True))
        for i, (case, kind, _) in enumerate(labels):
            r = res[i]
            if kind == "hardware":
                met["hardware"] = r
            elif kind == "latency":
                met["cases"][case].setdefault("latency", []).append(r)
            elif kind == "retrain":
                met["cases"][case].setdefault("retrain", []).append(r)
            elif kind == "solver_check":
                met["cases"][case]["solver_check"] = r
            elif kind == "solver1d":
                met["cases"][case]["solver1d"] = r
        met["analysis"] = analyse(met, cs)
        met["git"] = git_hash()
        met["figures"] = figures(met, a.tag, ex.dir, docs=not custom)
        report(met, table, a.tag)
        met["table"] = str(table.relative_to(ROOT))
    print(f"wrote {table}")


def _logged_wall(f: Path) -> float:
    import csv

    with open(f, encoding="utf-8") as fh:
        return float(list(csv.DictReader(fh))[-1]["wall_s"])


def _t(s) -> str:
    from aerosynapse.thermal.utils.costmodel import human_time

    return "—" if s is None else human_time(float(s))


def _n(v) -> str:
    return "never (solver cheaper per query)" if v is None else f"{v:,.0f}"


def report(met: dict, path: Path, tag: str) -> None:
    an = met["analysis"]
    hw = met.get("hardware") or {}
    L = [
        f"# Step 6.7: cost analysis — `{tag}` (auto-generated by experiments/p6_7_cost.py)",
        "",
        "Hardware: "
        + ", ".join(
            f"{k} {v}"
            for k, v in (
                ("CPU", hw.get("cpu")),
                ("logical cores", hw.get("cpus")),
                ("RAM", f"{hw['total_memory'] / 2**30:.1f} GiB" if hw.get("total_memory") else None),
                ("OS", hw.get("os")),
                ("Python", hw.get("python")),
                ("torch", hw.get("torch")),
                ("numpy", hw.get("numpy")),
                ("scipy", hw.get("scipy")),
                ("GPU", hw.get("gpu") or "none (CPU only)"),
            )
        )
        + ". All networks float64 on the CPU. Timings: one process at a time (nothing else running), median of repeats "
        "after an untimed warm-up; the solver ladders were measured the same way in Step 6.4b (one fresh process per grid).",
        "",
        "| case | members | offline: data + training (ensemble; clean / as logged) | training per member (mean): alone / logged (contention, same member) | PINN per query, ensemble: latency (batch 1) / best throughput (batch, threads) | PINN error (median) | solver per query at its grid (error) | speed-up per query | break-even queries: ensemble / one member | solver as accurate as the PINN (time, error) | break-even at equal accuracy |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case, a in an.items():
        if case.startswith("_"):
            continue
        eq = a.get("equal_accuracy")
        if "solver_grid" not in a:  # partial run (e.g. --parts retrain): no solver timing
            grid = "—"
        else:
            grid = (
                f"N {a['solver_grid']}"
                if a["dim"] == 1
                else (f"{a['solver_grid']} cells/H" if case == "D_graetz" else f"{a['solver_grid']}²")
            )
        eql = f"{eq['kind']} N {eq['N']}" if eq and "kind" in eq else (f"grid {eq['grid']}" if eq else "")
        eqs = f"{eql}: {_t(eq['time_s'])} ({100 * eq['rel_err']:.2g} %)" if eq else "—"
        clean = a["train_clean_s"]
        cont = f" ({a['contention']:.2f}x)" if a.get("contention") is not None else ""
        perr = f"{100 * a['pinn_error']:.3g} %" if a.get("pinn_error") is not None else "—"
        serr = f" ({100 * a['solver_err']:.2g} %)" if a.get("solver_err") is not None else ""
        sp = f"{a['speedup_per_query']:.3g}x" if a.get("speedup_per_query") is not None else "—"
        q = a.get("pinn_query_setting")
        qs = f" ({q['batch']}, {q['threads']})" if q else ""
        L.append(
            f"| {case} | {a['n_members']} | {_t(a['data_s'])} + training → {_t(a['offline_ensemble_s'])} / {_t(a['offline_ensemble_logged_s'])} | "
            f"{_t(np.mean(a['train_clean_est_s'])) if clean else '—'} / {_t(np.mean(a['train_logged_s']))}{cont} | "
            f"{_t(a.get('pinn_latency_ensemble_s'))} / {_t(a.get('pinn_query_ensemble_s'))}{qs} | "
            f"{perr} | {_t(a.get('solver_s'))} at {grid}{serr} | {sp} | "
            f"{_n(a.get('break_even_ensemble'))} / {_n(a.get('break_even_member'))} | {eqs} | "
            f"{_n(a.get('break_even_equal_accuracy')) if eq else 'none: the PINN is more accurate than every ladder grid'} |"
        )
    L.append("")
    rt = [(c, r) for c, d in met["cases"].items() for r in d.get("retrain") or []]
    if rt:
        L += [
            "Re-training alone (same seed and settings): clean time, data cost and reproducibility.",
            "",
            "| case / member | torch threads | data (snapshot solves) | training alone | logged (original run) | final loss: retrained / saved | max abs weight difference |",
            "|---|---|---|---|---|---|---|",
        ]
        for c, r in rt:
            L.append(
                f"| {c} / {r['member']} | {r.get('threads', '—')} | {_t(r['data_s'])} ({r['n_snapshots']} snapshots) | {_t(r['train_s'])} | {_t(r.get('logged_wall_saved'))} | "
                f"{r.get('final_total_new', float('nan')):.4g} / {r.get('final_total_saved', float('nan')):.4g} | {r['max_abs_weight_diff']:.3g} |"
            )
        L.append("")
    lat = [(c, d["latency"]) for c, d in met["cases"].items() if d.get("latency")]
    if lat:
        L += ["PINN inference per query (ensemble, all members) vs batch size and torch threads:", ""]
        for c, rs in lat:
            for r in rs:
                L.append(
                    f"- {c}, {r['threads']} thread(s): "
                    + ", ".join(f"batch {b['batch']}: {_t(np.sum(b['per_query_member_s']))}" for b in r["batches"])
                    + (
                        f"; full field ({r['field']['n_points']} points) {_t(np.sum(r['field']['member_s']))}"
                        if r.get("field") and r["field"].get("member_s")
                        else ""
                    )
                    + (
                        f"; extra memory {r['memory_extra_bytes_largest_batch'] / 2**20:.0f} MiB"
                        if r.get("memory_extra_bytes_largest_batch")
                        else ""
                    )
                )
        L.append("")
    for case, a in an.items():
        if case.startswith("_") or "solver_check_s" not in a:
            continue
        L.append(
            f"- {case}: solver at {a['solver_grid']} re-timed now {_t(a['solver_check_s'])} vs the Step 6.4b ladder {_t(a['solver_s'])}."
        )
    sa = an.get("_sobol_account")
    if sa:
        L += [
            "",
            "Project account — the Step 6.4 Sobol study (as run):",
            "",
            "| case | solver: runs, wall time | PINN: evaluations, evaluation time, + training and data (logged) | PINN median QoI error |",
            "|---|---|---|---|",
        ]
        for c, s in sa.items():
            L.append(
                f"| {c} | {s['solver_runs']}, {_t(s['solver_wall_s'])} | {s['pinn_runs']}, {_t(s['pinn_eval_wall_s'])}, total {_t(s['pinn_total_s'])} | {100 * s['pinn_qoi_err_median']:.3g} % |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(met: dict, tag: str, odir, docs: bool) -> list[str]:
    F.style()
    an = {k: v for k, v in met["analysis"].items() if not k.startswith("_")}
    paths = []
    # a partial run (e.g. --parts retrain) has no case with both PINN and solver timings: nothing to plot
    if not any(a.get("pinn_query_ensemble_s") and a.get("solver_s") for a in an.values()):
        return []
    paths += save_fig(F.amortisation(an), f"p6_7_amortisation_{tag}", odir, docs=docs)
    paths += save_fig(F.breakdown(an), f"p6_7_breakdown_{tag}", odir, docs=docs)
    if any(a.get("threads") for a in an.values()):
        paths += save_fig(F.latency(an), f"p6_7_latency_{tag}", odir, docs=docs)
    paths += save_fig(F.tradeoff(an), f"p6_7_tradeoff_{tag}", odir, docs=docs)
    paths += save_fig(F.decision3d(an), f"p6_7_decision3d_{tag}", odir, docs=docs)
    paths += save_fig(F.breakeven3d(an), f"p6_7_breakeven3d_{tag}", odir, docs=docs)
    return [str(p) for p in paths]


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"done in {time.perf_counter() - t0:.0f} s")
