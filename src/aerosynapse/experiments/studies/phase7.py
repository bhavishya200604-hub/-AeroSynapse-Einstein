"""Phase 7 study: trust/fallback on deliberately OOD queries, trust reports per model family,
resource-estimator accuracy, checkpoint-resume, notebooks, dataset catalogue, safety tier evidence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aerosynapse.ml.datasets import OperatorDataset, convdiff_operator_dataset
from aerosynapse.ml.fno import FNO1d
from aerosynapse.ml.nets import MLP, n_params
from aerosynapse.ml.pinn import PINNConfig, train_pinn
from aerosynapse.ml.problems import ConvDiff1D
from aerosynapse.ml.training import per_sample_rel_l2, predict_operator, train_operator
from aerosynapse.numerics.convdiff import solve_convdiff_fv
from aerosynapse.validation.resources import calibrate, estimate_fv_seconds, estimate_pinn_seconds
from aerosynapse.validation.trust import TrustPolicy, trust_report_markdown


def features(d: OperatorDataset) -> np.ndarray:
    u0 = d.inputs[:, 0]
    spec = np.abs(np.fft.rfft(u0 - u0.mean(1, keepdims=True), axis=1)) ** 2
    hf = spec[:, 5:].sum(1) / np.maximum(spec[:, 1:].sum(1), 1e-30)
    return np.column_stack([np.log10(d.params[:, 0]), u0.std(1), hf])


def trust_fallback(root: Path, epochs: int) -> dict[str, Any]:
    tr = convdiff_operator_dataset(300, n=64, pe_range=(5, 50), seed=11)
    cal = convdiff_operator_dataset(100, n=64, pe_range=(5, 50), seed=12)
    groups = {
        "in_distribution": convdiff_operator_dataset(100, n=64, pe_range=(5, 50), seed=13),
        "ood_pe": convdiff_operator_dataset(100, n=64, pe_range=(200, 1000), seed=14),
        "ood_input": convdiff_operator_dataset(100, n=64, pe_range=(5, 50), k_max=8, decay=1.0, seed=15),
    }
    members = []
    t0 = time.perf_counter()
    for s in range(3):
        torch.manual_seed(s)
        m, _, _ = train_operator(FNO1d(2, 1, 32, 12, 4), tr.inputs, tr.outputs, epochs=epochs, seed=s)
        members.append(m)
    train_s = time.perf_counter() - t0

    def ens(d: OperatorDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        P = np.stack([predict_operator(m, d.inputs) for m in members])
        mu, sd = P.mean(0), P.std(0, ddof=1)
        spread = np.sqrt((sd**2).sum((1, 2))) / np.sqrt((mu**2).sum((1, 2)))
        resid = np.abs(mu.mean(-1)[:, 0] - d.inputs[:, 0].mean(-1))
        return mu, spread, resid

    _, sp_c, rs_c = ens(cal)
    pol = TrustPolicy.calibrate(
        np.log10(tr.params), features(cal), sp_c, rs_c, q=0.99, log_path=root / "fallback_decisions.jsonl"
    )
    out: dict[str, Any] = {
        "ensemble_train_s": train_s,
        "thresholds": {
            "mahalanobis": pol.maha_threshold,
            "rel_spread": pol.unc_threshold,
            "mass_residual": pol.residual_threshold,
        },
    }
    t_solver = []
    for name, d in groups.items():
        mu, sp, rs = ens(d)
        F = features(d)
        err = per_sample_rel_l2(mu, d.outputs)
        used, guarded_err, reasons = [], [], {"domain": 0, "input_novelty": 0, "uncertainty": 0, "physics_residual": 0}
        for i in range(len(d.inputs)):
            dec = pol.decide(np.log10(d.params[i]), F[i], float(sp[i]), float(rs[i]), f"{name}/{i}")
            used.append(dec.use_surrogate)
            for r in dec.reasons:
                reasons[r] += 1
            if dec.use_surrogate:
                guarded_err.append(err[i])
            else:
                t0 = time.perf_counter()
                xc = (np.arange(64) + 0.5) / 64
                u0 = d.inputs[i, 0]
                r = solve_convdiff_fv(
                    0,
                    1,
                    64,
                    1.0,
                    1 / float(d.params[i, 0]),
                    lambda x, u0=u0, xc=xc: np.interp(x, xc, u0, period=1.0),
                    0.5,
                    scheme="vanleer",
                    cfl=0.4,
                )
                t_solver.append(time.perf_counter() - t0)
                guarded_err.append(float(np.linalg.norm(r.u - d.outputs[i, 0]) / np.linalg.norm(d.outputs[i, 0])))
        used_a = np.array(used)
        out[name] = {
            "fallback_rate": float(1 - used_a.mean()),
            "surrogate_err_median": float(np.median(err)),
            "surrogate_err_max": float(err.max()),
            "guarded_err_median": float(np.median(guarded_err)),
            "guarded_err_max": float(np.max(guarded_err)),
            "fallback_reasons": reasons,
            "err_of_accepted_max": float(err[used_a].max()) if used_a.any() else None,
        }
    out["solver_time_per_fallback_s"] = float(np.mean(t_solver)) if t_solver else None
    out["n_logged_decisions"] = len(pol.log)
    return out


def resource_estimator() -> dict[str, Any]:
    cal = calibrate(quick=False)
    p = ConvDiff1D(pe=10.0)
    m = MLP(2, 1, 32, 3, lo=p.lo, hi=p.hi, periodic_dims=(0,))
    it, n = 500, 1000
    pred = estimate_pinn_seconds(cal, it, n, n_params(m))
    r = train_pinn(p, m, PINNConfig(iterations=it, n_collocation=n, log_every=10**9))
    t0 = time.perf_counter()
    fv = solve_convdiff_fv(0, 1, 512, 1.0, 0.01, lambda x: np.sin(2 * np.pi * x), 0.5, scheme="vanleer")
    t_fv = time.perf_counter() - t0
    pred_fv = estimate_fv_seconds(cal, 512, fv.steps)
    return {
        "calibration": {"c_pinn": cal.c_pinn, "c_fv": cal.c_fv, "basis": cal.basis},
        "pinn": {
            "predicted_s": pred,
            "measured_s": r.wall_s,
            "rel_error": (pred - r.wall_s) / r.wall_s,
            "config": f"{it} it, {n} pts, {n_params(m)} params",
        },
        "fv": {
            "predicted_s": pred_fv,
            "measured_s": t_fv,
            "rel_error": (pred_fv - t_fv) / t_fv,
            "config": f"512 cells, {fv.steps} steps",
        },
    }


def trust_reports(root: Path, results_root: Path, fb: dict[str, Any]) -> dict[str, str]:
    reports = {}
    secs = {
        "Accuracy (in-distribution)": {
            "rel_l2_median": fb["in_distribution"]["surrogate_err_median"],
            "rel_l2_max": fb["in_distribution"]["surrogate_err_max"],
        },
        "Generalisation (out-of-distribution Pe 200-1000)": {
            "rel_l2_median": fb["ood_pe"]["surrogate_err_median"],
            "rel_l2_max": fb["ood_pe"]["surrogate_err_max"],
        },
        "Generalisation (high-frequency inputs)": {
            "rel_l2_median": fb["ood_input"]["surrogate_err_median"],
            "rel_l2_max": fb["ood_input"]["surrogate_err_max"],
        },
        "Fallback behaviour": {
            f"{g}_fallback_rate": fb[g]["fallback_rate"] for g in ("in_distribution", "ood_pe", "ood_input")
        },
        "Physical consistency": {"mass_residual_threshold": fb["thresholds"]["mass_residual"]},
    }
    p4 = results_root / "phase4_operators_uq" / "metrics.json"
    if p4.exists():
        m4 = json.loads(p4.read_text())
        e = m4.get("ensemble_uq", {})
        secs["Calibration (Phase 4 ensemble)"] = {f"{k}_coverage_95": v.get("coverage_95") for k, v in e.items()}
        secs["Performance (Phase 4)"] = {k: m4["operators"]["speed"][k] for k in ("speedup", "break_even_queries")}
    else:
        secs["Calibration"] = {"status": "Phase 4 results not available at report time"}
    reports["fno_operator"] = trust_report_markdown("FNO conv-diff operator", secs, "1/5 research simulation")
    p3 = results_root / "phase3_adaptive_collocation" / "metrics.json"
    if p3.exists():
        t = json.loads(p3.read_text())["table"]
        reports["pinn_convdiff"] = trust_report_markdown(
            "PINN conv-diff (Phase 3)",
            {
                "Accuracy vs exact (median rel. L2, uniform sampler)": {
                    k: v["median"] for k, v in t.items() if k.endswith("/uniform")
                }
            },
            "1/5 research simulation",
        )
    p5 = results_root / "phase5_optimization" / "metrics.json"
    if p5.exists():
        m5 = json.loads(p5.read_text())
        reports["optimisation_surrogates"] = trust_report_markdown(
            "Optimisation (Phase 5)",
            {"Optimisation reliability": {k: v["verdict"] for k, v in m5["protocol"].items()}},
            "2/5 engineering analysis (illustrative models)",
        )
    for k, md in reports.items():
        (root / f"trust_report_{k}.md").write_text(md, encoding="utf-8")
    return reports
