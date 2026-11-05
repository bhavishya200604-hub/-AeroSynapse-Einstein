"""Phase 4 study: neural operators vs baselines, generalisation, resolution transfer, multi-fidelity,
hybrids, data assimilation, UQ calibration, speed-up and break-even."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.sparse.linalg as spla
import torch

from aerosynapse.ml.datasets import (
    OperatorDataset,
    check_leakage,
    convdiff_operator_dataset,
    poisson2d_operator_dataset,
)
from aerosynapse.ml.deeponet import DeepONet
from aerosynapse.ml.fno import FNO1d, FNO2d
from aerosynapse.ml.hybrid import cg_iterations, coarse_solver_features, kalman_convdiff
from aerosynapse.ml.rom import CNN1d, FlatMLP, PODRegressor
from aerosynapse.ml.training import per_sample_rel_l2, predict_operator, train_operator
from aerosynapse.numerics.convdiff import solve_convdiff_fv
from aerosynapse.numerics.heat import laplacian_2d_dirichlet, solve_heat_1d
from aerosynapse.physics.library import heat_1d
from aerosynapse.uq.calibration import calibration_report
from aerosynapse.uq.gp import GaussianProcess
from aerosynapse.uq.mc import propagate
from aerosynapse.uq.sobol import ishigami, ishigami_analytic, sobol_indices
from aerosynapse.validation.metrics import break_even_queries

PROFILES = {
    "quick": {"n_train": 400, "n_test": 100, "epochs": 150, "ens": 5},
    "standard": {"n_train": 1000, "n_test": 200, "epochs": 400, "ens": 5},
    "full": {"n_train": 4000, "n_test": 500, "epochs": 1000, "ens": 10},
}


def dist(e: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(e)),
        "mean": float(np.mean(e)),
        "q90": float(np.quantile(e, 0.9)),
        "max": float(np.max(e)),
        "n": int(e.size),
    }


def deeponet_forward(n: int) -> Any:
    y = torch.tensor(((np.arange(n) + 0.5) / n)[:, None], dtype=torch.get_default_dtype())

    def fwd(m: torch.nn.Module, a: torch.Tensor) -> torch.Tensor:
        sens = torch.cat([a[:, 0], a[:, 1, :1]], 1)
        return m(sens, y).unsqueeze(1)

    return fwd


def resample(a: np.ndarray, n_new: int) -> np.ndarray:
    """Periodic linear resampling of cell-centred data along the last axis."""
    n = a.shape[-1]
    xc = (np.arange(n) + 0.5) / n
    xn = (np.arange(n_new) + 0.5) / n_new
    xp = np.concatenate([xc - 1, xc, xc + 1])
    return np.apply_along_axis(lambda v: np.interp(xn, xp, np.tile(v, 3)), -1, a)


def operators(P: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    t0 = time.perf_counter()
    tr = convdiff_operator_dataset(P["n_train"], n=64, pe_range=(5, 50), seed=seed)
    t_data = time.perf_counter() - t0
    te = convdiff_operator_dataset(P["n_test"], n=64, pe_range=(5, 50), seed=seed + 101)
    ood_pe = convdiff_operator_dataset(P["n_test"], n=64, pe_range=(200, 1000), seed=seed + 202)
    ood_in = convdiff_operator_dataset(P["n_test"], n=64, pe_range=(5, 50), k_max=8, decay=1.0, seed=seed + 303)
    out: dict[str, Any] = {
        "data_gen_s": t_data,
        "data_gen_s_per_sample": t_data / P["n_train"],
        "leakage": {
            "train_test": check_leakage(tr, te),
            "train_oodpe": check_leakage(tr, ood_pe),
            "train_oodin": check_leakage(tr, ood_in),
        },
        "reference_numerical_error_max": {
            "train": float(tr.numerical_error.max()),
            "test": float(te.numerical_error.max()),
            "ood_pe": float(ood_pe.numerical_error.max()),
        },
        "dataset_card": tr.card,
    }
    splits = {"test_id": te, "ood_pe": ood_pe, "ood_input": ood_in}
    models: dict[str, Any] = {}
    torch.manual_seed(seed)
    specs = {
        "fno": (FNO1d(2, 1, 32, 12, 4), None),
        "fno_mass_conserving": (FNO1d(2, 1, 32, 12, 4, conserve_mass=True), None),
        "deeponet": (DeepONet(64, 1, p=64, width=128, depth=3), deeponet_forward(64)),
        "cnn": (CNN1d(2, 32, 4), None),
        "mlp": (FlatMLP(2, 64, 256, 3), None),
    }
    for name, (m, fwd) in specs.items():
        m, hist, wall = train_operator(m, tr.inputs, tr.outputs, epochs=P["epochs"], seed=seed, forward=fwd)
        models[name] = (m, fwd)
        res: dict[str, Any] = {
            "train_s": wall,
            "final_train_loss": hist[-1],
            "n_params": sum(p.numel() for p in m.parameters()),
        }
        for sn, d in splits.items():
            pred = predict_operator(m, d.inputs, fwd)
            res[sn] = dist(per_sample_rel_l2(pred, d.outputs))
            res[sn]["mass_error_max"] = float(np.abs(pred.mean(-1) - d.inputs[:, :1].mean(-1)).max())
        out[name] = res
    A = lambda d: d.inputs[:, 0]  # noqa: E731
    lp = lambda d: np.log10(d.params)  # noqa: E731
    t0 = time.perf_counter()
    pod = PODRegressor(rank=16, poly_degree=3).fit(A(tr), lp(tr), tr.outputs[:, 0])
    out["pod"] = {"train_s": time.perf_counter() - t0, "energy": pod.energy}
    for sn, d in splits.items():
        out["pod"][sn] = dist(per_sample_rel_l2(pod.predict(A(d), lp(d))[:, None], d.outputs))
    # resolution transfer (same random fields regenerated at higher resolution)
    rt: dict[str, Any] = {}
    for n in (128, 256):
        dn = convdiff_operator_dataset(P["n_test"], n=n, pe_range=(5, 50), seed=seed + 101)
        rt[str(n)] = {
            "fno": dist(per_sample_rel_l2(predict_operator(models["fno"][0], dn.inputs), dn.outputs)),
            "cnn": dist(per_sample_rel_l2(predict_operator(models["cnn"][0], dn.inputs), dn.outputs)),
            # DeepONet sensors are fixed at the 64 training cell centres: resample the input to them,
            # then interpolate the 64-point output onto the new grid
            "deeponet_resampled_sensors": dist(
                per_sample_rel_l2(
                    resample(predict_operator(models["deeponet"][0], resample(dn.inputs, 64), deeponet_forward(64)), n),
                    dn.outputs,
                )
            ),
            "mlp": "not applicable: fixed-resolution input layer (raises ValueError)",
        }
    out["resolution_transfer"] = rt
    # speed-up / break-even (single query, batch 1, this machine)
    a1 = te.inputs[:1]
    t0 = time.perf_counter()
    for _ in range(20):
        predict_operator(models["fno"][0], a1)
    t_fno = (time.perf_counter() - t0) / 20
    u0 = lambda x: 1 + 0.2 * np.sin(2 * np.pi * x)  # noqa: E731
    t0 = time.perf_counter()
    for _ in range(5):
        solve_convdiff_fv(0, 1, 64, 1.0, 1 / 20, u0, 0.5, scheme="vanleer", cfl=0.4)
    t_fv = (time.perf_counter() - t0) / 5
    out["speed"] = {
        "t_fno_query_s": t_fno,
        "t_fv_solve_s": t_fv,
        "speedup": t_fv / t_fno,
        "fno_train_s": out["fno"]["train_s"],
        "data_gen_s": t_data,
        "break_even_queries": break_even_queries(out["fno"]["train_s"], t_data, t_fv, t_fno),
        "accuracy_note": "FNO rel. L2 (test_id median) vs FV discretisation error — see fno.test_id",
    }
    out["_models"] = models
    out["_data"] = (tr, te, ood_pe)
    return out


def ensemble_uq(P: dict[str, Any], tr: OperatorDataset, te: OperatorDataset, ood: OperatorDataset) -> dict[str, Any]:
    preds_te, preds_ood = [], []
    for s in range(P["ens"]):
        torch.manual_seed(100 + s)
        m, _, _ = train_operator(FNO1d(2, 1, 32, 12, 4), tr.inputs, tr.outputs, epochs=P["epochs"], seed=100 + s)
        preds_te.append(predict_operator(m, te.inputs))
        preds_ood.append(predict_operator(m, ood.inputs))
    out: dict[str, Any] = {}
    for name, preds, d in (("test_id", preds_te, te), ("ood_pe", preds_ood, ood)):
        Pm = np.stack(preds)
        mu, sd = Pm.mean(0), Pm.std(0, ddof=1)
        rep = calibration_report(d.outputs.ravel(), mu.ravel(), sd.ravel())
        err = per_sample_rel_l2(mu, d.outputs)
        spread = np.sqrt((sd**2).sum((1, 2))) / np.sqrt((d.outputs**2).sum((1, 2)))
        rep["error_spread_spearman"] = float(_spearman(err, spread))
        rep["ensemble_mean_rel_l2"] = dist(err)
        out[name] = {k: v for k, v in rep.items() if k != "pit_histogram"} | {"pit_histogram": rep["pit_histogram"]}
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def multifidelity(P: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    n_hf = 40
    tr = convdiff_operator_dataset(n_hf, n=64, seed=seed + 7)
    te = convdiff_operator_dataset(P["n_test"], n=64, seed=seed + 101)

    def lf(d: OperatorDataset) -> np.ndarray:
        return np.array(
            [coarse_solver_features(d.inputs[i, 0], float(d.params[i, 0]), 0.5, factor=4) for i in range(len(d.inputs))]
        )[:, None]

    lf_tr, lf_te = lf(tr), lf(te)
    m_hf, _, w1 = train_operator(FNO1d(2, 1, 32, 12, 4), tr.inputs, tr.outputs, epochs=P["epochs"], seed=seed)
    Xmf_tr = np.concatenate([tr.inputs, lf_tr], 1)
    Xmf_te = np.concatenate([te.inputs, lf_te], 1)
    m_mf, _, w2 = train_operator(FNO1d(3, 1, 32, 12, 4), Xmf_tr, tr.outputs - lf_tr, epochs=P["epochs"], seed=seed)
    return {
        "n_hf_train": n_hf,
        "hf_only_fno": dist(per_sample_rel_l2(predict_operator(m_hf, te.inputs), te.outputs)),
        "low_fidelity_solver_alone": dist(per_sample_rel_l2(lf_te, te.outputs)),
        "multifidelity_fno": dist(per_sample_rel_l2(lf_te + predict_operator(m_mf, Xmf_te), te.outputs)),
        "train_s": {"hf_only": w1, "mf": w2},
    }


def neural_correction(P: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    tr = convdiff_operator_dataset(P["n_train"], n=64, seed=seed + 11)
    te = convdiff_operator_dataset(P["n_test"], n=64, seed=seed + 101)

    def feats(d: OperatorDataset) -> np.ndarray:
        return np.array(
            [coarse_solver_features(d.inputs[i, 0], float(d.params[i, 0]), 0.5, factor=4) for i in range(len(d.inputs))]
        )[:, None]

    ctr, cte = feats(tr), feats(te)
    Xtr, Xte = np.concatenate([tr.inputs, ctr], 1), np.concatenate([te.inputs, cte], 1)
    m, _, w = train_operator(CNN1d(3, 32, 4), Xtr, tr.outputs - ctr, epochs=P["epochs"], seed=seed)
    m0, _, w0 = train_operator(CNN1d(2, 32, 4), tr.inputs, tr.outputs, epochs=P["epochs"], seed=seed)
    return {
        "coarse_solver_alone": dist(per_sample_rel_l2(cte, te.outputs)),
        "coarse_plus_neural_correction": dist(per_sample_rel_l2(cte + predict_operator(m, Xte), te.outputs)),
        "cnn_without_solver": dist(per_sample_rel_l2(predict_operator(m0, te.inputs), te.outputs)),
        "train_s": {"correction": w, "pure": w0},
    }


def residual_correction_cg(P: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    n = 32
    tr = poisson2d_operator_dataset(200, n=n, seed=seed)
    te = poisson2d_operator_dataset(30, n=n, seed=seed + 1)
    m, _, w = train_operator(FNO2d(1, 1, 24, 8, 4), tr.inputs, tr.outputs, epochs=P["epochs"], seed=seed)
    pred = predict_operator(m, te.inputs)
    h = 1.0 / n
    A = (-laplacian_2d_dirichlet(n, n, h, h)).tocsr()
    it0, it1, e_nn, e_cg = [], [], [], []
    for i in range(len(te.inputs)):
        b = te.inputs[i, 0, 1:-1, 1:-1].ravel()
        x_ref = spla.spsolve(A.tocsc(), b)
        x0 = pred[i, 0, 1:-1, 1:-1].ravel()
        x_a, k0 = cg_iterations(A, b, None)
        x_b, k1 = cg_iterations(A, b, x0)
        it0.append(k0)
        it1.append(k1)
        e_nn.append(float(np.linalg.norm(x0 - x_ref) / np.linalg.norm(x_ref)))
        e_cg.append(float(np.linalg.norm(x_b - x_ref) / np.linalg.norm(x_ref)))
    return {
        "fno_train_s": w,
        "cg_iters_zero_init": dist(np.array(it0, float)),
        "cg_iters_fno_init": dist(np.array(it1, float)),
        "fno_alone_rel_l2": dist(np.array(e_nn)),
        "fno_init_cg_final_rel_l2": dist(np.array(e_cg)),
        "tolerance": 1e-8,
    }


def classical_uq() -> dict[str, Any]:
    # GP calibration on a 1D heat QoI with additive measurement noise
    rng = np.random.default_rng(0)
    q = lambda a: float(np.exp(-a * np.pi**2 * 0.5))  # T(0.5, t=0.5) for heat1d (sin(pi/2) = 1)  # noqa: E731
    Xa = rng.uniform(0.05, 0.15, (30, 1))
    y = np.array([q(a) for a in Xa[:, 0]]) + rng.normal(0, 0.01, 30)
    gp = GaussianProcess(seed=0).fit(Xa, y)
    Xt = rng.uniform(0.05, 0.15, (500, 1))
    yt = np.array([q(a) for a in Xt[:, 0]]) + rng.normal(0, 0.01, 500)
    mu, sd = gp.predict(Xt, include_noise=True)
    gp_rep = calibration_report(yt, mu, sd)

    # Monte Carlo through the trusted FD solver vs analytic mean
    def qoi(x: np.ndarray) -> float:
        p = heat_1d(alpha=float(x[0]))
        xx, s = solve_heat_1d(p, 64, nt=64)
        return float(np.interp(0.5, xx, s.u))

    mc = propagate(qoi, lambda r, n: r.uniform(0.05, 0.15, (n, 1)), 400)
    c = np.pi**2 * 0.5
    analytic = (np.exp(-0.05 * c) - np.exp(-0.15 * c)) / (0.1 * c)
    sob = sobol_indices(ishigami, -np.pi * np.ones(3), np.pi * np.ones(3), n=2**14)
    ana = ishigami_analytic()
    kf = kalman_convdiff()
    return {
        "gp_heat_qoi": {
            "noise_std_est": gp.noise_std,
            "true_noise": 0.01,
            **{k: v for k, v in gp_rep.items() if k != "pit_histogram"},
        },
        "mc_heat": {
            "mean": mc["mean"],
            "se": mc["se_mean"],
            "analytic_mean": float(analytic),
            "z_score": float((mc["mean"] - analytic) / mc["se_mean"]),
            "n": 400,
            "note": "difference includes FD discretisation error (n=64)",
        },
        "sobol_ishigami": {
            "S1": sob["S1"].tolist(),
            "ST": sob["ST"].tolist(),
            "S1_analytic": ana["S1"].tolist(),
            "ST_analytic": ana["ST"].tolist(),
            "n_evals": int(sob["n_evals"]),
        },
        "kalman": {
            "final_rmse_free": kf.final_rmse_free,
            "final_rmse_kf": kf.final_rmse_kf,
            "coverage_2sigma": kf.coverage_2sigma,
        },
    }


def generalisation_map(model: torch.nn.Module, n_per_cell: int = 12, seed: int = 500) -> dict[str, Any]:
    """FNO error over (Pe, input bandwidth k_max); training region Pe in [5, 50], k_max = 4."""
    pes = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0, 1000.0, 3000.0]
    ks = [2, 3, 4, 6, 8, 10]
    E = np.zeros((len(pes), len(ks)))
    for i, pe in enumerate(pes):
        for j, k in enumerate(ks):
            d = convdiff_operator_dataset(n_per_cell, n=64, pe_range=(pe, pe * 1.0001), k_max=k, seed=seed + 31 * i + j)
            E[i, j] = float(np.median(per_sample_rel_l2(predict_operator(model, d.inputs), d.outputs)))
    return {"pe": pes, "k_max": ks, "median_rel_l2": E.tolist(), "train_region": {"pe": [5, 50], "k_max": [1, 4]}}
