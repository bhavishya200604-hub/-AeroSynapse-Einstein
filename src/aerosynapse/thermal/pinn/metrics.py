"""Error metrics (relative L2, L-infinity, relative H1) and grid prediction for 1D PINNs."""

from __future__ import annotations

import numpy as np
import torch

from aerosynapse.thermal.pinn.derivs import evaluate
from aerosynapse.thermal.pinn.problems import Problem


def rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-300))


def linf(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(pred - ref)))


def rel_h1(pred: np.ndarray, pred_dx: np.ndarray, ref: np.ndarray, ref_dx: np.ndarray) -> float:
    """Relative H1 error sqrt(||e||^2 + ||e_x||^2) / sqrt(||u||^2 + ||u_x||^2) on a uniform grid (equal weights)."""
    num = np.sum((pred - ref) ** 2) + np.sum((pred_dx - ref_dx) ** 2)
    den = np.sum(ref**2) + np.sum(ref_dx**2)
    return float(np.sqrt(num / max(den, 1e-300)))


def predict_grid(model, problem: Problem, x: np.ndarray, t: np.ndarray | None, p: float, chunk: int = 20000):
    """Network u and u_x on the tensor grid: steady -> (nx,), transient -> (nt, nx). Returns numpy arrays.
    MC-dropout: fix the masks first (aerosynapse.thermal.pinn.networks.sample_dropout_masks): one sampled function per call."""
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    if problem.has_time:
        T_, X_ = np.meshgrid(t, x, indexing="ij")
        flat = {"x": X_.ravel(), "t": T_.ravel()}
    else:
        flat = {"x": np.asarray(x)}
    if problem.parametric:
        flat["logp"] = np.full(flat["x"].shape, float(np.log(p)))
    n = flat["x"].size
    u = np.empty(n)
    ux = np.empty(n)
    model.eval()
    for i in range(0, n, chunk):
        cols = {k: torch.as_tensor(v[i : i + chunk], dtype=dt, device=dev) for k, v in flat.items()}
        d = evaluate(model, cols, problem.input_names, first=("x",), second=(), create_graph=False)
        u[i : i + chunk] = d["u"].cpu().numpy()
        ux[i : i + chunk] = d["d_x"].cpu().numpy()
    model.train()
    shape = (len(t), len(x)) if problem.has_time else (len(x),)
    return u.reshape(shape), ux.reshape(shape)


def evaluate_model(model, problem: Problem, p: float, nx: int, nt: int) -> dict:
    """All metrics against the analytic reference; also returns the fields for plotting."""
    x = np.linspace(0.0, 1.0, nx)
    t = (problem.T * np.arange(1, nt + 1) / nt) if problem.has_time else None
    u, ux = predict_grid(model, problem, x, t, p)
    ref = problem.reference(x, t, p)
    ref_dx = problem.reference_dx(x, t, p)
    return {
        "rel_l2": rel_l2(u, ref),
        "linf": linf(u, ref),
        "rel_linf": linf(u, ref) / max(float(np.max(np.abs(ref))), 1e-300),
        "rel_h1": rel_h1(u, ux, ref, ref_dx),
        "_x": x,
        "_t": t,
        "_u": u,
        "_ref": ref,
    }
