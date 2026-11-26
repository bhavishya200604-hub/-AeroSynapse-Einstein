"""Step 4.6: hybrid data + physics training — CFD snapshot data for the data loss.

A snapshot is a verified Phase 1 solver solution on its fine grid (``channel2d.channel_reference``). ``snapshot_points``
picks n fluid cells of that grid (seeded; random or Sobol placement) and returns their coordinates, the field values
(optionally with Gaussian noise of std = noise x RMS of the field) and a grid mask of the chosen cells, so the cells
used as training data can be EXCLUDED from the error evaluation (held-out test cells; ``evaluate_channel(exclude=)``).
``snapshot_term`` turns one or more snapshots into a ``userdata.DataTerm`` (loss terms ``data_<name>_<field>`` in the
loss group 'data', weight ``loss.weights.data``), exactly like user data files (D60).

Training modes (``MODES``): data_free = physics only (the 4.5b setting); hybrid = physics + data; data_only = the same
network and ansatz with every physics weight (pde, bc, gauge) set to 0 — a pure fit of the data (with exact psi
boundary values the ansatz itself still carries the inlet profile and impermeable walls, stated in the report).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from aerosynapse.thermal.data.design import sobol_unit
from aerosynapse.thermal.pinn.userdata import DataTerm

MODES = ("data_free", "data_only", "hybrid")
DATA_FIELDS = ("u", "v", "theta", "p")


def snapshot_points(
    ref: dict,
    n: int,
    fields: list[str] | tuple[str, ...],
    placement: str = "random",
    noise: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """n distinct fluid cells of the reference grid with their (noisy) field values and a grid mask. For a fixed
    seed the sets are NESTED in n (random: prefix of one permutation; sobol: prefix of one Sobol sequence)."""
    X, Y = np.meshgrid(np.asarray(ref["x"]), np.asarray(ref["y"]))
    fluid = ~np.asarray(ref["solid"], bool)
    idx = np.flatnonzero(fluid.ravel())
    if not 0 < n <= len(idx):
        raise ValueError(f"snapshot: n = {n} data points requested, the grid has {len(idx)} fluid cells")
    if placement == "random":
        pick = np.random.default_rng(seed).permutation(idx)[:n]  # nested: same seed, smaller n = subset
    elif placement == "sobol":  # Sobol points in the box, each snapped to the nearest unused fluid cell
        xs, ys = X.ravel()[idx], Y.ravel()[idx]
        u = sobol_unit(n, 2, seed) * [xs.max() - xs.min(), ys.max() - ys.min()] + [xs.min(), ys.min()]
        free = np.ones(len(idx), bool)
        pick = np.empty(n, int)
        for i, (a, b) in enumerate(u):
            d = np.where(free, (xs - a) ** 2 + (ys - b) ** 2, np.inf)
            j = int(np.argmin(d))
            free[j] = False
            pick[i] = idx[j]
    else:
        raise ValueError(f"snapshot placement '{placement}': random or sobol")
    targets = {}
    for i, f in enumerate(fields):
        rng = np.random.default_rng(seed + 1_000_003 + 7919 * i)  # one stream per field: noise also nested in n
        full = np.asarray(ref[f], float)
        vals = full.ravel()[pick].copy()
        if noise > 0:
            vals += noise * float(np.sqrt(np.mean(full[fluid] ** 2))) * rng.standard_normal(n)
        targets[f] = vals
    mask = np.zeros(fluid.size, bool)
    mask[pick] = True
    return {
        "xy": np.stack([X.ravel()[pick], Y.ravel()[pick]], 1),
        "targets": targets,
        "mask": mask.reshape(fluid.shape),
    }


def snapshot_term(name: str, problem: Any, snaps: list[tuple[dict, dict | None]], like: torch.Tensor) -> DataTerm:
    """DataTerm from [(snapshot_points output, parameter values or None for a fixed-parameter problem), ...]."""
    X = torch.cat([problem.inputs(s["xy"], vals, like) for s, vals in snaps])
    fields = list(snaps[0][0]["targets"])
    targets = {
        f: torch.as_tensor(np.concatenate([s["targets"][f] for s, _ in snaps]), dtype=like.dtype, device=like.device)
        for f in fields
    }
    return DataTerm(name, problem, X, targets)


def fit_error(term: DataTerm, model: torch.nn.Module) -> float:
    """Worst relative L2 misfit of the network on its own data points. u and v are measured relative to the norm of
    the velocity VECTOR of the data (as the field errors, D70: v ~ 0 in a channel makes v's own norm meaningless);
    theta and p relative to their own norm."""
    from aerosynapse.thermal.pinn.userdata import predict  # noqa: PLC0415

    pred = predict(term.problem, model, term.X)
    t = term.targets
    vel = float(torch.sqrt(sum((t[k] ** 2).sum() for k in ("u", "v") if k in t))) if {"u", "v"} & set(t) else 0.0
    out = []
    for k, tk in t.items():
        den = vel if k in ("u", "v") else float(torch.linalg.norm(tk))
        out.append(float(torch.linalg.norm(pred[k].detach() - tk)) / max(den, 1e-300))
    return max(out)


def mode_weights(spec_weights: dict, mode: str) -> dict:
    """Loss weights for a training mode: data_only switches every physics group off; data_free drops nothing (no
    data terms are attached)."""
    if mode not in MODES:
        raise ValueError(f"mode '{mode}': one of {MODES}")
    w = dict(spec_weights)
    if mode == "data_only":
        w.update({"pde": 0.0, "bc": 0.0, "ic": 0.0, "gauge": 0.0})
    return w
