r"""AirfRANS integration (NeurIPS 2022 Datasets & Benchmarks, OpenReview Zp8YmiQ_bDC; data licence ODbL-1.0).

The raw dataset (RANS fields over NACA 4/5-digit airfoils) is far larger than GitHub allows and is
**not** stored in the repository. This module:

1. parses simulation names ``airFoil2D_SST_<U>_<AoA deg>_<NACA params...>`` (3 params: 4-digit
   series m, p, t; 4 params: 5-digit series);
2. extracts per-simulation force coefficients with the official ``airfrans`` library
   (``Simulation(root, name, T).force_coefficient(reference=True)`` →
   ``((C_D, C_Dp, C_Dv), (C_L, C_Lp, C_Lv))``; API verified from the airfrans documentation) into a
   compact *derived table* (CSV) that may be redistributed under ODbL-1.0 with attribution;
3. trains a coefficient surrogate (U, AoA, geometry) → (C_D, C_L) with explicit convergence
   monitoring (validation loss plateau) on the official splits from ``manifest.json``;
4. quantifies **model discrepancy** of the inviscid panel method: panel C_L vs RANS C_L for the
   4-digit airfoils.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

LICENCE_NOTE = (
    "Derived from AirfRANS (NeurIPS 2022 Datasets and Benchmarks, OpenReview Zp8YmiQ_bDC), "
    "https://github.com/Extrality/airfrans_lib, licensed under ODbL-1.0. This derived table is shared "
    "under ODbL-1.0."
)
COLUMNS = ["name", "U", "aoa_deg", "series", "p1", "p2", "p3", "p4", "cd", "cdp", "cdv", "cl", "clp", "clv"]
NU_AIR = 1.56e-5  # kinematic viscosity used to report Re = U c / nu (assumption: chord 1 m, air near 298 K)


def parse_name(name: str) -> dict[str, Any]:
    parts = name.split("_")
    if len(parts) < 6 or not parts[0].startswith("airFoil2D"):
        raise ValueError(f"not an AirfRANS simulation name: {name}")
    vals = [float(v) for v in parts[2:]]
    U, aoa, geo = vals[0], vals[1], vals[2:]
    if len(geo) not in (3, 4):
        raise ValueError(f"unexpected number of NACA parameters in {name}")
    return {
        "name": name,
        "U": U,
        "aoa_deg": aoa,
        "series": 4 if len(geo) == 3 else 5,
        "p": geo + [0.0] * (4 - len(geo)),
    }


def extract_table(
    root: str | Path, names: list[str] | None = None, out_csv: str | Path | None = None, T: float = 298.15
) -> list[dict[str, Any]]:
    """Requires ``pip install airfrans`` and the downloaded dataset at ``root``."""
    import airfrans as af  # optional dependency

    root = Path(root)
    if names is None:
        names = sorted({n for v in json.loads((root / "manifest.json").read_text()).values() for n in v})
    rows = []
    for nm in names:
        meta = parse_name(nm)
        (cd, cdp, cdv), (cl, clp, clv) = af.Simulation(root=str(root), name=nm, T=T).force_coefficient(reference=True)
        rows.append(
            {
                "name": nm,
                "U": meta["U"],
                "aoa_deg": meta["aoa_deg"],
                "series": meta["series"],
                "p1": meta["p"][0],
                "p2": meta["p"][1],
                "p3": meta["p"][2],
                "p4": meta["p"][3],
                "cd": float(cd),
                "cdp": float(cdp),
                "cdv": float(cdv),
                "cl": float(cl),
                "clp": float(clp),
                "clv": float(clv),
            }
        )
    if out_csv is not None:
        write_table(rows, out_csv)
    return rows


def write_table(rows: list[dict[str, Any]], path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        fh.write(f"# {LICENCE_NOTE}\n")
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def read_table(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    out = []
    for r in csv.DictReader(lines):
        out.append({k: (r[k] if k == "name" else float(r[k])) for k in COLUMNS})
    return out


def features(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array(
        [[np.log10(r["U"] / NU_AIR), r["aoa_deg"], r["series"] == 5, r["p1"], r["p2"], r["p3"], r["p4"]] for r in rows],
        float,
    )


def targets(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([[r["cd"], r["cl"]] for r in rows], float)


@dataclass
class SurrogateResult:
    model: nn.Module
    history: list[dict[str, float]]
    converged: bool
    epochs_run: int
    wall_s: float
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            Z = torch.tensor((X - self.x_mean) / self.x_std, dtype=torch.float32)
            return self.model(Z).numpy() * self.y_std + self.y_mean


def train_coefficient_surrogate(
    X: np.ndarray,
    Y: np.ndarray,
    val_frac: float = 0.15,
    max_epochs: int = 5000,
    patience: int = 300,
    rel_tol: float = 1e-3,
    seed: int = 0,
    width: int = 64,
    lr: float = 1e-3,
) -> SurrogateResult:
    """MLP regression with early stopping. *Converged* means the best validation loss did not improve by
    more than ``rel_tol`` (relative) for ``patience`` consecutive epochs before ``max_epochs``."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idx = rng.permutation(len(X))
    nv = max(1, int(val_frac * len(X)))
    va, tr = idx[:nv], idx[nv:]
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-12
    ym, ys = Y[tr].mean(0), Y[tr].std(0) + 1e-12
    f = lambda A: torch.tensor(A, dtype=torch.float32)  # noqa: E731
    Xt, Yt = f((X[tr] - xm) / xs), f((Y[tr] - ym) / ys)
    Xv, Yv = f((X[va] - xm) / xs), f((Y[va] - ym) / ys)
    model = nn.Sequential(
        nn.Linear(X.shape[1], width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, Y.shape[1])
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, since, hist = np.inf, None, 0, []
    t0 = time.perf_counter()
    ep = 0
    for ep in range(1, max_epochs + 1):
        loss = torch.mean((model(Xt) - Yt) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            vl = float(torch.mean((model(Xv) - Yv) ** 2))
        hist.append({"epoch": ep, "train_mse": float(loss), "val_mse": vl})
        if vl < best * (1 - rel_tol):
            best, since = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return SurrogateResult(model, hist, since >= patience, ep, time.perf_counter() - t0, xm, xs, ym, ys)


def panel_vs_rans(rows: list[dict[str, Any]], n_max: int = 200, a_sound: float = 346.0) -> dict[str, Any]:
    """Inviscid panel C_L (with Prandtl–Glauert at M = U/a) vs RANS C_L for NACA 4-digit cases."""
    from aerosynapse.numerics.panel import solve_panel
    from aerosynapse.physics.airfoil import naca4_params

    four = [r for r in rows if r["series"] == 4][:n_max]
    d_cl = []
    for r in four:
        X, Y = naca4_params(r["p1"], r["p2"], r["p3"], 101)
        cl = solve_panel(X, Y, r["aoa_deg"], mach=min(r["U"] / a_sound, 0.69)).cl
        d_cl.append(cl - r["cl"])
    d = np.array(d_cl)
    return {
        "n": len(d),
        "mean_cl_panel_minus_rans": float(d.mean()) if d.size else None,
        "rmse_cl": float(np.sqrt(np.mean(d**2))) if d.size else None,
        "note": "inviscid panel model has no boundary layer: the difference is model discrepancy, not noise",
    }


def evaluate_on_splits(
    root_or_table: str | Path, manifest: str | Path, train_key: str, test_key: str, **kw: Any
) -> dict[str, Any]:
    rows = read_table(root_or_table)
    by = {r["name"]: r for r in rows}
    man = json.loads(Path(manifest).read_text())
    tr = [by[n] for n in man[train_key] if n in by]
    te = [by[n] for n in man[test_key] if n in by]
    res = train_coefficient_surrogate(features(tr), targets(tr), **kw)
    P = res.predict(features(te))
    Yt = targets(te)
    err = np.abs(P - Yt)
    return {
        "train_key": train_key,
        "test_key": test_key,
        "n_train": len(tr),
        "n_test": len(te),
        "converged": res.converged,
        "epochs": res.epochs_run,
        "wall_s": res.wall_s,
        "mae_cd": float(err[:, 0].mean()),
        "mae_cl": float(err[:, 1].mean()),
        "rel_mae_cd": float(np.mean(err[:, 0] / np.abs(Yt[:, 0]))),
        "max_abs_err_cl": float(err[:, 1].max()),
        "final_val_mse": res.history[-1]["val_mse"],
    }
