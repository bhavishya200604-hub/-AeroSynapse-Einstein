"""User-provided data for any PINN study case (Steps 4.1-4.4): extra evaluation truth and/or training data.

A data SOURCE is declared once under ``user_data.sources.<name>`` and attached to a study case with ``data: <name>``
(or a list of names). Nothing is attached by default, so the default studies are unchanged.

Formats (``format: auto`` picks by extension; .h5 files with a ``fields`` group are read as Phase 2 datasets):
    csv     header row with column names, one row per point
    npz     arrays named like the columns, all of equal length (one entry per point)
    h5      generic HDF5: datasets named like the columns, equal length
    phase2  a Phase 2 dataset file (data/datasets/<name>/case*.h5): one design point - ``select.index`` or the
            point whose parameters match ``select.params`` (default: the case's parameter value) - with its fields
            on the grid; ``select.time``: all | last | a time value (nearest snapshot)
Mapping: ``columns: {model_name: name_in_file}`` (inputs x, y, t; fields theta for 1D, u, v, p, theta for 2D; a
parameter column such as Pe for parametric problems). Unmapped model names are looked up under their own name.
``scale`` multiplies columns after loading (unit conversion), ``max_points`` subsamples (seeded), and points outside
the problem domain are dropped and counted. Roles: ``evaluate`` (errors vs these data are reported), ``train``
(adds the loss components ``data_<source>_<field>`` with weight ``loss.weights.data``), ``both``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from aerosynapse.thermal.utils.experiment import ROOT

ROLES = ("evaluate", "train", "both")
FORMATS = ("auto", "csv", "npz", "h5", "phase2")


@dataclass
class UserData:
    name: str
    inputs: dict[str, np.ndarray]  # model input name -> (N,)
    fields: dict[str, np.ndarray]  # field name -> (N,)
    role: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(next(iter(self.fields.values())))


def _path(f: str) -> Path:
    p = Path(f)
    return p if p.is_absolute() else ROOT / p


def _detect(path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    ext = path.suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext == ".npz":
        return "npz"
    if ext in (".h5", ".hdf5"):
        import h5py

        with h5py.File(path, "r") as f:
            return "phase2" if "fields" in f and "params" in f else "h5"
    raise ValueError(f"cannot infer the format of {path} (use format: csv | npz | h5 | phase2)")


def _read_columns(path: Path, fmt: str) -> dict[str, np.ndarray]:
    if fmt == "csv":
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"{path}: no data rows")
        return {k.strip(): np.array([float(r[k]) for r in rows]) for k in rows[0]}
    if fmt == "npz":
        with np.load(path) as z:
            return {k: np.asarray(z[k], float).reshape(-1) for k in z.files}
    import h5py

    with h5py.File(path, "r") as f:
        return {k: np.asarray(f[k][...], float).reshape(-1) for k in f if isinstance(f[k], h5py.Dataset)}


def _read_phase2(path: Path, select: dict, case_params: dict[str, float]) -> tuple[dict[str, np.ndarray], dict]:
    from aerosynapse.thermal.data.dataset_io import read_dataset

    ds = read_dataset(path)
    names = [d.name for d in ds.design.dims]
    params = ds.design.params
    if "index" in select:
        i = int(select["index"])
    else:
        want = {**case_params, **(select.get("params") or {})}
        use = [n for n in names if n in want]
        if not use:
            raise ValueError(f"{path}: give select.index or select.params (dataset parameters {names})")
        cols = np.array([names.index(n) for n in use])
        target = np.array([float(want[n]) for n in use])
        rel = np.abs(params[:, cols] - target) / np.maximum(np.abs(target), 1e-300)
        i = int(np.argmin(rel.max(axis=1)))
        tol = float(select.get("tolerance", 1e-9))
        if rel[i].max() > tol:
            raise ValueError(
                f"{path}: no design point with {dict(zip(use, target.tolist(), strict=True))} (closest: "
                f"{dict(zip(names, params[i].tolist(), strict=True))}); use select.index or a larger select.tolerance"
            )
    t = ds.coords.get("t")
    tsel = select.get("time", "all")
    fields = {k: v[i] for k, v in ds.fields.items()}  # (nt, [ny,] nx)
    tidx = np.arange(len(t)) if t is not None else None
    if t is not None and tsel != "all":
        tidx = np.array([len(t) - 1]) if tsel == "last" else np.array([int(np.argmin(np.abs(t - float(tsel))))])
    axes: list[tuple[str, np.ndarray]] = []
    if t is not None:
        axes.append(("t", t[tidx]))
        fields = {k: v[tidx] for k, v in fields.items()}
    if "y" in ds.coords:
        axes.append(("y", ds.coords["y"]))
    axes.append(("x", ds.coords["x"]))
    mesh = np.meshgrid(*[a for _, a in axes], indexing="ij")
    out = {n: m.reshape(-1) for (n, _), m in zip(axes, mesh, strict=True)}
    out.update({k: v.reshape(-1) for k, v in fields.items()})
    for j, n in enumerate(names):
        out[n] = np.full(out["x"].shape, params[i, j])
    meta = {"phase2_index": i, "phase2_params": dict(zip(names, params[i].tolist(), strict=True))}
    if tsel != "all" and t is not None:
        meta["phase2_time"] = float(t[tidx][0])
    return out, meta


def load_source(
    name: str,
    cfg: dict,
    input_names: list[str],
    field_names: list[str],
    case_params: dict[str, float],
    lo: list[float],
    hi: list[float],
    seed: int = 0,
) -> UserData:
    """Read a source and map it onto the model's inputs/fields; drop points outside [lo, hi]; subsample."""
    path = _path(cfg["file"])
    if not path.exists():
        raise FileNotFoundError(f"user_data.sources.{name}.file: {path} does not exist")
    fmt = _detect(path, cfg.get("format", "auto"))
    meta: dict[str, Any] = {"file": str(cfg["file"]), "format": fmt}
    if fmt == "phase2":
        cols, m = _read_phase2(path, cfg.get("select") or {}, case_params)
        meta.update(m)
    else:
        cols = _read_columns(path, fmt)
    mapping = cfg.get("columns") or {}
    scale = cfg.get("scale") or {}

    def col(model_name: str) -> np.ndarray | None:
        key = mapping.get(model_name, model_name)
        if key not in cols:
            return None
        return np.asarray(cols[key], float) * float(scale.get(model_name, 1.0))

    inputs = {}
    for n in input_names:
        if n == "logp":  # parametric 1D problem: log of the parameter column (or the case value)
            pname = cfg.get("parameter", "Pe")
            v = col(pname)
            if v is None and pname in case_params:
                v = np.full(len(next(iter(cols.values()))), float(case_params[pname]))
            if v is None:
                raise ValueError(f"user_data.sources.{name}: parametric problem needs a '{pname}' column")
            inputs[n] = np.log(v)
            continue
        if n.startswith("log") and n != "logp":  # parametric 2D input log<Param>: column <Param> (or a fixed value)
            pname = n[3:]
            v = col(pname)
            if v is None and pname in case_params:
                v = np.full(len(next(iter(cols.values()))), float(case_params[pname]))
            if v is None:
                raise ValueError(f"user_data.sources.{name}: parametric problem needs a '{pname}' column")
            inputs[n] = np.log(v)
            continue
        v = col(n)
        if v is None and n in case_params and n not in ("x", "y", "t"):
            v = np.full(len(next(iter(cols.values()))), float(case_params[n]))
        if v is None:
            raise ValueError(
                f"user_data.sources.{name}: input '{n}' not found (file columns {sorted(cols)}); "
                f"map it with columns: {{{n}: <name>}}"
            )
        inputs[n] = v
    fields = {f: v for f in field_names if (v := col(f)) is not None}
    if not fields:
        raise ValueError(
            f"user_data.sources.{name}: none of the fields {field_names} found (file columns {sorted(cols)}); "
            "map them with columns:"
        )
    N = len(next(iter(inputs.values())))
    ok = np.ones(N, bool)
    for j, n in enumerate(input_names):
        ok &= (inputs[n] >= lo[j] - 1e-12) & (inputs[n] <= hi[j] + 1e-12)
    for v in fields.values():
        ok &= np.isfinite(v)
    meta["n_file"], meta["n_dropped"] = int(N), int((~ok).sum())
    idx = np.flatnonzero(ok)
    mp = int(cfg.get("max_points", 0) or 0)
    if 0 < mp < idx.size:
        idx = np.sort(np.random.default_rng(seed + 4242).choice(idx, size=mp, replace=False))
    if idx.size == 0:
        raise ValueError(
            f"user_data.sources.{name}: no points inside the problem domain {list(zip(lo, hi, strict=True))}"
        )
    return UserData(
        name,
        {k: v[idx] for k, v in inputs.items()},
        {k: v[idx] for k, v in fields.items()},
        cfg.get("role", "evaluate"),
        meta,
    )


# --- model predictions at data points -------------------------------------------------------------------------------
def field_names(problem: Any) -> list[str]:
    return ["u", "v", "p", "theta"] if hasattr(problem, "fields") else ["theta"]


def problem_params(problem: Any, case: dict) -> dict[str, float]:
    if hasattr(problem, "params"):  # 2D; a parametric problem's swept parameters must come from the data columns
        swept = set(getattr(problem, "names", []))
        return {k: float(v) for k, v in problem.params.items() if k not in swept}
    out = {}
    if case.get("value") is not None:
        out[problem.param_name] = float(case["value"])
    return out


def predict(problem: Any, model: torch.nn.Module, X: Tensor) -> dict[str, Tensor]:
    if hasattr(problem, "fields"):  # 2D
        d = problem.fields(model, X, order=0)
        return {k: d[k] for k in ("u", "v", "p", "theta")}
    return {"theta": model(X).squeeze(-1)}


@dataclass
class DataTerm:
    """Fixed data points and targets; ``vectors(model)`` gives per-point residuals for the training loss."""

    source: str
    problem: Any
    X: Tensor
    targets: dict[str, Tensor]

    def vectors(self, model: torch.nn.Module) -> dict[str, Tensor]:
        pred = predict(self.problem, model, self.X)
        return {f"data_{self.source}_{k}": pred[k] - t for k, t in self.targets.items()}


def _X(problem: Any, ud: UserData, device: Any, dtype: Any) -> Tensor:
    names = list(problem.input_names)
    return torch.as_tensor(np.stack([ud.inputs[n] for n in names], 1), dtype=dtype, device=device)


def attach(
    spec: dict, case: dict, problem: Any, device: Any, dtype: Any, seed: int
) -> tuple[list[DataTerm], list[UserData]]:
    """(training terms, evaluation sets) for the sources named in ``case['data']``."""
    names = case.get("data") or []
    names = [names] if isinstance(names, str) else list(names)
    sources = (spec.get("user_data") or {}).get("sources") or {}
    terms, evals = [], []
    for n in names:
        if n not in sources:
            raise ValueError(f"case data '{n}' is not defined under user_data.sources (defined: {sorted(sources)})")
        ud = load_source(
            n,
            sources[n],
            list(problem.input_names),
            field_names(problem),
            problem_params(problem, case),
            list(problem.lo),
            list(problem.hi),
            seed,
        )
        if ud.role in ("train", "both"):
            X = _X(problem, ud, device, dtype)
            terms.append(
                DataTerm(
                    n, problem, X, {k: torch.as_tensor(v, dtype=dtype, device=device) for k, v in ud.fields.items()}
                )
            )
        if ud.role in ("evaluate", "both"):
            evals.append(ud)
    return terms, evals


def evaluate(problem: Any, model: torch.nn.Module, ud: UserData, chunk: int = 4096) -> dict[str, Any]:
    """Errors of the model at the user's points; pressure compared mean-free (gauge-invariant)."""
    p0 = next(model.parameters())
    X = _X(problem, ud, p0.device, p0.dtype)
    preds: dict[str, list[np.ndarray]] = {k: [] for k in ud.fields}
    for i in range(0, X.shape[0], chunk):
        out = predict(problem, model, X[i : i + chunk])
        for k in ud.fields:
            preds[k].append(out[k].detach().cpu().numpy())
    res: dict[str, Any] = {
        "role": ud.role,
        "n": ud.n,
        **{k: v for k, v in ud.meta.items() if k in ("n_file", "n_dropped", "format", "phase2_index", "phase2_time")},
    }
    for k, ref in ud.fields.items():
        pr = np.concatenate(preds[k])
        r = ref.copy()
        if k == "p":
            r, pr = r - r.mean(), pr - pr.mean()
        res[f"rel_l2_{k}"] = float(np.linalg.norm(pr - r) / max(np.linalg.norm(r), 1e-300))
        res[f"linf_{k}"] = float(np.abs(pr - r).max())
    return res


def evaluate_all(problem: Any, model: torch.nn.Module, evals: list[UserData]) -> dict[str, dict]:
    return {ud.name: evaluate(problem, model, ud) for ud in evals}


def describe_sources(spec: dict) -> str:
    s = (spec.get("user_data") or {}).get("sources") or {}
    return "user_data: " + (
        json.dumps({k: {kk: v[kk] for kk in ("file", "role") if kk in v} for k, v in s.items()}) if s else "none"
    )
