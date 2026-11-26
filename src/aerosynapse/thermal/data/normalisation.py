"""Normalisation: network inputs to [-1, 1] with FIXED design bounds; field statistics from the TRAIN split only.

Inputs  : (x, y, t, log p_1, ..., log p_d). Coordinates and time map their domain [lo, hi] linearly; each
          parameter maps log(lo)..log(hi) of the full design box (fixed by the design, not by the data).
Fields  : z-score with mean/std computed on the train points only (so no val/test/extrapolation information enters).
Persisted as JSON next to the dataset (``normalisation.json``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerosynapse.thermal.data.design import CaseDesign

Array = NDArray[np.float64]


@dataclass
class Normaliser:
    input_names: list[str]
    input_lo: list[float]  # in transformed space (log for parameters)
    input_hi: list[float]
    input_is_log: list[bool]
    field_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    fitted_on: str = "train"
    n_fit_points: int = 0

    # --- inputs -----------------------------------------------------------------------------
    def _tf(self, k: int, v: ArrayLike) -> Array:
        v = np.asarray(v, float)
        return np.log(v) if self.input_is_log[k] else v

    def normalise_inputs(self, **cols: ArrayLike) -> Array:
        """Stack the named columns (physical units) into an (n, d_in) array in [-1, 1] inside the design box."""
        out = []
        for k, name in enumerate(self.input_names):
            lo, hi = self.input_lo[k], self.input_hi[k]
            out.append(2.0 * (self._tf(k, cols[name]) - lo) / (hi - lo) - 1.0)
        return np.stack(np.broadcast_arrays(*out), axis=-1)

    def denormalise_inputs(self, z: ArrayLike) -> dict[str, Array]:
        z = np.asarray(z, float)
        out = {}
        for k, name in enumerate(self.input_names):
            v = 0.5 * (z[..., k] + 1.0) * (self.input_hi[k] - self.input_lo[k]) + self.input_lo[k]
            out[name] = np.exp(v) if self.input_is_log[k] else v
        return out

    # --- fields -----------------------------------------------------------------------------
    def normalise_field(self, name: str, f: ArrayLike) -> Array:
        s = self.field_stats[name]
        return (np.asarray(f, float) - s["mean"]) / s["std"]

    def denormalise_field(self, name: str, f: ArrayLike) -> Array:
        s = self.field_stats[name]
        return np.asarray(f, float) * s["std"] + s["mean"]

    # --- persistence ------------------------------------------------------------------------
    def to_json(self) -> dict:
        return {
            k: getattr(self, k)
            for k in ("input_names", "input_lo", "input_hi", "input_is_log", "field_stats", "fitted_on", "n_fit_points")
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Normaliser:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def fit_normaliser(
    design: CaseDesign, coord_bounds: dict[str, tuple[float, float]], fields: dict[str, Array]
) -> Normaliser:
    """Build the normaliser. ``fields[name]`` has the parameter point on axis 0; stats use design.idx('train') only.

    coord_bounds: e.g. {'x': (0, 1), 't': (0.05, 0.5)} (linear maps), added before the parameter inputs.
    """
    names = list(coord_bounds) + [d.name for d in design.dims]
    lo = [coord_bounds[k][0] for k in coord_bounds] + [float(np.log(d.lo) if d.is_log else d.lo) for d in design.dims]
    hi = [coord_bounds[k][1] for k in coord_bounds] + [float(np.log(d.hi) if d.is_log else d.hi) for d in design.dims]
    is_log = [False] * len(coord_bounds) + [d.is_log for d in design.dims]
    tr = design.idx("train")
    stats = {}
    for name, f in fields.items():
        a = np.asarray(f)[tr]
        std = float(a.std())
        stats[name] = {
            "mean": float(a.mean()),
            "std": std if std > 0 else 1.0,
            "min": float(a.min()),
            "max": float(a.max()),
        }
    return Normaliser(names, lo, hi, is_log, stats, "train", int(tr.size))
