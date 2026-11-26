"""HDF5 dataset writing/reading with provenance (config, git hash, checksums).

Layout (one file per case)::

    /params            (N, d)  physical parameters (columns = attrs['param_names'])
    /params_unit       (N, d)  unit-cube design coordinates
    /split             (N,)    0 train | 1 val | 2 interp | 3 extrap   (attrs['split_names'])
    /extrap_distance   (N,)    unit-cube overshoot beyond the training box (0 inside)
    /coords/{x,y,t}    grids (x,y: cell centres for 2D; vertices for 1D)
    /fields/<name>     (N, nt, [ny,] nx)  float64, gzip
    /scalars/<name>    (N, ...)
    attrs: case, config_json, git_hash, param_names, split_names, u_lim, min_dist, ...
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from aerosynapse.thermal.data.design import SPLITS, CaseDesign, Dim
from aerosynapse.thermal.utils.experiment import git_hash

Array = NDArray[np.float64]


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_dataset(
    path: str | Path,
    design: CaseDesign,
    coords: dict[str, Array],
    fields: dict[str, Array],
    scalars: dict[str, Array] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("params", data=design.params)
        f.create_dataset("params_unit", data=design.u)
        f.create_dataset("split", data=design.split)
        f.create_dataset("extrap_distance", data=design.extrap_distance())
        for k, v in coords.items():
            f.create_dataset(f"coords/{k}", data=np.asarray(v, float))
        for k, v in fields.items():
            f.create_dataset(
                f"fields/{k}", data=np.asarray(v, float), compression="gzip", compression_opts=4, shuffle=True
            )
        for k, v in (scalars or {}).items():
            f.create_dataset(f"scalars/{k}", data=np.asarray(v))
        f.attrs.update(
            {
                "case": design.name,
                "param_names": json.dumps([d.name for d in design.dims]),
                "param_lo": json.dumps([d.lo for d in design.dims]),
                "param_hi": json.dumps([d.hi for d in design.dims]),
                "param_scale": json.dumps([d.scale for d in design.dims]),
                "split_names": json.dumps(list(SPLITS)),
                "u_lim": design.u_lim,
                "min_dist": design.min_dist,
                "git_hash": git_hash(),
                "config_json": json.dumps(config or {}, default=str),
            }
        )
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


@dataclass
class Dataset:
    design: CaseDesign
    coords: dict[str, Array]
    fields: dict[str, Array]
    scalars: dict[str, Array]
    attrs: dict[str, Any]

    def split_idx(self, name: str) -> Array:
        return self.design.idx(name)


def read_dataset(path: str | Path) -> Dataset:
    with h5py.File(path, "r") as f:
        names = json.loads(f.attrs["param_names"])
        lo, hi = json.loads(f.attrs["param_lo"]), json.loads(f.attrs["param_hi"])
        scales = json.loads(f.attrs["param_scale"]) if "param_scale" in f.attrs else ["log"] * len(names)
        dims = [Dim(n, a, b, s) for n, a, b, s in zip(names, lo, hi, scales, strict=True)]
        des = CaseDesign(
            str(f.attrs["case"]),
            dims,
            f["params_unit"][:],
            f["split"][:],
            float(f.attrs["u_lim"]),
            float(f.attrs["min_dist"]),
        )
        coords = {k: f["coords"][k][:] for k in f["coords"]}
        fields = {k: f["fields"][k][:] for k in f["fields"]}
        scalars = {k: f["scalars"][k][:] for k in f["scalars"]} if "scalars" in f else {}
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
    return Dataset(des, coords, fields, scalars, attrs)
