"""Parameter-space design: Sobol/LHS sampling in log-space, physical mapping, parameter-based splits.

Unit cube u in [0,1]^d  <->  physical p_k = lo_k (hi_k/lo_k)^u_k (log-uniform). Train/val/interp-test samples
are drawn from the training box u <= u_lim; extrapolation samples from the shell where max_k u_k > u_lim. Every
split uses its own scrambled-Sobol seed and candidates closer than ``min_dist`` (unit-cube Euclidean) to an
already accepted point of ANY split are rejected, so no parameter point is shared between splits
(see :func:`check_leakage`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

Array = NDArray[np.float64]
SPLITS = ("train", "val", "interp", "extrap")


@dataclass(frozen=True)
class Dim:
    """One swept parameter: range [lo, hi] on a 'log' (log-uniform) or 'linear' scale."""

    name: str
    lo: float
    hi: float
    scale: str = "log"

    def __post_init__(self) -> None:
        if self.scale not in ("log", "linear"):
            raise ValueError(f"parameter {self.name}: scale must be 'log' or 'linear', got {self.scale!r}")
        if not self.lo < self.hi:
            raise ValueError(f"parameter {self.name}: need lo < hi, got [{self.lo}, {self.hi}]")
        if self.scale == "log" and self.lo <= 0:
            raise ValueError(f"parameter {self.name}: log scale requires lo > 0 (got {self.lo}); use scale: linear")

    @property
    def is_log(self) -> bool:
        return self.scale == "log"

    def to_phys(self, u: Array) -> Array:
        u = np.asarray(u)
        return self.lo * (self.hi / self.lo) ** u if self.is_log else self.lo + (self.hi - self.lo) * u

    def to_unit(self, p: Array) -> Array:
        p = np.asarray(p)
        return np.log(p / self.lo) / np.log(self.hi / self.lo) if self.is_log else (p - self.lo) / (self.hi - self.lo)


@dataclass
class CaseDesign:
    name: str
    dims: list[Dim]
    u: Array  # (N, d) unit coordinates, ordered train | val | interp | extrap
    split: Array  # (N,) integer index into SPLITS
    u_lim: float
    min_dist: float

    @property
    def d(self) -> int:
        return len(self.dims)

    @property
    def params(self) -> Array:
        return np.stack([dm.to_phys(self.u[:, k]) for k, dm in enumerate(self.dims)], axis=1)

    def idx(self, name: str) -> Array:
        return np.flatnonzero(self.split == SPLITS.index(name))

    def extrap_distance(self) -> Array:
        """Extrapolation distance in unit-cube coordinates: max_k (u_k - u_lim)_+ / (1 - u_lim), in [0, 1]
        (0 inside the training box, 1 at the edge of the full box)."""
        return np.max(np.maximum(self.u - self.u_lim, 0.0), axis=1) / (1.0 - self.u_lim)


def sobol_unit(n: int, d: int, seed: int) -> Array:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # balance warning for non-power-of-two n
        return qmc.Sobol(d, scramble=True, seed=seed).random(n)


def lhs_unit(n: int, d: int, seed: int) -> Array:
    return qmc.LatinHypercube(d, seed=seed).random(n)


def _min_dist(c: Array, pool: Array) -> Array:
    if pool.shape[0] == 0:
        return np.full(c.shape[0], np.inf)
    return np.min(np.linalg.norm(c[:, None, :] - pool[None, :, :], axis=2), axis=1)


def design_case(
    name: str,
    dims: list[Dim],
    sizes: dict[str, int],
    seed: int = 0,
    u_lim: float = 0.8,
    method: str = "sobol",
    min_dist_factor: float = 0.3,
) -> CaseDesign:
    """Draw the four splits. Sizes: {'train','val','interp','extrap'} point counts (0 allowed except train)."""
    d = len(dims)
    n_tr = max(sizes["train"], 1)
    min_dist = min_dist_factor * u_lim * (1.0 / n_tr) ** (1.0 / d)
    sampler = sobol_unit if method == "sobol" else lhs_unit
    accepted: list[Array] = []
    labels: list[int] = []
    pool = np.zeros((0, d))
    for si, split in enumerate(SPLITS):
        n = int(sizes.get(split, 0))
        got: list[Array] = []
        for attempt in range(20):
            cand = sampler(max(8 * n, 64) * (attempt + 1), d, seed * 1000 + 17 * si + attempt)
            if split == "extrap":
                cand = cand[np.max(cand, axis=1) > u_lim + 1e-9]
            else:
                cand = cand * u_lim
            for c in cand:
                if len(got) == n:
                    break
                ref = np.vstack([pool] + ([np.array(got)] if got else [])) if (pool.size or got) else pool
                if _min_dist(c[None, :], ref)[0] >= min_dist:
                    got.append(c)
            if len(got) == n:
                break
        if len(got) < n:
            raise RuntimeError(f"could not place {n} {split} points with min_dist={min_dist:.3f}")
        # order deterministically along the first coordinate within a split for readability
        arr = np.array(got).reshape(-1, d)
        arr = arr[np.argsort(arr[:, 0], kind="stable")]
        accepted.append(arr)
        labels += [si] * n
        pool = np.vstack([pool, arr])
    return CaseDesign(name, dims, np.vstack(accepted), np.array(labels, int), u_lim, min_dist)


def check_leakage(des: CaseDesign) -> dict[str, bool | float]:
    """Parameter-level leakage checks (all must be True)."""
    u, s = des.u, des.split
    tr, va, it, ex = (u[s == i] for i in range(4))
    inbox = lambda a: bool(np.all(a <= des.u_lim + 1e-12))  # noqa: E731
    out: dict[str, bool | float] = {
        "all_points_unique": bool(np.unique(np.round(u, 12), axis=0).shape[0] == u.shape[0]),
        "train_val_interp_inside_train_box": inbox(tr) and inbox(va) and inbox(it),
        "extrap_all_outside_train_box": bool(np.all(np.max(ex, axis=1) > des.u_lim)) if ex.size else True,
    }
    worst = np.inf
    for name, a, b in (
        ("val_vs_train", va, tr),
        ("interp_vs_train", it, tr),
        ("interp_vs_val", it, va),
        ("extrap_vs_train", ex, tr),
        ("extrap_vs_val", ex, va),
        ("extrap_vs_interp", ex, it),
    ):
        if a.size and b.size:
            m = float(np.min(_min_dist(a, b)))
            out[f"min_dist_{name}"] = m
            worst = min(worst, m)
    out["min_cross_split_distance_ge_threshold"] = bool(worst >= des.min_dist - 1e-12)
    out["all_ok"] = all(v for k, v in out.items() if isinstance(v, bool))
    return out
