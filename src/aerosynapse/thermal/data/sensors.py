"""Sparse sensor layouts and sampling of snapshot fields at sensor locations (inverse-problem data).

Layouts return coordinates in the unit domain [0,1] (1D) or [0,1]^2 (2D). Fields are interpolated linearly
(1D, ``np.interp``) or bilinearly (2D) from their cell-centre/vertex grids; 2D samples outside the cell-centre
hull are clamped to the nearest cell centre (sensors are placed inside [dx/2, 1-dx/2] by construction).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RegularGridInterpolator

from aerosynapse.thermal.data.design import sobol_unit
from aerosynapse.thermal.data.noise import add_noise

Array = NDArray[np.float64]
LAYOUTS_1D = ("uniform", "random", "clustered_right", "clustered_left")
LAYOUTS_2D = ("grid", "sobol", "wall_clustered", "centrelines")


def layout_1d(kind: str, n: int, seed: int = 0, margin: float = 0.0) -> Array:
    """n sensor positions in [margin, 1 - margin]. clustered_right: dense near x=1 (boundary layer of Case A)."""
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        x = np.linspace(0.0, 1.0, n + 2)[1:-1]
    elif kind == "random":
        x = np.sort(rng.uniform(0.0, 1.0, n))
    elif kind == "clustered_right":
        x = 1.0 - np.linspace(0.0, 1.0, n + 2)[1:-1] ** 2.5
    elif kind == "clustered_left":
        x = np.linspace(0.0, 1.0, n + 2)[1:-1] ** 2.5
    else:
        raise ValueError(f"kind must be one of {LAYOUTS_1D}")
    return np.sort(margin + (1 - 2 * margin) * x)


def layout_2d(kind: str, n: int, seed: int = 0, margin: float = 0.02) -> Array:
    """~n sensor positions (n x 2 array) in [margin, 1-margin]^2 (exactly n except 'grid': floor(sqrt n)^2)."""
    rng = np.random.default_rng(seed)
    if kind == "grid":
        k = max(int(np.floor(np.sqrt(n))), 1)
        g = np.linspace(0.0, 1.0, k + 2)[1:-1]
        pts = np.array([(a, b) for b in g for a in g])
    elif kind == "sobol":
        pts = sobol_unit(n, 2, seed)
    elif kind == "wall_clustered":
        side = rng.integers(0, 4, n)
        t = rng.uniform(0.0, 1.0, n)
        d = rng.uniform(0.0, 0.12, n)
        pts = np.stack(
            [
                np.where(side == 0, d, np.where(side == 1, 1 - d, t)),
                np.where(side == 2, d, np.where(side == 3, 1 - d, t)),
            ],
            axis=1,
        )
    elif kind == "centrelines":
        s = np.linspace(0.0, 1.0, n // 2 + 2)[1:-1]
        pts = np.vstack([np.stack([s, np.full_like(s, 0.5)], 1), np.stack([np.full_like(s, 0.5), s], 1)])[:n]
    else:
        raise ValueError(f"kind must be one of {LAYOUTS_2D}")
    return margin + (1 - 2 * margin) * pts


def sample_1d(x_grid: ArrayLike, field: ArrayLike, xs: ArrayLike) -> Array:
    """field[..., nx] sampled at xs -> [..., n_sensors] (linear interpolation)."""
    f = np.asarray(field, float)
    flat = f.reshape(-1, f.shape[-1])
    out = np.stack([np.interp(xs, x_grid, row) for row in flat])
    return out.reshape(*f.shape[:-1], len(xs))


def sample_2d(x_grid: ArrayLike, y_grid: ArrayLike, field: ArrayLike, pts: ArrayLike) -> Array:
    """field[..., ny, nx] sampled at pts (n,2) -> [..., n_sensors] (bilinear; clamped to the grid hull)."""
    f = np.asarray(field, float)
    pts = np.asarray(pts, float)
    xg, yg = np.asarray(x_grid), np.asarray(y_grid)
    q = np.stack([np.clip(pts[:, 1], yg[0], yg[-1]), np.clip(pts[:, 0], xg[0], xg[-1])], axis=1)
    flat = f.reshape(-1, *f.shape[-2:])
    out = np.stack([RegularGridInterpolator((yg, xg), a, method="linear")(q) for a in flat])
    return out.reshape(*f.shape[:-2], len(pts))


def make_sensor_data(clean_fn_values: Array, level: float, model: str, seed: int) -> dict[str, Array]:
    """Clean -> noisy pair for readings array (any shape); the noise stream depends only on ``seed``."""
    return {
        "clean": np.asarray(clean_fn_values, float),
        "noisy": add_noise(clean_fn_values, level, model, seed),
    }
