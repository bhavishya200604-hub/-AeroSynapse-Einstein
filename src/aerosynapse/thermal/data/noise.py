"""Measurement-noise models for synthetic sensor data (reproducible: every call takes a seed or Generator).

* ``gaussian``  : y = x + eps,  eps ~ N(0, (level * s_ref)^2)   with s_ref = RMS of the CLEAN signal over the whole
                  sample passed in (so 5 % means noise RMS = 5 % of signal RMS; a global absolute sensor noise).
* ``relative``  : y = x (1 + level * z), z ~ N(0, 1)              (multiplicative, per-reading proportional noise).
* ``gaussian_range`` : eps ~ N(0, (level * (max - min))^2)         (noise relative to the signal range).
level = 0 returns the clean data unchanged (identical array values).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]
MODELS = ("gaussian", "relative", "gaussian_range")
PAPER_LEVELS = (0.0, 0.01, 0.02, 0.05, 0.10)


def rng_of(seed: int | np.random.Generator) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def noise_scale(clean: ArrayLike, level: float, model: str = "gaussian") -> Array:
    """Per-reading noise standard deviation (array broadcastable to ``clean``)."""
    x = np.asarray(clean, float)
    if model == "gaussian":
        return np.full(x.shape, level * float(np.sqrt(np.mean(x**2))))
    if model == "relative":
        return level * np.abs(x)
    if model == "gaussian_range":
        return np.full(x.shape, level * float(np.ptp(x)))
    raise ValueError(f"model must be one of {MODELS}")


def add_noise(clean: ArrayLike, level: float, model: str = "gaussian", seed: int | np.random.Generator = 0) -> Array:
    """Noisy copy of ``clean``."""
    x = np.asarray(clean, float)
    if level < 0:
        raise ValueError("level must be >= 0")
    if level == 0.0:
        return x.copy()
    return x + noise_scale(x, level, model) * rng_of(seed).standard_normal(x.shape)
