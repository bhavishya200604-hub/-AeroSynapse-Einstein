"""Deep ensembles (Lakshminarayanan et al. 2017): M independently initialised models; predictive
mean and spread. The spread estimates *epistemic* (model) uncertainty only; it must be calibrated
(see :mod:`aerosynapse.uq.calibration`) before being read as a probability."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class Ensemble:
    def __init__(self, members: list[Any], predict: Callable[[Any, np.ndarray], np.ndarray]) -> None:
        self.members, self._predict = members, predict

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P = np.stack([self._predict(m, X) for m in self.members])
        return P.mean(0), P.std(0, ddof=1)

    @classmethod
    def train(
        cls, factory: Callable[[int], Any], predict: Callable[[Any, np.ndarray], np.ndarray], m: int = 5
    ) -> Ensemble:
        return cls([factory(s) for s in range(m)], predict)
