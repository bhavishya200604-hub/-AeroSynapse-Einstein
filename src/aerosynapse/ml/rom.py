r"""Reduced-order and simple baselines for operator learning.

* :class:`PODRegressor` — proper orthogonal decomposition (SVD of centred snapshots) with a ridge
  regression from (input POD coefficients, parameters) to output POD coefficients. Linear in the
  input field, so it is *exact* (up to truncation) for linear operators at fixed parameters.
* :class:`CNN1d` — small convolutional encoder baseline.
* :class:`FlatMLP` — fully connected baseline on the flattened grid (fixed resolution).
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

FloatArr = NDArray[np.float64]


class PODRegressor:
    def __init__(self, rank: int = 16, ridge: float = 1e-8, poly_degree: int = 2) -> None:
        self.rank, self.ridge, self.deg = rank, ridge, poly_degree

    def _features(self, a_coef: FloatArr, p: FloatArr) -> FloatArr:
        cols = [np.ones((len(a_coef), 1))]
        P = np.column_stack([p**d for d in range(1, self.deg + 1)]) if p.size else np.zeros((len(a_coef), 0))
        cols += [a_coef, P]
        if P.size:  # bilinear terms: coefficient × parameter powers (parameter-dependent linear operator)
            cols += [a_coef * P[:, [j]] for j in range(P.shape[1])]
        return np.hstack(cols)

    def fit(self, A: FloatArr, P: FloatArr, Y: FloatArr) -> PODRegressor:
        self.am, self.ym = A.mean(0), Y.mean(0)
        _, sa, Va = np.linalg.svd(A - self.am, full_matrices=False)
        _, sy, Vy = np.linalg.svd(Y - self.ym, full_matrices=False)
        self.Va, self.Vy = Va[: self.rank], Vy[: self.rank]
        self.energy = float(np.sum(sy[: self.rank] ** 2) / np.sum(sy**2))
        F = self._features((A - self.am) @ self.Va.T, P)
        C = (Y - self.ym) @ self.Vy.T
        self.W = np.linalg.solve(F.T @ F + self.ridge * np.eye(F.shape[1]), F.T @ C)
        return self

    def predict(self, A: FloatArr, P: FloatArr) -> FloatArr:
        return np.asarray(self._features((A - self.am) @ self.Va.T, P) @ self.W @ self.Vy + self.ym)


class CNN1d(nn.Module):
    def __init__(self, cin: int, width: int = 32, depth: int = 4, kernel: int = 5) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = cin
        for _ in range(depth):
            layers += [nn.Conv1d(c, width, kernel, padding=kernel // 2, padding_mode="circular"), nn.GELU()]
            c = width
        layers.append(nn.Conv1d(c, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)


class FlatMLP(nn.Module):
    def __init__(self, cin: int, n: int, width: int = 256, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        d = cin * n
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU()]
            d = width
        layers.append(nn.Linear(d, n))
        self.net = nn.Sequential(*layers)
        self.n = n

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if a.shape[-1] != self.n:
            raise ValueError("FlatMLP is tied to its training resolution")
        return self.net(a).unsqueeze(1)
