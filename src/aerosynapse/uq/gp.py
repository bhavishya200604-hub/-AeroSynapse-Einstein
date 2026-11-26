r"""Gaussian-process regression with an anisotropic RBF kernel and Gaussian noise.

:math:`k(x,x')=s^2\exp(-\tfrac12\sum_d (x_d-x'_d)^2/\ell_d^2)`; hyper-parameters
:math:`(\log s,\log\ell,\log\sigma_n)` maximise the log marginal likelihood (L-BFGS-B, restarts).
Estimates *epistemic* uncertainty of the regression function plus *aleatoric* noise ``sigma_n``;
assumes stationarity and Gaussian noise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

FloatArr = NDArray[np.float64]


class GaussianProcess:
    def __init__(self, noise: float | None = None, restarts: int = 3, seed: int = 0, normalize_y: bool = True) -> None:
        self.fixed_noise, self.restarts, self.seed, self.normalize_y = noise, restarts, seed, normalize_y

    def _k(self, A: FloatArr, B: FloatArr, s: float, ell: FloatArr) -> FloatArr:
        d = (A[:, None, :] - B[None, :, :]) / ell
        return np.asarray(s**2 * np.exp(-0.5 * np.sum(d**2, -1)))

    def _nll(self, th: FloatArr) -> float:
        d = self.X.shape[1]
        s, ell = np.exp(th[0]), np.exp(th[1 : 1 + d])
        sn = self.fixed_noise if self.fixed_noise is not None else np.exp(th[1 + d])
        K = self._k(self.X, self.X, s, ell) + (sn**2 + 1e-10) * np.eye(len(self.X))
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e25
        a = cho_solve(c, self.y)
        return float(0.5 * self.y @ a + np.sum(np.log(np.diag(c[0]))) + 0.5 * len(self.y) * np.log(2 * np.pi))

    def fit(self, X: FloatArr, y: FloatArr) -> GaussianProcess:
        self.X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float).ravel()
        self.ym, self.ys = (float(y.mean()), float(y.std() or 1.0)) if self.normalize_y else (0.0, 1.0)
        self.y = (y - self.ym) / self.ys
        d = self.X.shape[1]
        rng = np.random.default_rng(self.seed)
        span = np.ptp(self.X, 0) + 1e-12
        best = None
        for r in range(self.restarts):
            th0 = np.concatenate([[0.0], np.log(span * (0.3 if r == 0 else rng.uniform(0.05, 1.0, d)))])
            if self.fixed_noise is None:
                th0 = np.append(th0, np.log(0.1))
            res = minimize(
                self._nll,
                th0,
                method="L-BFGS-B",
                bounds=[(-5, 5)]
                + [(np.log(span[i] * 1e-3), np.log(span[i] * 1e2)) for i in range(d)]
                + ([(-12, 2)] if self.fixed_noise is None else []),
            )
            if best is None or res.fun < best.fun:
                best = res
        assert best is not None
        th = best.x
        self.s, self.ell = float(np.exp(th[0])), np.exp(th[1 : 1 + d])
        self.sn = self.fixed_noise if self.fixed_noise is not None else float(np.exp(th[1 + d]))
        K = self._k(self.X, self.X, self.s, self.ell) + (self.sn**2 + 1e-10) * np.eye(len(self.X))
        self.c = cho_factor(K, lower=True)
        self.alpha = cho_solve(self.c, self.y)
        self.log_marginal_likelihood = -float(best.fun)
        return self

    @property
    def noise_std(self) -> float:
        """Estimated observation-noise standard deviation in the units of y."""
        return float(self.sn * self.ys)

    def predict(self, Xs: FloatArr, include_noise: bool = False) -> tuple[FloatArr, FloatArr]:
        """Posterior mean and standard deviation (latent f, or y if ``include_noise``)."""
        Xs = np.atleast_2d(np.asarray(Xs, float))
        Ks = self._k(Xs, self.X, self.s, self.ell)
        mu = Ks @ self.alpha
        v = cho_solve(self.c, Ks.T)
        var = np.clip(self.s**2 - np.sum(Ks * v.T, 1), 1e-300, None)
        if include_noise:
            var = var + self.sn**2
        return mu * self.ys + self.ym, np.sqrt(var) * self.ys
