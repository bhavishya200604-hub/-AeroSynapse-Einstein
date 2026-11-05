r"""Collocation samplers under a fixed point budget ``n``.

Non-adaptive: ``uniform`` (fixed), ``uniform_resample``, ``lhs``, ``sobol`` (scrambled).
Residual-based adaptive:

* ``rar``  — residual-based adaptive refinement (Lu et al. 2021): start with n/2 uniform points, add
  the top-k residual points from a uniform candidate pool at each update until n points are reached.
* ``rad``  — residual-based adaptive distribution (Wu et al. 2023): resample all n points from
  :math:`p(x)\propto |r|^k/E|r|^k + c` over a candidate pool (k = 1, c = 1 by default).
* ``r3``   — retain–resample–release (Daw et al. 2023): keep points with residual above the mean,
  replace the rest with fresh uniform points.

Every sampler counts the residual evaluations it spends (``extra_evals``), so compute is comparable.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

FloatArr = NDArray[np.float64]
ResidualFn = Callable[[FloatArr], FloatArr]


class Sampler:
    adaptive = False

    def __init__(
        self,
        lo: list[float],
        hi: list[float],
        n: int,
        seed: int = 0,
        inside: Callable[[FloatArr], NDArray[np.bool_]] | None = None,
        pool_factor: int = 4,
    ) -> None:
        if n < 2:
            raise ValueError("budget n must be >= 2")
        self.lo, self.hi, self.n = np.asarray(lo, float), np.asarray(hi, float), n
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.inside = inside
        self.pool_factor = pool_factor
        self.extra_evals = 0

    def _scale(self, U: FloatArr) -> FloatArr:
        return self.lo + U * (self.hi - self.lo)

    def uniform(self, m: int) -> FloatArr:
        if self.inside is None:
            return self._scale(self.rng.random((m, self.lo.size)))
        out: list[FloatArr] = []
        got = 0
        while got < m:
            c = self._scale(self.rng.random((2 * m, self.lo.size)))
            c = c[self.inside(c)]
            out.append(c)
            got += len(c)
        return np.vstack(out)[:m]

    def initial(self) -> FloatArr:
        return self.uniform(self.n)

    def update(self, X: FloatArr, residual: ResidualFn, step: int) -> FloatArr:
        return X


class UniformResample(Sampler):
    def update(self, X: FloatArr, residual: ResidualFn, step: int) -> FloatArr:
        return self.uniform(self.n)


class LHS(Sampler):
    def initial(self) -> FloatArr:
        return self._scale(qmc.LatinHypercube(d=self.lo.size, seed=self.seed).random(self.n))


class Sobol(Sampler):
    def initial(self) -> FloatArr:
        s = qmc.Sobol(d=self.lo.size, scramble=True, seed=self.seed)
        m = int(np.ceil(np.log2(self.n)))
        return self._scale(s.random_base2(m)[: self.n])


class RAR(Sampler):
    adaptive = True

    def __init__(self, *a: object, n_add: int | None = None, **k: object) -> None:
        super().__init__(*a, **k)  # type: ignore[arg-type]
        self.n_add = n_add

    def initial(self) -> FloatArr:
        return self.uniform(self.n // 2)

    def update(self, X: FloatArr, residual: ResidualFn, step: int) -> FloatArr:
        if len(X) >= self.n:
            return X
        k = min(self.n_add or max(1, self.n // 10), self.n - len(X))
        cand = self.uniform(self.pool_factor * self.n)
        r = np.abs(residual(cand))
        self.extra_evals += len(cand)
        return np.vstack([X, cand[np.argsort(r)[-k:]]])


class RAD(Sampler):
    adaptive = True

    def __init__(self, *a: object, k: float = 1.0, c: float = 1.0, **kw: object) -> None:
        super().__init__(*a, **kw)  # type: ignore[arg-type]
        self.k, self.c = k, c

    def update(self, X: FloatArr, residual: ResidualFn, step: int) -> FloatArr:
        cand = self.uniform(self.pool_factor * self.n)
        r = np.abs(residual(cand)) ** self.k
        self.extra_evals += len(cand)
        p = r / max(float(r.mean()), 1e-300) + self.c
        p = p / p.sum()
        idx = self.rng.choice(len(cand), size=self.n, replace=False, p=p)
        return cand[idx]


class R3(Sampler):
    adaptive = True

    def update(self, X: FloatArr, residual: ResidualFn, step: int) -> FloatArr:
        r = np.abs(residual(X))
        self.extra_evals += len(X)
        keep = X[r > r.mean()]
        return np.vstack([keep, self.uniform(self.n - len(keep))])


SAMPLERS: dict[str, type[Sampler]] = {
    "uniform": Sampler,
    "uniform_resample": UniformResample,
    "lhs": LHS,
    "sobol": Sobol,
    "rar": RAR,
    "rad": RAD,
    "r3": R3,
}


def make_sampler(name: str, lo: list[float], hi: list[float], n: int, seed: int = 0, **kw: object) -> Sampler:
    try:
        cls = SAMPLERS[name]
    except KeyError:
        raise KeyError(f"unknown sampler {name!r}; available {sorted(SAMPLERS)}") from None
    return cls(lo, hi, n, seed, **kw)  # type: ignore[arg-type]
