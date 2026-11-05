r"""EXPERIMENTAL (P3) demonstration plugin: Hamiltonian harmonic oscillator.

:math:`H(q,p)=p^2/(2m)+kq^2/2`. Compares explicit Euler (energy grows like :math:`(1+\omega^2\Delta t^2)^n`)
with the symplectic Störmer–Verlet scheme (energy error bounded, O(Δt²), no secular drift).
Registered in :data:`aerosynapse.registry.PHYSICS` as ``"harmonic_oscillator"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aerosynapse.registry import PHYSICS

FloatArr = NDArray[np.float64]


@dataclass
class Oscillator:
    m: float = 1.0
    k: float = 1.0
    maturity: str = "experimental (P3)"

    def energy(self, q: FloatArr, p: FloatArr) -> FloatArr:
        return np.asarray(p**2 / (2 * self.m) + 0.5 * self.k * q**2)

    def integrate(self, q0: float, p0: float, dt: float, n: int, scheme: str = "verlet") -> tuple[FloatArr, FloatArr]:
        q, p = np.empty(n + 1), np.empty(n + 1)
        q[0], p[0] = q0, p0
        for i in range(n):
            if scheme == "euler":
                q[i + 1] = q[i] + dt * p[i] / self.m
                p[i + 1] = p[i] - dt * self.k * q[i]
            elif scheme == "verlet":
                ph = p[i] - 0.5 * dt * self.k * q[i]
                q[i + 1] = q[i] + dt * ph / self.m
                p[i + 1] = ph - 0.5 * dt * self.k * q[i + 1]
            else:
                raise ValueError(f"unknown scheme {scheme}")
        return q, p


@PHYSICS.register("harmonic_oscillator")
def harmonic_oscillator(m: float = 1.0, k: float = 1.0) -> Oscillator:
    return Oscillator(m, k)
