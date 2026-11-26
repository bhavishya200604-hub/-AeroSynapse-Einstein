r"""A library of mathematical curves and surfaces that bodies can be built from.

Euclidean: Lame superellipse, Gielis superformula, Fourier descriptors, Bernstein/Bezier curves,
Euler spiral (clothoid, curvature linear in arc length), Sears-Haack and von Karman area laws,
Joukowski and Karman-Trefftz conformal maps.

Non-Euclidean: these are curved geometries, used here as a source of shapes and of deformations.

* Hyperbolic (constant negative curvature): the pseudosphere, built on the tractrix, has Gaussian
  curvature K = -1/a^2. Moebius maps are the rigid motions of the Poincare disk: applied to a cross
  section they deform it as a Euclidean shape while preserving hyperbolic distance exactly.
* Elliptic (constant positive curvature): spherical caps with K = +1/R^2, and inverse stereographic
  projection, which wraps a planar outline onto a sphere.
* Minimal surfaces (mean curvature zero, K <= 0): the catenoid.

Air does not care which geometry a shape was drawn in. These families matter only because they
give the optimiser a richer set of shapes. Whether they help is an empirical result of a run.

``gaussian_curvature`` computes K from the first and second fundamental forms, which lets the tests
confirm each surface has the curvature its theory says it must.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import comb, fresnel

FloatArr = NDArray[np.float64]
CplxArr = NDArray[np.complex128]


# ------------------------------------------------------------------ Euclidean cross sections
def superellipse(phi: FloatArr, a: float, b: float, n: float = 2.0) -> tuple[FloatArr, FloatArr]:
    """Lame curve |y/a|^n + |z/b|^n = 1 at polar parameter phi."""
    c, s = np.cos(phi), np.sin(phi)
    return a * np.sign(c) * np.abs(c) ** (2 / n), b * np.sign(s) * np.abs(s) ** (2 / n)


def superformula(phi: FloatArr, m: float, n1: float, n2: float, n3: float, a: float = 1.0, b: float = 1.0) -> FloatArr:
    """Gielis superformula: radius at angle phi. One equation spans circles to stars and polygons."""
    t = m * phi / 4
    r = (np.abs(np.cos(t) / a) ** n2 + np.abs(np.sin(t) / b) ** n3) ** (-1.0 / n1)
    return np.asarray(r)


def fourier_radius(phi: FloatArr, a: FloatArr, b: FloatArr) -> FloatArr:
    """r(phi) = 1 + sum a_k cos(k phi) + b_k sin(k phi): any star-shaped outline."""
    r = np.ones_like(phi)
    for k, (ak, bk) in enumerate(zip(a, b, strict=False), start=1):
        r = r + ak * np.cos(k * phi) + bk * np.sin(k * phi)
    return r


def random_fourier(rng: np.random.Generator, modes: int = 6, decay: float = 1.5, amp: float = 0.12) -> tuple[FloatArr, FloatArr]:
    """Random smooth outline coefficients, with spectrum falling as k^-decay."""
    k = np.arange(1, modes + 1, dtype=float)
    return amp * rng.standard_normal(modes) / k**decay, amp * rng.standard_normal(modes) / k**decay


def bernstein(t: FloatArr, ctrl: FloatArr) -> FloatArr:
    """Bezier curve value at t from control values (n,) or control points (n, d)."""
    ctrl = np.asarray(ctrl, float)
    n = len(ctrl) - 1
    B = np.stack([comb(n, i) * t**i * (1 - t) ** (n - i) for i in range(n + 1)], axis=-1)
    return B @ ctrl


def clothoid(s: FloatArr, scale: float = 1.0) -> tuple[FloatArr, FloatArr]:
    """Euler spiral: curvature grows linearly with arc length. Used for smooth blends."""
    S, C = fresnel(s / (scale * np.sqrt(np.pi)))
    k = scale * np.sqrt(np.pi)
    return k * C, k * S


# ------------------------------------------------------------------ Euclidean area laws (x in [0, 1])
def sears_haack(x: FloatArr) -> FloatArr:
    """Normalised radius of the minimum wave drag body for given length and volume."""
    return np.asarray(np.clip(4 * x * (1 - x), 0, None) ** 0.75)


def von_karman(x: FloatArr) -> FloatArr:
    """Haack ogive radius, nose only, normalised to 1 at x = 1."""
    th = np.arccos(np.clip(1 - 2 * x, -1, 1))
    return np.asarray(np.sqrt((th - np.sin(2 * th) / 2) / np.pi))


def power_series(x: FloatArr, n: float = 0.5) -> FloatArr:
    return np.asarray(np.clip(x, 0, 1) ** n)


# ------------------------------------------------------------------ conformal maps
def joukowski(z: CplxArr, c: float = 1.0) -> CplxArr:
    return np.asarray(z + c**2 / z)


def karman_trefftz(z: CplxArr, tau_deg: float = 10.0, c: float = 1.0) -> CplxArr:
    """Joukowski with a finite trailing-edge angle tau."""
    n = 2 - tau_deg / 180
    p, q = (z + c) ** n, (z - c) ** n
    return np.asarray(n * c * (p + q) / (p - q))


# ------------------------------------------------------------------ hyperbolic geometry
def tractrix(t: FloatArr, a: float = 1.0) -> tuple[FloatArr, FloatArr]:
    """Tractrix (x, r). Revolved it gives the pseudosphere, with K = -1/a^2."""
    return a * (t - np.tanh(t)), a / np.cosh(t)


def pseudosphere(t: FloatArr, phi: FloatArr, a: float = 1.0) -> tuple[FloatArr, FloatArr, FloatArr]:
    x, r = tractrix(t, a)
    return np.outer(x, np.ones_like(phi)), np.outer(r, np.cos(phi)), np.outer(r, np.sin(phi))


def moebius(z: CplxArr, w: complex) -> CplxArr:
    """Hyperbolic translation of the unit disk taking 0 to w, with |w| < 1."""
    return np.asarray((z + w) / (1 + np.conj(w) * z))


def poincare_distance(u: complex, v: complex) -> float:
    return float(2 * np.arctanh(abs((u - v) / (1 - np.conj(v) * u))))


def hyperbolic_warp(y: FloatArr, z: FloatArr, w: complex, radius: float) -> tuple[FloatArr, FloatArr]:
    """Deform an outline by a Moebius map of the disk of the given radius, keeping its size."""
    c = np.asarray((y + 1j * z) / radius, dtype=np.complex128)
    m = moebius(c, w)
    m = m - np.mean(m)
    m = m * (np.max(np.abs(c)) / max(float(np.max(np.abs(m))), 1e-9))
    return m.real * radius, m.imag * radius


# ------------------------------------------------------------------ elliptic and minimal surfaces
def inverse_stereographic(y: FloatArr, z: FloatArr, R: float = 1.0) -> tuple[FloatArr, FloatArr, FloatArr]:
    """Wrap plane points onto a sphere of radius R, with the plane through its centre."""
    d = y**2 + z**2 + R**2
    return 2 * R**2 * y / d, 2 * R**2 * z / d, R * (y**2 + z**2 - R**2) / d


def spherical_cap(theta: FloatArr, phi: FloatArr, R: float = 1.0) -> tuple[FloatArr, FloatArr, FloatArr]:
    return R * np.outer(np.sin(theta), np.cos(phi)), R * np.outer(np.sin(theta), np.sin(phi)), R * np.outer(np.cos(theta), np.ones_like(phi))


def catenoid(u: FloatArr, phi: FloatArr, c: float = 1.0) -> tuple[FloatArr, FloatArr, FloatArr]:
    r = c * np.cosh(u / c)
    return np.outer(u, np.ones_like(phi)), np.outer(r, np.cos(phi)), np.outer(r, np.sin(phi))


def gaussian_curvature(X: FloatArr, Y: FloatArr, Z: FloatArr) -> FloatArr:
    """K = (LN - M^2)/(EG - F^2) on a grid, from the fundamental forms. Index units cancel."""
    P = np.stack([X, Y, Z], -1)
    Pu, Pv = np.gradient(P, axis=0), np.gradient(P, axis=1)
    Puu, Puv = np.gradient(Pu, axis=0), np.gradient(Pu, axis=1)
    Pvv = np.gradient(Pv, axis=1)
    n = np.cross(Pu, Pv)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-14)
    E, F, G = (Pu * Pu).sum(-1), (Pu * Pv).sum(-1), (Pv * Pv).sum(-1)
    L, M, N = (Puu * n).sum(-1), (Puv * n).sum(-1), (Pvv * n).sum(-1)
    return np.asarray((L * N - M**2) / np.maximum(E * G - F**2, 1e-14))
