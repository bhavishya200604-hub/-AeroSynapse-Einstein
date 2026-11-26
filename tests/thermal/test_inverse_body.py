"""Step 5.2 (shapes): 1D body forward models for slab / cylinder / sphere — eigenvalue conditions, the generic
quadrature coefficients against the closed-form slab series, steady states, the radial finite-volume solver against the
series, and the dimensional wrapper."""

import numpy as np
import pytest
from scipy.special import j0, j1

from aerosynapse.thermal.inverse import body as B
from aerosynapse.thermal.inverse import spec as IS

X = np.linspace(0.0, 1.0, 9)
T = np.array([0.02, 0.1, 0.5, 2.0])


@pytest.mark.parametrize("n", [0, 1, 2])
def test_eigenvalues_satisfy_the_robin_condition(n):
    lam = B.eigenvalues(0.7, 40, n)
    g = {
        0: lam * np.sin(lam) - 0.7 * np.cos(lam),
        1: lam * j1(lam) - 0.7 * j0(lam),
        2: lam * np.cos(lam) - 0.3 * np.sin(lam),
    }[n]
    assert np.abs(g).max() < 1e-12 and np.all(np.diff(lam) > 0) and abs(np.diff(lam)[-1] - np.pi) < 1e-2


def test_generic_coefficients_reproduce_the_closed_form_slab_series():
    lam = B.eigenvalues(0.7, 120, 0)
    g, w = np.polynomial.legendre.leggauss(960)
    r, w = 0.5 * (g + 1), 0.5 * w
    Xm = B._mode(lam, r, 0)
    C = (Xm * (1 - B.steady(r, 0.7, 0.3, 0)) * w).sum(1) / (Xm**2 * w).sum(1)
    th = B.steady(X, 0.7, 0.3, 0)[None, :] + (np.exp(-np.outer(T, lam**2)) * C) @ B._mode(lam, X, 0)
    assert np.abs(th - B.theta_analytic(X, T, 0.7, 0.3, 120, 0)).max() < 1e-12


@pytest.mark.parametrize("n", [0, 1, 2])
def test_series_initial_and_steady_states(n):
    assert np.abs(B.theta_analytic(X, np.array([300.0]), 0.7, 0.3, 60, n)[0] - B.steady(X, 0.7, 0.3, n)).max() < 1e-12
    early = B.theta_analytic(np.array([0.0, 0.3, 0.6]), np.array([1e-4]), 0.7, 0.3, 400, n)[0]
    assert np.abs(early - 1.0).max() < 1e-3  # theta(r, 0) = 1 away from the surface
    s = B.steady(np.array([1.0]), 2.0, 0.6, n)[0]  # Robin: -theta_r(1) = Q/(n+1) = Bi theta(1)
    assert abs(2.0 * s - 0.6 / (n + 1)) < 1e-15


@pytest.mark.parametrize("n", [0, 1, 2])
def test_finite_volume_solver_matches_the_series(n):
    th = B.theta_analytic(X, T, 0.7, 0.3, 120, n)
    assert np.abs(B.theta_fv(X, T, 0.7, 0.3, n, 200, 1e-3) - th).max() < 1e-4
    coarse = np.abs(B.theta_fv(X, T, 0.7, 0.3, n, 50, 4e-3) - th).max()
    assert coarse < 2e-3  # converges with the grid


def test_dimensional_wrapper_shapes_and_validation():
    s = IS.load_spec(overrides={"body": {"shape": "sphere"}})
    b = IS.body_of(s)
    assert b.n == 2 and b.shape == "sphere"
    p = {k: float(v) for k, v in s["parameters"].items()}
    xs, ts = np.array([0.0, 0.01, 0.02]), np.array([10.0, 100.0])
    Ta = B.temperatures(p, b, xs, ts, "analytic", 60)
    Tf = B.temperatures(p, b, xs, ts, "fdm", 0, IS.fdm_settings(s))
    assert np.abs(Ta - Tf).max() < 5e-3 and Ta.shape == (2, 3)
    with pytest.raises(ValueError, match="body.shape"):
        IS.load_spec(overrides={"body": {"shape": "cube"}})
    with pytest.raises(ValueError, match="one of"):
        B.Body("torus", 0.02, 80.0, 20.0, 0.0)
