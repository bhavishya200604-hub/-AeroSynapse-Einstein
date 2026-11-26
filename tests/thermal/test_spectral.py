"""Step 1.4: Chebyshev collocation and the solve_ivp method-of-lines cross-check."""

import numpy as np
import pytest

from aerosynapse.thermal.numerics import fd1d
from aerosynapse.thermal.numerics import spectral as sp_
from aerosynapse.thermal.numerics.mol import solve_mol
from aerosynapse.thermal.numerics.time_integrators import integrate
from aerosynapse.thermal.physics import analytic as an
from aerosynapse.thermal.physics import mms


def test_cheb_exact_for_polynomials():
    for N in (4, 9, 16):
        Dm, xi = sp_.cheb(N)
        assert np.allclose(Dm @ xi**N, N * xi ** (N - 1), atol=1e-10 * N**2)
        assert np.allclose(Dm @ np.ones(N + 1), 0, atol=1e-12)


def test_cheb_spectral_derivative_accuracy():
    D1, x = sp_.cheb_mapped(40, -1.0, 2.0)
    f = np.exp(x) * np.sin(5 * x)
    df = np.exp(x) * (np.sin(5 * x) + 5 * np.cos(5 * x))
    assert np.max(np.abs(D1 @ f - df)) < 1e-9


def test_clenshaw_curtis():
    w = sp_.clencurt(20, 0.0, 1.0)
    _, x = sp_.cheb_mapped(20, 0.0, 1.0)
    assert w @ np.exp(x) == pytest.approx(np.e - 1, abs=1e-14)
    w9 = sp_.clencurt(9, 0.0, 2.0)
    _, x9 = sp_.cheb_mapped(9, 0.0, 2.0)
    assert w9 @ x9**3 == pytest.approx(4.0, abs=1e-13)


def test_row_replacement_equals_elimination():
    p = sp_.ChebConvDiff1D(
        24, a=1.0, D=0.1, s=0.3, source=lambda x, t: np.cos(3 * x), left=fd1d.robin(2.0, 0.5), right=fd1d.neumann(0.7)
    )
    y_rr = p.steady()
    y_el = np.linalg.solve(p.A_dense, -p.b(0.0))
    assert np.allclose(y_rr, y_el, atol=1e-11)


def test_steady_case_a_and_case_b_exact():
    p = sp_.ChebConvDiff1D(48, a=1.0, D=0.1, left=fd1d.dirichlet(0.0), right=fd1d.dirichlet(1.0))
    assert np.max(np.abs(p.nodal(p.steady(), 0) - an.steady_convdiff(p.x, 10.0))) < 1e-11
    Bi, Q, m = 3.0, 1.0, 2.0
    q = sp_.ChebConvDiff1D(
        24, a=0.0, D=1.0, s=m * m, source=lambda x, t: Q + 0 * x, left=fd1d.neumann(0.0), right=fd1d.robin(Bi)
    )
    assert np.max(np.abs(q.nodal(q.steady(), 0) - an.slab_steady(q.x, Bi, Q, m))) < 1e-12


def test_transient_case_b_radau_vs_series():
    Bi, Q, m, T = 5.0, 1.0, 1.0, 0.2
    p = sp_.ChebConvDiff1D(
        24, a=0.0, D=1.0, s=m * m, source=lambda x, t: Q + 0 * x, left=fd1d.neumann(0.0), right=fd1d.robin(Bi)
    )
    sol = solve_mol(p, np.ones(p.idx.size), 0.0, T, "Radau", rtol=1e-12, atol=1e-14)
    assert np.max(np.abs(p.nodal(sol.y[:, -1], T) - an.slab_transient(p.x, T, Bi, Q, m))) < 1e-8


def test_mol_cross_checks_own_integrators():
    ms = mms.mms_1d_default(Pe=20.0, a=1.0)
    p = fd1d.ConvDiff1D(
        64,
        a=1.0,
        D=0.05,
        source=ms.forcing,
        scheme="central",
        left=fd1d.BC(1, 0, lambda t: float(ms.exact(0.0, t))),
        right=fd1d.BC(1, 0, lambda t: float(ms.exact(1.0, t))),
    )
    y0 = ms.exact(p.x_active, 0 * p.x_active)
    ref = solve_mol(p, y0, 0.0, 0.5, "Radau", rtol=1e-12, atol=1e-14).y[:, -1]
    for m, dt, tol in [("crank_nicolson", 1e-3, 1e-6), ("bdf2", 1e-3, 5e-6), ("rk4", 2e-4, 1e-10)]:
        assert np.max(np.abs(integrate(p, y0, 0.0, 0.5, dt, m) - ref)) < tol, m
