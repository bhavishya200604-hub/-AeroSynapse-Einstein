"""Step 1.1: nondimensionalisation, analytic solutions, MMS generator."""

import numpy as np
import pytest
import sympy as sp

from aerosynapse.thermal.physics import analytic as an
from aerosynapse.thermal.physics import mms
from aerosynapse.thermal.physics import nondim as nd

# --- nondimensionalisation --------------------------------------------------------------------


def test_groups_consistent():
    p = nd.DimensionalParams(
        L=0.1, U=0.2, nu=1.5e-5, alpha=2.2e-5, k_fluid=0.026, h=25.0, k_solid=200.0, dT=10.0, beta=3.4e-3
    )
    g = nd.to_nondim(p, bi_k="solid")
    assert g.Pe == pytest.approx(g.Re * g.Pr, rel=1e-14)
    assert g.Bi == pytest.approx(25.0 * 0.1 / 200.0)
    assert nd.to_nondim(p, bi_k="fluid").Bi == pytest.approx(25.0 * 0.1 / 0.026)
    with pytest.raises(ValueError):
        nd.to_nondim(nd.DimensionalParams(L=1, U=1, nu=1, alpha=1, k_fluid=1), bi_k="solid")


def test_scaling_roundtrip():
    rng = np.random.default_rng(0)
    T = rng.uniform(280, 350, 10)
    assert np.allclose(nd.unscale_T(nd.scale_T(T, 300.0, 20.0), 300.0, 20.0), T)
    tt = rng.uniform(0, 5, 10)
    assert np.allclose(nd.unscale_t(nd.scale_t(tt, 0.3, 2.0), 0.3, 2.0), tt)


def test_buckingham_pi_group_count_and_dimensionless():
    vars_ = ["L", "U", "nu", "alpha", "k", "h", "rho", "cp", "dT", "q_vol"]
    n, groups = nd.buckingham_pi(vars_)
    assert n == 6 and len(groups) == 6
    for gr in groups:
        dims = np.zeros(4)
        for v, e in gr.items():
            dims += float(e) * np.array(nd.DIMENSIONS[v])
        assert np.allclose(dims, 0)


# --- Case A analytic --------------------------------------------------------------------------

X, T, PE, S0, X0, C = sp.symbols("x t Pe s0 x0 c", positive=True)


def test_steady_formula_symbolic():
    th = (sp.exp(PE * X) - 1) / (sp.exp(PE) - 1)
    assert sp.simplify(sp.diff(th, X) - sp.diff(th, X, 2) / PE) == 0


@pytest.mark.parametrize("Pe", [0.5, 10.0, 200.0])
def test_steady_stable_form_matches_naive(Pe):
    xs = np.linspace(0, 1, 101)
    naive = np.expm1(Pe * xs) / np.expm1(Pe)
    assert np.allclose(an.steady_convdiff(xs, Pe), naive, rtol=1e-12, atol=1e-14)
    naive_dx = Pe * np.exp(Pe * xs) / np.expm1(Pe)
    assert np.allclose(an.steady_convdiff_dx(xs, Pe), naive_dx, rtol=1e-12, atol=1e-14)


def test_steady_extreme_pe_finite():
    xs = np.linspace(0, 1, 2001)
    th = an.steady_convdiff(xs, 1e4)
    assert np.all(np.isfinite(th)) and th[0] == 0 and th[-1] == pytest.approx(1)
    assert np.all(np.diff(th) >= 0)


def test_gaussian_symbolic():
    s2 = S0**2 + 2 * T / PE
    th = sp.sqrt(S0**2 / s2) * sp.exp(-((X - X0 - C * T) ** 2) / (2 * s2))
    res = sp.diff(th, T) + C * sp.diff(th, X) - sp.diff(th, X, 2) / PE
    assert sp.simplify(res / th) == 0


# --- Case B analytic --------------------------------------------------------------------------


@pytest.mark.parametrize("Bi", [0.1, 1.0, 100.0])
def test_robin_eigenvalues(Bi):
    lam = an.robin_eigenvalues(Bi, 50)
    assert np.allclose(lam * np.sin(lam) - Bi * np.cos(lam), 0, atol=1e-10)
    assert np.all(np.diff(lam) > 0)


@pytest.mark.parametrize("Bi,Q,m", [(1.0, 0.0, 0.0), (0.1, 2.0, 0.0), (10.0, 1.0, 1.5)])
def test_slab_steady_symbolic_and_bc(Bi, Q, m):
    xs = np.linspace(0, 1, 4001)
    th = an.slab_steady(xs, Bi, Q, m)
    d1 = np.gradient(th, xs, edge_order=2)
    d2 = np.gradient(d1, xs, edge_order=2)
    assert np.max(np.abs(d2 - m**2 * th + Q)[5:-5]) < 1e-4 * max(1, Q)
    assert abs(d1[0]) < 1e-5 and abs(-d1[-1] - Bi * th[-1]) < 1e-5 * max(1, Q)


@pytest.mark.parametrize("Bi,Q,m", [(1.0, 0.0, 0.0), (5.0, 1.0, 0.0), (2.0, 1.0, 2.0)])
def test_slab_transient_pde_bc_limits(Bi, Q, m):
    xs = np.linspace(0, 1, 801)
    t0, dt = 0.05, 1e-5
    th = an.slab_transient(xs, t0, Bi, Q, m)
    th_t = (an.slab_transient(xs, t0 + dt, Bi, Q, m) - an.slab_transient(xs, t0 - dt, Bi, Q, m)) / (2 * dt)
    d1 = np.gradient(th, xs, edge_order=2)
    d2 = np.gradient(d1, xs, edge_order=2)
    assert np.max(np.abs(th_t - d2 + m**2 * th - Q)[5:-5]) < 1e-3
    assert abs(-d1[-1] - Bi * th[-1]) < 1e-4
    # t -> 0 recovers IC in the interior; t -> inf recovers steady state
    assert np.allclose(an.slab_transient(xs[:600], 1e-4, Bi, Q, m, n_terms=600), 1.0, atol=5e-3)
    assert np.allclose(an.slab_transient(xs, 60.0, Bi, Q, m), an.slab_steady(xs, Bi, Q, m), atol=1e-10)


# --- MMS --------------------------------------------------------------------------------------


def test_mms_1d_exact_symbolic_residual():
    ms = mms.mms_1d_default(Pe=20.0, a=1.0, s=0.5)
    res = mms.residual_operator(ms.theta, ms.velocity, ms.D, ms.s) - ms.source
    assert sp.simplify(res) == 0


def test_mms_2d_divergence_free_and_numeric_residual():
    ms = mms.mms_2d_default(Pe=50.0)
    assert mms.divergence(ms.velocity) == 0
    # numeric check of the forcing with finite differences at random interior points
    rng = np.random.default_rng(1)
    X_, Y_, T_ = rng.uniform(0.1, 0.9, (3, 20))
    h = 1e-4
    f = ms.exact
    th_t = (f(X_, Y_, T_ + h) - f(X_, Y_, T_ - h)) / (2 * h)
    th_x = (f(X_ + h, Y_, T_) - f(X_ - h, Y_, T_)) / (2 * h)
    th_y = (f(X_, Y_ + h, T_) - f(X_, Y_ - h, T_)) / (2 * h)
    lap = (f(X_ + h, Y_, T_) + f(X_ - h, Y_, T_) + f(X_, Y_ + h, T_) + f(X_, Y_ - h, T_) - 4 * f(X_, Y_, T_)) / h**2
    uf = mms._lambdify(ms.velocity[0], ms.args)
    vf = mms._lambdify(ms.velocity[1], ms.args)
    Q = th_t + uf(X_, Y_, T_) * th_x + vf(X_, Y_, T_) * th_y - lap / 50.0
    assert np.allclose(Q, ms.forcing(X_, Y_, T_), atol=1e-5)


def test_robin_data():
    ms = mms.mms_1d_default()
    g = ms.robin_data(normal=(1,), a=2.0, b=1.0)
    xv, tv = np.array(1.0), np.array(0.3)
    assert g(xv, tv) == pytest.approx(2 * ms.exact(xv, tv) + ms.grad[0](xv, tv))
