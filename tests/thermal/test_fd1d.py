"""Step 1.2: Thomas algorithm, FD operator assembly, BCs, schemes, integrators (fast checks)."""

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sps
import sympy as sp

from aerosynapse.thermal.numerics import fd1d
from aerosynapse.thermal.numerics.convergence import fit_order
from aerosynapse.thermal.numerics.linalg import ThomasFactor, bandwidth, factorize, thomas
from aerosynapse.thermal.numerics.time_integrators import integrate
from aerosynapse.thermal.physics import mms


def _random_tridiag(n, rng, batch=()):
    lo, up = rng.standard_normal((2, *batch, n))
    di = 4 + np.abs(rng.standard_normal((*batch, n)))
    return lo, di, up


def test_thomas_matches_scipy_banded():
    rng = np.random.default_rng(0)
    lo, di, up = _random_tridiag(50, rng)
    d = rng.standard_normal(50)
    ab = np.vstack([np.r_[0, up[:-1]], di, np.r_[lo[1:], 0]])
    assert np.allclose(thomas(lo, di, up, d), sla.solve_banded((1, 1), ab, d), atol=1e-13)
    assert np.allclose(ThomasFactor(lo, di, up).solve(d), thomas(lo, di, up, d), atol=1e-14)


def test_thomas_batched():
    rng = np.random.default_rng(1)
    lo, di, up = _random_tridiag(30, rng, batch=(7,))
    d = rng.standard_normal((7, 30))
    x = thomas(lo, di, up, d)
    for k in range(7):
        M = np.diag(di[k]) + np.diag(lo[k, 1:], -1) + np.diag(up[k, :-1], 1)
        assert np.allclose(M @ x[k], d[k], atol=1e-12)


@pytest.mark.parametrize("scheme", fd1d.LINEAR_SCHEMES)
@pytest.mark.parametrize("a", [1.0, -0.7])
@pytest.mark.parametrize("bcs", ["DD", "NR", "RN"])
def test_probed_matrix_reproduces_rhs(scheme, a, bcs):
    mk = {"D": fd1d.dirichlet(0.3), "N": fd1d.neumann(-0.2), "R": fd1d.robin(3.0, 0.5)}
    p = fd1d.ConvDiff1D(
        23, a=a, D=0.03, s=0.4, source=lambda x, t: np.sin(x + t), left=mk[bcs[0]], right=mk[bcs[1]], scheme=scheme
    )
    y = np.random.default_rng(2).standard_normal(p.idx.size)
    for t in (0.0, 0.7):
        assert np.allclose(p.A @ y + p.b(t), p.rhs(y, t), atol=1e-11)
    lb, ub = bandwidth(p.A)
    assert max(lb, ub) <= 3
    if scheme in ("upwind1", "central"):
        assert max(lb, ub) == 1  # tridiagonal -> Thomas is used by 'auto'


def test_tvd_is_nonlinear():
    p = fd1d.ConvDiff1D(10, scheme="tvd_vanleer")
    with pytest.raises(ValueError):
        _ = p.A


def test_limiter_properties():
    r = np.linspace(-3, 6, 181)
    for name, psi in fd1d.LIMITERS.items():
        v = psi(r)
        assert np.all(v[r <= 0] == 0), name
        assert np.all(v <= np.maximum(0, np.minimum(2 * r, 2)) + 1e-12), name  # Sweby TVD region
        assert psi(np.array([1.0]))[0] == pytest.approx(1.0), name


@pytest.mark.parametrize("lim", list(fd1d.LIMITERS))
def test_tvd_flat_regions_finite(lim):
    """Regression: den = 0 in flat regions once produced NaN for van Albada (r*r overflow)."""
    U = np.array([1.0, 1.0, 0.0, 0.5, 2.0])
    UU = np.array([0.0, 1.0, 0.0, 0.2, 1e-300])
    Dn = np.array([1.0, 1.0, 0.0, 0.5, 3.0])
    f = fd1d.face_values(U, UU, Dn, f"tvd_{lim}")
    assert np.all(np.isfinite(f)) and np.allclose(f[:4], U[:4])


@pytest.mark.parametrize("scheme,order", [("upwind1", 1), ("central", 2), ("upwind2", 2), ("quick", 2)])
def test_negative_velocity_order(scheme, order):
    th = sp.sin(sp.pi * mms.x + sp.Rational(1, 3)) + sp.cos(2 * mms.x) / 2
    ms = mms.manufacture(th, [-1.0], sp.Rational(1, 10), 0)
    errs, Ns = [], [32, 64, 128]
    for N in Ns:
        p = fd1d.ConvDiff1D(
            N,
            a=-1.0,
            D=0.1,
            source=lambda X, t: ms.forcing(X, 0 * X),
            left=fd1d.dirichlet(float(ms.exact(0.0, 0.0))),
            right=fd1d.dirichlet(float(ms.exact(1.0, 0.0))),
            scheme=scheme,
        )
        errs.append(np.max(np.abs(p.nodal(p.steady(), 0) - ms.exact(p.x, 0 * p.x))))
    assert abs(fit_order(1 / np.array(Ns), errs)[0] - order) < 0.2


def test_neumann_diffusion_conserves_trapezoidal_mass():
    p = fd1d.ConvDiff1D(40, a=0.0, D=1.0, left=fd1d.neumann(0.0), right=fd1d.neumann(0.0))
    y0 = np.exp(-50 * (p.x - 0.3) ** 2)
    w = np.full(p.x.size, p.h)
    w[[0, -1]] *= 0.5
    for m in ("crank_nicolson", "bdf2", "rk4"):
        y = integrate(p, y0, 0.0, 0.05, 1e-4, m)
        assert w @ y == pytest.approx(w @ y0, rel=1e-12), m


def test_thomas_and_splu_same_solution():
    p = fd1d.ConvDiff1D(64, a=1.0, D=0.05, left=fd1d.dirichlet(0), right=fd1d.robin(2.0, 1.0))
    I = sps.identity(p.idx.size)
    rhs = np.random.default_rng(3).standard_normal(p.idx.size)
    M = I - 0.01 * p.A
    assert np.allclose(factorize(M, "thomas")(rhs), factorize(M, "splu")(rhs), atol=1e-13)
