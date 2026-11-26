"""Step 1.3: von Neumann analysis, modified equation, and consistency with the fd1d operators."""

import numpy as np
import pytest
import sympy as sp

from aerosynapse.thermal.numerics import fd1d
from aerosynapse.thermal.numerics import stability as st


def _num(expr, **vals):
    f = sp.lambdify(tuple(sp.Symbol(k) for k in vals), expr, "numpy")
    return complex(f(*vals.values()))


def test_ftcs_convection_amplification_classical():
    G2 = st.abs2_simplified(st.amplification_factor("central", "euler").subs(st.r, 0))
    rng = np.random.default_rng(0)
    for Cv, xv in rng.uniform(0, 2, (10, 2)):
        assert _num(G2, C=Cv, xi=xv).real == pytest.approx(1 + Cv**2 * np.sin(xv) ** 2, rel=1e-12)


def test_upwind_euler_amplification_classical():
    G2 = st.abs2_simplified(st.amplification_factor("upwind1", "euler").subs(st.r, 0))
    rng = np.random.default_rng(1)
    for Cv, xv in rng.uniform(0, 2, (10, 2)):
        ref = 1 - 2 * Cv * (1 - Cv) * (1 - np.cos(xv))
        assert _num(G2, C=Cv, xi=xv).real == pytest.approx(ref, rel=1e-12)


@pytest.mark.parametrize(
    "scheme,integ,limit",
    [
        ("upwind1", "euler", 1.0),
        ("central", "rk4", 2 * np.sqrt(2)),
        ("central", "ssprk3", np.sqrt(3)),
    ],
)
def test_courant_limits(scheme, integ, limit):
    assert st.max_stable_courant(scheme, integ) == pytest.approx(limit, rel=1e-4)


def test_ftcs_central_advection_unconditionally_unstable():
    # |G|^2 = 1 + C^2 sin^2(xi) > 1 for every C > 0; bisection with tol 1e-10 stops at C = sqrt(2e-10)
    assert st.max_stable_courant("central", "euler", tol=1e-10) == pytest.approx(np.sqrt(2e-10), rel=1e-3)
    assert st.max_amplification("central", "euler", 1e-3, 0.0) > 1


def test_ftcs_diffusion_limit_and_convdiff_region():
    assert st.max_amplification("central", "euler", 0.0, 0.5) <= 1 + 1e-12
    assert st.max_amplification("central", "euler", 0.0, 0.51) > 1
    # FTCS convection-diffusion: stable iff C^2 <= 2r <= 1
    for Cv, rv in [(0.5, 0.2), (0.3, 0.45), (0.9, 0.45)]:
        stable = Cv**2 <= 2 * rv <= 1
        assert (st.max_amplification("central", "euler", Cv, rv) <= 1 + 1e-12) == stable


@pytest.mark.parametrize("integ", ["implicit_euler", "crank_nicolson", "bdf2"])
@pytest.mark.parametrize("scheme", ["upwind1", "central", "upwind2", "quick"])
def test_implicit_a_stable(integ, scheme):
    for Cv, rv in [(0.5, 0.1), (10.0, 0.0), (50.0, 100.0)]:
        assert st.max_amplification(scheme, integ, Cv, rv) <= 1 + 1e-12


def test_modified_equation_terms():
    a, h, dt = sp.symbols("a h dt", positive=True)
    assert st.modified_equation_terms("upwind1")[2] == a * h / 2
    ct = st.modified_equation_terms("central")
    assert 2 not in ct and sp.simplify(ct[3] + a * h**2 / 6) == 0
    assert 2 not in st.modified_equation_terms("upwind2")
    assert 2 not in st.modified_equation_terms("quick")
    assert sp.simplify(st.numerical_diffusion_euler("upwind1") - a * h / 2 * (1 - a * dt / h)) == 0


@pytest.mark.parametrize("scheme", ["upwind1", "central", "upwind2", "quick"])
def test_stencils_match_solver_interior_rows(scheme):
    a, D, N = 1.3, 0.07, 30
    p = fd1d.ConvDiff1D(N, a=a, D=D, scheme=scheme)
    h = p.h
    row = p.A.toarray()[15]
    i = 15  # active index 15 <-> node 16 (left Dirichlet node removed)
    expect = np.zeros_like(row)
    for j, c in st.CONV_STENCILS[scheme].items():
        expect[i + j] += -a * float(c) / h
    for j, c in st.DIFF_STENCIL.items():
        expect[i + j] += D * float(c) / h**2
    assert np.allclose(row, expect, rtol=1e-12, atol=1e-9)


def test_central_ratio_sign():
    assert st.central_steady_ratio(1.9) > 0 and st.central_steady_ratio(2.1) < 0
