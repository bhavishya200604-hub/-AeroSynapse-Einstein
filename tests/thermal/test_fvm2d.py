"""Step 1.5: 2D finite-volume energy solver, ADI, multigrid, CG."""

import numpy as np
import pytest
import scipy.sparse.linalg as spla
import sympy as sp

from aerosynapse.thermal.numerics import fvm2d as fv
from aerosynapse.thermal.numerics.adi import DouglasGunnADI, integrate_adi
from aerosynapse.thermal.numerics.convergence import fit_order
from aerosynapse.thermal.numerics.linalg import cg
from aerosynapse.thermal.numerics.mol import solve_mol
from aerosynapse.thermal.numerics.multigrid import MultigridHelmholtz, laplacian_fv
from aerosynapse.thermal.numerics.time_integrators import integrate
from aerosynapse.thermal.physics import mms

X, Y, T = mms.x, mms.y, mms.t
PSI = sp.sin(sp.pi * X) * sp.sin(sp.pi * Y) / sp.pi
PSIF = sp.lambdify((X, Y, T), PSI, "numpy")


mms_bcs = fv.bcs_from_mms

SPEC = {"left": (1, 0), "right": (2, 1), "bottom": (0, 1), "top": (3, 1)}


def make(n, scheme="central", steady=True, Pe=50.0):
    if steady:
        th = sp.cos(sp.pi * X / 2 + sp.Rational(1, 5)) * sp.sin(sp.pi * Y + sp.Rational(1, 7)) + X * Y**2 / 2
        u, v = mms.velocity_from_streamfunction(PSI)
        ms = mms.manufacture(th, [u, v], 1 / sp.nsimplify(Pe), 0)
    else:
        ms = mms.mms_2d_default(Pe=Pe)
    g = fv.Grid2D(n, n)
    vel = fv.FaceVelocity.from_streamfunction(g, lambda x, y, t: PSIF(x, y, t))
    p = fv.FVEnergy2D(g, 1 / Pe, vel, source=ms.forcing, bcs=mms_bcs(ms, SPEC), scheme=scheme)
    return p, ms


def test_streamfunction_velocity_discretely_divergence_free():
    g = fv.Grid2D(24, 17, 1.0, 1.3)
    U, V = fv.FaceVelocity.from_streamfunction(g, lambda x, y, t: np.sin(3 * x) * np.cos(2 * y) + x * y).at(0)
    assert np.max(np.abs(fv.discrete_divergence(g, U, V))) < 1e-12


@pytest.mark.parametrize("scheme", ["upwind1", "central", "quick"])
def test_probed_operator_and_split(scheme):
    p, _ = make(12, scheme)
    y = np.random.default_rng(0).standard_normal(p.g.n)
    for t in (0.0, 0.4):
        assert np.allclose(p.A @ y + p.b(t), p.rhs(y, t), atol=1e-10)
    Ax, Ay = p.split_operators()
    assert np.allclose((Ax + Ay - p.A).toarray(), 0, atol=1e-11)


def test_steady_second_order_and_energy_balance():
    errs, Ns = [], [16, 32, 64]
    for n in Ns:
        p, ms = make(n)
        y = p.steady()
        errs.append(np.max(np.abs(y.reshape(n, n) - ms.exact(p.g.Xc, p.g.Yc, 0 * p.g.Xc))))
        eb = p.energy_balance(y, 0.0)
        assert abs(eb["residual"]) < 1e-12 * max(1.0, eb["scale"])
        assert abs(eb["storage_rate"]) < 1e-10 * max(1.0, eb["scale"])
    assert abs(fit_order(1 / np.array(Ns), errs)[0] - 2.0) < 0.1


@pytest.mark.parametrize("order,expected", [(1, 1.0), (2, 2.0)])
def test_wall_gradient_order_depends_on_bc_closure(order, expected):
    """Step 1.7 finding: the half-cell closure gives O(h^2) theta but only O(h) wall gradients."""
    x = mms.x
    th = sp.cos(sp.pi * x / 2 + sp.Rational(1, 5)) * sp.sin(sp.pi * mms.y + sp.Rational(1, 7)) + x * mms.y**2 / 2
    ms = mms.manufacture(th, [0, 0], sp.Rational(1, 2), 0)
    exact = float(sp.N(sp.integrate(sp.diff(th, x).subs(x, 0), (mms.y, 0, 1))))
    errs = []
    for n in (32, 64, 128):
        g = fv.Grid2D(n, n)
        p = fv.FVEnergy2D(
            g, 0.5, source=ms.forcing, bcs=fv.bcs_from_mms(ms, {k: (1, 0) for k in fv.SIDES}), bc_order=order
        )
        y = p.steady()
        errs.append(abs(np.mean(p.wall_gradient(y, 0.0, "left")) - exact))
        assert abs(p.energy_balance(y, 0.0)["residual"]) < 1e-12
    assert abs(np.log2(errs[1] / errs[2]) - expected) < 0.2


def test_bc_order2_stays_adi_compatible_and_m_matrix():
    bcs = {
        "left": fv.dirichlet2(1.0),
        "right": fv.robin2(5.0),
        "bottom": fv.neumann2(0.0),
        "top": fv.robin2(0.5),
    }
    p = fv.FVEnergy2D(fv.Grid2D(10, 10), 1.0, bcs=bcs, bc_order=2)
    A = p.A.toarray()
    off = A - np.diag(np.diag(A))
    assert np.all(off >= -1e-12) and np.all(np.diag(A) < 0) and np.all(A.sum(axis=1) <= 1e-9)
    y0 = np.zeros(p.g.n)
    a = integrate_adi(p, y0, 0.0, 0.05, 1e-3)
    b = integrate(p, y0, 0.0, 0.05, 1e-3, "crank_nicolson")
    assert np.abs(a - b).max() < 1e-4


def test_fv_laplacian_symmetric():
    g = fv.Grid2D(9, 7, 1.0, 0.8)
    L = laplacian_fv(
        g,
        {"left": fv.dirichlet2(0), "right": fv.robin2(2.0), "bottom": fv.neumann2(0), "top": fv.robin2(0.3)},
    )
    assert abs(L - L.T).max() < 1e-10


def test_adi_second_order_vs_radau():
    p, ms = make(16, steady=False, Pe=20.0)
    y0 = ms.exact(p.g.Xc, p.g.Yc, 0 * p.g.Xc).ravel()
    ref = solve_mol(p, y0, 0.0, 0.5, "Radau", 1e-12, 1e-14).y[:, -1]
    dts = [0.05, 0.025, 0.0125]
    e_adi = [np.max(np.abs(integrate_adi(p, y0, 0.0, 0.5, dt) - ref)) for dt in dts]
    e_cn = [np.max(np.abs(integrate(p, y0, 0.0, 0.5, dt, "crank_nicolson") - ref)) for dt in dts]
    assert abs(fit_order(dts, e_adi)[0] - 2) < 0.15
    assert abs(fit_order(dts, e_cn)[0] - 2) < 0.15


def test_adi_rejects_wide_stencil():
    p, _ = make(8, "quick")
    with pytest.raises(ValueError):
        DouglasGunnADI(p, 0.01)


@pytest.mark.parametrize("kind", ["dirichlet", "mixed", "neumann"])
def test_multigrid_solves_and_is_h_independent(kind):
    bcs = {
        "dirichlet": {k: fv.dirichlet2(0) for k in fv.SIDES},
        "mixed": {
            "left": fv.dirichlet2(0),
            "right": fv.robin2(5.0),
            "bottom": fv.neumann2(0),
            "top": fv.robin2(0.5),
        },
        "neumann": {k: fv.neumann2(0) for k in fv.SIDES},
    }[kind]
    rates = []
    for n in (32, 64):
        g = fv.Grid2D(n, n)
        mg = MultigridHelmholtz(g, 0.0, 1.0, bcs)
        f = (np.cos(np.pi * g.Xc) * np.cos(2 * np.pi * g.Yc) + g.Xc).ravel()
        x, h = mg.solve(f, rtol=1e-10)
        assert h[-1] < 1e-10 and len(h) - 1 <= 15
        rates.append((h[-1] / h[0]) ** (1 / (len(h) - 1)))
        if kind == "neumann":
            fp = f - f.mean()
            assert abs(x.mean()) < 1e-12
            assert np.linalg.norm(mg.M @ x - fp) < 1e-9 * np.linalg.norm(fp)
        else:
            assert np.allclose(x, spla.spsolve(mg.M.tocsc(), f), atol=1e-9)
    assert max(rates) < 0.2


def test_mg_preconditioned_cg():
    g = fv.Grid2D(64, 64)
    bcs = {k: fv.dirichlet2(0) for k in fv.SIDES}
    mg = MultigridHelmholtz(g, 10.0, 1.0, bcs)
    f = np.sin(3 * g.Xc * g.Yc).ravel()
    x, h = cg(mg.M, f, M=mg.precondition, rtol=1e-11)
    assert len(h) - 1 <= 12
    assert np.allclose(x, spla.spsolve(mg.M.tocsc(), f), atol=1e-10)
