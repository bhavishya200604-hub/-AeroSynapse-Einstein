"""Phase 1 verification: observed orders, MMS for every PDE and BC type, conservation, stability guards."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aerosynapse.exceptions import ConfigurationError, SolverDivergenceError, StabilityError
from aerosynapse.numerics import amr, burgers, convdiff, fem, heat, poisson, wave
from aerosynapse.numerics import verification as V
from aerosynapse.numerics.fd1d import assemble_1d
from aerosynapse.numerics.guards import check_field
from aerosynapse.numerics.timestep import theta_integrate
from aerosynapse.physics import Dirichlet, Neumann, Periodic, Robin
from aerosynapse.physics import library as L

pytestmark = pytest.mark.verification
TOL = 0.15  # |observed - theoretical| order tolerance on the finest pair


def _order(fn, ns, length=1.0):
    return V.convergence_study(fn, ns, length).pairwise_orders[-1]


@pytest.mark.parametrize("theta,expected", [(0.5, 2.0), (1.0, 1.0)])
def test_heat1d_order(theta, expected):
    p = L.heat_1d()
    if theta == 0.5:

        def err(n):
            x, s = heat.solve_heat_1d(p, n, nt=n, theta=theta)
            return np.abs(s.u - p.exact(x, p.t_final)).max()

        assert abs(_order(err, [16, 32, 64]) - 2.0) < TOL
    else:  # time order with fine space grid

        def err_t(nt):
            x, s = heat.solve_heat_1d(p, 400, nt=nt, theta=theta)
            return np.abs(s.u - p.exact(x, p.t_final)).max()

        assert abs(V.convergence_study(err_t, [10, 20, 40]).pairwise_orders[-1] - expected) < TOL


def test_heat1d_explicit_stability_guard():
    p = L.heat_1d(alpha=1.0)
    with pytest.raises(StabilityError):
        heat.solve_heat_1d(p, 64, nt=10, theta=0.0)
    x, s = heat.solve_heat_1d(p, 16, nt=400, theta=0.0)
    assert np.abs(s.u - p.exact(x, p.t_final)).max() < 1e-3


def test_heat1d_adaptive_dt():
    p = L.heat_1d()
    x, s = heat.solve_heat_1d(p, 64, nt=4, adaptive=True, tol=1e-8)
    assert s.steps > 4 and np.abs(s.u - p.exact(x, p.t_final)).max() < 5e-4
    assert max(s.dt_history) > min(s.dt_history)  # step size actually adapted


def test_heat2d_order():
    p = L.heat_2d()

    def err(n):
        X, Y, T = heat.solve_heat_2d(p, n, nt=n)
        return np.abs(T - p.exact(X, Y, p.t_final)).max()

    assert abs(_order(err, [8, 16, 32]) - 2.0) < TOL


@pytest.mark.parametrize("bc", ["neumann", "robin"])
def test_heat1d_flux_bcs_mms(bc):
    """MMS: T = e^{-t} cos(x) + 1 on [0,1], s = T_t - alpha T_xx."""
    alpha = 0.5
    ex = lambda x, t: np.exp(-t) * np.cos(x) + 1
    src = lambda x, t: -np.exp(-t) * np.cos(x) + alpha * np.exp(-t) * np.cos(x)
    right_flux = lambda t: -np.exp(-t) * np.sin(1.0)  # outward derivative at x=1
    if bc == "neumann":
        left, right = Neumann(0.0), Neumann(right_flux)
    else:  # a T + b dT/dn = g with a=2, b=1
        left, right = (
            Robin(2.0, 1.0, lambda t: 2 * ex(0.0, t)),
            Robin(2.0, 1.0, lambda t: 2 * ex(1.0, t) + right_flux(t)),
        )

    def err(n):
        op = assemble_1d(0, 1, n, alpha, 0.0, left, right)
        s = theta_integrate(op, ex(op.x, 0.0), 0.5, 0.5 / n, 0.5, source=src)
        return np.abs(s.u - ex(op.x, 0.5)).max()

    assert abs(_order(err, [16, 32, 64]) - 2.0) < TOL


@pytest.mark.parametrize("scheme,p_expected", [("upwind", 1.0), ("central", 2.0), ("quick", 2.0), ("vanleer", 2.0)])
def test_convdiff_fv_orders(scheme, p_expected):
    prob = L.convdiff_1d(pe=50.0)

    def err(n):
        r = convdiff.solve_convdiff_fv(0, 1, n, 1.0, prob.p("D"), prob.initial_condition, 0.5, scheme=scheme, cfl=0.2)
        return np.abs(r.u - prob.exact(r.x, 0.5)).max()

    ns = [128, 256, 512] if scheme == "upwind" else [64, 128, 256]
    assert abs(_order(err, ns) - p_expected) < (0.12 if scheme != "upwind" else 0.1)


def test_convdiff_mass_conservation_and_theta():
    prob = L.convdiff_1d(pe=20.0)
    r = convdiff.solve_convdiff_fv(0, 1, 128, 1.0, 0.05, prob.initial_condition, 0.5, scheme="vanleer")
    assert max(r.mass) - min(r.mass) < 1e-12
    r2 = convdiff.solve_convdiff_fv(
        0, 1, 128, 1.0, 0.05, prob.initial_condition, 0.5, scheme="central", integrator="theta", dt=0.5 / 200
    )
    assert np.abs(r2.u - prob.exact(r2.x, 0.5)).max() < 5e-3
    assert max(r2.mass) - min(r2.mass) < 1e-12


def test_convdiff_dirichlet_neumann_steady():
    """Steady boundary layer reached by time marching with Dirichlet in / Dirichlet out."""
    p = L.convdiff_steady(pe=10.0)
    r = convdiff.solve_convdiff_fv(
        0, 1, 100, 1.0, 0.1, lambda x: x, 5.0, scheme="central", left=Dirichlet(0.0), right=Dirichlet(1.0)
    )
    assert np.abs(r.u - p.exact(r.x)).max() < 2e-3
    r = convdiff.solve_convdiff_fv(
        0, 1, 40, 0.0, 1.0, lambda x: 0 * x, 4.0, scheme="central", left=Neumann(1.0), right=Dirichlet(0.0)
    )  # outward du/dn = -u_x = 1: u = 1 - x
    assert np.abs(r.u - (1 - r.x)).max() < 1e-3


def test_convdiff_guards():
    with pytest.raises(StabilityError):
        convdiff.solve_convdiff_fv(0, 1, 64, 1.0, 0.01, np.sin, 0.1, dt=0.1)
    with pytest.raises(ConfigurationError):
        convdiff.solve_convdiff_fv(0, 1, 64, 1.0, 0.01, np.sin, 0.1, scheme="nope")
    with pytest.raises(ConfigurationError):
        convdiff.solve_convdiff_fv(0, 1, 64, 1.0, 0.01, np.sin, 0.1, scheme="vanleer", integrator="theta")


def test_central_oscillates_at_high_cell_peclet():
    """Documented failure mode: central scheme at Pe_h > 2 is non-monotone, upwind is not."""
    x = np.linspace(0, 1, 21)
    uc = amr.solve_nonuniform(x, 1.0, 1e-3, 0.0, 1.0, "central")
    uu = amr.solve_nonuniform(x, 1.0, 1e-3, 0.0, 1.0, "upwind")
    assert uc.min() < -0.1 and np.all(np.diff(uu) >= -1e-12)


def test_amr_beats_uniform():
    p = L.convdiff_steady(pe=100.0)
    x, u, hist = amr.adaptive_solve(1.0, 0.01, max_nodes=80)
    xu = np.linspace(0, 1, x.size)
    uu = amr.solve_nonuniform(xu, 1.0, 0.01, 0.0, 1.0)
    assert np.abs(u - p.exact(x)).max() < 0.1 * np.abs(uu - p.exact(xu)).max()
    assert hist == sorted(hist) and len(hist) > 1


def test_burgers_order_and_conservation():
    b = L.burgers_1d(nu=0.1)

    def err(n):
        r = burgers.solve_burgers(0, 2 * np.pi, n, 0.1, b.initial_condition, 0.5, cfl=0.3)
        return np.abs(r.u - b.exact(r.x, 0.5)).max()

    assert _order(err, [128, 256, 512], 2 * np.pi) > 1.7
    r = burgers.solve_burgers(0, 2 * np.pi, 128, 0.1, b.initial_condition, 0.5)
    assert max(r.mass) - min(r.mass) < 1e-11


def test_wave_order_energy_and_cfl():
    w = L.wave_1d(t_final=0.75)

    def err(n):
        x, u, _ = wave.solve_wave(1.0, n, 1.0, w.initial_condition, 0.75)
        return np.abs(u - w.exact(x, 0.75)).max()

    assert abs(_order(err, [32, 64, 128]) - 2.0) < TOL
    _, _, E = wave.solve_wave(1.0, 64, 1.0, w.initial_condition, 2.0)
    assert (max(E) - min(E)) / E[0] < 1e-12
    with pytest.raises(StabilityError):
        wave.solve_wave(1.0, 64, 1.0, w.initial_condition, 1.0, courant=1.2)


def test_poisson2d_order_and_neumann():
    p = L.poisson_2d()

    def err(n):
        X, Y, U = poisson.solve_poisson_2d(p, n)
        return np.abs(U - p.exact(X, Y)).max()

    assert abs(_order(err, [16, 32, 64]) - 2.0) < TOL
    # mixed: u = cos(pi x) sin(pi y) with Neumann (du/dn = 0) at x = 0, 1, Dirichlet at y = 0, 1
    from aerosynapse.physics.problem import PDEProblem

    q = PDEProblem(
        "m",
        "-lap u = f",
        ((0.0, 1.0), (0.0, 1.0)),
        bcs={"left": Neumann(0.0), "right": Neumann(0.0), "bottom": Dirichlet(0.0), "top": Dirichlet(0.0)},
        source=lambda x, y: 2 * np.pi**2 * np.cos(np.pi * x) * np.sin(np.pi * y),
    )

    def err2(n):
        X, Y, U = poisson.solve_poisson_2d(q, n)
        return np.abs(U - np.cos(np.pi * X) * np.sin(np.pi * Y)).max()

    assert abs(_order(err2, [16, 32, 64]) - 2.0) < TOL


@pytest.mark.parametrize("bcs", ["DD", "NR", "RD", "PP", "NN"])
def test_poisson1d_all_bcs(bcs):
    if bcs in ("PP", "NN"):
        ue, f = (lambda x: np.cos(2 * np.pi * x)), (lambda x: 4 * np.pi**2 * np.cos(2 * np.pi * x))
        lb, rb = (Periodic(), Periodic()) if bcs == "PP" else (Neumann(0.0), Neumann(0.0))
    else:
        ue, f = (lambda x: np.cos(np.pi * x) + x), (lambda x: np.pi**2 * np.cos(np.pi * x))
        lb, rb = {
            "DD": (Dirichlet(1.0), Dirichlet(0.0)),
            "NR": (Neumann(-1.0), Robin(2.0, 1.0, 1.0)),
            "RD": (Robin(1.0, 1.0, 0.0), Dirichlet(0.0)),
        }[bcs]

    def err(n):
        x, u = poisson.solve_poisson_1d(0, 1, n, f, lb, rb)
        return np.abs(u - ue(x)).max()

    assert abs(_order(err, [16, 32, 64]) - 2.0) < TOL


def test_pure_neumann_incompatible_data_raises():
    with pytest.raises(ConfigurationError):
        poisson.solve_poisson_1d(0, 1, 32, lambda x: 1.0 + 0 * x, Neumann(0.0), Neumann(0.0))


def test_fem_orders_and_crosscheck():
    p = L.poisson_2d()
    grad = lambda x, y: (
        np.pi * np.cos(np.pi * x) * np.sin(2 * np.pi * y),
        2 * np.pi * np.sin(np.pi * x) * np.cos(2 * np.pi * y),
    )
    errs = [
        fem.fem_errors(*fem.solve_poisson_fem(n, p.source, lambda x, y: 0 * x), p.exact, grad) for n in (16, 32, 64)
    ]
    hs = [1 / 16, 1 / 32, 1 / 64]
    assert abs(V.observed_order(hs, [e[0] for e in errs]).pairwise_orders[-1] - 2.0) < TOL
    assert abs(V.observed_order(hs, [e[1] for e in errs]).pairwise_orders[-1] - 1.0) < TOL
    nodes, _, u = fem.solve_poisson_fem(32, p.source, lambda x, y: 0 * x)
    X, Y, U = poisson.solve_poisson_2d(p, 32)
    assert np.abs(u - U.ravel()).max() < 5e-3  # FD and FEM agree to discretisation error


def test_divergence_guard():
    with pytest.raises(SolverDivergenceError):
        check_field(np.array([1.0, np.nan]))
    with pytest.raises(SolverDivergenceError):
        check_field(np.array([1e9]), bound=1e8)


def test_verification_tools():
    r = V.observed_order([0.1, 0.05, 0.025], [1e-2, 2.5e-3, 6.25e-4])
    assert r.fitted_order == pytest.approx(2.0) and r.as_dict()["pairwise_orders"][0] == pytest.approx(2.0)
    g = V.gci(1.0 + 0.01 / 4, 1.0 + 0.01, 1.0 + 0.04, 2.0)
    assert g.p == pytest.approx(2.0) and g.f_extrapolated == pytest.approx(1.0)
    assert g.asymptotic_ratio == pytest.approx(1.0, rel=1e-2)
    with pytest.raises(ValueError):
        V.gci(1.0, 1.1, 1.0, 2.0)
    with pytest.raises(ValueError):
        V.observed_order([1, 2], [0.0, 1.0])
    assert V.richardson(1.0025, 1.01, 2, 2) == pytest.approx(1.0)
    assert V.is_von_neumann_stable(0.5, 10.0, 5.0, "central")
    assert not V.is_von_neumann_stable(0.0, 0.6, 0.0, "central")
    assert V.is_von_neumann_stable(0.0, 0.25, 0.5, "upwind")
    assert not V.is_von_neumann_stable(0.0, 0.0, 0.5, "central")  # FTCS pure advection unstable


@settings(max_examples=25, deadline=None)
@given(r=st.floats(0.01, 0.5), c=st.floats(0.0, 0.5))
def test_explicit_upwind_stable_region(r, c):
    """Property: explicit upwind is von Neumann stable whenever 2r + c <= 1."""
    if 2 * r + c <= 1.0:
        assert V.is_von_neumann_stable(0.0, r, c, "upwind")
