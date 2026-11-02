import numpy as np
import pytest

from aerosynapse.exceptions import ConfigurationError
from aerosynapse.physics import Dirichlet, Interface, Neumann, Parameter, PDEProblem, Periodic, Robin
from aerosynapse.physics import library as L
from aerosynapse.physics.bc import BC_CATALOGUE, ghost_value
from aerosynapse.physics.problem import cell_peclet, cfl_number, diffusion_number, dimensionless_numbers
from aerosynapse.registry import PHYSICS

pytestmark = pytest.mark.unit


def test_parameter_range_validation():
    with pytest.raises(ConfigurationError):
        Parameter("alpha", -1.0, low=0.0)


def test_problem_validation():
    with pytest.raises(ConfigurationError):
        PDEProblem("p", "e", ((1.0, 0.0),))
    with pytest.raises(ConfigurationError):
        PDEProblem("p", "e", ((0.0, 1.0),), bcs={"top": Dirichlet()})
    with pytest.raises(ConfigurationError):
        PDEProblem("p", "e", ((0.0, 1.0),), bcs={"left": Periodic(), "right": Dirichlet()})
    with pytest.raises(ConfigurationError):
        PDEProblem("p", "e", ((0.0, 1.0),), t_final=-1)
    with pytest.raises(ConfigurationError):
        L.heat_1d().p("nope")


def test_bc_types():
    assert Dirichlet(2.0).value(0.0) == 2.0
    assert Neumann(lambda t: t).value(3.0) == 3.0
    with pytest.raises(ConfigurationError):
        Robin(1.0, 0.0)
    with pytest.raises(ConfigurationError):
        Robin(0.0, 1.0)
    with pytest.raises(ConfigurationError):
        Periodic().value()
    with pytest.raises(ConfigurationError):
        Interface(-1, 1)
    assert Interface(1.0, 3.0).harmonic_conductivity() == pytest.approx(1.5)
    assert ghost_value(Neumann(1.0), 2.0, 0.5) == pytest.approx(3.0)
    with pytest.raises(ConfigurationError):
        ghost_value(Dirichlet(1.0), 2.0, 0.5)
    for k in ("dirichlet", "neumann", "robin", "periodic", "initial", "interface"):
        assert set(BC_CATALOGUE[k]) == {
            "definition",
            "meaning",
            "implementation",
            "compatibility",
            "failure_modes",
        }


def test_dimensionless_numbers():
    d = dimensionless_numbers(L.convdiff_1d(pe=40.0))
    assert d["Pe"] == pytest.approx(40.0) and d["Fo"] == pytest.approx(0.5 / 40)
    assert dimensionless_numbers(L.burgers_1d(nu=0.1))["Re"] == pytest.approx(4 * 2 * np.pi / 0.1)
    assert cfl_number(2, 0.1, 0.5) == pytest.approx(0.4)
    assert diffusion_number(1, 0.01, 0.1) == pytest.approx(1.0)
    assert cell_peclet(1, 0.01, 0.1) == pytest.approx(10.0)
    p = PDEProblem("b", "e", ((0.0, 2.0),), parameters={"h": Parameter("h", 10.0), "k": Parameter("k", 5.0)})
    assert dimensionless_numbers(p)["Bi"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    "name", ["heat1d", "heat2d", "convdiff1d", "convdiff_steady", "burgers1d", "wave1d", "poisson2d"]
)
def test_library_problems_build(name):
    p = PHYSICS.get(name)()
    assert p.exact is not None and p.summary()["name"] == name


def test_exact_solutions_satisfy_pdes():
    """Finite-difference residual of each analytical solution is O(h^2) small."""
    h, t = 1e-4, 0.3
    x = np.linspace(0.1, 0.9, 7)
    p = L.convdiff_1d(pe=20.0)
    u = p.exact
    D = p.p("D")
    res = (
        (u(x, t + h) - u(x, t - h)) / (2 * h)
        + (u(x + h, t) - u(x - h, t)) / (2 * h)
        - D * (u(x + h, t) - 2 * u(x, t) + u(x - h, t)) / h**2
    )
    assert np.abs(res).max() < 1e-4
    b = L.burgers_1d(nu=0.1)
    ub = b.exact
    xx = np.linspace(0.5, 5.5, 9)
    r = (
        (ub(xx, t + h) - ub(xx, t - h)) / (2 * h)
        + ub(xx, t) * (ub(xx + h, t) - ub(xx - h, t)) / (2 * h)
        - 0.1 * (ub(xx + h, t) - 2 * ub(xx, t) + ub(xx - h, t)) / h**2
    )
    assert np.abs(r).max() < 1e-3
    s = L.convdiff_steady(pe=30.0)
    assert s.exact(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-14)
    assert s.exact(np.array([1.0]))[0] == pytest.approx(1.0)
