"""Phase 5: optimisation engine, KKT, robust measures, protocol, surrogate loop, inverse problem."""

import numpy as np
import pytest

from aerosynapse.exceptions import BudgetExhaustedError, ConfigurationError, ConstraintViolationError
from aerosynapse.optim import demos as D
from aerosynapse.optim.algorithms import CAPABILITIES, minimize
from aerosynapse.optim.inverse import HeatInverseSetup
from aerosynapse.optim.kkt import NONCONVEX_WARNING, kkt_check
from aerosynapse.optim.pareto import hypervolume_2d, hypervolume_3d, non_dominated_sort
from aerosynapse.optim.problem import OptimizationProblem
from aerosynapse.optim.protocol import OptimizationStudy, require_trusted_feasible
from aerosynapse.optim.robust import chance, cvar, expectation, worst_case
from aerosynapse.optim.variables import DesignSpace, Variable
from aerosynapse.registry import OPTIMIZERS

pytestmark = pytest.mark.unit


def test_design_space_roundtrip():
    s = DesignSpace(
        [
            Variable("a", 1, 3),
            Variable("n", 0, 10, vtype="integer"),
            Variable("m", vtype="categorical", choices=("x", "y", "z")),
        ]
    )
    d = s.decode(np.array([0.5, 0.33, 0.9]))
    assert d == {"a": 2.0, "n": 3, "m": "z"}
    assert s.decode(s.encode(d)) == d
    with pytest.raises(ConfigurationError):
        Variable("c", vtype="categorical")
    with pytest.raises(ConfigurationError):
        Variable("c", 1, 0)


def test_budget_enforced_and_history():
    p = OptimizationProblem("r", D.rosenbrock_space(), D.rosenbrock, budget=3)
    for _ in range(3):
        p.evaluate(np.full(2, 0.5))
    with pytest.raises(BudgetExhaustedError):
        p.evaluate(np.full(2, 0.5))
    assert p.n_evals == 3 and p.history[0].feasible


@pytest.mark.parametrize(
    "method,budget,tol",
    [("lbfgs", 400, 1e-4), ("sqp", 400, 1e-4), ("de", 3000, 1e-3), ("ga", 4000, 5e-2), ("adam", 6000, 5e-2)],
)
def test_rosenbrock_solved(method, budget, tol):
    p = OptimizationProblem("r", D.rosenbrock_space(), D.rosenbrock, budget=budget)
    r = minimize(p, method, z0=np.array([0.2, 0.2]), seed=0)
    assert r.best is not None and r.best.f[0] < tol and r.n_evals <= budget


def test_constrained_qp_and_kkt():
    for m in ("sqp", "de", "cbo"):
        p = OptimizationProblem("qp", D.qp_space(), D.qp, budget=40 if m == "cbo" else 2000)
        r = minimize(p, m, seed=0)
        assert r.best is not None and r.best.feasible
        tol = 2e-2 if m == "cbo" else 2e-3
        assert abs(r.best.f[0] - 0.5) < tol, m
    rep = kkt_check(lambda x: x @ x, np.array([0.5, 0.5]), g=lambda x: np.array([1 - x[0] - x[1]]), convex=True)
    assert rep.kkt_ok and rep.licq and rep.second_order_ok and abs(rep.mu[0] - 1.0) < 1e-5 and rep.warning == ""
    bad = kkt_check(lambda x: x @ x, np.array([1.0, 1.0]), g=lambda x: np.array([1 - x[0] - x[1]]))
    assert not bad.kkt_ok and bad.warning == NONCONVEX_WARNING
    eq = kkt_check(lambda x: x[0] + x[1], np.array([-1, -1]) / np.sqrt(2), h=lambda x: np.array([x @ x - 1]))
    assert eq.kkt_ok and eq.second_order_ok


def test_bo_beats_random_on_branin():
    p = OptimizationProblem("b", D.branin_space(), D.branin, budget=30)
    r = minimize(p, "bo", seed=0)
    rng = np.random.default_rng(0)
    rand = min(D.branin(D.branin_space().decode(z)).f[0] for z in rng.random((30, 2)))
    assert r.best is not None and r.best.f[0] < rand and r.best.f[0] < 0.6


def test_nsga2_and_hypervolume():
    p = OptimizationProblem("z", D.zdt1_space(6), D.zdt1, n_obj=2, budget=3000)
    r = minimize(p, "nsga2", seed=0, pop=40)
    F = np.array([e.f for e in r.pareto])
    hv = hypervolume_2d(F, np.array([1.1, 1.1]))
    assert hv > 0.95 * D.zdt1_true_hv()
    assert hypervolume_2d(np.array([[0.5, 0.5]]), np.array([1.0, 1.0])) == pytest.approx(0.25)
    assert hypervolume_3d(np.array([[0.5, 0.5, 0.5]]), np.ones(3)) == pytest.approx(0.125)
    assert non_dominated_sort(np.array([[0, 1], [1, 0], [1, 1]]))[0] == [0, 1]


def test_ehvi_runs():
    p = OptimizationProblem("z", D.zdt1_space(2), D.zdt1, n_obj=2, budget=12)
    r = minimize(p, "ehvi", seed=0, n_cand=50, n_mc=8)
    assert r.n_evals == 12 and r.pareto


def test_robust_measures():
    s = np.random.default_rng(0).normal(size=20000)
    m, se = expectation(s)
    assert abs(m) < 4 * se
    assert cvar(s, 0.9) == pytest.approx(1.755, abs=0.05)  # CVaR_0.9 of N(0,1) = φ(1.2816)/0.1
    assert worst_case(s) == s.max()
    c = chance(s - 3.0, eps=0.05)
    assert c["satisfied"] and not chance(s, eps=0.05)["satisfied"]


def test_protocol_enforced_and_reverification():
    st = OptimizationStudy("qp")
    with pytest.raises(ConfigurationError):
        st.run("sqp", "de")
    st = OptimizationStudy(
        "qp",
        baseline_design={"x1": 1.0, "x2": 1.0},
        space=D.qp_space(),
        objective="min |x|^2",
        constraints=["1 - x1 - x2 <= 0"],
        constraints_declared=True,
        budget=300,
        stopping="budget or SLSQP ftol",
        model=D.qp,
        trusted=D.qp,
    )
    rep = st.run("sqp", "de")
    assert rep["verdict"] == "ACCEPT" and rep["method"]["final"]["trusted_feasible"]
    assert len(rep["method"]["history"]) == rep["method"]["n_evals"]
    require_trusted_feasible(rep)
    # a biased "surrogate" that ignores the constraint: trusted re-check must reject it
    fake = lambda d: D.EvalResult(f=[d["x1"] ** 2 + d["x2"] ** 2], g=[-1.0])
    st2 = OptimizationStudy(
        "qp2",
        baseline_design={"x1": 1.0, "x2": 1.0},
        space=D.qp_space(),
        objective="min",
        constraints=["1-x1-x2<=0"],
        constraints_declared=True,
        budget=200,
        stopping="budget",
        model=fake,
        trusted=D.qp,
    )
    rep2 = st2.run("sqp", "de")
    assert rep2["verdict"].startswith("REJECT")
    with pytest.raises(ConstraintViolationError):
        require_trusted_feasible(rep2)


def test_surrogate_trust_region_on_thermal():
    p = OptimizationProblem("t", D.thermal_space(False), D.thermal_model, budget=25)
    r = minimize(p, "surrogate_tr", seed=0, n_cand=300)
    assert r.best is not None and r.best.feasible and r.best.source == "trusted"


def test_demo_models_physics():
    x, th = D.thermal_field(1.0, 0.5)
    assert th[0] == pytest.approx(0.0, abs=1e-12) and th.max() > 0
    _, th2 = D.thermal_field(3.0, 0.5)
    assert th2.max() < th.max()  # stronger lateral cooling lowers peak temperature
    b = D.beam_response({"b": 0.05, "t": 0.1, "material": "steel"})
    assert b["defl"] == pytest.approx(4 * 5e3 / (200e9 * 0.05 * 0.1**3))
    r = D.source_model({"s": 0.5, "q": 1.0})
    assert len(r.g) == 1 and r.f[0] > 0
    assert set(CAPABILITIES) <= set(OPTIMIZERS.names())
    for row in CAPABILITIES.values():
        assert {
            "gradients",
            "expensive_objective",
            "constraints",
            "noise",
            "multi_objective",
            "complexity",
            "failure_modes",
        } <= set(row)


def test_adjoint_gradient_matches_fd_and_identification():
    s = HeatInverseSetup(n=32, nt=40)
    d = s.observe(s.forward(0.1))
    J, g = s.misfit_and_grad(0.08, d)
    h = 1e-6
    fd = (s.misfit_and_grad(0.08 + h, d)[0] - s.misfit_and_grad(0.08 - h, d)[0]) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-5)
    rng = np.random.default_rng(0)
    out = s.identify(d + rng.normal(0, 0.01, d.shape), noise_sigma=0.01)
    assert abs(out["alpha_hat"] - 0.1) < 4 * out["se"]
