"""Tests for the whole-design shape optimiser. Physics checks use known analytic results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aerosynapse.numerics.panel import solve_panel
from aerosynapse.physics.airfoil import cosine_spacing
from aerosynapse.shapeopt import dynamics, multielement, topology, vii
from aerosynapse.shapeopt import geometry as G
from aerosynapse.shapeopt import wing as wg
from aerosynapse.shapeopt.cmaes import cmaes

DATA = Path(__file__).resolve().parents[1] / "src" / "aerosynapse" / "data"


def section(code: str = "naca2412"):
    return G.fit_cst(G.load_shape(code))


@pytest.mark.unit
def test_cst_fit_represents_airfoils_and_rejects_blocks():
    assert section()[1] < 2e-4
    assert G.fit_cst(G.load_shape("square"))[1] > 3e-3


@pytest.mark.unit
def test_selig_and_panel_orders_are_opposite():
    xy = G.load_shape("naca2412")
    X, Y = G.to_panel_order(xy)
    assert xy[5, 1] > 0  # Selig starts along the upper surface
    assert Y[5] < 0  # the panel order starts along the lower surface


@pytest.mark.unit
def test_baseline_satisfies_its_own_limits():
    W, _ = section()
    assert np.all(G.section_violations(W[None], G.SectionLimits.from_baseline(W)) <= 1e-9)


@pytest.mark.unit
def test_load_shape_rejects_unknown():
    with pytest.raises(Exception, match="unknown shape"):
        G.load_shape("not_a_shape")


@pytest.mark.unit
def test_cmaes_solves_rosenbrock():
    def f(X):
        x = X * 4 - 2
        return np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, axis=1)

    r = cmaes(f, np.full(5, 0.3), 0.2, seed=1, budget=20000)
    assert r.f < 1e-6


@pytest.mark.verification
def test_elliptic_wing_has_unit_span_efficiency():
    p = wg.Planform(16.0, 8.0, 1.0, 0.0, 1.0, 0.0)
    m = wg.build_mesh(p, n=60)
    eta = m.y / (p.span / 2)
    c = np.sqrt(np.maximum(1 - eta**2, 1e-6)) * 4 * p.area / (np.pi * p.span)
    me = wg.Mesh(m.y, m.z, c, m.ds, m.phi, m.twist, m.is_wing, m.A, p.area)
    _, g, _, _ = wg.linear_load(me, 0.5, -2.0, 0.11)
    cdi = -2 * 2 * (g * (g @ me.A.T) * me.ds).sum() / me.S
    assert 0.25 / (np.pi * 8 * cdi) == pytest.approx(1.0, abs=0.02)


@pytest.mark.verification
def test_newton_lifting_line_matches_linear_solution():
    p = wg.Planform(16.0, 8.0, 1.0, 0.0, 1.0, 0.0)
    m = wg.build_mesh(p, n=40)
    ag = np.arange(-14.0, 22.0, 2.0)
    r = wg.solve_llt(m, np.array([0.5]), ag, (0.11 * (ag + 2))[None], np.zeros((1, len(ag))), iters=30)
    _, g, a, _ = wg.linear_load(m, 0.5, -2.0, 0.11)
    assert r.converged
    assert r.alpha[0] == pytest.approx(a, abs=1e-6)
    assert np.abs(r.gamma[0] - g).max() < 1e-6


@pytest.mark.verification
def test_winglet_reduces_induced_drag():
    ag = np.arange(-14.0, 22.0, 2.0)
    tab, z = (0.11 * (ag + 2))[None], np.zeros((1, len(ag)))
    cdi = [
        wg.solve_llt(wg.build_mesh(wg.Planform(16.0, 8.0, 1.0, 0.0, 1.0, h), n=30), np.array([0.5]), ag, tab, z).CDi[0]
        for h in (0.0, 0.12)
    ]
    assert cdi[1] < cdi[0]


@pytest.mark.verification
@pytest.mark.parametrize("mu,xa,sg", [(20, 0.1, 0.4), (10, 0.2, 0.3), (40, 0.15, 0.3)])
def test_flutter_k_method_agrees_with_state_space(mu, xa, sg):
    s = dynamics.Section2DOF(mu=mu, a=-0.2, x_alpha=xa, r2=0.24, sigma=sg)
    assert dynamics.flutter_k(s)[0] == pytest.approx(dynamics.flutter_statespace(s), rel=0.03)


@pytest.mark.unit
def test_theodorsen_limits():
    assert dynamics.theodorsen(np.array([1e-4]))[0].real == pytest.approx(1.0, abs=0.01)
    assert dynamics.theodorsen(np.array([1e3]))[0].real == pytest.approx(0.5, abs=0.01)


@pytest.mark.unit
def test_quasi_static_pitching_has_no_hysteresis():
    a = np.arange(-6.0, 26.0, 0.5)
    cl = 0.1 * (a + 2) * (1 - 0.5 / (1 + np.exp(-(a - 15))))
    fast = dynamics.dynamic_stall(a, cl, 12.0, 8.0, k=0.1)
    slow = dynamics.dynamic_stall(a, cl, 12.0, 8.0, k=0.001, steps=4000, cycles=2)
    assert abs(slow["loop_area"]) < 0.25 * abs(fast["loop_area"])


@pytest.mark.validation
def test_independent_solver_tracks_nasa_data_before_stall():
    d = json.loads((DATA / "naca0012_ladson.json").read_text())
    L = np.array(d["rows"])
    L = L[(L[:, 0] >= -2) & (L[:, 0] <= 12.5)]
    X, Y = G.to_panel_order(G.load_shape("naca0012"))
    p = vii.polar(X, Y, L[:, 0], 6e6, 9.0, 0.05)
    assert np.mean(np.abs(p["CD"] / L[:, 2] - 1)) < 0.08
    assert np.sqrt(np.mean((p["CL"] - L[:, 1]) ** 2)) < 0.1


@pytest.mark.validation
def test_independent_solver_stalls():
    X, Y = G.to_panel_order(G.load_shape("naca0012"))
    p = vii.polar(X, Y, np.array([8.0, 14.0, 20.0]), 6e6, 9.0, 0.05)
    assert p["CL"][2] < p["CL"][1]


@pytest.mark.verification
def test_multi_element_reduces_to_repository_panel_solver():
    W, _ = section()
    x = cosine_spacing(61)
    yu, yl = G.surfaces(W, x)
    one = multielement.to_panel(np.c_[x, yu], np.c_[x, yl])
    r = multielement.solve_multi([one], np.array([4.0]))
    assert r["cl"][0] == pytest.approx(solve_panel(one[:, 0], one[:, 1], 4.0).cl, rel=1e-9)
    far = multielement.solve_multi([one, one + np.array([0.0, 200.0])], np.array([4.0]))
    assert far["cl"][0] == pytest.approx(2 * r["cl"][0], rel=1e-3)


@pytest.mark.unit
def test_stalled_layout_is_not_reported_as_lift():
    W, _ = section()
    x = cosine_spacing(61)
    yu, yl = G.surfaces(W, x)
    one = multielement.to_panel(np.c_[x, yu], np.c_[x, yl])
    assert multielement.clmax([one], 1e-3)[0] == 0.0


@pytest.fixture(scope="module")
def small_flow():
    return topology.BrinkmanNS(topology.Grid(lx=1.5, ly=1.0, h=1 / 8, box=(0.3, 1.0, 0.3, 0.7)))


@pytest.mark.verification
@pytest.mark.parametrize("U", [(1.0, 0.0), (0.98, 0.2)])
def test_free_stream_is_an_exact_solution(small_flow, U):
    ns = small_flow
    R, _ = ns.residual_jacobian(ns.initial(U), np.zeros(ns.Nu), np.zeros(ns.Nv), 0.01, U)
    assert np.abs(R).max() < 1e-12


@pytest.mark.verification
def test_flow_jacobian_matches_finite_difference(small_flow):
    ns = small_flow
    rng = np.random.default_rng(0)
    U = (0.98, 0.2)
    w = ns.initial(U) + 0.1 * rng.standard_normal(ns.N)
    au, av = rng.random(ns.Nu), rng.random(ns.Nv)
    _, J = ns.residual_jacobian(w, au, av, 0.02, U)
    d = rng.standard_normal(ns.N)
    e = 1e-6
    fd = (ns.residual_jacobian(w + e * d, au, av, 0.02, U)[0] - ns.residual_jacobian(w - e * d, au, av, 0.02, U)[0]) / (2 * e)
    assert np.abs(fd - J @ d).max() < 1e-7 * np.abs(J @ d).max()


@pytest.mark.verification
def test_adjoint_gradient_matches_finite_difference():
    g = topology.Grid(lx=3, ly=2, h=1 / 8, box=(0.6, 1.8, 0.6, 1.4))
    T = topology.Topology(g, [topology.FlowState(30.0, 5.0)], a_max=500.0, r_filter=0.2, area_cells=25.0)
    X, Y = np.meshgrid(g.xc, g.yc, indexing="ij")
    x = np.where(((X - 1.2) / 0.4) ** 2 + ((Y - 1.0) / 0.25) ** 2 < 1, 0.8, 0.1) * T.mask
    _, gr, *_ = T.objective(x, 2.0)
    dx = np.zeros_like(x)
    dx[10:13, 7:10] = 1.0
    e = 1e-4
    fd = (T.objective(x + e * dx, 2.0)[0] - T.objective(x - e * dx, 2.0)[0]) / (2 * e)
    assert float((gr * dx).sum()) == pytest.approx(fd, rel=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("beta", [1.0, 4.0, 16.0])
def test_physical_area_is_exact_even_for_thinly_spread_material(beta):
    """A limit on the raw field lets material be smeared below the threshold until the body vanishes."""
    g = topology.Grid(lx=3, ly=2, h=1 / 8, box=(0.6, 1.8, 0.6, 1.4))
    T = topology.Topology(g, [topology.FlowState(30.0)], r_filter=0.2, area_cells=20.0)
    smeared = np.where(T.mask, 0.05, 0.0)
    assert T.field(smeared, beta)[0].sum() == pytest.approx(20.0, rel=1e-6)


@pytest.mark.unit
def test_body_and_hole_counting():
    a = np.zeros((20, 20))
    a[2:8, 2:8] = 1
    a[12:18, 12:18] = 1
    a[4:6, 4:6] = 0
    assert topology.count_bodies(a) == (2, 1)


@pytest.mark.slow
def test_neuralfoil_matches_published_lift_when_order_is_right():
    pytest.importorskip("neuralfoil")
    from aerosynapse.shapeopt import aero2d

    W, _ = section("naca0012")
    p = aero2d.polar(W, 6e6, 9.0, 0.05, alpha=np.array([4.04]), model="large")
    assert p["CL"][0] == pytest.approx(0.4316, abs=0.03)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [11, 1000, 1001, 2001, 3001])
def test_problem_scores_the_baseline_as_exactly_one(seed):
    """Fails if a random state is impossible to fly: the baseline itself would be penalised."""
    pytest.importorskip("neuralfoil")
    from aerosynapse.shapeopt.objective import Problem

    W, _ = section()
    plan = wg.Planform(16.2, 7.5, 1.0, 0.0, 1.0, 0.0)
    P = Problem(W, plan, n_random_2d=6, n_random_3d=10, seed=seed)
    e = P.evaluate(P.encode(W, plan))
    assert e["J"][0] == pytest.approx(1.0, abs=1e-9)
    assert np.all(e["g"][0] <= 1e-6)


@pytest.mark.unit
def test_random_flight_states_stay_inside_the_envelope():
    from aerosynapse.shapeopt.objective import RHO_SL, flight_states

    v_s = 27.0
    fs = flight_states(400, 3, v_s)
    sig = fs.rho / RHO_SL
    assert np.all(fs.n <= np.maximum(0.8 * sig * (fs.V / v_s) ** 2, 1.0) + 1e-9)
    assert np.all(fs.V[7:] >= 1.25 * v_s / np.sqrt(sig[7:]) - 1e-9)


def test_pick_best_prefers_feasible_designs():
    """A feasible design is reported before any infeasible one, whatever their penalised scores (KI-69)."""
    from aerosynapse.shapeopt.search import pick_best, run_feasible

    runs = [{"method": "cma-es", "seed": 0, "holdout_J": 0.9254, "holdout_feasible": False},
            {"method": "cma-es", "seed": 1, "holdout_J": 0.9260, "holdout_feasible": True},
            {"method": "differential-evolution", "seed": 7, "holdout_J": 0.90, "holdout_feasible": True}]
    assert pick_best(runs)["seed"] == 1
    # older records carry feasibility only in the trace round that produced the reported score
    old = {"method": "cma-es", "seed": 2, "holdout_J": 0.9251, "trace": [{"holdout_J": 0.93, "feasible": True}, {"holdout_J": 0.9251, "feasible": False}]}
    assert not run_feasible(old)
    assert pick_best([old, runs[1]])["seed"] == 1
    assert pick_best([old])["seed"] == 2  # with no feasible run, the lowest score is still reported
