"""Step 2.1: parameter design and snapshot generators (spec-driven)."""

import numpy as np
import pytest

from aerosynapse.thermal.data import design as D
from aerosynapse.thermal.data import generators as G
from aerosynapse.thermal.data import spec as SP
from aerosynapse.thermal.physics import analytic as an

SPEC = SP.load_spec(None, "smoke")
DIMS3 = [D.Dim("Re", 100, 1000), D.Dim("Pr", 0.7, 7), D.Dim("Bi", 0.1, 100)]
SIZES = {"train": 12, "val": 3, "interp": 4, "extrap": 4}


def test_dim_log_and_linear_roundtrip():
    d = D.Dim("Pe", 1.0, 1000.0)
    u = np.array([0.0, 0.5, 1.0])
    assert np.allclose(d.to_phys(u), [1.0, 10**1.5, 1000.0]) and np.allclose(d.to_unit(d.to_phys(u)), u)
    lin = D.Dim("Ri", -1.0, 3.0, "linear")
    assert np.allclose(lin.to_phys(u), [-1.0, 1.0, 3.0]) and np.allclose(lin.to_unit(lin.to_phys(u)), u)
    with pytest.raises(ValueError):
        D.Dim("x", -1.0, 1.0, "log")
    with pytest.raises(ValueError):
        D.Dim("x", 2.0, 1.0)


def test_design_deterministic_sizes_ranges():
    a = D.design_case("c", DIMS3, SIZES, seed=3)
    b = D.design_case("c", DIMS3, SIZES, seed=3)
    c = D.design_case("c", DIMS3, SIZES, seed=4)
    assert np.array_equal(a.u, b.u) and not np.array_equal(a.u, c.u)
    assert [int((a.split == i).sum()) for i in range(4)] == [12, 3, 4, 4]
    p = a.params
    for k, dm in enumerate(DIMS3):
        assert dm.lo * (1 - 1e-12) <= p[:, k].min() and p[:, k].max() <= dm.hi * (1 + 1e-12)
    assert np.all(a.extrap_distance()[a.idx("extrap")] > 0) and np.all(a.extrap_distance()[a.idx("train")] == 0)


def test_design_allows_zero_size_splits_and_custom_limit():
    des = D.design_case("c", DIMS3[:1], {"train": 6, "val": 0, "interp": 2, "extrap": 0}, seed=1, u_lim=0.6)
    assert des.idx("val").size == 0 and des.idx("extrap").size == 0 and np.all(des.u <= 0.6 + 1e-12)
    assert D.check_leakage(des)["all_ok"]


@pytest.mark.parametrize("method", ["sobol", "lhs"])
def test_leakage_checks_pass_and_detect_violations(method):
    des = D.design_case("c", DIMS3, SIZES, seed=1, method=method)
    assert D.check_leakage(des)["all_ok"]
    bad = D.design_case("c", DIMS3, SIZES, seed=1, method=method)
    bad.u[bad.idx("val")[0]] = bad.u[bad.idx("train")[0]]
    res = D.check_leakage(bad)
    assert not res["all_ok"] and not res["all_points_unique"]
    bad2 = D.design_case("c", DIMS3, SIZES, seed=1, method=method)
    bad2.u[bad2.idx("interp")[0], 0] = 0.95
    assert not D.check_leakage(bad2)["train_val_interp_inside_train_box"]


def test_case_a_matches_analytic_and_user_conditions():
    cfg = SPEC["cases"]["caseA"]
    out = G.gen_case_a(np.array([1.0, 50.0, 800.0]), cfg)
    assert out["theta"].shape == (3, 10, 513)
    i = np.argmin(abs(out["x"] - 0.5))
    pu = cfg["conditions"]["pulse"]
    assert out["theta"][1, 4, i] == pytest.approx(
        float(an.gaussian_pulse(0.5, out["t"][4], 50.0, pu["sigma0"], pu["x0"], pu["c"]))
    )
    assert out["theta_steady"][2][-1] == pytest.approx(1.0) and out["theta_steady"][2][0] == pytest.approx(0.0)
    cfg2 = SP.deep_merge(
        cfg,
        {
            "conditions": {
                "pulse": {"amplitude": 3.0, "baseline": 0.5},
                "steady": {"theta_left": 2.0, "theta_right": 5.0},
            }
        },
    )
    o2 = G.gen_case_a(np.array([50.0]), cfg2)
    assert o2["theta"][0, 4, i] == pytest.approx(0.5 + 3.0 * out["theta"][1, 4, i])
    assert o2["theta_steady"][0, 0] == pytest.approx(2.0) and o2["theta_steady"][0, -1] == pytest.approx(5.0)


def test_case_b_monotone_relaxation_and_conditions():
    cfg = SPEC["cases"]["caseB"]
    out = G.gen_case_b(np.array([0.1, 10.0]), cfg)
    s = cfg["conditions"]["slab"]
    err = [np.max(np.abs(out["theta"][1, k] - out["theta_steady"][1])) for k in range(10)]
    assert np.all(np.diff(err) < 0) and err[-1] < 0.25 * err[0]
    late = an.slab_transient(out["x"], 30.0, 10.0, s["Q"], s["m"], s["theta0"])
    assert np.allclose(late, out["theta_steady"][1], atol=1e-10)
    cfg2 = SP.deep_merge(cfg, {"conditions": {"slab": {"Q": 0.0, "theta0": 2.0}}})
    o2 = G.gen_case_b(np.array([1.0]), cfg2)
    assert np.all(o2["theta_steady"] == 0) and o2["theta"][0, 0].max() <= 2.0 + 1e-6


def _small_c(**cond):
    cfg = SP.deep_merge(SPEC["cases"]["caseC"], {"grid": 16, "times": {"kind": "list", "values": [0.5, 1.0, 1.5]}})
    return SP.deep_merge(cfg, cond) if cond else cfg


PARAMS = {"Re": 200.0, "Pr": 1.0, "Bi": 5.0, "Ri": 0.0}


def test_case_c_default_physical():
    cfg = _small_c()
    r = G.gen_case_c_point(PARAMS, G.snapshot_times(cfg["times"]), cfg)
    assert r["theta"].shape == (3, 16, 16) and r["max_div"] < 1e-11
    assert -1e-6 <= r["theta"].min() and r["theta"].max() <= 1.0 + 1e-6
    assert np.all(np.diff(r["ke"]) > 0) and r["nu_top"][-1] > 0


def test_case_c_user_defined_walls_change_the_physics():
    base = _small_c()
    times = G.snapshot_times(base["times"])
    r0 = G.gen_case_c_point(PARAMS, times, base)
    cfg = _small_c(
        conditions={
            "thermal": {
                "top": {"type": "robin", "Bi": 3.0, "theta_inf": 0.0},
                "bottom": {"type": "dirichlet", "value": 1.0},
                "left": {"type": "dirichlet", "value": 0.5},
                "right": {"type": "dirichlet", "value": 0.5},
            },
            "velocity": {"top": {"profile": "expr", "expr": "sin(pi*s)", "amplitude": 0.7}},
            "initial_theta": 0.5,
        }
    )
    r1 = G.gen_case_c_point(PARAMS, times, cfg)
    assert np.abs(r1["theta"] - r0["theta"]).max() > 0.1
    assert r1["theta"][-1][0, :].mean() > r1["theta"][-1][-1, :].mean()  # bottom row hotter than top row
    assert np.all(r1["theta"][:, :, 0] > 0.3)  # next to the 0.5 side wall
    assert r1["ke"][-1] < r0["ke"][-1]  # weaker 0.7 sin lid drives less kinetic energy


def test_case_c_bi_as_number_and_buoyancy_runs():
    cfg = _small_c(conditions={"thermal": {"bottom": {"type": "robin", "Bi": 5.0}}})
    times = G.snapshot_times(cfg["times"])
    a = G.gen_case_c_point(PARAMS, times, cfg)
    b = G.gen_case_c_point(PARAMS, times, _small_c())
    assert np.allclose(a["theta"], b["theta"])  # numeric Bi = 5 identical to the named parameter Bi = 5
    c = G.gen_case_c_point({**PARAMS, "Ri": 2.0}, times, cfg)
    assert np.abs(c["v"] - a["v"]).max() > 1e-4 and c["max_div"] < 1e-11


def test_restriction_conserves_mean():
    f = np.random.default_rng(0).standard_normal((2, 8, 8))
    c = G.restrict_cell_average(f, 2)
    assert c.shape == (2, 4, 4) and np.allclose(c.mean(axis=(-2, -1)), f.mean(axis=(-2, -1)))
