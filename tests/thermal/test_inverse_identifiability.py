"""Step 5.1: dimensional slab forward models, Jacobian / Fisher / SVD / CRLB algebra, the structural k-h confounding
without a source, bound scaling laws, the measurement design and the config validation."""

import numpy as np
import pytest

from aerosynapse.thermal.experiments import p5_1_identifiability as E
from aerosynapse.thermal.inverse import body as SL
from aerosynapse.thermal.inverse import identifiability as ID
from aerosynapse.thermal.inverse import spec as IS
from aerosynapse.thermal.physics import analytic as an

SPEC = IS.load_spec()
P = {k: float(v) for k, v in SPEC["parameters"].items()}
SLAB = IS.body_of(SPEC)


def test_groups_and_slab_validation():
    g = SLAB.groups(P)
    assert abs(g["Bi"] - 100 * 0.02 / 15) < 1e-15 and abs(g["t_scale"] - 100.0) < 1e-9
    assert abs(g["Q"] - 1e5 * 0.02**2 / (15 * 60)) < 1e-15
    with pytest.raises(ValueError, match="T0 must differ"):
        SL.Body("slab", 0.02, 20.0, 20.0, 0.0)


def test_vectorised_series_matches_phase1_solution_and_steady_limit():
    x = np.linspace(0, 1, 7)
    t = np.array([0.01, 0.1, 1.0])
    th = SL.theta_analytic(x, t, 0.7, 0.3, 200)
    for i, tt in enumerate(t):
        assert np.abs(th[i] - an.slab_transient(x, tt, 0.7, 0.3, 0.0, 1.0, 200)).max() < 1e-13
    late = SL.theta_analytic(x, np.array([200.0]), 0.7, 0.3, 50)[0]
    assert np.abs(late - an.slab_steady(x, 0.7, 0.3)).max() < 1e-12


def test_fdm_forward_model_agrees_with_the_series():
    xs, ts = np.array([0.0, 0.01, 0.02]), np.array([5.0, 30.0, 120.0])
    Ta = SL.temperatures(P, SLAB, xs, ts, "analytic", 60)
    Tf = SL.temperatures(P, SLAB, xs, ts, "fdm", 0, {"N": 100, "dt_fourier": 5e-3})
    assert np.abs(Ta - Tf).max() < 3e-3 and Ta.shape == (3, 3)  # incl. the surface at t* = 0.05 (Rannacher start-up)


def test_jacobian_and_bounds_on_a_linear_model():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 3))
    p0 = {"a": 2.0, "b": 0.5, "c": 3.0}
    fn = lambda p: A @ np.log([p["a"], p["b"], p["c"]])  # noqa: E731  (linear in log p: J = A exactly)
    J = ID.jacobian(fn, p0, ["a", "b", "c"])
    assert np.abs(J - A).max() < 1e-9
    r = ID.analyse(J, 0.1, ["a", "b", "c"])
    C = 0.01 * np.linalg.inv(A.T @ A)
    assert np.allclose(r["crlb_log"], np.sqrt(np.diag(C)), rtol=1e-12) and r["rank"] == 3 and not r["null_directions"]
    assert np.allclose(r["correlation"], C / np.sqrt(np.outer(np.diag(C), np.diag(C))), atol=1e-12)
    Q, _ = np.linalg.qr(A)
    assert abs(ID.analyse(Q, 1.0, ["a", "b", "c"])["collinearity_index"] - 1.0) < 1e-12  # orthogonal columns: gamma = 1


def test_singular_fisher_reports_inf_and_the_null_direction():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal(30), rng.standard_normal(30)
    J = np.stack([a, b, b], 1)  # parameters 2 and 3 enter only through their sum
    r = ID.analyse(J, 1.0, ["x", "y", "z"])
    v = np.asarray(r["null_directions"][0])
    assert r["rank"] == 2 and np.allclose(np.abs(v), [0, 2**-0.5, 2**-0.5], atol=1e-12)
    assert r["crlb_percent"]["y"] == np.inf and r["crlb_percent"]["z"] == np.inf and r["correlation"] is None
    # the estimable parameter x keeps the bound of the reduced model (x and the sum y + z)
    Jr = np.stack([a, b], 1)
    assert abs(r["crlb_log"][0] - np.sqrt(np.linalg.inv(Jr.T @ Jr)[0, 0])) < 1e-10
    assert r["collinearity_index"] == np.inf


def test_structural_confounding_without_source_and_identifiability_with_it():
    xs, ts = IS.sensor_positions(SPEC), IS.measurement_times(SPEC)
    n = IS.n_terms(SPEC)
    for q, rank in ((0.0, 2), (1e5, 3)):
        sl = SL.Body("slab", SLAB.L, SLAB.T0, SLAB.T_inf, q)
        fn = lambda p, sl=sl: SL.temperatures(p, sl, xs, ts, "analytic", n)  # noqa: E731
        J = ID.jacobian(fn, P, SL.PARAMS)
        r = ID.analyse(J, IS.noise_sigma(SPEC, fn(P), 0.02), SL.PARAMS)
        assert r["rank"] == rank
        if q == 0.0:  # T depends on k and h only through Bi = hL/k: dT/dln k = -dT/dln h exactly
            assert np.abs(J[:, 1] + J[:, 2]).max() < 1e-9 * np.abs(J[:, 1]).max()
            assert np.allclose(np.abs(r["null_directions"][0]), [0, 2**-0.5, 2**-0.5], atol=1e-6)
            assert np.isfinite(r["crlb_percent"]["alpha"]) and r["crlb_percent"]["k"] == np.inf
        else:
            assert all(np.isfinite(v) for v in r["crlb_percent"].values())


def test_bound_scaling_with_noise_and_replication():
    rng = np.random.default_rng(2)
    J = rng.standard_normal((50, 3))
    c1 = np.asarray(ID.analyse(J, 0.1, list("abc"))["crlb_log"])
    assert np.allclose(np.asarray(ID.analyse(J, 0.2, list("abc"))["crlb_log"]), 2 * c1, rtol=1e-12)
    assert np.allclose(
        np.asarray(ID.analyse(np.vstack([J, J]), 0.1, list("abc"))["crlb_log"]), c1 / np.sqrt(2), rtol=1e-12
    )
    with pytest.raises(ValueError, match="noise-free"):
        ID.analyse(J, 0.0, list("abc"))


def test_measurement_design():
    s = IS.load_spec(overrides={"measurement": {"sensors": {"positions": [0.0, 0.5, 1.0]}}})
    assert np.allclose(IS.sensor_positions(s), [0.0, 0.01, 0.02])
    s = IS.load_spec(overrides={"measurement": {"sensors": {"n": 3, "include_surface": True}}})
    assert np.allclose(IS.sensor_positions(s) / 0.02, [0.25, 0.5, 0.75, 1.0])
    s = IS.load_spec(overrides={"measurement": {"times": {"kind": "geom", "t0": 1.0, "t1": 100.0, "n": 3}}})
    assert np.allclose(IS.measurement_times(s), [1, 10, 100])
    assert IS.n_terms(SPEC) >= SL.truncation_terms(5.0 * 4e-6 / 0.02**2)
    sig = IS.noise_sigma(SPEC, np.array([20.0, 80.0]), 0.1)  # theta = 0, 1: RMS 1/sqrt(2); dT = 60
    assert np.allclose(sig, 0.1 * 60 / np.sqrt(2))


@pytest.mark.parametrize(
    "over,msg",
    [
        ({"body": {"L": -1.0}}, "body.L"),
        ({"body": {"q_vol": -5.0}}, "body.q_vol"),
        ({"parameters": {"k": 0.0}}, "parameters.k"),
        ({"measurement": {"sensors": {"positions": [0.5, 1.5]}}}, "fractions of L"),
        ({"measurement": {"sensors": {"layout": "zigzag"}}}, "measurement.sensors.layout"),
        ({"measurement": {"times": {"t0": 10.0, "t1": 5.0}}}, "0 < t0 < t1"),
        ({"measurement": {"noise": {"levels": [0.0, 0.01]}}}, "levels > 0"),
        ({"forward": {"model": "fem"}}, "analytic or fdm"),
        ({"identifiability": {"thresholds": [50.0, 10.0]}}, "thresholds"),
        ({"studies": {"sweeps": {"rho": [1.0]}}}, "studies.sweeps.rho"),
    ],
)
def test_validation(over, msg):
    with pytest.raises(ValueError, match=msg):
        IS.load_spec(overrides=over)


def test_small_monte_carlo_runs():
    s = IS.load_spec(overrides={"measurement": {"sensors": {"n": 3}, "times": {"n": 10}}})
    mc = E.monte_carlo(s, 0.02, 6, 0)
    assert mc["failed"] == 0 and all(np.isfinite(v) and v > 0 for v in mc["sample_std_log"].values())


def test_fdm_output_is_smooth_in_alpha():
    """Fixed step counts (dimensional dt_seconds): the FDM Jacobian matches the analytic one in every column,
    including alpha (with a Fourier-time step the step count jumps with alpha: 2.4% column error before the fix)."""
    s = IS.load_spec(overrides={"measurement": {"sensors": {"n": 3}, "times": {"n": 8, "t1": 200.0}}})
    sl, xs, ts = IS.body_of(s), IS.sensor_positions(s), IS.measurement_times(s)
    f, n = IS.fdm_settings(s), IS.n_terms(s)
    assert np.all(SL.fdm_steps(ts, f["dt_seconds"]) >= 1) and SL.fdm_steps(ts, f["dt_seconds"])[0] >= 4
    Ja = ID.jacobian(lambda p: SL.temperatures(p, sl, xs, ts, "analytic", n), P, SL.PARAMS)
    Jf = ID.jacobian(lambda p: SL.temperatures(p, sl, xs, ts, "fdm", n, f), P, SL.PARAMS)
    assert (np.linalg.norm(Jf - Ja, axis=0) / np.linalg.norm(Ja, axis=0)).max() < 1e-3
