"""Phase 10.1 estimation engine (src/inverse/realdata.py) on synthetic data with known answers."""

import numpy as np
import pytest

from aerosynapse.thermal.inverse import realdata as R

DAY = R.DAY


def wave_series(alpha=4e-7, days=6, zs=(0.05, 0.10, 0.20, 0.50), noise=0.0, seed=0, mean=15.0, amp=8.0):
    """Exact damped diurnal wave at the sensor depths (hourly), optionally with Gaussian noise."""
    w = 2 * np.pi / DAY
    d = np.sqrt(2 * alpha / w)
    t = np.arange(0, days * DAY, 3600.0)
    z = np.array(zs)
    T = mean + amp * np.exp(-z[None] / d) * np.sin(w * t[:, None] - z[None] / d)
    T += np.random.default_rng(seed).normal(0, noise, T.shape) if noise else 0
    return t, z, T


def test_forward_model_second_order_in_space_and_time():
    e = [R.verify_wave(5e-7, 10.0, DAY, 0.05, 0.45, dz, 60.0, days=1.0) for dz in (0.02, 0.01, 0.005)]
    assert np.polyfit(np.log([0.02, 0.01, 0.005])[-2:], np.log(e[-2:]), 1)[0] == pytest.approx(2.0, abs=0.15)
    e = [R.verify_wave(5e-7, 10.0, DAY, 0.05, 0.45, 0.005, dt, days=1.0) for dt in (3600.0, 1800.0, 900.0)]
    assert np.polyfit(np.log([3600.0, 1800.0, 900.0])[-2:], np.log(e[-2:]), 1)[0] == pytest.approx(2.0, abs=0.2)


def test_fast_propagator_equals_stepwise_solver():
    t, z, T = wave_series()
    m = R.Conduction1D(t, z, T, 0.05, 0.50, [0.10, 0.20], dz=0.005, dt=600, layer_edges=[0.15])
    a = np.array([3e-7, 6e-7])
    assert np.max(np.abs(m.run(a) - m.run_stepwise(a))) < 1e-10
    ac = a + 1e-25j
    assert np.max(np.abs(m.run(ac) - m.run_stepwise(ac))) < 1e-10
    full = m.run(a, full=True)
    assert np.allclose(m.at(a, [0.10, 0.20]), m.run(a), atol=1e-10)  # cubic interpolation is exact at nodes
    assert full.shape == (len(t), m.N + 1)


def test_complex_step_jacobian_matches_central_differences():
    t, z, T = wave_series(noise=0.05)
    win = {"t": t, "z": z, "T": T}
    prob = R.conduction_problem(
        win,
        {"top_m": 0.05, "bottom_m": 0.5, "inner_m": [0.1, 0.2], "dz_m": 0.005, "dt_s": 600, "spinup_h": 24},
        {"layer_edges_m": [0.15], "depths": True},
    )
    x = np.array([np.log(4e-7), np.log(5e-7), 0.004, -0.003])
    Jc = prob.jacobian(x)
    Jf = np.stack(
        [
            (prob.residual(x + h * e) - prob.residual(x - h * e)) / (2 * h)
            for e, h in zip(np.eye(4), (1e-6, 1e-6, 1e-7, 1e-7), strict=True)
        ],
        1,
    )
    assert np.linalg.norm(Jc - Jf) / np.linalg.norm(Jc) < 1e-5


def test_fit_recovers_alpha_from_noisy_profile_and_holdout():
    t, z, T = wave_series(alpha=4e-7, days=8, noise=0.05)
    win = {"t": t[: 4 * 24], "z": z, "T": T[: 4 * 24]}
    test = {"t": t[4 * 24 :], "z": z, "T": T[4 * 24 :]}
    mcfg = {"top_m": 0.05, "bottom_m": 0.5, "inner_m": [0.1, 0.2], "dz_m": 0.005, "dt_s": 600, "spinup_h": 24}
    fcfg = {"alpha_init": 1e-6, "alpha_bounds": [1e-8, 1e-5], "tol": 1e-10, "hac_lag_h": 12}
    r = R.fit_conduction(win, mcfg, fcfg)
    assert r["fit"]["converged"]
    row = r["rows"][0]
    assert row["estimate"] == pytest.approx(4e-7, rel=0.03)
    assert row["ci95_hac"][0] < 4e-7 < row["ci95_hac"][1]
    h = R.holdout(win, test, mcfg, fcfg, r["x"])
    assert all(s > 0.5 for s in h["skill"])  # physics beats linear interpolation in depth


def test_pipeline_converges_from_a_poor_start_and_reports_attempts():
    t, z, T = wave_series(alpha=4e-7, days=5, noise=0.05)
    win = {"t": t, "z": z, "T": T}
    mcfg = {"top_m": 0.05, "bottom_m": 0.5, "inner_m": [0.1, 0.2], "dz_m": 0.005, "dt_s": 600, "spinup_h": 24}
    fcfg = {
        "alpha_init": 9.9e-6,
        "alpha_bounds": [1e-8, 1e-5],
        "tol": 1e-10,
        "strategies": ["multistart"],
        "multistart_n": 4,
    }
    r = R.fit_conduction(win, mcfg, fcfg)
    assert r["fit"]["converged"] and r["rows"][0]["estimate"] == pytest.approx(4e-7, rel=0.03)
    assert r["attempts"][0]["strategy"] == "levenberg-marquardt"


def test_identifiability_reports_rank_deficiency_and_null_direction():
    rng = np.random.default_rng(0)
    a = rng.normal(size=50)
    J = np.stack([a, -a, rng.normal(size=50)], 1)  # columns 1 and 2 confounded (as k and h without a source)
    ident = R.identifiability(J, ["log_k", "log_h", "log_alpha"])
    assert ident["rank"] == 2
    nd = ident["null_directions"][0]
    assert abs(abs(nd["log_k"]) - abs(nd["log_h"])) < 1e-8 and abs(nd["log_alpha"]) < 1e-8


def test_stationarity_is_zero_at_a_least_squares_solution():
    rng = np.random.default_rng(1)
    J = rng.normal(size=(40, 3))
    y = rng.normal(size=40)
    x = np.linalg.lstsq(J, y, rcond=None)[0]
    assert R.stationarity(J, J @ x - y) < 1e-12
    assert R.stationarity(J, J @ (x + 0.1) - y) > 1e-3


def test_harmonic_estimators_recover_alpha_of_a_damped_wave():
    t, z, T = wave_series(alpha=5e-7, days=10)
    h = R.harmonic(t, T, z)
    for p in h["pairs"]:
        assert p["alpha_amplitude"] == pytest.approx(5e-7, rel=1e-3)
        assert p["alpha_phase"] == pytest.approx(5e-7, rel=1e-3)


def test_lumped_cooling_recovers_tau_and_exact_integrator():
    t = np.arange(0, 7200.0, 30.0)
    amb = 22 + 0.5 * np.sin(2 * np.pi * t / 3600)
    m = R.Lumped(t, np.zeros_like(t), amb)
    T = m.run([np.log(900.0), 85.0]).real[:, 0]
    # exact for a constant ambient: T = Ta + (T0 - Ta) exp(-t / tau)
    mc = R.Lumped(t, np.zeros_like(t), 22 + 0 * t)
    assert np.allclose(mc.run([np.log(900.0), 85.0]).real[:, 0], 22 + 63 * np.exp(-t / 900), atol=1e-10)
    data = T + np.random.default_rng(0).normal(0, 0.1, t.size)
    m2 = R.Lumped(t, data, amb)
    prob = R.Problem(m2, m2.names, np.array([np.log(3000.0), 80.0]), np.array([0.0, -50.0]), np.array([20.0, 500.0]))
    r = R.fit(prob, {"tol": 1e-10, "strategies": ["multistart"]})
    assert r["result"]["converged"] and np.exp(r["x"][0]) == pytest.approx(900.0, rel=0.01)


def test_load_table_parses_iso_times_and_positions():
    csv = "time,T_5cm,T_10cm\n2023-07-01T00:30:00Z,20.0,19.5\n2023-07-01T01:30:00Z,20.5,19.6\n"
    s = R.load_table(csv, "time", {"T_5cm": 0.05, "T_10cm": 0.10})
    assert s.t.tolist() == [0.0, 3600.0] and s.z.tolist() == [0.05, 0.10] and s.T[1, 0] == 20.5
    s2 = R.load_table("t,a\n0,1\n2,3\n", "t", {"a": 0.0}, time_unit="h")
    assert s2.t.tolist() == [0.0, 7200.0]


def test_load_table_never_treats_text_as_a_path(tmp_path):
    # a long table (beyond the OS path limit: CI run 39544679087 failed with ENAMETOOLONG on Linux) parses as text
    rows = "\n".join(f"{3600 * i},{20 + 0.001 * i:.5f}" for i in range(500))
    assert R.load_table("t,a\n" + rows, "t", {"a": 0.0}).t.size == 500
    # a str naming an existing file is NOT read (the web server passes uploads as str); a Path is
    f = tmp_path / "table.csv"
    f.write_text("t,a\n0,1\n3600,2\n", encoding="utf-8")
    assert R.load_table(f, "t", {"a": 0.0}).t.tolist() == [0.0, 3600.0]
    with pytest.raises(KeyError, match="column 't' not in the table"):
        R.load_table(str(f), "t", {"a": 0.0})


def test_window_fills_short_gaps_only():
    y = np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, 8.0])
    f = R.fill_gaps(y, 2)
    assert f[1] == 2.0 and np.isnan(f[3:7]).all()


def test_freezing_model_reduces_to_conduction_and_has_exact_derivatives():
    t, z, T = wave_series(alpha=4e-7, days=3, mean=0.5, amp=3.0)  # crosses 0 degC: the latent heat matters
    args = (t, z, T, 0.05, 0.50, [0.10, 0.20])
    m0 = R.Conduction1D(*args, dz=0.01, dt=900)
    mf = R.FreezingConduction1D(*args, dz=0.01, dt=900, width=0.5)
    assert np.max(np.abs(mf.run([4e-7, 0.0]) - m0.run(np.array([4e-7])))) < 1e-9  # S = 0: plain conduction
    # the capacity bump integrates to the latent temperature S
    Tg = np.linspace(-10, 10, 20001)
    assert np.trapezoid(mf.capacity(Tg, 7.0) - 1.0, Tg) == pytest.approx(7.0, rel=1e-6)
    # latent heat slows the temperature swings near 0 degC
    assert np.ptp(mf.run([4e-7, 20.0])[24:, 0]) < np.ptp(m0.run(np.array([4e-7]))[24:, 0])
    win = {"t": t, "z": z, "T": T}
    mcfg = {"top_m": 0.05, "bottom_m": 0.5, "inner_m": [0.1, 0.2], "dz_m": 0.01, "dt_s": 900, "spinup_h": 24}
    prob = R.freezing_problem(win, mcfg, {"fit_freezing_point": True, "freeze_width_k": 0.5})
    x = np.array([np.log(4e-7), np.log(5.0), -0.2])
    Jc = prob.jacobian(x)
    Jf = np.stack(
        [
            (prob.residual(x + h * e) - prob.residual(x - h * e)) / (2 * h)
            for e, h in zip(np.eye(3), (1e-6, 1e-6, 1e-6), strict=True)
        ],
        1,
    )
    assert np.linalg.norm(Jc - Jf) / np.linalg.norm(Jc) < 1e-5
