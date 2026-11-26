"""Step 5.5: parameter uncertainty (src/inverse/uq.py) and the experiment's helpers (experiments/p5_5_uq.py)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import lfilter

from aerosynapse.thermal.experiments import p5_5_uq as P55
from aerosynapse.thermal.inverse import classical as C
from aerosynapse.thermal.inverse import data as D
from aerosynapse.thermal.inverse import spec as IS
from aerosynapse.thermal.inverse import uq as U
from aerosynapse.thermal.viz import inverse_figs as VF

SMALL = {"measurement": {"sensors": {"n": 3}, "times": {"n": 12}}}
UNK = ["alpha", "k", "h"]


def _setup():
    s = IS.load_spec(overrides=SMALL)
    d = D.load(s)
    f = C.fit(s, d, UNK, {k: float(s["initial_guess"][k]) for k in UNK}, "analytic")
    return s, d, f


def test_autocorrelation_time():
    rng = np.random.default_rng(0)
    assert U.autocorr_time(rng.standard_normal(20000)) == pytest.approx(1.0, abs=0.15)
    phi = 0.9  # AR(1): tau = (1 + phi) / (1 - phi) = 19; estimator sd ~ 4 % at 2e5 samples
    x = lfilter([1.0], [1.0, -phi], rng.standard_normal(200000))
    assert U.autocorr_time(x) == pytest.approx((1 + phi) / (1 - phi), rel=0.1)


def test_metropolis_samples_a_gaussian():
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    P = np.linalg.inv(cov)
    r = U.metropolis(lambda x: -0.5 * float(x @ P @ x), np.zeros(2), np.eye(2), 20000, 2000, 250, 1)
    S = r["samples_log"]
    assert np.allclose(S.mean(0), 0, atol=0.1) and np.allclose(np.cov(S.T), cov, atol=0.12)
    assert 0.15 < r["acceptance"] < 0.6 and min(r["ess"]) > 500


def test_fit_without_covariance():
    s, d, f = _setup()
    g = C.fit(s, d, UNK, f["estimates"], "analytic", with_cov=False)
    assert "sd_log" not in g and all(g["estimates"][k] == pytest.approx(f["estimates"][k], rel=1e-5) for k in UNK)


def test_bootstrap_and_posterior_agree_with_gauss_newton_in_the_linear_regime():
    s, d, f = _setup()
    gn = np.array([f["sd_log"][k] for k in UNK])
    for kind in ("parametric", "residual"):
        b = U.bootstrap(s, d, UNK, f, 60, kind, 3)
        sd = np.array([b["sd_log"][k] for k in UNK])
        assert b["failed"] == 0 and b["samples_log"].shape == (60, 3)
        assert np.all(np.abs(sd / gn - 1) < 0.35), (kind, sd / gn)
        assert set(b["bias_corrected"]) == set(UNK)
    p = U.posterior(s, d, UNK, f, 6000, 1500, 250, 100.0, 4)
    sd = p["samples_log"].std(0)
    assert np.all(np.abs(sd / gn - 1) < 0.35) and max(p["at_prior_edge"].values()) == 0.0
    lo, hi = np.asarray(p["bounds_log"]).T
    assert np.all(p["samples_log"] >= lo) and np.all(p["samples_log"] <= hi)


def test_intervals():
    S = np.random.default_rng(0).standard_normal((20000, 2))
    iv = U.percentile_interval(S, ["a", "b"], 0.95)
    assert iv["a"][0] == pytest.approx(-1.96, abs=0.05) and iv["b"][1] == pytest.approx(1.96, abs=0.05)
    fit = {"estimates": {"a": 1.0}, "sd_log": {"a": 0.1}}
    assert U.gn_interval(fit, ["a"], 0.68)["a"][1] == pytest.approx(0.0994, abs=1e-3)


def test_analyse_and_calibration_bookkeeping():
    s = IS.load_spec(overrides=IS.deep_merge(SMALL, {"uq": {"mcmc": {"steps": 800, "burn": 300}}}))
    d = D.load(s)
    r = P55.analyse(s, d, 0, 10, True)
    q = P55.quantities(UNK)
    assert q == UNK + ["Bi"] and set(r["intervals"]) == {"0.5", "0.68", "0.9", "0.95"}
    for lv in r["intervals"]:
        for m in P55.METHODS:
            assert all(r["intervals"][lv][m][qq][0] < r["intervals"][lv][m][qq][1] for qq in q)
    assert np.isfinite(r["sd_gn"][3]) and len(r["points"]["bias_corrected"]) == 4
    cal = P55.calibration([r, r], s)
    for cv in cal["coverage"].values():
        assert all(0.0 <= v <= 1.0 for lv in cv for v in cv[lv].values())
    xt = P55.truth_log(s)
    assert xt[3] == pytest.approx(np.log(100.0 / 15.0))


def test_ensemble_statistics():
    members = [{"log_ratio": [0.1 + 0.01 * i, -0.5 + 0.2 * i]} for i in range(5)]
    e = P55.ensemble(members, ["a", "b"])
    assert e["members"] == 5 and e["covers_truth"]["a"] is False and e["covers_truth"]["b"] is True
    assert e["t_quantile"] == pytest.approx(2.776, abs=1e-3)


@pytest.mark.parametrize(
    "over, msg",
    [
        ({"uq": {"model": "x"}}, "uq.model"),
        ({"uq": {"levels": [1.2]}}, "uq.levels"),
        ({"uq": {"bootstrap": {"kinds": ["wild"]}}}, "uq.bootstrap.kinds"),
        ({"uq": {"mcmc": {"prior_factor": 1.0}}}, "prior_factor"),
        ({"studies": {"uq": {"pinn": {"variant": "nope"}}}}, "studies.uq.pinn.variant"),
        ({"studies": {"uq": {"batch": 0}}}, "studies.uq.batch"),
    ],
)
def test_validation(over, msg):
    with pytest.raises(ValueError, match=msg):
        IS.load_spec(overrides=over)


def test_figures():
    q = UNK + ["Bi"]
    lv = [0.5, 0.68, 0.9, 0.95]
    rel = {"base": {m: {"levels": lv} | {qq: [0.5, 0.7, 0.88, 0.94] for qq in q} for m in P55.METHODS}}
    iv = {
        "base": {m: {"95": {qq: [-0.1, 0.1] for qq in q}, "68": {qq: [-0.05, 0.05] for qq in q}} for m in P55.METHODS}
        | {"_point": {qq: 0.01 for qq in q}}
    }
    rng = np.random.default_rng(0)
    pairs = {
        "base": {
            "posterior": rng.normal(0, 0.05, (300, 3)),
            "bootstrap": rng.normal(0, 0.05, (50, 3)),
            "gn_cov": np.eye(3) * 0.0025,
            "x_hat": np.zeros(3),
        }
    }
    p = {
        "members": [{"seed": i, "log_ratio": [0.1 * i] * 4} for i in range(3)],
        "ensemble": P55.ensemble([{"log_ratio": [0.1 * i] * 4} for i in range(3)], q),
    }
    for f in (
        VF.reliability(rel, q, P55.LABEL),
        VF.uq_intervals(iv, q, P55.LABEL),
        VF.posterior_pairs(pairs, UNK),
        VF.pinn_ensemble(p, {qq: [-0.1, 0.1] for qq in q}),
    ):
        assert isinstance(f, plt.Figure)
        plt.close(f)
