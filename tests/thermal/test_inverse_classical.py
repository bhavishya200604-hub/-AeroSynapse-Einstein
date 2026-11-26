"""Step 5.3: classical least-squares baseline (src/inverse/classical.py) and the experiment's helpers."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aerosynapse.thermal.experiments import p5_3_classical as P53
from aerosynapse.thermal.inverse import classical as C
from aerosynapse.thermal.inverse import data as D
from aerosynapse.thermal.inverse import identifiability as ID
from aerosynapse.thermal.inverse import spec as IS
from aerosynapse.thermal.viz import inverse_figs as VF

SMALL = {"measurement": {"sensors": {"n": 3}, "times": {"n": 12}}}
UNK = ["alpha", "k", "h"]


def _spec(**over):
    return IS.load_spec(overrides=IS.deep_merge(SMALL, over))


def _guess(s):
    return {k: float(s["initial_guess"][k]) for k in s["unknown"]}


def test_forward_settings_use_the_guess_not_the_truth():
    s = _spec()
    f = C.forward_settings(s, {"alpha": 2e-6})
    assert f["N"] == s["classical"]["fdm"]["N"] and f["dt_fourier"] == s["classical"]["fdm"]["dt_fourier"]
    assert f["dt_seconds"] == pytest.approx(f["dt_fourier"] * 0.02**2 / 2e-6)
    assert C.forward_settings(s, {"alpha": 4e-6})["dt_seconds"] == pytest.approx(f["dt_seconds"] / 2)


def test_noise_free_data_recover_the_truth():
    s = _spec()
    d = D.synthetic(s, 0.02, 0)
    d["T"] = d["T_clean"].copy()
    r = C.fit(s, d, UNK, _guess(s), "analytic")
    assert r["success"] and r["chi2_per_reading"] < 1e-12
    for k in UNK:
        assert r["estimates"][k] == pytest.approx(float(s["parameters"][k]), rel=1e-6)
    assert r["n_model_calls"] == len(r["path"]) and r["path"][-1]["cost"] >= 0


def test_fdm_and_analytic_estimators_agree_and_intervals_match_the_bound():
    s = _spec()
    d = D.load(s)
    fa = C.fit(s, d, UNK, _guess(s), "analytic")
    ff = C.fit(s, d, UNK, _guess(s), "fdm")
    for k in UNK:  # FDM discretisation bias far below the statistical spread
        assert abs(np.log(ff["estimates"][k] / fa["estimates"][k])) < 0.05 * fa["sd_log"][k]
        lo, hi = ff["ci95"][k]
        assert lo < ff["estimates"][k] < hi
    # Gauss-Newton sd at the estimate ~ the Cramer-Rao bound at the truth (linear regime)
    b = P53.bound(s, float(s["data"]["level"]))
    for k, c in zip(UNK, b["crlb_log"], strict=True):
        assert fa["sd_log"][k] == pytest.approx(c, rel=0.15)
    R = np.asarray(fa["correlation"])
    assert np.allclose(R, R.T) and np.allclose(np.diag(R), 1.0)


def test_no_source_case_resolves_only_the_biot_number():
    """q_vol = 0: only h/k enters, so the central-difference Jacobian is rank 2 at the estimate; k and h get an
    infinite sd, ln Bi a finite one, and the Biot number is still estimated."""
    s = _spec(body={"q_vol": 0.0})
    d = D.load(s)
    r = C.fit(s, d, UNK, _guess(s), "analytic")
    assert r["rank"] == 2 and r["correlation"] is None
    assert np.allclose(np.abs(r["null_directions"][0]), [0, 2**-0.5, 2**-0.5], atol=1e-4)
    assert np.isinf(r["sd_log"]["k"]) and np.isinf(r["sd_log"]["h"]) and np.isfinite(r["sd_log"]["alpha"])
    sd_bi = P53.bi_log_sd(r, UNK)
    assert np.isfinite(sd_bi) and sd_bi > 0
    bi, bi_true = r["estimates"]["h"] / r["estimates"]["k"], float(s["parameters"]["h"]) / float(s["parameters"]["k"])
    assert abs(np.log(bi / bi_true)) < 4 * sd_bi
    assert np.isinf(C.log_sd(r, UNK, {"k": 1.0}))
    assert P53.bound(s, 0.02)["rank"] == 2


def test_multistart_starts_and_summary():
    s = _spec()
    d = D.load(s)
    g = _guess(s)
    fits = C.multistart(s, d, UNK, g, 3, 4.0, 0, "analytic")
    assert len(fits) == 3 and fits[0]["start"] == g and all("path" not in f for f in fits)
    for f in fits[1:]:
        for k in UNK:
            assert g[k] / 4.0 <= f["start"][k] <= g[k] * 4.0
    m = P53.ms_summary(fits, UNK, len(d["T"]))
    assert 1 <= m["n_at_best"] <= 3 and m["best_chi2"] == min(m["chi2_per_reading"])
    assert set(m["spread_log"]) == set(UNK) and m["spread_log_Bi"] is not None


def test_bound_on_a_subset_of_unknowns():
    s = _spec(unknown=["alpha", "k"])
    b = P53.bound(s, 0.02)
    full = P53.bound(_spec(), 0.02)
    assert b["names"] == ["alpha", "k"] and len(b["crlb_log"]) == 2
    assert b["crlb_log"][0] <= full["crlb_log"][0] + 1e-12  # fixing h can only tighten the bound


def test_bi_log_sd_delta_method():
    C_ = np.array([[0.01, 0, 0], [0, 0.04, 0.03], [0, 0.03, 0.09]])
    fit = {"fisher": np.linalg.inv(C_).tolist(), "null_directions": []}
    assert P53.bi_log_sd(fit, UNK) == pytest.approx(np.sqrt(0.04 + 0.09 - 2 * 0.5 * 0.06))
    assert P53.bi_log_sd(fit, ["alpha", "k"]) is None


def test_mc_summary_coverage_and_ratio():
    s = _spec()
    b = P53.bound(s, 0.02)
    rng = np.random.default_rng(0)
    truth = {k: float(s["parameters"][k]) for k in UNK}
    fits = []
    for i in range(400):  # draws from the bound itself: std / CRLB ~ 1, coverage ~ 95 %
        x = rng.multivariate_normal(np.zeros(3), np.asarray(b["correlation"]) * np.outer(b["crlb_log"], b["crlb_log"]))
        est = {k: truth[k] * np.exp(x[j]) for j, k in enumerate(UNK)}
        sd = dict(zip(UNK, b["crlb_log"], strict=True))
        fits.append(
            {"estimates": est, "sd_log": sd, "success": True, "n_model_calls": 10, "chi2_per_reading": 1.0, "seed": i}
        )
    m = P53.mc_summary(fits, s, 0.02)
    for k in UNK:
        assert m["std_log"][k] / m["crlb_log"][k] == pytest.approx(1.0, abs=0.12)
        assert m["coverage95"][k] == pytest.approx(0.95, abs=0.04)
    assert m["failed"] == 0 and np.asarray(m["samples_log"]).shape == (400, 3)


@pytest.mark.parametrize(
    "over, msg",
    [
        ({"classical": {"model": "spectral"}}, "classical.model"),
        ({"classical": {"method": "nelder"}}, "classical.method"),
        ({"classical": {"multistart": {"n": 0}}}, "classical.multistart"),
        ({"classical": {"monte_carlo": {"batch": 0}}}, "classical.monte_carlo"),
        ({"studies": {"classical": {"cases": ["torus"]}}}, "studies.classical.cases"),
    ],
)
def test_validation(over, msg):
    with pytest.raises(ValueError, match=msg):
        IS.load_spec(overrides=over)


def test_multistart_figure():
    truth = {"alpha": 4e-6, "k": 15.0, "h": 100.0}
    ms = {
        "base": {"starts": [{"k": 30.0, "h": 50.0}], "estimates": [{"k": 14.0, "h": 98.0}], "n_at_best": 1},
        "no source": {"starts": [{"k": 5.0, "h": 400.0}], "estimates": [{"k": 30.0, "h": 200.0}], "n_at_best": 1},
    }
    fig = VF.multistart_kh(ms, truth)
    assert len(fig.axes) == 2
    plt.close(fig)
    rows = [{"name": "ls", "errors": {k: [1.0] for k in UNK}, "ci": {k: [-2.0, 4.0] for k in UNK}}]
    fig = VF.error_dots(rows, UNK, {k: 3.0 for k in UNK})
    plt.close(fig)


def test_identifiability_analyse_subset_consistency():
    """bound() equals ID.analyse on the corresponding Jacobian columns (no re-ordering mistakes)."""
    s = _spec(unknown=["k", "h"])
    b = P53.bound(s, 0.02)
    from aerosynapse.thermal.experiments.p5_1_identifiability import analyse_spec

    a = analyse_spec(s, [0.02])
    sig = IS.noise_sigma(s, a["_T"], 0.02)
    r = ID.analyse(a["_J"][:, [1, 2]], sig, ["k", "h"])
    assert np.allclose(b["crlb_log"], r["crlb_log"])


def test_analyse_handles_a_zero_sensitivity_column():
    """A parameter the readings do not depend on (alpha -> 0 plateau of a stalled start): rank deficient, infinite
    bound and collinearity index, no exception (the Step 5.3 full run crashed here before)."""
    rng = np.random.default_rng(0)
    J = np.column_stack([np.zeros(20), rng.standard_normal(20), rng.standard_normal(20)])
    r = ID.analyse(J, 1.0, UNK)
    assert r["rank"] == 2 and np.isinf(r["crlb_log"][0]) and np.isinf(r["collinearity_index"])
    assert ID.analyse(np.zeros((5, 2)), 1.0, ["k", "h"])["rank"] == 0


def test_multistart_records_stalled_starts():
    """Starts that stall on the alpha -> 0 plateau end with a much larger chi2 and are flagged, not dropped."""
    s = _spec()
    d = D.load(s)
    g = _guess(s)
    bad = {"alpha": 1e-13, "k": 28.0, "h": 2.0e4}
    fits = [C.safe_fit(s, d, UNK, g, "analytic"), C.safe_fit(s, d, UNK, bad, "fdm")]  # series root-finding fails there
    for f, st in zip(fits, (g, bad), strict=True):
        f["start"] = st
    m = P53.ms_summary(fits, UNK, len(d["T"]))
    assert m["at_best"] == [True, False] and m["n_failed"] == 0
    assert m["chi2_per_reading"][1] > 10 * m["chi2_per_reading"][0] and fits[1]["rank"] < 3
    assert C.safe_fit(s, d, UNK, bad, "analytic")["success"] is False  # recorded, not raised
