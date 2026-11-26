"""Step 5.4: robustness study helpers (experiments/p5_4_robustness.py) and its figures."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aerosynapse.thermal.experiments import p5_3_classical as P53
from aerosynapse.thermal.experiments import p5_4_robustness as P54
from aerosynapse.thermal.inverse import spec as IS
from aerosynapse.thermal.viz import inverse_figs as VF

SMALL = {"measurement": {"sensors": {"n": 3}, "times": {"n": 12}}}
UNK = ["alpha", "k", "h"]


def _spec(**over):
    return IS.load_spec(overrides=IS.deep_merge(SMALL, over))


def test_zero_noise_data_keep_clean_values_and_get_weights():
    s = _spec()
    d = P54.data_of(s, 0.0, 0, 0.01)
    assert np.array_equal(d["T"], d["T_clean"]) and d["noise_sigma_mean_K"] == 0.0
    assert np.allclose(d["sigma"], IS.noise_sigma(s, d["T_clean"], 0.01)) and np.all(d["sigma"] > 0)


def test_zero_noise_single_fits():
    """Analytic fit of analytic data is exact (inverse crime); the FDM estimator shows its model-error floor."""
    r = P54.setting_task(_spec(), 0.0, 0.01)
    est = r["analytic"]["estimates"]
    truth = {"alpha": 4e-6, "k": 15.0, "h": 100.0}
    assert all(est[k] == pytest.approx(truth[k], rel=1e-6) for k in UNK)
    assert r["fdm"]["estimates"] is not None and 0 < r["fdm_vs_analytic_dlog"] < 0.02


def test_mc_stats_counts_stalled_and_failed_fits():
    s = _spec()
    b = P53.bound(s, 0.02)
    crlb = dict(zip(UNK, b["crlb_log"], strict=True))
    cov = np.asarray(b["correlation"]) * np.outer(b["crlb_log"], b["crlb_log"])
    rng = np.random.default_rng(1)
    truth = {k: float(s["parameters"][k]) for k in UNK}
    fits = []
    for i in range(400):
        x = rng.multivariate_normal(np.zeros(3), cov)
        fits.append(
            {
                "estimates": {k: truth[k] * np.exp(x[j]) for j, k in enumerate(UNK)},
                "sd_log": crlb,
                "chi2_per_reading": 1.0,
                "seed": i,
            }
        )
    fits.append(
        {"estimates": {k: truth[k] * 1e-6 for k in UNK}, "sd_log": crlb, "chi2_per_reading": 600.0, "seed": 998}
    )
    fits.append({"estimates": None, "sd_log": None, "chi2_per_reading": float("inf"), "seed": 999})
    m = P54.mc_stats(fits, s, crlb, 36, 10.0)
    assert m["stalled"] == 1 and m["failed"] == 1 and m["used"] == 400
    assert m["stall_threshold"] == pytest.approx(1 + 10 * np.sqrt(2 / 36))
    for k in UNK:
        assert m["std_over_crlb"][k] == pytest.approx(1.0, abs=0.12)
        assert m["coverage95"][k] == pytest.approx(0.95, abs=0.04)
        assert abs(m["bias_over_crlb"][k]) < 0.2


def test_build_settings_layouts_and_duplicates():
    cli = {"studies": {"robustness": {"noise": {"designs": ["base"], "levels": [0.0, 0.02]}, "sensors": {"n": [1, 8]}}}}
    rb = IS.load_spec(overrides=cli)["studies"]["robustness"]
    st = P54.build_settings([], cli, rb)
    assert [x["study"] for x in st].count("placement") == len(rb["placement"]["layouts"])
    by = {(x["study"], x["label"]): x for x in st}
    assert by[("placement", "centre_only")]["positions_frac"] == [0.0] * 8
    assert by[("placement", "surface_only")]["positions_frac"] == [1.0] * 8
    assert by[("sensors", "1")]["positions_frac"] == [0.5]
    # the base problem at 2 % appears three times (noise, 8 sensors, uniform placement): one computation
    assert by[("noise", "2 %")]["key"] == by[("sensors", "8")]["key"] == by[("placement", "uniform")]["key"]
    assert by[("noise", "0 %")]["key"] != by[("noise", "2 %")]["key"]


@pytest.mark.parametrize(
    "over, msg",
    [
        ({"mc_model": "spectral"}, "mc_model"),
        ({"noise": {"levels": [-0.01]}}, "noise levels"),
        ({"noise": {"designs": ["no_source"]}}, "noise.designs"),
        ({"sensors": {"n": [0]}}, "sensors.n"),
        ({"placement": {"layouts": {"bad": {"positions": [1.5]}}}}, "positions"),
        ({"pinn": {"variant": "nope"}}, "robustness.pinn.variant"),
    ],
)
def test_validation(over, msg):
    with pytest.raises(ValueError, match=msg):
        IS.load_spec(overrides={"studies": {"robustness": over}})


def test_figures():
    lv = [1.0, 2.0, 5.0, 10.0]
    des = {
        "base": {
            "levels": lv,
            "crlb": {k: [3 * x for x in lv] for k in UNK},
            "std": {k: [3.1 * x for x in lv] for k in UNK},
            "floor": {k: 0.01 for k in UNK},
        },
        "optimised design": {
            "levels": lv,
            "crlb": {k: [1.5 * x for x in lv] for k in UNK},
            "std": {k: [1.6 * x for x in lv] for k in UNK},
            "floor": None,
        },
    }
    pinn = {"levels": [0.0, 2.0, 10.0], "err": {k: [50.0, 60.0, 80.0] for k in UNK}}
    figs = [VF.noise_robustness(des, pinn, UNK)]
    rows = [
        {"label": str(n), "crlb": {k: 10 / n for k in UNK}, "std": {k: 11 / n for k in UNK}, "rank": 3}
        for n in (1, 2, 4)
    ]
    plc = [
        {"label": "uniform", "crlb": {k: 5.0 for k in UNK}, "std": {k: 5.2 for k in UNK}, "rank": 3},
        {"label": "centre only", "crlb": {k: 50.0 for k in UNK}, "std": None, "rank": 3},
    ]
    figs.append(VF.sensor_studies(rows, plc, UNK))
    sec = [
        {
            "title": "noise",
            "labels": ["1 %", "2 %"],
            "ratio": {k: [1.0, 1.05] for k in UNK},
            "coverage": {k: [95.0, 94.0] for k in UNK},
            "ratio_band": 0.05,
            "coverage_band": 3.0,
        }
    ]
    figs.append(VF.calibration(sec, UNK))
    for f in figs:
        assert isinstance(f, plt.Figure)
        plt.close(f)
