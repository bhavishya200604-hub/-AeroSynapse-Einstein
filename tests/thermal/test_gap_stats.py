"""Pre-registered gap-run statistics (src/utils/stats.py)."""

import numpy as np
import pytest
from scipy import stats

from aerosynapse.thermal.utils import stats as S


def test_holm_known_values():
    # classic example: p = 0.01, 0.04, 0.03, 0.005 -> sorted 0.005*4, 0.01*3, 0.03*2, 0.04*1 with monotonicity
    adj = S.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_ignores_nan_and_caps():
    adj = S.holm([0.5, float("nan"), 0.9])
    assert np.isnan(adj[1]) and adj[0] == pytest.approx(1.0) and adj[2] == pytest.approx(1.0)


def test_paired_log_ratio_matches_scipy():
    ref = np.array([0.1, 0.2, 0.15, 0.3, 0.25])
    test = ref * np.array([2.1, 1.9, 2.5, 2.0, 2.2])
    r = S.paired_log_ratio(test, ref)
    d = np.log(test / ref)
    assert r["gm_ratio"] == pytest.approx(np.exp(d.mean()))
    assert r["p_t"] == pytest.approx(stats.ttest_1samp(d, 0).pvalue)
    assert r["p_wilcoxon"] == pytest.approx(0.0625)  # n = 5, all same sign: the exact minimum
    lo, hi = r["gm_ratio_ci95"]
    assert lo <= r["gm_ratio"] <= hi


def test_welch_less_direction_and_ci():
    a = np.array([-0.99, -0.95, -1.0, -0.97, -0.9])
    b = np.array([0.1, -0.2, 0.3, 0.05, 0.2])
    r = S.welch_less(a, b)
    assert r["p"] < 0.01 and r["diff"] < 0
    assert r["diff_ci95"][0] <= r["diff"] <= r["diff_ci95"][1]
    assert S.welch_less(b, a)["p"] > 0.99


def test_bootstrap_is_seeded():
    x = np.arange(10.0)
    assert S.bootstrap_mean_ci(x) == S.bootstrap_mean_ci(x)
