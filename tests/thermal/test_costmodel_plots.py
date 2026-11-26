"""Step 6.4b: cost model (power-law fit, prediction interval, memory probes) and the PCE variance decomposition."""

import numpy as np
import pytest

from aerosynapse.thermal.pinn import sensitivity as SA
from aerosynapse.thermal.utils import costmodel as CM
from aerosynapse.thermal.viz import sensitivity_plots as F


def test_fit_and_predict_power_law():
    n = np.array([1e3, 2e3, 4e3, 8e3, 1.6e4])
    y = 3e-4 * n**1.5 * np.exp(np.random.default_rng(0).normal(0, 0.02, 5))
    f = CM.fit_power(n, y)
    assert f["b"] == pytest.approx(1.5, abs=0.05)
    p = CM.predict(f, 1e6)
    assert p["lo"] < 3e-4 * 1e9 < p["hi"]
    assert p["hi"] / p["value"] > CM.predict(f, 2e4)["hi"] / CM.predict(f, 2e4)["value"]  # wider when extrapolating


def test_human_time_and_memory_probes():
    assert CM.human_time(30) == "30 s" and CM.human_time(7200) == "2 h" and CM.human_time(3 * 86400) == "3 days"
    assert CM.peak_memory_bytes() > 0 and CM.current_memory_bytes() > 0
    m = CM.machine()
    assert m["cpus"] >= 1
    with CM.MemorySampler() as ms:
        x = np.ones(30_000_000)  # 240 MB
        x[::1000] = 2.0
        del x
    assert ms.extra >= 1


def test_decomposition_of_known_pce():
    u = np.random.default_rng(1).random((4000, 3))
    x = 2 * u - 1
    f = x[:, 0] + 0.5 * x[:, 1] * x[:, 2]  # first order in x0, one interaction x1 x2
    m = SA.pce_fit(u, f, 2)
    d = F.decompose(m)
    v0, v12 = 1 / 3, 0.25 / 9  # Var(x0) and Var(0.5 x1 x2) for U(-1, 1)
    assert d["S"][0] == pytest.approx(v0 / (v0 + v12), abs=1e-8)
    assert d["S_ij"]["Pr×Bi"] == pytest.approx(v12 / (v0 + v12), abs=1e-8)
    assert d["first"] + d["pairs"] + d["triple"] == pytest.approx(1.0, abs=1e-10)
