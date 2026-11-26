import json
import logging

import numpy as np
import pytest

from aerosynapse.experiments import stats as ST
from aerosynapse.experiments.parallel import run_parallel
from aerosynapse.logging_utils import get_logger, log_event

pytestmark = pytest.mark.unit


def test_stats_functions():
    est, lo, hi = ST.bootstrap_ci(np.arange(1, 11), "mean")
    assert est == 5.5 and lo < 5.5 < hi
    assert ST.paired_wilcoxon([1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]) == pytest.approx(0.0078125)
    assert ST.paired_wilcoxon([1, 2], [1, 2]) == 1.0
    h = ST.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert h["a"]["p_holm"] == pytest.approx(0.03) and h["b"]["p_holm"] == pytest.approx(0.08) and not h["b"]["reject"]
    assert h["c"]["p_holm"] == pytest.approx(0.5)
    assert ST.friedman({"x": [1, 2, 3, 4], "y": [2, 3, 4, 5], "z": [3, 4, 5, 6]}) < 0.05
    assert ST.effect_size_log_ratio([1, 10], [10, 100]) == pytest.approx(-1.0)


def _sq(x):
    return x * x


def test_parallel_and_logging(caplog):
    assert run_parallel(_sq, [1, 2, 3], workers=1) == [1, 4, 9]
    assert run_parallel(_sq, [1, 2, 3], workers=2) == [1, 4, 9]
    lg = get_logger("t")
    with caplog.at_level(logging.INFO, logger="aerosynapse.t"):
        log_event(lg, "evt", a=1)
    assert json.loads(caplog.records[0].getMessage()) == {"a": 1, "event": "evt"}
