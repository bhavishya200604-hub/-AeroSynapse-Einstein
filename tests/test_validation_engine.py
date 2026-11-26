"""Phase 7: metrics, trust/fallback, resources, checkpoint resume, notebooks, dataset catalogue, safety."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aerosynapse.experiments.notebooks import write_notebooks
from aerosynapse.ml.nets import MLP
from aerosynapse.ml.pinn import train_pinn_checkpointed
from aerosynapse.ml.problems import Heat1D
from aerosynapse.safety import TIER_DESCRIPTIONS, ValidationTier, banner
from aerosynapse.validation import metrics as M
from aerosynapse.validation.resources import array_memory_mb, calibrate, estimate_fv_seconds, estimate_pinn_seconds
from aerosynapse.validation.trust import TrustPolicy, guarded_predict, trust_report_markdown

pytestmark = pytest.mark.unit


def test_metrics_hand_calculations():
    p, r = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])
    assert M.linf(p, r) == 2.0 and M.mae(p, r) == pytest.approx(2 / 3)
    assert M.rmse(p, r) == pytest.approx(np.sqrt(4 / 3)) and M.rel_l2(p, r) == pytest.approx(2 / np.sqrt(30))
    assert M.abs_l2(p, r, 0.5) == pytest.approx(np.sqrt(2.0))
    d = M.error_distribution(p, r)
    assert d["q100"] == 2.0 and d["q50"] == 0.0
    assert M.conservation_error([2.0, 2.0, 2.2]) == pytest.approx(0.1)
    assert M.energy_drift([1.0, 1.1, 0.9]) == pytest.approx(0.2) and M.bc_error(p, r) == 2.0
    assert M.speedup(10.0, 2.0) == 5.0 and M.break_even_queries(10, 10, 1.0, 0.0) == 20.0
    assert M.break_even_queries(1, 1, 1.0, 2.0) == float("inf")
    assert M.measure(lambda: sum(range(100)))["time_s_median"] >= 0
    assert set(M.field_metrics(p, r)) >= {"rel_l2", "linf", "mae", "rmse"}


def test_trust_policy_and_fallback(tmp_path):
    rng = np.random.default_rng(0)
    params = rng.uniform(0, 1, (200, 1))
    feats = rng.normal(size=(200, 2))
    pol = TrustPolicy.calibrate(
        params, feats, rng.uniform(0, 0.01, 200), rng.uniform(0, 0.01, 200), log_path=tmp_path / "log.jsonl"
    )
    ok = pol.decide(np.array([0.5]), np.zeros(2), 0.001, 0.001, "id")
    assert ok.use_surrogate and not ok.reasons
    bad = pol.decide(np.array([5.0]), np.array([10.0, 10.0]), 1.0, 1.0, "ood")
    assert not bad.use_surrogate and set(bad.reasons) == {"domain", "input_novelty", "uncertainty", "physics_residual"}
    pred, src, _ = guarded_predict(
        pol, lambda: (np.zeros(3), 1.0, 0.0), lambda: np.ones(3), np.array([0.5]), np.zeros(2)
    )
    assert src == "classical_solver" and pred.sum() == 3
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 3
    md = trust_report_markdown("x", {"Accuracy": {"rel_l2": 0.01}}, "1/5")
    assert "RESEARCH USE ONLY" in md and "| rel_l2 | 0.01 |" in md


def test_resource_estimator_and_memory():
    cal = calibrate(quick=True)
    assert cal.c_pinn > 0 and cal.c_fv > 0 and "PINN" in cal.basis
    assert estimate_pinn_seconds(cal, 10, 10, 10) == pytest.approx(10 * (cal.c0_pinn + cal.c_pinn * 100))
    assert estimate_fv_seconds(cal, 10, 10) == pytest.approx(cal.c_fv * 100)
    assert array_memory_mb(1024, 128) == pytest.approx(1.0)


def test_checkpoint_resume_is_bit_identical(tmp_path):
    p = Heat1D()

    def model():
        torch.manual_seed(0)
        return MLP(2, 1, 8, 2, lo=p.lo, hi=p.hi)

    m_full = model()
    train_pinn_checkpointed(p, m_full, 40, 64, str(tmp_path / "a.pt"), 10)
    m_int = model()
    assert train_pinn_checkpointed(p, m_int, 40, 64, str(tmp_path / "b.pt"), 10, stop_at=20) == 20
    m_res = model()
    train_pinn_checkpointed(p, m_res, 40, 64, str(tmp_path / "b.pt"), 10, resume=True)
    X = torch.rand(16, 2)
    assert torch.equal(m_full(X), m_res(X))


def test_notebooks_catalogue_and_safety(tmp_path):
    nbs = write_notebooks(tmp_path, (3,))
    nb = json.loads(nbs[0].read_text())
    assert nb["nbformat"] == 4 and "not executed" in nb["cells"][0]["source"]
    cat = json.loads((Path(__file__).parents[1] / "src/aerosynapse/data/datasets.json").read_text())
    for name, e in cat.items():
        if name.startswith("_"):
            continue
        for k in (
            "source",
            "licence",
            "physical_system",
            "format",
            "units",
            "split",
            "leakage_risks",
            "limitations",
            "access",
            "verification_status",
        ):
            assert k in e, (name, k)
    assert len(TIER_DESCRIPTIONS) == 5 and ValidationTier.SAFETY_CERTIFIED == 5
    assert "not certified" in banner()
