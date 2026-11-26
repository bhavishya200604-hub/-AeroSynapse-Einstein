"""Step 6.2: calibration metrics, conformal multiplier, MC-dropout masks, validation, analysis, report, figures."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from aerosynapse.thermal.experiments import p6_2_uq as P62
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn import uq as UQ
from aerosynapse.thermal.pinn.networks import MLP, MCDropout, clear_dropout_masks, sample_dropout_masks
from aerosynapse.thermal.viz import uq_figs as F


def test_calibrated_gaussian_is_calibrated():
    rng = np.random.default_rng(0)
    mu, sd = rng.normal(size=200_000), rng.uniform(0.5, 2.0, 200_000)
    y = mu + sd * rng.normal(size=mu.size)
    c = UQ.coverage(mu, sd, y, [0.5, 0.9, 0.95])
    assert all(abs(c[k] - k) < 0.005 for k in c)
    assert UQ.ece(mu, sd, y) < 0.005
    assert UQ.dispersion(mu, sd, y) == pytest.approx(np.sqrt(np.mean(sd**2 * 1)) / np.sqrt(np.mean(sd**2)), abs=0.01)
    assert UQ.rank_corr(mu, sd, y) > 0.2  # larger sd -> larger errors


def test_overconfident_is_detected():
    rng = np.random.default_rng(1)
    mu = np.zeros(50_000)
    y = rng.normal(size=mu.size)
    s = UQ.summary(mu, 0.1 * np.ones_like(mu), y, [0.95])
    assert s["dispersion"] == pytest.approx(10.0, rel=0.02)
    assert s["coverage"]["0.95"] < 0.2


def test_crps_gaussian_closed_form():
    assert UQ.crps_gaussian(0.0, 2.0, 0.0) == pytest.approx(2.0 * (2 * 0.3989422804 - 1 / np.sqrt(np.pi)), rel=1e-9)
    rng = np.random.default_rng(2)  # Monte-Carlo check: CRPS = E|X - y| - 0.5 E|X - X'|
    x, x2 = rng.normal(1.0, 0.5, 400_000), rng.normal(1.0, 0.5, 400_000)
    mc = np.mean(np.abs(x - 2.0)) - 0.5 * np.mean(np.abs(x - x2))
    assert UQ.crps_gaussian(1.0, 0.5, 2.0) == pytest.approx(mc, rel=0.01)


def test_conformal_rank():
    s = np.arange(1.0, 20.0)  # 19 scores 1..19 (mu 0, sd 1, |y| = score)
    assert UQ.conformal_scale(np.zeros(19), np.ones(19), s, 0.1) == 18.0  # ceil(20 * 0.9) = 18th smallest
    assert UQ.conformal_scale(np.zeros(3), np.ones(3), np.ones(3), 0.05) == np.inf


def test_mc_dropout_masks_and_default_unchanged():
    torch.manual_seed(0)
    plain = MLP([0, 0], [1, 1], 8, 3)
    assert not any(isinstance(m, MCDropout) for m in plain.modules())
    torch.manual_seed(0)
    drop = MLP([0, 0], [1, 1], 8, 3, dropout=0.3).double().eval()
    assert sum(isinstance(m, MCDropout) for m in drop.modules()) == 3
    assert drop.n_params() == plain.n_params()
    X = torch.rand(6, 2, dtype=torch.float64)
    det = drop(X)
    g = torch.Generator().manual_seed(1)
    sample_dropout_masks(drop, g)
    a = drop(X)
    assert torch.allclose(a, drop(X)) and not torch.allclose(a, det)
    m = next(x for x in drop.modules() if isinstance(x, MCDropout)).mask
    assert m.shape == (8,) and set(np.unique(m.numpy()).round(6)) <= {0.0, round(1 / 0.7, 6)}
    clear_dropout_masks(drop)
    assert torch.allclose(drop(X), det)


@pytest.mark.parametrize(
    "bad",
    [
        {"network": {"dropout": 1.0}},
        {"network": {"dropout": 0.1, "arch": "fourier"}},
        {"uq_study": {"problems": {"B_slab": {"range": [3.0, 1.0]}}}},
        {"uq_study": {"problems": {"nope": {"range": [1.0, 2.0]}}}},
        {"uq_study": {"ensemble": {"members": 3, "sizes": [2, 5]}}},
        {"uq_study": {"mc_dropout": {"rates": [1.5]}}},
        {"uq_study": {"conformal_alpha": [0.0]}},
    ],
)
def test_validation(bad):
    with pytest.raises(ValueError, match="dropout|uq_study"):
        SP.load_spec(None, "smoke", bad)


def test_values_sets():
    st = SP.load_spec(None, "smoke")["uq_study"]
    allv, sets = P62.values([0.3, 3.0], st)
    g = [allv[i] for i in sets["grid"]]
    assert len(g) == 13 and g[0] == pytest.approx(0.3) and g[-1] == pytest.approx(3.0)
    assert len(sets["calibration"]) == 48 and len(sets["test"]) == 48
    rnd = [allv[i] for i in sets["calibration"] + sets["test"]]
    assert all(0.3 <= v <= 3.0 for v in rnd)
    assert P62.values([0.3, 3.0], st) == (allv, sets)  # seeded


def _fake_analysis():
    rng = np.random.default_rng(3)
    st = SP.load_spec(None, "smoke")["uq_study"]
    st = dict(st, random_values={"calibration": 40, "test": 40, "seed": 0})
    allv, sets = P62.values([1.0, 10.0], st)
    x = np.linspace(0, 1, 30)
    truth = np.stack([np.sin(3 * x) * v for v in allv])
    ens = {f"ens_s{i}": truth + 0.02 * rng.normal(size=truth.shape) for i in range(8)}
    mcd = {"mcd_0.05_s0": (truth + 0.05 * rng.normal(size=truth.shape), 0.05 * np.ones_like(truth))}
    return st, P62.analyse(st, allv, sets, truth, ens, mcd) | {
        "grid_values": [allv[i] for i in sets["grid"]],
        "n_calibration": 40,
        "n_test": 40,
    }


def test_analyse_report_figures(tmp_path):
    st, an = _fake_analysis()
    assert set(an["methods"]) == {"ensemble_M2", "ensemble_M3", "ensemble_M5", "ensemble_M8", "mcd_0.05_s0"}
    r = an["methods"]["mcd_0.05_s0"]
    assert 0.9 < r["coverage"]["0.95"] < 1.0 and len(r["per_value"]) == 13
    c = r["conformal"]["0.05"]
    assert abs(c["test_coverage"] - 0.95) < 0.02 and c["field_test_coverage"] >= 0.8
    met = {"analysis": {"B_slab": an}, "members": []}
    P62.report(met, tmp_path / "u.md", "test")
    assert "ensemble_M8" in (tmp_path / "u.md").read_text(encoding="utf-8")
    F.style()
    for fn in (F.reliability, F.spread_vs_error, F.bands):
        plt.close(fn(met["analysis"]))
