"""Phase 4: datasets, operators, hybrids, UQ."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from aerosynapse.ml.datasets import check_leakage, convdiff_operator_dataset, poisson2d_operator_dataset
from aerosynapse.ml.deeponet import DeepONet
from aerosynapse.ml.fno import FNO1d, FNO2d
from aerosynapse.ml.hybrid import cg_iterations, coarse_solver_features, kalman_convdiff, periodic_upsample
from aerosynapse.ml.rom import CNN1d, FlatMLP, PODRegressor
from aerosynapse.ml.training import per_sample_rel_l2, predict_operator, train_operator
from aerosynapse.uq.calibration import binomial_ci, calibration_report
from aerosynapse.uq.cards import UQ_CARDS
from aerosynapse.uq.ensembles import Ensemble
from aerosynapse.uq.gp import GaussianProcess
from aerosynapse.uq.mc import lhs_uniform, propagate
from aerosynapse.uq.sobol import ishigami, ishigami_analytic, sobol_indices

pytestmark = pytest.mark.unit


def test_dataset_and_leakage():
    d = convdiff_operator_dataset(6, n=32, seed=0)
    assert d.inputs.shape == (6, 2, 32) and d.outputs.shape == (6, 1, 32)
    # stored FV-vs-exact-spectral error must converge with resolution (same random fields via same seed)
    d128 = convdiff_operator_dataset(6, n=128, seed=0)
    assert np.all(d128.numerical_error < 0.25 * d.numerical_error) and d128.numerical_error.max() < 2e-3
    assert check_leakage(d.subset(np.arange(3)), d.subset(np.arange(3, 6))) == 0
    assert check_leakage(d, d) == 6
    assert set(d.card) >= {
        "source",
        "licence",
        "physical_system",
        "format",
        "units",
        "split",
        "leakage_risks",
        "limitations",
    }
    p = poisson2d_operator_dataset(2, n=16)
    assert p.inputs.shape == (2, 1, 17, 17) and np.allclose(p.outputs[:, :, 0, :], 0)


def test_fno_shapes_and_resolution_invariance():
    torch.manual_seed(0)
    m = FNO1d(2, 1, width=8, modes=4, layers=2, conserve_mass=True)
    for n in (32, 64):
        a = torch.randn(3, 2, n)
        out = m(a)
        assert out.shape == (3, 1, n)
        assert torch.allclose(out.mean(-1), a[:, :1].mean(-1), atol=1e-5)  # exact mass conservation
    m2 = FNO2d(1, 1, width=8, modes=3, layers=2)
    assert m2(torch.randn(2, 1, 17, 17)).shape == (2, 1, 17, 17)


def test_deeponet_cnn_mlp_pod():
    d = DeepONet(16, 1, p=8, width=16, depth=2)
    assert d(torch.randn(4, 17), torch.rand(10, 1)).shape == (4, 10)
    assert CNN1d(2, 8, 2)(torch.randn(2, 2, 32)).shape == (2, 1, 32)
    f = FlatMLP(2, 32, 16, 1)
    assert f(torch.randn(2, 2, 32)).shape == (2, 1, 32)
    with pytest.raises(ValueError):
        f(torch.randn(2, 2, 64))
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 16))
    M = rng.normal(size=(16, 16))
    pod = PODRegressor(rank=16, poly_degree=1).fit(A, np.zeros((50, 0)), A @ M)
    assert np.abs(pod.predict(A, np.zeros((50, 0))) - A @ M).max() < 1e-6  # exact for linear operator


def test_operator_training_reduces_error():
    d = convdiff_operator_dataset(32, n=32, seed=1)
    m = FNO1d(2, 1, width=8, modes=6, layers=2)
    X, Y = d.inputs, d.outputs
    e0 = per_sample_rel_l2(predict_operator(m, X), Y).mean()
    m, hist, wall = train_operator(m, X, Y, epochs=20, batch=8)
    assert per_sample_rel_l2(predict_operator(m, X), Y).mean() < e0 and hist[-1] < hist[0] and wall > 0


def test_hybrid_components():
    u = np.sin(2 * np.pi * (np.arange(8) + 0.5) / 8)
    up = periodic_upsample(u, 4)
    assert up.size == 32 and np.abs(up).max() <= 1.0
    x = (np.arange(64) + 0.5) / 64
    f = coarse_solver_features(1 + 0.3 * np.sin(2 * np.pi * x), 20.0, 0.2)
    assert f.shape == (64,) and abs(f.mean() - 1.0) < 1e-3
    A = sp.diags([-1, 2.0, -1], [-1, 0, 1], shape=(50, 50)).tocsr()
    b = np.ones(50)
    xs, it0 = cg_iterations(A, b, None)
    _, it1 = cg_iterations(A, b, xs + 1e-6)
    assert it1 < it0
    r = kalman_convdiff(steps=40, n=32)
    assert r.final_rmse_kf < 0.5 * r.final_rmse_free and 0 <= r.coverage_2sigma <= 1


def test_gp_calibration_on_synthetic_data():
    """Mean 95% coverage over independent replicates is near nominal (ML-II hyper-parameters vary per replicate)."""
    f = lambda x: np.sin(6 * x[:, 0])
    covs, noise = [], []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 1, (150, 1))
        gp = GaussianProcess(seed=0).fit(X, f(X) + rng.normal(0, 0.1, 150))
        Xt = rng.uniform(0, 1, (400, 1))
        mu, sd = gp.predict(Xt, include_noise=True)
        covs.append(calibration_report(f(Xt) + rng.normal(0, 0.1, 400), mu, sd)["coverage_95"])
        noise.append(gp.noise_std)
    lo, hi = binomial_ci(0.95, 5 * 400)
    assert lo - 0.01 <= np.mean(covs) <= hi + 0.01
    assert 0.08 < np.mean(noise) < 0.12  # recovers the true noise level 0.1


def test_sobol_matches_analytic_ishigami():
    est = sobol_indices(ishigami, -np.pi * np.ones(3), np.pi * np.ones(3), n=2**14)
    ana = ishigami_analytic()
    assert np.allclose(est["S1"], ana["S1"], atol=0.03)
    assert np.allclose(est["ST"], ana["ST"], atol=0.03)


def test_monte_carlo_and_ensemble():
    out = propagate(lambda x: float(x[0] ** 2), lambda rng, n: rng.uniform(0, 1, (n, 1)), 4000)
    assert abs(out["mean"] - 1 / 3) < 3 * out["se_mean"] + 1e-12
    assert lhs_uniform(np.zeros(2), np.ones(2), 10).shape == (10, 2)
    ens = Ensemble.train(lambda s: s, lambda m, X: X * m, m=3)
    mu, sd = ens.predict(np.ones(2))
    assert np.allclose(mu, 1.0) and np.allclose(sd, 1.0)
    assert {"estimates", "assumptions", "calibration", "evaluation", "decision_impact"} <= set(
        UQ_CARDS["gaussian_process"]
    )
