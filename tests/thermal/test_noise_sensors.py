"""Step 2.2: noise models and sensor layouts."""

import numpy as np
import pytest

from aerosynapse.thermal.data import noise as N
from aerosynapse.thermal.data import sensors as S


def test_noise_zero_level_identity_and_determinism():
    x = np.linspace(-1, 2, 50)
    assert np.array_equal(N.add_noise(x, 0.0, "gaussian", 1), x)
    a, b, c = (N.add_noise(x, 0.05, "gaussian", s) for s in (3, 3, 4))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


@pytest.mark.parametrize("model", N.MODELS)
def test_noise_statistics_match_definition(model):
    rng = np.random.default_rng(0)
    x = rng.uniform(0.2, 1.0, 200_000)
    lvl = 0.05
    eps = N.add_noise(x, lvl, model, 7) - x
    expect = N.noise_scale(x, lvl, model)
    z = eps / expect
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01
    if model == "gaussian":
        assert expect[0] == pytest.approx(lvl * np.sqrt(np.mean(x**2)))
    if model == "gaussian_range":
        assert expect[0] == pytest.approx(lvl * np.ptp(x))
    if model == "relative":
        assert np.allclose(expect, lvl * x)


def test_noise_rejects_bad_input():
    with pytest.raises(ValueError):
        N.add_noise([1.0], -0.1)
    with pytest.raises(ValueError):
        N.noise_scale([1.0], 0.1, "bogus")


@pytest.mark.parametrize("kind", S.LAYOUTS_1D)
def test_layout_1d(kind):
    x = S.layout_1d(kind, 16, seed=2)
    assert x.shape == (16,) and np.all(np.diff(x) >= 0) and x.min() > 0 and x.max() < 1
    assert np.array_equal(x, S.layout_1d(kind, 16, seed=2))
    assert (x > 0.75).sum() > (x < 0.25).sum() if kind == "clustered_right" else True


@pytest.mark.parametrize("kind", S.LAYOUTS_2D)
def test_layout_2d(kind):
    p = S.layout_2d(kind, 36, seed=1)
    assert p.shape[1] == 2 and 25 <= len(p) <= 36
    assert p.min() >= 0.02 - 1e-12 and p.max() <= 0.98 + 1e-12
    assert np.array_equal(p, S.layout_2d(kind, 36, seed=1))


def test_sampling_exact_for_linear_fields():
    xg = np.linspace(0, 1, 33)
    assert np.allclose(S.sample_1d(xg, 3 * xg + 1, np.array([0.1, 0.77])), [1.3, 3.31])
    xc = (np.arange(16) + 0.5) / 16
    X, Y = np.meshgrid(xc, xc)
    f = np.stack([2 * X - Y + 0.3, X * 0 + 5])
    pts = np.array([[0.3, 0.4], [0.9, 0.2]])
    got = S.sample_2d(xc, xc, f, pts)
    assert got.shape == (2, 2) and np.allclose(got[0], 2 * pts[:, 0] - pts[:, 1] + 0.3) and np.allclose(got[1], 5)
