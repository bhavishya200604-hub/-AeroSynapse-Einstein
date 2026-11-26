"""Phase 2 ML verification: autograd vs FD, exact residual ≈ 0, determinism, serialisation, samplers."""

import math

import numpy as np
import pytest
import torch

from aerosynapse.exceptions import SolverDivergenceError
from aerosynapse.ml.autodiff import grad, second
from aerosynapse.ml.baselines import train_supervised
from aerosynapse.ml.bc_strategies import train_dd_boundary_layer
from aerosynapse.ml.io import load_checkpoint, load_model, save_checkpoint, save_model
from aerosynapse.ml.nets import MLP, FourierFeatures, n_params, set_seed
from aerosynapse.ml.pinn import PINNConfig, predict, residual_numpy, train_pinn
from aerosynapse.ml.problems import Burgers1D, ConvDiff1D, DiskPoisson, Heat1D, Poisson2D, SteadyBoundaryLayer
from aerosynapse.ml.sampling import SAMPLERS, make_sampler
from aerosynapse.ml.weighting import AnnealingWeights, gradient_imbalance

pytestmark = pytest.mark.unit


@pytest.fixture
def f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


class Exact(torch.nn.Module):
    """Wraps a closed-form solution as a 'model' to test residual implementations."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, X):
        return self.fn(X)


def test_autograd_matches_finite_differences(f64):
    set_seed(0)
    m = MLP(2, 1, 16, 2, fourier_features=4, fourier_scale=1.0, periodic_dims=(0,), lo=[0, 0], hi=[1, 1])
    X = torch.rand(5, 2, requires_grad=True)
    u = m(X)
    g = grad(u, X)
    uxx = second(u, X, 0)
    h = 1e-6  # central-difference truncation ~h^2 |u'''| stays below 1e-6 relative
    for i in range(2):
        e = torch.zeros(2)
        e[i] = h
        fd = (m(X + e) - m(X - e)) / (2 * h)
        assert torch.allclose(g[:, i : i + 1], fd, rtol=1e-6, atol=1e-8)
    e0 = torch.tensor([1e-4, 0.0])
    fd2 = (m(X + e0) - 2 * m(X) + m(X - e0)) / 1e-8
    assert torch.allclose(uxx, fd2, rtol=1e-4, atol=1e-5)
    # parameter gradients (torch.autograd.gradcheck on the loss)
    Xc = torch.rand(4, 2)
    assert torch.autograd.gradcheck(
        lambda w: torch.sum(torch.tanh(Xc @ w) ** 2), (torch.rand(2, 3, requires_grad=True),)
    )


def test_exact_solutions_have_zero_residual(f64):
    X = torch.rand(64, 2, requires_grad=True)
    heat = Heat1D()
    m = Exact(lambda X: torch.exp(-heat.alpha * math.pi**2 * X[:, 1:2]) * torch.sin(math.pi * X[:, :1]))
    assert heat.residual(m, X).abs().max() < 1e-10
    P = Poisson2D()
    m2 = Exact(lambda X: torch.sin(math.pi * X[:, :1]) * torch.sin(2 * math.pi * X[:, 1:2]))
    assert P.residual(m2, X).abs().max() < 1e-9
    cd = ConvDiff1D(pe=20.0)

    def cd_exact(X):
        x, t = X[:, :1], X[:, 1:2]
        s2 = cd.sigma0**2 + 2 * cd.D * t
        return sum(
            torch.sqrt(cd.sigma0**2 / s2) * torch.exp(-((x - 0.25 - t - k) ** 2) / (2 * s2)) for k in range(-3, 4)
        )

    Xc = torch.rand(64, 2) * torch.tensor([1.0, 0.5])
    Xc.requires_grad_(True)
    assert cd.residual(Exact(cd_exact), Xc).abs().max() < 1e-9
    bl = SteadyBoundaryLayer(pe=10.0)
    m4 = Exact(lambda X: (torch.exp(10 * (X - 1)) - math.exp(-10)) / (1 - math.exp(-10)))
    Xb = torch.rand(32, 1, requires_grad=True)
    assert bl.residual(m4, Xb).abs().max() < 1e-10
    dp = DiskPoisson()
    assert dp.residual(Exact(lambda X: 1 - X[:, :1] ** 2 - X[:, 1:2] ** 2), X).abs().max() < 1e-12
    # Burgers exact via numpy -> FD residual check of the problem's exact()
    b = Burgers1D(nu=0.1)
    x = np.linspace(0.3, 6.0, 7)
    t, h = 0.2, 1e-5
    E = lambda xx, tt: b.exact(np.column_stack([xx, np.full_like(xx, tt)]))
    r = (
        (E(x, t + h) - E(x, t - h)) / (2 * h)
        + E(x, t) * (E(x + h, t) - E(x - h, t)) / (2 * h)
        - 0.1 * (E(x + h, t) - 2 * E(x, t) + E(x - h, t)) / h**2
    )
    assert np.abs(r).max() < 1e-3


def test_hard_constraints_exact_on_boundary(f64):
    set_seed(1)
    for prob, Xb, target in (
        (Heat1D(hard=True), np.array([[0.0, 0.3], [1.0, 0.2], [0.4, 0.0]]), np.array([0.0, 0.0, np.sin(0.4 * np.pi)])),
        (Poisson2D(hard=True), np.array([[0.0, 0.3], [0.5, 1.0]]), np.zeros(2)),
        (SteadyBoundaryLayer(hard=True), np.array([[0.0], [1.0]]), np.array([0.0, 1.0])),
        (DiskPoisson(hard=True), np.array([[1.0, 0.0], [0.0, -1.0]]), np.zeros(2)),
    ):
        m = MLP(len(prob.lo), 1, 8, 2, lo=prob.lo, hi=prob.hi)
        assert np.allclose(predict(prob, m, Xb), target, atol=1e-12)
        assert prob.constraint_losses(m) == {}


def test_periodic_embedding_is_exactly_periodic():
    m = MLP(2, 1, 8, 2, lo=[0, 0], hi=[1, 1], periodic_dims=(0,))
    X = torch.tensor([[0.0, 0.3], [1.0, 0.3]])
    out = m(X)
    assert torch.allclose(out[0], out[1], atol=1e-6)


def test_determinism_and_serialisation(tmp_path):
    def run():
        set_seed(3)
        p = Heat1D()
        m = MLP(2, 1, 8, 2, lo=p.lo, hi=p.hi, seed=3)
        train_pinn(p, m, PINNConfig(iterations=30, n_collocation=64, log_every=10), seed=3)
        return m, p

    m1, p = run()
    m2, _ = run()
    G = p.eval_grid(11)
    assert np.array_equal(predict(p, m1, G), predict(p, m2, G))
    save_model(m1, tmp_path / "m.pt", {"k": 1})
    m3 = MLP(2, 1, 8, 2, lo=p.lo, hi=p.hi, seed=99)
    assert load_model(m3, tmp_path / "m.pt") == {"k": 1}
    assert np.array_equal(predict(p, m1, G), predict(p, m3, G))
    opt = torch.optim.Adam(m1.parameters())
    save_checkpoint(tmp_path / "c.pt", m1, opt, 7)
    assert load_checkpoint(tmp_path / "c.pt", m3, opt) == 7
    assert n_params(m1) > 0 and FourierFeatures(2, 3, 1.0).out_dim == 6


@pytest.mark.parametrize("name", sorted(SAMPLERS))
def test_samplers_respect_bounds_and_budget(name):
    s = make_sampler(name, [0.0, 0.0], [1.0, 2.0], 64, seed=0)
    X = s.initial()
    res = lambda P: np.exp(-((P[:, 0] - 0.5) ** 2) / 0.01)
    for step in range(1, 12):
        X = s.update(X, res, step)
        assert X.shape[1] == 2 and len(X) <= 64
        assert np.all(X >= 0) and np.all(X[:, 0] <= 1) and np.all(X[:, 1] <= 2)
    if name in ("rar", "rad", "r3"):
        assert len(X) == 64 and s.extra_evals > 0
        assert np.mean(np.abs(X[:, 0] - 0.5) < 0.1) > 0.2 + 0.05  # concentrates where residual is high
    with pytest.raises(KeyError):
        make_sampler("nope", [0], [1], 4)
    with pytest.raises(ValueError):
        make_sampler("uniform", [0], [1], 1)


def test_inside_rejection_sampling():
    dp = DiskPoisson()
    s = make_sampler("rad", dp.lo, dp.hi, 100, inside=dp.inside)
    assert dp.inside(s.initial()).all()


def test_training_reduces_error_and_weighting():
    set_seed(0)
    p = Heat1D(hard=True)
    m = MLP(2, 1, 16, 2, lo=p.lo, hi=p.hi)
    G = p.eval_grid(21)
    e0 = np.linalg.norm(predict(p, m, G) - p.exact(G))
    r = train_pinn(p, m, PINNConfig(iterations=400, n_collocation=256, lbfgs_iterations=20, log_every=100))
    assert np.linalg.norm(predict(p, m, G) - p.exact(G)) < 0.2 * e0
    assert r.history[-1]["loss"] < r.history[0]["loss"] and r.residual_evals >= 400 * 256
    p2 = Heat1D()
    m2 = MLP(2, 1, 8, 2, lo=p2.lo, hi=p2.hi)
    r2 = train_pinn(
        p2,
        m2,
        PINNConfig(
            iterations=60, n_collocation=64, adaptive_weights="annealing", anneal_every=20, causal_eps=1.0, log_every=20
        ),
        data=(G[:5], p2.exact(G[:5])),
    )
    assert set(r2.final_weights) == {"ic", "bc", "data"}
    X = torch.tensor(G[:16], dtype=torch.get_default_dtype(), requires_grad=True)
    Lr = p2.residual(m2, X).pow(2).mean()
    imb = gradient_imbalance(Lr, p2.constraint_losses(m2), list(m2.parameters()))
    assert set(imb) == {"ic", "bc"}
    aw = AnnealingWeights(["ic"])
    assert "ic" in aw.update(Lr, {"ic": p2.constraint_losses(m2)["ic"]}, list(m2.parameters()))
    cd = ConvDiff1D(pe=5.0)
    m3 = MLP(2, 1, 8, 2, lo=cd.lo, hi=cd.hi, periodic_dims=(0,))
    train_pinn(cd, m3, PINNConfig(iterations=5, n_collocation=32, conservation=1.0))
    assert residual_numpy(cd, m3, cd.eval_grid(5)).shape == (25,)


def test_divergence_raises():
    p = Heat1D()
    m = MLP(2, 1, 8, 2, lo=p.lo, hi=p.hi)
    with pytest.raises(SolverDivergenceError):
        train_pinn(p, m, PINNConfig(iterations=5, n_collocation=16, lr=1e30, weights={"residual": 1e30}))


def test_baseline_and_domain_decomposition():
    p = Heat1D()
    G = p.eval_grid(11)
    m, wall = train_supervised(G, p.exact(G), p.lo, p.hi, width=8, depth=2, iterations=50)
    assert wall > 0 and m(torch.tensor(G[:2], dtype=torch.get_default_dtype())).shape == (2, 1)
    r = train_dd_boundary_layer(10.0, iterations=50, n=32)
    assert np.isfinite(r.rel_l2) and r.interface_jump_u >= 0
