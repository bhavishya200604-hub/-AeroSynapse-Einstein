"""Step 3.1: network + autograd derivatives vs analytic derivatives and finite differences."""

import numpy as np
import pytest
import torch

from aerosynapse.thermal.pinn.derivs import evaluate, grad
from aerosynapse.thermal.pinn.networks import MLP, resolve_device

torch.set_default_dtype(torch.float64)


def analytic_model(X):  # u = sin(3x) exp(-2t) + x t^2 + p x   (columns x, t, p)
    x, t, p = X[:, 0], X[:, 1], X[:, 2]
    return (torch.sin(3 * x) * torch.exp(-2 * t) + x * t**2 + p * x).unsqueeze(-1)


def test_derivatives_match_analytic_exactly():
    g = torch.Generator().manual_seed(0)
    cols = {k: torch.rand(50, generator=g) for k in ("x", "t", "p")}
    d = evaluate(analytic_model, cols, ["x", "t", "p"], first=("x", "t", "p"), second=("x", "t"))
    x, t, p = cols["x"], cols["t"], cols["p"]
    assert torch.allclose(d["u"], torch.sin(3 * x) * torch.exp(-2 * t) + x * t**2 + p * x)
    assert torch.allclose(d["d_x"], 3 * torch.cos(3 * x) * torch.exp(-2 * t) + t**2 + p, atol=1e-13)
    assert torch.allclose(d["d_t"], -2 * torch.sin(3 * x) * torch.exp(-2 * t) + 2 * x * t, atol=1e-13)
    assert torch.allclose(d["d_p"], x, atol=1e-13)
    assert torch.allclose(d["dd_x"], -9 * torch.sin(3 * x) * torch.exp(-2 * t), atol=1e-12)
    assert torch.allclose(d["dd_t"], 4 * torch.sin(3 * x) * torch.exp(-2 * t) + 2 * x, atol=1e-12)


@pytest.mark.parametrize("act", ["tanh", "sin", "silu", "gelu"])
def test_mlp_derivatives_match_finite_differences(act):
    torch.manual_seed(1)
    net = MLP([0.0, 0.0], [1.0, 2.0], width=16, depth=3, activation=act)
    x = torch.rand(40) * 0.8 + 0.1
    t = torch.rand(40) * 1.6 + 0.2
    d = evaluate(net, {"x": x, "t": t}, ["x", "t"], first=("x", "t"), second=("x",), create_graph=False)
    h, h2 = 1e-5, 1e-4  # FD truncation error ~ h^2 f''' (measured: 1.2e-8 at h=1e-5), round-off ~ eps/h

    def f(xx, tt):
        return net(torch.stack([xx, tt], 1)).squeeze(-1).detach()

    fd_x = (f(x + h, t) - f(x - h, t)) / (2 * h)
    fd_t = (f(x, t + h) - f(x, t - h)) / (2 * h)
    fd_xx = (f(x + h2, t) - 2 * f(x, t) + f(x - h2, t)) / h2**2
    assert torch.allclose(d["d_x"], fd_x, rtol=1e-6, atol=1e-6)
    assert torch.allclose(d["d_t"], fd_t, rtol=1e-6, atol=1e-6)
    assert torch.allclose(d["dd_x"], fd_xx, rtol=1e-4, atol=1e-5)


def test_graph_retained_for_training_and_detached_otherwise():
    net = MLP([0.0], [1.0], width=8, depth=2)
    x = torch.rand(10)
    d = evaluate(net, {"x": x}, ["x"], first=("x",), second=("x",))
    loss = (d["dd_x"] ** 2).mean()
    loss.backward()  # second derivative must be differentiable w.r.t. the weights
    # every parameter except the output bias (d2u/dx2 does not depend on it) receives a finite gradient
    params = list(net.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params[:-1])
    assert params[-1].grad is None or torch.all(params[-1].grad == 0)
    d2 = evaluate(net, {"x": x}, ["x"], first=("x",), second=("x",), create_graph=False)
    assert not d2["dd_x"].requires_grad


def test_input_scaling_is_fixed_by_bounds_and_missing_column_errors():
    net = MLP([0.0, 10.0], [2.0, 20.0], width=4, depth=1)
    X = torch.tensor([[0.0, 10.0], [2.0, 20.0], [1.0, 15.0]])
    z = 2 * (X - net.lo) / (net.hi - net.lo) - 1
    assert torch.allclose(z, torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(KeyError):
        evaluate(net, {"x": torch.rand(3)}, ["x", "t"], first=("x", "t"))
    assert net.n_params() == 2 * 4 + 4 + 4 + 1 and resolve_device("cpu").type == "cpu"


def test_grad_unused_returns_zeros():
    x = torch.rand(5, requires_grad=True)
    y = torch.rand(5, requires_grad=True)
    assert torch.equal(grad((x**2), y), torch.zeros(5)) and np.allclose(
        grad(x**2, x).detach().numpy(), 2 * x.detach().numpy()
    )
