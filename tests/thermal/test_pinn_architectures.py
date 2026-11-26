"""Step 4.1: architectures — shapes, structure, derivatives, reproducibility, spec validation, short training."""

import math

import pytest
import torch

from aerosynapse.thermal.pinn import architectures as A
from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import spec as S
from aerosynapse.thermal.pinn.derivs import evaluate
from aerosynapse.thermal.pinn.trainer import Trainer

torch.set_default_dtype(torch.float64)
SPEC = S.load_spec(None, "smoke")
LO, HI = [0.0, 0.0], [1.0, 2.0]


def net_cfg(**over):
    return S.deep_merge(SPEC["network"], over)


def build(arch, seed=0, **over):
    torch.manual_seed(seed)
    return A.build_network(net_cfg(arch=arch, **over), LO, HI)


@pytest.mark.parametrize("arch", A.ARCHS)
def test_shapes_scaling_and_second_derivatives(arch):
    net = build(arch)
    X = torch.rand(20, 2) * torch.tensor([1.0, 2.0])
    assert net(X).shape == (20, 1) and torch.isfinite(net(X)).all()
    # fixed input scaling: the network output depends on X only through (X - lo) / (hi - lo)
    assert torch.allclose(net(X), net(X.clone()))
    d = evaluate(net, {"x": X[:, 0], "t": X[:, 1]}, ["x", "t"], first=("x", "t"), second=("x",), create_graph=False)
    h = 1e-5
    f = lambda a, b: net(torch.stack([a, b], 1)).squeeze(-1).detach()  # noqa: E731
    fd_x = (f(X[:, 0] + h, X[:, 1]) - f(X[:, 0] - h, X[:, 1])) / (2 * h)
    assert torch.allclose(d["d_x"], fd_x, rtol=1e-5, atol=1e-5)
    h2 = 1e-4
    fd_xx = (f(X[:, 0] + h2, X[:, 1]) - 2 * f(X[:, 0], X[:, 1]) + f(X[:, 0] - h2, X[:, 1])) / h2**2
    assert torch.allclose(d["dd_x"], fd_xx, rtol=1e-3, atol=1e-3)
    assert net.n_params() > 0


@pytest.mark.parametrize("arch", A.ARCHS)
def test_seed_reproducible_and_different_seeds_differ(arch):
    a, b, c = build(arch, 1), build(arch, 1), build(arch, 2)
    X = torch.rand(5, 2)
    assert torch.equal(a(X), b(X)) and not torch.equal(a(X), c(X))


def test_parameter_counts():
    w, d = 32, 3
    mlp = build("mlp").n_params()
    assert mlp == (2 * w + w) + 2 * (w * w + w) + (w + 1)
    # fourier (non-trainable B): first layer takes 2*n_features inputs; B is a buffer
    f = build("fourier")
    assert f.n_params() == (64 * w + w) + 2 * (w * w + w) + (w + 1) and "B" not in dict(f.named_parameters())
    assert "B" in dict(build("fourier", fourier={"trainable": True}).named_parameters())
    # rwf: plain-MLP count + one scale s per output neuron
    assert build("rwf").n_params() == mlp + (w + w + w + 1)
    # resnet: embed + depth blocks of 2 linear layers + output
    assert build("resnet").n_params() == (2 * w + w) + d * 2 * (w * w + w) + (w + 1)
    # modified mlp: three input layers + (depth-1) hidden + output
    assert build("modified_mlp").n_params() == 3 * (2 * w + w) + (d - 1) * (w * w + w) + (w + 1)
    assert build("siren").n_params() == (2 * w + w) + 2 * (w * w + w) + (w + 1)


def test_fourier_features_follow_formula_and_per_input_sigma():
    net = build("fourier", fourier={"sigma": [1.0, 5.0], "n_features": 8, "include_input": True})
    X = torch.rand(6, 2) * torch.tensor([1.0, 2.0])
    z = 2 * (X - net.lo) / (net.hi - net.lo) - 1
    E = net.embed(X)
    proj = 2 * math.pi * z @ net.B.T
    assert E.shape == (6, 2 * 8 + 2)
    assert (
        torch.allclose(E[:, :8], torch.sin(proj))
        and torch.allclose(E[:, 8:16], torch.cos(proj))
        and torch.allclose(E[:, 16:], z)
    )
    # larger sigma on the second column => larger |B| entries there (statistically, 8 draws each)
    big = build("fourier", fourier={"sigma": [0.1, 50.0], "n_features": 400})
    assert big.B[:, 1].abs().mean() > 50 * big.B[:, 0].abs().mean()
    with pytest.raises(ValueError):
        build("fourier", fourier={"sigma": [1.0, 2.0, 3.0]})


def test_modified_mlp_gating_structure():
    net = build("modified_mlp", depth=3)
    z = torch.rand(4, 2) * 2 - 1
    X = (z + 1) / 2 * (net.hi - net.lo) + net.lo
    act = net.act
    U, V, H = act(net.enc_u(z)), act(net.enc_v(z)), act(net.first(z))
    for lin in net.hidden:
        Zk = act(lin(H))
        H = (1 - Zk) * U + Zk * V
    assert torch.allclose(net(X), net.out(H))
    # gate extremes: Z=0 -> H=U, Z=1 -> H=V (convex combination property)
    assert torch.allclose((1 - torch.zeros(3)) * U[0, :3] + torch.zeros(3) * V[0, :3], U[0, :3])


def test_resnet_is_identity_plus_residual():
    net = build("resnet", depth=2)
    for blk in net.blocks:  # zero residual branches -> h unchanged by blocks (act(0) = 0 for tanh)
        for lin in blk:
            torch.nn.init.zeros_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
    X = torch.rand(5, 2)
    h = net.act(net.embed(net.scale(X)))
    assert torch.allclose(net(X), net.out(h))


def test_rwf_reparametrisation_and_gradients():
    net = build("rwf", rwf={"mu": 1.0, "sigma": 0.1})
    lay = net.layers[0]
    W = lay.weight()
    assert torch.allclose(W, torch.exp(lay.s)[:, None] * lay.V)
    # the initial effective weight has the xavier scale (V = W0 / exp(s) undoes the factor exactly at init)
    std = W.std().item()
    assert 0.3 * math.sqrt(2 / (2 + 32)) * 1.2 < std < 3.0 * math.sqrt(2 / (2 + 32)) * 1.67
    net(torch.rand(4, 2)).sum().backward()
    assert lay.s.grad is not None and lay.s.grad.abs().sum() > 0 and lay.V.grad.abs().sum() > 0
    # sigma = 0 and mu = 0: identical forward values to a plain linear with the same effective weights
    z = torch.rand(3, 2)
    assert torch.allclose(lay(z), z @ lay.weight().T + lay.b)


def test_siren_initialisation_ranges_and_frequency():
    net = build("siren", siren={"omega0": 10.0})
    assert net.layers[0].weight.abs().max() <= 1 / 2 + 1e-12
    assert net.layers[1].weight.abs().max() <= math.sqrt(6 / 32) / 10 + 1e-12
    assert net.out.weight.abs().max() <= math.sqrt(6 / 32) / 10 + 1e-12
    with pytest.raises(ValueError):
        build("siren", siren={"omega0": 0.0})


def test_factory_rejects_unknown_arch_and_validation_messages():
    with pytest.raises(ValueError, match="unknown architecture"):
        A.build_network(net_cfg(arch="transformer"), LO, HI)
    for over, needle in [
        ({"network": {"arch": "gnn"}}, "network.arch"),
        ({"network": {"fourier": {"sigma": -1.0}}}, "fourier.sigma"),
        ({"network": {"fourier": {"n_features": 0}}}, "n_features"),
        ({"network": {"resnet": {"block_layers": 3}}}, "block_layers"),
        ({"network": {"siren": {"omega0": 0}}}, "omega0"),
        ({"arch_study": {"seeds": []}}, "arch_study.seeds"),
        ({"arch_study": {"cases": [{"problem": "X", "value": 1.0}]}}, "arch_study.cases[0].problem"),
        ({"arch_study": {"architectures": [{"name": "bad", "arch": "mlp", "depth": 0}]}}, "architectures[0].depth"),
    ]:
        with pytest.raises(ValueError, match="invalid setting at") as e:
            S.load_spec(None, "smoke", over)
        assert needle in str(e.value)


def test_user_variants_label_and_override():
    spec = S.load_spec(None, "smoke")
    label, net = S.network_variant(
        spec, {"name": "fourier_s4", "arch": "fourier", "fourier": {"sigma": 4.0}, "width": 16}
    )
    assert (
        label == "fourier_s4"
        and net["fourier"]["sigma"] == 4.0
        and net["fourier"]["n_features"] == 32
        and net["width"] == 16
    )
    assert S.network_variant(spec, "siren")[1]["arch"] == "siren"


@pytest.mark.parametrize("arch", A.ARCHS)
def test_each_architecture_trains_a_few_epochs_without_nan(arch, tmp_path):
    spec = S.load_spec(
        None, "smoke", S.parse_set(["training.adam.epochs=25", "training.lbfgs.iterations=0", "training.log_every=5"])
    )
    spec["network"] = net_cfg(arch=arch, width=12, depth=2)
    spec["sampling"].update({"n_pde": 64, "n_ic": 16, "n_bc": 8})
    prob = P.build_problem("B_slab", spec["problems"]["B_slab"], 5.0)
    torch.manual_seed(0)
    net = A.build_network(spec["network"], prob.lo, prob.hi)
    out = Trainer(net, prob, spec, tmp_path).run()
    assert out["status"] == "ok"
    rows = (tmp_path / "log.csv").read_text().splitlines()  # columns: phase, step, total, ...
    first, last = float(rows[1].split(",")[2]), float(rows[-1].split(",")[2])
    assert last < first
