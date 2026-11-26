"""Step 6.5: Jacobian / NTK, Hessian-vector products, landscape and spectra (checked against finite differences)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from aerosynapse.thermal.pinn import diagnostics as D
from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn.architectures import build_network
from aerosynapse.thermal.viz import diagnostics_figs as F


def _setup(name="A_steady", value=10.0, width=6):
    s = SP.load_spec(None, "smoke", {"network": {"width": width, "depth": 1}})
    prob = P.build_problem(name, s["problems"][name], value)
    torch.manual_seed(0)
    net = build_network(s["network"], prob.lo, prob.hi).to(dtype=torch.float64)
    pts = D.point_set(prob, 12, 6, 3)
    return prob, net, pts


def test_flat_roundtrip_and_loss():
    prob, net, pts = _setup()
    th = D.flat(net)
    D.set_flat(net, th + 1.0)
    assert torch.allclose(D.flat(net), th + 1.0)
    D.set_flat(net, th)
    L = float(D.loss(net, prob, pts))
    r = D.residuals(net, prob, pts)
    assert L == pytest.approx(sum(float(torch.mean(v**2)) for v in r.values()))


def test_jacobian_matches_finite_differences():
    prob, net, pts = _setup("B_slab", 1.0)
    J, lab = D.jacobian(net, prob, pts)
    th = D.flat(net)
    i = 5
    h = 1e-6
    e = torch.zeros_like(th)
    e[i] = h

    def rvec(v):
        D.set_flat(net, v)
        return torch.cat([x.detach().reshape(-1) for x in D.residuals(net, prob, pts).values()]).numpy()

    fd = (rvec(th + e) - rvec(th - e)) / (2 * h)
    D.set_flat(net, th)
    assert np.allclose(J[:, i], fd, rtol=1e-5, atol=1e-7)
    st = D.ntk_stats(J, lab)
    st.pop("_K")
    assert set(st["trace_per_point"]) == {"pde", "ic", "bc_left", "bc_right"}
    assert st["eigenvalues"][0] >= st["eigenvalues"][-1] >= 0


def test_hvp_and_full_hessian():
    prob, net, pts = _setup(width=4)
    th = D.flat(net)
    n = th.numel()
    v = torch.as_tensor(np.random.default_rng(0).standard_normal(n))

    def grad(w):
        D.set_flat(net, w)
        g = torch.autograd.grad(D.loss(net, prob, pts), D.params(net))
        return torch.cat([x.reshape(-1) for x in g]).detach()

    h = 1e-6
    fd = (grad(th + h * v) - grad(th - h * v)) / (2 * h)
    D.set_flat(net, th)
    hv = D.hvp(net, prob, pts, v)
    assert torch.allclose(hv, fd, rtol=1e-4, atol=1e-6)
    H = D.hessian_full(net, prob, pts)
    assert np.allclose(H, H.T) and np.allclose(H @ v.numpy(), hv.numpy(), rtol=1e-6, atol=1e-8)
    ex = D.hessian_extremes(net, prob, pts)
    lam = np.linalg.eigvalsh(H)
    assert ex["max"] == pytest.approx(lam[-1], rel=1e-4) and ex["min"] == pytest.approx(lam[0], rel=1e-3, abs=1e-6)


def test_landscape_interpolation_alignment_spectra():
    prob, net, pts = _setup()
    th = D.flat(net)
    L0 = float(D.loss(net, prob, pts))
    d1 = D.filter_normalised(net, np.random.default_rng(1))
    d2 = D.filter_normalised(net, np.random.default_rng(2))
    Z = D.landscape(net, prob, pts, d1, d2, np.array([-1.0, 0.0, 1.0]), np.array([0.0]))
    assert Z[0, 1] == pytest.approx(L0) and torch.allclose(D.flat(net), th)
    s, L = D.interpolate(net, prob, pts, th + d1, th, n=15)  # s = -0.2, -0.1, ..., 1.2: s = 1 is the end point
    k = int(np.argmin(np.abs(s - 1.0)))
    assert s[k] == pytest.approx(1.0) and L[k] == pytest.approx(L0)
    y = np.random.default_rng(3).standard_normal(10)
    assert D.alignment(np.outer(y, y), y) == pytest.approx(1.0)
    x = (np.arange(64) + 0.5) / 64
    a = D.dct_amplitudes(np.cos(np.pi * 5 * x))
    assert np.argmax(a) == 5
    B, labs = F.bands(np.array([a]))
    assert labs[:4] == ["0", "1", "2-3", "4-7"] and B.shape[1] == len(labs)
    c1, c2, coords = D.pca_directions([th + 2 * d1, th + d1, th], th)
    assert coords.shape == (3, 2) and abs(coords[-1]).max() < 1e-12


def test_figures_from_fake_results():
    prob, net, pts = _setup()
    J, lab = D.jacobian(net, prob, pts)
    st = D.ntk_stats(J, lab)
    st.pop("_K")
    e = {
        "snapshot": 0,
        "phase": "init",
        "step": 0,
        "drift": 0.0,
        "residual_share_top": [0.1, 0.3, 0.5, 0.7, 1.0],
        "hessian": {"max": 10.0, "min": -1.0},
    } | st
    e2 = dict(e, snapshot=2, drift=0.3)
    r = {
        "case": "c",
        "variant": "mlp",
        "problem": "A_steady",
        "value": 10.0,
        "steps": [("init", 0), ("adam", 50), ("lbfgs", 0)],
        "rel_l2": [1.0, 0.1, 0.01],
        "loss": [1.0, 0.1, 0.01],
        "spectrum": [list(np.abs(np.random.default_rng(i).standard_normal(256))) for i in range(3)],
        "ntk": [e, e2],
        "hessian_spectrum": list(np.linspace(-1, 10, 50)),
        "points": {"n_pde": 12, "n_ic": 6, "n_t": 3},
        "landscape": {
            "a": [-1, 0, 1],
            "b": [-1, 0, 1],
            "loss": [[1, 2, 3], [2, 0.1, 2], [3, 2, 1]],
            "directions": "pca",
        },
        "trajectory": [[1.0, 0.5], [0.5, 0.2], [0.0, 0.0]],
        "interpolation": {"s": [0, 1], "loss": [1, 0.01]},
    }
    cases = [r, dict(r, variant="fourier")]
    F.style()
    for fn in (
        F.spectral_surfaces,
        F.error_spectra,
        F.ntk_spectra,
        F.ntk_waterfall,
        F.ntk_eigenfunctions,
        F.dynamics,
        F.hessian,
        F.landscapes3d,
        F.landscapes2d,
    ):
        plt.close(fn(cases))
    w = [{"width": 16, "n_params": 100, "drift": [0, 0.5], "lambda_max": [1, 2], "param_change": 0.1}]
    s = [{"sigma": 0.5, "alignment_fd": 0.3, "rel_l2": 0.01}, {"sigma": 2.0, "alignment_fd": 0.2, "rel_l2": 0.1}]
    plt.close(F.width_sigma(w, s))
    t = {
        "variant": "mlp",
        "frequencies": [1, 4],
        "epochs": [0, 25],
        "amplitude": [[1, 1], [0.1, 0.9]],
        "spectrum": [[1] * 64, [0.5] * 64],
    }
    plt.close(F.toy([t]))
