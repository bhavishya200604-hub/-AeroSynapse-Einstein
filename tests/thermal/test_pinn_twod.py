"""Step 6.6b: designs (rays at a given distance outside the box, slices), UQ / extrapolation analysis, ROC, the 2D
dispatch of the diagnostics (problems with residual_vectors) and the figures (fake data)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from aerosynapse.thermal.experiments import p6_6b_2d as T
from aerosynapse.thermal.pinn import diagnostics as D
from aerosynapse.thermal.pinn import validity as VA
from aerosynapse.thermal.viz import twod_figs as F

BOX = {"Re": [10.0, 63.1], "Pr": [0.7, 1.62], "Bi": [0.5, 19.9]}


def test_ray_points_distance():
    P, meta = T.ray_points(BOX, [0.0, 0.25, 1.0], 3, 0)
    assert len(P) == 9 * 3 and sum(m["axis"] for m in meta) == 6 * 3
    for p, m in zip(P, meta, strict=True):
        d = max(float(VA.distance(p[i], *BOX[k])) for i, k in enumerate(T.NAMES))
        assert d == pytest.approx(m["d"], abs=1e-9)  # d = 0: strictly inside
        if m["d"] == 0:
            assert all(BOX[k][0] < p[i] < BOX[k][1] for i, k in enumerate(T.NAMES))


def test_slice_and_uq_points():
    P, a, b = T.slice_points(BOX, ["Re", "Pr"], 5, 0.5)
    assert P.shape == (25, 3) and a[0] == pytest.approx(1.0 - 0.5) and np.allclose(P[:, 2], np.sqrt(0.5 * 19.9))
    U = T.uq_points(BOX, 50, 1)
    assert all((U[:, i] >= BOX[k][0]).all() and (U[:, i] <= BOX[k][1]).all() for i, k in enumerate(T.NAMES))


def test_analyse_uq_calibrated_ensemble():
    rng = np.random.default_rng(0)
    n, m = 400, 40
    truth_sd = 0.1
    y = rng.standard_normal(n)
    M = y[None, :] + truth_sd * rng.standard_normal((m, n))
    # members scatter around the truth -> mean ~ y, sd ~ truth_sd / err ~ truth_sd / sqrt(m): over-dispersed
    out = T.analyse_uq(y, M, [0.5, 0.9], 0.1, 50, 0)
    assert out["gaussian"]["coverage"]["0.9"] > 0.95  # over-dispersed -> over-covers
    assert out["conformal"]["coverage_mean"] == pytest.approx(0.9, abs=0.05)  # conformal fixes it
    assert out["conformal"]["q_median"] < out["conformal"]["z_gauss"]


def test_roc_and_extrapolation():
    fpr, tpr = T.roc([3, 2, 1, 0], [True, True, False, False])
    assert tpr[2] == 1.0 and fpr[2] == 0.0
    rec = [
        {"y": 1.0, "rel_err": 0.01 * (1 + 10 * d), "spread": 0.01 * (1 + 5 * d), "residual": 0.1 + d, "d": d}
        for d in (0, 0, 0.5, 1)
    ]
    inr = {"spread": np.array([0.01, 0.011]), "residual": np.array([0.1, 0.11])}
    x = T.analyse_extrap(rec, 0.05, 0.9, inr)
    assert x["indicators"]["spread"]["auc"] == 1.0 and x["by_distance"][-1]["d"] == 1


class _Toy:
    """A problem in the 2D interface: residual_vectors(model, batch) with two groups."""

    def residual_vectors(self, model, batch):
        X = batch["pde"].clone().requires_grad_(True)
        u = model(X).squeeze(-1)
        ux = torch.autograd.grad(u.sum(), X, create_graph=True)[0][:, 0]
        return {"pde_a": ux - 1.0, "bc_u": model(batch["walls"]["w"]).squeeze(-1)}


def test_diagnostics_dispatch_2d():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 5), torch.nn.Tanh(), torch.nn.Linear(5, 1)).double()
    prob = _Toy()
    pts = {"pde": torch.rand(6, 2, dtype=torch.float64), "walls": {"w": torch.rand(3, 2, dtype=torch.float64)}}
    J, lab = D.jacobian(net, prob, pts)
    assert J.shape == (9, D.flat(net).numel()) and lab.count("pde_a") == 6
    th = D.flat(net)
    i, h = 3, 1e-6
    e = torch.zeros_like(th)
    e[i] = h

    def rv(v):
        D.set_flat(net, v)
        return torch.cat([x.detach().reshape(-1) for x in prob.residual_vectors(net, pts).values()]).numpy()

    fd = (rv(th + e) - rv(th - e)) / (2 * h)
    D.set_flat(net, th)
    assert np.allclose(J[:, i], fd, rtol=1e-5, atol=1e-8)
    st = D.ntk_stats(J, lab)
    assert len(st["pde_eigenvalues"]) == 6  # the pde_* rows form the PDE block
    c = T._conflict(net, prob, pts, {"pde_a": 2.0, "bc_u": 1.0})
    assert np.allclose(np.diag(c["cosine"]), 1.0) and c["weighted"]["grad_pde"] == pytest.approx(
        2 * c["unweighted"]["grad_pde"]
    )


def test_figures_from_fake_results():
    rng = np.random.default_rng(1)
    y = rng.uniform(5, 10, 20)
    M = y + 0.1 * rng.standard_normal((3, 20))
    an_uq = T.analyse_uq(y, M, [0.5, 0.9], 0.1, 10, 0)
    P, meta = T.ray_points(BOX, [0.0, 0.5], 1, 0)
    rec = [
        {
            "y": 1.0,
            "rel_err": 0.01 + m["d"] * rng.uniform(),
            "spread": 0.01 + 0.1 * m["d"],
            "residual": 0.1 + m["d"],
            "d": m["d"],
        }
        for m in meta
    ]
    inr = {"spread": np.array([0.01, 0.02]), "residual": np.array([0.1, 0.12])}
    ex = T.analyse_extrap(rec, 0.05, 0.9, inr) | {"records": rec}
    n = 4
    sl = {"rel_err": list(rng.uniform(0.01, 0.2, n * n)), "spread": list(rng.uniform(0.01, 0.1, n * n))}
    k = 3
    conf = {
        "groups": ["pde_mom_x", "pde_energy", "bc_u"],
        "loss": {"pde_mom_x": 1.0, "pde_energy": 1.0, "bc_u": 1.0},
        "grad_norm": [1.0, 2.0, 3.0],
        "cosine": np.eye(k).tolist(),
        "unweighted": {"cosine": -0.5, "cancellation": 0.3, "grad_pde": 1, "grad_bc": 1},
        "weighted": None,
    }
    tagd = {
        "conflict": conf,
        "ntk": {"eigenvalues": [3.0, 2.0, 1.0], "effective_rank": 2.5},
        "hessian": {"max": 1.0, "min": -0.1},
    }
    diag = {
        "member": "seed 0",
        "init": tagd,
        "final": tagd | {"hessian_spectrum": [-0.1, 0.0, 1.0]},
        "landscape": {"a": [-1, 0, 1], "loss": [[2, 1, 2], [1, 0.1, 1], [2, 1, 2]]},
        "interpolation": {"s": [0, 0.5, 1], "loss": [1, 0.5, 0.1]},
        "reload_max_rel_diff": 0.0,
        "ntk_drift": 0.5,
    }
    cases = {
        "D_graetz": {
            "qoi": "mean Nu",
            "box": BOX,
            "members": ["a", "b", "c"],
            "ray_meta": meta,
            "slice_grid": {
                "pair": ["Re", "Pr"],
                "a": list(np.linspace(0.5, 2.3, n)),
                "b": list(np.linspace(-0.6, 0.7, n)),
            },
            "fields": [{"values": {"Re": 20, "Pr": 1, "Bi": 3}}],
            "diagnostics": [diag, dict(diag, member="seed 1")],
        }
    }
    an = {"D_graetz": {"uq": an_uq, "extrapolation": ex, "slice": sl}}
    x, yy = np.linspace(0, 8, 9), np.linspace(0, 1, 4)
    fields = {"D_graetz_0_x": x, "D_graetz_0_y": yy} | {
        f"D_graetz_0_{k}": rng.uniform(size=(4, 9)) for k in ("theta", "theta_mean", "theta_sd")
    }
    F.style()
    for fig in (
        F.uq(cases, an),
        F.uq_field3d(cases, fields),
        F.extrapolation(cases, an),
        F.slice3d(cases, an),
        F.conflict(cases),
        F.spectra(cases),
        F.landscapes3d(cases),
    ):
        plt.close(fig)
