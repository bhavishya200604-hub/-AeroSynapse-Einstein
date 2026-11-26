"""Step 5.2: inverse-PINN residual assembly, known / unknown parameters, collocation, tiny joint / two-stage /
grad-balanced trainings, measurement data (synthetic, user files and their validation), the least-squares reference
and the settings validation."""

import math

import numpy as np
import pytest
import torch
from torch import nn

from aerosynapse.thermal.inverse import body as B
from aerosynapse.thermal.inverse import data as D
from aerosynapse.thermal.inverse import pinn as PI
from aerosynapse.thermal.inverse import spec as IS

SPEC = IS.load_spec()
TRUE = {k: float(v) for k, v in SPEC["parameters"].items()}
DT = torch.float64


class Linear(nn.Module):
    """N(s, t) = a + b s  ->  theta = 1 + t (a + b s)."""

    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = nn.Parameter(torch.tensor(a, dtype=DT)), nn.Parameter(torch.tensor(b, dtype=DT))

    def forward(self, X):
        return (self.a + self.b * X[:, 0]).unsqueeze(-1)


@pytest.mark.parametrize("shape", ["slab", "cylinder", "sphere"])
def test_residual_assembly_on_a_polynomial_ansatz(shape):
    body = B.Body(shape, 0.02, 80.0, 20.0, 1e5)
    m = PI.InversePINN(IS.pinn_settings(SPEC)["network"], body, 50.0, 10.0, ["alpha", "k", "h"], TRUE, TRUE)
    m.net = Linear(0.3, -0.2)
    s, t = torch.tensor([0.1, 0.5, 0.9], dtype=DT), torch.tensor([0.2, 1.0, 3.0], dtype=DT)
    F = TRUE["alpha"] * 50.0 / 0.02**2
    Q = 1e5 * 0.02**2 / (TRUE["k"] * 60.0)
    Bi = TRUE["h"] * 0.02 / TRUE["k"]
    n = body.n
    expected = (
        (0.3 - 0.2 * s) - F * 2 * (n + 1) * t * (-0.2) - F * Q
    )  # theta_t - F (2(n+1) theta_s + 4 s theta_ss) - F Q
    assert torch.allclose(m.pde_residual(s, t).detach(), expected, rtol=1e-12)
    bc = -2 * t * (-0.2) - Bi * (1 + t * (0.3 - 0.2))
    assert torch.allclose(m.bc_residual(t).detach(), bc, rtol=1e-12)
    assert torch.allclose(
        m.theta(s, torch.zeros(3, dtype=DT)).detach(), torch.ones(3, dtype=DT)
    )  # exact initial condition


def test_known_and_unknown_parameters():
    m = PI.InversePINN(
        IS.pinn_settings(SPEC)["network"],
        IS.body_of(SPEC),
        100.0,
        12.0,
        ["alpha", "h"],
        {"alpha": 2e-6, "h": 50.0},
        TRUE,
    )
    assert set(m.logp) == {"alpha", "h"} and m.estimates()["k"] == pytest.approx(15.0, rel=1e-14)
    assert m.estimates()["alpha"] == pytest.approx(2e-6, rel=1e-14) and m.estimates()["h"] == pytest.approx(
        50.0, rel=1e-14
    )


def test_collocation_points():
    c = PI.collocation(64, 16, 12.0, 0)
    assert c["s"].shape == (64,) and float(c["s"].min()) >= 0 and float(c["s"].max()) <= 1
    assert float(c["t"].max()) <= 12.0 and c["t_bc"].shape == (16,)
    b = PI.collocation(32, 4, 12.0, 0, (0.2, 0.8, 1.0, 5.0))
    assert float(b["s"].min()) >= 0.04 - 1e-12 and float(b["s"].max()) <= 0.64 + 1e-12 and float(b["t"].min()) >= 1.0


@pytest.mark.parametrize("mode,bal", [("joint", "fixed"), ("joint", "grad_balance"), ("two_stage", "fixed")])
def test_tiny_trainings(mode, bal):
    s = IS.load_spec(
        overrides={
            "measurement": {"sensors": {"n": 3}, "times": {"n": 6}},
            "pinn": {
                "collocation": {"n_pde": 64, "n_bc": 8},
                "adam": {"epochs": 12},
                "lbfgs": {"iterations": 3},
                "two_stage": {"fit_epochs": 8, "fit_lbfgs": 2, "param_epochs": 6},
                "log_every": 4,
                "balancing": {"method": bal, "every": 4},
            },
        }
    )
    d = D.load(s)
    init = {k: float(s["initial_guess"][k]) for k in s["unknown"]}
    r = PI.train(IS.pinn_settings(s), IS.body_of(s), d, s["unknown"], init, TRUE, mode, 0)
    assert all(np.isfinite(v) and v > 0 for v in r["estimates"].values()) and r["history"]
    stages = {h["stage"] for h in r["history"]}
    assert stages == (
        {"fit", "fit_lbfgs", "params", "params_lbfgs"} if mode == "two_stage" else {"joint", "joint_lbfgs"}
    )
    if bal == "grad_balance":
        assert r["w_data_final"] != 1.0
    if mode == "two_stage":  # the network is fitted first: the parameters move only in the 'params' stages
        fit = [h for h in r["history"] if h["stage"].startswith("fit")]
        assert all(h["alpha"] == pytest.approx(init["alpha"], rel=1e-14) for h in fit)


def test_synthetic_data_and_file_round_trip(tmp_path):
    d = D.synthetic(SPEC, 0.02, 5)
    assert len(d["T"]) == 8 * 60 and np.all(d["sigma"] > 0) and np.abs(d["T"] - d["T_clean"]).max() > 0
    f = tmp_path / "m.csv"
    with open(f, "w", encoding="utf-8") as fh:
        fh.write("# comment line\nx,t,T,sigma\n")
        np.savetxt(fh, np.stack([d["r"], d["t"], d["T"], d["sigma"]], 1), delimiter=",", fmt="%.12g")
    r = D.read_file(f, SPEC, 0.02)
    assert np.allclose(r["T"], d["T"]) and np.allclose(r["sigma"], d["sigma"]) and r["T_clean"] is None
    np.savez(tmp_path / "m.npz", x=d["r"], t=d["t"], T=d["T"])  # no sigma column: Phase 2 model at the level
    assert np.all(D.read_file(tmp_path / "m.npz", SPEC, 0.02)["sigma"] > 0)


@pytest.mark.parametrize(
    "rows,msg",
    [
        ("x,t\n0.01,5\n", "column 'T' missing"),
        ("x,t,T\n0.05,5,60\n", "x must lie"),
        ("x,t,T\n0.01,0,60\n", "t > 0"),
        ("x,t,T,sigma\n0.01,5,60,0\n", "sigma must be > 0"),
    ],
)
def test_data_file_validation(tmp_path, rows, msg):
    f = tmp_path / "bad.csv"
    f.write_text(rows, encoding="utf-8")
    with pytest.raises(ValueError, match=msg):
        D.read_file(f, SPEC, 0.02)
    with pytest.raises(FileNotFoundError):
        D.read_file(tmp_path / "missing.csv", SPEC, 0.02)


def test_least_squares_recovers_noise_free_parameters():
    d = D.synthetic(SPEC, 0.02, 0)
    d = {**d, "T": d["T_clean"]}
    est = D.least_squares_estimate(SPEC, d, ["alpha", "k", "h"], {"alpha": 2e-6, "k": 30.0, "h": 50.0})["estimates"]
    assert all(abs(est[k] / TRUE[k] - 1) < 1e-6 for k in TRUE)
    s0 = IS.load_spec(overrides={"body": {"q_vol": 0.0}, "unknown": ["alpha", "h"]})  # no source, k KNOWN: identifiable
    d0 = D.synthetic(s0, 0.02, 0)
    e0 = D.least_squares_estimate(s0, {**d0, "T": d0["T_clean"]}, ["alpha", "h"], {"alpha": 2e-6, "h": 50.0})[
        "estimates"
    ]
    assert abs(e0["alpha"] / TRUE["alpha"] - 1) < 1e-6 and abs(e0["h"] / TRUE["h"] - 1) < 1e-6 and e0["k"] == 15.0


def test_settings():
    p = IS.pinn_settings(SPEC, {"name": "x", "weights": {"data": 10.0}, "balancing": {"method": "grad_balance"}})
    assert p["weights"]["data"] == 10.0 and p["weights"]["pde"] == 1.0 and p["balancing"]["method"] == "grad_balance"
    assert "fourier" in p["network"] and p["network"]["width"] == 32
    assert math.isclose(IS.body_of(SPEC).groups(TRUE)["t_scale"], 100.0)


@pytest.mark.parametrize(
    "over,msg",
    [
        ({"unknown": ["alpha", "rho"]}, "unknown"),
        ({"unknown": ["alpha", "alpha"]}, "unknown"),
        ({"initial_guess": {"k": -1.0}}, "initial_guess.k"),
        ({"data": {"source": "file", "file": None}}, "data.file"),
        ({"data": {"source": "lab"}}, "data.source"),
        ({"pinn": {"weights": {"physics": 1.0}}}, "pinn.weights.physics"),
        ({"pinn": {"balancing": {"method": "ntk"}}}, "pinn.balancing.method"),
        ({"studies": {"pinn_inverse": {"variants": [{"name": "a", "mode": "three_stage"}]}}}, "mode: joint"),
        ({"studies": {"pinn_inverse": {"shapes": ["cube"]}}}, "studies.pinn_inverse.shapes"),
        ({"studies": {"pinn_inverse": {"shape_variant": "best"}}}, "shape_variant"),
    ],
)
def test_validation(over, msg):
    with pytest.raises(ValueError, match=msg):
        IS.load_spec(overrides=over)


def test_forward_mode_keeps_the_parameters_and_evaluates_a_grid():
    s = IS.load_spec(
        overrides={
            "measurement": {"sensors": {"n": 3}, "times": {"n": 6}},
            "pinn": {
                "collocation": {"n_pde": 64, "n_bc": 8},
                "adam": {"epochs": 6},
                "lbfgs": {"iterations": 2},
                "log_every": 3,
            },
        }
    )
    r = PI.train(
        IS.pinn_settings(s),
        IS.body_of(s),
        D.load(s),
        s["unknown"],
        TRUE,
        TRUE,
        "forward",
        0,
        {"r": [0.0, 0.02], "t": [5.0, 50.0, 100.0]},
    )
    assert all(r["estimates"][k] == pytest.approx(TRUE[k], rel=1e-14) for k in TRUE)
    assert r["theta_grid"].shape == (3, 2) and np.isfinite(r["residual"]["pde_rms"])


@pytest.mark.parametrize("shape", ["slab", "sphere"])
def test_sqrt_time_coordinate_residual(shape):
    """theta = 1 + c (a + b s) with c = sqrt(t_hat): residual = theta_c - 2 c (F Lap + F Q), Lap = 2 (n+1) c b."""
    body = B.Body(shape, 0.02, 80.0, 20.0, 1e5)
    m = PI.InversePINN(IS.pinn_settings(SPEC)["network"], body, 50.0, 9.0, ["alpha", "k", "h"], TRUE, TRUE, "sqrt")
    assert m.c_max == pytest.approx(3.0)
    m.net = Linear(0.3, -0.2)
    s, c = torch.tensor([0.1, 0.5, 0.9], dtype=DT), torch.tensor([0.2, 1.0, 2.0], dtype=DT)
    F = TRUE["alpha"] * 50.0 / 0.02**2
    Q = 1e5 * 0.02**2 / (TRUE["k"] * 60.0)
    expected = (0.3 - 0.2 * s) - 2 * c * (F * 2 * (body.n + 1) * c * (-0.2) + F * Q)
    assert torch.allclose(m.pde_residual(s, c).detach(), expected, rtol=1e-12)
    th = torch.tensor([0.04, 1.0, 4.0], dtype=DT)  # theta(s, t_hat) evaluates the network at c = sqrt(t_hat)
    assert torch.allclose(m.theta(s, th).detach(), 1 + torch.sqrt(th) * (0.3 - 0.2 * s), rtol=1e-14)


def test_staged_mode_freezes_the_parameters_during_warmup():
    s = IS.load_spec(
        overrides={
            "measurement": {"sensors": {"n": 3}, "times": {"n": 6}},
            "pinn": {
                "collocation": {"n_pde": 64, "n_bc": 8},
                "adam": {"epochs": 6},
                "lbfgs": {"iterations": 2},
                "staged": {"warmup_epochs": 6},
                "log_every": 3,
                "time_coordinate": "sqrt",
            },
        }
    )
    init = {k: float(s["initial_guess"][k]) for k in s["unknown"]}
    r = PI.train(IS.pinn_settings(s), IS.body_of(s), D.load(s), s["unknown"], init, TRUE, "staged", 0)
    warm = [h for h in r["history"] if h["stage"] == "warmup"]
    assert warm and all(h["k"] == pytest.approx(init["k"], rel=1e-14) for h in warm)
    assert {h["stage"] for h in r["history"]} == {"warmup", "joint", "joint_lbfgs"}
