"""Step 3.2: PDE residual, IC and BC losses — zero for the exact solution, positive for wrong ones."""

import numpy as np
import pytest
import torch

from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import spec as S

torch.set_default_dtype(torch.float64)
SPEC = S.load_spec(None, "smoke")


class RefModel(torch.nn.Module):
    """Differentiable torch version of the exact solution as the 'network' (columns in problem.input_names order)."""

    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def forward(self, X):
        n = self.prob.name
        x = X[:, 0]
        c = self.prob.cond
        if n == "A_steady":
            Pe = self.prob.param * c["c"]
            u = c["theta_left"] + (c["theta_right"] - c["theta_left"]) * (
                torch.exp(Pe * (x - 1)) - torch.exp(-Pe * torch.ones_like(x))
            ) / (1 - np.exp(-Pe))
        elif n == "A_pulse":
            u = P.pulse_torch(x, X[:, 1], self.prob.param, c["pulse"])
        else:  # slab: first-eigenmode-free check below uses steady solution for Q-only problem
            raise NotImplementedError
        return u.unsqueeze(-1)


@pytest.mark.parametrize("name,val", [("A_steady", 5.0), ("A_pulse", 20.0)])
def test_exact_solution_has_zero_losses(name, val):
    prob = P.build_problem(name, SPEC["problems"][name], val)
    batch = prob.sample(200, 50, 40, "random", 0, "cpu", torch.float64)
    L = P.component_losses(RefModel(prob), prob, batch)
    for k, v in L.items():
        assert float(v) < 1e-14, (name, k, float(v))


class SteadySlab(torch.nn.Module):
    """Exact steady slab solution theta = Q/m^2 + A cosh(m x) (time-independent), as a time-dependent model."""

    def __init__(self, prob):
        super().__init__()
        c = prob.cond
        self.m, self.Q, self.Bi = c["m"], c["Q"], prob.param

    def forward(self, X):
        x = X[:, 0]
        m, Q, Bi = self.m, self.Q, self.Bi
        A = -Bi * Q / m**2 / (m * np.sinh(m) + Bi * np.cosh(m))
        return (Q / m**2 + A * torch.cosh(m * x)).unsqueeze(-1)


def test_slab_steady_solution_satisfies_pde_and_bcs_but_not_ic():
    prob = P.build_problem("B_slab", SPEC["problems"]["B_slab"], 3.0)
    L = P.component_losses(SteadySlab(prob), prob, prob.sample(100, 30, 30, "random", 0, "cpu", torch.float64))
    assert float(L["pde"]) < 1e-20 and float(L["bc_left"]) < 1e-20 and float(L["bc_right"]) < 1e-20
    assert float(L["ic"]) > 1e-3  # IC theta0 = 1 differs from the steady state


def test_wrong_model_gives_positive_losses_and_weights_combine():
    prob = P.build_problem("A_pulse", SPEC["problems"]["A_pulse"], 20.0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    L = P.component_losses(net, prob, prob.sample(64, 16, 16, "sobol", 1, "cpu", torch.float64))
    assert all(float(v) > 0 for v in L.values())
    w = {"pde": 2.0, "ic": 3.0, "bc": 5.0}
    tot = P.weighted_total(L, w)
    assert float(tot) == pytest.approx(
        2 * float(L["pde"]) + 3 * float(L["ic"]) + 5 * (float(L["bc_left"]) + float(L["bc_right"]))
    )


def test_references_satisfy_their_own_conditions():
    for name, val in (("A_steady", 50.0), ("A_pulse", 10.0), ("B_slab", 5.0)):
        prob = P.build_problem(name, SPEC["problems"][name], val)
        x = np.linspace(0, 1, 401)
        t = None if name == "A_steady" else np.array([0.05, 0.2, 0.5])
        ref = prob.reference(x, t, val)
        dx = prob.reference_dx(x, t, val)
        assert ref.shape == ((401,) if t is None else (3, 401)) and np.all(np.isfinite(dx))
        if name == "B_slab":
            assert np.abs(dx[:, 0]).max() < 1e-6  # symmetry
            assert np.abs(dx[:, -1] + val * ref[:, -1]).max() < 1e-5  # Robin
        if name == "A_steady":
            assert ref[0] == pytest.approx(0.0, abs=1e-12) and ref[-1] == pytest.approx(1.0)


def test_parametric_columns_sampling_and_ranges():
    pc = SPEC["problems"]["B_slab"]
    prob = P.build_problem("B_slab", {**pc, "parametric": True}, None)
    assert prob.input_names == ["x", "t", "logp"] and prob.prange == (0.1, 100.0)
    b = prob.sample(100, 20, 10, "sobol", 3, "cpu", torch.float64)
    assert set(b) == {"pde", "ic", "bc_left", "bc_right"} and set(b["pde"]) == {"x", "t", "logp"}
    p = torch.exp(b["pde"]["logp"])
    assert (
        p.min() >= 0.1 - 1e-9
        and p.max() <= 100 + 1e-9
        and torch.all(b["bc_right"]["x"] == 1)
        and torch.all(b["ic"]["t"] == 0)
    )
    L = P.component_losses(torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)), prob, b)
    assert all(torch.isfinite(v) for v in L.values())
    # same seed -> identical collocation points; different seed -> different
    b2 = prob.sample(100, 20, 10, "sobol", 3, "cpu", torch.float64)
    b3 = prob.sample(100, 20, 10, "sobol", 4, "cpu", torch.float64)
    assert torch.equal(b["pde"]["x"], b2["pde"]["x"]) and not torch.equal(b["pde"]["x"], b3["pde"]["x"])


def test_spec_layers_and_validation():
    s = S.load_spec(
        None,
        "smoke",
        S.parse_set(
            [
                "problems.A_steady.parameter.values=[2, 4]",
                "network.width=8",
                "problems.B_slab.enabled=false",
                "studies.loss_weights.enabled=false",
            ]
        ),
    )
    assert S.param_values(s["problems"]["A_steady"]["parameter"]) == [2.0, 4.0] and s["network"]["width"] == 8
    assert S.param_values({"name": "Pe", "lo": 1, "hi": 100, "n": 3}) == pytest.approx([1.0, 10.0, 100.0])
    assert S.load_spec(None, "full")["network"]["width"] == 64
    for over, needle in [
        ({"network": {"activation": "relu"}}, "network.activation"),
        ({"loss": {"weights": {"foo": 1}}}, "loss.weights.foo"),
        ({"training": {"adam": {"epochs": -1}}}, "training.adam.epochs"),
        ({"problems": {"A_steady": {"parameter": {"values": [1, -2]}}}}, "must be > 0"),
        ({"problems": {"A_pulse": {"T": 0}}}, "must be > 0"),
        ({"dtype": "float16"}, "dtype"),
        ({"problems": {"B_slab": {"parameter": {"name": "Pe"}}}}, "must be 'Bi'"),
    ]:
        with pytest.raises(ValueError, match="invalid setting at") as e:
            S.load_spec(None, "smoke", over)
        assert needle in str(e.value)
