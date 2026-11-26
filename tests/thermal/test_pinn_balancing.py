"""Step 4.3: loss-balancing suite - each method's update against a hand computation of its formula, resume
exactness for every method, frozen weights during L-BFGS, 2D compatibility and spec validation."""

import math

import numpy as np
import pytest
import torch

from aerosynapse.thermal.pinn import balancing as B
from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import problems2d as P2
from aerosynapse.thermal.pinn import spec as S
from aerosynapse.thermal.pinn.networks import MLP
from aerosynapse.thermal.pinn.trainer import Trainer

DT = torch.float64
SPEC = S.load_spec(None, "smoke")
GW = {"pde": 1.0, "ic": 1.0, "bc": 1.0, "gauge": 1.0}


def cfg(method, **over):
    return S.deep_merge(SPEC["loss"]["balancing"], {"method": method, method: over} if over else {"method": method})


def setup(name="B_slab", val=10.0, seed=0):
    prob = P.build_problem(name, SPEC["problems"][name], val)
    torch.manual_seed(seed)
    net = MLP(prob.lo, prob.hi, 16, 2).to(DT)
    b = prob.sample(64, 16, 8, "sobol", 0, "cpu", DT)
    vecs = P.residual_vectors(net, prob, b)
    comp = {k: (v**2).mean() for k, v in vecs.items()}
    return prob, net, comp, vecs


def flat(loss, net):
    g = torch.autograd.grad(loss, list(net.parameters()), retain_graph=True, allow_unused=True)
    return torch.cat(
        [(x if x is not None else torch.zeros_like(p)).reshape(-1) for x, p in zip(g, net.parameters(), strict=True)]
    )


def test_residual_vectors_reproduce_component_losses():
    prob, net, comp, vecs = setup()
    ref = P.component_losses(net, prob, prob.sample(64, 16, 8, "sobol", 0, "cpu", DT))
    assert set(ref) == set(comp) and all(torch.equal(ref[k], comp[k]) for k in ref)


def test_lr_annealing_matches_formula():
    _, net, comp, vecs = setup()
    bal = B.build_balancer(cfg("lr_annealing", alpha=1.0, every=1), GW, 0)
    bal.prepare(net, comp, vecs, 0)
    gmax = flat(comp["pde"], net).abs().max()
    for k in ("ic", "bc_left", "bc_right"):
        expect = float(gmax / flat(comp[k], net).abs().mean())
        assert bal.w[k] == pytest.approx(expect, rel=1e-12)
    assert bal.w["pde"] == 1.0
    bal2 = B.build_balancer(cfg("lr_annealing", alpha=0.9, every=1), GW, 0)
    bal2.prepare(net, comp, vecs, 0)
    assert bal2.w["ic"] == pytest.approx(0.1 + 0.9 * bal.w["ic"], rel=1e-12)  # lam = (1-a) lam + a lam_hat, lam0 = 1


def test_gradnorm_one_step_and_renormalisation():
    _, net, comp, vecs = setup()
    bal = B.build_balancer(cfg("gradnorm", alpha=1.5, lr=0.01, every=1), GW, 0)
    bal.prepare(net, comp, vecs, 0)
    names = list(comp)
    W = B.last_layer_params(net)
    gn = np.array(
        [
            float(
                torch.sqrt(
                    sum(
                        (g**2).sum()
                        for g in torch.autograd.grad(comp[k], W, retain_graph=True, allow_unused=True)
                        if g is not None
                    )
                )
            )
            for k in names
        ]
    )
    target = gn.mean() * 1.0  # at t = 0 every loss ratio is 1
    w = 1.0 - 0.01 * np.sign(gn - target)  # first Adam step = lr * sign(grad) (up to eps)
    w = w * len(names) / w.sum()
    assert np.allclose([bal.w[k] for k in names], w, rtol=0, atol=1e-6)
    assert sum(bal.w.values()) == pytest.approx(len(names), rel=1e-12)


@pytest.mark.parametrize("trace", ["mean", "sum"])
def test_ntk_trace_weights_match_full_computation(trace):
    _, net, comp, vecs = setup()
    bal = B.build_balancer(cfg("ntk", n_points=1000, every=1, trace=trace), GW, 0)  # n_points >= all points: exact
    bal.prepare(net, comp, vecs, 0)
    tr = {}
    for k, v in vecs.items():
        sq = [float((flat(v[i], net) ** 2).sum()) for i in range(v.numel())]
        tr[k] = float(np.mean(sq)) * (v.numel() if trace == "sum" else 1)
    tot = sum(tr.values())
    for k in vecs:
        assert bal.w[k] == pytest.approx(tot / tr[k], rel=1e-10)


def test_softadapt_formula_and_beta_zero():
    _, net, comp, vecs = setup()
    bal = B.build_balancer(cfg("softadapt", beta=0.0, every=1), GW, 0)
    for step in range(5):
        bal.prepare(net, {k: v * (1 + step) for k, v in comp.items()}, vecs, step)
    assert all(w == pytest.approx(1.0) for w in bal.w.values())  # beta = 0: equal weights
    bal = B.build_balancer(cfg("softadapt", beta=2.0, every=1, window=3, normalized=True), GW, 0)
    hist = [[1.0, 2.0, 3.0, 4.0], [0.5, 2.5, 3.0, 1.0], [0.25, 3.0, 3.0, 0.5]]
    names = list(comp)
    for step, L in enumerate(hist):
        bal.prepare(net, {k: torch.tensor(x, dtype=DT) for k, x in zip(names, L, strict=True)}, vecs, step)
    H = np.array(hist)
    s = (H[-1] - H[0]) / 2
    s = s / (np.abs(s).sum() + 1e-8)
    a = np.exp(2.0 * (s - s.max()))
    a = 4 * a / (a.sum() + 1e-8)
    assert np.allclose([bal.w[k] for k in names], a, rtol=1e-12)


def test_relobralo_formula_and_infinite_temperature():
    names = ["pde", "ic", "bc_left", "bc_right"]
    L0, L1 = [1.0, 2.0, 4.0, 8.0], [0.5, 1.0, 4.0, 1.0]
    bal = B.build_balancer(cfg("relobralo", alpha=0.9, rho=1.0, temperature=0.5, every=1), GW, 0)
    _, net, _, vecs = setup()
    for step, L in enumerate((L0, L1)):
        bal.prepare(net, {k: torch.tensor(x, dtype=DT) for k, x in zip(names, L, strict=True)}, vecs, step)
    z = np.array(L1) / (0.5 * np.array(L0))  # bal(t, t-1) with t-1 = step 0
    bal_prev = 4 * np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    w0 = 0.9 * 1.0 + 0.1 * np.ones(4)  # step 0: both balances uniform -> weights stay 1
    expect = 0.9 * w0 + 0.1 * bal_prev  # rho = 1: history carries the previous weights
    assert np.allclose([bal.w[k] for k in names], expect, rtol=1e-12)
    hot = B.build_balancer(cfg("relobralo", temperature=1e12, every=1), GW, 0)
    for step, L in enumerate((L0, L1, L0)):
        hot.prepare(net, {k: torch.tensor(x, dtype=DT) for k, x in zip(names, L, strict=True)}, vecs, step)
    assert all(w == pytest.approx(1.0, abs=1e-9) for w in hot.w.values())


def test_uncertainty_weights_converge_to_inverse_loss():
    """Stationary point of exp(-s)/2 L + s/2 is s = log L, i.e. effective weight exp(-s)/2 = 1/(2L)."""
    _, net, _, vecs = setup()
    L = {"pde": 0.5, "ic": 2.0, "bc_left": 0.1, "bc_right": 4.0}
    bal = B.build_balancer(cfg("uncertainty", lr=0.05), GW, 0)
    for step in range(3000):
        comp = {k: torch.tensor(v, dtype=DT) for k, v in L.items()}
        bal.prepare(net, comp, vecs, step)
        bal.total(comp, vecs).backward()
        bal.after_backward(step)
    for i, k in enumerate(bal.names):
        assert float(bal.s[i]) == pytest.approx(math.log(L[k]), abs=1e-2)


def test_self_adaptive_ascent_and_fixed_points():
    _, net, comp, vecs = setup()
    bal = B.build_balancer(cfg("self_adaptive", lr=0.01, init=1.0, mask="square"), GW, 0)
    bal.prepare(net, comp, vecs, 0)
    bal.total(comp, vecs).backward()
    g = {k: bal.lam[k].grad.detach().clone() for k in vecs}
    for k, v in vecs.items():  # d/dlam_i mean(lam^2 r^2) = 2 lam_i r_i^2 / N >= 0
        assert torch.allclose(g[k], 2 * v.detach() ** 2 / v.numel(), rtol=1e-12, atol=0)
    bal.after_backward(0)
    for k in vecs:  # first Adam ASCENT step: lam += lr * g / (|g| + eps)
        assert torch.allclose(bal.lam[k].detach(), 1.0 + 0.01 * g[k] / (g[k].abs() + 1e-8), rtol=0, atol=1e-12)
    short = {k: v[:-1] if v.numel() > 1 else v for k, v in vecs.items()}
    with pytest.raises(RuntimeError, match="collocation set must stay fixed"):
        bal.prepare(net, {k: (v**2).mean() for k, v in short.items()}, short, 1)


def small_spec(method, **over):
    s = S.deep_merge(
        SPEC,
        {
            "training": {
                "adam": {"epochs": 12},
                "lbfgs": {"iterations": 4, "inner": 2},
                "checkpoint_every": 5,
                "log_every": 1,
                "eval_every": 100,
            },
            "sampling": {"n_pde": 48, "n_ic": 16, "n_bc": 8},
            "network": {"width": 12, "depth": 2},
            "loss": {"balancing": {"method": method}},
        },
    )
    for m in ("lr_annealing", "gradnorm", "ntk", "softadapt"):
        s["loss"]["balancing"][m]["every"] = 3
    s["loss"]["balancing"]["ntk"]["n_points"] = 4
    return S.deep_merge(s, over)


@pytest.mark.parametrize("method", [m for m in B.METHODS if m != "fixed"])
def test_resume_is_exact_and_weights_frozen_in_lbfgs(tmp_path, method):
    s = small_spec(method)
    prob = P.build_problem("B_slab", s["problems"]["B_slab"], 10.0)

    def net():
        torch.manual_seed(0)
        return MLP(prob.lo, prob.hi, 12, 2).to(DT)

    a = net()
    ta = Trainer(a, prob, s, tmp_path / "a")
    out = ta.run()
    assert out["status"] == "ok" and ta.balancer.n_updates > 0
    adam_rows = [r for r in ta.history if r["phase"] == "adam"]
    lb_rows = [r for r in ta.history if r["phase"] == "lbfgs"]
    wk = [k for k in adam_rows[-1] if k.startswith("w_")]
    assert wk and all(r[k] == adam_rows[-1][k] for r in lb_rows for k in wk)  # frozen during L-BFGS
    b = net()
    assert Trainer(b, prob, s, tmp_path / "b").run(stop_after_adam_epoch=7)["status"] == "stopped"
    c = net()
    tc = Trainer(c, prob, s, tmp_path / "b")
    tc.run(resume=True)
    for pa, pc in zip(a.parameters(), c.parameters(), strict=True):
        assert torch.equal(pa, pc), method
    assert ta.balancer.effective() == tc.balancer.effective()


def test_balancer_on_2d_problem(tmp_path):
    s = small_spec("lr_annealing", sampling={"n_pde": 32, "n_bc": 6})
    prob = P2.build_problem2d("C_mms", s, s["formulation"])
    torch.manual_seed(0)
    from aerosynapse.thermal.pinn.architectures import build_network

    m = build_network(s["network"], list(prob.lo), list(prob.hi), prob.out_dim).to(DT)
    tr = Trainer(m, prob, s, tmp_path)
    assert tr.run()["status"] == "ok"
    eff = tr.balancer.effective()
    assert {"pde_mom_x", "pde_cont", "bc_u", "bc_theta"} <= set(eff) and eff["pde_mom_x"] == 1.0
    assert any(v != 1.0 for k, v in eff.items() if k.startswith("bc_"))


@pytest.mark.parametrize(
    "over,msg",
    [
        ({"loss": {"balancing": {"method": "magic"}}}, "loss.balancing.method"),
        ({"loss": {"balancing": {"method": "self_adaptive"}}, "sampling": {"resample_every": 10}}, "resample_every=0"),
        ({"loss": {"balancing": {"weight_bounds": [1.0, 0.5]}}}, "weight_bounds"),
        ({"loss": {"balancing": {"relobralo": {"rho": 1.5}}}}, "relobralo.rho"),
        ({"loss": {"balancing": {"ntk": {"trace": "max"}}}}, "ntk.trace"),
        ({"balance_study": {"cases": [{"problem": "Z"}]}}, "balance_study.cases\\[0\\].problem"),
        ({"balance_study": {"methods": ["fixed", "fixed"]}}, "labels must be unique"),
        ({"balance_study": {"cases": [{"problem": "C_mms", "formulation": {"kind": "x"}}]}}, "formulation.kind"),
    ],
)
def test_spec_validation(over, msg):
    with pytest.raises(ValueError, match=msg):
        S.load_spec(None, "smoke", over)


def test_resampling_allowed_unless_self_adaptive_is_used():
    s = S.load_spec(None, "smoke", {"sampling": {"resample_every": 10}})  # other studies may resample
    with pytest.raises(ValueError, match="resample_every=0"):
        S.check_balance_study_runnable(s)  # ... but the 4.3 study with self_adaptive cannot
    S.check_balance_study_runnable(S.load_spec(None, "smoke"))


def test_example_config_is_runnable():
    ex = S.load_spec(S.ROOT / "configs" / "pinn" / "examples" / "balancing_custom.yaml", "smoke")
    S.check_balance_study_runnable(ex)
    labels = []
    for entry in ex["balance_study"]["methods"]:
        label, b = S.balancing_variant(ex, entry)
        bal = B.build_balancer(b, ex["loss"]["weights"], 0)
        assert (bal is None) == (b["method"] == "fixed")
        labels.append(label)
    assert "relobralo_T001" in labels and ex["loss"]["weights"]["gauge"] == 1.0
