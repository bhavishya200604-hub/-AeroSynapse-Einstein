"""Step 4.4: sampling and training strategies - RAD/RAR-D density and updates, causal weights (exact identity with the
paper's weighted loss), time-marching windows, composite model, curriculum, LHS, Trainer integration with exact resume,
spec validation and report wording."""

import numpy as np
import pytest
import torch

from aerosynapse.thermal.experiments.p4_report import strategy_table
from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import problems2d as P2
from aerosynapse.thermal.pinn import spec as S
from aerosynapse.thermal.pinn import strategies as ST
from aerosynapse.thermal.pinn.networks import MLP
from aerosynapse.thermal.pinn.trainer import Trainer

DT = torch.float64
SPEC = S.load_spec(None, "smoke")


def prob(name="B_slab", v=10.0):
    return P.build_problem(name, SPEC["problems"][name], v)


def net(p, seed=0):
    torch.manual_seed(seed)
    return MLP(p.lo, p.hi, 12, 2).to(DT)


def acfg(method, **kw):
    return {**SPEC["sampling"]["adaptive"], "method": method, **kw}


def test_density_formula_and_limits():
    s = ST.AdaptiveSampler(acfg("rad", k=2.0, c=0.5), 64, 1000, 0)
    eps = np.array([0.1, 1.0, 3.0, 0.0])
    p = s.density(eps)
    q = eps**2 / np.mean(eps**2) + 0.5
    assert np.allclose(p, q / q.sum(), rtol=1e-14)
    assert np.allclose(ST.AdaptiveSampler(acfg("rad", k=0.0, c=0.0), 64, 1000, 0).density(eps), 0.25)  # k = 0: uniform
    assert (ST.AdaptiveSampler(acfg("rad"), 64, 1000, 0).k, ST.AdaptiveSampler(acfg("rar_d"), 64, 1000, 0).c) == (
        1.0,
        0.0,
    )


def test_pde_residual_abs_matches_loss_residuals():
    p, m = prob(), net(prob())
    b = p.sample(40, 8, 4, "sobol", 0, "cpu", DT)
    r = P.residual_vectors(m, p, b)["pde"].detach().abs().numpy()
    assert np.allclose(ST.pde_residual_abs(m, p, b["pde"]), r, rtol=1e-14)
    q = P2.build_problem2d("C_mms", SPEC, S.deep_merge(SPEC["formulation"], {"kind": "streamfunction"}))
    torch.manual_seed(0)
    from aerosynapse.thermal.pinn.architectures import build_network

    m2 = build_network(SPEC["network"], list(q.lo), list(q.hi), q.out_dim).to(DT)
    X = q.sample(30, 0, 4, "sobol", 0, "cpu", DT)["pde"]
    v = q.residual_vectors(m2, {"pde": X, "walls": q.sample(30, 0, 4, "sobol", 0, "cpu", DT)["walls"]})
    ref = np.sqrt(sum(v[k].detach().numpy() ** 2 for k in v if k.startswith("pde_")))
    assert np.allclose(ST.pde_residual_abs(m2, q, X), ref, rtol=1e-12)


def test_sampler_updates_rad_and_rar_d():
    p, m = prob(), net(prob())
    b = p.sample(64, 8, 4, "sobol", 0, "cpu", DT)
    rad = ST.AdaptiveSampler(acfg("rad", period=10, n_candidates=256), 64, 101, 3)
    rad.initial(b["pde"])
    rad.update(m, p, 10, "cpu", DT, "sobol")
    assert ST._size(rad.points) == 64 and rad.n_updates == 1
    again = ST.AdaptiveSampler(acfg("rad", period=10, n_candidates=256), 64, 101, 3)
    again.initial(b["pde"])
    again.update(m, p, 10, "cpu", DT, "sobol")
    assert all(torch.equal(rad.points[k], again.points[k]) for k in rad.points)  # deterministic
    rar = ST.AdaptiveSampler(acfg("rar_d", period=10, n_candidates=256, start_fraction=0.5), 64, 101, 3)
    assert ST._size(rar.initial(b["pde"])) == 32 and rar.n_add == 4  # (64 - 32) / 10 updates
    for step in range(10, 120, 10):
        rar.update(m, p, step, "cpu", DT, "sobol")
    assert ST._size(rar.points) == 64  # grows to n_pde and stops


def test_causal_transform_equals_weighted_loss():
    cw = ST.CausalWeights({"n_chunks": 4, "eps": [0.5, 5.0], "delta": 0.99}, 0.0, 1.0)
    t = torch.tensor([0.05, 0.1, 0.3, 0.6, 0.65, 0.7, 0.95], dtype=DT)
    r = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.1, 0.2, 3.0], dtype=DT, requires_grad=True)
    out = cw.transform(r, t, step=None)
    chunks = [[0, 1], [2], [3, 4, 5], [6]]
    Lc = [float(np.mean([r[j].item() ** 2 for j in ch])) for ch in chunks]
    w = [np.exp(-0.5 * sum(Lc[:i])) for i in range(4)]
    assert float((out**2).mean()) == pytest.approx(np.mean([wi * li for wi, li in zip(w, Lc, strict=True)]), rel=1e-12)
    g = torch.autograd.grad((out**2).mean(), r)[0]  # weights carry no gradient: d/dr_j = 2 r_j scale_j / N
    scale = (out.detach() / r.detach()) ** 2
    assert torch.allclose(g, 2 * r.detach() * scale / r.numel(), rtol=1e-12)
    small = ST.CausalWeights({"n_chunks": 4, "eps": [0.5, 5.0], "delta": 0.99}, 0.0, 1.0)
    small.transform(1e-6 * torch.ones(7, dtype=DT), t, step=1)
    assert small.eps == 5.0 and small.eps_history == [(1, 5.0)]


def test_window_problem_and_composite_model():
    base = prob("A_pulse", 10.0)
    prev = net(base, 1)
    wp = ST.WindowProblem(base, 0.25, 0.5, prev)
    b = wp.sample(50, 10, 5, "sobol", 0, "cpu", DT)
    assert float(b["pde"]["t"].min()) >= 0.25 and float(b["pde"]["t"].max()) <= 0.5
    assert torch.all(b["ic"]["t"] == 0.25) and float(b["bc_left"]["t"].min()) >= 0.25
    X = torch.stack([b["ic"]["x"], b["ic"]["t"]], 1)
    assert torch.allclose(wp.ic_target(b["ic"]), prev(X).squeeze(-1).detach())
    first = ST.WindowProblem(base, 0.0, 0.25, None)
    c0 = first.sample(10, 10, 5, "sobol", 0, "cpu", DT)["ic"]
    assert torch.allclose(first.ic_target(c0), base.ic_target(c0))

    class Const(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = torch.nn.Parameter(torch.tensor([float(k)], dtype=DT))

        def forward(self, X):
            return self.k * torch.ones(X.shape[0], 1, dtype=DT)

    cm = ST.CompositeModel([Const(0), Const(1), Const(2)], [0.0, 0.1, 0.2, 0.3], t_index=1)
    t = torch.tensor([0.0, 0.05, 0.1, 0.15, 0.2, 0.3], dtype=DT)
    out = cm(torch.stack([torch.zeros_like(t), t], 1)).squeeze(-1)
    assert out.tolist() == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]


def test_curriculum_values_and_budget_split():
    v = ST.curriculum_values(100.0, 0.1, 3)
    assert v[0] == pytest.approx(10.0) and v[1] == pytest.approx(np.sqrt(1000.0)) and v[-1] == 100.0
    assert ST.curriculum_values(5.0, 0.1, 1) == [5.0] and ST.split_budget(10, 3) == [4, 3, 3]


def test_lhs_sampling_1d_and_2d():
    p = prob()
    b = p.sample(20, 8, 4, "lhs", 1, "cpu", DT)
    x = np.sort(b["pde"]["x"].numpy())
    assert np.all(np.diff(np.floor(x * 20)) == 1)  # one point per stratum (Latin hypercube)
    q = P2.build_problem2d("C_mms", SPEC, SPEC["formulation"])
    X = q.sample(16, 0, 4, "lhs", 1, "cpu", DT)["pde"].numpy()
    assert X.shape == (16, 2) and np.all((X >= 0) & (X <= 1))


def tiny(**over):
    s = S.deep_merge(
        SPEC,
        {
            "training": {
                "adam": {"epochs": 24},
                "lbfgs": {"iterations": 4, "inner": 2},
                "checkpoint_every": 7,
                "log_every": 1,
            },
            "sampling": {"n_pde": 40, "n_ic": 10, "n_bc": 5},
        },
    )
    return S.deep_merge(s, over)


@pytest.mark.parametrize(
    "over",
    [
        {"sampling": {"adaptive": {"method": "rad", "period": 6, "n_candidates": 128}}},
        {"sampling": {"adaptive": {"method": "rar_d", "period": 6, "n_candidates": 128}}},
        {"training": {"causal": {"enabled": True, "n_chunks": 4}}},
    ],
)
def test_trainer_strategies_run_and_resume_exactly(tmp_path, over):
    s = tiny(**over)
    p = prob("A_pulse", 10.0)
    a = net(p)
    ta = Trainer(a, p, s, tmp_path / "a")
    assert ta.run()["status"] == "ok"
    if ta.sampler is not None:
        assert ta.sampler.n_updates >= 3 and ST._size(ta.sampler.points) == 40
    if ta.causal is not None:
        lb = [r for r in ta.history if r["phase"] == "lbfgs"]
        assert ta.causal.frozen and "causal_eps" in ta.history[0] and len({r["causal_eps"] for r in lb}) == 1
    b = net(p)
    assert Trainer(b, p, s, tmp_path / "b").run(stop_after_adam_epoch=13)["status"] == "stopped"
    c = net(p, seed=5)
    Trainer(c, p, s, tmp_path / "b").run(resume=True)
    for pa, pc in zip(a.parameters(), c.parameters(), strict=True):
        assert torch.equal(pa, pc)


@pytest.mark.parametrize(
    "over,msg",
    [
        ({"sampling": {"adaptive": {"method": "rad"}, "resample_every": 10}}, "resample_every"),
        ({"sampling": {"adaptive": {"method": "magic"}}}, "sampling.adaptive.method"),
        ({"sampling": {"method": "grid"}}, "sampling.method"),
        ({"training": {"causal": {"n_chunks": 0}}}, "training.causal.n_chunks"),
        ({"training": {"time_marching": {"windows": 0}}}, "training.time_marching.windows"),
        ({"training": {"curriculum": {"start_fraction": 1.5}}}, "training.curriculum.start_fraction"),
        ({"strategy_study": {"strategies": [{"name": "a"}, {"name": "a"}]}}, "unique"),
        (
            {"strategy_study": {"strategies": [{"name": "a", "optimizer": {}}]}},
            "strategy_study.strategies\\[0\\].optimizer",
        ),
        ({"strategy_study": {"cases": [{"problem": "Q"}]}}, "strategy_study.cases\\[0\\].problem"),
    ],
)
def test_spec_validation(over, msg):
    with pytest.raises(ValueError, match=msg):
        S.load_spec(None, "smoke", over)


def test_conflicts_checked_at_load_for_active_settings_and_at_run_for_the_study():
    with pytest.raises(ValueError, match="cannot be combined with loss.balancing.method=self_adaptive"):
        S.load_spec(
            None,
            "smoke",
            {"sampling": {"adaptive": {"method": "rad"}}, "loss": {"balancing": {"method": "self_adaptive"}}},
        )
    with pytest.raises(ValueError, match="enable at most one"):
        S.load_spec(None, "smoke", {"training": {"causal": {"enabled": True}, "curriculum": {"enabled": True}}})
    s = S.load_spec(None, "smoke", {"sampling": {"resample_every": 7}})  # unrelated runs may resample ...
    with pytest.raises(ValueError, match="strategies\\[4\\].sampling.resample_every"):
        S.check_strategy_study_runnable(s)  # ... but the default study's rad strategy then cannot run
    S.check_strategy_study_runnable(S.load_spec(None, "smoke"))


def rec(strategy, err, case="A_pulse 10", dim=1, wall=10.0, **kw):
    return {
        "case": case,
        "problem": case.split()[0],
        "dim": dim,
        "strategy": strategy,
        "seed": 0,
        "err": err,
        "rel_l2_p": err,
        "val_loss": err,
        "wall_s": wall,
        "status": "ok",
        **kw,
    }


def test_strategy_report_wording(tmp_path):
    recs = [
        rec("sobol", 1e-1),
        rec("rad", 2e-2, n_pde_final=512, k=1.0, c=1.0, sampler_updates=5),
        rec("rar_d", 3e-1, n_pde_final=300, k=2.0, c=0.0, sampler_updates=5),
        rec("causal", 8e-2, causal_eps_final=1.0, causal_min_w=0.5),
        rec("sobol", 1e-1, case="A_steady 100"),
        rec("rad", 1e-1, case="A_steady 100", n_pde_final=512, k=1.0, c=1.0, sampler_updates=5),
    ]
    skipped = [
        {
            "case": "A_steady 100",
            "strategy": "causal",
            "reason": "causal weighting / time-marching need a transient problem",
        }
    ]
    txt = "\n".join(strategy_table(recs, skipped, {}, SPEC, tmp_path / "t.md", "t"))
    assert (
        "≥2x lower: rad (5.0x)" in txt
        and "≥2x HIGHER: rar_d (3.0x)" in txt
        and "within 2x (no effect claimed): causal" in txt
    )
    assert "Not applicable (not run): causal (causal weighting / time-marching need a transient problem)" in txt
    assert "Final PDE point count below the baseline's" in txt and "rar_d (300 < 512)" in txt
    assert "| causal | 1.25 | n/a | 0 | 0 |" in txt  # 1e-1 / 8e-2; not applicable on A_steady
    assert "rad (2 cases)" in txt and "1 seed(s)" in txt
