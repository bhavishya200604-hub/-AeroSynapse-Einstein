"""Step 4.2: 2D residuals, formulations, hard constraints, gauge, normalisation, reference and spec validation."""

import itertools

import numpy as np
import pytest
import torch
from torch import nn

from aerosynapse.thermal.pinn import problems2d as P2
from aerosynapse.thermal.pinn import spec as S
from aerosynapse.thermal.pinn.architectures import build_network
from aerosynapse.thermal.pinn.ref2d import cavity_reference
from aerosynapse.thermal.pinn.trainer import Trainer

SPEC = S.load_spec(None, "smoke")
DT = torch.float64


def form(**kw):
    return S.deep_merge(SPEC["formulation"], kw)


def mms(**kw):
    return P2.build_problem2d("C_mms", SPEC, form(**kw))


def fake_ref():
    x = (np.arange(8) + 0.5) / 8
    z = np.zeros((8, 8))
    return {"x": x, "y": x, "u": z, "v": z, "p": z, "theta": z, "nu_top": 1.0}


def cavity(**kw):
    return P2.build_problem2d("C_cavity", SPEC, form(**kw), reference=fake_ref())


def net(prob, seed=0):
    torch.manual_seed(seed)
    return build_network(SPEC["network"], list(prob.lo), list(prob.hi), prob.out_dim).to(DT)


class Exact(nn.Module):
    def __init__(self, fns):
        super().__init__()
        self.fns = fns
        self.dummy = nn.Parameter(torch.zeros(1, dtype=DT))

    def forward(self, X):
        return torch.stack([f(X[:, 0], X[:, 1]) for f in self.fns], -1) + 0 * self.dummy


def pts(n=300, seed=1):
    return torch.as_tensor(np.random.default_rng(seed).uniform(size=(n, 2)), dtype=DT)


@pytest.mark.parametrize("kind", ["primitive", "streamfunction"])
def test_exact_solution_has_zero_residuals(kind):
    """Forcing, residual code, wall data and normals agree with the symbolic manufactured solution."""
    prob = mms(kind=kind)
    fn = prob.truth["exact_fn"]
    names = ("u", "v", "p", "theta") if kind == "primitive" else ("psi", "p", "theta")
    m = Exact([fn[k] for k in names])
    r = prob.residuals(prob.fields(m, pts(), order=2))
    for k, v in r.items():
        assert float(v.detach().abs().max()) < 1e-11, k
    s = np.linspace(0, 1, 50)
    walls = {w: torch.as_tensor(P2.wall_points(w, s), dtype=DT) for w in P2.WALLS}
    for w, comp in prob.wall_residuals(m, walls).items():
        for c, v in comp.items():
            assert float(v.detach().abs().max()) < 1e-12, (w, c)
    ev = P2.evaluate2d(prob, m, n_wall=50)
    assert ev["rel_l2_max"] < 1e-13 and ev["rel_l2_p"] < 1e-13 and ev["max_div"] < 1e-11
    assert P2.validation_loss(prob, m, 256, 32, 0)["val_loss"] < 1e-20


def test_transfinite_interpolant_matches_data_on_every_wall_subset():
    f = lambda x, y: torch.sin(x + 2 * y) + x * y  # noqa: E731
    s = np.linspace(0, 1, 41)
    for k in range(1, 5):
        for ws in itertools.combinations(P2.WALLS, k):
            G, D = P2.transfinite({w: f for w in ws}), P2.distance(ws)
            for w in ws:
                X = torch.as_tensor(P2.wall_points(w, s), dtype=DT)
                assert float((G(X[:, 0], X[:, 1]) - f(X[:, 0], X[:, 1])).detach().abs().max()) < 1e-14, ws
                assert float(D(X[:, 0], X[:, 1]).detach().abs().max()) == 0.0
            Xi = pts(200)
            assert float(D(Xi[:, 0], Xi[:, 1]).detach().min()) > 0


@pytest.mark.parametrize("builder", [mms, cavity])
def test_hard_primitive_constraints_exact_for_random_network(builder):
    prob = builder(hard_bc=True)
    s = np.random.default_rng(3).uniform(size=60)
    walls = {w: torch.as_tensor(P2.wall_points(w, s), dtype=DT) for w in P2.WALLS}
    wr = prob.wall_residuals(net(prob), walls)
    for w in P2.WALLS:
        assert float(wr[w]["u"].detach().abs().max()) < 1e-13 and float(wr[w]["v"].detach().abs().max()) < 1e-13
        if w in prob.th_dir:
            assert float(wr[w]["theta"].detach().abs().max()) < 1e-13
        else:
            assert float(wr[w]["theta"].detach().abs().max()) > 1e-6  # Robin/Neumann walls remain loss terms
    comp = prob.loss_components(net(prob), prob.sample(64, 0, 16, "sobol", 0, "cpu", DT))
    assert "bc_u" not in comp and "bc_v" not in comp and "pde_cont" in comp


def test_streamfunction_divergence_free_and_hard_normal_velocity():
    prob = mms(kind="streamfunction", hard_bc=True)
    m = net(prob)
    d = prob.fields(m, pts(), order=1)
    assert float((d["u_x"] + d["v_y"]).detach().abs().max()) < 1e-12
    s = np.random.default_rng(4).uniform(size=50)
    walls = {w: torch.as_tensor(P2.wall_points(w, s), dtype=DT) for w in P2.WALLS}
    wr = prob.wall_residuals(m, walls)
    fn = prob.truth["exact_fn"]
    for w in P2.WALLS:
        nrm = "v" if w in ("bottom", "top") else "u"
        X = walls[w]
        target = fn[nrm](X[:, 0], X[:, 1])
        assert float((wr[w][nrm] + target).detach().abs().max()) < 1e-13  # predicted normal velocity is exactly 0
    comp = prob.loss_components(m, prob.sample(64, 0, 16, "sobol", 0, "cpu", DT))
    assert "pde_cont" not in comp and {"bc_u", "bc_v", "bc_theta"} <= set(comp)


def test_hard_point_gauge_and_soft_gauge_terms():
    prob = mms(pressure_gauge={"kind": "point", "mode": "hard", "x": 0.3, "y": 0.7, "value": 0.25})
    p = prob.fields(net(prob), torch.tensor([[0.3, 0.7]], dtype=DT), order=0)["p"]
    assert abs(float(p[0].detach()) - 0.25) < 1e-15
    b = prob.sample(64, 0, 16, "sobol", 0, "cpu", DT)
    assert "gauge" not in prob.loss_components(net(prob), b)
    for kind in ("point", "mean"):
        q = mms(pressure_gauge={"kind": kind, "mode": "soft"})
        assert float(q.loss_components(net(q), b)["gauge"].detach()) >= 0


def test_init_rms_normalisation_gives_unit_initial_residuals():
    prob = mms(normalise={"kind": "init_rms"})
    m = net(prob)
    b = prob.sample(128, 0, 16, "sobol", 0, "cpu", DT)
    sc = prob.fit_normalisation(m, b)
    comp = prob.loss_components(m, b)
    for k in ("mom_x", "mom_y", "cont", "energy"):
        assert sc[k] > 0 and abs(float(comp["pde_" + k].detach()) - 1.0) < 1e-12


def test_infeasible_hard_constraints_and_bad_expressions_are_rejected():
    uni = S.deep_merge(
        SPEC,
        {
            "problems2d": {
                "C_cavity": {
                    "conditions": {
                        "velocity": {"top": {"profile": "uniform"}},
                        "thermal": {w: {"type": "neumann", "value": 0.0} for w in P2.WALLS},
                    }
                }
            }
        },
    )
    with pytest.raises(ValueError, match="discontinuous at the"):
        P2.build_problem2d("C_cavity", uni, form(hard_bc=True), reference=fake_ref())
    P2.build_problem2d("C_cavity", uni, form(hard_bc=False), reference=fake_ref())  # soft: allowed
    bad = S.deep_merge(SPEC, {"problems2d": {"C_mms": {"exact": {"psi": "x*y"}}}})
    with pytest.raises(ValueError, match="psi = 0 on all walls"):
        P2.build_problem2d("C_mms", bad, form(kind="streamfunction", hard_bc=True))
    tdep = S.deep_merge(SPEC, {"problems2d": {"C_mms": {"exact": {"theta": "x*t"}}}})
    with pytest.raises(ValueError, match="only x and y"):
        P2.build_problem2d("C_mms", tdep, form())


@pytest.mark.parametrize(
    "over,msg",
    [
        ({"formulation": {"kind": "vorticity"}}, "formulation.kind"),
        ({"formulation": {"pressure_gauge": {"kind": "mean", "mode": "hard"}}}, "pressure_gauge.mode"),
        ({"formulation_study": {"variants": [{"name": "a", "hard": True}]}}, "variants\\[0\\].hard"),
        ({"formulation_study": {"variants": [{"name": "a"}, {"name": "a"}]}}, "unique"),
        ({"formulation_study": {"cases": ["C_x"]}}, "formulation_study.cases"),
        ({"problems2d": {"C_mms": {"parameters": {"Re": -1.0}}}}, "C_mms.parameters.Re"),
        ({"problems2d": {"C_mms": {"thermal_bc": {"left": [0.0, 0.0]}}}}, "thermal_bc.left"),
        ({"formulation": {"normalise": {"kind": "manual", "scales": {"cont": 0.0}}}}, "scales.cont"),
        ({"loss": {"weights": {"gauge": -1.0}}}, "loss.weights.gauge"),
    ],
)
def test_spec_validation_messages(over, msg):
    with pytest.raises(ValueError, match=msg):
        S.load_spec(None, "smoke", over)


def test_cavity_reference_converges_and_respects_walls():
    cond = P2.cavity_conditions(SPEC["problems2d"]["C_cavity"])
    r = cavity_reference({"Re": 100.0, "Pr": 0.71, "Bi": 1.0, "Ri": 0.0}, cond, 8, 2)
    assert r["newton"]["fine"]["newton_residual"] < 1e-10 and r["n_fine"] == 16
    assert r["newton"]["fine"]["max_div"] < 1e-11
    assert 0.0 < r["theta"].min() and r["theta"].max() < 1.0  # top wall theta = 1, cooled bottom: interior in (0, 1)
    assert set(r["coarse_fine_diff"]) == {"u", "v", "theta", "p", "nu_top"}


def test_trainer_2d_runs_and_resumes_exactly(tmp_path):
    s = S.deep_merge(
        SPEC,
        {
            "training": {
                "adam": {"epochs": 6},
                "lbfgs": {"iterations": 4, "inner": 2},
                "checkpoint_every": 3,
                "log_every": 1,
                "eval_every": 3,
            },
            "sampling": {"n_pde": 64, "n_bc": 8},
        },
    )
    prob = mms(kind="streamfunction", hard_bc=True, normalise={"kind": "init_rms"})
    a = net(prob)
    tr = Trainer(a, prob, s, tmp_path / "a")
    prob.fit_normalisation(a, tr.batch(0))
    out = tr.run()
    assert out["status"] == "ok" and np.isfinite(out["final"]["total"])
    b = net(prob)
    tb = Trainer(b, prob, s, tmp_path / "b")
    assert tb.run(stop_after_adam_epoch=3)["status"] == "stopped"
    b2 = net(prob, seed=99)
    Trainer(b2, prob, s, tmp_path / "b").run(resume=True)
    for pa, pb in zip(a.parameters(), b2.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_hard_point_gauge_leaves_training_unchanged(tmp_path):
    """p enters the residuals only through grad p, and p - N_p(X0) differs from N_p by a constant: the hard point
    gauge must not change any parameter update (Step 4.2 finding, verified here directly)."""
    s = S.deep_merge(
        SPEC,
        {
            "training": {"adam": {"epochs": 5}, "lbfgs": {"iterations": 2, "inner": 1}},
            "sampling": {"n_pde": 64, "n_bc": 8},
        },
    )
    nets = []
    for g in ({"kind": "none"}, {"kind": "point", "mode": "hard", "x": 0.2, "y": 0.9, "value": 3.0}):
        prob = mms(pressure_gauge=g)
        m = net(prob)
        Trainer(m, prob, s, tmp_path / g["kind"]).run()
        nets.append(m)
    for pa, pb in zip(nets[0].parameters(), nets[1].parameters(), strict=True):
        assert torch.allclose(pa, pb, rtol=0, atol=1e-13)


def test_example_config_builds_every_case_and_variant():
    """The shipped example must be feasible for every (case, variant), incl. hard-constraint corner checks."""
    ex = S.load_spec(S.ROOT / "configs" / "pinn" / "examples" / "formulations_custom.yaml", "smoke")
    for case in ex["formulation_study"]["cases"]:
        for entry in ex["formulation_study"]["variants"]:
            _, f = S.formulation_variant(ex, entry)
            P2.build_problem2d(case, ex, f, reference=fake_ref() if case == "C_cavity" else None)
