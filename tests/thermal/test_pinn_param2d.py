"""Step 4.5a: parametric 2D problem (inputs x, y, log Re, log Pr, log Bi), the decoupled reference, the Phase 2 design
reuse, parametric user data, spec validation and report wording."""

import numpy as np
import pytest
import torch
from torch import nn

from aerosynapse.thermal.experiments import p4_5_parametric as E
from aerosynapse.thermal.experiments.p4_report import param_table
from aerosynapse.thermal.data import spec as DS
from aerosynapse.thermal.data.design import Dim, design_case
from aerosynapse.thermal.pinn import param2d as PP
from aerosynapse.thermal.pinn import problems2d as P2
from aerosynapse.thermal.pinn import spec as S
from aerosynapse.thermal.pinn import userdata as U
from aerosynapse.thermal.pinn.architectures import build_network
from aerosynapse.thermal.pinn.ref2d import solve_steady, solve_steady_decoupled
from aerosynapse.thermal.pinn.trainer import Trainer

DT = torch.float64
SPEC = S.load_spec(None, "smoke")
DIMS = [Dim("Re", 100.0, 1000.0), Dim("Pr", 0.7, 7.0), Dim("Bi", 0.1, 100.0)]
CAV = SPEC["problems2d"]["C_cavity"]


def fake_ref():
    x = (np.arange(6) + 0.5) / 6
    return {"x": x, "y": x, "nu_top": 1.0, **{k: np.zeros((6, 6)) for k in P2.FIELDS}}


def param_problem(**form):
    return PP.build_param_cavity(CAV, S.deep_merge(SPEC["formulation"], form), DIMS, {"Ri": 0.0}, 0.8)


class Fixed(nn.Module):
    """The parametric network evaluated at fixed parameter columns: a non-parametric (x, y) network."""

    def __init__(self, net, cols):
        super().__init__()
        self.net, self.cols = net, torch.as_tensor(cols, dtype=DT)

    def forward(self, X):
        return self.net(torch.cat([X, self.cols.expand(X.shape[0], -1)], 1))


@pytest.mark.parametrize("form", [{}, {"kind": "streamfunction", "hard_bc": True}])
def test_parametric_residuals_equal_fixed_parameter_problem(form):
    pp = param_problem(**form)
    vals = {"Re": 250.0, "Pr": 2.0, "Bi": 7.5, "Ri": 0.0}
    torch.manual_seed(0)
    net = build_network(SPEC["network"], list(pp.lo), list(pp.hi), pp.out_dim).to(DT)
    fixed = P2.build_cavity({**CAV, "parameters": vals}, S.deep_merge(SPEC["formulation"], form), reference=fake_ref())
    m = Fixed(net, pp.encode(vals))
    xy = torch.as_tensor(np.random.default_rng(0).uniform(size=(40, 2)), dtype=DT)
    Xp = torch.cat([xy, torch.as_tensor(pp.encode(vals), dtype=DT).expand(40, -1)], 1)
    rp = pp.residuals(pp.fields(net, Xp, 2))
    rf = fixed.residuals(fixed.fields(m, xy, 2))
    for k in rf:
        assert torch.allclose(rp[k], rf[k], rtol=1e-12, atol=1e-12), k
    s = np.linspace(0.05, 0.95, 9)
    wp = {
        w: torch.cat(
            [torch.as_tensor(P2.wall_points(w, s), dtype=DT), torch.as_tensor(pp.encode(vals), dtype=DT).expand(9, -1)],
            1,
        )
        for w in P2.WALLS
    }
    wf = {w: torch.as_tensor(P2.wall_points(w, s), dtype=DT) for w in P2.WALLS}
    a, b = pp.wall_residuals(net, wp), fixed.wall_residuals(m, wf)
    for w in P2.WALLS:
        for c in ("u", "v", "theta"):
            assert torch.allclose(a[w][c], b[w][c], rtol=1e-12, atol=1e-12), (w, c)
    assert pp.robin_param == {"bottom": ("Bi", 0.0)}


def test_sampling_box_scaling_and_hard_gauge():
    pp = param_problem()
    b = pp.sample(256, 0, 16, "sobol", 0, "cpu", DT)
    P = b["pde"].numpy()
    assert P.shape == (256, 5) and np.all((P[:, :2] >= 0) & (P[:, :2] <= 1))
    for j, d in enumerate(DIMS):
        v = np.exp(P[:, 2 + j])
        assert v.min() >= d.lo * (1 - 1e-12) and v.max() <= d.to_phys(0.8) * (1 + 1e-12)  # training box only
    assert pp.lo[2:] == tuple(np.log([d.lo for d in DIMS])) and pp.hi[2:] == tuple(np.log([d.hi for d in DIMS]))
    assert all(t.shape == (16, 5) for t in b["walls"].values())
    g = param_problem(pressure_gauge={"kind": "point", "mode": "hard", "x": 0.3, "y": 0.6, "value": 0.5})
    torch.manual_seed(1)
    net = build_network(SPEC["network"], list(g.lo), list(g.hi), g.out_dim).to(DT)
    X0 = torch.cat([torch.tensor([[0.3, 0.6]], dtype=DT).expand(5, 2), b["pde"][:5, 2:]], 1)
    assert torch.allclose(g.fields(net, X0, 0)["p"], torch.full((5,), 0.5, dtype=DT), atol=1e-14)


def test_evaluate_point_against_own_prediction_is_zero():
    pp = param_problem(kind="streamfunction", hard_bc=True)
    torch.manual_seed(2)
    net = build_network(SPEC["network"], list(pp.lo), list(pp.hi), pp.out_dim).to(DT)
    vals = {"Re": 300.0, "Pr": 1.5, "Bi": 2.0, "Ri": 0.0}
    x = (np.arange(10) + 0.5) / 10
    pr = P2._batched(pp, net, pp.grid_inputs(x, x, vals, next(net.parameters())), 0, P2.FIELDS)
    top = torch.as_tensor(
        np.concatenate([np.stack([x, np.ones(10)], 1), np.tile(pp.encode(vals), (10, 1))], 1), dtype=DT
    )
    nu = float(pp.fields(net, top, 1)["theta_y"].detach().mean())
    ref = {"x": x, "y": x, "nu_top": nu, **{k: pr[k].reshape(10, 10) for k in P2.FIELDS}}
    ev = PP.evaluate_point(pp, net, ref, vals)
    assert (
        ev["rel_l2_max"] < 1e-14 and ev["rel_l2_p"] < 1e-12 and ev["nu_top_rel_err"] < 1e-14 and ev["max_div"] < 1e-12
    )


def test_decoupled_reference_equals_coupled_for_ri_zero():
    cond = P2.cavity_conditions(CAV)
    par = {"Re": 100.0, "Pr": 0.71, "Bi": 1.0, "Ri": 0.0}
    a = solve_steady(par, cond, 12)
    b = solve_steady_decoupled(par, cond, 12, scheme="central")  # same scheme -> same discrete problem
    for k in ("u", "v", "p", "theta"):
        assert np.allclose(a[k], b[k], rtol=0, atol=1e-10), k
    assert abs(a["nu_top"] - b["nu_top"]) < 1e-10 and abs(b["energy_balance"]["residual"]) < 1e-12
    with pytest.raises(ValueError, match="needs Ri = 0"):
        solve_steady_decoupled({**par, "Ri": 1.0}, cond, 8)


def test_study_design_reproduces_phase2_points_and_user_ranges():
    d = E.study_design(SPEC)
    dspec = DS.load_spec(None, "smoke")
    ccfg = dspec["cases"]["caseC"]
    dims, _ = DS.case_dims(ccfg, "caseC")
    z = dspec["design"]
    des = design_case(
        "caseC",
        dims,
        ccfg["sizes"],
        dspec["seed"] + list(dspec["cases"]).index("caseC"),
        z["unit_train_limit"],
        z["method"],
        z["min_dist_factor"],
    )
    test_idx = [i for i, s in enumerate(des.split) if s in (2, 3)]
    assert [p["index"] for p in d["points"]] == test_idx
    assert all(abs(p["values"]["Re"] - des.params[p["index"], 0]) < 1e-12 for p in d["points"])
    assert d["leakage"]["extrap_all_outside_train_box"] and d["fixed"] == {"Ri": 0.0}
    user = S.deep_merge(
        SPEC,
        {
            "param_study": {
                "parameters": {"Re": {"lo": 50.0, "hi": 200.0}, "Pr": 1.0, "Bi": {"lo": 1.0, "hi": 10.0}},
                "design": {"sizes": {"train": 4, "val": 0, "interp": 2, "extrap": 1}},
            }
        },
    )
    du = E.study_design(user)
    assert [p["split"] for p in du["points"]] == ["interp", "interp", "extrap"] and du["fixed"] == {"Pr": 1.0}


def test_parametric_user_data_needs_parameter_columns(tmp_path):
    pp = param_problem()
    f = tmp_path / "d.csv"
    f.write_text("x,y,Re,Pr,Bi,theta\n0.5,0.5,200,1.5,3,0.2\n0.2,0.7,150,1.0,1,0.4\n", encoding="utf-8")
    ud = U.load_source(
        "d",
        {"file": str(f)},
        list(pp.input_names),
        ["u", "v", "p", "theta"],
        U.problem_params(pp, {}),
        list(pp.lo),
        list(pp.hi),
    )
    assert np.allclose(ud.inputs["logRe"], np.log([200, 150])) and ud.n == 2
    g = tmp_path / "e.csv"
    g.write_text("x,y,Pr,Bi,theta\n0.5,0.5,1.5,3,0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs a 'Re' column"):
        U.load_source(
            "e", {"file": str(g)}, list(pp.input_names), ["theta"], U.problem_params(pp, {}), list(pp.lo), list(pp.hi)
        )


def test_tiny_parametric_training_runs(tmp_path):
    s = S.deep_merge(
        SPEC,
        {
            "training": {"adam": {"epochs": 6}, "lbfgs": {"iterations": 2, "inner": 1}},
            "sampling": {"n_pde": 48, "n_bc": 6},
        },
    )
    pp = param_problem(kind="streamfunction", hard_bc=True)
    torch.manual_seed(0)
    net = build_network(s["network"], list(pp.lo), list(pp.hi), pp.out_dim).to(DT)
    assert Trainer(net, pp, s, tmp_path).run()["status"] == "ok"
    assert np.isfinite(P2.validation_loss(pp, net, 64, 8, 0)["val_loss"])


@pytest.mark.parametrize(
    "over,msg",
    [
        ({"param_study": {"case": "D_channel"}}, "param_study.case"),
        ({"param_study": {"parameters": {"Re": {"lo": 10.0, "hi": 5.0}, "Pr": 1.0}}}, "param_study.parameters.Re"),
        ({"param_study": {"parameters": {"Re": 100.0, "Pr": 1.0}}}, "sweep at least one"),
        ({"param_study": {"reference": {"scheme": "tvd_vanleer"}}}, "param_study.reference.scheme"),
        ({"param_study": {"variants": [{"name": "a", "kind": "global"}]}}, "param_study.variants\\[0\\].kind"),
        (
            {"param_study": {"variants": [{"name": "a", "kind": "parametric"}, {"name": "a", "kind": "per_point"}]}},
            "unique",
        ),
        ({"param_study": {"data": ["nope"]}}, "param_study.data"),
    ],
)
def test_spec_validation(over, msg):
    with pytest.raises(ValueError, match=msg):
        S.load_spec(None, "smoke", over)


def rec(variant, kind, point, split, err, unc=1e-2, nu=0.1, wall=100.0):
    return {
        "variant": variant,
        "kind": kind,
        "seed": 0,
        "point": point,
        "split": split,
        "Re": 100.0,
        "Pr": 1.0,
        "Bi": 1.0,
        "Pe": 100.0,
        "rel_l2_max": err,
        "rel_l2_p": err,
        "nu_top_rel_err": nu,
        "ref_unc_max": unc,
        "train_wall_s": wall,
        "n_params": 1000,
    }


def test_param_report_wording(tmp_path):
    design = {
        "source": "test",
        "dims": [["Re", 100.0, 1000.0, "log"]],
        "fixed": {},
        "train_box": {"Re": [100.0, 600.0]},
        "leakage": {"ok": True},
        "points": [
            {"index": i, "split": s, "values": {"Re": 100.0, "Pr": 1.0, "Bi": 1.0}}
            for i, s in ((0, "interp"), (1, "interp"), (2, "extrap"))
        ],
    }
    refs = [
        {
            "kind": "ref",
            "n_coarse": 8,
            "n_fine": 16,
            "coarse_fine_diff": {"u": 1e-2, "v": 1e-2, "theta": 1e-3, "p": 5e-2},
            "nu_top": 0.7,
        }
    ] * 3
    recs = [
        rec("par", "parametric", 0, "interp", 1e-1),
        rec("par", "parametric", 1, "interp", 1e-1),
        rec("par", "parametric", 2, "extrap", 5e-1),
        rec("pp", "per_point", 0, "interp", 2e-2, wall=50),
        rec("pp", "per_point", 1, "interp", 8e-2, wall=50),
        rec("pp", "per_point", 2, "extrap", 5e-3, wall=50),
    ]
    runs = [{"variant": "par", "seed": 0, "wall_s": 100.0}] + [
        {"variant": "pp", "seed": 0, "point": i, "wall_s": 50.0} for i in range(3)
    ]
    txt = "\n".join(param_table(recs, runs, design, refs, SPEC, tmp_path / "t.md", "t"))
    assert (
        "par: mean extrapolation error is 5.00x the mean interpolation error — a degradation beyond a factor 2" in txt
    )
    assert "Within 2x of the per-point networks at 1/3 points; ≥2x better at none; ≥2x worse at [0, 2]" in txt
    assert (
        "Target rel L2 < 0.01 met by: pp." in txt and "| pp | per_point |" in txt and "[2]" in txt
    )  # p2 error 5e-3 < unc 1e-2
    assert "Cost: one par training 100 s vs 3 pp trainings 150 s in total" in txt and "1 seed(s)" in txt
    assert "solution not captured" not in txt  # here most errors are below 0.3
    failing = [dict(r, rel_l2_max=0.8) for r in recs]
    ftxt = "\n".join(param_table(failing, runs, design, refs, SPEC, tmp_path / "f.md", "f"))
    assert "100% of all test-point errors are ≥ 0.3 (solution not captured)" in ftxt
