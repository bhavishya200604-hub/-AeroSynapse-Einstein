"""Step 4.5b: Case D channel problems - geometry and sampling, residuals against the exact Poiseuille solution at
per-point Re, Robin walls with per-point Bi, obstacle no-slip / isothermal faces, the drag integral, the references,
the evaluation identity, spec validation and report wording."""

import numpy as np
import pytest
import torch
from torch import nn

from aerosynapse.thermal.experiments import p4_5b_channel as E
from aerosynapse.thermal.experiments.p4_report import channel_table
from aerosynapse.thermal.data.design import Dim
from aerosynapse.thermal.pinn import channel2d as CH
from aerosynapse.thermal.pinn import problems2d as P2
from aerosynapse.thermal.pinn import spec as S
from aerosynapse.thermal.pinn.architectures import build_network
from aerosynapse.thermal.pinn.trainer import Trainer

DT = torch.float64
SPEC = S.load_spec(None, "smoke")
FORM = S.deep_merge(SPEC["formulation"], {"kind": "primitive", "hard_bc": False})
HARD = S.deep_merge(SPEC["formulation"], {"kind": "streamfunction", "hard_bc": True})
DIMS = [Dim("Re", 10.0, 100.0), Dim("Pr", 0.7, 2.0), Dim("Bi", 0.5, 50.0)]


def graetz(sweep=True):
    return CH.build_channel(
        "D_graetz",
        SPEC["problems2d"]["D_graetz"],
        FORM,
        DIMS if sweep else [],
        {"Ri": 0.0} if sweep else {"Re": 30.0, "Pr": 1.0, "Bi": 5.0, "Ri": 0.0},
        0.8,
    )


def obstacle(sweep=True):
    return CH.build_channel(
        "D_obstacle",
        SPEC["problems2d"]["D_obstacle"],
        FORM,
        DIMS if sweep else [],
        {"Ri": 0.0} if sweep else {"Re": 40.0, "Pr": 0.7, "Bi": 1.0, "Ri": 0.0},
        0.8,
    )


class Poiseuille(nn.Module):
    """u = 6y(1-y), v = 0, p = -12 x / Re (exact steady channel flow), theta = c; inputs (x, y, logRe, logPr, logBi)."""

    def __init__(self, c=0.3):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(c, dtype=DT))

    def forward(self, X):
        x, y, Re = X[:, 0], X[:, 1], torch.exp(X[:, 2])
        return torch.stack([6 * y * (1 - y), 0 * x, -12 * x / Re, self.c + 0 * x], -1)


def test_geometry_bounds_sampling_and_segments():
    g, o = graetz(), obstacle()
    assert g.lo[:2] == (0.0, 0.0) and g.hi[:2] == (8.0, 1.0) and o.hi[:2] == (6.0, 2.0)
    assert g.segments == ["inlet", "outlet", "bottom", "top"] and len(o.segments) == 8
    b = o.sample(400, 0, 10, "sobol", 0, "cpu", DT)
    X = b["pde"].numpy()
    assert X.shape == (400, 5) and not o.inside_obstacle(X[:, 0], X[:, 1]).any()
    ob = o.geom["obstacle"]
    L = b["walls"]["obs_left"].numpy()
    assert np.allclose(L[:, 0], ob["x0"]) and L[:, 1].min() >= ob["y0"] and L[:, 1].max() <= ob["y1"]
    T = b["walls"]["obs_top"].numpy()
    assert np.allclose(T[:, 1], ob["y1"]) and T[:, 0].min() >= ob["x0"]


def test_exact_poiseuille_gives_zero_flow_residuals_at_every_re():
    g = graetz()
    m = Poiseuille()
    b = g.sample(200, 0, 20, "sobol", 1, "cpu", DT)
    r = g.residuals(g.fields(m, b["pde"], order=2))
    for k in ("mom_x", "mom_y", "cont"):
        assert float(r[k].detach().abs().max()) < 1e-12, k
    w = g.wall_residuals(m, b["walls"])
    for seg in ("inlet", "outlet", "bottom", "top"):
        for c in ("u", "v"):
            assert float(w[seg][c].detach().abs().max()) < 1e-12, (seg, c)
    Bi = torch.exp(b["walls"]["bottom"][:, 4])
    assert torch.allclose(w["bottom"]["theta"], Bi * (0.3 - 1.0), rtol=1e-12)  # Robin with per-point Bi, theta_inf = 1
    assert torch.allclose(w["inlet"]["theta"], torch.full_like(w["inlet"]["theta"], 0.3))


def test_obstacle_faces_and_drag_integral():
    o = obstacle(sweep=False)

    class LinearP(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(3.0, dtype=DT))

        def forward(self, X):
            z = 0 * X[:, 0]
            return torch.stack([z, z, -self.a * X[:, 0], z + 1.0], -1)

    m = LinearP()
    assert abs(CH.drag_coefficient(o, m, None) - 3.0) < 1e-12  # F_x = a (x1 - x0) h; C_D = 2 F_x / D = a
    b = o.sample(64, 0, 8, "sobol", 0, "cpu", DT)
    w = o.wall_residuals(m, b["walls"])
    for f in ("obs_left", "obs_right", "obs_bottom", "obs_top"):
        assert float(w[f]["theta"].detach().abs().max()) < 1e-14  # theta = 1 = theta_obs


def test_references_flow_and_obstacle():
    g = SPEC["problems2d"]["D_graetz"]
    r = CH.channel_reference({"Re": 30.0, "Pr": 1.0, "Bi": 5.0, "Ri": 0.0}, g, 4, 2)
    Y = r["y"][:, None]
    assert (
        np.abs(r["u"] - 6 * Y * (1 - Y)).max() < 0.05 and r["q_bottom"] < 0
    )  # heated through the wall: theta falls away from it (inward derivative < 0)
    assert set(r["coarse_fine_diff"]) == {"u", "v", "theta", "p", "q_bottom"} and "decoupled" in r["kind"]
    o = SPEC["problems2d"]["D_obstacle"]
    ro = CH.solve_channel({"Re": 30.0, "Pr": 0.7, "Bi": 1.0, "Ri": 0.0}, o, 8)
    assert ro["solid"].sum() == 4 * 4 and ro["C_D"] > 0 and np.allclose(ro["theta"][ro["solid"]], 1.0)
    assert ro["newton_residual"] < 1e-9 and "coupled" in ro["kind"]


def test_evaluation_against_own_prediction_is_zero():
    o = obstacle()
    torch.manual_seed(0)
    net = build_network(SPEC["network"], list(o.lo), list(o.hi), o.out_dim).to(DT)
    vals = {"Re": 40.0, "Pr": 0.7, "Bi": 1.0, "Ri": 0.0}
    x, y = (np.arange(24) + 0.5) / 4, (np.arange(8) + 0.5) / 4
    X, Y = np.meshgrid(x, y)
    pr = P2._batched(o, net, o.inputs(np.stack([X.ravel(), Y.ravel()], 1), vals, next(net.parameters())), 0, P2.FIELDS)
    xb = np.stack([x, np.zeros_like(x)], 1)
    q = float(o.fields(net, o.inputs(xb, vals, next(net.parameters())), 1)["theta_y"].detach().mean())
    ref = {
        "x": x,
        "y": y,
        "solid": o.inside_obstacle(X, Y),
        "q_bottom": q,
        "C_D": CH.drag_coefficient(o, net, vals),
        **{k: pr[k].reshape(X.shape) for k in P2.FIELDS},
    }
    ev = CH.evaluate_channel(o, net, ref, vals)
    assert (
        ev["rel_l2_max"] < 1e-14
        and ev["rel_l2_p"] < 1e-12
        and ev["q_bottom_rel_err"] < 1e-14
        and ev["C_D_rel_err"] < 1e-14
    )


def test_tiny_parametric_channel_training_and_designs(tmp_path):
    s = S.deep_merge(
        SPEC,
        {
            "training": {"adam": {"epochs": 5}, "lbfgs": {"iterations": 2, "inner": 1}},
            "sampling": {"n_pde": 48, "n_bc": 6},
        },
    )
    o = obstacle()
    torch.manual_seed(0)
    net = build_network(s["network"], list(o.lo), list(o.hi), o.out_dim).to(DT)
    assert Trainer(net, o, s, tmp_path).run()["status"] == "ok"
    assert np.isfinite(P2.validation_loss(o, net, 64, 6, 0)["val_loss"])
    d = E.case_design(SPEC, "D_obstacle")
    assert [p["split"] for p in d["points"]] == ["interp"] * 3 + ["extrap"] * 2 and d["leakage"]["all_ok"]


@pytest.mark.parametrize(
    "over,msg",
    [
        (
            {"problems2d": {"D_obstacle": {"geometry": {"obstacle": {"x0": 1.51}}}}},
            "must lie on the reference grid lines",
        ),
        ({"problems2d": {"D_obstacle": {"geometry": {"obstacle": {"x1": 9.0}}}}}, "D_obstacle.geometry.obstacle"),
        ({"channel_study": {"formulation": {"hard_bc": True}}}, "need kind: streamfunction"),
        (
            {"problems2d": {"D_obstacle": {"geometry": {"obstacle": {"psi": "zero"}}}}},
            "auto, symmetric, learned or soft",
        ),
        (
            {
                "channel_study": {"formulation": {"kind": "streamfunction", "hard_bc": True}},
                "problems2d": {"D_obstacle": {"geometry": {"obstacle": {"y0": 0.5, "y1": 1.0, "psi": "symmetric"}}}},
            },
            "centred obstacle",
        ),
        ({"problems2d": {"D_graetz": {"parameters": {"Gr": 1.0}}}}, "D_graetz.parameters.Gr"),
        ({"problems2d": {"D_graetz": {"thermal": {"bottom": {"type": "robin", "Bi": "Bk"}}}}}, "names 'Bk'"),
        ({"channel_study": {"cases": ["C_cavity"]}}, "channel_study.cases"),
    ],
)
def test_spec_validation(over, msg):
    with pytest.raises(ValueError, match=msg):
        S.load_spec(None, "smoke", over)


def test_channel_report_wording(tmp_path):
    designs = {
        "D_graetz": {
            "dims": [["Re", 10.0, 100.0, "log"]],
            "fixed": {},
            "train_box": {"Re": [10.0, 63.0]},
            "leakage": {"all_ok": True},
            "points": [
                {"index": i, "split": s, "values": {"Re": 20.0, "Pr": 1.0, "Bi": 1.0}}
                for i, s in ((0, "interp"), (1, "extrap"))
            ],
        }
    }
    refs = [
        {
            "case": "D_graetz",
            "kind": "ref",
            "n_coarse": 8,
            "n_fine": 16,
            "q_bottom": 1.0,
            "C_D": None,
            "coarse_fine_diff": {"u": 1e-3, "v": 1e-3, "theta": 1e-3, "p": 1e-3, "q_bottom": 1e-3},
        }
    ] * 2

    def rec(v, k, pt, sp, e):
        return {
            "case": "D_graetz",
            "variant": v,
            "kind": k,
            "seed": 0,
            "point": pt,
            "split": sp,
            "Re": 20.0,
            "Pr": 1.0,
            "Bi": 1.0,
            "Pe": 20.0,
            "rel_l2_max": e,
            "rel_l2_p": e,
            "q_bottom_rel_err": e,
            "ref_unc_max": 1e-3,
            "train_wall_s": 10.0,
        }

    recs = [
        rec("par", "parametric", 0, "interp", 0.05),
        rec("par", "parametric", 1, "extrap", 0.2),
        rec("pp", "per_point", 0, "interp", 0.008),
        rec("pp", "per_point", 1, "extrap", 0.02),
    ]
    runs = [
        {"case": "D_graetz", "variant": "par", "seed": 0, "wall_s": 10.0},
        {"case": "D_graetz", "variant": "pp", "seed": 0, "point": 0, "wall_s": 10.0},
        {"case": "D_graetz", "variant": "pp", "seed": 0, "point": 1, "wall_s": 10.0},
    ]
    txt = "\n".join(channel_table(recs, runs, designs, refs, SPEC, tmp_path / "t.md", "t"))
    assert "Target rel L2 < 0.01 met by: pp." in txt and "par: mean extrapolation error 4.00x" in txt
    assert (
        "≥2x worse at [0, 1]" in txt
        and "solution not captured" not in txt
        and "Cost: one par training 10 s vs 2 pp trainings 20 s" in txt
    )


def test_flow_rates_conserved_and_decaying():
    g = graetz()
    vals = {"Re": 30.0, "Pr": 1.0, "Bi": 5.0, "Ri": 0.0}
    f = CH.flow_rates(g, Poiseuille(), vals)
    assert abs(f["flux_inlet"] - 1.0) < 1e-3 and abs(f["flux_ratio"] - 1.0) < 1e-12  # Q = int 6y(1-y) dy = 1

    class Decay(nn.Module):  # u = 6y(1-y)(1 - x/16): Q_out / Q_in = 1 - 8/16 = 0.5 on the 8-long channel
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(1.0, dtype=DT))

        def forward(self, X):
            x, y = X[:, 0], X[:, 1]
            z = 0 * x
            return torch.stack([self.a * 6 * y * (1 - y) * (1 - x / 16), z, z, z], -1)

    assert abs(CH.flow_rates(g, Decay(), vals)["flux_ratio"] - 0.5) < 1e-12


def hard(name, psi="auto", geom=None, inlet=None):
    pc = S.deep_merge(SPEC["problems2d"][name], {"inlet": inlet or {}, "geometry": geom or {}})
    if pc["geometry"].get("obstacle"):
        pc["geometry"]["obstacle"]["psi"] = psi
    return CH.build_channel(name, pc, HARD, DIMS, {"Ri": 0.0}, 0.8)


def random_net(prob, seed=0):
    torch.manual_seed(seed)
    return build_network(SPEC["network"], list(prob.lo), list(prob.hi), prob.out_dim).to(DT)


@pytest.mark.parametrize("name,psi", [("D_graetz", "auto"), ("D_obstacle", "symmetric"), ("D_obstacle", "learned")])
def test_hard_psi_exact_on_inlet_walls_and_obstacle(name, psi):
    """An untrained network already satisfies psi = psi_in on the inlet, 0 / Q on the walls, psi_obs on the obstacle:
    u = u_in at the inlet, no normal velocity on walls and obstacle faces, Q_out / Q_in = 1."""
    o = hard(name, psi)
    assert o.psi_obs_mode == (psi if o.obs else None)
    net = random_net(o)
    b = o.sample(64, 0, 32, "sobol", 3, "cpu", DT)
    w = o.wall_residuals(net, b["walls"])
    assert float(w["inlet"]["u"].detach().abs().max()) < 1e-12
    for seg in ("bottom", "top"):
        assert float(w[seg]["v"].detach().abs().max()) < 1e-12
        psi_w = o.transform(net, b["walls"][seg])["psi"].detach()
        assert torch.allclose(psi_w, torch.full_like(psi_w, 0.0 if seg == "bottom" else o.Q), atol=1e-12)
    assert abs(o.Q - (1.0 if name == "D_graetz" else 2.0)) < 1e-14  # parabolic mean 1 (amplitude 1.5 over Ly 2)
    if o.obs:
        normal = {"obs_left": "u", "obs_right": "u", "obs_bottom": "v", "obs_top": "v"}
        ps = torch.cat([o.transform(net, b["walls"][f])["psi"].detach() for f in normal])
        if psi == "symmetric":
            assert torch.allclose(ps, torch.full_like(ps, 0.5 * o.Q), atol=1e-12)
        for f, c in normal.items():
            assert float(w[f][c].detach().abs().max()) < 1e-11, f
    vals = {"Re": 30.0, "Pr": 1.0, "Bi": 2.0, "Ri": 0.0}
    # exact Q at every section follows from psi = 0 / Q on the whole walls (above); here only the 64-point midpoint rule
    assert abs(CH.flow_rates(o, net, vals)["flux_ratio"] - 1.0) < 1e-3
    x, y = np.meshgrid((np.arange(48) + 0.5) / 8, (np.arange(16) + 0.5) / 8)  # whole box incl. the obstacle
    F = o.fields(net, o.inputs(np.stack([x.ravel(), y.ravel()], 1), vals, next(net.parameters())), order=2)
    assert all(bool(torch.isfinite(F[k]).all()) for k in ("u", "v", "u_xx", "v_yy", "theta"))


@pytest.mark.parametrize(
    "vel",
    [
        {"profile": "parabolic", "amplitude": 1.5},
        {"profile": "uniform", "amplitude": 0.7},
        {"profile": "cosh", "amplitude": 1.0, "C": 8.0},
        {"profile": "expr", "amplitude": 1.0, "expr": "sin(pi * s)"},
    ],
)
def test_psi_inlet_is_the_integral_of_the_profile(vel):
    o = hard("D_graetz", inlet={"velocity": vel})
    y = torch.linspace(0.0, 1.0, 41, dtype=DT, requires_grad=True)
    ps = o.psi_inlet(y)
    (dpsi,) = torch.autograd.grad(ps.sum(), y)
    assert abs(float(ps[0])) <= 1e-15  # zero up to libm round-off (torch.sinh vs math.sinh differ by 1 ulp on Linux)
    assert torch.allclose(dpsi, P2.profile_torch(vel, y.detach() / o.Ly), atol=1e-12)


def test_obstacle_psi_auto_and_symmetry_of_the_reference():
    assert hard("D_obstacle").psi_obs_mode == "symmetric"
    off = hard("D_obstacle", "auto", geom={"obstacle": {"y0": 0.5, "y1": 1.0}})
    assert off.psi_obs_mode == "learned"
    with pytest.raises(ValueError, match="centred obstacle"):
        hard("D_obstacle", "symmetric", geom={"obstacle": {"y0": 0.5, "y1": 1.0}})
    with pytest.raises(ValueError, match="streamfunction"):
        CH.build_channel("D_graetz", SPEC["problems2d"]["D_graetz"], {**FORM, "hard_bc": True}, DIMS, {"Ri": 0.0}, 0.8)
    # verification of psi_obs = Q/2: the reference flow rate between the bottom wall and the obstacle is Q/2
    o = SPEC["problems2d"]["D_obstacle"]
    r = CH.solve_channel({"Re": 40.0, "Pr": 0.7, "Bi": 1.0, "Ri": 0.0}, o, 8)
    ob = o["geometry"]["obstacle"]
    i = int(np.flatnonzero((r["x"] > ob["x0"]) & (r["x"] < ob["x1"]))[0])
    below = r["y"] < ob["y0"]
    dy = r["y"][1] - r["y"][0]
    q_col = float(r["u"][~r["solid"][:, i], i].sum() * dy)  # discrete Q (midpoint inlet profile: 2.0039 on 16 cells)
    assert abs(q_col - 2.0) < 5e-3 and abs(float(r["u"][below, i].sum() * dy) - 0.5 * q_col) < 1e-8


def test_tiny_training_with_hard_psi(tmp_path):
    s = S.deep_merge(
        SPEC,
        {
            "training": {"adam": {"epochs": 5}, "lbfgs": {"iterations": 2, "inner": 1}},
            "sampling": {"n_pde": 48, "n_bc": 6},
        },
    )
    o = hard("D_obstacle", "learned")
    net = random_net(o)
    assert Trainer(net, o, s, tmp_path).run()["status"] == "ok"
    assert np.isfinite(P2.validation_loss(o, net, 64, 6, 0)["val_loss"])
    assert abs(CH.flow_rates(o, net, {"Re": 30.0, "Pr": 0.7, "Bi": 1.0, "Ri": 0.0})["flux_ratio"] - 1.0) < 1e-3


def test_hard_psi_with_soft_obstacle():
    """psi exact on inlet and walls only: inlet u, wall v and Q at every section exact; the obstacle faces are NOT
    constrained (their normal velocity stays a loss term)."""
    o = hard("D_obstacle", "soft")
    assert o.psi_obs_mode == "soft"
    net = random_net(o)
    b = o.sample(64, 0, 32, "sobol", 3, "cpu", DT)
    w = o.wall_residuals(net, b["walls"])
    assert float(w["inlet"]["u"].detach().abs().max()) < 1e-12
    assert max(float(w[s]["v"].detach().abs().max()) for s in ("bottom", "top")) < 1e-12
    assert float(w["obs_left"]["u"].detach().abs().max()) > 1e-3
    vals = {"Re": 30.0, "Pr": 0.7, "Bi": 1.0, "Ri": 0.0}
    assert abs(CH.flow_rates(o, net, vals)["flux_ratio"] - 1.0) < 1e-3


def test_rad_on_obstacle_channel_keeps_points_in_the_fluid(tmp_path):
    """RAD (4.4) on a channel: redrawn PDE points come from the channel's own sampler (never inside the obstacle)."""
    s = S.deep_merge(
        SPEC,
        {
            "training": {"adam": {"epochs": 6}, "lbfgs": {"iterations": 1, "inner": 1}},
            "sampling": {"n_pde": 48, "n_bc": 6, "adaptive": {"method": "rad", "period": 2, "n_candidates": 96}},
        },
    )
    o = hard("D_obstacle", "soft")
    tr = Trainer(random_net(o), o, s, tmp_path)
    assert tr.run()["status"] == "ok" and tr.sampler.n_updates >= 2
    P = tr.sampler.points.numpy()
    assert P.shape == (48, 5) and not o.inside_obstacle(P[:, 0], P[:, 1]).any()
