"""Step 4.6: snapshot data (selection, nesting, noise, held-out mask), data terms, held-out evaluation, mode weights,
train design points, spec validation, report wording and a tiny end-to-end hybrid run."""

import numpy as np
import pytest
import torch

from aerosynapse.thermal.experiments import p4_6_hybrid as E
from aerosynapse.thermal.experiments.p4_5b_channel import case_design
from aerosynapse.thermal.experiments.p4_report import hybrid_table
from aerosynapse.thermal.pinn import channel2d as CH
from aerosynapse.thermal.pinn import hybrid as H
from aerosynapse.thermal.pinn import problems2d as P2
from aerosynapse.thermal.pinn import spec as S
from aerosynapse.thermal.pinn.architectures import build_network

DT = torch.float64
SPEC = S.load_spec(None, "smoke")
HARD = S.deep_merge(SPEC["formulation"], {"kind": "streamfunction", "hard_bc": True})
VALS = {"Re": 30.0, "Pr": 1.0, "Bi": 2.0, "Ri": 0.0}


def own_reference(prob, net, nx=24, ny=8):
    """A 'reference' that is the network's own prediction on a cell-centred grid (obstacle cells marked solid)."""
    x, y = (np.arange(nx) + 0.5) * prob.Lx / nx, (np.arange(ny) + 0.5) * prob.Ly / ny
    X, Y = np.meshgrid(x, y)
    vals = None if not prob.sweep else VALS
    pr = P2._batched(
        prob, net, prob.inputs(np.stack([X.ravel(), Y.ravel()], 1), vals, next(net.parameters())), 0, P2.FIELDS
    )
    xb = np.stack([x, np.zeros_like(x)], 1)
    q = float(prob.fields(net, prob.inputs(xb, vals, next(net.parameters())), 1)["theta_y"].detach().mean())
    return {
        "x": x,
        "y": y,
        "solid": prob.inside_obstacle(X, Y),
        "q_bottom": q,
        **{k: pr[k].reshape(X.shape) for k in P2.FIELDS},
    }


def problem(name="D_obstacle", sweep=False):
    pc = S.deep_merge(
        SPEC["problems2d"][name], {"geometry": {"obstacle": {"psi": "soft"}}} if name == "D_obstacle" else {}
    )
    dims = case_design(SPEC, name)["dims"] if sweep else []
    prob = CH.build_channel(name, pc, HARD, dims, {"Ri": 0.0} if sweep else VALS, 0.8)
    torch.manual_seed(0)
    return prob, build_network(SPEC["network"], list(prob.lo), list(prob.hi), prob.out_dim).to(DT)


@pytest.mark.parametrize("placement", ["random", "sobol"])
def test_snapshot_points_distinct_fluid_nested(placement):
    prob, net = problem()
    ref = own_reference(prob, net)
    a = H.snapshot_points(ref, 40, ["u", "theta"], placement, 0.0, 3)
    b = H.snapshot_points(ref, 10, ["u", "theta"], placement, 0.0, 3)
    assert a["mask"].sum() == 40 and not (a["mask"] & ref["solid"]).any()
    assert len({tuple(p) for p in a["xy"]}) == 40
    assert np.array_equal(a["xy"][:10], b["xy"]) and (b["mask"] <= a["mask"]).all()  # nested in n
    ix, iy = np.searchsorted(ref["x"], a["xy"][:, 0]), np.searchsorted(ref["y"], a["xy"][:, 1])
    for f in ("u", "theta"):  # noise 0: exact solver values at the chosen cells, in the order of xy
        assert np.array_equal(a["targets"][f], np.asarray(ref[f])[iy, ix])


def test_snapshot_noise_and_errors():
    prob, net = problem("D_graetz")
    ref = own_reference(prob, net, 64, 16)
    fluid = ~ref["solid"]
    clean = H.snapshot_points(ref, 800, ["theta"], "random", 0.0, 1)
    noisy = H.snapshot_points(ref, 800, ["theta"], "random", 0.1, 1)
    rms = float(np.sqrt(np.mean(np.asarray(ref["theta"])[fluid] ** 2)))
    dev = noisy["targets"]["theta"] - clean["targets"]["theta"]
    assert abs(dev.std() / (0.1 * rms) - 1) < 0.1
    small = H.snapshot_points(ref, 100, ["theta"], "random", 0.1, 1)
    assert np.allclose(small["targets"]["theta"], noisy["targets"]["theta"][:100])  # noise nested too
    with pytest.raises(ValueError, match="fluid cells"):
        H.snapshot_points(ref, 10**6, ["u"], "random")
    with pytest.raises(ValueError, match="random or sobol"):
        H.snapshot_points(ref, 5, ["u"], "grid")


@pytest.mark.parametrize("sweep", [False, True])
def test_snapshot_term_zero_for_the_generating_network(sweep):
    prob, net = problem("D_obstacle", sweep)
    ref = own_reference(prob, net)
    snaps = [
        (H.snapshot_points(ref, 30, ["u", "v", "theta"], "random", 0.0, k), VALS if sweep else None)
        for k in range(1 + sweep)
    ]
    term = H.snapshot_term("snap", prob, snaps, next(net.parameters()))
    assert term.X.shape == (30 * (1 + sweep), prob.lo.__len__())
    vec = term.vectors(net)
    assert set(vec) == {"data_snap_u", "data_snap_v", "data_snap_theta"}
    assert max(float(v.detach().abs().max()) for v in vec.values()) < 1e-12


def test_evaluate_channel_exclude_mask():
    prob, net = problem()
    ref = own_reference(prob, net)
    snap = H.snapshot_points(ref, 25, ["u"], "random", 0.0, 0)
    bad = {**ref, "u": np.where(snap["mask"], 99.0, ref["u"]), "theta": np.where(snap["mask"], -5.0, ref["theta"])}
    assert CH.evaluate_channel(prob, net, bad, None)["rel_l2_max"] > 0.1  # the bad cells count without the mask
    assert CH.evaluate_channel(prob, net, bad, None, exclude=snap["mask"])["rel_l2_max"] < 1e-12


def test_mode_weights():
    w = SPEC["loss"]["weights"]
    assert H.mode_weights(w, "hybrid") == w and H.mode_weights(w, "data_free") == w
    d = H.mode_weights(w, "data_only")
    assert d["pde"] == d["bc"] == d["ic"] == d["gauge"] == 0.0 and d["data"] == w["data"]
    with pytest.raises(ValueError, match="one of"):
        H.mode_weights(w, "physics")


def test_train_design_points_are_the_unused_train_split():
    for c in S.CHANNEL:
        tp = E.train_design_points(SPEC, c)
        d = case_design(SPEC, c)
        assert len(tp) == SPEC["channel_study"]["design"]["sizes"]["train"]
        assert all(p["split"] == "train" for p in tp)
        assert not {p["index"] for p in tp} & {p["index"] for p in d["points"]}
        for p in tp:  # inside the training box of the design
            for k, (lo, hi) in d["train_box"].items():
                assert lo * (1 - 1e-12) <= p["values"][k] <= hi * (1 + 1e-12)


@pytest.mark.parametrize(
    "over,msg",
    [
        ({"hybrid_study": {"modes": ["hybrid", "physics"]}}, "hybrid_study.modes"),
        ({"hybrid_study": {"data": {"placement": "grid"}}}, "random or sobol"),
        ({"hybrid_study": {"data": {"fields": ["T"]}}}, "hybrid_study.data.fields"),
        ({"hybrid_study": {"data": {"noise": -0.1}}}, "must be >= 0"),
        ({"hybrid_study": {"parametric": {"n_snapshots": 99}}}, "sizes.train"),
        ({"hybrid_study": {"per_point": {"n_data": [16, 0]}}}, "positive integers"),
        ({"hybrid_study": {"cases": ["C_cavity"]}}, "hybrid_study.cases"),
        ({"hybrid_study": {"user_data": {"D_graetz": ["nope"]}}}, "not defined under user_data.sources"),
        ({"hybrid_study": {"formulation": {"kind": "primitive", "hard_bc": True}}}, "need kind: streamfunction"),
        ({"hybrid_study": {"per_point": {"enabled": False}, "parametric": {"enabled": False}}}, "nothing to run"),
    ],
)
def test_spec_validation(over, msg):
    with pytest.raises(ValueError, match=msg):
        S.load_spec(None, "smoke", over)


def test_hybrid_report_wording(tmp_path):
    def rec(mode, n, e, kind="per_point", point=8, split="interp"):
        return {
            "case": "D_graetz",
            "kind": kind,
            "mode": mode,
            "n_data": n,
            "n_snapshots": 0 if mode == "data_free" else 1,
            "seed": 0,
            "point": point,
            "split": split,
            "Re": 20.0,
            "Pr": 1.0,
            "Bi": 1.0,
            "Pe": 20.0,
            "n_eval_cells": 100,
            "data_fit": None if mode == "data_free" else 1e-3,
            "ref_unc_max": 1e-3,
            "train_wall_s": 10.0,
            "status": "ok",
            **{f"rel_l2_{k}": e for k in ("max", "u", "v", "theta", "p")},
            "q_bottom_rel_err": e,
        }

    recs = [rec("data_free", 0, 0.1)] + [rec("data_only", n, e) for n, e in ((16, 0.8), (64, 0.4), (256, 0.2))]
    recs += [rec("hybrid", n, e) for n, e in ((16, 0.09), (64, 0.04), (256, 0.005))]
    recs += [
        rec(m, 0 if m == "data_free" else 64, e, "parametric", p, s)
        for m, e in (("data_free", 0.5), ("hybrid", 0.1))
        for p, s in ((8, "interp"), (11, "extrap"))
    ]
    txt = "\n".join(hybrid_table(recs, [], SPEC, tmp_path / "h.md", "t"))
    assert "target rel L2 < 0.01 from n = 256" in txt
    assert "≥2x better than data_free (1.00e-01) from n = 64" in txt
    assert "≥2x better than data_only at the same n from n = 16" in txt
    assert "data_only: 3/3 errors ≥ 0.3" not in txt and "≥2x better at [8, 11]" in txt
    assert "data_free: 2/2 errors ≥ 0.3 — solution not captured." in txt


def test_tiny_hybrid_run_end_to_end(tmp_path):
    spec = S.load_spec(
        None,
        "smoke",
        {
            "training": {"adam": {"epochs": 4}, "lbfgs": {"iterations": 1, "inner": 1}},
            "sampling": {"n_pde": 32, "n_bc": 4},
            "hybrid_study": {"cases": ["D_graetz"]},
        },
    )
    pc = spec["problems2d"]["D_graetz"]
    ref = CH.channel_reference(VALS, pc, 4, 2)
    pt = {"case": "D_graetz", "index": 8, "split": "interp", "values": VALS}
    big = H.snapshot_points(ref, 64, ["u", "v", "theta"], "random", 0.0, 0)
    r = E.run_point(spec, "D_graetz", "hybrid", 16, 0, pt, ref, big["mask"], tmp_path / "h", "t")
    assert r["status"] == "ok" and r["n_data"] == 16 and np.isfinite(r["data_fit"])
    assert r["n_eval_cells"] == int((~np.asarray(ref["solid"])).sum()) - 64 and abs(r["flux_ratio"] - 1) < 1e-3
    f = E.run_point(spec, "D_graetz", "data_free", 0, 0, pt, ref, big["mask"], tmp_path / "f", "t")
    assert f["data_fit"] is None and f["n_data"] == 0


def test_fit_error_uses_velocity_vector_norm_for_u_and_v():
    """v ~ 0 (channel): a misfit relative to v's own norm explodes; fit_error divides u, v by |(u, v)| (D70)."""
    prob, net = problem("D_graetz")
    ref = own_reference(prob, net)
    snap = H.snapshot_points(ref, 30, ["u", "v", "theta"], "random", 0.0, 0)
    term = H.snapshot_term("snap", prob, [(snap, None)], next(net.parameters()))
    assert H.fit_error(term, net) < 1e-12
    vel = float(torch.sqrt((term.targets["u"] ** 2).sum() + (term.targets["v"] ** 2).sum()))
    term.targets["v"] = term.targets["v"] + 0.01 * vel / np.sqrt(30)  # misfit 0.01 of the velocity-vector norm
    vel_new = float(torch.sqrt((term.targets["u"] ** 2).sum() + (term.targets["v"] ** 2).sum()))
    assert abs(H.fit_error(term, net) - 0.01 * vel / vel_new) < 1e-12  # denominator = norm of the (new) data
