"""Step 4.7: ablation configurations, per-configuration specs, validation, aggregation / report rules, and a tiny
end-to-end training of the 'full' recipe (every component at once) for both kinds."""

import numpy as np
import pytest

from aerosynapse.thermal.experiments import p4_6_hybrid as HY
from aerosynapse.thermal.experiments import p4_7_ablation as A
from aerosynapse.thermal.experiments.p4_5b_channel import case_design
from aerosynapse.thermal.experiments.p4_report import ablation_aggregate, ablation_table
from aerosynapse.thermal.pinn import channel2d as CH
from aerosynapse.thermal.pinn import spec as S

SPEC = S.load_spec(None, "smoke")
AB = SPEC["ablation_study"]


def test_configurations_leave_one_out_and_add_one_in():
    cs = A.configurations(AB)
    names = list(AB["components"])
    assert len(cs) == 2 * (len(names) + 1)
    full = next(c for c in cs if c["name"] == "full")
    assert full["on"] == names and full["changed"] is None
    for n in names:
        lo = next(c for c in cs if c["name"] == f"full-{n}")
        ai = next(c for c in cs if c["name"] == f"base+{n}")
        assert set(lo["on"]) == set(names) - {n} and ai["on"] == [n] and lo["changed"] == ai["changed"] == n
    assert [c["name"] for c in A.configurations({**AB, "designs": ["add_one_in"]})][0] == "base"


def test_config_spec_applies_exactly_the_components():
    full = A.config_spec([], "smoke", {}, AB, list(AB["components"]))
    assert full["formulation"]["kind"] == "streamfunction" and full["formulation"]["hard_bc"]
    assert full["network"]["arch"] == "siren" and full["loss"]["balancing"]["method"] == "ntk"
    assert full["sampling"]["adaptive"]["method"] == "rad" and full["training"]["adam"]["grad_clip"] == 1.0
    assert full["ablation_data"] == {"n_snapshots": 8, "n_data": 64}
    assert full["hybrid_study"]["formulation"] == {} and full["hybrid_study"]["common"] == {}
    assert full["hybrid_study"]["parametric"]["n_data"] == 64
    base = A.config_spec([], "smoke", {}, AB, [])
    assert (
        base["formulation"]["kind"] == SPEC["formulation"]["kind"]
        and base["network"]["arch"] == SPEC["network"]["arch"]
    )
    assert base["ablation_data"] is None and base["loss"]["balancing"]["method"] == "fixed"
    common = A.config_spec([], "smoke", {}, {**AB, "common": {"network": {"width": 16}}}, ["ntk"])
    assert common["network"]["width"] == 16 and common["loss"]["balancing"]["method"] == "ntk"


@pytest.mark.parametrize(
    "over,msg",
    [
        ({"ablation_study": {"case": "C_cavity"}}, "ablation_study.case"),
        ({"ablation_study": {"kind": "both"}}, "parametric or per_point"),
        ({"ablation_study": {"components": {"x": {"optimizer": {}}}}}, "unknown key"),
        ({"ablation_study": {"components": {"x": {}}}}, "non-empty dict"),
        ({"ablation_study": {"components": {"data": {"data": {"n_snapshots": 99, "n_data": 4}}}}}, "sizes.train"),
        (
            {"ablation_study": {"components": {"d2": {"data": {"n_snapshots": 2, "n_data": 4}}}}},
            "at most one component",
        ),
        ({"ablation_study": {"designs": ["leave_two_out"]}}, "ablation_study.designs"),
        ({"ablation_study": {"seeds": [0, 0, 1]}}, "distinct integers"),
    ],
)
def test_spec_validation(over, msg):
    with pytest.raises(ValueError, match=msg):
        S.load_spec(None, "smoke", over)


def _recs(name, design, changed, on, errs, time=100.0):
    out = []
    for s, e in enumerate(errs):
        for p, split in ((8, "interp"), (9, "interp"), (11, "extrap")):
            out.append(
                {
                    "config": name,
                    "design": design,
                    "changed": changed,
                    "components": on,
                    "kind": "parametric",
                    "seed": s,
                    "point": p,
                    "split": split,
                    "rel_l2_max": e if split == "interp" else 3 * e,
                    "q_bottom_rel_err": e / 2,
                    "train_wall_s": time + s,
                }
            )
    return out


def test_aggregate_and_report_rules(tmp_path):
    recs = _recs("full", "leave_one_out", None, ["a", "b", "c"], [0.10, 0.11, 0.12])
    recs += _recs("full-a", "leave_one_out", "a", ["b", "c"], [0.40, 0.42, 0.44])  # needed (4x, separated)
    recs += _recs("full-b", "leave_one_out", "b", ["a", "c"], [0.05, 0.30, 0.55])  # 2.7x but within spread
    recs += _recs("full-c", "leave_one_out", "c", ["a", "b"], [0.12, 0.13, 0.14])  # no effect
    recs += _recs("base", "add_one_in", None, [], [0.50, 0.52, 0.54], 50.0)
    recs += _recs("base+a", "add_one_in", "a", ["a"], [0.20, 0.21, 0.22], 60.0)  # helps alone (2.5x)
    recs += _recs("base+b", "add_one_in", "b", ["b"], [0.50, 0.50, 0.50], 60.0)
    recs += _recs("base+c", "add_one_in", "c", ["c"], [1.30, 1.31, 1.32], 60.0)  # hurts alone
    g = ablation_aggregate(recs)
    assert abs(g["full"]["err_mean"] - 0.11) < 1e-12 and abs(g["full"]["err_std"] - 0.01) < 1e-12
    assert abs(g["full"]["ext_mean"] - 0.33) < 1e-12 and g["full"]["time_mean"] == 101.0
    spec = S.deep_merge(
        SPEC, {"ablation_study": {"components": {"a": {"loss": {}}, "b": {"loss": {}}, "c": {"loss": {}}}}}
    )
    spec["ablation_study"]["components"] = {"a": {"loss": {}}, "b": {"loss": {}}, "c": {"loss": {}}}
    txt = "\n".join(ablation_table(recs, spec, tmp_path / "a.md", "t"))
    assert "- a: needed: removing it is ≥2x worse | helps alone: ≥2x better than base" in txt
    assert "- b: ≥2x but within the seed spread | no effect beyond 2x" in txt
    assert "- c: no effect beyond 2x | hurts alone: ≥2x worse than base" in txt
    assert "Best mean error: `full`" in txt and "Only" not in txt
    one = ablation_table([r for r in recs if r["seed"] == 0], spec, tmp_path / "b.md", "t")
    assert any("Only 1 seed(s)" in x for x in one)


@pytest.mark.parametrize("kind", ["parametric", "per_point"])
def test_tiny_full_recipe_trains(tmp_path, kind):
    """Every default component at once (exact psi, grad_clip, ntk, rad, siren, data) through the 4.6 run functions."""
    tiny = {
        "training": {"adam": {"epochs": 4}, "lbfgs": {"iterations": 1, "inner": 1}},
        "sampling": {"n_pde": 32, "n_bc": 4},
        "ablation_study": {"kind": kind},
    }
    ab = S.load_spec(None, "smoke", tiny)["ablation_study"]
    ab["components"]["ntk"] = {"loss": {"balancing": {"method": "ntk", "ntk": {"every": 2, "n_points": 4}}}}
    ab["components"]["rad"] = {"sampling": {"adaptive": {"method": "rad", "period": 2, "n_candidates": 48}}}
    ab["components"]["data"] = {"data": {"n_snapshots": 2, "n_data": 8}}
    s = A.config_spec([], "smoke", tiny, ab, list(ab["components"]))
    design = case_design(s, "D_graetz")
    pc = s["problems2d"]["D_graetz"]
    if kind == "parametric":
        refs = [CH.channel_reference(p["values"], pc, 4, 2) for p in design["points"]]
        snaps = [(p, CH.channel_reference(p["values"], pc, 4, 2)) for p in HY.train_design_points(s, "D_graetz")[:2]]
        out = HY.run_parametric(s, "D_graetz", "hybrid", 0, design, refs, snaps, tmp_path / "r", "t")
        assert len(out) == len(design["points"])
    else:
        pt = design["points"][0]
        ref = CH.channel_reference(pt["values"], pc, 4, 2)
        out = [
            HY.run_point(s, "D_graetz", "hybrid", 8, 0, pt, ref, np.zeros_like(ref["solid"], bool), tmp_path / "r", "t")
        ]
    assert all(r["status"] == "ok" and np.isfinite(r["rel_l2_max"]) and np.isfinite(r["data_fit"]) for r in out)
