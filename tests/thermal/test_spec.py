"""User-definable conditions: layering, overrides, validation messages."""

import pytest
import yaml

from aerosynapse.thermal.data import spec as SP


def write(tmp_path, d, name="u.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    return p


def test_defaults_load_and_describe():
    s = SP.load_spec(None, "smoke")
    assert set(s["cases"]) == {"caseA", "caseB", "caseC"} and s["cases"]["caseC"]["grid"] == 32
    assert SP.load_spec(None, "full")["cases"]["caseC"]["grid"] == 64
    assert "caseC" in SP.describe(s)


def test_user_layer_overrides_defaults_and_profile_applies_last(tmp_path):
    u = write(
        tmp_path,
        {
            "cases": {
                "caseC": {"parameters": {"Re": {"lo": 50, "hi": 500, "scale": "log"}, "Pr": 0.71}, "grid": 24},
                "caseA": {"enabled": False},
            },
            "profiles": {"smoke": {"cases": {"caseC": {"sizes": {"train": 5}}}}},
        },
    )
    s = SP.load_spec(u, "smoke")
    c = s["cases"]["caseC"]
    dims, fixed = SP.case_dims(c, "caseC")
    assert [d.name for d in dims] == ["Re", "Bi"] and dims[0].lo == 50 and fixed == {"Pr": 0.71, "Ri": 0.0}
    assert c["grid"] == 24 and c["sizes"]["train"] == 5 and c["sizes"]["val"] == 3  # deep merge keeps other keys
    assert s["cases"]["caseA"]["enabled"] is False


def test_command_line_overrides():
    ov = SP.parse_set(
        [
            "cases.caseC.grid=48",
            "noise.levels=[0, 0.05]",
            "design.method=lhs",
            "cases.caseC.conditions.thermal.top.value=0.25",
        ]
    )
    s = SP.load_spec(None, "smoke", ov)
    assert s["cases"]["caseC"]["grid"] == 48 and s["noise"]["levels"] == [0, 0.05] and s["design"]["method"] == "lhs"
    assert s["cases"]["caseC"]["conditions"]["thermal"]["top"]["value"] == 0.25
    assert s["cases"]["caseC"]["conditions"]["thermal"]["top"]["type"] == "dirichlet"  # untouched sibling kept


@pytest.mark.parametrize(
    "over,needle",
    [
        ({"cases": {"caseC": {"parameters": {"Re": {"lo": 10, "hi": 5}}}}}, "lo < hi"),
        ({"cases": {"caseC": {"parameters": {"Re": {"lo": 0, "hi": 5, "scale": "log"}}}}}, "log scale requires lo > 0"),
        ({"cases": {"caseC": {"parameters": {"Foo": 1.0}}}}, "unknown ['Foo']"),
        ({"cases": {"caseA": {"parameters": {"Pe": 10.0}}}}, "must be swept"),
        ({"cases": {"caseC": {"grid": 31}}}, "even integer"),
        ({"cases": {"caseC": {"scheme": "bogus"}}}, "scheme"),
        (
            {"cases": {"caseC": {"conditions": {"thermal": {"top": {"type": "robin", "Bi": "Zeta"}}}}}},
            "not defined in `parameters`",
        ),
        (
            {"cases": {"caseC": {"conditions": {"velocity": {"top": {"profile": "expr", "expr": "sin(q)"}}}}}},
            "cannot evaluate",
        ),
        ({"cases": {"caseC": {"times": {"kind": "linear", "t0": 5, "t1": 1, "n": 3}}}}, "t0 < t1"),
        ({"cases": {"caseC": {"sizes": {"train": 0}}}}, "at least 1 training point"),
        ({"design": {"unit_train_limit": 1.2}}, "unit_train_limit"),
        ({"noise": {"model": "speckle"}}, "noise.model"),
        ({"cases": {"caseD": {}}}, "unknown case"),
    ],
)
def test_validation_errors_name_the_offending_key(over, needle):
    with pytest.raises(ValueError, match="invalid condition at") as e:
        SP.load_spec(None, "smoke", over)
    assert needle in str(e.value)


def test_unknown_profile_and_all_disabled():
    with pytest.raises(ValueError, match="profile"):
        SP.load_spec(None, "nonexistent")
    with pytest.raises(ValueError, match="no case enabled"):
        SP.load_spec(None, "smoke", {"cases": {c: {"enabled": False} for c in ("caseA", "caseB", "caseC")}})


def test_trivial_thermal_problem_rejected():
    trivial = {
        "thermal": {w: {"type": "neumann", "value": 0.0} for w in ("top", "bottom", "left", "right")},
        "initial_theta": 0.0,
    }
    with pytest.raises(ValueError, match="trivially 0"):
        SP.load_spec(None, "smoke", {"cases": {"caseC": {"conditions": trivial}}})


def test_profile_eval():
    import numpy as np

    s = np.linspace(0, 1, 5)
    assert np.allclose(SP.eval_profile({"profile": "uniform", "amplitude": 2.0}, s, 0.0), 2.0)
    assert np.allclose(SP.eval_profile({"profile": "parabolic"}, s, 0.0), 4 * s * (1 - s))
    assert SP.eval_profile({"profile": "cosh", "C": 50.0}, np.array([0.0, 1.0]), 0.0)[0] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(SP.eval_profile({"profile": "expr", "expr": "s*(1+t)", "amplitude": 2.0}, s, 1.0), 4 * s)
