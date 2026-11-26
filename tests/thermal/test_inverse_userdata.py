"""Your own data and body shape in every Phase 5 step: design from the data file, noise level from its sigma column,
the user template config, and the 'synthetic what-if' rule for derived cases."""

import numpy as np
import pytest

from aerosynapse.thermal.experiments import p5_1_identifiability as P51
from aerosynapse.thermal.experiments import p5_3_classical as P53
from aerosynapse.thermal.experiments import p5_5_uq as P55
from aerosynapse.thermal.inverse import data as D
from aerosynapse.thermal.inverse import design as DS
from aerosynapse.thermal.inverse import spec as IS
from aerosynapse.thermal.utils.experiment import ROOT

USER = "configs/inverse/user_case.yaml"
CYL = "configs/inverse/examples/user_cylinder.yaml"


def _write(tmp_path, x, t, T, sigma=None, name="d.csv"):
    p = tmp_path / name
    cols = [x, t, T] + ([sigma] if sigma is not None else [])
    hdr = "x,t,T" + (",sigma" if sigma is not None else "")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("# comment line\n" + hdr + "\n")
        np.savetxt(fh, np.column_stack(cols), delimiter=",")
    return str(p)


def test_user_template_and_cylinder_example_load():
    s = IS.load_spec(USER)
    assert s["data"]["source"] == "file" and s["data"]["level"] == pytest.approx(0.02, abs=1e-3)
    assert s["data"]["level_from_file"] is True
    d = D.load(s)
    assert len(IS.sensor_positions(s)) * len(IS.measurement_times(s)) == len(d["T"]) == 480
    c = IS.load_spec([USER, CYL])
    assert c["body"]["shape"] == "cylinder" and len(IS.sensor_positions(c)) == 6 and len(IS.measurement_times(c)) == 50
    assert IS.data_design(c)["full_grid"]


def test_design_from_data_and_precedence(tmp_path):
    xs, ts = np.array([0.0, 0.01, 0.02]), np.array([10.0, 20.0, 40.0, 80.0])
    X, Tt = np.meshgrid(xs, ts)
    f = _write(tmp_path, X.ravel(), Tt.ravel(), 50 + 0 * X.ravel(), 0.5 + 0 * X.ravel())
    over = {"data": {"file": f}, "measurement": {"sensors": {"from_data": True}, "times": {"from_data": True}}}
    s = IS.load_spec(overrides=over)
    assert np.allclose(IS.sensor_positions(s), xs) and np.allclose(IS.measurement_times(s), ts)
    # explicit positions still win over the file's
    e = IS.load_spec(overrides=IS.deep_merge(over, {"measurement": {"sensors": {"positions": [1.0]}}}))
    assert np.allclose(IS.sensor_positions(e), [0.02])
    ex = IS.explicit_design(s, keep_time_grid=False)["measurement"]
    assert ex["sensors"]["positions"] == [0.0, 0.5, 1.0] and ex["times"]["n"] == 4 and ex["times"]["t0"] == 10.0


def test_scattered_file_is_flagged(tmp_path):
    f = _write(tmp_path, np.array([0.0, 0.01, 0.02]), np.array([10.0, 20.0, 30.0]), np.array([50.0, 50.0, 50.0]))
    s = IS.load_spec(overrides={"data": {"file": f}, "measurement": {"sensors": {"from_data": True}}})
    assert not IS.data_design(s)["full_grid"]


def test_level_auto_and_validation(tmp_path):
    T = np.linspace(80, 40, 6)
    f = _write(tmp_path, np.full(6, 0.01), np.arange(1.0, 7.0), T, np.full(6, 0.6))
    s = IS.load_spec(overrides={"data": {"source": "file", "file": f, "level": "auto"}})
    theta = (T - 20.0) / 60.0
    assert s["data"]["level"] == pytest.approx(0.6 / (np.sqrt(np.mean(theta**2)) * 60.0))
    g = _write(tmp_path, np.full(6, 0.01), np.arange(1.0, 7.0), T, name="nosig.csv")
    with pytest.raises(ValueError, match="sigma column"):
        IS.load_spec(overrides={"data": {"source": "file", "file": g, "level": "auto"}})
    with pytest.raises(ValueError, match="from_data"):
        IS.load_spec(overrides={"measurement": {"sensors": {"from_data": True}}})


def test_designer_and_sweeps_with_a_file_design():
    s = IS.load_spec(USER)
    d = DS.Designer(s)
    b = d.base_design()
    assert len(b["positions"]) == 8 and b["t1"] == pytest.approx(1200.0)
    o = d.spec_of({**b, "t1": 2400.0})
    assert o["measurement"]["times"]["from_data"] is False and len(IS.measurement_times(o)) == 60
    assert IS.measurement_times(o).max() == pytest.approx(2400.0)
    ov = P51.sweep_override(s, "n_sensors", 4)
    assert len(IS.sensor_positions(IS.load_spec(USER, ov))) == 4
    ov = P51.sweep_override(s, "t1", 600.0)
    t = IS.measurement_times(IS.load_spec(USER, ov))
    assert len(t) == 60 and t.max() == pytest.approx(600.0)


def test_derived_cases_are_synthetic_what_ifs():
    s = P53.case_spec([USER], {}, "no_source")
    assert s["data"]["source"] == "synthetic" and s["body"]["q_vol"] == 0.0 and s["data"]["compare_to_parameters"]
    assert P53.case_spec([USER], {}, "base")["data"]["source"] == "file"


def test_saved_pinn_runs_are_reused_only_for_the_same_problem():
    cfg = ROOT / "results" / "p5_2_pinn_inverse_default" / "config.yaml"
    if not cfg.exists():
        pytest.skip("no saved Step 5.2 run")
    assert P55.same_problem(cfg, IS.load_spec())
    assert not P55.same_problem(cfg, IS.load_spec(USER))
    assert not P55.same_problem(cfg, IS.load_spec(overrides={"body": {"shape": "sphere"}}))


def test_replace_marker_replaces_a_default_dict():
    from aerosynapse.thermal.data.spec import deep_merge

    assert deep_merge({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"_replace": True, "z": 3}}) == {"a": {"z": 3}, "b": 1}
    assert deep_merge({"a": {"x": 1}}, {"a": {"z": 3}}) == {"a": {"x": 1, "z": 3}}
    s = IS.load_spec(USER)
    assert list(s["studies"]["uq"]["cases"]) == ["mine"]
    assert "base_2pct" in IS.load_spec()["studies"]["uq"]["cases"]  # the default case is unchanged
