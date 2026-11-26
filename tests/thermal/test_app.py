"""Phase 8 interface: the shared backend (solvers, validity guard, inverse, export), the CLI, and the Streamlit app
run headless (streamlit.testing AppTest). PINN parts are skipped where the git-ignored checkpoints are absent."""

import json
from pathlib import Path

import numpy as np
import pytest

from aerosynapse.thermal.app import backend as B
from aerosynapse.thermal.app import cli

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
APP = str(__import__("aerosynapse.thermal._home", fromlist=["HOME"]).CODE / "app" / "main.py")
has_1d = B.pinn_available("B_slab")[0]
needs_1d = pytest.mark.skipif(not has_1d, reason="Step 6.2 checkpoints not present (git-ignored)")


# ---------------------------------------------------------------------------- backend
def test_cases_are_complete():
    for k, c in B.CASES.items():
        assert c.key == k and c.params and c.title
        for p in c.params:
            assert p.lo <= p.default <= p.hi
            if p.train:
                assert p.lo <= p.train[0] < p.train[1] <= p.hi
    with pytest.raises(KeyError):
        B.case("nope")


def test_validity_guard():
    assert [m["level"] for m in B.validity("B_slab", {"Bi": 1.0})] == ["ok"]
    stop = B.validity("B_slab", {"Bi": 30.0})
    assert stop[0]["level"] == "stop" and "EXTRAPOLATION" in stop[0]["text"]
    assert any(m["level"] == "warn" for m in B.validity("A_steady", {"Pe": 500.0}, {"N": 40, "scheme": "central"}))
    cav = B.validity("cavity", {"Re": 12000.0})
    if (ROOT / "results/p6_3_extrapolation_default/metrics.json").exists():
        assert cav[0]["level"] == "stop"  # above the critical Re of Step 6.3: never presented as a valid answer


def test_steady_solver_and_order():
    r = B.solve("A_steady", {"Pe": 10.0}, {"N": 80})
    assert r["err_inf"] < 1e-3
    cv = B.convergence("A_steady", {"Pe": 10.0}, [10, 20, 40, 80, 160])
    assert abs(cv["schemes"]["central"]["observed_order"] - 2.0) < 0.1
    assert abs(cv["schemes"]["upwind1"]["observed_order"] - 1.0) < 0.15


def test_slab_solver_is_second_order():
    """Crank-Nicolson with Rannacher start-up and dt ~ h: second order against the eigen-series (Bi 0.1 to 10)."""
    for Bi in (0.1, 10.0):
        cv = B.convergence("B_slab", {"Bi": Bi}, [10, 20, 40, 80])
        assert abs(cv["schemes"]["central"]["observed_order"] - 2.0) < 0.1


def test_inverse_identifiable_and_not():
    r = B.inverse()
    assert r["identifiable"] and len(r["rows"]) == 3
    gate = {"alpha": -2.99, "k": -3.98, "h": -0.24}  # docs/tables/phase5_gate.md (same data, analytic model)
    for row in r["rows"]:
        assert abs(row["error_pct"] - gate[row["parameter"]]) < 0.1
        lo, hi = row["ci95"]
        assert lo < row["truth"] < hi
    none = B.inverse(q_vol=0.0)
    assert not none["identifiable"] and none["rank"] == 2 and not none["rows"]


def test_export_columns_and_json():
    r = B.solve("B_slab", {"Bi": 1.0}, {"N": 20, "n_out": 3})
    csv = B.to_csv(B.field_columns("B_slab", r, "theta")).decode().splitlines()
    assert csv[0] == "t,x,theta" and len(csv) == 1 + 3 * 21
    json.dumps(B.jsonable(r))


# ---------------------------------------------------------------------------- CLI
def test_cli_strict_validity(capsys):
    assert cli.main(["validity", "--case", "B_slab", "--set", "Bi=30", "--strict"]) == 2
    assert cli.main(["validity", "--case", "B_slab", "--set", "Bi=1"]) == 0
    capsys.readouterr()


def test_cli_solve_writes_csv(tmp_path, capsys):
    out = tmp_path / "s.csv"
    assert cli.main(["solve", "--case", "A_steady", "--set", "Pe=5", "--opt", "N=20", "--out", str(out)]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["result"]["err_inf"] < 1e-2 and out.read_text().startswith("x,theta")


@needs_1d
def test_pinn_prediction_and_comparison():
    c = B.compare("B_slab", {"Bi": 1.0})
    assert c["rel_l2"] < 0.05 and c["pinn"]["n_members"] == 8
    two = B.predict("B_slab", {"Bi": 1.0}, {"members": ["ens_s0", "ens_s1"]})
    assert two["n_members"] == 2
    sw = B.sweep_1d("B_slab", np.array([1.0, 30.0]))
    assert sw["rel_l2"][1] > 5 * sw["rel_l2"][0]  # error grows out of range (Step 6.3)


# ---------------------------------------------------------------------------- the app, headless
def _app(case, section):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["case"] = case
    at.query_params["section"] = section
    return at.run()


@pytest.mark.parametrize("section", ["Overview", "Solver", "Inverse", "Convergence", "Export"])
def test_app_sections_without_networks(section):
    at = _app("A_steady", section)
    assert not at.exception and not at.error
    assert at.title[0].value.startswith("Physics-informed")


@needs_1d
@pytest.mark.parametrize("section", ["PINN", "Comparison", "Training monitor", "Uncertainty & validity"])
def test_app_sections_with_networks(section):
    at = _app("B_slab", section)
    assert not at.exception and not at.error


@needs_1d
def test_app_extrapolation_is_flagged():
    at = _app("B_slab", "Comparison")
    at.number_input(key="p_B_slab_Bi").set_value(30.0).run()
    assert not at.exception
    assert any("EXTRAPOLATION" in e.value or "Extrapolation" in e.value for e in at.error)


@pytest.mark.slow
def test_app_2d_cases():
    at = _app("cavity", "Solver")
    at.button(key="run_solver").click().run()
    assert not at.exception and at.metric[1].value.startswith("-0.10")
    if B.pinn_available("D_graetz")[0]:
        at = _app("D_graetz", "Comparison")
        at.button(key="run_cmp").click().run()
        assert not at.exception and not at.error


def test_app_section_slugs():
    at = _app("B_slab", "uncertainty")  # deep links use the first word of a section: '&' needs no escaping
    assert at.radio(key="section").value == "Uncertainty & validity"


def test_app_run_request_persists():
    """A 2D run, once requested, stays shown when another widget changes (a plain button lasts one rerun)."""
    at = _app("cavity", "Solver")
    assert at.info  # waits for the request
    at.button(key="run_solver").click().run()
    assert at.metric and not at.info
    at.selectbox(key="c_N").set_value(16).run()
    assert at.metric and not at.info and not at.exception
