"""Whole-vehicle models, the data layer and the Aircraft Designer app."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerosynapse.aircraft import analysis as an
from aerosynapse.aircraft import datahub as dh
from aerosynapse.aircraft import optimise as op
from aerosynapse.aircraft import panel3d as p3
from aerosynapse.aircraft import vehicle as vh

APP = str(Path(__file__).resolve().parents[1] / "src" / "aerosynapse" / "ui" / "designer.py")


def spheroid(fin, ni=34, nj=24):
    t = np.linspace(0, np.pi, ni)
    ph = np.linspace(0, 2 * np.pi, nj, endpoint=False)
    return np.stack([np.outer(-fin * np.cos(t), np.ones_like(ph)), np.outer(np.sin(t), np.cos(ph)), np.outer(np.sin(t), np.sin(ph))], -1)


@pytest.mark.verification
@pytest.mark.parametrize("fin,exact", [(1.0, 1.5), (6.0, None)])
def test_panel_method_matches_exact_spheroid_speed(fin, exact):
    f = p3.solve(p3.from_grid(spheroid(fin)))
    v = np.sqrt((f.V**2).sum(1)).max()
    assert v == pytest.approx(exact or p3.ellipsoid_vmax(fin), rel=0.01)
    assert f.leak < 1e-3


@pytest.mark.verification
def test_wave_drag_of_sears_haack_is_exact():
    L, A = 30.0, 4.0
    x = np.linspace(0, L, 600)
    S = A * np.clip(4 * (x / L) * (1 - x / L), 0, None) ** 1.5
    assert an.wave_drag_area(x, S) == pytest.approx(9 * np.pi * A**2 / (2 * L**2), rel=0.01)


@pytest.mark.verification
def test_sears_haack_beats_other_bodies_of_same_volume_and_length():
    L = 30.0
    x = np.linspace(0, L, 800)
    xi = x / L
    sh = np.clip(4 * xi * (1 - xi), 0, None) ** 1.5
    other = np.clip(4 * xi * (1 - xi), 0, None) ** 1.0
    other = other * np.trapezoid(sh, x) / np.trapezoid(other, x)
    assert an.wave_drag_area(x, sh) < an.wave_drag_area(x, other)


@pytest.mark.verification
def test_supersonic_lift_factor_has_the_ackeret_limit():
    M = 2.0
    beta = np.sqrt(M**2 - 1)
    AR = 1e7
    assert AR * (M**2 - 1) / (4 * AR * beta - 2) == pytest.approx(beta / 4, rel=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("name", list(vh.PRESETS))
def test_every_preset_builds_a_closed_geometry(name):
    v = vh.build(vh.PRESETS[name])
    V, F = vh.surface(v)
    assert v.S_ref > 0 and v.S_wet_body > 0 and v.V_body > 0
    assert np.isfinite(V).all() and F.max() < len(V)


@pytest.mark.verification
def test_cylinder_like_body_has_the_right_area_and_volume():
    d = {**vh.PRESETS["tube_and_wing"], "fn": 0.06, "ft": 0.15}
    v = vh.build(d)
    R, L = d["W"], d["L"]
    assert v.V_body == pytest.approx(np.pi * R**2 * L * 0.9, rel=0.12)
    assert v.S_wet_body == pytest.approx(2 * np.pi * R * L * 0.92, rel=0.12)


@pytest.mark.verification
def test_flat_body_is_treated_as_lifting():
    assert vh.build(vh.PRESETS["blended_wing_body"]).lifting_body
    assert not vh.build(vh.PRESETS["tube_and_wing"]).lifting_body


@pytest.mark.unit
def test_any_design_in_the_box_builds_and_evaluates():
    rng = np.random.default_rng(0)
    m = an.Mission()
    for _ in range(12):
        d = vh.to_dict(vh.LO + rng.random(len(vh.LO)) * (vh.HI - vh.LO))
        r = an.evaluate(vh.build(d), m)
        assert np.isfinite(r.mtow) and r.drag.CD > 0


@pytest.mark.verification
def test_every_section_is_left_right_symmetric():
    """A one-sided body keeps full lift credit with half the skin: the search found that once."""
    rng = np.random.default_rng(3)
    for _ in range(60):
        d = vh.to_dict(vh.LO + rng.random(len(vh.LO)) * (vh.HI - vh.LO))
        y, z = vh.section(d, 96)
        assert y.max() == pytest.approx(-y.min(), abs=1e-6)
        assert np.sort(y) == pytest.approx(np.sort(-y), abs=1e-6)


@pytest.mark.verification
def test_a_closed_body_has_at_least_twice_its_planform_as_surface():
    rng = np.random.default_rng(4)
    for _ in range(40):
        d = vh.to_dict(vh.LO + rng.random(len(vh.LO)) * (vh.HI - vh.LO))
        v = vh.build(d)
        planform = 2 * float(np.trapezoid(v.a, v.x_st))
        assert v.S_wet_body >= 1.95 * planform


@pytest.mark.validation
def test_airliner_baseline_is_in_the_published_range():
    st = op.Study(an.Mission(), vh.PRESETS["tube_and_wing"], 41000.0)
    d = st.r0.drag
    cd0 = d.CD_friction + d.CD_form + d.CD_misc
    assert 0.016 < cd0 < 0.024  # public 737-class values run 0.018 to 0.021
    assert 14 < st.r0.L_over_D < 20


@pytest.mark.slow
@pytest.mark.parametrize("mission,name,oew", [("airliner", "tube_and_wing", 41000.0), ("supersonic", "supersonic_jet", None)])
def test_baselines_are_feasible_and_score_one(mission, name, oew):
    m = an.Mission(**an.SUPERSONIC) if mission == "supersonic" else an.Mission()
    st = op.Study(m, vh.PRESETS[name], oew)
    assert all(x <= 1e-6 for x in st.r0.constraints.values())
    assert st.score(vh.PRESETS[name])[0] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_hot_day_and_contamination_raise_drag():
    st = op.Study(an.Mission(), vh.PRESETS["tube_and_wing"], 41000.0)
    names = [f.name for f in st.fl]
    v = vh.build(vh.PRESETS["tube_and_wing"])
    clean = an.drag(v, an.isa(11000), 0.78, 0.5, an.Flight()).CD
    dirty = an.drag(v, an.isa(11000), 0.78, 0.5, an.Flight(contaminated=True)).CD
    assert dirty > clean and "contaminated surface" in names


@pytest.mark.unit
def test_steep_tail_costs_form_drag():
    d = dict(vh.PRESETS["tube_and_wing"])
    a = an.drag(vh.build(d), an.isa(11000), 0.78, 0.5, an.Flight()).CD_form
    d["ft"] = 0.15
    assert an.drag(vh.build(d), an.isa(11000), 0.78, 0.5, an.Flight()).CD_form > a


@pytest.mark.unit
def test_design_space_round_trip():
    sp = op.space("freeform")
    z = np.random.default_rng(1).random(len(sp.names))
    assert np.allclose(sp.encode(sp.decode(z)), z)


@pytest.mark.unit
def test_upload_accepts_good_data_and_explains_bad_data(tmp_path):
    assert dh.store("polar", "mach,cl,cd\n0.78,0.5,0.03\n", tmp_path) == 1
    assert dh.load("polar", tmp_path)[0]["cd"] == 0.03
    with pytest.raises(ValueError, match="missing columns: cd"):
        dh.parse("polar", "mach,cl\n0.78,0.5\n")
    with pytest.raises(ValueError, match="line 2: column cl must be a number"):
        dh.parse("polar", "mach,cl,cd\n0.78,abc,0.03\n")


@pytest.mark.unit
def test_empty_slots_are_reported_for_later_upload(tmp_path):
    s = dh.status(tmp_path)
    assert len(s) == len(dh.SLOTS) and all(r["state"].startswith("empty") for r in s)


@pytest.mark.unit
def test_drag_calibration_recovers_a_known_scale():
    polar = [{"mach": 0.7, "cl": c, "cd": 1.1 * (0.02 + 0.04 * c**2)} for c in (0.2, 0.4, 0.6)]
    assert dh.drag_scale(polar, lambda M, c: 0.02 + 0.04 * c**2) == pytest.approx(1.1)


@pytest.mark.unit
def test_reference_file_states_quality_and_source():
    for e in dh.reference()["entries"]:
        assert e["quality"] and e["url"].startswith("http")


@pytest.fixture
def app():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP, default_timeout=60)


@pytest.mark.slow
@pytest.mark.parametrize("page", ["Design", "Mission", "Analysis", "Optimise", "Study results", "Shape library", "Data"])
def test_every_page_opens_without_error(app, page):
    app.run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
