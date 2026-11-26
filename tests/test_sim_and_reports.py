"""Math features of the kernel, the simulation suite and the report formats."""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path

import pytest

from aerosynapse.cad import kernel

ROOT = Path(__file__).resolve().parents[1]


def vol(f, h=0.02):
    return kernel.info({"features": [f]}, h=h)["volume"]


def test_math_features_match_exact_volumes():
    assert vol({"id": "i", "type": "implicit", "expr": "(x/2)^2 + y^2 + (z/0.5)^2 - 1", "size": [4.4, 2.2, 1.2]}) == pytest.approx(4 / 3 * math.pi, rel=2e-3)
    assert vol({"id": "s", "type": "superellipsoid", "radii": [1, 1, 1], "n1": 1, "n2": 1}) == pytest.approx(4 / 3 * math.pi, rel=2e-3)
    v_box = vol({"id": "s", "type": "superellipsoid", "radii": [1, 1, 1], "n1": 0.1, "n2": 0.1})
    assert 7.6 < v_box < 8.0  # a slightly rounded cube
    r, R, p = 0.06, 0.5, 0.25
    L = 6 * 2 * math.pi * math.sqrt(R * R + p * p)
    helix = {"id": "h", "type": "sweep", "x": "0.25*t", "y": "0.5*cos(t)", "z": "0.5*sin(t)", "t0": 0, "t1": 12 * math.pi, "radius": 0.06, "samples": 600}
    assert vol(helix, h=0.01) == pytest.approx(math.pi * r * r * L, rel=5e-3)


def test_bad_equations_are_reported():
    c = kernel.check({"features": [{"id": "bad", "type": "implicit", "expr": "x^2 + foo(y)"}]})
    assert any("unknown name 'foo'" in e for e in c["errors"])
    c = kernel.check({"features": [{"id": "bad", "type": "sweep", "x": "t*(", "y": "0", "z": "0"}]})
    assert not c["ok"]
    assert {"implicit", "superellipsoid", "sweep", "lattice"} <= set(kernel.templates())


def test_analytic_simulation(tmp_path):
    from aerosynapse.sim import analytic

    d = json.loads((ROOT / "results" / "fighter" / "baseline_design.json").read_text())
    r = analytic.run(d, tmp_path)
    assert (tmp_path / "summary.txt").exists() and len(list(tmp_path.glob("*.png"))) == 4
    assert 0.9 < r["panel"]["lift_slope_per_rad"] / (2 * math.pi) < 1.2  # thin sections: close to 2 pi
    assert r["area_rule"]["D_over_q_sears_haack_numeric_m2"] == pytest.approx(r["area_rule"]["D_over_q_sears_haack_m2"], rel=0.05)
    assert r["area_rule"]["D_over_q_design_m2"] >= r["area_rule"]["D_over_q_sears_haack_m2"] * 0.95  # Sears-Haack is the minimum
    ns = analytic.normal_shock(2.0)
    assert ns["M2"] == pytest.approx(0.5774, abs=1e-3) and ns["p02_p01"] == pytest.approx(0.7209, abs=1e-3)  # NACA 1135 tables


def test_lifting_line_elliptic_limit():
    from aerosynapse.sim.analytic import lifting_line

    # an untwisted wing of taper 0.4 is close to elliptic: e above 0.97, CL_alpha below the 2D slope
    r = lifting_line({"span": 5.0, "root_chord": 2.0, "taper": 0.4, "translate": [0, 0, 0]}, 4.0)
    assert r["e"] > 0.97
    assert 0 < r["CL"] / math.radians(4.0) < 2 * math.pi


def test_report_formats(tmp_path):
    pytest.importorskip("docx")
    from aerosynapse.workflow import Project
    from aerosynapse.workflow import report_formats as RF
    from aerosynapse.workflow import stages as S

    p = Project.create("rep", where=tmp_path)
    S.run_stage(p, "concept", "quick")
    out = RF.write_all(p)
    assert set(out) == {"pdf", "docx", "txt"}
    assert out["pdf"].read_bytes()[:5] == b"%PDF-"
    with zipfile.ZipFile(out["docx"]) as z:
        assert "word/document.xml" in z.namelist()
        assert "Basic design" in z.read("word/document.xml").decode("utf-8")
    t = out["txt"].read_text(encoding="utf-8")
    assert "REP: DESIGN WORKFLOW REPORT" in t and "1 BASIC DESIGN" in t


def test_icons_cover_every_tool():
    import re

    inc = (ROOT / "cpp" / "studio" / "tools.inc").read_text(encoding="utf-8")
    names = re.findall(r'\{"\w+", "(\w+)", "[^"]*", "[^"]*",', inc) + re.findall(r'\{"\w+", "[^"]*", \d+, \d+, \d+, "(\w+)"\}', inc)
    missing = [n for n in names if not (ROOT / "assets" / "icons" / f"{n}.png").exists()]
    if missing and shutil.which("python"):
        pytest.fail(f"icons missing (run scripts/make_icons.py): {missing}")
    assert len(names) > 100
