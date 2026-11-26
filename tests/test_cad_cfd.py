"""Geometry kernel, 3D flow solver, exact solutions and the Studio build."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aerosynapse.cad import analyse, kernel
from aerosynapse.cfd import exact, runner


def test_kernel_volumes_and_areas_match_exact_shapes():
    s = kernel.info({"features": [{"id": "s", "type": "sphere", "r": 1.0}]}, h=0.03)
    assert s["volume"] == pytest.approx(4 / 3 * math.pi, rel=2e-3)
    assert s["wetted_area"] == pytest.approx(4 * math.pi, rel=2e-3)
    t = kernel.info({"features": [{"id": "t", "type": "torus", "R": 1.0, "r": 0.3}]}, h=0.03)
    assert t["volume"] == pytest.approx(2 * math.pi**2 * 0.09, rel=5e-3)
    b = kernel.info({"features": [{"id": "b", "type": "box", "size": [2, 2, 2]}, {"id": "h", "type": "cylinder", "op": "cut", "r": 0.5, "length": 3, "axis": "z"}]}, h=0.03)
    assert b["volume"] == pytest.approx(8 - math.pi * 0.25 * 2, rel=3e-3)


def test_hollow_shell_and_mirror():
    shell = kernel.info({"features": [{"id": "s", "type": "sphere", "r": 1.0, "shell": 0.1}]}, h=0.02)
    assert shell["volume"] == pytest.approx(4 / 3 * math.pi * (1.05**3 - 0.95**3), rel=0.03)
    m = kernel.info({"features": [{"id": "s", "type": "sphere", "r": 0.5, "translate": [0, 2, 0], "mirror": True}]}, h=0.03)
    assert m["volume"] == pytest.approx(2 * 4 / 3 * math.pi * 0.125, rel=5e-3)


def test_curve_tools_revolve_extrude_loft_sheet_volumes():
    def vol(f, h=0.02):
        return kernel.info({"features": [f]}, h=h)["volume"]

    # revolve: a rectangle profile is a cylinder, a triangle is a cone
    assert vol({"id": "c", "type": "revolve", "profile": [[0, 0], [0, 1], [2, 1], [2, 0]]}) == pytest.approx(2 * math.pi, rel=3e-3)
    assert vol({"id": "k", "type": "revolve", "profile": [[0, 0], [0, 1], [2, 0]]}) == pytest.approx(2 * math.pi / 3, rel=3e-3)
    # extrude: unit square, height 0.5
    assert vol({"id": "e", "type": "extrude", "outline": [[0, 0], [1, 0], [1, 1], [0, 1]], "height": 0.5}) == pytest.approx(0.5, rel=3e-3)
    # loft: square side 1 to side 2 over length 1 is a square frustum, h/3 (A1 + A2 + sqrt(A1 A2)) = 7/3
    sq = lambda a: [[-a / 2, -a / 2], [a / 2, -a / 2], [a / 2, a / 2], [-a / 2, a / 2]]  # noqa: E731
    lo = {"id": "l", "type": "loft", "sections": [{"x": 0, "points": sq(1)}, {"x": 1, "points": sq(2)}]}
    assert vol(lo) == pytest.approx(7 / 3, rel=3e-3)
    # sheet: a flat unit patch of thickness t is the square swept by a ball of radius t/2
    net = [[i / 3, j / 3, 0.0] for j in range(4) for i in range(4)]
    r = 0.05
    exact_sheet = 2 * r + 4 * math.pi * r**2 / 2 + 4 / 3 * math.pi * r**3
    assert vol({"id": "s", "type": "sheet", "control": net, "thickness": 2 * r}) == pytest.approx(exact_sheet, rel=0.02)


def test_new_tools_are_listed_and_showcase_builds():
    import json
    from pathlib import Path

    c = kernel.check(json.loads((Path(__file__).resolve().parents[1] / "projects" / "tools_showcase.json").read_text(encoding="utf-8")))
    assert c["ok"], c


def test_distance_sign_and_section():
    d = kernel.distance({"features": [{"id": "s", "type": "sphere", "r": 1.0}]}, np.array([[0, 0, 0], [2, 0, 0], [0, 0.5, 0]], float))
    assert d[0] < 0 < d[1] and d[2] == pytest.approx(-0.5, abs=1e-9)
    seg = kernel.section({"features": [{"id": "s", "type": "sphere", "r": 1.0}]}, "z", 0.0, 0.02)
    assert np.allclose(np.hypot(seg[:, 0], seg[:, 1]), 1.0, atol=0.01)


def test_bad_designs_are_reported_not_crashed():
    c = kernel.check({"features": [{"id": "x", "type": "teapot"}, {"id": "f", "type": "fuselage", "stations": [[1, 1, 1], [0, 1, 1]]}]})
    assert not c["ok"] and len(c["errors"]) == 3  # both bad features, and nothing left to build


def test_exact_solutions_match_tables():
    o = exact.oblique_shock(2.0, 10.0)
    assert o["beta_deg"] == pytest.approx(39.31, abs=0.02) and o["p2_p1"] == pytest.approx(1.7066, abs=1e-3)
    assert exact.taylor_maccoll(2.0, 10.0)["shock_deg"] == pytest.approx(31.2, abs=0.2)
    assert exact.spheroid(4.0)["v_max"] == pytest.approx(1.0816, abs=2e-3)


def test_sod_shock_tube_against_exact(tmp_path):
    case = {"case": "sod", "grid": {"uniform": {"n": [400, 2, 2], "lo": [0, 0, 0], "hi": [1, 0.005, 0.005]},
            "bc": {"x": ["extrapolate", "extrapolate"], "y": ["periodic", "periodic"], "z": ["periodic", "periodic"]}},
            "model": {"viscous": False, "sgs": "none", "wall_model": False, "cfl": 0.6}, "run": {"steps": 100000, "t_end": 0.2, "log_every": 1000}}
    res = runner.run_special(case, tmp_path)
    p = runner.plane(tmp_path, "plane_z")
    ex = exact.sod(p["coord_u"], res["time"])
    assert np.mean(np.abs(p["rho"][0] - ex[0])) < 0.004  # second-order MUSCL-HLLC on 400 cells gives about 0.002


def test_quick_analysis_of_a_design():
    des = {"features": [{"id": "body", "type": "fuselage", "stations": [[0, 0.02, 0.02, 0, 2], [2, 0.4, 0.4, 0, 2], [8, 0.5, 0.5, 0, 2], [10, 0.3, 0.3, 0, 2]]},
                        {"id": "wing", "type": "wing", "span": 3, "root_chord": 3, "taper": 0.3, "sweep": 40, "tc": 0.05, "translate": [4, 0.4, 0], "mirror": True}]}
    a = analyse.analyse(des, h=0.08)
    assert 2.0 < a["lattice"]["M0.3"]["CL_alpha_per_rad"] < 4.5
    assert a["wave_drag_CD"]["M1.6"] > 0



def _sheet_wing(tc: float) -> dict:
    return {"id": "w", "type": "wing", "span": 3.0, "root_chord": 1.0, "taper": 1.0, "sweep": 0.0, "tc": tc, "profile": "naca", "translate": [0, 0, 0], "mirror": True}


_SHEET_REF = {"length": 1.0, "area": 6.0, "chord": 1.0, "span": 6.0, "moment_point": [0.25, 0, 0]}


def test_wing_thinner_than_a_cell_is_walled(tmp_path):
    """A 2 % wing on 12 cells per chord is 0.24 cells thick: no cell centre lies inside it. It used to be missed
    entirely (no wall, C_L 0); it must be closed by face walls between the cell centres it passes."""
    r = runner.run({"features": [_sheet_wing(0.02)]}, runner.Flight(mach=0.5, alpha=4.0, altitude=0.0), _SHEET_REF, tmp_path,
                   runner.Numerics(cells_per_length=12, steps=2, cfl=1.2, viscous=False, sgs="none", wall_model=False, domain=[2, 3, 1.0, 2]))
    assert r["wall_faces"] >= 400  # 18 spanwise x 12 chordwise cells on the half model, the whole planform (432 measured)


@pytest.mark.slow
def test_sheet_wing_lift_matches_vortex_lattice(tmp_path):
    """Inviscid 2 % wing, aspect ratio 6, Mach 0.5, 4 deg: lift within 10 % of the vortex lattice (1.02 measured, 0 before
    the face walls)."""
    from aerosynapse.fighter import vlm
    w = _sheet_wing(0.02)
    lat = vlm.derivatives([w], 0.5, 6.0, 1.0, 6.0, 0.25, nc=8, ns=24).CL_alpha * math.radians(4.0)
    r = runner.run({"features": [w]}, runner.Flight(mach=0.5, alpha=4.0, altitude=0.0), _SHEET_REF, tmp_path,
                   runner.Numerics(cells_per_length=12, steps=2000, cfl=1.2, residual_drop=1e-5, viscous=False, sgs="none", wall_model=False,
                                   domain=[2, 3, 1.0, 2], force_band=0.005))
    assert r["coefficients"]["CL"] == pytest.approx(lat, rel=0.10)

@pytest.mark.slow
def test_studio_builds():
    exe = kernel.build("AeroSynapseStudio")
    assert exe.exists() and exe.stat().st_size > 100_000
