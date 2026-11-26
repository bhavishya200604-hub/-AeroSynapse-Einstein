"""Structures (finite elements, panels, flutter, design codes, wing sizing) and the workflow's export writers."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from aerosynapse.cad import kernel
from aerosynapse.structures import aeroelastic as ae
from aerosynapse.structures import codes as C
from aerosynapse.structures import fem, panels

ROOT = Path(__file__).resolve().parents[1]


def cantilever(n: int = 10, L: float = 2.0) -> fem.Frame:
    F = fem.Frame(np.c_[np.linspace(0, L, n + 1), np.zeros(n + 1), np.zeros(n + 1)])
    s = fem.Section(70e9, 27e9, 1e-3, 1e-6, 1e-6, 2e-6, rho=2700)
    for i in range(n):
        F.add(i, i + 1, s)
    return F


def test_cantilever_deflection_twist_and_modes():
    F = cantilever()
    ld = np.zeros((11, 6))
    ld[-1, 2], ld[-1, 3] = 1000.0, 100.0
    u = F.solve(ld, list(range(6)))["u"]
    assert u[-1, 2] == pytest.approx(1000 * 8 / (3 * 70e9 * 1e-6), rel=1e-9)
    assert u[-1, 3] == pytest.approx(100 * 2 / (27e9 * 2e-6), rel=1e-9)
    f = F.modes(list(range(6)), 4)["freq_hz"]
    ref = np.array([1.8751, 1.8751, 4.6941, 4.6941]) ** 2 / (2 * np.pi) * np.sqrt(70e9 * 1e-6 / (2.7 * 16))
    assert f == pytest.approx(ref, rel=2e-3)


def test_truss_and_closed_section():
    T = fem.Frame(np.array([[0, 0, 0], [2, 0, 0], [1, 0, 1.0]]))
    T.truss = [(0, 2, 200e9, 1e-4), (1, 2, 200e9, 1e-4)]
    ld = np.zeros((3, 6))
    ld[2, 2] = -1000
    u = T.solve(ld, [0, 1, 2, 6, 7, 8, 13])["u"]
    assert T.truss_forces(u) == pytest.approx([-1000 / math.sqrt(2)] * 2, rel=1e-9)
    p = panels.closed_section(np.array([0, 1, 1, 0.0]), np.array([0, 0, 0.5, 0.5]), np.full(4, 0.01))
    assert p["A_enclosed"] == pytest.approx(0.5)
    assert p["J"] == pytest.approx(4 * 0.25 / (3 / 0.01))


def test_flutter_matches_hodges_pierce_and_buckling_formulas():
    fl = ae.flutter(20, -0.2, 0.1, 0.24, 0.4)
    assert fl["U_flutter"] == pytest.approx(2.17, rel=0.03)
    assert abs(ae.theodorsen(1e-4) - 1) < 2e-3 and abs(ae.theodorsen(50.0) - (0.5 - 1j / 400)) < 2e-4
    s = panels.plate_compression(70e9, 0.002, 0.1)
    assert s == pytest.approx(4 * math.pi**2 * 70e9 / (12 * (1 - 0.33**2)) * 0.02**2)
    assert panels.interaction(1.0, 2.0, 0.0, 1.0) == pytest.approx(1.0)


def test_design_codes_and_vn():
    code = C.CODES["mil_fighter"]
    assert code.ultimate(code.n_pos) == pytest.approx(13.5)
    assert C.margin(150.0, 100.0, code) == pytest.approx(0.0)
    vn = C.vn_diagram(code, 12000 * 9.81, 30.0, 1.2, -0.7, 3.5, 3.5, 1.225, 320, 400)
    assert vn["V_A"] == pytest.approx(vn["V_S1"] * 3, rel=1e-9)
    assert 0 < vn["K_g"] < 0.88


def test_wing_box_sizing_closes_every_margin():
    from aerosynapse.structures import wing as SW

    d = json.loads((ROOT / "results" / "fighter" / "freeform.design.json").read_text())
    r = SW.analyse(d, 11000, "mil_fighter", "al7075", 330.0, v_d_eas=400, fuel_kg=3000)
    w = r["wing"]
    assert w["all_margins_positive"]
    assert 200 < w["box_mass_both_sides_kg"] < 3000
    assert w["box_mass_both_sides_kg"] >= w["strength_sized_box_mass_kg"] - 1e-6
    assert w["root_bending_moment_kNm"] > 0 and w["tip_deflection_limit_m"] > 0


def test_new_transformations_keep_exact_volumes():
    def vol(f, h=0.02):
        return kernel.info({"features": [f]}, h=h)["volume"]

    assert vol({"id": "s", "type": "sphere", "r": 1, "stretch": [2, 1, 0.5]}) == pytest.approx(4 / 3 * math.pi, rel=2e-3)
    assert vol({"id": "b", "type": "box", "size": [1, 1, 1], "shear": [0.5, 0]}) == pytest.approx(1.0, rel=2e-3)
    assert vol({"id": "c", "type": "cylinder", "r": 0.5, "length": 2, "axis": "x", "taper_rate": 0.3}) == pytest.approx(math.pi * 0.25 * (2 + 0.06), rel=3e-3)
    assert vol({"id": "c", "type": "cylinder", "r": 0.1, "length": 3, "axis": "x", "bend": 0.5}, h=0.01) == pytest.approx(math.pi * 0.03, rel=5e-3)
    i = kernel.info({"features": [{"id": "b", "type": "box", "size": [2, 0.2, 0.2], "pivot": [-1, 0, 0], "rotate": [0, 0, 90]}]}, h=0.02)
    assert i["bbox_lo"][1] == pytest.approx(0, abs=1e-6) and i["bbox_hi"][1] == pytest.approx(2, abs=1e-6)
    g = vol({"id": "l", "type": "lattice", "size": [1, 1, 1], "cell": 0.25, "wall": 0.02}, h=0.01)
    assert g == pytest.approx(3.09 * 0.02 / 0.25, rel=0.06)  # gyroid wall: area per volume 3.09 / cell


def test_export_writers(tmp_path):
    from aerosynapse.workflow import export as X

    d = {"features": [{"id": "s", "type": "sphere", "r": 1.0}]}
    kernel.mesh(d, tmp_path / "s.obj", h=0.1)
    V, F = X.read_obj(tmp_path / "s.obj")
    X.write_glb(tmp_path / "s.glb", V, F, "s")
    b = (tmp_path / "s.glb").read_bytes()
    magic, ver, total = struct.unpack("<4sII", b[:12])
    assert magic == b"glTF" and ver == 2 and total == len(b)
    X.write_step(tmp_path / "s.step", V, F, "s")
    t = (tmp_path / "s.step").read_text()
    assert t.startswith("ISO-10303-21;") and t.count("FACE(") == len(F) and "FACETED_BREP(" in t
    outl = X.silhouette(V, F, (0, 1), res=0.02)
    assert outl and np.allclose(np.hypot(*outl[0].T), 1.0, atol=0.05)
    X.write_dxf(tmp_path / "s.dxf", {"TOP": (outl, np.zeros(2))}, [("TITLE", 0, 0, 5, "sphere")])
    assert (tmp_path / "s.dxf").read_text().rstrip().endswith("EOF")
