"""Part-design modifiers, sheet metal, templates, engineering checks and the full-3D flow post-processing."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from aerosynapse import engineering as E
from aerosynapse.cad import kernel

ROOT = Path(__file__).resolve().parents[1]


def vol(features, h=0.01):
    return kernel.info({"features": features}, h=h)["volume"]


def square_extrude(**kw):
    f = {"id": "e", "type": "extrude", "outline": [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], "height": 0.4, "smooth": False}
    f.update(kw)
    return f


def test_draft_gives_the_frustum_volume():
    a, H, deg = 1.0, 0.4, 10.0
    t = math.tan(math.radians(deg))
    b = a - 2 * H * t  # each wall moves in by H tan(draft) over the height
    exact = H / 3 * (a * a + a * b + b * b)
    assert vol([square_extrude(draft=deg)]) == pytest.approx(exact, rel=1e-2)
    assert vol([square_extrude()]) == pytest.approx(a * a * H, rel=5e-3)


def test_chamfer_adds_a_bevel_at_the_join():
    base = {"id": "b", "type": "box", "size": [1, 1, 1], "round": 0}
    post = {"id": "p", "type": "box", "size": [0.4, 0.4, 0.6], "round": 0, "translate": [0, 0, 0.8]}
    plain = vol([base, post], h=0.015)
    bevel = vol([base, dict(post, chamfer=0.08)], h=0.015)
    assert plain == pytest.approx(1 + 0.4 * 0.4 * 0.6, rel=1e-2)
    assert bevel > plain * 1.005  # a 45-degree fillet of material around the post foot


def test_bent_plate_volume_is_thickness_width_and_flat_length():
    pr = [[0, 0], [0.4, 0], [0.4, 0.3], [0.7, 0.3]]
    t, w, r = 0.01, 0.3, 0.02
    f = {"id": "s", "type": "bentplate", "profile": pr, "thickness": t, "width": w, "bend_radius": r}
    straight = 0.4 + 0.3 + 0.3
    v = vol([f], h=0.004)
    assert v == pytest.approx(t * w * straight, rel=0.05)  # bends shorten the mid-line a little


def test_every_template_is_a_buildable_design():
    index = json.loads((ROOT / "assets" / "templates" / "index.json").read_text(encoding="utf-8"))
    keys = [t["key"] for t in index]
    assert "blank" in keys and len(keys) >= 20
    groups = {t["group"] for t in index}
    assert {"Start", "Jet fighters"} <= groups
    for t in index:
        d = json.loads((ROOT / "assets" / "templates" / (t["key"] + ".json")).read_text(encoding="utf-8"))
        assert isinstance(d["features"], list)
        assert (ROOT / "assets" / "templates" / (t["key"] + ".png")).exists()
    d = json.loads((ROOT / "assets" / "templates" / "jet_conventional.json").read_text(encoding="utf-8"))
    assert kernel.info(d, h=0.1)["volume"] > 1.0


def test_quasi_isotropic_laminate_is_isotropic_in_plane():
    r = E.laminate("[0/45/-45/90]s", "cfrp_im7")
    Ex, Ey, Gxy, nu = r["Ex_GPa"], r["Ey_GPa"], r["Gxy_GPa"], r["nuxy"]
    assert Ex == pytest.approx(Ey, rel=1e-6)
    assert Gxy == pytest.approx(Ex / (2 * (1 + nu)), rel=1e-6)
    assert E.parse_layup("[0/90]s") == [0.0, 90.0, 90.0, 0.0]


def test_rules_are_evaluated_safely(tmp_path):
    assert E.safe_eval("2 * max(a, 3) + abs(-1)", {"a": 4}) == 9
    with pytest.raises(ValueError, match="only min, max"):
        E.safe_eval("__import__('os').system('echo')", {})
    with pytest.raises(ValueError, match="not allowed"):
        E.safe_eval("a.__class__", {"a": 1})
    d = json.loads((ROOT / "assets" / "templates" / "jet_conventional.json").read_text(encoding="utf-8"))
    r = E.rules(d, tmp_path)
    assert len(r["rules"]) > 0 and all(x["pass"] is not None for x in r["rules"]) and (tmp_path / "summary.txt").exists()


def test_clash_reports_overlapping_parts(tmp_path):
    d = {"features": [{"id": "a", "type": "box", "size": [1, 1, 1], "round": 0},
                      {"id": "b", "type": "box", "size": [1, 1, 1], "round": 0, "translate": [0.5, 0, 0]},
                      {"id": "c", "type": "sphere", "r": 0.2, "translate": [5, 0, 0]}]}
    r = E.clash(d, tmp_path, h=0.04)
    pairs = {tuple(sorted((c["a"], c["b"]))): c for c in r["pairs"]}
    assert ("a", "b") in pairs and pairs[("a", "b")]["volume_m3"] == pytest.approx(0.5, rel=0.1)
    assert not any("c" in p for p in pairs)


@pytest.mark.skipif(not (ROOT / "results" / "cfd3d" / "tiny" / "volume.bin").exists(), reason="needs a saved full-3D flow volume")
def test_full_3d_flow_postprocessing(tmp_path):
    from aerosynapse.sim import flow3d

    d = tmp_path / "tiny"
    shutil.copytree(ROOT / "results" / "cfd3d" / "tiny", d, ignore=shutil.ignore_patterns("*.png", "viz3d.json", "iso_*.obj"))
    flow3d.postprocess(d)
    viz = json.loads((d / "viz3d.json").read_text(encoding="utf-8"))
    assert len(viz["lines"]) > 100  # streamlines all around the body, both sides
    ys = [p[1] for ln in viz["lines"] for p in ln["pts"][:1]]
    assert min(ys) < 0 < max(ys)  # full model, not a half model
    assert (d / "summary.txt").exists() and (d / "2_surface_pressure.png").exists()


def test_new_project_in_a_chosen_folder(tmp_path):
    import subprocess
    import sys

    (tmp_path / "plane1").mkdir()
    src = tmp_path / "plane1" / "design.json"
    src.write_text(json.dumps({"name": "plane1", "features": []}), encoding="utf-8")
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, "-m", "aerosynapse.cli", "workflow", "new", "plane1", "--from", str(src), "--where", str(tmp_path)],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    meta = json.loads((tmp_path / "plane1" / "project.json").read_text(encoding="utf-8"))
    assert meta["name"] == "plane1" and (tmp_path / "plane1" / "input" / "design.json").exists()


def test_capsule_has_its_exact_volume():
    r, L = 0.25, 5.0
    f = {"id": "c", "type": "capsule", "a": [4, 0, -0.4], "b": [9, 0, -0.4], "r": r}
    assert vol([f], h=0.01) == pytest.approx(math.pi * r * r * L + 4 / 3 * math.pi * r**3, rel=5e-3)


def test_display_thickening_is_for_the_editor_only():
    plate = {"id": "p", "type": "bentplate", "profile": [[0, 0], [0.4, 0], [0.4, 0.3]], "thickness": 0.01, "width": 0.3, "bend_radius": 0.02}
    wing = json.loads((ROOT / "assets" / "templates" / "jet_conventional.json").read_text(encoding="utf-8"))
    wing["features"] = [f for f in wing["features"] if f["id"] == "wing"]
    plain = kernel.info({"features": [plate]}, h=0.003)["volume"]
    assert plain == pytest.approx(0.01 * 0.3 * 0.7, rel=0.06)  # exports and analyses: the true plate
    assert kernel.info({"features": [plate]}, h=0.045)["volume"] < 0.5 * plain  # a 10 mm plate falls apart on a 45 mm mesh...
    assert kernel.info({"features": [plate], "display_h": 0.045}, h=0.045)["volume"] > plain  # ...unless the editor draws it thicker
    assert kernel.info(wing, h=0.05)["volume"] < kernel.info({**wing, "display_h": 0.05}, h=0.05)["volume"]  # only the editor key thickens
