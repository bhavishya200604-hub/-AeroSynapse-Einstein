"""The browser apps run natively: the Streamlit stand-in, the Plotly redraw, the design link and real pages."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("plotly")

from aerosynapse.nativelab import context as C  # noqa: E402
from aerosynapse.nativelab import run as R  # noqa: E402

CTX = {
    "part": {
        "id": "pinion1",
        "material": {"key": "steel_mild", "label": "Carbon steel", "rho": 7850, "k": 50, "cp": 486},
        "size_m": [0.044, 0.044, 0.02],
        "rpm": 300,
    },
    "environment": {
        "medium": "Air",
        "T_fluid_C": 15,
        "T_part_C": 60,
        "U_m_s": 2.0,
        "fluid": {"rho": 1.225, "mu": 1.8e-5, "k": 0.025, "cp": 1006, "Pr": 0.71, "beta": 1 / 288},
    },
    "design": {
        "features": [{"type": "wing", "span": 4.5, "root_chord": 5, "taper": 0.25, "sweep": 40, "mirror": True}]
    },
}


def test_design_link_numbers():
    d = C.derived(CTX)
    Re = 2.0 * 0.044 / (1.8e-5 / 1.225)
    assert d["Re"][0] == pytest.approx(Re, rel=1e-9)
    h = 0.664 * Re**0.5 * 0.71 ** (1 / 3) * 0.025 / 0.044
    assert d["h"][0] == pytest.approx(h, rel=1e-9)
    assert d["Bi"][0] == pytest.approx(h * 0.01 / 50, rel=1e-9)
    assert d["wing_area"][0] == pytest.approx(2 * 4.5 * 5 * 1.25 / 2)
    assert d["aspect_ratio"][0] == pytest.approx(9.0**2 / d["wing_area"][0])
    lab = C.label_defaults(CTX)
    assert any(v[0] == pytest.approx(d["Bi"][0], rel=1e-5) for k, v in lab.items() if "Bi" in k)


def test_every_app_is_listed():
    apps = {a["key"]: a for a in R.listing()["apps"]}
    assert set(apps) == {"thermal_lab", "airfrans", "thermal_app", "designer", "workbench"}
    assert len(apps["thermal_lab"]["pages"]) >= 10 and len(apps["designer"]["pages"]) >= 10


@pytest.mark.parametrize(
    ("app", "page"), [("thermal_lab", "Cavity flow"), ("workbench", "Physics"), ("designer", "Materials")]
)
def test_pages_run_natively(tmp_path, app, page):
    values = {"fl_grid": 12, "fl_frames": 3, "fl_tend": 0.5} if page == "Cavity flow" else {}
    done = R.run_page({"app": app, "page": page, "values": values, "context": CTX}, tmp_path)
    assert done["ok"], json.loads((tmp_path / "blocks.json").read_text())
    ui = json.loads((tmp_path / "ui.json").read_text())["widgets"]
    blocks = json.loads((tmp_path / "blocks.json").read_text())["blocks"]
    assert blocks[0]["type"] == "context" and "Inputs taken from the design" in blocks[0]["text"]
    assert any(b["type"] in ("figure", "table", "metric") for b in blocks)
    for b in blocks:
        if b["type"] in ("figure", "table"):
            assert (tmp_path / b["file"]).exists()
    if page == "Cavity flow":  # the design's Reynolds and Prandtl numbers are the starting values
        src = {w["label"].split(" ")[0]: w["source"] for w in ui if w["kind"] == "number_input"}
        assert src["Re"].startswith("design") and src["Pr"].startswith("design")


def test_plotly_redraw(tmp_path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from aerosynapse.nativelab.plotly2mpl import render

    fig = make_subplots(rows=1, cols=2, specs=[[{}, {"type": "scene"}]])
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[3, 1, 2], name="a"), 1, 1)
    fig.add_trace(go.Heatmap(z=[[1, 2], [3, 4]]), 1, 1)
    fig.add_trace(go.Surface(z=[[1, 2], [2, 3]]), 1, 2)
    render(fig, tmp_path / "f.png")
    assert (tmp_path / "f.png").stat().st_size > 5000
