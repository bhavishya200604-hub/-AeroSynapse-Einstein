"""Phase 8: workbench smoke tests (Streamlit AppTest) and CLI tests."""

import time
from pathlib import Path

import pytest

from aerosynapse.cli import main

APP = str(Path(__file__).parents[1] / "src" / "aerosynapse" / "ui" / "app.py")
pytestmark = pytest.mark.unit


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP, default_timeout=120)


@pytest.mark.parametrize("ws", ["Physics", "Simulation", "ML", "Optimization", "Research"])
def test_workspace_renders(ws):
    pytest.importorskip("streamlit")
    at = _app().run()
    assert not at.exception
    at.sidebar.radio(key="workspace").set_value(ws).run()
    assert not at.exception, at.exception
    assert any("RESEARCH USE ONLY" in w.value for w in at.warning)


def test_simulation_and_optimisation_actions():
    pytest.importorskip("streamlit")
    at = _app().run()
    at.sidebar.radio(key="workspace").set_value("Simulation").run()
    at.button(key="sim_run").click().run()
    assert not at.exception and len(at.metric) == 1
    at.sidebar.radio(key="workspace").set_value("Optimization").run()
    at.slider(key="opt_budget").set_value(40).run()
    at.button(key="opt_run").click().run()
    assert not at.exception
    assert any(s.value.startswith("ACCEPT") for s in at.success) or any(e.value.startswith("REJECT") for e in at.error)


def test_ml_background_job_completes():
    pytest.importorskip("streamlit")
    at = _app().run()
    at.sidebar.radio(key="workspace").set_value("ML").run()
    at.slider(key="ml_iters").set_value(50).run()
    at.button(key="ml_train").click().run()
    job = at.session_state["ml_job"]
    for _ in range(120):
        if job.done:
            break
        time.sleep(0.5)
    at.run()
    assert job.done and job.error is None and not at.exception
    assert any(m.label == "relative L2 error" for m in at.metric)


def test_cli_doctor_and_run(tmp_path, capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "RESEARCH USE ONLY" in out and "numpy" in out
    cfg = tmp_path / "wf.yaml"
    cfg.write_text(
        (Path(__file__).parents[1] / "configs" / "workflow_quick.yaml")
        .read_text()
        .replace("iterations: 600", "iterations: 20")
        .replace("budget: 200", "budget: 30")
    )
    assert main(["run", str(cfg), "--results", str(tmp_path / "res")]) == 0
    out = capsys.readouterr().out
    assert "verdict" in out and (tmp_path / "res" / "workflow_quick" / "report.html").exists()
