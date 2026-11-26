"""CFD design optimisation (with a stand-in solver of known optimum) and the flow-picture helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FALCON = ROOT / "results/cfd3d/falcon_cruise_full"


def _design():
    return json.loads((FALCON / "case.json").read_text(encoding="utf-8"))["design"]


@pytest.mark.skipif(not (FALCON / "case.json").exists(), reason="needs the falcon run")
def test_apply_keeps_wing_area():
    from aerosynapse.sim import optimise as O

    d = _design()
    w0 = O.wing_of(d)
    a0 = 0.5 * w0["span"] * w0["root_chord"] * (1 + w0["taper"])
    d2 = O.apply(d, {"taper": 0.5, "span": 1.15, "sweep": 40.0}, keep_area=True)
    w = O.wing_of(d2)
    assert w["sweep"] == 40.0 and abs(w["span"] - 1.15 * w0["span"]) < 1e-9
    assert abs(0.5 * w["span"] * w["root_chord"] * (1 + w["taper"]) - a0) < 1e-9
    assert O.wing_of(d)["sweep"] == w0["sweep"]  # the original is untouched


@pytest.mark.skipif(not (FALCON / "case.json").exists(), reason="needs the falcon run")
def test_bayesian_loop_finds_known_optimum(tmp_path, monkeypatch):
    """A stand-in for the CFD: L/D peaks at sweep 38, taper 0.42; the loop must get close within 14 runs."""
    from aerosynapse.sim import flow3d
    from aerosynapse.sim import optimise as O

    def fake(design, setup, out):
        w = O.wing_of(design)
        ld = 14 - 0.012 * (w["sweep"] - 38) ** 2 - 60 * (w["taper"] - 0.42) ** 2
        Path(out).mkdir(parents=True, exist_ok=True)
        return {"coefficients": {"CL": 0.2, "CD": 0.2 / ld}, "steps": 100, "stop_reason": "forces"}

    monkeypatch.setattr(flow3d, "run_setup", fake)
    res = O.run(_design(), {"budget": 14, "initial_runs": 6, "vary_twist": False}, tmp_path)
    b = res["best"]
    assert abs(b["sweep"] - 38) < 4 and abs(b["taper"] - 0.42) < 0.06
    assert b["LD"] > 13.7 and res["gain_percent"] > 0
    for f in ("1_optimisation_history.png", "2_variables.png", "3_surrogate_map.png", "4_planforms.png", "history.csv", "best_design.json"):
        assert (tmp_path / f).exists(), f


def test_expected_improvement_prefers_unexplored_good_regions():
    from aerosynapse.sim.optimise import Surrogate

    X = np.array([[0.1], [0.3], [0.5], [0.9]])
    y = (X[:, 0] - 0.7) ** 2
    gp = Surrogate(1).fit(X, y)
    g = np.linspace(0, 1, 201)[:, None]
    ei = gp.expected_improvement(g, float(y.min()))
    assert 0.55 < g[int(np.argmax(ei)), 0] < 0.85


@pytest.mark.skipif(not (FALCON / "volume.bin").exists(), reason="needs the falcon run")
def test_flow_section_fields():
    from aerosynapse.sim import flowviz

    F = flowviz.section_frames(FALCON, ("y", 0.0), n=200)
    T = F["temp"][~F["solid"]]
    assert np.nanpercentile(T, 50) < 0.1  # free stream: no temperature rise
    assert np.nanmax(T) > 0.5  # a stagnation point or the wake
    assert F["solid"].any() and np.isnan(F["vort"][F["solid"]]).all()
    tr = flowviz.Tracers(F, n=400)
    x0 = tr.px.copy()
    tr.step(0.01)
    assert np.mean(tr.px - x0) > 0  # carried downstream
