"""The design workflow: project files, revisions, the concept stage and the grid-convergence routine."""

from __future__ import annotations

import json

import pytest

from aerosynapse.workflow import Project
from aerosynapse.workflow import stages as S


def test_richardson_recovers_order_and_limit():
    h = [0.4, 0.4 / 2**0.5, 0.2]
    f = [1.0 + 0.3 * x**2 for x in h]
    r = S.richardson(f, h)
    assert r["observed_order"] == pytest.approx(2.0, rel=1e-6)
    assert r["extrapolated"] == pytest.approx(1.0, abs=1e-9)
    assert r["monotone"] and r["gci_fine"] > 0
    osc = S.richardson([1.0, 1.2, 1.1], h)
    assert not osc["monotone"] and osc["extrapolated"] == 1.1
    assert osc["gci_fine"] == pytest.approx(1.25 * 0.1 / 1.1)


def test_project_and_concept_stage(tmp_path):
    src = tmp_path / "body.json"
    src.write_text(json.dumps({"features": [{"id": "body", "type": "fuselage", "stations": [[0, 0.02, 0.02, 0, 2], [3, 0.6, 0.5, 0, 2], [9, 0.7, 0.6, 0, 2.5], [12, 0.4, 0.4, 0, 2]]},
                                            {"id": "wing", "type": "wing", "span": 4.0, "root_chord": 4.0, "taper": 0.3, "sweep": 40, "tc": 0.05, "translate": [5, 0.5, 0], "mirror": True}]}))
    p = Project.create("t1", src, where=tmp_path)
    assert p.meta["origin"].startswith("input body") and p.meta["requirements"]["family"] == "conventional"
    with pytest.raises(FileExistsError):
        Project.create("t1", src, where=tmp_path)
    res = S.run_stage(p, "concept", "quick")
    assert p.status("concept") == "done" and res["scorecard"]["volume_m3"] > 0
    assert len(p.meta["revisions"]) == 1 and len(p.meta["revisions"][0]["sha256"]) == 64
    d, k = p.design_before("design")
    assert k == "concept" and d["revision"] == 1
    # rerunning an earlier stage marks the later ones stale
    p.set_stage("design", status="done")
    S.run_stage(p, "concept", "quick")
    assert Project(p.dir).status("design") == "stale"
