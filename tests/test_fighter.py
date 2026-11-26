"""Fighter design space, vortex lattice, materials, performance model and the data store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aerosynapse.aircraft import materials as mt
from aerosynapse.aircraft.atmosphere import isa
from aerosynapse.cad import kernel
from aerosynapse.fighter import design as D
from aerosynapse.fighter import performance as P
from aerosynapse.fighter import vlm
from aerosynapse.store import shards

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "fighter" / "baseline_design.json"
ZONES = ("radome", "fwd_fuselage", "centre_fuselage", "wing_box", "wing_le", "control", "aft_fuselage", "nozzle", "gear")


def test_lattice_matches_classic_wings():
    rect = {"type": "wing", "span": 3.0, "root_chord": 1.0, "taper": 1.0, "sweep": 0.0, "translate": [0, 0, 0], "mirror": True}
    d = vlm.derivatives([rect], 0.0, 6.0, 1.0, 6.0, 0.25, nc=6, ns=24)
    assert d.CL_alpha == pytest.approx(2 * np.pi * 6 / (2 + np.sqrt(40)), rel=0.07)  # Helmbold
    assert 0.95 < 1 / (np.pi * 6 * d.K) < 1.03
    tap = {"type": "wing", "span": 4.0, "root_chord": 1.428571, "taper": 0.4, "sweep": 0.0, "translate": [0, 0, 0], "mirror": True}
    S = 0.5 * 1.4 * 1.428571 * 8
    d0 = vlm.derivatives([tap], 0.0, S, 1.0, 8.0, 0.25, nc=6, ns=30)
    d6 = vlm.derivatives([tap], 0.6, S, 1.0, 8.0, 0.25, nc=6, ns=30)
    pg = lambda b: 2 * np.pi * 8 / (2 + np.sqrt(64 * b * b + 4))  # noqa: E731
    assert d6.CL_alpha / d0.CL_alpha == pytest.approx(pg(0.8) / pg(1.0), rel=0.03)  # Prandtl-Glauert trend
    assert abs(d0.Cn_beta) < 1e-6


def test_materials_follow_temperature():
    cool = {c.zone: c.material for c in mt.select({k: 300.0 for k in ZONES})}
    hot = {c.zone: c.material for c in mt.select({k: 520.0 for k in ZONES})}
    assert cool["radome"] in ("glass_epoxy", "cfrp_bmi")
    assert mt.LIBRARY[cool["nozzle"]].T_max > 900
    assert mt.LIBRARY[hot["wing_box"]].T_max >= 540  # aluminium and epoxy composites are out at 520 K
    assert mt.LIBRARY["al7075"].strength(473) < 0.5 * mt.LIBRARY["al7075"].sigma


def test_tolerance_study_gives_spread():
    t = mt.tolerance_study(mt.select({}), isa(11000.0), 0.85, 15.0, 150.0, n=300)
    assert t["drag_N"]["p95"] > t["drag_N"]["p05"] >= 0
    assert t["mass_kg"]["p95"] > 0 > t["mass_kg"]["p05"]


@pytest.mark.skipif(not BASE.exists(), reason="reference design not generated")
def test_reference_fighter_is_feasible_and_plausible():
    base = json.loads(BASE.read_text())["variables"]
    cal = json.loads((ROOT / "results" / "fighter" / "calibration.json").read_text())["calib"]
    r = P.evaluate("conventional", base, P.Mission(calib=cal))
    assert r.feasible, {k: v for k, v in r.constraints.items() if v > 0}
    assert r.empty == pytest.approx(8570, rel=0.01)
    assert 0.015 < r.CD0 < 0.03 and 10000 < r.W0 < 16000
    assert r.metrics["Cn_beta"] > 0


def test_every_family_builds_valid_geometry():
    for fam in D.FAMILIES:
        d = D.decode(fam, np.full(len(D.variables(fam)), 0.5))
        assert kernel.check(D.build(d, fam))["ok"], fam
        assert D.encode(fam, d) == pytest.approx(np.full(len(D.variables(fam)), 0.5), abs=1e-9)


def test_topology_options_switch_features():
    base = D.decode("freeform", np.full(len(D.variables("freeform")), 0.5))
    ids = {f["id"] for f in D.build({**base, "c_size": 0.0, "b_count": 0.0, "d_radius": 0.0}, "x")["features"]}
    assert "canard" not in ids and "blades" not in ids and "inlet_duct" not in ids
    ids = {f["id"] for f in D.build({**base, "c_size": 0.1, "b_count": 2.0, "d_radius": 0.3}, "x")["features"]}
    assert {"canard", "blades", "inlet_duct"} <= ids


def test_shard_store_round_trip_and_bounded_stream(tmp_path):
    w = shards.ShardWriter(tmp_path, "recs", ["a"], max_records=7)
    for i in range(20):
        w.add({"a": i})
    w.close()
    assert [r["a"] for r in shards.records(tmp_path / "recs", verify=True)] == list(range(20))
    aw = shards.ArrayWriter(tmp_path, "arr", (3, 4), max_bytes=4 * 12 * 5)
    for i in range(23):
        aw.add(np.full((3, 4), i, np.float32))
    aw.close()
    rd = shards.ArrayReader(tmp_path / "arr")
    assert len(rd) == 23 and len(rd.man["shards"]) == 5 and rd.verify()
    seen = np.concatenate([b[:, 0, 0] for b in shards.stream(rd, batch=4, buffer=8, seed=1)])
    assert sorted(seen.tolist()) == list(range(23))


def test_streaming_training_from_disk_learns_and_resumes(tmp_path):
    from aerosynapse.store import train as T

    rng = np.random.default_rng(0)
    w = shards.ArrayWriter(tmp_path, "rows", (4,), max_bytes=4 * 4 * 150)
    for _ in range(900):
        x = rng.uniform(-1, 1, 2)
        w.add(np.array([x[0], x[1], np.sin(2 * x[0]) + x[1] ** 2, x[0] * x[1]], np.float32))
    w.close()
    rep = T.train(tmp_path / "rows", 2, tmp_path / "net", epochs=25, batch=64, buffer=256, width=48, depth=3)
    assert min(rep["val_r2"]) > 0.9
    assert set(rep["val_shard_ids"]).isdisjoint(rep["train_shard_ids"])
    again = T.train(tmp_path / "rows", 2, tmp_path / "net", epochs=25, batch=64, buffer=256, width=48, depth=3)  # resumes at the last epoch
    assert len(again["history"]) == 25


def test_surrogate_prediction_labels_reliability():
    from aerosynapse.store import predict as SP

    if not SP.MODEL.exists():
        pytest.skip("surrogate not trained yet (scripts/train_surrogate.py)")
    d = json.loads((Path(__file__).resolve().parents[1] / "results" / "fighter" / "freeform.design.json").read_text(encoding="utf-8"))
    p = SP.predict(d)
    assert p and all(np.isfinite(v["value"]) for v in p.values())
    assert any(v["trusted"] for v in p.values())
    assert all(v["trusted"] == (v["held_out_r2"] >= SP.GOOD_R2) for v in p.values())
    with pytest.raises(ValueError):
        SP.predict({"features": []})
