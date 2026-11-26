"""Turbulence and instability analysis, and the setup schema that drives every analysis form."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aerosynapse.sim import setup as SU
from aerosynapse.sim import turbulence as T

ROOT = Path(__file__).resolve().parents[1]


def test_rayleigh_solver_matches_michalke_for_the_tanh_mixing_layer():
    res = T.most_unstable(np.tanh, lambda z: -2 * np.tanh(z) / np.cosh(z) ** 2, 12.0, np.linspace(0.40, 0.49, 19), 80)
    best = max(res["best"].values(), key=lambda m: m["growth"])
    assert best["growth"] == pytest.approx(0.1897, rel=0.02)  # Michalke (1964), U = tanh z
    assert best["k"] == pytest.approx(0.4446, abs=0.02)
    assert abs(best["c"][0]) < 1e-6  # the unstable wave travels at the mean speed (zero here)


def test_gaussian_wake_has_a_sinuous_mode_stronger_than_the_varicose():
    Uf, d2 = T.gauss_wake(0.5)
    res = T.most_unstable(Uf, d2, 7.0, np.linspace(0.1, 2.0, 20), 80)
    assert res["best"]["sinuous"]["growth"] > res["best"]["varicose"]["growth"] > 0


def test_mack_critical_n_and_flat_plate_friction():
    assert T.mack_ncrit(0.07) == pytest.approx(9.0, abs=0.05)  # quiet flight air gives the classic N = 9
    assert T.mack_ncrit(1.0) < T.mack_ncrit(0.1)
    re = 1e7
    assert T.friction_cd(re, 0.0) == pytest.approx(0.074 / re**0.2)
    assert T.friction_cd(re, 0.5) < T.friction_cd(re, 0.0)  # a laminar run lowers friction


def test_blasius_layer_from_thwaites():
    s = np.linspace(1e-4, 1.0, 400)  # flat plate: constant edge velocity, no transition (N crit huge)
    bl = T.boundary_layer(s, np.full_like(s, 10.0), 1.5e-5, 1e9)
    th_blasius = 0.664 * np.sqrt(1.5e-5 * s[-1] / 10.0)
    assert bl["theta"][-1] == pytest.approx(th_blasius, rel=0.02)
    assert bl["H"][-1] == pytest.approx(2.61, abs=0.01)


def test_setup_schema_defaults_ranges_and_choices():
    for kind in SU.SCHEMA:
        d = SU.defaults(kind)
        assert SU.merged(kind, None) == d
        for f in SU.SCHEMA[kind]["fields"]:
            if "min" in f and f["type"] in ("num", "int"):
                assert f["min"] <= f["default"] <= f["max"], (kind, f["key"])
            if f["type"] == "choice":
                assert f["default"] in f["choices"], (kind, f["key"])
    with pytest.raises(ValueError, match="outside"):
        SU.merged("cfd", {"mach": 9.0})
    with pytest.raises(ValueError, match="not one of"):
        SU.merged("structure", {"code": "far25"})
    assert SU.merged("cfd", {"viscous": "false"})["viscous"] is False
    for name, p in SU.PRESETS.items():  # presets only set fields the form has, within range
        SU.merged("cfd", {**p, "preset": name})


def test_schema_file_matches_the_code(tmp_path):
    f = SU.write_schema(tmp_path / "s.json")
    d = json.loads(f.read_text(encoding="utf-8"))
    assert d["kinds"] == list(SU.SCHEMA)
    shipped = json.loads((ROOT / "assets" / "sim_setup_schema.json").read_text(encoding="utf-8"))
    assert shipped == d, "assets/sim_setup_schema.json is stale: run `python -m aerosynapse.cli simulate schema -`"


def test_turbulence_analysis_without_a_flow_run(tmp_path):
    d = json.loads((ROOT / "assets" / "templates" / "jet_conventional.json").read_text(encoding="utf-8"))
    r = T.run(d, tmp_path, {"mach": 0.5, "altitude": 3000, "alpha": 1.0, "turbulence_intensity": 0.5, "wake_stations": "0.2, 0.6"})
    tr = r["transition"]
    assert 0 <= tr["sides"]["upper"]["x_tr_c"] <= 1 and tr["n_crit"] == pytest.approx(T.mack_ncrit(0.5))
    xs = [row["x_tr_upper"] for row in tr["sweep"]]
    assert xs[0] >= xs[-1]  # more free-stream turbulence, earlier transition
    assert len(r["wakes"]) == 2 and all(w["source"].startswith("model") for w in r["wakes"])
    assert r["rayleigh_check"]["growth"] == pytest.approx(0.1897, rel=0.02)
    assert (tmp_path / "1_transition.png").exists() and (tmp_path / "2_wake_instability.png").exists()
