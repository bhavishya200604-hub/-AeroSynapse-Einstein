"""AirfRANS integration: name parsing, table I/O, surrogate convergence logic, panel comparison mechanics.
Uses a tiny synthetic table in the AirfRANS *format* (no reported results come from it)."""

import json

import numpy as np
import pytest

from aerosynapse.ml import airfrans as A

pytestmark = pytest.mark.unit


def test_parse_names():
    m = A.parse_name("airFoil2D_SST_43.597_5.932_3.551_3.1_1.0_18.252")
    assert m["U"] == 43.597 and m["aoa_deg"] == 5.932 and m["series"] == 5 and m["p"][3] == 18.252
    m4 = A.parse_name("airFoil2D_SST_31.677_-0.742_2.926_5.236_12.538")
    assert m4["series"] == 4 and m4["p"] == [2.926, 5.236, 12.538, 0.0]
    with pytest.raises(ValueError):
        A.parse_name("cavity_1_2")


def _fake_rows(n=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        U, a, m, p, t = (
            rng.uniform(30, 90),
            rng.uniform(-4, 10),
            rng.uniform(0, 6),
            rng.uniform(2, 6),
            rng.uniform(6, 18),
        )
        cl = 0.11 * a + 0.1 * m + rng.normal(0, 0.01)  # format fixture only (synthetic relation + noise floor)
        rows.append(
            {
                "name": f"airFoil2D_SST_{U:.3f}_{a:.3f}_{m:.3f}_{p:.3f}_{t:.3f}",
                "U": U,
                "aoa_deg": a,
                "series": 4,
                "p1": m,
                "p2": p,
                "p3": t,
                "p4": 0.0,
                "cd": 0.01 + 1e-4 * a**2,
                "cdp": 0.0,
                "cdv": 0.0,
                "cl": cl,
                "clp": cl,
                "clv": 0.0,
            }
        )
    return rows


def test_table_roundtrip_surrogate_and_splits(tmp_path):
    rows = _fake_rows()
    A.write_table(rows, tmp_path / "t.csv")
    assert (tmp_path / "t.csv").read_text().startswith("# Derived from AirfRANS")
    back = A.read_table(tmp_path / "t.csv")
    assert len(back) == len(rows) and back[0]["name"] == rows[0]["name"]
    res = A.train_coefficient_surrogate(A.features(rows), A.targets(rows), max_epochs=3000, patience=200)
    assert res.converged and res.epochs_run < 3000
    assert res.history[-1]["val_mse"] < res.history[0]["val_mse"]
    names = [r["name"] for r in rows]
    (tmp_path / "manifest.json").write_text(json.dumps({"full_train": names[:100], "full_test": names[100:]}))
    out = A.evaluate_on_splits(
        tmp_path / "t.csv", tmp_path / "manifest.json", "full_train", "full_test", max_epochs=1500, patience=200
    )
    assert out["n_train"] == 100 and out["n_test"] == 20 and out["mae_cl"] < 0.1
    pv = A.panel_vs_rans(rows[:3])
    assert pv["n"] == 3 and np.isfinite(pv["rmse_cl"])
