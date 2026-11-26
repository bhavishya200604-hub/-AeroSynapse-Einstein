"""AirfRANS graph data (Kaggle download): safe reading, the models, a tiny end-to-end run."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aerosynapse.ml import airfrans_graph as G  # noqa: E402

FILES = G.graph_files()
need_data = pytest.mark.skipif(not FILES, reason="the AirfRANS graph folders are not in the project")


@need_data
def test_safe_read_one_simulation():
    g = G.read_graph(FILES[0][1])
    assert g["x"].shape[1] == 5 and g["y"].shape[1] == 4 and g["pos"].shape[1] == 2
    assert len(g["x"]) == len(g["y"]) == len(g["pos"]) > 10000
    assert np.ptp(g["x"][:, 0]) == 0 and np.ptp(g["x"][:, 1]) == 0  # the inlet: the same at every node
    assert (g["x"][:, 2] <= g["x"][:, 2].min() + 1e-7).sum() > 100  # wall nodes


def test_models_shapes():
    x, pos = torch.rand(3, 50, 5), torch.rand(3, 50, 2)
    for m in (G.PointMLP(), G.PointNetFlow()):
        assert m(x, pos).shape == (3, 50, 4)


@need_data
def test_tiny_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setitem(G.PROFILES, "smoke", (10, 1))
    res = G.run(tmp_path, "smoke")
    assert res["simulations"] == 10 and res["split"] == {"train": 8, "val": 1, "test": 1}
    for k in ("mlp", "pointnet"):
        assert set(res["results"][k]["test_full_mesh"]) == set(G.FIELDS)
    assert (tmp_path / "03_fields_test_airfoil.png").exists()
    assert json.loads((tmp_path / "result.json").read_text())["figures"]
