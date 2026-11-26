"""The fin PINN study: data extraction from every optimiser, PINN training (a few steps) and the figures."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from aerosynapse.thermal import fin_pinn as F  # noqa: E402


def test_fd_energy_balance_closes():
    # the finite-difference fin conserves energy: sink + lateral + tip = source
    x, th = F.fd_field(2.0, 0.5, 1600)
    b = F.energy_balance(x, th, (th[1] - th[0]) / (x[1] - x[0]), 2.0, 0.5)
    assert abs(b["imbalance"]) < 5e-3
    assert b["sink"] > 0 and b["lateral"] > 0 and b["tip"] > 0


def test_smoke_run_writes_the_analysis(tmp_path):
    res = F.run(tmp_path, "smoke")
    assert res["n_training_designs"] > 100
    assert set(res["settings"]) == {s[0] for s in F.SETTINGS}
    for name in (
        "FinPINN_ALL_01_extracted-data_2D.png",
        "FinPINN_ALL_09_theta-surface_3D.png",
        "FinPINN_ALL_11_fins_3D.png",
        "summary.txt",
        "evaluations.csv",
        "fin_data.npz",
        "settings.json",
        "training_history.json",
    ):
        assert (tmp_path / name).exists(), name
    for key in F.SHORT.values():  # every optimisation setting has its own 2D and 3D analysis, in PNG and vector form
        for dim in ("2D", "3D"):
            assert (tmp_path / f"FinPINN_{key}_analysis_{dim}.png").exists(), (key, dim)
            assert (tmp_path / "vector" / f"FinPINN_{key}_analysis_{dim}.pdf").exists(), (key, dim)
    man = json.loads((tmp_path / "settings.json").read_text())
    assert [m["key"] for m in man["settings"]] == list(F.SHORT.values())
    d = np.load(tmp_path / "fin_data.npz")
    assert d["fields"].shape[1] == F.NX and d["fields"].shape[0] == res["n_training_designs"]
    assert json.loads((tmp_path / "result.json").read_text())["figures"]
