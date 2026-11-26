"""Step 6.7: break-even arithmetic, ladder summary, analysis on fake measurements, timing helper, figures."""

import json

import matplotlib.pyplot as plt
import pytest

from aerosynapse.thermal.experiments import p6_7_cost as C
from aerosynapse.thermal.viz import cost_figs as F


def test_break_even_and_helpers():
    assert C.break_even(100.0, 1.0, 0.0) == pytest.approx(100.0)
    assert C.break_even(100.0, 1.0, 0.5) == pytest.approx(200.0)
    assert C.break_even(100.0, 0.1, 0.2) is None  # the solver is cheaper per query: never
    x, y = C.solver_grid_xy("D_graetz", 4)
    assert len(x) == 32 and len(y) == 4 and x[0] == pytest.approx(0.125) and y[-1] == pytest.approx(0.875)
    x, y = C.solver_grid_xy("C_cavity", 8)
    assert len(x) == len(y) == 8
    P = C.box_points({"Re": [10, 100], "Pr": [1, 2], "Bi": [0.1, 1]}, 50, 0)
    assert P.shape == (50, 3) and (P[:, 0] >= 10).all() and (P[:, 0] <= 100).all()
    calls = []
    t = C._timeit(lambda: calls.append(1), 2, 3)
    assert len(calls) == 5 and len(t["repeats"]) == 3 and t["min_s"] <= t["median_s"]


def _ladder():
    pt = {"Re": 1.0, "Pr": 1.0, "Bi": 1.0}
    rows = [
        {"grid": g, "cells": g * g, "point": pt, "qoi": 1.0 + 1.0 / g**2, "wall_s": 0.01 * g, "mem_bytes": 1e3 * g}
        for g in (8, 16, 32)
    ]
    conv = [{"point": pt, "gci": {"f_ext": 1.0}}]
    return {"grid": {"D_graetz": {"rows": rows, "convergence": conv}}}


def test_ladder_and_analysis(tmp_path, monkeypatch):
    L = C.ladder_2d(_ladder(), "D_graetz")
    assert [r["grid"] for r in L] == [8, 16, 32] and L[0]["rel_err"] == pytest.approx(1 / 64)
    (tmp_path / "lad").mkdir()
    (tmp_path / "lad" / "metrics.json").write_text(json.dumps(_ladder()), encoding="utf-8")
    monkeypatch.setattr(C, "ROOT", tmp_path)
    lat = {
        "threads": 1,
        "batches": [
            {"batch": 1, "per_query_member_s": [0.01, 0.01]},
            {"batch": 16, "per_query_member_s": [0.001, 0.001]},
        ],
        "field": {"n_points": 10, "member_s": [0.002, 0.002]},
    }
    met = {
        "cases": {
            "D_graetz": {
                "dim": 2,
                "train_logged_s": [200.0, 220.0],
                "pinn_error": 0.02,
                "latency": [lat],
                "retrain": [{"train_s": 100.0, "data_s": 50.0, "member_index": 0}],
            }
        }
    }
    cs = {"cases": {"D_graetz": {"dim": 2, "solver_grid": 16}}, "ladder": "lad", "sobol_run": "none"}
    a = C.analyse(met, cs)["D_graetz"]
    assert a["contention"] == pytest.approx(2.0)  # member 0: logged 200 / alone 100
    assert a["train_clean_est_s"] == pytest.approx([100.0, 110.0])  # member 1 scaled by the factor
    assert a["offline_ensemble_s"] == pytest.approx(50 + 210)
    assert a["pinn_query_ensemble_s"] == pytest.approx(0.002)
    assert a["solver_s"] == pytest.approx(0.16)
    assert a["break_even_ensemble"] == pytest.approx(260 / (0.16 - 0.002))
    assert a["equal_accuracy"]["grid"] == 8  # error 1/64 < 0.02: the coarsest grid is already as accurate
    F.style()
    an = {"D_graetz": a}
    for fn in (F.amortisation, F.breakdown, F.latency, F.tradeoff, F.decision3d, F.breakeven3d):
        plt.close(fn(an))
