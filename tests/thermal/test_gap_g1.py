"""Gap run G1 (experiments/p6b_g1_stats.py): references per design, pre-registered verdict, Holm family."""

import numpy as np

from aerosynapse.thermal.experiments import p6b_g1_stats as G


def test_default_config_is_the_preregistered_design():
    cfg = G.load_cfg([], [])
    assert cfg["seeds"] == [0, 1, 2, 3, 4] and cfg["references"] == {"leave_one_out": "full", "add_one_in": "base"}
    assert cfg["metrics"]["primary"] == {"split": "interp", "field": "rel_l2_max"}
    assert cfg["analysis"]["ratio_min_effect"] == 2.0


def test_compare_verdicts_and_holm():
    cfg = G.load_cfg([], [])
    rng = np.random.default_rng(1)
    base = {s: 0.8 * (1 + 0.05 * rng.standard_normal()) for s in range(5)}
    full = {s: 0.14 * (1 + 0.05 * rng.standard_normal()) for s in range(5)}
    tab = {
        "full": full,
        "base": base,
        "full-x": {s: full[s] * 3.0 * (1 + 0.02 * rng.standard_normal()) for s in range(5)},  # 3x worse: established
        "full-y": {s: full[s] * 1.3 * (1 + 0.02 * rng.standard_normal()) for s in range(5)},  # significant but < 2x
        "base+x": {s: base[s] * (1 + 0.3 * rng.standard_normal()) for s in range(5)},  # noise
    }
    design = {
        "full": "leave_one_out",
        "full-x": "leave_one_out",
        "full-y": "leave_one_out",
        "base": "add_one_in",
        "base+x": "add_one_in",
    }
    res = {e["config"]: e for e in G.compare(tab, design, cfg)}
    assert set(res) == {"full-x", "full-y", "base+x"}  # references are not compared with themselves
    assert res["full-x"]["reference"] == "full" and res["base+x"]["reference"] == "base"
    assert res["full-x"]["verdict"] == "ESTABLISHED"
    assert res["full-y"]["p_holm"] < 0.05 and res["full-y"]["verdict"] == "NOT ESTABLISHED"
    assert res["base+x"]["verdict"] == "NOT ESTABLISHED"
    assert all(e["p_holm"] >= e["p_t"] for e in res.values())


def test_tables_reproduce_r4_7a_full():
    tab, design = G.table(["results/p4_7_ablation_smoke"], "interp", "rel_l2_max")
    assert design["full"] == "leave_one_out" and design["base"] == "add_one_in"
    assert abs(np.mean([tab["full"][s] for s in (0, 1, 2)]) - 0.139) < 5e-4  # R4.7a: 0.139 +- 0.064
