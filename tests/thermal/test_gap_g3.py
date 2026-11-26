"""Gap run G3 (experiments/p6b_g3_budget.py): pre-registered verdict logic and report."""

from aerosynapse.thermal.experiments import p6b_g3_budget as G


def _d(smoke, full, seeds=(0, 1, 2)):
    def rec(v):
        return {"error": v, "train_wall_s": 1.0, "status": "ok"}

    return {
        "smoke": {s: rec(v) for s, v in zip(seeds, smoke, strict=True)},
        "full": {s: rec(v) for s, v in zip(seeds, full, strict=True)},
        "seeds": list(seeds),
    }


def test_default_config_is_the_preregistered_design():
    cfg = G.load_cfg([], [])
    assert cfg["graetz"]["configuration"] == "full-exact_psi" and cfg["graetz"]["seeds"] == [0, 1, 2]
    assert cfg["slab"]["profile"] == "full" and cfg["slab"]["problem"] == "B_slab" and cfg["slab"]["value"] == 1.0
    assert cfg["analysis"]["ratio_max"] == 0.5 and cfg["analysis"]["alpha"] == 0.05


def test_budget_limited_needs_significance_and_a_factor_two():
    cfg = G.load_cfg([], [])
    data = {
        "graetz": _d([0.1, 0.08, 0.06], [0.01, 0.0082, 0.0059]),  # ~10x lower, consistent
        "slab": _d([0.002, 0.0026, 0.002], [0.0015, 0.0019, 0.0016]),
    }  # lower but only ~0.75x
    an = G.analyse(data, cfg)
    assert an["graetz"]["verdict"] == "BUDGET-LIMITED"
    assert an["slab"]["verdict"].startswith("NOT BUDGET-LIMITED")


def test_pending_seeds_are_marked_interim_or_untested(tmp_path):
    cfg = G.load_cfg([], [])
    d = _d([0.1, 0.08, 0.06], [0.01, 0.008, 0.006])
    del d["full"][2]
    one = _d([0.1, 0.08, 0.06], [0.01, 0.008, 0.006])
    one["full"] = {0: one["full"][0]}
    an = G.analyse({"graetz": d, "slab": one}, cfg)
    assert "INTERIM" in an["graetz"]["verdict"] and an["graetz"]["pending"] == [2]
    assert an["slab"]["test"] is None and an["slab"]["verdict"].startswith("NO TEST")
    G.report({"data": {"graetz": d, "slab": one}, "analysis": an}, tmp_path / "g3.md", "t")
    txt = (tmp_path / "g3.md").read_text(encoding="utf-8")
    assert "pending" in txt and "NO TEST" in txt
