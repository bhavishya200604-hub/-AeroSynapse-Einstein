"""Step 6.1: power-law fits, equal-error selection, Pareto front, settings validation, report and figures."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aerosynapse.thermal.experiments import p6_1_scaling as P61
from aerosynapse.thermal.pinn import scaling as S
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.viz import scaling_figs as F


def test_loglog_recovers_exact_power_law():
    x = np.array([8, 16, 32, 64, 128.0])
    f = S.loglog(x, 3.0 * x**-1.5)
    assert f["slope"] == pytest.approx(-1.5, abs=1e-12)
    assert f["C"] == pytest.approx(3.0, rel=1e-10)
    assert f["r2"] == pytest.approx(1.0)


def test_saturating_recovers_floor_and_is_preferred():
    x = np.geomspace(8, 2048, 7)
    e = 2.0 * x**-2.0 + 1e-3
    s = S.saturating(x, e)
    assert s["q"] == pytest.approx(2.0, rel=1e-3)
    assert s["floor"] == pytest.approx(1e-3, rel=1e-3)
    f = S.fit(x, [np.array([v, v, v]) for v in e], n_boot=50)
    assert f["best"] == "saturating" and f["exponent"] == pytest.approx(-2.0, rel=1e-3)


def test_fit_bootstrap_interval_contains_true_exponent():
    rng = np.random.default_rng(1)
    x = np.array([8, 16, 32, 64, 128, 256.0])
    vals = [0.5 * v**-1.0 * np.exp(rng.normal(0, 0.05, 5)) for v in x]
    f = S.fit(x, vals, n_boot=300, seed=0)
    assert f["exponent"] == pytest.approx(-1.0, abs=0.1)
    lo, hi = f["exponent_ci95"]
    assert lo <= -1.0 <= hi
    assert f["n_boot"] == 300


def test_fit_three_levels_always_power_law():
    x = np.array([8, 32, 128.0])
    f = S.fit(x, [np.array([v]) for v in 1e-2 * x**-0.5 + 1e-4], n_boot=20)
    assert f["best"] == "loglog"


def test_cheapest_and_pareto():
    e, t = np.array([0.1, 0.01, 0.005, 0.2]), np.array([1.0, 5.0, 3.0, 0.5])
    c = S.cheapest(e, t, 0.01)
    assert c == {"index": 2, "error": 0.005, "wall_s": 3.0}
    assert S.cheapest(e, t, 1e-4) is None
    assert S.pareto(e, t) == [3, 0, 2]


@pytest.mark.parametrize(
    "bad",
    [
        {"problems": {"nope": {"value": 1.0}}},
        {"sweeps": {"lr": [1, 2]}},
        {"sweeps": {"width": [0, 8]}},
        {"sweeps": {"width": []}},
        {"seeds": []},
        {"targets": [2.0]},
        {"solvers": {"fd_N": [2]}},
    ],
)
def test_validation_rejects_bad_settings(bad):
    with pytest.raises(ValueError, match="scaling_study"):
        SP.load_spec(None, "smoke", {"scaling_study": bad})


def test_defaults_valid_and_overrides_apply():
    spec = SP.load_spec(None, "smoke", P61.OVERRIDE["width"](8))
    assert spec["network"]["width"] == 8
    b = P61.base_levels(SP.load_spec(None, "smoke"))
    st = SP.load_spec(None, "smoke")["scaling_study"]
    assert set(st["sweeps"]) <= set(P61.OVERRIDE)
    assert all(b[k] in v for k, v in st["sweeps"].items())  # every sweep passes through the base setting


def _fake_met():
    x = np.array([8, 32, 128.0])
    fits = {
        p: {k: S.fit(x, [np.array([0.1 * v**-0.5, 0.12 * v**-0.5]) for v in x], n_boot=20) for k in ("width", "n_pde")}
        for p in ("B_slab", "A_steady")
    }
    base = {"width": 32, "depth": 3, "n_pde": 32, "adam_epochs": 1500, "lbfgs_iterations": 200}
    runs = [
        {"problem": p, "factor": "base" if lv == 32 else k, "level": "base" if lv == 32 else lv, "seed": sd}
        | {"rel_l2": (0.1 + 0.02 * sd) * lv**-0.5, "wall_s": 10.0 + lv / 10}
        for p in fits
        for k in ("width", "n_pde")
        for lv in (8, 32, 128)
        for sd in (0, 1)
    ]
    curve = [{"N": n, "rel_l2": 1.0 / n**2, "wall_s": 1e-4 * n} for n in (10, 20, 40)]
    solvers = {p: {"fd": curve, "spectral": curve} for p in fits}
    equal = {
        p: [
            {
                "target": 1e-2,
                "pinn": None,
                "fd": {"index": 1, "error": 2.5e-3, "wall_s": 2e-3, "N": 20},
                "spectral": None,
            }
        ]
        for p in fits
    }
    return {
        "runs": runs,
        "fits": fits,
        "solvers": solvers,
        "equal_error": equal,
        "base": base,
        "problems": {"B_slab": {"value": 1.0}, "A_steady": {"value": 10.0}},
        "seeds": [0, 1],
    }


def test_report_and_figures(tmp_path):
    met = _fake_met()
    P61.report(met, tmp_path / "t.md", "test")
    txt = (tmp_path / "t.md").read_text(encoding="utf-8")
    assert "## B_slab (Bi = 1)" in txt and "## A_steady (Pe = 10)" in txt
    assert "not reached" in txt and "N 20" in txt
    F.style()
    fig = F.scaling_grid(met["runs"], met["fits"], met["base"], P61.LABEL)
    assert len(fig.axes) == 4
    assert fig.axes[0].get_title(loc="left").startswith("slope")
    plt.close(fig)
    fig = F.cost_vs_error(met["runs"], met["solvers"], [1e-2])
    assert len(fig.axes) == 2
    plt.close(fig)
    paths = P61.figures(met, "test", tmp_path, docs=False)
    assert len(paths) == 4 and all(str(tmp_path) in p for p in paths)


def test_step_to_floor_is_unresolved_and_labelled():
    x = np.array([32, 64, 128, 256, 512.0])
    e = np.array([6e-3, 2.2e-3, 2.3e-3, 2.2e-3, 2.3e-3])
    f = S.fit(x, [np.array([v, v * 1.01]) for v in e], n_boot=30)
    assert f["best"] == "saturating" and not f["resolved"]
    assert P61._exponent(f) == "unresolved (step to the floor)"
    g = S.fit(np.array([8, 16, 32, 64.0]), [np.array([v]) for v in [1e-2, 5e-3, 2.5e-3, 1.25e-3]], n_boot=20)
    assert g["resolved"] and P61._exponent(g).startswith("-1.00")


def test_by_setting_medians():
    runs = [
        {"factor": "width", "level": 8, "rel_l2": v, "wall_s": w} for v, w in ((1e-3, 10.0), (3e-3, 20.0), (2e-3, 30.0))
    ] + [{"factor": "base", "level": "base", "rel_l2": 5e-4, "wall_s": 40.0}]
    s = {(r["factor"], r["level"]): r for r in S.by_setting(runs)}
    assert s[("width", 8)]["rel_l2"] == pytest.approx(2e-3) and s[("width", 8)]["rel_l2_max"] == pytest.approx(3e-3)
    assert s[("width", 8)]["wall_s"] == pytest.approx(20.0) and s[("width", 8)]["n"] == 3
    assert s[("base", "base")]["n"] == 1


def test_fd_solver_setting_validated():
    with pytest.raises(ValueError, match="fd_solver"):
        SP.load_spec(None, "smoke", {"scaling_study": {"solvers": {"fd_solver": "magic"}}})
