"""Step 5.1 (optimised): optimal experiment design and the Phase 5 figure functions (smoke tests)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aerosynapse.thermal.inverse import design as DS
from aerosynapse.thermal.inverse import spec as IS
from aerosynapse.thermal.viz import inverse_figs as VF

SMALL = {"measurement": {"sensors": {"n": 3}, "times": {"n": 12}}}


def test_time_spacing_default_is_the_old_linear_grid():
    s = IS.load_spec()
    assert np.allclose(IS.measurement_times(s), np.linspace(5.0, 1200.0, 60))
    e = IS.load_spec(overrides={"measurement": {"times": {"spacing": 2.0}}})
    u = np.linspace(0, 1, 60) ** 2
    assert np.allclose(IS.measurement_times(e), 5.0 + 1195.0 * u)
    with pytest.raises(ValueError, match="spacing"):
        IS.load_spec(overrides={"measurement": {"times": {"spacing": 0.0}}})


def test_explicit_positions_keep_replicates():
    s = IS.load_spec(overrides={"measurement": {"sensors": {"positions": [1.0, 0.0, 0.0]}}})
    assert np.allclose(IS.sensor_positions(s), [0.0, 0.0, 0.02])


def test_encode_decode_round_trip_and_base_design():
    d = DS.Designer(IS.load_spec(overrides=SMALL))
    b = d.base_design()
    assert np.allclose(b["positions"], [0.25, 0.5, 0.75]) and b["t1"] == 1200.0 and b["spacing"] == 1.0
    back = d.decode(d.encode(b))
    assert np.allclose(back["positions"], b["positions"], atol=1e-9) and abs(back["t1"] - 1200.0) < 1e-6
    r = d.evaluate(b)  # identical to the 5.1 analysis of the base design
    s = IS.load_spec(overrides=SMALL)
    from aerosynapse.thermal.experiments.p5_1_identifiability import analyse_spec

    ref = analyse_spec(s, [0.02])["levels"]["0.02"]["crlb_percent"]
    assert all(abs(r["crlb_percent"][k] - ref[k]) < 1e-9 * ref[k] for k in ref)


def test_optimisation_improves_the_criterion_and_respects_bounds():
    s = IS.load_spec(overrides={**SMALL, "design_optimisation": {"starts": 1}})
    d = DS.Designer(s)
    out = d.optimise(1, 0)
    lo, hi = s["design_optimisation"]["bounds"]["t1"]
    assert lo <= out["optimal_design"]["t1"] <= hi and all(0 <= v <= 1 for v in out["optimal_design"]["positions"])
    worst = lambda r: max(r["crlb_percent"].values())  # noqa: E731
    assert worst(out["optimal"]) < worst(out["base"])


@pytest.mark.parametrize(
    "over,msg",
    [
        ({"design_optimisation": {"criterion": "e_optimal"}}, "design_optimisation.criterion"),
        ({"design_optimisation": {"variables": ["positions", "budget"]}}, "design_optimisation.variables"),
        ({"design_optimisation": {"bounds": {"t1": [500.0, 100.0]}}}, "bounds.t1"),
        ({"design_optimisation": {"starts": 0}}, "starts"),
    ],
)
def test_validation(over, msg):
    with pytest.raises(ValueError, match=msg):
        IS.load_spec(overrides=over)


def test_figure_functions_return_figures():
    VF.style()
    P = ["alpha", "k", "h"]
    rng = np.random.default_rng(0)
    pan = [
        {
            "title": "a",
            "t": np.linspace(1, 10, 5),
            "J": rng.standard_normal((5, 2, 3)),
            "x_frac": [0.1, 0.9],
            "sigma_K": 0.5,
        }
    ]
    figs = [VF.sensitivities(pan, P)]
    sw = {
        "h": {
            "x": [10.0, 100.0],
            "xlabel": "h",
            "log_x": True,
            "title": "h",
            "base": 100.0,
            "crlb": {k: [5.0, 1.0] for k in P},
        },
        "q": {
            "x": [0.0, 1e3, 1e5],
            "xlabel": "q",
            "log_x": "symlog",
            "title": "q",
            "base": 1e5,
            "crlb": {"alpha": [5.0, 5.0, 5.0], "k": [np.inf, 100.0, 5.0], "h": [np.inf, 90.0, 2.0]},
        },
        "lay": {
            "x": ["uniform", "random"],
            "xlabel": "",
            "log_x": False,
            "title": "layout",
            "base": "uniform",
            "crlb": {k: [3.0, 4.0] for k in P},
        },
    }
    figs.append(VF.crlb_sweeps(sw, P, [10.0, 50.0], 0.02))
    cov = np.diag([0.003, 0.003, 0.0003])
    figs.append(
        VF.confidence_ellipses(
            [{"name": "x", "color": VF.EMPH, "cov": cov, "samples": rng.standard_normal((20, 3)) * 0.05}], P
        )
    )
    figs.append(
        VF.design_comparison(
            {"positions": [0.2, 0.5], "times": [1, 2], "crlb": dict.fromkeys(P, 5.0)},
            {"positions": [0.0, 1.0], "times": [1, 3], "crlb": dict.fromkeys(P, 2.0)},
            P,
            0.02,
        )
    )
    hist = [[{"alpha": 1.0, "k": 2.0, "h": 3.0}, {"alpha": 1.1, "k": 1.9, "h": 2.5}]]
    figs.append(VF.convergence({"joint": hist}, P, {"alpha": 1.0, "k": 2.0, "h": 3.0}, dict.fromkeys(P, 5.0)))
    figs.append(VF.error_dots([{"name": "a", "errors": {k: [1.0, -2.0] for k in P}}], P, dict.fromkeys(P, 5.0)))
    assert all(isinstance(f, plt.Figure) for f in figs)
    for f in figs:
        plt.close(f)
