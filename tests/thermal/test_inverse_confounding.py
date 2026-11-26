"""Step 5.6: profile likelihood (src/inverse/profile.py), priors in the posterior, the case builder and figures."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aerosynapse.thermal.experiments import p5_6_confounding as P56
from aerosynapse.thermal.inverse import classical as C
from aerosynapse.thermal.inverse import data as D
from aerosynapse.thermal.inverse import profile as PR
from aerosynapse.thermal.inverse import spec as IS
from aerosynapse.thermal.inverse import uq as U
from aerosynapse.thermal.viz import inverse_figs as VF

SMALL = {"measurement": {"sensors": {"n": 3}, "times": {"n": 12}}}
UNK = ["alpha", "k", "h"]


def _fit(over=None):
    s = IS.load_spec(overrides=IS.deep_merge(SMALL, over or {}))
    d = D.load(s)
    f = C.fit(s, d, list(s["unknown"]), {k: float(s["initial_guess"][k]) for k in s["unknown"]}, "analytic")
    return s, d, f


def test_offsets_two_scale():
    o = PR.offsets(0.05, 100.0, 4, 3)
    assert len(o) == 7 and o[3] == pytest.approx(0.2) and o[-1] == pytest.approx(np.log(100.0))
    assert np.all(np.diff(o) > 0)
    assert PR.offsets(np.inf, 100.0, 4, 3)[3] == pytest.approx(0.4)


def test_profile_matches_gauss_newton_when_identifiable():
    s, d, f = _fit()
    for k in UNK:
        pr = PR.profile(s, d, UNK, f, k, 100.0, 12, 6)
        lo, hi = pr["ci95_log"]
        assert lo is not None and hi is not None
        x = np.log(f["estimates"][k])
        assert (hi - lo) / 2 == pytest.approx(1.96 * f["sd_log"][k], rel=0.25)
        assert lo < x < hi and float(np.nanmin(pr["delta_chi2"])) == pytest.approx(0.0, abs=1e-6)


def test_profile_is_flat_without_a_source():
    s, d, f = _fit({"body": {"q_vol": 0.0}})
    for k in ("k", "h"):
        pr = PR.profile(s, d, UNK, f, k, 100.0, 6, 4)
        assert pr["ci95_log"] == [None, None] and pr["max_delta_chi2"] < 1e-3
    pa = PR.profile(s, d, UNK, f, "alpha", 100.0, 6, 4)
    assert all(v is not None for v in pa["ci95_log"])


def test_profile_single_unknown_and_interval_helper():
    s, d, f = _fit({"unknown": ["h"]})
    pr = PR.profile(s, d, ["h"], f, "h", 10.0, 8, 4)
    assert all(v is not None for v in pr["ci95_log"])
    g = np.linspace(-1, 1, 5)
    assert PR._interval(g, np.array([10.0, 2.0, 0.0, 2.0, np.nan]), 2) == [
        pytest.approx(-0.5 - 0.5 * (3.841 - 2) / 8, abs=1e-3),
        None,
    ]


def test_prior_makes_the_confounded_posterior_proper():
    """Full default design (near-linear Bi posterior): h = Bi L / k follows k's prior along the Biot ridge."""
    s = IS.load_spec(overrides={"body": {"q_vol": 0.0}})
    d = D.load(s)
    f = C.fit(s, d, UNK, {k: float(s["initial_guess"][k]) for k in UNK}, "analytic")
    mean_k = float(np.log(15.0))
    post = U.posterior(s, d, UNK, f, 6000, 1500, 250, 100.0, 3, "analytic", {"k": (mean_k, 0.1)})
    S = post["samples_log"]
    sk, sh = S[:, 1].std(), S[:, 2].std()
    assert sk == pytest.approx(0.1, rel=0.3)  # k follows its prior (the data do not inform it alone)
    assert sh == pytest.approx(np.sqrt(sk**2 + C.log_sd(f, UNK, {"h": 1.0, "k": -1.0}) ** 2), rel=0.3)
    assert abs(np.corrcoef(S[:, 1], S[:, 2])[0, 1]) > 0.8  # h follows k along the Biot ridge


def test_case_builder():
    s = IS.load_spec()
    cfg = s["studies"]["confounding"]
    cases = P56.build_cases(s, cfg)
    groups = [c["group"] for c in cases]
    assert groups.count("q_vol") == 5 and groups.count("h") == 6 and groups.count("priors") == 3
    assert groups.count("known") == len(cfg["known"]) * len(cfg["known_at_q_vol"])
    h = next(c for c in cases if c["name"] == "h=1000")
    assert h["spec"]["initial_guess"]["h"] == pytest.approx(500.0)  # the guess keeps its ratio (50 / 100) to the truth
    pr = next(c for c in cases if c["group"] == "priors" and c["label"] == "10 %")
    assert pr["priors"]["k"][0] == pytest.approx(np.log(15.0) + 0.1)
    u = P56.build_cases(IS.load_spec("configs/inverse/user_case.yaml"), cfg)
    assert u[0]["spec"]["data"]["source"] == "file" and all(c["spec"]["data"]["source"] == "synthetic" for c in u[1:])


def test_validation():
    with pytest.raises(ValueError, match="studies.confounding.profile"):
        IS.load_spec(overrides={"studies": {"confounding": {"profile": {"span": 1.0}}}})
    with pytest.raises(ValueError, match="studies.confounding.known"):
        IS.load_spec(overrides={"studies": {"confounding": {"known": {"x": ["rho"]}}}})


def test_figures():
    x = np.linspace(-4, 4, 21)
    cases = [
        {"label": lab, "unknown": UNK, "profiles": {k: {"x": x.tolist(), "d": (x**2 * s).tolist()} for k in UNK}}
        for lab, s in (("configured", 50.0), ("q_vol = 0", 0.0), ("q_vol = 1000", 0.5), ("h = 3", 0.1))
    ]
    sw = {
        "q_vol": {
            "x": [0.0, 1e-3, 0.04],
            "xlabel": "Q",
            "crlb": {k: [np.inf, 50.0, 5.0] for k in UNK},
            "profile": {k: [np.inf, np.inf, 11.0] for k in UNK},
            "bi": [5.0, 5.0, 4.0],
        }
    }
    row = {
        "unknown": ["alpha", "h"],
        "label": "k known",
        "parameters": {"alpha": 4e-6, "k": 15.0, "h": 100.0},
        "profiles": {k: {"ci95_log": [np.log(v) - 0.1, None]} for k, v in (("alpha", 4e-6), ("h", 100.0))},
    }
    pri = {
        "unknown": UNK,
        "parameters": {"alpha": 4e-6, "k": 15.0, "h": 100.0},
        "posterior": {
            "priors": {"k": [np.log(15.0), 0.1]},
            "median_log": {k: np.log(v) for k, v in (("alpha", 4e-6), ("k", 15.0), ("h", 100.0))},
            "ci95_log": {
                k: [np.log(v) - 0.2, np.log(v) + 0.2] for k, v in (("alpha", 4e-6), ("k", 15.0), ("h", 100.0))
            },
        },
    }
    for f in (VF.profiles(cases), VF.identifiability_map(sw), VF.prior_knowledge([row], [pri])):
        assert isinstance(f, plt.Figure)
        plt.close(f)


def test_phase5_gate_aggregates_saved_results():
    """The gate only reads saved metrics: every checklist item is present and the classical baseline passes."""
    from aerosynapse.thermal.experiments import p5_gate as G
    from aerosynapse.thermal.utils.experiment import ROOT

    if not all((ROOT / "results" / r / "metrics.json").exists() for r in G.RUNS.values()):
        pytest.skip("saved Phase 5 results not present")
    g = G.gather()
    items = {it["item"]: it["status"] for it in G.checklist(g)}
    assert items["Classical baseline within its 95 % band and the CRLB"] == "PASS"
    assert len(items) == 7 and all(q in g["crlb_log"] for q in ("alpha", "k", "h", "Bi"))
    f1, f2 = VF.gate_convergence(g["histories"], g["truth"], g["crlb_log"]), VF.gate_errors(
        [{"name": "x", "group": "classical", "err": {"alpha": 1.0}, "ci": {"alpha": [-1.0, 2.0]}, "seeds": {}}],
        {q: 5.0 for q in ("alpha", "k", "h", "Bi")},
    )
    plt.close(f1)
    plt.close(f2)
