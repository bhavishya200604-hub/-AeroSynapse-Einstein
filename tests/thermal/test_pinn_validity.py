"""Step 6.3: distance, residual indicators, detection scores, cavity stability, settings, report and figures."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from aerosynapse.thermal.experiments import p6_3_extrapolation as P63
from aerosynapse.thermal.numerics.ns_stability import crossing
from aerosynapse.thermal.pinn import problems as P
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn import validity as V
from aerosynapse.thermal.viz import validity_figs as F


def test_distance_decades():
    d = V.distance([0.03, 0.3, 1.0, 3.0, 300.0], 0.3, 3.0)
    assert d == pytest.approx([1.0, 0.0, 0.0, 0.0, 2.0])


def test_auc_and_detection():
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert V.auc(s, [0, 0, 0, 1, 1, 1]) == 1.0
    assert V.auc(s, [1, 1, 1, 0, 0, 0]) == 0.0
    assert V.auc(s, [0, 1, 0, 1, 0, 1]) == pytest.approx(2 / 3)
    assert V.auc(s, [0, 0, 0, 0, 0, 0]) is None
    d = V.detection(s, [0.001, 0.001, 0.02, 0.05, 0.2, 0.3], tol=0.01, threshold=0.35)
    assert d["n_bad"] == 4 and d["tpr"] == pytest.approx(3 / 4) and d["fpr"] == 0.0 and d["missed_bad"] == 1


class _ExactSteady(torch.nn.Module):
    """theta = (exp(Pe c x) - 1) / (exp(Pe c) - 1) with inputs (x, logp): the exact A_steady solution (0 -> 1)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))  # dtype / device carrier

    def forward(self, X):
        x, pe = X[:, 0], torch.exp(X[:, 1])
        return ((torch.exp(pe * x) - 1) / (torch.exp(pe) - 1)).unsqueeze(-1)


class _Const(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.c = c

    def forward(self, X):
        return self.c + 0 * X[:, :1] ** 3 + self.w  # cubic: keeps an autograd graph for the 2nd derivative


def test_residuals_of_exact_and_constant_fields():
    spec = SP.load_spec(
        None, "smoke", {"problems": {"A_steady": {"parametric": True, "parameter": {"values": [1, 20]}}}}
    )
    prob = P.build_problem("A_steady", spec["problems"]["A_steady"])
    x = np.linspace(0, 1, 51)
    r = V.residual_vectors(_ExactSteady().double(), prob, 5.0, x, None)
    ind = V.physics_indicator(r)
    assert ind["total"] < 1e-10
    spec = SP.load_spec(
        None, "smoke", {"problems": {"B_slab": {"parametric": True, "parameter": {"values": [0.3, 3]}}}}
    )
    prob = P.build_problem("B_slab", spec["problems"]["B_slab"])
    c = spec["problems"]["B_slab"]["conditions"]
    r = V.residual_vectors(_Const(c["theta0"]), prob, 2.0, x, np.linspace(0, prob.T, 5))
    assert np.allclose(r["pde"], c["m"] ** 2 * c["theta0"] - c["Q"]) and np.allclose(r["ic"], 0.0)
    assert np.allclose(r["bc_left"], 0.0) and np.allclose(r["bc_right"], 2.0 * c["theta0"])
    m = V.mean_residuals([r, {k: 3 * v for k, v in r.items()}])
    assert np.allclose(m["bc_right"], 4.0 * c["theta0"])


def test_cavity_stability_stokes_limit_and_crossing():
    from aerosynapse.thermal.numerics import fvm2d as fv
    from aerosynapse.thermal.numerics import ns_projection as ns
    from aerosynapse.thermal.numerics.ns_stability import rightmost
    from aerosynapse.thermal.numerics.ns_steady import SteadyNewtonNS

    s = ns.MACNavierStokes2D(fv.Grid2D(16, 16), 100.0, {"top": ns.wall(u=1.0)})
    N = SteadyNewtonNS(s)
    info = N.solve(N.pack(s.u, s.v, None, None), tol=1e-9, max_iter=25)
    assert info["converged"]
    r = rightmost(N, N.x, [0.0, 2.0], k=4)
    assert not r["unstable"] and -60 < r["growth_rate"] * 100 < -45  # slowest Stokes decay of the unit square ~ -52/Re
    assert crossing([1, 2, 3], [-2.0, -1.0, 1.0]) == pytest.approx(2.5)
    assert crossing([1, 2], [-1.0, -0.5]) is None


@pytest.mark.parametrize(
    "bad",
    [
        {"extrapolation_study": {"problems": {"nope": {}}}},
        {"extrapolation_study": {"problems": {"B_slab": {"decades_below": -1}}}},
        {"extrapolation_study": {"time": {"factor": 0.5}}},
        {"extrapolation_study": {"threshold_quantile": 1.5}},
        {"extrapolation_study": {"cavity": {"Re": [5000, 1000]}}},
        {"extrapolation_study": {"cavity": {"N": [4]}}},
    ],
)
def test_validation(bad):
    with pytest.raises(ValueError, match="extrapolation_study"):
        SP.load_spec(None, "smoke", bad)


def test_user_configs_load():
    for f in ("configs/pinn/user_phase6.yaml", "configs/pinn/examples/user_phase6_example.yaml"):
        s = SP.load_spec(f, "smoke")
        assert {"scaling_study", "uq_study", "extrapolation_study"} <= set(s)
    s = SP.load_spec("configs/pinn/examples/user_phase6_example.yaml", "smoke")
    assert (
        s["problems"]["B_slab"]["conditions"]["Q"] == 0.5
        and s["problems"]["A_steady"]["conditions"]["theta_left"] == 1.0
    )
    assert s["scaling_study"]["solvers"]["fd_solver"] == "splu"  # partial blocks keep the defaults


def _fake():
    rng = np.random.default_rng(0)
    vals = np.geomspace(0.03, 30, 13)
    d = V.distance(vals, 0.3, 3.0)
    err = 0.003 * 10 ** (1.5 * d) * np.exp(rng.normal(0, 0.1, 13))
    ev = {
        "value": vals.tolist(),
        "rel_l2": err.tolist(),
        "member_rel_l2_min": (0.7 * err).tolist(),
        "member_rel_l2_max": (1.5 * err).tolist(),
        "rms_sd": (0.5 * err).tolist(),
        "coverage_95": np.clip(1 - d / 3, 0, 1).tolist(),
        "residual": (2 * err).tolist(),
        "sd_mcd_0.01_s0": (0.05 + 0 * err).tolist(),
    }
    cv = rng.uniform(0.002, 0.004, 20)
    cal = {"rms_sd": (0.5 * cv).tolist(), "residual": (2 * cv).tolist(), "sd_mcd_0.01_s0": (0.05 + 0 * cv).tolist()}
    es = SP.load_spec(None, "smoke")["extrapolation_study"]
    rows = [
        {
            "value": v,
            "t": tt,
            "t_over_T": tt / 0.5,
            "rel_l2": 0.003 * (1 + max(0, tt / 0.5 - 1)) ** 4,
            "rms_sd": 0.001 * (1 + tt),
            "residual": 0.01 * (1 + tt),
            "coverage_95": 1.0,
        }
        for v in (0.3, 3.0)
        for tt in np.linspace(0.05, 1.5, 10)
    ]
    tm = {
        "T": 0.5,
        "factor": 3.0,
        "rows": rows,
        "scores": {
            k: {"threshold": 0.002 if k == "rms_sd" else 0.02, "spearman": 0.9, "detection": []}
            for k in ("rms_sd", "residual")
        },
    }
    par = {
        "B_slab": {
            "range": [0.3, 3.0],
            "param_name": "Bi",
            "values": ev,
            "distance": d.tolist(),
            "calibration": cal,
            "scores": P63.score(es, cal, ev, d),
            "time": tm,
        }
    }

    def recs(n):
        return [
            {
                "Re": r,
                "N": n,
                "converged": True,
                "psi_min": -0.1 + r * 1e-6,
                "u_min": -0.3,
                "v_max": 0.3,
                "v_min": -0.4,
                "max_div": 1e-14,
                "stability": {"growth_rate": (r - 8000) * 1e-6, "frequency": 2.8, "unstable": r > 8000},
            }
            for r in (1000.0, 5000.0, 10000.0)
        ]

    cav = {
        "grids": {str(n): {"N": n, "records": recs(n)} for n in (16, 32)},
        "per_Re": {
            "1000": {
                "Re": 1000.0,
                "label": "laminar",
                "per_grid": {
                    "16": {"psi_min": -0.1, "growth_rate": -0.007, "frequency": 0.0},
                    "32": {"psi_min": -0.1, "growth_rate": -0.007, "frequency": 0.0},
                },
            }
        },
        "critical_Re": {"16": 8000.0, "32": None},
        "label_from_Re": 5000.0,
    }
    return par, cav


def test_scores_report_figures(tmp_path):
    par, cav = _fake()
    sc = par["B_slab"]["scores"]
    assert {"rms_sd", "residual", "combined", "distance", "sd_mcd_0.01_s0"} <= set(sc)
    assert sc["residual"]["spearman"] > 0.9 and sc["sd_mcd_0.01_s0"]["detection"][0]["auc"] in (0.5, None)
    P63.report({"parametric": par, "cavity": cav}, tmp_path / "e.md", "test")
    txt = (tmp_path / "e.md").read_text(encoding="utf-8")
    assert "Bi trained on [0.3, 3]" in txt and "stable at Re 5000, unstable at Re 10000" in txt
    F.style()
    for fig in (F.extrapolation(par), F.indicators(par), F.time_extrapolation(par), F.cavity(cav)):
        plt.close(fig)
