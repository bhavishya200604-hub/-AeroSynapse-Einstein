"""Phase 4 gate: benchmark quantities on analytic fields (Nu, Graetz Nu, centrelines, Ghia deviation), literature
values, the compile step, validation, and a tiny end-to-end run of every benchmark kind."""

import json

import numpy as np
import pytest
import torch
from torch import nn

from aerosynapse.thermal.experiments import p4_gate as G
from aerosynapse.thermal.analysis import benchmarks as bm
from aerosynapse.thermal.pinn import channel2d as CH
from aerosynapse.thermal.pinn import problems2d as P2
from aerosynapse.thermal.pinn import spec as S

DT = torch.float64
SPEC = S.load_spec(None, "smoke")
PRIM = S.deep_merge(SPEC["formulation"], {"kind": "primitive", "hard_bc": False})


class Field(nn.Module):
    """Primitive outputs (u, v, p, theta) from a python function of (x, y)."""

    def __init__(self, fn):
        super().__init__()
        self.fn, self.a = fn, nn.Parameter(torch.tensor(1.0, dtype=DT))

    def forward(self, X):
        return torch.stack(self.fn(X[:, 0], X[:, 1]), -1) * self.a


def cell_grid(nx, ny, lx=1.0, ly=1.0):
    x, y = (np.arange(nx) + 0.5) * lx / nx, (np.arange(ny) + 0.5) * ly / ny
    return x, y, *np.meshgrid(x, y)


def test_nu_hot_linear_profile_is_one():
    x, y, X, Y = cell_grid(32, 32)
    assert abs(G.ref_nu_hot({"x": x, "theta": 1.0 - X}) - 1.0) < 1e-12  # 2nd-order one-sided gradient exact for linear
    _, _, pcfg = G.cavity_setup(SPEC["gate"]["benchmarks"]["dvd_ra1e4"], SPEC["gate"]["reference"])
    ref = {"x": x, "y": y, "nu_top": 0.0, **{k: np.zeros_like(X) for k in P2.FIELDS}}
    prob = P2.build_cavity(pcfg, PRIM, ref)
    z = lambda x, y: 0 * x  # noqa: E731
    net = Field(lambda x, y: (z(x, y), z(x, y), z(x, y), 1.0 - x))
    assert abs(G.pinn_nu_hot(prob, net) - 1.0) < 1e-12


def test_graetz_nu_on_an_analytic_field():
    """u = 6y(1-y), theta = 1 - exp(-x) sin(pi y): Nu_Dh = 2 pi / (24 / pi^3) = pi^4 / 12 at every x."""
    exact = np.pi**4 / 12
    x, y, X, Y = cell_grid(64, 128, 8.0, 1.0)
    ref = {"x": x, "y": y, "u": 6 * Y * (1 - Y), "theta": 1 - np.exp(-X) * np.sin(np.pi * Y)}
    xs, nu = G.ref_nu_graetz(ref, 1.0, 3.0, 6.0)
    assert len(xs) and xs.min() >= 3 and xs.max() <= 6 and np.abs(nu / exact - 1).max() < 2e-3
    vals, pcfg = G.graetz_setup(SPEC, SPEC["gate"]["benchmarks"]["graetz_cwt"])
    prob = CH.build_channel("D_graetz", pcfg, PRIM, [], vals, 1.0)
    net = Field(lambda x, y: (6 * y * (1 - y), 0 * x, 0 * x, 1 - torch.exp(-x) * torch.sin(np.pi * y)))
    nu_p = G.pinn_nu_graetz(prob, net, np.array([3.0, 4.5, 6.0]), 1.0)
    assert np.abs(nu_p / exact - 1).max() < 1e-3


def test_centrelines_and_ghia_deviation():
    c, val, mask = bm.ghia_points(100, "u")
    o = np.argsort(c)  # Ghia's tables list the coordinates in DESCENDING order
    s, prof = c[o], val[o]  # sampled exactly at the table points
    assert G.ghia_deviation(s, prof, 100, "u")["max"] < 1e-12
    d = G.ghia_deviation(s, prof + 0.01, 100, "u")
    assert abs(d["rms"] - 0.01) < 1e-12 and d["n"] == int((mask & (c > 0) & (c < 1)).sum())
    x, y, X, Y = cell_grid(16, 16)
    cl = G.ref_centrelines({"x": x, "u": Y * 1.0, "v": X * 2.0})
    assert np.allclose(cl["u"], y) and np.allclose(cl["v"], 2 * x)


def test_literature_values():
    b = SPEC["gate"]["benchmarks"]
    assert G.literature(b["dvd_ra1e4"])["Nu"] == 2.243 and G.literature(b["dvd_ra1e4"])["Nu_hortmann"] == 2.245
    lg = G.literature(b["graetz_cwt"])
    assert abs(lg["Nu_no_axial"] - 7.54) < 0.01 and lg["Nu_axial"] > lg["Nu_no_axial"] and lg["Pe"] == 20.0


def test_compile_rows(tmp_path, monkeypatch):
    d = tmp_path / "results" / "runA"
    d.mkdir(parents=True)
    recs = [
        {
            "variant": v,
            "split": sp,
            "seed": 0,
            "rel_l2_max": e,
            "ref_unc_max": 0.01,
            "case": "D_x",
            "q_bottom_rel_err": e / 10,
        }
        for v, sp, e in (("a", "interp", 0.2), ("a", "interp", 0.4), ("a", "extrap", 9.0), ("b", "interp", 0.005))
    ]
    (d / "metrics.json").write_text(json.dumps({"records": recs, "git": "abcdef123"}), encoding="utf-8")
    monkeypatch.setattr(G, "ROOT", tmp_path)
    rows = G.compile_rows(
        [
            {"label": "L", "run": "runA", "group": "variant", "where": {"case": "D_x"}},
            {"label": "M", "run": "nope", "group": "variant"},
        ]
    )
    a = next(r for r in rows if r.get("group") == "a")
    assert a["n"] == 2 and abs(a["err_mean"] - 0.3) < 1e-12 and a["err_max"] == 0.4 and a["git"] == "abcdef1"
    assert a["q_bottom_rel_err"] == (0.02, 0.04) and rows[-1]["missing"]


@pytest.mark.parametrize(
    "over,msg",
    [
        ({"gate": {"benchmarks": {"x": {"kind": "karman"}}}}, "one of"),
        ({"gate": {"benchmarks": {"x": {"kind": "dvd", "Pr": 0.71}}}}, "gate.benchmarks.x.Ra"),
        ({"gate": {"benchmarks": {"x": {"kind": "dvd", "Ra": 1.0e3, "Pr": 0.71}}}}, "verified de Vahl Davis"),
        ({"gate": {"benchmarks": {"x": {"kind": "ghia_cavity", "Re": 150, "Pr": 1.0}}}}, "Ghia et al. table"),
        ({"gate": {"benchmarks": {"graetz_cwt": {"window": [6.0, 3.0]}}}}, "inside the channel"),
        ({"gate": {"seeds": []}}, "gate.seeds"),
    ],
)
def test_spec_validation(over, msg):
    with pytest.raises(ValueError, match=msg):
        S.load_spec(None, "smoke", over)


@pytest.mark.parametrize("name", ["ghia_re100", "graetz_cwt", "dvd_ra1e4"])
def test_tiny_benchmarks_end_to_end(tmp_path, name):
    spec = S.load_spec(
        None,
        "smoke",
        {
            "training": {"adam": {"epochs": 4}, "lbfgs": {"iterations": 1, "inner": 1}},
            "sampling": {"n_pde": 32, "n_bc": 4, "adaptive": {"period": 2, "n_candidates": 48}},
            "loss": {"balancing": {"ntk": {"every": 2, "n_points": 4}}},
            "gate": {"reference": {"grid": 8, "refine": 2}},
            "problems2d": {"D_graetz": {"reference": {"cells_per_unit": 4}}},
        },
    )
    ref = G.bench_reference(spec, name)
    r = G.run_benchmark(spec, name, 0, ref, tmp_path / name, "t")
    assert r["status"] == "ok" and np.isfinite(r["rel_l2_max"])
    if name == "ghia_re100":
        assert set(ref["quantities"]) == {"ghia_u", "ghia_v"} and np.isfinite(r["ghia_u"]["rms"])
    elif name == "dvd_ra1e4":
        assert 1.0 < ref["quantities"]["Nu_hot"] < 4.0 and np.isfinite(r["Nu_hot"])
    else:
        assert len(r["Nu_x"]) == len(ref["quantities"]["Nu_x"]) > 0
