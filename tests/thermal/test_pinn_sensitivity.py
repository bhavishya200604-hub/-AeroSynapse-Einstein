"""Step 6.4: Sobol estimators and PCE verified on the Ishigami function (analytic indices)."""

import numpy as np
import pytest

from aerosynapse.thermal.pinn import sensitivity as SA

A_, B_ = 7.0, 0.1


def ishigami(u):
    x = -np.pi + 2 * np.pi * np.asarray(u)
    return np.sin(x[:, 0]) + A_ * np.sin(x[:, 1]) ** 2 + B_ * x[:, 2] ** 4 * np.sin(x[:, 0])


def ishigami_exact():
    V1 = 0.5 * (1 + B_ * np.pi**4 / 5) ** 2
    V2 = A_**2 / 8
    V13 = B_**2 * np.pi**8 * (1 / 18 - 1 / 50)
    V = V1 + V2 + V13
    return [V1 / V, V2 / V, 0.0], [(V1 + V13) / V, V2 / V, V13 / V]


def test_to_phys_log_and_linear():
    u = np.array([[0.0, 0.5], [1.0, 1.0]])
    x = SA.to_phys(u, [(1.0, 100.0, "log"), (2.0, 4.0, "linear")])
    assert x == pytest.approx(np.array([[1.0, 3.0], [100.0, 4.0]]))


def test_saltelli_jansen_on_ishigami():
    n, d = 2**14, 3
    A, B, AB = SA.saltelli(n, d, seed=1)
    f = ishigami(SA.stack(A, B, AB))
    ix = SA.indices(f, n, d)
    S, ST = ishigami_exact()
    assert ix["S"] == pytest.approx(S, abs=0.02)
    assert ix["ST"] == pytest.approx(ST, abs=0.02)
    ci = SA.bootstrap(f, n, d, n_boot=200)
    for k, (lo, hi) in enumerate(ci["ST_ci95"]):
        assert lo - 0.01 <= ST[k] <= hi + 0.01


def test_pce_on_ishigami_and_polynomial():
    u = np.random.default_rng(0).random((3000, 3))
    m = SA.pce_fit(u, ishigami(u), 10)
    S, ST = ishigami_exact()
    assert m["S"] == pytest.approx(S, abs=0.01) and m["ST"] == pytest.approx(ST, abs=0.01)
    assert m["loo_rel_error"] < 0.05
    f = 1 + 2 * (2 * u[:, 0] - 1) + 0.5 * (2 * u[:, 1] - 1) ** 2  # exact in degree 2
    m2 = SA.pce_fit(u, f, 2)
    assert m2["loo_rel_error"] < 1e-10 and m2["ST"][2] == pytest.approx(0.0, abs=1e-12)
    assert SA.pce_eval(m2, u[:5]) == pytest.approx(f[:5])


@pytest.mark.parametrize(
    "bad",
    [
        {"sensitivity_study": {"cases": {"nope": {}}}},
        {"sensitivity_study": {"cases": {"D_graetz": {"solver": {"n": 100}}}}},
        {"sensitivity_study": {"cases": {"D_graetz": {"ranges": {"Re": [10, 1]}}}}},
        {"sensitivity_study": {"convergence": [3]}},
    ],
)
def test_validation(bad):
    from aerosynapse.thermal.pinn import spec as SP

    with pytest.raises(ValueError, match="sensitivity_study"):
        SP.load_spec(None, "smoke", bad)


def test_report_and_figures(tmp_path):
    import matplotlib.pyplot as plt

    from aerosynapse.thermal.experiments import p6_4_sensitivity as P64
    from aerosynapse.thermal.viz import sensitivity_figs as F

    n, d = 64, 3
    A, B, AB = SA.saltelli(n, d, 0)
    U = SA.stack(A, B, AB)
    f = 1 + U[:, 0] + 2 * U[:, 2] ** 2
    ix = SA.indices(f, n, d) | SA.bootstrap(f, n, d, 50)
    box = {k: [1.0, 10.0] for k in P64.NAMES}
    case = {
        "qoi": "mean Nu",
        "box": box,
        "train_box": box,
        "outside_train_box": {k: [False, False] for k in P64.NAMES},
        "members": ["seed 0", "seed 1"],
        "reload_check": {"n": 3, "max_rel_diff": 1e-16},
        "grid_check": {"max_rel_diff": 0.01},
        "solver": ix | {"n": n, "n_runs": 5 * n, "wall_s": 1.0, "settings": {"n": n}},
        "convergence": [
            {"N": m}
            | SA.indices(f.reshape(5, n)[:, :m].ravel(), m, d)
            | SA.bootstrap(f.reshape(5, n)[:, :m].ravel(), m, d, 20)
            for m in (16, 32, 64)
        ],
        "pce": {k: v for k, v in SA.pce_fit(U, f, 2).items() if k not in ("coef", "alphas")},
        "pinn": {
            "ensemble": ix,
            "ensemble_same_design": ix,
            "seed 0": ix,
            "seed 1": ix,
            "n": n,
            "n_runs": 5 * n,
            "wall_s": 1.0,
        },
        "paired": {
            "rel_err_median": 0.01,
            "rel_err_p95": 0.02,
            "rel_err_max": 0.03,
            "bias_mean_rel": 0.0,
            "rel_err_median_seed 0": 0.01,
            "sample": {"solver": f.tolist(), "pinn": (1.01 * f).tolist()},
        },
    }
    met = {"cases": {"D_graetz": case}}
    P64.report(met, tmp_path / "s.md", "test")
    assert "| ST | Bi |" in (tmp_path / "s.md").read_text(encoding="utf-8")
    F.style()
    for fn in (F.indices, F.convergence, F.paired):
        plt.close(fn(met["cases"]))
