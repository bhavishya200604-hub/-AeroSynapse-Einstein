"""Step 6.6: Graetz eigen-series limits, wall Nusselt numbers, the disk cache, the Berkovsky-Polevikov correlation,
the analysis and the figures (fake data)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aerosynapse.thermal.experiments import p6_6_validation as V
from aerosynapse.thermal.physics import graetz as G
from aerosynapse.thermal.utils.cache import cached, key_hash
from aerosynapse.thermal.viz import validation_figs as F


def test_entry_series_limits():
    s = G.entry_series(np.array([1e-4, 1e-3, 0.5]))
    assert s["mu"][0] == pytest.approx(G.NU_FD_CWT, abs=1e-3)  # fully developed limit
    assert s["Nu_local"][-1] == pytest.approx(G.NU_FD_CWT, abs=1e-3)
    assert s["Nu_local"][0] == pytest.approx(float(G.leveque_local(1e-4)), rel=0.01)  # short-entry asymptote
    assert s["Nu_mean"][0] == pytest.approx(float(G.leveque_mean(1e-4)), rel=0.01)
    assert np.all(np.diff(s["Nu_local"]) < 0) and np.all(s["Nu_mean"] > s["Nu_local"])
    # bulk temperature decays monotonically, mean Nu = energy balance
    assert np.all(np.diff(s["theta_b"]) < 0)
    assert s["Nu_mean"] == pytest.approx(np.log(1 / s["theta_b"]) / (4 * s["xstar"]))


def test_axial_conduction_reference():
    assert G.nu_cwt_spectral(64, Pe=1e4)["Nu"] == pytest.approx(G.NU_FD_CWT, abs=2e-3)
    assert G.nu_cwt_spectral(64, Pe=10.0)["Nu"] > G.NU_FD_CWT  # axial conduction raises the fully developed Nu


def test_wall_nusselt_on_the_series_profile():
    """A fully developed profile theta = 1 - f(y) e^{-lambda x} (f = first eigenfunction) gives Nu = mu_1."""
    N = 64
    D1, D2, y, u = G._operators(N)
    from scipy import linalg as sla

    I = np.arange(1, N)
    vals, vecs = sla.eig(-D2[np.ix_(I, I)], np.diag(u[I]))
    k = np.argmin(np.where(np.isreal(vals) & (np.real(vals) > 0), np.real(vals), np.inf))
    f = np.zeros(N + 1)
    f[I] = np.real(vecs[:, k])
    f /= f[N // 2]
    df = D1 @ f
    Pe = 50.0
    x = np.linspace(0.5, 3.0, 6)
    dec = np.exp(-np.real(vals[k]) * x / Pe)  # theta~ ~ exp(-4 mu x*) = exp(-mu x / Pe)
    th = 1.0 - np.outer(f, dec)
    U = np.repeat(u[:, None], x.size, 1)
    tw = {"bottom": np.ones_like(x), "top": np.ones_like(x)}
    dn = {"bottom": -df[0] * dec, "top": df[-1] * dec}  # inward normal: +y at y = 0, -y at y = 1
    # Clenshaw-Curtis weighted bulk: wall_nusselt uses plain sums, so feed the weights through u
    w = G.clencurt(N, 0.0, 1.0)
    r = V.wall_nusselt(x, y, U * w[:, None], th, tw, dn, Pe)
    assert r["Nu_local"] == pytest.approx(np.full(x.size, np.real(vals[k])), rel=1e-6)


def test_cache_roundtrip(tmp_path):
    calls = []

    def fn():
        calls.append(1)
        return {"a": np.arange(3)}

    k = {"case": "x", "n": 3}
    r1, h1 = cached(tmp_path, "g", k, fn)
    r2, h2 = cached(tmp_path, "g", dict(reversed(list(k.items()))), fn)  # key order does not matter
    assert (h1, h2) == (False, True) and len(calls) == 1 and np.array_equal(r1["a"], r2["a"])
    assert key_hash(k) != key_hash(k | {"n": 4})
    assert not list((tmp_path / "g").glob("*.tmp"))


def test_berkovsky_polevikov():
    assert V.bp_nu(1e4, 0.71) == pytest.approx(0.18 * (0.71 / 0.91 * 1e4) ** 0.29)
    assert V.bp_nu(1e5, 7.0) > V.bp_nu(1e5, 0.71)


def _fake_met():
    xs = np.geomspace(1e-4, 0.2, 40)
    ser = G.entry_series(xs)
    g = []
    for Re, Pr in ((10, 1.0), (100, 5.0)):
        Pe = Re * Pr
        x = xs * 4 * Pe
        keep = x <= 8.0
        g.append(
            {"Re": Re, "Pr": Pr, "Pe": Pe, "xstar": xs[keep].tolist()}
            | {k: (ser[k][keep] * 1.01).tolist() for k in ("Nu_local", "Nu_mean", "theta_b")}
        )
    nat = [
        {"Pr": 0.71, "N": n, "records": [{"Ra": Ra, "Nu_hot": V.bp_nu(Ra, 0.71) * (1 + 1 / n**2)} for Ra in (1e4, 1e5)]}
        for n in (16, 24, 32)
    ]
    met = {"graetz": g, "graetz_check": [], "natural": nat}
    vs = {"graetz": {"Lx": 8.0, "exclude_outlet": 1.0, "series": {"n_modes": 40, "N": 120}}}
    met["analysis"] = V.analyse(met, vs)
    s = G.entry_series(np.geomspace(1e-5, 1, 50))
    met["series"] = {k: s[k].tolist() for k in ("xstar", "Nu_local", "Nu_mean")} | {"mu1": float(s["mu"][0])}
    return met


def test_analyse_and_figures():
    met = _fake_met()
    an = met["analysis"]
    for e in an["graetz"]:
        assert e["x_station"] <= 7.0 + 1e-9
        assert e["dev_local_median"] == pytest.approx(0.01, rel=0.05)
    e = an["natural"]["Pr0.71_Ra10000"]
    assert e["richardson"] == pytest.approx(V.bp_nu(1e4, 0.71), rel=1e-3)  # second-order data -> exact extrapolation
    assert "dVD" in e and abs(e["dev_BP"]) < 1e-3
    xx = np.linspace(0.1, 8, 20)
    pinn = [
        {
            "point": {"Re": 25.0, "Pr": 1.0, "Bi": 3.0},
            "xstar": (xx / 100).tolist(),
            "solver": {"Nu_local": (8 + 1 / xx).tolist()},
            "pinn": {"Nu_local": (8 + 0.9 / xx).tolist()},
            "members": [(8 + 0.8 / xx).tolist(), (8 + 1.0 / xx).tolist()],
        }
    ]
    n = 9
    xc = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xc, xc)
    fields = {
        f"nat_0.71_{Ra:g}_{k}": v
        for Ra in (1e4, 1e5)
        for k, v in (("xc", xc), ("yc", xc), ("theta", 1 - X), ("psi", np.sin(np.pi * X) * np.sin(np.pi * Y)))
    }
    gx, gy = np.linspace(0, 8, 17), np.linspace(0, 1, 5)
    fields |= {"graetz_10_1_x": gx, "graetz_10_1_y": gy, "graetz_10_1_theta": np.outer(np.ones(5), 1 - np.exp(-gx))}
    met["plot_Ra"] = [1e4, 1e5]
    met["plot_pair"] = [10, 1.0]
    F.style()
    for fig in (
        F.graetz_nu(met),
        F.graetz_surface3d(met),
        F.graetz_field3d(met, fields),
        F.pinn_nu(pinn),
        F.natural_nu(met),
        F.natural_fields(met, fields),
        F.natural_field3d(met, fields),
    ):
        plt.close(fig)
