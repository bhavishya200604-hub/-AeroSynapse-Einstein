"""Step 1.7: GCI/Richardson, Graetz limits, steady Newton NS solver, body force, benchmark data."""

import numpy as np
import pytest
import sympy as sp

from aerosynapse.thermal.analysis import benchmarks as bm
from aerosynapse.thermal.numerics import fvm2d as fv
from aerosynapse.thermal.numerics import ns_projection as ns
from aerosynapse.thermal.numerics.convergence import gci, richardson
from aerosynapse.thermal.numerics.ns_steady import SteadyNewtonNS
from aerosynapse.thermal.physics import graetz as gz


def test_gci_recovers_order_and_limit():
    # f(h) = 3 + 0.7 h^2 + 0.2 h^3 on h = 0.1, 0.2, 0.4 (constant r) and a non-uniform ratio set
    f = lambda h: 3 + 0.7 * h**2 + 0.2 * h**3  # noqa: E731
    for h1, h2, h3 in [(0.01, 0.02, 0.04), (0.01, 0.013, 0.021)]:
        r = gci(f(h1), f(h2), f(h3), h1, h2, h3)
        assert r["type"] == "monotone"
        assert abs(r["p"] - 2) < 0.1
        assert abs(r["f_ext"] - 3) < abs(f(h1) - 3) / 10
        assert r["GCI_fine21"] > abs(f(h1) - 3) / 3  # conservative band
    assert richardson(f(0.01), f(0.02), 2.0, 2.0) == pytest.approx(3, abs=1e-5)


def test_graetz_chf_exact_and_cwt_spectral():
    assert gz.nu_chf_symbolic() == sp.Rational(140, 17)
    a, b = gz.nu_cwt_spectral(32), gz.nu_cwt_spectral(64)
    assert abs(a["Nu"] - b["Nu"]) < 1e-9 and abs(b["Nu"] - 7.5407) < 1e-4
    c = gz.nu_cwt_spectral(48, Pe=10.0)
    assert c["Nu"] > b["Nu"]  # axial conduction raises Nu
    assert c["Nu"] == pytest.approx(c["Nu_energy_balance"], rel=1e-9)


def _cavity(n, Re=100.0, energy=False, Ri=0.0):
    en = None
    if energy:
        en = ns.EnergySpec(
            Pe=71.0,
            bcs={
                "top": fv.dirichlet2(1.0),
                "bottom": fv.dirichlet2(0.0),
                "left": fv.neumann2(0.0),
                "right": fv.neumann2(0.0),
            },
        )
    return ns.MACNavierStokes2D(fv.Grid2D(n, n), Re, {"top": ns.wall(u=1.0)}, energy=en, Ri=Ri)


def test_newton_jacobian_exact_and_matches_time_marching():
    s = _cavity(12, energy=True, Ri=0.3)
    N = SteadyNewtonNS(s)
    rng = np.random.default_rng(0)
    x = 0.3 * rng.standard_normal(N.n)
    J, dropped = N.jacobian(x)
    w = rng.standard_normal(N.n)
    fd = (N.residual(x + 1e-6 * w) - N.residual(x - 1e-6 * w)) / 2e-6
    assert np.linalg.norm(J @ w - fd) < 1e-8 * np.linalg.norm(fd) and dropped == 0.0
    info = N.solve(N.pack(s.u, s.v, None, np.zeros((12, 12))))
    assert info["converged"] and info["iterations"] <= 8
    N.store()
    s2 = _cavity(12, energy=True, Ri=0.3)
    s2.set_state(np.zeros_like(s2.u), np.zeros_like(s2.v), theta=np.zeros((12, 12)))
    s2.run(300.0, steady_tol=1e-10)
    assert np.abs(s.u - s2.u).max() < 1e-8 and np.abs(s.theta - s2.theta).max() < 1e-7
    assert s.max_div() < 1e-12


def test_body_force_interior_wall_shear_second_order():
    errs = []
    for n in (8, 16, 32):
        k, y0, Re = 2, 2 / n, 20.0
        solid = np.zeros((n + k, 2 * n), bool)
        solid[:k, :] = True
        prof = lambda s_, t, y0=y0: np.where(s_ > y0, 4 * (s_ - y0) * (1 + y0 - s_), 0.0)  # noqa: E731
        s = ns.MACNavierStokes2D(
            fv.Grid2D(2 * n, n + k, 2.0, 1 + y0),
            Re,
            {"left": ns.wall(u=prof), "right": ns.outflow()},
            solid=solid,
        )
        N = SteadyNewtonNS(s)
        N.solve(N.pack(s.u, s.v, None, None))
        p = N.store()
        P = np.where(s.solid, 0.0, np.nan_to_num(p))
        Nu, _ = s.momentum_rhs(s.u, s.v, None, 0.0)
        Gx = np.zeros_like(s.u)
        Gx[:, 1:-1] = (P[:, 1:] - P[:, :-1]) / s.g.dx
        cols = np.arange(s.g.nx + 1)
        mid = (cols >= s.g.nx // 4) & (cols < 3 * s.g.nx // 4)
        errs.append(abs(np.sum(((Nu - Gx) * s.solid_u)[:, mid]) * s.g.vol - 4 / Re))
    assert np.log2(errs[1] / errs[2]) > 1.9


def test_body_force_symmetric_obstacle_no_lift():
    g = fv.Grid2D(48, 16, 3.0, 1.0)
    solid = (np.abs(g.Xc - 1.0) < 0.2) & (np.abs(g.Yc - 0.5) < 0.2)
    s = ns.MACNavierStokes2D(
        g, 10.0, {"left": ns.wall(u=lambda s_, t: 4 * s_ * (1 - s_)), "right": ns.outflow()}, solid=solid
    )
    N = SteadyNewtonNS(s)
    N.solve(N.pack(s.u, s.v, None, None))
    Fx, Fy = s.body_force(N.store())
    assert Fx > 0 and abs(Fy) < 1e-10 * Fx


def test_convective_outflow_channel_steady_and_divergence_free():
    """Convective outflow (added after the 2D-2 blow-up) still yields the Poiseuille profile to O(h^2)."""
    errs = []
    for n in (8, 16):
        s = ns.MACNavierStokes2D(
            fv.Grid2D(2 * n, n, 2.0, 1.0),
            20.0,
            {"left": ns.wall(u=lambda s_, t: 4 * s_ * (1 - s_)), "right": ns.outflow(True)},
        )
        s.set_state(np.zeros_like(s.u), np.zeros_like(s.v))
        info = s.run(60.0, steady_tol=1e-8)
        assert info["last_change_rate"] < 1e-8 and info["max_div"] < 1e-12
        errs.append(np.abs(s.u - 4 * s.Yu * (1 - s.Yu)).max())
    assert errs[1] < errs[0] / 2.5


def test_benchmark_tables_consistent():
    for Re in (100, 400, 1000, 3200):
        for which in ("u", "v"):
            c, v, m = bm.ghia_points(Re, which)
            assert c.size == v.size == m.size == 17
    assert not bm.ghia_points(3200, "u")[2][9] and not bm.ghia_points(400, "v")[2][5]
    assert bm.SCHAFER_TUREK_2D1["C_D"] == pytest.approx(5.57953523384)
