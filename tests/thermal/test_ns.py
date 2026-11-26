"""Step 1.6: MAC Navier–Stokes projection solver (fast checks)."""

import numpy as np
import pytest
import sympy as sp

from aerosynapse.thermal.numerics import fvm2d as fv
from aerosynapse.thermal.numerics import ns_projection as ns
from aerosynapse.thermal.numerics.convergence import fit_order
from aerosynapse.thermal.numerics.multigrid import laplacian_fv
from aerosynapse.thermal.physics import mms


def tg_bcs(tg):
    return {
        "left": ns.wall(v=lambda s, t: tg.v_fn(0 * s, s, t)),
        "right": ns.wall(v=lambda s, t: tg.v_fn(1 + 0 * s, s, t)),
        "bottom": ns.wall(u=lambda s, t: tg.u_fn(s, 0 * s, t)),
        "top": ns.wall(u=lambda s, t: tg.u_fn(s, 1 + 0 * s, t)),
    }


def tg_solver(n, Re=100.0, rk="ssprk3", poisson="splu"):
    tg = mms.taylor_green(Re)
    s = ns.MACNavierStokes2D(fv.Grid2D(n, n), Re, tg_bcs(tg), rk=rk, poisson=poisson)
    s.set_state(tg.u_fn(s.Xu, s.Yu, 0 * s.Xu), tg.v_fn(s.Xv, s.Yv, 0 * s.Xv), project=False)
    return s, tg


def test_taylor_green_is_exact_unforced_solution():
    tg = mms.taylor_green(37.0)
    assert sp.simplify(tg.fx) == 0 and sp.simplify(tg.fy) == 0
    assert mms.divergence((tg.u, tg.v)) == 0


def test_mms_ns_forcing_consistency():
    ms = mms.mms_ns_default()
    assert mms.divergence((ms.u, ms.v)) == 0
    # psi vanishes on the boundary => zero normal velocity on all walls
    for xb in (0, 1):
        assert sp.simplify(ms.u.subs(mms.x, xb)) == 0
        assert sp.simplify(ms.v.subs(mms.y, xb)) == 0


@pytest.mark.parametrize("right", ["wall", "outflow"])
def test_DG_equals_fv_laplacian(right):
    g = fv.Grid2D(10, 7, 1.0, 0.7)
    bcs = {"right": ns.outflow()} if right == "outflow" else {}
    s = ns.MACNavierStokes2D(g, 10.0, bcs)
    fvb = {k: fv.neumann2(0.0) for k in fv.SIDES}
    if right == "outflow":
        fvb["right"] = fv.dirichlet2(0.0)
    assert abs(s.L - laplacian_fv(g, fvb)).max() < 1e-12
    assert s.singular == (right == "wall")


@pytest.mark.parametrize("with_solid", [False, True])
def test_projection_divergence_free_idempotent(with_solid):
    g = fv.Grid2D(20, 16)
    solid = None
    if with_solid:
        solid = np.zeros((16, 20), bool)
        solid[6:10, 5:9] = True
    s = ns.MACNavierStokes2D(g, 10.0, {"left": ns.wall(u=1.0), "right": ns.outflow()}, solid=solid)
    rng = np.random.default_rng(0)
    u, v, _ = s.project(rng.standard_normal(s.u.shape), rng.standard_normal(s.v.shape), 0.0)
    assert np.max(np.abs(s.divergence(u, v))) < 1e-12
    u2, v2, _ = s.project(u.copy(), v.copy(), 0.0)
    assert max(np.abs(u2 - u).max(), np.abs(v2 - v).max()) < 1e-12
    if with_solid:
        assert np.all(u[s.solid_u & ~s._mirror_u] == 0) and np.all(v[s.solid_v & ~s._mirror_v] == 0)


@pytest.mark.parametrize("method", ["mg", "mgcg"])
def test_multigrid_pressure_matches_splu(method):
    a, _ = tg_solver(32)
    b, _ = tg_solver(32, poisson=method)
    a.run(0.05, dt=5e-3)
    b.run(0.05, dt=5e-3)
    assert np.abs(a.u - b.u).max() < 1e-10 and b.max_div() < 1e-10


def test_taylor_green_second_order_velocity_and_pressure():
    eu, ep, Ns = [], [], [16, 32, 64]
    for n in Ns:
        s, tg = tg_solver(n)
        s.run(0.25)
        eu.append(np.abs(s.u - tg.u_fn(s.Xu, s.Yu, s.t + 0 * s.Xu)).max())
        p, pe = s.pressure(), tg.p_fn(s.g.Xc, s.g.Yc, s.t + 0 * s.g.Xc)
        ep.append(np.abs((p - p.mean()) - (pe - pe.mean())).max())
        assert s.max_div() < 1e-11
    h = 1 / np.array(Ns)
    assert fit_order(h[-2:], eu[-2:])[0] > 1.8
    assert fit_order(h[-2:], ep[-2:])[0] > 1.9


def test_temporal_orders_taylor_green():
    ref, _ = tg_solver(16)
    ref.run(0.4, dt=1e-3)
    for rk, order in [("euler", 1), ("ssprk2", 2), ("ssprk3", 3)]:
        errs, dts = [], [0.02, 0.01, 0.005]
        for dt in dts:
            s, _ = tg_solver(16, rk=rk)
            s.run(0.4, dt=dt)
            errs.append(np.abs(s.u - ref.u).max())
        assert abs(fit_order(dts, errs)[0] - order) < 0.25, (rk, errs)


def test_rk3_third_order_with_forcing_needs_projected_initial_state():
    """Regression (Step 1.6): a sampled MMS velocity is not discretely solenoidal; starting from it gave a
    spurious O(dt) error. With a projected initial state SSP-RK3 is third order."""
    x, y, t = mms.x, mms.y, mms.t
    psi = 64 * x**2 * (1 - x) ** 2 * y**2 * (1 - y) ** 2 * (1 + sp.sin(2 * t) / 2)
    ms = mms.manufacture_ns(psi, sp.cos(sp.pi * x) * sp.exp(-t), 50.0)

    def make():
        s = ns.MACNavierStokes2D(fv.Grid2D(12, 12), 50.0, {}, force=(ms.fx_fn, ms.fy_fn))
        s.set_state(ms.u_fn(s.Xu, s.Yu, 0 * s.Xu), ms.v_fn(s.Xv, s.Yv, 0 * s.Xv), project=True)
        return s

    ref = make()
    ref.run(0.3, dt=5e-4)
    dts, errs = [0.01, 0.005, 0.0025], []
    for dt in dts:
        s = make()
        s.run(0.3, dt=dt)
        errs.append(np.abs(s.u - ref.u).max())
    assert abs(fit_order(dts, errs)[0] - 3) < 0.2, errs
    s = ns.MACNavierStokes2D(fv.Grid2D(12, 12), 50.0, {})
    with pytest.warns(UserWarning, match="divergence-free"):
        s.set_state(ms.u_fn(s.Xu, s.Yu, 0 * s.Xu), ms.v_fn(s.Xv, s.Yv, 0 * s.Xv), project=False)


def _channel(nx, ny, k, Re=20.0):
    y0 = k / ny
    solid = None
    if k:
        solid = np.zeros((ny + k, nx), bool)
        solid[:k, :] = True
    g = fv.Grid2D(nx, ny + k, 2.0, 1.0 + y0)
    prof = lambda s, t: np.where(s > y0, 4 * (s - y0) * (1 + y0 - s), 0.0)  # noqa: E731
    s = ns.MACNavierStokes2D(g, Re, {"left": ns.wall(u=prof), "right": ns.outflow()}, solid=solid)
    s.set_state(np.zeros_like(s.u), np.zeros_like(s.v))
    return s


def test_solid_mask_equals_true_wall_horizontal():
    a, b = _channel(24, 12, 0), _channel(24, 12, 3)
    a.run(0.3, dt=2e-3)
    b.run(0.3, dt=2e-3)
    assert np.abs(a.u - b.u[3:]).max() < 1e-13 and np.abs(a.v - b.v[3:]).max() < 1e-13
    assert np.abs(a.pressure() - b.pressure()[3:]).max() < 1e-11


def test_solid_mask_equals_true_wall_vertical():
    n, k = 12, 2
    lid = lambda s, t: np.where(s < 1 + 1e-12, np.sin(np.pi * s) ** 2, 0.0)  # noqa: E731
    a = ns.MACNavierStokes2D(fv.Grid2D(n, n), 100.0, {"top": ns.wall(u=lid)})
    solid = np.zeros((n, n + k), bool)
    solid[:, n:] = True
    b = ns.MACNavierStokes2D(fv.Grid2D(n + k, n, 1 + k / n, 1.0), 100.0, {"top": ns.wall(u=lid)}, solid=solid)
    for s in (a, b):
        s.set_state(np.zeros_like(s.u), np.zeros_like(s.v))
        s.run(0.2, dt=4e-3)
    assert np.abs(a.u - b.u[:, : n + 1]).max() < 1e-13 and np.abs(a.v - b.v[:, :n]).max() < 1e-13


def test_hydrostatic_boussinesq_rest_state():
    """Stratified theta(y) with fluid at rest: buoyancy is a discrete gradient -> velocity stays 0."""
    g = fv.Grid2D(12, 12)
    en = ns.EnergySpec(
        Pe=50.0,
        bcs={
            "bottom": fv.dirichlet2(0.0),
            "top": fv.dirichlet2(1.0),
            "left": fv.neumann2(0.0),
            "right": fv.neumann2(0.0),
        },
    )
    s = ns.MACNavierStokes2D(g, 50.0, {}, energy=en, Ri=2.0)
    s.set_state(np.zeros_like(s.u), np.zeros_like(s.v), theta=g.Yc.copy())
    s.run(0.5, dt=1e-2)
    assert max(np.abs(s.u).max(), np.abs(s.v).max()) < 1e-12


def test_passive_scalar_conserved_in_closed_adiabatic_cavity():
    g = fv.Grid2D(16, 16)
    en = ns.EnergySpec(Pe=70.0, bcs={k: fv.neumann2(0.0) for k in fv.SIDES})
    s = ns.MACNavierStokes2D(g, 100.0, {"top": ns.wall(u=1.0)}, energy=en)
    th0 = np.exp(-30 * ((g.Xc - 0.3) ** 2 + (g.Yc - 0.6) ** 2))
    s.set_state(np.zeros_like(s.u), np.zeros_like(s.v), theta=th0)
    s.run(0.5)
    assert abs(s.theta.sum() - th0.sum()) < 1e-12 * th0.sum()


def test_checkpoint_roundtrip(tmp_path):
    a, _ = tg_solver(12)
    a.run(0.05, dt=5e-3)
    a.save_state(tmp_path / "ck.npz")
    a.run(0.1, dt=5e-3)
    b, _ = tg_solver(12)
    b.load_state(tmp_path / "ck.npz")
    b.run(0.1, dt=5e-3)
    assert np.array_equal(a.u, b.u) and np.array_equal(a.v, b.v) and a.nsteps == b.nsteps
