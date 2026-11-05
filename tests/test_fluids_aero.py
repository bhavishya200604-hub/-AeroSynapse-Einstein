"""Phase 6: NS analytic flows, every fluid BC, cavity vs Ghia, regime guard, BL, panel method, plugins."""

import json
from pathlib import Path

import numpy as np
import pytest

from aerosynapse.exceptions import ConfigurationError, RegimeError
from aerosynapse.numerics.boundary_layer import blasius, march_boundary_layer
from aerosynapse.numerics.external import import_line_csv, preflight, write_cavity_case
from aerosynapse.numerics.ns2d import FLUID_BC_CATALOGUE, FluidBC, NSSolver, cavity_centerlines, lid_driven_cavity
from aerosynapse.numerics.panel import solve_panel
from aerosynapse.physics.airfoil import cst, max_thickness, naca4
from aerosynapse.physics.regime import classify, guard, mach_guard
from aerosynapse.plugins.oscillator import Oscillator
from aerosynapse.registry import PHYSICS

pytestmark = pytest.mark.validation
NU = 0.1
WALLS = {"bottom": FluidBC("wall"), "top": FluidBC("wall")}
PER = {"left": FluidBC("periodic"), "right": FluidBC("periodic")}


def _uc(r):
    return r.centre_velocity()[0]


def test_couette_moving_wall_and_periodic():
    s = NSSolver(16, 8, 2.0, 1.0, NU, {**PER, "bottom": FluidBC("wall"), "top": FluidBC("wall", 1.0)})
    r = s.run(50, 1e-9)
    assert r.converged and np.abs(_uc(r) - s.yc).max() < 1e-8 and r.divergence_max < 1e-10


def test_poiseuille_body_force_pressure_and_velocity_inlet():
    ex = lambda y: 1.0 / (2 * NU) * y * (1 - y)
    s = NSSolver(16, 16, 1.0, 1.0, NU, {**PER, **WALLS}, body_force=(1.0, 0.0))
    r = s.run(50, 1e-9)
    assert np.abs(_uc(r) - ex(s.yc)).max() / ex(s.yc).max() < 5e-3
    s = NSSolver(
        32, 16, 2.0, 1.0, NU, {"left": FluidBC("pressure_inlet", pressure=2.0), "right": FluidBC("outlet"), **WALLS}
    )
    r = s.run(50, 1e-9)
    assert np.abs(_uc(r)[16] - ex(s.yc)).max() / ex(s.yc).max() < 5e-3 and r.divergence_max < 1e-10
    prof = lambda y: 6 * y * (1 - y)
    s = NSSolver(32, 16, 2.0, 1.0, NU, {"left": FluidBC("inlet", prof), "right": FluidBC("outlet"), **WALLS})
    r = s.run(50, 1e-9)
    assert np.abs(_uc(r)[24] - prof(s.yc)).max() / 1.5 < 5e-3


@pytest.mark.parametrize("kind", ["slip", "symmetry"])
def test_slip_symmetry_half_channel(kind):
    s = NSSolver(8, 16, 0.5, 1.0, NU, {**PER, "bottom": FluidBC("wall"), "top": FluidBC(kind)}, body_force=(1.0, 0.0))
    r = s.run(120, 1e-9)
    ex = 1.0 / (2 * NU) * (2 * s.yc - s.yc**2)
    assert np.abs(_uc(r) - ex).max() / ex.max() < 3e-3


def test_farfield_uniform_flow_is_preserved():
    ff = FluidBC("farfield", 1.0)
    s = NSSolver(
        16,
        16,
        1.0,
        1.0,
        NU,
        {
            "left": ff,
            "right": FluidBC("outlet"),
            "bottom": FluidBC("farfield", 0.0, tangential=1.0),
            "top": FluidBC("farfield", 0.0, tangential=1.0),
        },
    )
    r = s.run(5, 1e-10)
    assert np.abs(_uc(r) - 1.0).max() < 1e-8


def test_fluid_bc_validation():
    with pytest.raises(ConfigurationError):
        FluidBC("bogus")
    with pytest.raises(ConfigurationError):
        NSSolver(4, 4, 1.0, 1.0, NU, {"left": FluidBC("periodic"), "right": FluidBC("wall"), **WALLS})
    with pytest.raises(ConfigurationError):
        NSSolver(4, 4, 1.0, 2.0, NU, {**PER, **WALLS})
    for k in ("wall", "slip", "symmetry", "inlet", "pressure_inlet", "outlet", "periodic", "farfield"):
        assert {"definition", "implementation", "compatibility", "failure"} <= set(FLUID_BC_CATALOGUE[k])


def test_cavity_re100_vs_ghia():
    g = json.loads((Path(__file__).parents[1] / "src/aerosynapse/data/ghia1982.json").read_text())
    r = lid_driven_cavity(100, n=32, t_end=40)
    y, u, x, v = cavity_centerlines(r)
    gu, gv = g["u_vertical_centerline"], g["v_horizontal_centerline"]
    rms_u = np.sqrt(np.mean((np.interp(gu["coord"], y, u) - np.array(gu["100"])) ** 2))
    rms_v = np.sqrt(np.mean((np.interp(gv["coord"], x, v) - np.array(gv["100"])) ** 2))
    assert r.converged and rms_u < 0.02 and rms_v < 0.02


def test_regime_guards():
    assert classify("channel", 100) == "laminar" and classify("channel", 3000) == "transitional"
    assert classify("channel", 1e5) == "turbulent"
    with pytest.raises(RegimeError):
        guard("channel", 1e5)
    with pytest.raises(RegimeError):
        guard("cavity", 5000)  # laminar by rule of thumb but outside the validated range
    assert guard("cavity", 5000, extrapolate=True).startswith("EXTRAPOLATION")
    assert guard("cavity", 100).startswith("validated")
    with pytest.raises(RegimeError):
        mach_guard(0.8)
    with pytest.raises(RegimeError):
        classify("nope", 1)


def test_blasius_and_marching_boundary_layer():
    eta, fp, fpp0 = blasius()
    assert abs(fp[-1] - 1) < 1e-8 and 0.3 < fpp0 < 0.35
    U, nu = 1.0, 1e-4
    xs, y, u = march_boundary_layer(U, nu, 1.0, ny=200, nx=200)
    prof = np.interp(y * np.sqrt(U / (nu * xs[-1])), eta, fp)
    assert np.abs(u[-1] - prof).max() < 0.02


def test_panel_method_verification():
    errs = []
    for n in (32, 64):
        t = np.linspace(0, -2 * np.pi, n + 1)
        r = solve_panel(np.cos(t), np.sin(t), 0.0, lifting=False)
        errs.append(np.abs(r.cp - (1 - 4 * np.sin(np.arctan2(r.yc, r.xc)) ** 2)).max())
    assert max(errs) < 1e-10
    X, Y = naca4("0012", 101)
    assert abs(max_thickness(X, Y) - 0.12) < 1e-3
    assert abs(solve_panel(X, Y, 0.0).cl) < 1e-10
    r4 = solve_panel(X, Y, 4.0)
    ratio = r4.cl / (2 * np.pi * np.deg2rad(4))
    assert 1.0 < ratio < 1.15 and abs(r4.cl - r4.cl_pressure) / r4.cl < 0.01
    assert solve_panel(*naca4("2412", 101), 0.0).cm_c4 < 0  # cambered: nose-down moment
    r_pg = solve_panel(X, Y, 4.0, mach=0.5)
    assert r_pg.cl == pytest.approx(r4.cl / np.sqrt(0.75))
    with pytest.raises(RegimeError):
        solve_panel(X, Y, 4.0, mach=0.75)
    Xc, Yc = cst(np.array([0.17, 0.16, 0.15]), -np.array([0.17, 0.16, 0.15]))
    assert abs(solve_panel(Xc, Yc, 0.0).cl) < 1e-10 and max_thickness(Xc, Yc) > 0.05
    with pytest.raises(ConfigurationError):
        naca4("24")


def test_external_adapter_and_plugin(tmp_path):
    pf = preflight()
    assert isinstance(pf.available, bool) and pf.message
    case = write_cavity_case(tmp_path / "cav", 100, 16)
    assert (case / "system" / "blockMeshDict").read_text().count("16 16 1") == 1
    f = tmp_path / "line.csv"
    f.write_text("# y u\n0,0\n0.5,-0.2\nbad,row\n1,1\n")
    x, v = import_line_csv(f)
    assert x.tolist() == [0, 0.5, 1]
    osc = PHYSICS.get("harmonic_oscillator")()
    assert isinstance(osc, Oscillator)
    for scheme, bounded in (("verlet", True), ("euler", False)):
        q, p = osc.integrate(1.0, 0.0, 0.1, 2000, scheme)
        E = osc.energy(q, p)
        drift = abs(E[-1] - E[0]) / E[0]
        assert (drift < 0.01) == bounded
    with pytest.raises(ValueError):
        osc.integrate(1, 0, 0.1, 2, "rk9")
