"""Physics checks for the whole-aircraft models, against exact or published values."""

from __future__ import annotations

import numpy as np
import pytest

from aerosynapse.aircraft import curves as C
from aerosynapse.aircraft import layers as Y
from aerosynapse.aircraft import propulsion as P
from aerosynapse.aircraft.atmosphere import isa


@pytest.mark.validation
@pytest.mark.parametrize("h,T,p,rho", [(0, 288.15, 101325, 1.225), (5000, 255.65, 54020, 0.7361), (11000, 216.65, 22632, 0.3639), (20000, 216.65, 5475, 0.08803)])
def test_isa_matches_standard_table(h, T, p, rho):
    a = isa(h)
    assert pytest.approx(T, abs=0.02) == a.T
    assert a.p == pytest.approx(p, rel=2e-3)
    assert a.rho == pytest.approx(rho, rel=2e-3)


@pytest.mark.unit
def test_hot_day_keeps_pressure_and_thins_the_air():
    a, b = isa(0), isa(0, 30.0)
    assert a.p == b.p and b.rho < a.rho and b.a > a.a


@pytest.mark.unit
def test_isa_refuses_unsupported_altitude():
    with pytest.raises(ValueError, match="outside"):
        isa(90000)


@pytest.mark.verification
def test_pseudosphere_has_constant_negative_curvature():
    t, ph = np.linspace(0.6, 2.2, 90), np.linspace(0.3, 2.6, 90)
    K = C.gaussian_curvature(*C.pseudosphere(t, ph, 1.7))
    assert np.median(K[8:-8, 8:-8]) == pytest.approx(-1 / 1.7**2, rel=0.02)


@pytest.mark.verification
def test_sphere_has_constant_positive_curvature():
    K = C.gaussian_curvature(*C.spherical_cap(np.linspace(0.4, 1.4, 90), np.linspace(0.2, 2.6, 90), 2.5))
    assert np.median(K[8:-8, 8:-8]) == pytest.approx(1 / 2.5**2, rel=0.02)


@pytest.mark.verification
def test_catenoid_is_negatively_curved_as_theory_says():
    u, ph, c = np.linspace(-1, 1, 120), np.linspace(0.2, 2.6, 120), 1.3
    K = C.gaussian_curvature(*C.catenoid(u, ph, c))
    exact = -1 / (c**2 * np.cosh(u / c) ** 4)
    assert np.allclose(K[10:-10, 60], exact[10:-10], rtol=0.03)


@pytest.mark.verification
def test_moebius_preserves_hyperbolic_distance():
    u, v, w = 0.3 + 0.2j, -0.1 + 0.5j, 0.4 - 0.3j
    a, b = C.moebius(np.array([u]), w)[0], C.moebius(np.array([v]), w)[0]
    assert C.poincare_distance(a, b) == pytest.approx(C.poincare_distance(u, v), rel=1e-10)


@pytest.mark.unit
def test_superformula_circle_and_superellipse_limits():
    ph = np.linspace(0, 2 * np.pi, 50)
    assert np.allclose(C.superformula(ph, 4, 2, 2, 2), 1.0)
    y, z = C.superellipse(ph, 2.0, 1.0, 4.0)
    assert np.allclose(np.abs(y / 2) ** 4 + np.abs(z) ** 4, 1.0, atol=1e-9)


@pytest.mark.verification
def test_clothoid_curvature_is_linear_in_arc_length():
    s = np.linspace(0.05, 2.0, 400)
    x, y = C.clothoid(s, 1.0)
    th = np.unwrap(np.arctan2(np.gradient(y, s), np.gradient(x, s)))
    k = np.gradient(th, s)[5:-5]
    fit = np.polyfit(s[5:-5], k, 1)
    assert np.abs(np.polyval(fit, s[5:-5]) - k).max() < 2e-3


@pytest.mark.validation
def test_blasius_and_transition_limits():
    assert Y.cf_laminar(1e6) == pytest.approx(0.001328)
    assert Y.cf_mixed(1e7, 0.0) == pytest.approx(Y.cf_turbulent(1e7))
    assert Y.cf_mixed(1e7, 1.0) == pytest.approx(Y.cf_laminar(1e7))
    assert Y.cf_laminar(1e7) < Y.cf_mixed(1e7, 0.4) < Y.cf_turbulent(1e7)


@pytest.mark.validation
def test_compressibility_lowers_friction_about_as_measured():
    """Adiabatic turbulent friction at Mach 2 is about 0.8 of the incompressible value."""
    a = isa(12000)
    L = 20.0
    ratio = Y.friction(a, 2.0, L, 0.0) / Y.cf_turbulent(a.rho * 2 * a.a * L / a.mu)
    assert 0.70 < ratio < 0.88


@pytest.mark.unit
def test_roughness_raises_friction_only_when_it_exceeds_the_smooth_value():
    a = isa(11000)
    smooth = Y.friction(a, 0.78, 4.0, 0.0)
    assert Y.friction(a, 0.78, 4.0, 0.0, roughness=30e-6) > smooth
    assert Y.friction(a, 0.78, 4.0, 0.0, roughness=1e-9) == smooth


@pytest.mark.unit
def test_hot_wall_lowers_friction():
    a = isa(12000)
    assert Y.friction(a, 2.0, 20.0, 0.0, Tw=500.0) < Y.friction(a, 2.0, 20.0, 0.0, Tw=300.0)


@pytest.mark.validation
def test_concorde_total_temperature():
    """Concorde's 127 C limit: Mach 2.02 at about 16 km on an ISA+ day gives that total temperature."""
    T0 = Y.t_total(isa(16000, 5.0).T, 2.02)
    assert pytest.approx(127, abs=8) == T0 - 273.15


@pytest.mark.verification
def test_skin_cannot_exceed_total_temperature_and_radiation_cools_it():
    a = isa(20000)
    s = Y.thermal_state(a, 3.2, 30.0, 400.0, 0.05, Y.MATERIALS["titanium"])
    assert s.T_nose <= s.T_total + 1e-6
    assert s.T_skin < s.T_adiabatic
    assert s.T_nose > s.T_skin_aft


@pytest.mark.verification
def test_cycle_reduces_to_ideal_turbojet():
    a = isa(11000)
    e = P.Engine.ideal_turbojet(20.0, 1500.0)
    c = P.cycle(e, a, 0.8)
    # the closed form neglects the fuel mass, so allow for f
    assert c.spec_thrust == pytest.approx(P.ideal_turbojet_thrust(a, 0.8, 20.0, 1500.0), rel=0.04)


@pytest.mark.validation
def test_turbofan_cruise_tsfc_is_realistic():
    c = P.cycle(P.Engine(), isa(11000), 0.78)
    assert c.valid
    assert 13e-6 < c.tsfc < 19e-6  # high-bypass cruise, kg/(N s)


@pytest.mark.verification
def test_bypass_improves_propulsive_efficiency():
    a = isa(11000)
    jet = P.cycle(P.Engine(bpr=0.0, fan_pr=1.0), a, 0.8)
    fan = P.cycle(P.Engine(bpr=8.0), a, 0.8)
    assert fan.eta_propulsive > jet.eta_propulsive
    assert fan.tsfc < jet.tsfc


@pytest.mark.unit
def test_hot_day_costs_thrust():
    ref = isa(0)
    e = P.Engine()
    t0, _ = P.available_thrust(e, ref, 0.2, 100.0, ref, 0.2)
    t1, _ = P.available_thrust(e, isa(0, 30.0), 0.2, 100.0, ref, 0.2, hot_dT=30.0)
    assert t1 < t0
