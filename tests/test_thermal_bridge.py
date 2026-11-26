"""The merged PINN project's slab problem as an airframe skin model."""

from __future__ import annotations

import numpy as np

from aerosynapse.aircraft import materials as mt
from aerosynapse.thermal import bridge as B


def test_fd_solver_matches_exact_series():
    fo, x = np.array([0.05, 0.5, 1.5]), np.linspace(0, 1, 11)
    for bi in (0.3, 3.0, 15.0):
        assert np.abs(B.theta_fd(bi, fo, x) - B.theta_exact(bi, fo, x)).max() < 1e-4


def test_panel_without_network_uses_the_analytic_series():
    r = B.panel(None, None, mt.LIBRARY["ti64"], 0.004, 800.0, 220.0, 360.0, 300.0)
    assert r["source"].get("pinn", 0) == 0
    assert min(r["T_inner_K"]) >= 220.0 and max(r["T_outer_K"]) <= 360.0 + 1e-6
    assert r["T_outer_K"][-1] > r["T_inner_K"][0]
