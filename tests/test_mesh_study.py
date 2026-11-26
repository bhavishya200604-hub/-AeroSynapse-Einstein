"""Grid-convergence index (Celik et al. 2008) used by scripts/cfd_mesh_study.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("cfd_mesh_study", Path(__file__).resolve().parents[1] / "scripts" / "cfd_mesh_study.py")
MS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MS)


@pytest.mark.parametrize("p_true", [1.0, 2.0, 1.5])
def test_gci_recovers_order_and_limit(p_true):
    """f = f0 + C h^p on grids with unequal ratios: the order and the extrapolated value come back exactly."""
    f0, C = 0.05, 0.8
    h = [15.16 / 68, 15.16 / 48, 15.16 / 34]
    f = [f0 + C * x**p_true for x in h]
    g = MS.gci3(h, f)
    assert g["kind"] == "monotonic"
    assert g["p"] == pytest.approx(p_true, rel=1e-6)
    assert g["f_ext"] == pytest.approx(f0, rel=1e-6)
    # the GCI (relative to the fine value, safety factor 1.25) bounds the true fine-grid error in absolute terms
    assert abs(f[0] - f0) <= g["gci_fine"] * abs(f[0])


def test_gci_reports_oscillation_without_an_order():
    g = MS.gci3([0.2, 0.3, 0.45], [0.10, 0.12, 0.09])
    assert g["kind"] == "oscillatory" and g["p"] is None and g["band"] == [0.09, 0.12]


def test_gci_flags_growing_change_as_divergent():
    """The Falcon full-model CL of 2026-11-11 (34, 48, 68 cells per length): the change 48 -> 68 is larger than
    34 -> 48, so no order may be reported (the formula's absolute value would give a spurious p of about 4.2)."""
    L = 15.16
    g = MS.gci3([L / 68, L / 48, L / 34], [0.08169, 0.09358, 0.09627])
    assert g["kind"] == "divergent" and g["p"] is None and g["gci_fine"] is None
    assert g["R"] == pytest.approx(0.01189 / 0.00269, rel=1e-3)


def test_gci_celik_worked_example():
    """Celik et al. (2008), Table 1, phi_1 column: N = 18000, 8000, 4500 cells in 2D; phi = 6.063, 5.972, 5.863.
    Published: p = 1.53, phi_ext = 6.1685, GCI_fine = 2.2 %."""
    h = [(1 / 18000) ** 0.5, (1 / 8000) ** 0.5, (1 / 4500) ** 0.5]
    g = MS.gci3(h, [6.063, 5.972, 5.863])
    assert g["p"] == pytest.approx(1.53, abs=0.01)
    assert g["f_ext"] == pytest.approx(6.1685, abs=2e-3)
    assert 100 * g["gci_fine"] == pytest.approx(2.2, abs=0.05)
