"""Phase 6 gate: registry / decision parsing, the traceability audit (found / mismatch / no data), the checklist and
the figures from fake inputs."""

import matplotlib.pyplot as plt

from aerosynapse.thermal.experiments import p6_gate as G
from aerosynapse.thermal.viz import gate6_figs as F

REG = """## Step 6.1 — scaling (`python experiments/p6_1_scaling.py`)
| ID | Quantity | Value |
|---|---|---|
| R6.1a | base | B_slab 0.0022 |
| R6.1j | equal error | PINN Adam 250, 0.0024 [max 0.0027], 20 s vs FD N 10, 26 ms (781x) and Chebyshev N 6 (191x) |
## Step 6.2 — UQ
| R6.2b | ensemble | B_slab 67 / 87 / 92 / 95 %, 1.30, 0.29 |
| R6.2d | ece | B_slab 0.119 → 0.019 |
## Phase 7
| R7.1a | other | 1 |
"""
DEC = """| D103 | 2026-10-09 | Step 6.1 design (brief ...) | | |
| D104 | 2026-10-09 | Step 6.1 result: ... uses Step 6.2 members | | |
| D114 | 2026-10-10 | User request "1024" (Step 6.4b, experiments/p6_4_extended.py) | | |
| D121 | 2026-10-11 | Step 6.6b design (the Step 6.4 adapters) | | |
"""


def test_registry_and_decisions():
    s = G.registry_sections(REG)
    assert set(s) == {"6.1", "6.2"} and sorted(s["6.1"]["ids"]) == ["R6.1a", "R6.1j"]
    d = G.decisions_for(DEC)
    assert d["6.1"] == ["D103", "D104"] and d["6.2"] == [] and d["6.4b"] == ["D114"] and d["6.6b"] == ["D121"]
    assert d["6.4"] == []  # mentioning "the Step 6.4 adapters" is not a Step 6.4 entry


def _mets(ratio_fd=781.2, ece=0.1188):
    return {
        "6.1": {
            "equal_error": {
                "B_slab": [
                    {
                        "target": 0.01,
                        "pinn": {"wall_s": ratio_fd, "error": 0.00238},
                        "fd": {"wall_s": 1.0},
                        "spectral": {"wall_s": ratio_fd / 190.8},
                    }
                ]
            }
        },
        "6.2": {
            "analysis": {
                "B_slab": {
                    "methods": {
                        "ensemble_M2": {},
                        "ensemble_M8": {"dispersion": 1.2998, "coverage": {"0.95": 0.9485}, "ece": ece},
                    }
                }
            }
        },
    }


def test_audit_found_mismatch_nodata():
    a = {(x["id"], x["label"]): x for x in G.audit(_mets(), REG)}
    assert a[("R6.1j", "B_slab equal error 1e-2: PINN / FD time")]["status"] == "FOUND"
    assert a[("R6.1j", "B_slab equal error 1e-2: PINN / Chebyshev time")]["status"] == "FOUND"
    assert a[("R6.2b", "B_slab ensemble M 8 dispersion")]["status"] == "FOUND"
    assert a[("R6.2d", "B_slab ensemble M 8 ECE (raw)")]["status"] == "FOUND"
    assert a[("R6.4c", "Graetz solver ST_Re (Monte Carlo)")]["status"] == "NO DATA"
    bad = {(x["id"], x["label"]): x for x in G.audit(_mets(ratio_fd=783.4, ece=0.2), REG)}
    assert bad[("R6.1j", "B_slab equal error 1e-2: PINN / FD time")]["status"] == "MISMATCH"
    assert bad[("R6.2d", "B_slab ensemble M 8 ECE (raw)")]["status"] == "MISMATCH"
    assert G.ens_key({"ensemble_M2": {}, "ensemble_M8": {}, "mcd_0.01_s0": {}}) == "ensemble_M8"


def test_checklist_headlines_and_figures():
    mets = _mets()
    runs = {s: None for s in G.STEPS}
    inv = G.inventory(runs, mets, G.registry_sections(REG), G.decisions_for(DEC))
    g = {"inventory": inv, "audit": G.audit(mets, REG), "headlines": G.headlines(mets)}
    items = {it["item"]: it["status"] for it in G.checklist(g)}
    assert items["Every brief item traced to RESULTS_REGISTRY"] == "FAIL"  # the fake registry lacks most IDs
    assert items["Headline numbers re-derived and found in the registry"] == "PASS"
    assert any(h["step"] == "6.2" and "95 %" in h["text"] for h in g["headlines"])
    F.style()
    for fig in (F.overview(mets), F.traceability(g), F.completeness3d(g)):
        plt.close(fig)
    m67 = {
        "analysis": {
            "D_graetz": {"pinn_error": 0.047, "speedup_per_query": 191.0, "break_even_ensemble": 615.0},
            "A_steady": {"pinn_error": 0.008, "speedup_per_query": 0.29, "break_even_ensemble": None},
        }
    }
    plt.close(F.landscape3d(m67))
