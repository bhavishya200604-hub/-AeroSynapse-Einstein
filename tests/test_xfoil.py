"""XFOIL adapter. Parsing is always tested; running XFOIL needs the program, so those tests skip without it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aerosynapse.exceptions import ConfigurationError
from aerosynapse.shapeopt import geometry as G
from aerosynapse.shapeopt import xfoil

SAMPLE = """
 xtrf =   1.000 (top)        1.000 (bottom)
 Mach =   0.000     Re =     6.000 e 6     Ncrit =   9.000

   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr
  ------ -------- --------- --------- -------- -------- --------
   0.000   0.0000   0.00792   0.00125  -0.0000   0.0500   0.0500
   4.000   0.4580   0.00824   0.00168  -0.0013   0.0500   0.0500
   4.000   0.4600   0.00830   0.00168  -0.0013   0.0500   0.0500
   8.000   0.9038   0.00987   0.00341  -0.0010   0.0235   0.0500
"""


def has_xfoil() -> bool:
    try:
        xfoil.executable()
    except ConfigurationError:
        return False
    return True


needs_xfoil = pytest.mark.skipif(not has_xfoil(), reason="XFOIL is not installed")


@pytest.mark.unit
def test_polar_parser_reads_rows_and_keeps_the_last_repeat():
    p = xfoil.parse_polar(SAMPLE)
    assert p["alpha"].tolist() == [0.0, 4.0, 8.0]
    assert p["CL"][1] == 0.46


@pytest.mark.unit
def test_empty_polar_is_empty_rather_than_an_error():
    assert len(xfoil.parse_polar("")["alpha"]) == 0


@pytest.mark.unit
def test_lift_above_the_maximum_gives_no_drag_value():
    p = xfoil.parse_polar(SAMPLE + "  10.000   0.8000   0.02000   0.00341  -0.0010   0.0235   0.0500\n")
    assert np.isnan(xfoil.at_cl(p, 1.2)["cd"])
    assert xfoil.at_cl(p, 0.4580)["cd"] == pytest.approx(0.0083, abs=1e-3)
    assert xfoil.at_cl(p, 1.2)["cl_max"] == pytest.approx(0.9038)


@pytest.mark.unit
def test_failed_runs_are_not_turned_into_numbers():
    assert np.isnan(xfoil.at_cl(xfoil.parse_polar(""), 0.5)["cd"])


@pytest.mark.unit
def test_mach_above_the_limit_is_refused():
    with pytest.raises(ConfigurationError, match="above"):
        xfoil.polar(G.load_shape("naca0012"), 1e6, mach=0.7)


@pytest.mark.validation
@needs_xfoil
def test_xfoil_matches_nasa_data_on_naca0012():
    d = json.loads((Path(__file__).resolve().parents[1] / "src" / "aerosynapse" / "data" / "naca0012_ladson.json").read_text(encoding="utf-8"))
    L = np.array(d["rows"])
    L = L[(L[:, 0] >= 0) & (L[:, 0] <= 10.5)]
    p = xfoil.polar(G.load_shape("naca0012"), 6e6, 0.0, 9.0, 0.05, -2, 12, 0.5)
    assert np.interp(L[:, 0], p["alpha"], p["CD"]) == pytest.approx(L[:, 2], rel=0.1)
    assert np.interp(L[:, 0], p["alpha"], p["CL"]) == pytest.approx(L[:, 1], abs=0.05)
