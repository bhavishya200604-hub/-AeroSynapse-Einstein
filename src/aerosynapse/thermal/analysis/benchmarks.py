"""Reference benchmark data with provenance (see docs/LITERATURE.md for verification status).

Every constant here is VERIFIED or VERIFIED-SECONDARY (labelled); flagged entries are excluded from error
statistics but kept for transparency.
"""

from __future__ import annotations

import numpy as np

GHIA_SOURCE = (
    "Ghia, Ghia & Shin (1982) J. Comput. Phys. 48:387-411, Tables I/II; VERIFIED-SECONDARY via "
    "gist.github.com/ivan-pi (3e9326d1..., caa6c673...); original not accessible"
)

# Table I: u along the vertical line through the geometric centre (x = 0.5)
GHIA_Y = np.array(
    [
        1.0000,
        0.9766,
        0.9688,
        0.9609,
        0.9531,
        0.8516,
        0.7344,
        0.6172,
        0.5000,
        0.4531,
        0.2813,
        0.1719,
        0.1016,
        0.0703,
        0.0625,
        0.0547,
        0.0000,
    ]
)
GHIA_U = {
    100: [
        1.00000,
        0.84123,
        0.78871,
        0.73722,
        0.68717,
        0.23151,
        0.00332,
        -0.13641,
        -0.20581,
        -0.21090,
        -0.15662,
        -0.10150,
        -0.06434,
        -0.04775,
        -0.04192,
        -0.03717,
        0.00000,
    ],
    400: [
        1.00000,
        0.75837,
        0.68439,
        0.61756,
        0.55892,
        0.29093,
        0.16256,
        0.02135,
        -0.11477,
        -0.17119,
        -0.32726,
        -0.24299,
        -0.14612,
        -0.10338,
        -0.09266,
        -0.08186,
        0.00000,
    ],
    1000: [
        1.00000,
        0.65928,
        0.57492,
        0.51117,
        0.46604,
        0.33304,
        0.18719,
        0.05702,
        -0.06080,
        -0.10648,
        -0.27805,
        -0.38289,
        -0.29730,
        -0.22220,
        -0.20196,
        -0.18109,
        0.00000,
    ],
    3200: [
        1.00000,
        0.53236,
        0.48296,
        0.46547,
        0.46101,
        0.34682,
        0.19791,
        0.07156,
        -0.04272,
        -0.86636,
        -0.24427,
        -0.34323,
        -0.41933,
        -0.37827,
        -0.35344,
        -0.32407,
        0.00000,
    ],
}
# Table II: v along the horizontal line through the geometric centre (y = 0.5)
GHIA_X = np.array(
    [
        1.0000,
        0.9688,
        0.9609,
        0.9531,
        0.9453,
        0.9063,
        0.8594,
        0.8047,
        0.5000,
        0.2344,
        0.2266,
        0.1563,
        0.0938,
        0.0781,
        0.0703,
        0.0625,
        0.0000,
    ]
)
GHIA_V = {
    100: [
        0.00000,
        -0.05906,
        -0.07391,
        -0.08864,
        -0.10313,
        -0.16914,
        -0.22445,
        -0.24533,
        0.05454,
        0.17527,
        0.17507,
        0.16077,
        0.12317,
        0.10890,
        0.10091,
        0.09233,
        0.00000,
    ],
    400: [
        0.00000,
        -0.12146,
        -0.15663,
        -0.19254,
        -0.22847,
        -0.23827,
        -0.44993,
        -0.38598,
        0.05186,
        0.30174,
        0.30203,
        0.28124,
        0.22965,
        0.20920,
        0.19713,
        0.18360,
        0.00000,
    ],
    1000: [
        0.00000,
        -0.21388,
        -0.27669,
        -0.33714,
        -0.39188,
        -0.51500,
        -0.42665,
        -0.31966,
        0.02526,
        0.32235,
        0.33075,
        0.37095,
        0.32627,
        0.30353,
        0.29012,
        0.27485,
        0.00000,
    ],
    3200: [
        0.00000,
        -0.39017,
        -0.47425,
        -0.52357,
        -0.54053,
        -0.44307,
        -0.37401,
        -0.31184,
        0.00999,
        0.28188,
        0.29030,
        0.37119,
        0.42768,
        0.41906,
        0.40917,
        0.39560,
        0.00000,
    ],
}
# (Re, coordinate) entries excluded from statistics (inconsistent with neighbours; see LITERATURE.md)
GHIA_EXCLUDE_U = {(3200, 0.4531)}
GHIA_EXCLUDE_V = {(400, 0.9063)}

SCHAFER_TUREK_2D1 = {
    "source": "FEATFLOW DFG benchmark 2D-1 reference page (VERIFIED 2026-10-01)",
    "C_D": 5.57953523384,
    "C_L": 0.010618948146,
    "dp": 0.11752016697,
    "H": 0.41,
    "L": 2.2,
    "xc": 0.2,
    "yc": 0.2,
    "r": 0.05,
    "U_max": 0.3,
    "U_mean": 0.2,
    "nu": 1e-3,
    "rho": 1.0,
    "p_points": ((0.15, 0.2), (0.25, 0.2)),
}

SCHAFER_TUREK_2D2 = {
    "source": "FEATFLOW DFG benchmark 2D-2 reference page, level 6, dt = 1/200 (VERIFIED 2026-10-01)",
    "C_D_max": 3.2200,
    "C_L_max": 0.9859,
    "St": 0.30188,
    "U_max": 1.5,
    "U_mean": 1.0,
    "nu": 1e-3,
}

DE_VAHL_DAVIS_NU = {
    "source": "de Vahl Davis (1983) via Wasberg et al. arXiv physics/0305049 Table 1 (VERIFIED-SECONDARY)",
    "Pr": 0.71,
    1e4: 2.243,
    1e5: 4.519,
    1e6: 8.800,
}
HORTMANN_NU = {"source": "Hortmann et al. (1990) via the same table", 1e4: 2.245, 1e5: 4.522, 1e6: 8.825}


def ghia_points(Re: int, which: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(coordinate, value, include-mask) for Table I ('u') or Table II ('v')."""
    if which == "u":
        c, v, ex = GHIA_Y, np.array(GHIA_U[Re]), GHIA_EXCLUDE_U
    else:
        c, v, ex = GHIA_X, np.array(GHIA_V[Re]), GHIA_EXCLUDE_V
    mask = np.array([(Re, float(ci)) not in ex for ci in c])
    return c, v, mask
