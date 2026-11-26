"""Fighter design space: variables -> feature tree.

Every design is built from one dictionary of named variables. Families fix which topology options exist;
the "freeform" family switches each option on by a continuous size variable, so the optimiser can add or
remove canards, strakes, tails, fins, blades, cuts and ducts (a feature whose size falls below its threshold
is left out: a topology change).

Main-body constraints (``envelopes``): the engine (a cylinder with its length and diameter), the cockpit and
the fuel volume must fit inside the outer body. They are checked with the geometry kernel's distance field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Var = tuple[str, float, float, str]  # name, lower, upper, meaning

BODY: list[Var] = [
    ("length", 13.0, 19.5, "body length, m"),
    ("width", 0.75, 1.5, "maximum half-width of the body, m (wide = blended lifting body)"),
    ("height", 0.55, 0.95, "maximum half-height of the body, m"),
    ("nose_frac", 0.16, 0.34, "nose length / body length"),
    ("max_frac", 0.38, 0.62, "station of the largest section / length"),
    ("tail_frac", 0.35, 0.85, "half-size of the tail end / largest half-size"),
    ("squareness", 2.0, 4.0, "superellipse exponent of the sections: 2 round, 4 boxy with chines"),
    ("camber_line", -0.15, 0.15, "nose droop (-) or rise (+) of the body axis, m"),
]
WING: list[Var] = [
    ("w_area", 26.0, 60.0, "wing reference area (both sides), m^2"),
    ("w_ar", 1.8, 4.2, "wing aspect ratio"),
    ("w_taper", 0.08, 0.45, "tip / root chord"),
    ("w_sweep", 25.0, 62.0, "leading-edge sweep, deg"),
    ("w_tc", 0.03, 0.065, "thickness / chord"),
    ("w_twist", -5.0, 0.0, "tip washout, deg"),
    ("w_camber", 0.0, 0.025, "camber / chord"),
    ("w_x", 0.38, 0.60, "wing root leading edge station / body length"),
    ("w_z", -0.4, 0.4, "wing vertical position / body half-height (low to high)"),
    ("w_dihedral", -6.0, 6.0, "dihedral, deg"),
]
OPTIONS: dict[str, list[Var]] = {
    "canard": [("c_size", 0.0, 0.18, "canard area / wing area"), ("c_sweep", 35.0, 62.0, "canard sweep, deg"), ("c_x", 0.16, 0.34, "canard station / length")],
    "strake": [("s_size", 0.0, 0.12, "strake (LERX) area / wing area"), ("s_sweep", 65.0, 80.0, "strake sweep, deg")],
    "htail": [("h_size", 0.0, 0.30, "horizontal tail area / wing area"), ("h_sweep", 30.0, 55.0, "tail sweep, deg")],
    "fins": [("v_size", 0.08, 0.26, "total fin area / wing area"), ("v_cant", 0.0, 40.0, "cant of twin fins from vertical, deg (0 = single fin)"), ("v_sweep", 30.0, 60.0, "fin sweep, deg")],
    "ventral": [("vf_size", 0.0, 0.04, "ventral fin area / wing area")],
    "blades": [("b_count", 0.0, 3.0, "blades (wing fences) per side"), ("b_height", 0.1, 0.35, "blade height, m")],
    "notch": [("n_depth", 0.0, 0.06, "leading-edge notch (dogtooth cut) width / root chord"), ("n_eta", 0.3, 0.75, "notch span station")],
    "duct": [("d_radius", 0.0, 0.42, "through-flow inlet duct radius, m (0 = closed body)"), ("d_inlet", 0.22, 0.40, "inlet station / length")],
    "bay": [("bay_volume", 0.0, 3.0, "internal bay volume, m^3")],
}
PROPULSION: list[Var] = [("tw", 0.8, 1.3, "sea-level static afterburning thrust / take-off weight")]

FAMILIES: dict[str, dict[str, Any]] = {
    "conventional": {"options": ["strake", "htail", "fins", "ventral", "duct", "bay"], "fixed": {"c_size": 0.0, "v_cant": 0.0, "b_count": 0.0, "n_depth": 0.0}},
    "canard_delta": {"options": ["canard", "fins", "duct", "bay"], "fixed": {"h_size": 0.0, "s_size": 0.0, "v_cant": 0.0, "b_count": 0.0, "n_depth": 0.0}},
    "twin_tail_lerx": {"options": ["strake", "htail", "fins", "duct", "bay"], "fixed": {"c_size": 0.0, "b_count": 0.0, "n_depth": 0.0}, "min": {"v_cant": 10.0}},
    "tailless_blended": {"options": ["strake", "fins", "blades", "notch", "duct", "bay"], "fixed": {"c_size": 0.0, "h_size": 0.0}, "min": {"v_cant": 15.0, "width": 1.0}},
    "freeform": {"options": list(OPTIONS), "fixed": {}},
}
ENGINE: dict[str, float] = {"length": 5.0, "diameter": 1.08, "aft_gap": 0.3}  # one afterburning turbofan of about 120 kN class
COCKPIT_X = 0.17  # cockpit station / length
COCKPIT_RADII = (1.2, 0.42, 0.45)
OPTION_ON = {"c_size": 0.02, "s_size": 0.01, "h_size": 0.03, "vf_size": 0.004, "b_count": 0.5, "n_depth": 0.01, "d_radius": 0.12, "bay_volume": 0.2}


def variables(family: str) -> list[Var]:
    fam = FAMILIES[family]
    vs = BODY + WING + [v for o in fam["options"] for v in OPTIONS[o]] + PROPULSION
    return [v for v in vs if v[0] not in fam["fixed"]]


def bounds(family: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vs = variables(family)
    lo = np.array([max(v[1], FAMILIES[family].get("min", {}).get(v[0], v[1])) for v in vs])
    return lo, np.array([v[2] for v in vs])


def decode(family: str, z: NDArray[np.float64]) -> dict[str, float]:
    lo, hi = bounds(family)
    d = {v[0]: float(lo[i] + (hi[i] - lo[i]) * np.clip(z[i], 0, 1)) for i, v in enumerate(variables(family))}
    return complete(family, d)


def encode(family: str, d: dict[str, float]) -> NDArray[np.float64]:
    lo, hi = bounds(family)
    return np.array([np.clip((d.get(v[0], lo[i]) - lo[i]) / (hi[i] - lo[i]), 0, 1) for i, v in enumerate(variables(family))])


def complete(family: str, d: dict[str, float]) -> dict[str, float]:
    """Fill every variable the family does not use with its fixed value or 'off'."""
    out = {v[0]: v[1] for v in BODY + WING + PROPULSION}
    for vs in OPTIONS.values():
        for v in vs:
            out[v[0]] = 0.0 if v[0] in OPTION_ON else v[1]
    out["v_size"], out["v_cant"], out["v_sweep"] = 0.16, 0.0, 45.0
    out.update(FAMILIES[family]["fixed"])
    out.update(d)
    return out


def _r(x: float) -> float:
    return round(float(x), 5)


def build(d: dict[str, float], name: str = "design") -> dict[str, Any]:
    """Feature tree of a design."""
    L, W, H = d["length"], d["width"], d["height"]
    nose, xm, tail, n = d["nose_frac"] * L, d["max_frac"] * L, d["tail_frac"], d["squareness"]
    dz = d["camber_line"]
    st = [
        [0.0, 0.02, 0.02, dz, 2.0],
        [0.35 * nose, 0.45 * W, 0.50 * H, 0.6 * dz, 2.0],
        [nose, 0.80 * W, 0.85 * H, 0.2 * dz, 0.5 * (2 + n)],
        [xm, W, H, 0.0, n],
        [0.5 * (xm + L), 0.5 * (1 + tail) * W, 0.5 * (1 + tail) * H, 0.0, n],
        [L, tail * W, tail * H, 0.0, 0.5 * (2 + n)],
    ]
    # stations must increase: squeeze the nose stations if the largest section is close to the nose
    st[2][0] = min(st[2][0], 0.9 * st[3][0])
    st[1][0] = min(st[1][0], 0.9 * st[2][0])
    feats: list[dict[str, Any]] = [{"id": "body", "type": "fuselage", "op": "add", "stations": [[_r(v) for v in s] for s in st]}]
    feats.append({"id": "canopy", "type": "ellipsoid", "op": "add", "radii": list(COCKPIT_RADII), "translate": [_r(COCKPIT_X * L + 1.0), 0.0, _r(0.72 * H)], "blend": 0.25})
    # wing: span and chord from area and aspect ratio; the root sits at the body side
    S, AR, lam = d["w_area"], d["w_ar"], d["w_taper"]
    b = np.sqrt(S * AR)
    cr = 2 * S / (b * (1 + lam))
    y0 = 0.55 * W  # wing root inside the body
    semi = b / 2 - y0
    wing = {"id": "wing", "type": "wing", "op": "add", "span": _r(semi), "root_chord": _r(cr), "taper": _r(lam), "sweep": _r(d["w_sweep"]), "dihedral": _r(d["w_dihedral"]),
            "twist": _r(d["w_twist"]), "tc": _r(d["w_tc"]), "tip_tc": _r(0.8 * d["w_tc"]), "camber": _r(d["w_camber"]), "camber_pos": 0.35,
            "profile": "naca" if d["w_tc"] > 0.045 else "biconvex", "translate": [_r(d["w_x"] * L), _r(y0), _r(d["w_z"] * H)], "mirror": True, "blend": 0.12}
    feats.append(wing)
    x_te_root = d["w_x"] * L + cr
    if d["c_size"] > OPTION_ON["c_size"]:
        Sc = d["c_size"] * S
        bc = np.sqrt(Sc * 2.6)
        crc = 2 * Sc / (bc * 1.35)
        feats.append({"id": "canard", "type": "canard", "op": "add", "span": _r(bc / 2), "root_chord": _r(crc), "taper": 0.35, "sweep": _r(d["c_sweep"]), "dihedral": 0.0,
                      "tc": 0.04, "profile": "biconvex", "translate": [_r(d["c_x"] * L), _r(0.6 * W), _r(d["w_z"] * H + 0.35 * H)], "mirror": True, "blend": 0.05})
    if d["s_size"] > OPTION_ON["s_size"]:
        Ss = d["s_size"] * S
        cs = min(0.42 * L, d["w_x"] * L - nose * 0.6)
        sp = max(2 * Ss / cs, 0.15)  # nearly triangular: tip chord ~ 0
        feats.append({"id": "strake", "type": "strake", "op": "add", "span": _r(sp), "root_chord": _r(cs), "taper": 0.03, "sweep": _r(d["s_sweep"]), "dihedral": 0.0, "tc": 0.025,
                      "profile": "diamond", "translate": [_r(d["w_x"] * L - cs + 0.25 * cr), _r(0.75 * W), _r(d["w_z"] * H)], "mirror": True, "blend": 0.08})
    if d["h_size"] > OPTION_ON["h_size"]:
        Sh = d["h_size"] * S
        bh = np.sqrt(Sh * 3.0)
        crh = 2 * Sh / (bh * 1.35)
        xh = min(L - 0.9 * crh, max(x_te_root + 0.2, L - 1.3 * crh))
        feats.append({"id": "htail", "type": "tail", "op": "add", "span": _r(bh / 2), "root_chord": _r(crh), "taper": 0.35, "sweep": _r(d["h_sweep"]), "dihedral": -2.0, "tc": 0.04,
                      "profile": "biconvex", "translate": [_r(xh), _r(0.5 * W * tail + 0.3), _r(d["w_z"] * H - 0.25 * H)], "mirror": True, "blend": 0.05})
    Sv = d["v_size"] * S
    twin = d["v_cant"] > 5.0
    each = Sv / (2 if twin else 1)
    hv = np.sqrt(each * 1.4)
    crv = 2 * each / (hv * 1.3)
    xv = min(L - 0.85 * crv, max(x_te_root - 0.4 * crv, L - 1.6 * crv))
    feats.append({"id": "fins", "type": "fin", "op": "add", "span": _r(hv), "root_chord": _r(crv), "taper": 0.3, "sweep": _r(d["v_sweep"]), "dihedral": _r(90 - d["v_cant"]),
                  "tc": 0.045, "profile": "biconvex", "translate": [_r(xv), _r(0.55 * W if twin else 0.0), _r(0.6 * H)], "mirror": twin, "blend": 0.06})
    if d["vf_size"] > OPTION_ON["vf_size"]:
        Svf = d["vf_size"] * S / 2
        hvf = np.sqrt(Svf * 1.2)
        feats.append({"id": "ventral", "type": "fin", "op": "add", "span": _r(hvf), "root_chord": _r(2 * Svf / (hvf * 1.4)), "taper": 0.4, "sweep": 45.0, "dihedral": -70.0,
                      "tc": 0.04, "profile": "diamond", "translate": [_r(0.8 * L), _r(0.45 * W), _r(-0.6 * H)], "mirror": True})
    if d["b_count"] > OPTION_ON["b_count"]:
        nb = int(round(d["b_count"]))
        ch = 0.6 * cr
        feats.append({"id": "blades", "type": "blade", "op": "add", "span": _r(d["b_height"]), "root_chord": _r(ch), "taper": 0.5, "sweep": 55.0, "dihedral": 90.0, "tc": 0.05,
                      "profile": "diamond", "translate": [_r(d["w_x"] * L + 0.45 * semi * np.tan(np.radians(d["w_sweep"])) - 0.1 * ch), _r(y0 + 0.45 * semi), _r(d["w_z"] * H + 0.02)],
                      "pattern": {"count": nb, "step": [_r(0.18 * semi * np.tan(np.radians(d["w_sweep"]))), _r(0.18 * semi), 0.0]}, "mirror": True})
    if d["n_depth"] > OPTION_ON["n_depth"]:
        e = d["n_eta"]
        xl = d["w_x"] * L + e * semi * np.tan(np.radians(d["w_sweep"]))
        feats.append({"id": "notch", "type": "box", "op": "cut", "size": [_r(0.5 * cr), _r(d["n_depth"] * cr), 1.0], "translate": [_r(xl), _r(y0 + e * semi), _r(d["w_z"] * H)], "mirror": True})
    if d["d_radius"] > OPTION_ON["d_radius"]:
        r = d["d_radius"]
        xi = d["d_inlet"] * L
        feats.append({"id": "inlet_duct", "type": "duct", "op": "cut", "path": [[_r(xi - 0.6), _r(0.95 * W), _r(-0.25 * H)], [_r(xi + 1.5), _r(0.6 * W), _r(-0.2 * H)],
                      [_r(L - ENGINE["length"] - 0.5), _r(0.25 * W), 0.0], [_r(L + 0.3), _r(0.25 * W * tail), 0.0]], "radius": [_r(r), _r(0.95 * r), _r(0.9 * r), _r(0.9 * r)], "mirror": True})
    if d["bay_volume"] > OPTION_ON["bay_volume"]:
        v = d["bay_volume"]
        lb = min(4.2, 0.3 * L)
        wb = min(0.9 * W, v / (lb * 0.45))
        feats.append({"id": "bay", "type": "box", "op": "cut", "size": [_r(lb), _r(wb), 0.45], "round": 0.05, "translate": [_r(0.5 * L), 0.0, _r(-0.45 * H)]})
    return {"name": name, "units": "m", "variables": {k: _r(v) for k, v in d.items()}, "features": feats}


@dataclass
class Geometry:
    S: float
    b: float
    c_mac: float
    x_mac_le: float
    L: float
    wing: dict[str, Any]


def reference(design: dict[str, Any]) -> Geometry:
    """Reference area, span and mean chord: the wing extended to the centre line."""
    w = next(f for f in design["features"] if f["id"] == "wing")
    v = design["variables"]
    S, AR, lam = v["w_area"], v["w_ar"], v["w_taper"]
    b = float(np.sqrt(S * AR))
    cr = 2 * S / (b * (1 + lam))
    mac = 2 / 3 * cr * (1 + lam + lam**2) / (1 + lam)
    ymac = b / 6 * (1 + 2 * lam) / (1 + lam)
    x_le = w["translate"][0] - w["translate"][1] * np.tan(np.radians(v["w_sweep"])) + ymac * np.tan(np.radians(v["w_sweep"]))
    return Geometry(float(S), b, float(mac), float(x_le), float(v["length"]), w)


def envelope_points(d: dict[str, float], n: int = 24) -> dict[str, NDArray[np.float64]]:
    """Points on the engine and cockpit envelopes that must lie inside the outer body."""
    L = d["length"]
    e = ENGINE
    x = np.linspace(L - e["aft_gap"] - e["length"], L - e["aft_gap"], 10)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r0 = 0.5 * e["diameter"] + 0.04  # engine plus clearance; the afterburner and nozzle taper to 80 % over the last 1.5 m
    r = r0 * (1 - 0.2 * np.clip((x - (L - e["aft_gap"] - 1.5)) / 1.5, 0, 1))
    eng = np.array([[xx, rr * np.cos(t), rr * np.sin(t)] for xx, rr in zip(x, r, strict=True) for t in th])
    cx = COCKPIT_X * L + 1.0
    ck = np.array([[cx + 0.9 * np.cos(t), 0.35 * np.sin(t), 0.25 * d["height"]] for t in th])
    return {"engine": eng, "cockpit": ck}


def outer(design: dict[str, Any]) -> dict[str, Any]:
    """The design without internal cuts (ducts and bays), for envelope checks of the outer body."""
    return {**design, "features": [f for f in design["features"] if not (f.get("op") == "cut" and f["id"] in ("inlet_duct", "bay"))]}
