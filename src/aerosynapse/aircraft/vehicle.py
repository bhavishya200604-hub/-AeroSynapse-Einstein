r"""One geometry for every aircraft: a body of any section, plus a lifting surface.

A tube-and-wing airliner, a blended wing body, a flying wing and a freeform lifting body are not
separate models. They are regions of the same design space:

* Lengthwise, separate nose and tail laws meet at the widest section. The nose length and tail
  length are free, so the outline runs from a constant-section tube with a long tail cone to a
  spindle close to Sears-Haack. The tail can blend into a tractrix, the generating curve of the
  pseudosphere.
* The cross section is a Gielis superformula outline with Fourier terms, then a Moebius map of the
  Poincare disk, so each section is a hyperbolic-geometry deformation of a Euclidean outline.
* The wing has area, aspect ratio, taper, sweep, thickness, twist and winglet.

Wide flat bodies carry lift: their planform is sliced into strips and joined to the wing, so the
same lifting-line solution serves any shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from aerosynapse.aircraft import curves as C
from aerosynapse.shapeopt import wing as wg

FloatArr = NDArray[np.float64]

#        name          low    high  meaning
VARS = [
    ("L", 12.0, 60.0),  # body length, m
    ("W", 1.0, 16.0),  # body maximum half width, m
    ("H", 0.8, 3.0),  # body maximum half height, m
    ("fn", 0.06, 0.60),  # nose length, fraction of body
    ("ft", 0.15, 0.75),  # tail length, fraction of body
    ("p", 1.3, 4.0),  # nose fullness
    ("tq", 1.0, 2.5),  # tail fullness
    ("tract", 0.0, 1.0),  # tractrix tail blend
    ("n1", 1.2, 6.0),  # superformula exponent: section squareness
    ("wi", -0.45, 0.45),  # hyperbolic warp: Moebius translation along the vertical axis
    ("f2", -0.15, 0.15),  # Fourier 2-lobe term
    ("f3", -0.10, 0.10),  # Fourier 3-lobe term
    ("upsweep", 0.0, 1.0),  # tail upsweep, in body half heights
    ("S", 20.0, 500.0),  # outer wing area, m^2
    ("AR", 3.0, 16.0),  # wing aspect ratio
    ("taper", 0.12, 0.9),
    ("sweep", 0.0, 65.0),  # quarter-chord sweep, deg
    ("tc", 0.05, 0.17),  # wing thickness ratio
    ("twist", -8.0, 2.0),  # tip twist, deg
    ("winglet", 0.0, 0.14),  # height as a fraction of semi-span
    ("x_wing", 0.2, 0.75),  # wing root position along the body
    ("Vh", 0.0, 1.3),  # horizontal tail volume coefficient
    ("Vv", 0.0, 0.12),  # vertical tail volume coefficient
]
NAMES = [v[0] for v in VARS]
LO = np.array([v[1] for v in VARS])
HI = np.array([v[2] for v in VARS])


def to_dict(x: FloatArr) -> dict[str, float]:
    return {k: float(v) for k, v in zip(NAMES, x, strict=False)}


def to_array(d: dict[str, float]) -> FloatArr:
    return np.array([d[k] for k in NAMES])


def lengthwise(xi: FloatArr, fn: float, ft: float, p: float, tq: float, tract: float) -> FloatArr:
    """Normalised body outline along its length: a nose law times a tail law."""
    sn = np.clip(xi / fn, 0.0, 1.0)
    nose = (1 - (1 - sn) ** p) ** 0.5
    u = np.clip((1 - xi) / ft, 0.0, 1.0)
    tail = (1 - (1 - u) ** 1.7) ** (1 / tq)
    if tract > 0:
        k = 1 / np.cosh(3.2)
        tail = (1 - tract) * tail + tract * (1 / np.cosh(3.2 * (1 - u)) - k) / (1 - k)
    r = nose * tail
    return r / max(float(r.max()), 1e-9)


def section(d: dict[str, float], n: int = 48) -> tuple[FloatArr, FloatArr]:
    """Unit cross section (y, z): superformula, Fourier terms, then a hyperbolic deformation.

    An aircraft is left-right symmetric, so every term here must be too: the 3-lobe term is a sine,
    and the Moebius translation runs along the vertical axis only. A sideways translation would
    build a body on one side of the centreline while the lift model counted both.
    """
    ph = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    r = C.superformula(ph, 4.0, d["n1"], d["n1"], d["n1"]) * C.fourier_radius(ph, np.array([0.0, d["f2"], 0.0]), np.array([0.0, 0.0, d["f3"]]))
    y, z = r * np.cos(ph), r * np.sin(ph)
    if abs(d["wi"]) > 1e-6:
        s = max(float(np.hypot(y, z).max()), 1e-9) * 1.02
        y, z = C.hyperbolic_warp(y, z, complex(0.0, d["wi"]), s)
    z = z - 0.5 * (z.max() + z.min())
    return y / max(float(np.abs(y).max()), 1e-9), z / max(float(np.abs(z).max()), 1e-9)


@dataclass
class Strip:
    y: FloatArr  # strip centres on one side
    z: FloatArr
    c: FloatArr
    tc: FloatArr
    dy: FloatArr
    x_le: FloatArr
    twist: FloatArr
    is_body: NDArray[np.bool_]


@dataclass
class Vehicle:
    d: dict[str, float]
    x_st: FloatArr  # body stations
    a: FloatArr  # half width along the body
    b: FloatArr  # half height along the body
    zc: FloatArr
    sec_y: FloatArr
    sec_z: FloatArr
    strips: Strip
    S_ref: float
    span: float
    mac: float
    x_mac: float  # leading edge of the mean chord
    S_wet_body: float
    S_wet_wing: float
    S_wet_tail: float
    V_body: float
    V_wing: float
    floor_area: float
    fineness: float
    squareness: float
    closure_deg: float
    lifting_body: bool
    sweep_le: float
    sweep_mid: float
    tc_wing: float
    area_x: FloatArr = field(default_factory=lambda: np.zeros(0))
    area_law: FloatArr = field(default_factory=lambda: np.zeros(0))
    x_tail: float = 0.0
    S_h: float = 0.0
    S_v: float = 0.0
    mesh: wg.Mesh | None = None


def _trefftz(ny: FloatArr, nz: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr, FloatArr, FloatArr]:
    """Non-planar lifting line nodes -> control points, panel length, angle and influence matrix."""
    y, z = 0.5 * (ny[1:] + ny[:-1]), 0.5 * (nz[1:] + nz[:-1])
    ds = np.hypot(np.diff(ny), np.diff(nz))
    phi = np.arctan2(np.diff(nz), np.diff(ny))
    nrm = np.stack([-np.sin(phi), np.cos(phi)], 1)

    def wash(sy: FloatArr, sz: FloatArr) -> FloatArr:
        ry, rz = y[:, None] - sy[None, :], z[:, None] - sz[None, :]
        r2 = ry**2 + rz**2 + 1e-12
        return (-rz / r2 * nrm[:, 0:1] + ry / r2 * nrm[:, 1:2]) / (4 * np.pi)

    A = (wash(ny[1:], nz[1:]) - wash(ny[:-1], nz[:-1])) - (wash(-ny[1:], nz[1:]) - wash(-ny[:-1], nz[:-1]))
    return y, z, ds, phi, A


def build(d: dict[str, float], nx: int = 72, n_body: int = 7, n_wing: int = 16, n_winglet: int = 3) -> Vehicle:
    L, W, H = d["L"], d["W"], d["H"]
    xi = (0.5 * (1 - np.cos(np.linspace(0, np.pi, nx))))
    prof = lengthwise(xi, d["fn"], d["ft"], d["p"], d["tq"], d["tract"])
    a, b = W * prof, H * prof
    x = xi * L
    zc = d["upsweep"] * H * np.clip((xi - 0.6) / 0.4, 0, 1) ** 2
    sy, sz = section(d)
    # exact section metrics on the unit outline
    y2, z2 = np.r_[sy, sy[0]], np.r_[sz, sz[0]]
    area_u = 0.5 * abs(float(np.sum(y2[:-1] * z2[1:] - y2[1:] * z2[:-1])))
    # surface area from the lofted grid
    Yg = a[:, None] * sy[None, :]
    Zg = b[:, None] * sz[None, :] + zc[:, None]
    Xg = np.repeat(x[:, None], len(sy), 1)
    P = np.stack([Xg, Yg, Zg], -1)
    P = np.concatenate([P, P[:, :1]], 1)
    e1 = P[1:, :-1] - P[:-1, :-1]
    e2 = P[:-1, 1:] - P[:-1, :-1]
    e3 = P[1:, 1:] - P[1:, :-1]
    e4 = P[1:, 1:] - P[:-1, 1:]
    S_body = float(0.5 * (np.linalg.norm(np.cross(e1, e2), axis=-1).sum() + np.linalg.norm(np.cross(e3, e4), axis=-1).sum()))
    A_x = area_u * a * b
    V_body = float(np.trapezoid(A_x, x))
    # a section far from a circle needs the 30-40 % square-body correction
    per = float(np.hypot(np.diff(y2) * W, np.diff(z2) * H).sum())
    circ = 4 * np.pi * (area_u * W * H) / max(per, 1e-9) ** 2
    squareness = float(np.clip(d["n1"] / 2 - 1, 0, 1))
    D_eq = 2 * np.sqrt(max(float(A_x.max()), 1e-6) / np.pi)
    fin = L / D_eq
    # closure: the mean boat-tail angle in the vertical plane. The tip itself is excluded because
    # any closed outline ends in a steep point that the flow never follows.
    x0 = 1 - d["ft"]
    b0, b1 = float(np.interp(x0, xi, b)), float(np.interp(0.96, xi, b))
    closure = float(np.degrees(np.arctan(max(b0 - b1, 0.0) / max(L * (0.96 - x0), 1e-6))))
    # cabin floor: planform area with 2 m of headroom (inside the outline at each spanwise point)
    head = 1.0
    gy = np.linspace(0, 1, 15)
    y_half = sy[sz >= 0]
    z_half = sz[sz >= 0]
    o = np.argsort(y_half)
    prof_z = np.interp(gy, y_half[o], z_half[o])
    floor = 0.0
    dxs = np.gradient(x)
    for i in range(nx):
        if a[i] < 0.3:
            continue
        floor += dxs[i] * 2 * a[i] * float(np.mean(b[i] * prof_z >= head))
    lifting = bool(W / max(H, 1e-6) > 1.6)

    # wing
    S, AR, tp = d["S"], d["AR"], d["taper"]
    span_w = float(np.sqrt(S * AR))
    i_w = int(np.argmin(np.abs(xi - d["x_wing"])))
    root_y = float(a[i_w]) if lifting else 0.0
    half = root_y + span_w / 2
    c_r = 2 * S / (span_w * (1 + tp))
    tan_q = np.tan(np.radians(d["sweep"]))
    sw_le = float(np.degrees(np.arctan(tan_q + (1 - tp) * c_r / (2 * max(span_w / 2, 1e-6)) * 0.25 * 2)))
    sw_mid = float(np.degrees(np.arctan(tan_q - (1 - tp) * c_r / (2 * max(span_w / 2, 1e-6)) * 0.25 * 2)))
    th = np.linspace(0, np.pi / 2, n_wing + 1)
    wy = root_y + (span_w / 2) * np.sin(th)
    nodes_y, nodes_z = [], []
    cs, tcs, xl, tw, isb = [], [], [], [], []
    if lifting:
        by = np.linspace(0.0, root_y, n_body + 1)
        for k in range(n_body):
            yc = 0.5 * (by[k] + by[k + 1])
            inside = a > yc
            chord = float(np.trapezoid(inside.astype(float), x))
            if chord < 0.2:
                chord = 0.2
            frac = np.sqrt(np.clip(1 - (yc / np.maximum(a, 1e-6)) ** 2, 0, 1))
            thick = float(np.max(2 * b * frac * inside))
            cs.append(chord)
            tcs.append(min(thick / chord, 0.30))
            xl.append(float(x[inside][0]) if inside.any() else 0.0)
            tw.append(0.0)
            isb.append(True)
        nodes_y += list(by[:-1])
        nodes_z += [0.0] * n_body
    nodes_y += list(wy)
    nodes_z += [0.0] * len(wy)
    eta = (0.5 * (wy[1:] + wy[:-1]) - root_y) / max(span_w / 2, 1e-6)
    cw = c_r * (1 - (1 - tp) * eta)
    cs += list(cw)
    tcs += [d["tc"]] * len(cw)
    x0 = x[i_w] - 0.25 * c_r
    xl += list(x0 + 0.25 * c_r + (0.5 * (wy[1:] + wy[:-1]) - root_y) * tan_q - 0.25 * cw)
    tw += list(d["twist"] * eta)
    isb += [False] * len(cw)
    if d["winglet"] > 1e-3:
        hw = d["winglet"] * half
        zt = hw * np.sin(np.linspace(0, np.pi / 2, n_winglet + 1))[1:]
        nodes_y += [half] * n_winglet
        nodes_z += list(zt)
        c_t = c_r * tp
        cs += list(c_t * (1 - 0.6 * (np.arange(n_winglet) + 0.5) / n_winglet))
        tcs += [0.08] * n_winglet
        xl += [xl[-1]] * n_winglet
        tw += [0.0] * n_winglet
        isb += [False] * n_winglet
    ny_, nz_ = np.array(nodes_y), np.array(nodes_z)
    y, z, ds, phi, Am = _trefftz(ny_, nz_)
    c_a, tc_a = np.array(cs), np.array(tcs)
    is_wl = np.abs(phi) > 0.3
    planar = ~is_wl
    S_ref = float(2 * np.sum((c_a * ds)[planar]))
    mac = float(2 * np.sum((c_a**2 * ds)[planar]) / max(S_ref, 1e-6))
    xl_a = np.array(xl)
    x_mac = float(2 * np.sum((xl_a * c_a * ds)[planar]) / max(S_ref, 1e-6))
    st = Strip(y, z, c_a, tc_a, ds, xl_a, np.array(tw), np.array(isb))
    S_wing_exposed = float(2 * np.sum((c_a * ds)[~st.is_body]))
    S_wet_wing = 2 * S_wing_exposed * (1 + 0.2 * float(np.mean(tc_a[~st.is_body])))
    V_wing = float(2 * np.sum((0.68 * tc_a * c_a**2 * ds)[~st.is_body]))
    # tails, sized by volume coefficient about the mean chord
    l_t = max(0.92 * L - (x_mac + 0.25 * mac), 0.15 * L)
    b_ref = 2 * half
    S_h = d["Vh"] * S_ref * mac / l_t
    S_v = d["Vv"] * S_ref * b_ref / l_t
    S_wet_tail = 2.05 * (S_h + S_v)
    mesh = wg.Mesh(y, z, c_a, ds, phi, st.twist, planar, Am, S_ref)
    # total area law for wave drag: body plus wing volume spread along the chord
    grid = np.linspace(0, max(L, float(np.max(xl_a + c_a))), 96)
    law = np.interp(grid, x, A_x, left=0.0, right=0.0)
    for j in range(len(y)):
        if st.is_body[j] or is_wl[j]:
            continue
        u = (grid - xl_a[j]) / max(c_a[j], 1e-6)
        m = (u > 0) & (u < 1)
        thick = 5.2 * tc_a[j] * c_a[j] * np.sqrt(np.clip(u, 0, 1)) * (1 - u) * 0.92  # thickness distribution
        law = law + np.where(m, 2 * ds[j] * thick, 0.0)
    return Vehicle(
        d, x, a, b, zc, sy, sz, st, S_ref, 2 * half, mac, x_mac, S_body, S_wet_wing, S_wet_tail, V_body, V_wing, floor, fin,
        squareness + (1 - circ) * 0.5, closure, lifting, sw_le, sw_mid, d["tc"], grid, law, x_mac + 0.25 * mac + l_t, S_h, S_v, mesh,
    )  # fmt: skip


def surface(v: Vehicle, n_span: int = 26) -> tuple[FloatArr, NDArray[np.int64]]:
    """Whole-vehicle triangle mesh for the 3D view and STL export."""
    V: list[FloatArr] = []
    F: list[tuple[int, int, int]] = []

    def add(G: FloatArr, wrap: bool) -> None:
        o = sum(len(q) for q in V)
        ni, nj = G.shape[:2]
        V.append(G.reshape(-1, 3))
        for i in range(ni - 1):
            for j in range(nj if wrap else nj - 1):
                jn = (j + 1) % nj
                p, q, r, s = o + i * nj + j, o + (i + 1) * nj + j, o + (i + 1) * nj + jn, o + i * nj + jn
                F.extend([(p, q, r), (p, r, s)])

    x = v.x_st[::2]
    k = slice(None, None, 2)
    G = np.stack([np.repeat(x[:, None], len(v.sec_y), 1), v.a[k][:, None] * v.sec_y[None, :], v.b[k][:, None] * v.sec_z[None, :] + v.zc[k][:, None]], -1)
    add(G, True)
    t = np.linspace(0, 1, 15)
    thick = 5 * (0.2969 * np.sqrt(t) - 0.126 * t - 0.3516 * t**2 + 0.2843 * t**3 - 0.1036 * t**4)
    s = v.strips
    for sgn in (1.0, -1.0):
        idx = [j for j in range(len(s.y)) if not s.is_body[j]]
        rows = []
        for j in idx:
            xs = np.r_[t, t[::-1][1:]]
            ys = np.r_[thick, -thick[::-1][1:]] * s.tc[j]
            wl = abs(v.mesh.phi[j]) > 0.3 if v.mesh is not None else False
            rows.append(np.c_[s.x_le[j] + xs * s.c[j], np.full(len(xs), sgn * s.y[j]) + (ys * s.c[j] * sgn if wl else np.zeros(len(xs))), s.z[j] + (np.zeros(len(xs)) if wl else ys * s.c[j])])
        add(np.array(rows), False)
    return np.vstack(V), np.array(F, dtype=np.int64)


# Starting points. Each is only a region of the same space, not a separate model.
PRESETS = {
    "tube_and_wing": dict(L=38.0, W=1.98, H=1.98, fn=0.14, ft=0.34, p=2.2, tq=1.2, tract=0.0, n1=2.0, wi=0.0, f2=0.0, f3=0.0, upsweep=0.6, S=125.0, AR=9.4, taper=0.27, sweep=25.0, tc=0.125, twist=-3.0, winglet=0.05, x_wing=0.42, Vh=1.0, Vv=0.085),
    "blended_wing_body": dict(L=30.0, W=9.0, H=1.7, fn=0.40, ft=0.60, p=2.2, tq=1.4, tract=0.0, n1=2.0, wi=0.0, f2=0.0, f3=0.0, upsweep=0.0, S=120.0, AR=7.5, taper=0.25, sweep=30.0, tc=0.11, twist=-3.0, winglet=0.08, x_wing=0.55, Vh=0.0, Vv=0.04),
    "flying_wing": dict(L=22.0, W=12.0, H=1.35, fn=0.45, ft=0.55, p=2.0, tq=1.4, tract=0.0, n1=1.8, wi=0.0, f2=0.0, f3=0.0, upsweep=0.0, S=150.0, AR=8.0, taper=0.25, sweep=30.0, tc=0.11, twist=-4.0, winglet=0.09, x_wing=0.5, Vh=0.0, Vv=0.03),
    "freeform_lifting_body": dict(L=34.0, W=6.0, H=2.0, fn=0.30, ft=0.60, p=2.0, tq=1.4, tract=0.4, n1=2.6, wi=0.05, f2=0.05, f3=0.02, upsweep=0.3, S=130.0, AR=9.0, taper=0.3, sweep=28.0, tc=0.11, twist=-3.0, winglet=0.06, x_wing=0.55, Vh=0.3, Vv=0.05),
    "supersonic_jet": dict(L=50.0, W=1.35, H=1.35, fn=0.45, ft=0.50, p=1.6, tq=1.3, tract=0.0, n1=2.0, wi=0.0, f2=0.0, f3=0.0, upsweep=0.0, S=165.0, AR=2.2, taper=0.12, sweep=60.0, tc=0.03, twist=-2.0, winglet=0.0, x_wing=0.58, Vh=0.05, Vv=0.05),
}
