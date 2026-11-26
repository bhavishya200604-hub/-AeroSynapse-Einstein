r"""Adding shapes: multi-element high-lift systems (slat, flap, or both).

Adding an element changes the topology of the section from one body to two or three, which the
18-number parameterisation cannot express and which single-element viscous models cannot analyse.
The analysis here is a multi-body Hess-Smith panel method: one source strength per panel, one
vortex strength and one Kutta condition per element.

Maximum lift uses the pressure-difference idea of Valarezo and Chin: an element stalls when the
difference between its suction peak and its trailing-edge pressure reaches a critical value. That
value depends on Reynolds and Mach number and is not assumed here. It is calibrated so that the
single-element section reaches its own c_l,max, which is supplied by a viscous analysis.

Limits: inviscid flow plus a stall rule. Confluent boundary layers and slot flow physics are not
modelled, so gap and overlap optima are trends and the absolute c_l,max is an estimate. Compare
the result with typical values for each device before using it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve

from aerosynapse.shapeopt.cmaes import cmaes

FloatArr = NDArray[np.float64]


def solve_multi(elements: list[FloatArr], alphas_deg: FloatArr) -> dict[str, FloatArr | list[FloatArr]]:
    """Solve all elements at every incidence. Elements are in panel order (TE, lower, LE, upper, TE).

    Potential flow is linear, so two right-hand sides (unit x and y free streams) give every
    incidence at no extra cost.
    """
    x0, y0, x1, y1 = [], [], [], []
    starts = [0]
    for e in elements:
        x0.append(e[:-1, 0])
        y0.append(e[:-1, 1])
        x1.append(e[1:, 0])
        y1.append(e[1:, 1])
        starts.append(starts[-1] + len(e) - 1)
    X0, Y0, X1, Y1 = map(np.concatenate, (x0, y0, x1, y1))
    n, K = len(X0), len(elements)
    xc, yc = 0.5 * (X0 + X1), 0.5 * (Y0 + Y1)
    th = np.arctan2(Y1 - Y0, X1 - X0)
    L = np.hypot(X1 - X0, Y1 - Y0)
    dx0, dy0 = X0[None] - xc[:, None], Y0[None] - yc[:, None]
    dx1, dy1 = X1[None] - xc[:, None], Y1[None] - yc[:, None]
    r0, r1 = np.hypot(dx0, dy0), np.hypot(dx1, dy1)
    beta = np.arctan2(dx0 * dy1 - dy0 * dx1, dx0 * dx1 + dy0 * dy1)
    np.fill_diagonal(beta, np.pi)
    lr = np.log(r1 / r0)
    np.fill_diagonal(lr, 0.0)
    d = th[:, None] - th[None, :]
    s, c = np.sin(d), np.cos(d)
    An, Bn = (s * lr + c * beta) / (2 * np.pi), (c * lr - s * beta) / (2 * np.pi)
    At, Bt = (s * beta - c * lr) / (2 * np.pi), (s * lr + c * beta) / (2 * np.pi)
    Bn_e = np.stack([Bn[:, starts[k] : starts[k + 1]].sum(1) for k in range(K)], 1)
    Bt_e = np.stack([Bt[:, starts[k] : starts[k + 1]].sum(1) for k in range(K)], 1)
    M = np.zeros((n + K, n + K))
    M[:n, :n] = An
    M[:n, n:] = Bn_e
    rhs = np.zeros((n + K, 2))
    nrm_x, nrm_y = -np.sin(th), np.cos(th)
    rhs[:n, 0], rhs[:n, 1] = -nrm_x, -nrm_y
    tx, ty = np.cos(th), np.sin(th)
    for k in range(K):
        a, b = starts[k], starts[k + 1] - 1
        M[n + k, :n] = At[a] + At[b]
        M[n + k, n:] = Bt_e[a] + Bt_e[b]
        rhs[n + k, 0] = -(tx[a] + tx[b])
        rhs[n + k, 1] = -(ty[a] + ty[b])
    S = lu_solve(lu_factor(M), rhs)
    Vt = At @ S[:n] + Bt_e @ S[n:] + np.stack([tx, ty], 1)  # (n, 2)
    ang = np.deg2rad(np.atleast_1d(alphas_deg))
    bas = np.stack([np.cos(ang), np.sin(ang)])  # (2, A)
    vt = Vt @ bas
    gam = S[n:] @ bas  # (K, A)
    per = np.array([L[starts[k] : starts[k + 1]].sum() for k in range(K)])
    cl = 2 * (gam * per[:, None]).sum(0)
    cp = 1 - vt**2
    cp_el = [cp[starts[k] : starts[k + 1]] for k in range(K)]
    return {"cl": cl, "cp": cp_el, "xc": [xc[starts[k] : starts[k + 1]] for k in range(K)], "yc": [yc[starts[k] : starts[k + 1]] for k in range(K)], "gamma": gam}


def delta_cp(cp_el: list[FloatArr], xc_el: list[FloatArr]) -> FloatArr:
    """Peak-to-trailing-edge pressure difference per element: (K, A).

    Only the upper surface counts (panels after the nose). The cuts that form a cove or a slat
    rear leave sharp corners on the lower side, and inviscid flow gives them velocity spikes with
    no bearing on stall.
    """
    out = []
    for c, x in zip(cp_el, xc_el, strict=False):
        le = int(np.argmin(x))
        out.append(0.5 * (c[0] + c[-1]) - c[le:].min(0))
    return np.stack(out)


def to_panel(upper: FloatArr, lower: FloatArr) -> FloatArr:
    """Upper and lower arrays, each running LE to TE -> closed panel-order contour."""
    return np.r_[lower[::-1], upper[1:]]


def _nose(p_up: FloatArr, p_lo: FloatArr, length: float, n: int = 13) -> FloatArr:
    """Rounded nose joining two points, bulging towards -x. Runs from upper point to lower point."""
    mid, half = 0.5 * (p_up + p_lo), 0.5 * (p_up - p_lo)
    t = np.linspace(0.0, np.pi, n)
    return mid[None] + np.outer(np.cos(t), half) + np.outer(np.sin(t), [-length, 0.0])


@dataclass
class Layout:
    """Deployment. Lengths are in chords, angles in degrees (positive deflects down / droops)."""

    has_slat: bool
    has_flap: bool
    c_flap: float = 0.30
    d_flap: float = 30.0
    gap_flap: float = 0.02
    ovl_flap: float = 0.01
    c_slat: float = 0.15
    d_slat: float = 25.0
    gap_slat: float = 0.02
    ovl_slat: float = 0.0


def _rot(p: FloatArr, deg: float, about: FloatArr) -> FloatArr:
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])  # clockwise: trailing edge down
    return (p - about) @ R.T + about


def build(xu: FloatArr, yu: FloatArr, yl: FloatArr, lay: Layout) -> list[FloatArr]:
    """Cut one section (surfaces on stations xu, LE to TE) into elements and deploy them."""
    up, lo = np.c_[xu, yu], np.c_[xu, yl]
    x_f = 1.0 - lay.c_flap if lay.has_flap else 1.0
    x_s = lay.c_slat if lay.has_slat else 0.0
    els: list[FloatArr] = []
    mu = (xu >= x_s) & (xu <= x_f)
    main_u, main_l = up[mu], lo[mu]
    if lay.has_slat:
        nose = _nose(main_u[0], main_l[0], 0.035)
        main_u = np.r_[nose[len(nose) // 2 :: -1][:-1], main_u]
        main_l = np.r_[nose[len(nose) // 2 :][:-1], main_l]
    if lay.has_flap:  # cove: the lower surface rises to the upper lip, giving a sharp trailing edge
        lip = main_u[-1]
        main_l = np.r_[main_l, (main_l[-1] + 0.5 * (lip - main_l[-1]))[None], lip[None]]
    main = to_panel(main_u, main_l) if lay.has_slat or lay.has_flap else to_panel(up, lo)
    els.append(main)
    if lay.has_flap:
        mf = xu >= x_f
        fu, fl = up[mf], lo[mf]
        nose = _nose(fu[0], fl[0], 0.03)
        k = len(nose) // 2
        fu2 = np.r_[nose[k::-1], fu[1:]]
        fl2 = np.r_[nose[k:], fl[1:]]
        flap = to_panel(fu2, fl2)
        lip = main_u[-1]
        le = flap[np.argmin(flap[:, 0])]
        flap = _rot(flap, lay.d_flap, le)
        # place the flap so that its nose sits below and behind the lip by the gap and overlap
        tip = flap[np.argmin(flap[:, 0])]
        flap = flap + (lip + np.array([-lay.ovl_flap, -lay.gap_flap]) - tip)
        els.append(flap)
    if lay.has_slat:
        ms = xu <= x_s
        su, sl = up[ms], lo[ms]
        # concave rear: straight closure from the lower cut to the upper trailing point
        te = su[-1]
        sl2 = np.r_[sl, (0.5 * (sl[-1] + te))[None], te[None]]
        slat = to_panel(su, sl2)
        pivot = te
        slat = _rot(slat, -lay.d_slat, pivot)  # droop: counter-clockwise about the slat trailing edge
        main_nose = main[np.argmin(main[:, 0])]
        tgt = main_nose + np.array([-lay.gap_slat + lay.ovl_slat, lay.gap_slat])
        slat = slat + (tgt - slat[0])
        els.insert(0, slat)
    return els


def min_clearance(els: list[FloatArr]) -> float:
    """Smallest distance between different elements; negative if an element pierces another."""
    from matplotlib.path import Path as MplPath

    best = np.inf
    for i in range(len(els)):
        for j in range(i + 1, len(els)):
            d = np.hypot(*(els[i][:, None, :] - els[j][None, :, :]).transpose(2, 0, 1)).min()
            if MplPath(els[i]).contains_points(els[j]).any() or MplPath(els[j]).contains_points(els[i]).any():
                d = -abs(d) - 0.01
            best = min(best, float(d))
    return float(best)


def clmax(els: list[FloatArr], dcp_crit: float, alphas: FloatArr | None = None) -> tuple[float, float, int]:
    """Lift, incidence and element index at first stall under the pressure-difference rule."""
    a = np.arange(-10.0, 30.1, 0.5) if alphas is None else alphas
    r = solve_multi(els, a)
    d = delta_cp(r["cp"], r["xc"])  # type: ignore[arg-type]
    # scan upward from the zero-load region: stall is the first exceedance at positive lift
    bad = (~np.all(d < dcp_crit, axis=0)) & (np.asarray(r["cl"]) > 0.3)
    i = int(np.where(bad)[0][0]) - 1 if bad.any() else len(a) - 1
    if i < 0 or np.any(d[:, i] >= dcp_crit):  # past the limit before it carries useful lift: stalled, not good
        return 0.0, float(a[max(i, 0)]), int(np.argmax(d[:, max(i, 0)]))
    return float(r["cl"][i]), float(a[i]), int(np.argmax(d[:, min(i + 1, len(a) - 1)]))


def calibrate(xu: FloatArr, yu: FloatArr, yl: FloatArr, cl_max_section: float, visc_factor: float = 0.92) -> float:
    """Critical pressure difference that makes the single element stall at its known c_l,max.

    The panel lift overstates viscous lift, so ``visc_factor`` relates the two (the independent
    solver reads about 7 % high against NASA data before stall).
    """
    els = [to_panel(np.c_[xu, yu], np.c_[xu, yl])]
    a = np.arange(0.0, 26.0, 0.25)
    r = solve_multi(els, a)
    cl = np.asarray(r["cl"]) * visc_factor
    i = int(np.argmin(np.abs(cl - cl_max_section)))
    return float(delta_cp(r["cp"], r["xc"])[0, i])  # type: ignore[arg-type]


BOUNDS = {  # name: (low, high)
    "c_flap": (0.20, 0.38),
    "d_flap": (10.0, 45.0),
    "gap_flap": (0.012, 0.05),
    "ovl_flap": (-0.02, 0.05),
    "c_slat": (0.10, 0.20),
    "d_slat": (10.0, 40.0),
    "gap_slat": (0.012, 0.05),
    "ovl_slat": (-0.02, 0.03),
}


def optimise_layout(
    xu: FloatArr, yu: FloatArr, yl: FloatArr, dcp_crit: float, has_slat: bool, has_flap: bool, seed: int = 0, budget: int = 500, visc_factor: float = 0.92
) -> tuple[Layout, float, float]:
    names = [k for k in BOUNDS if (has_flap and "flap" in k) or (has_slat and "slat" in k)]
    lo = np.array([BOUNDS[k][0] for k in names])
    hi = np.array([BOUNDS[k][1] for k in names])

    def lay_of(z: FloatArr) -> Layout:
        return Layout(has_slat, has_flap, **dict(zip(names, lo + z * (hi - lo), strict=False)))

    def f(Z: FloatArr) -> FloatArr:
        out = np.empty(len(Z))
        for i, z in enumerate(Z):
            els = build(xu, yu, yl, lay_of(z))
            clr = min_clearance(els)
            if clr < 0.008:
                out[i] = 5.0 + 100 * (0.008 - clr)
                continue
            out[i] = -clmax(els, dcp_crit)[0] * visc_factor
        return out

    r = cmaes(f, np.full(len(names), 0.5), sigma=0.2, seed=seed, budget=budget, pop=10, restarts=1, stall_gens=25)
    lay = lay_of(r.x)
    cl, a, _ = clmax(build(xu, yu, yl, lay), dcp_crit)
    return lay, cl * visc_factor, a
