r"""Independent viscous analysis: Hess-Smith panels + integral boundary layers.

This module is the *judge*, not the optimiser's evaluator. It shares no code, data or training
with NeuralFoil, so a gain that appears in both is unlikely to be a surrogate artefact.

Laminar: Thwaites, :math:`\theta^2=0.45\nu U_e^{-6}\int U_e^5ds`, with the Cebeci-Bradshaw
closures for H(lambda) and l(lambda).

Transition: e^N envelope method of Drela and Giles (1987), with amplification from the
Falkner-Skan-based correlation. Laminar separation forces transition (short bubble assumption).

Turbulent: Head's entrainment method with the Ludwieg-Tillmann skin friction law.

Coupling: the boundary layer acts on the outer flow through surface transpiration
:math:`v_n=d(U_e\delta^*)/ds` (mass-defect sources), which leaves the geometry and the panel
matrix unchanged. Drag: Squire-Young at x/c = X_END, just ahead of the trailing edge, because the
inviscid velocity falls to zero at a closed trailing edge. Lift: circulation of the coupled
solution, with a Kirchhoff factor :math:`((1+\sqrt{f})/2)^2` once the upper surface separates.

Weakly coupled (a fixed number of passes), so it is reliable for attached flow and gives only
trends through stall. It does not model the drag of laminar separation bubbles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve

from aerosynapse.numerics.panel import _geometry, _influence

FloatArr = NDArray[np.float64]
X_END = 0.98


def _thwaites(lam: FloatArr) -> tuple[FloatArr, FloatArr]:
    lam = np.clip(lam, -0.0899, 0.25)
    pos = lam >= 0
    H = np.where(pos, 2.61 - 3.75 * lam + 5.24 * lam**2, 2.088 + 0.0731 / (0.14 + lam))
    L = np.where(pos, 0.22 + 1.57 * lam - 1.8 * lam**2, 0.22 + 1.402 * lam + 0.018 * lam / (0.107 + lam))
    return H, L


def _dn_dx(H: float, theta: float, re_theta: float) -> float:
    """Drela-Giles envelope amplification rate dN/ds."""
    Hk = min(max(H, 2.05), 6.0)
    g = 1.0 / (Hk - 1.0)
    log_re0 = (1.415 * g - 0.489) * np.tanh(20.0 * g - 12.9) + 3.295 * g + 0.44
    if np.log10(max(re_theta, 1e-9)) < log_re0:
        return 0.0
    dn = 0.01 * np.sqrt((2.4 * Hk - 3.7 + 2.5 * np.tanh(1.5 * Hk - 4.65)) ** 2 + 0.25)
    ell = (6.54 * Hk - 14.07) / Hk**2
    m = (0.058 * (Hk - 4.0) ** 2 / (Hk - 1.0) - 0.068) / ell
    return float(dn * 0.5 * (m + 1.0) * ell / theta)


def _h1_of_h(H: float) -> float:
    return 0.8234 * (H - 1.1) ** -1.287 + 3.3 if H <= 1.6 else 1.5501 * (H - 0.6778) ** -3.064 + 3.3


def _h_of_h1(H1: float) -> float:
    H1 = max(H1, 3.32)
    if H1 >= 5.3:
        return 1.1 + ((H1 - 3.3) / 0.8234) ** (-1 / 1.287)
    return 0.6778 + ((H1 - 3.3) / 1.5501) ** (-1 / 3.064)


@dataclass
class BLResult:
    x_tr: float
    x_sep: float
    theta_te: float
    H_te: float
    ue_te: float
    delta_star: FloatArr
    cf: FloatArr


def march(s: FloatArr, ue: FloatArr, x: FloatArr, Re: float, ncrit: float, xtr: float, h_sep: float = 2.4) -> BLResult:
    """March one surface from the stagnation point. Lengths are per chord, velocities per U_inf."""
    n = len(s)
    nu = 1.0 / Re
    ue = np.maximum(ue, 1e-4)
    due = np.gradient(ue, s)
    th = np.zeros(n)
    H = np.full(n, 2.6)
    cf = np.zeros(n)
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (ue[1:] ** 5 + ue[:-1] ** 5) * np.diff(s))])
    th_l = np.sqrt(0.45 * nu * np.maximum(integ, 1e-30) / ue**6)
    th_l[0] = th_l[1] if n > 1 else 0.0
    H_l, L_l = _thwaites(th_l**2 / nu * due)
    N = 0.0
    i_tr = n - 1
    x_tr = 1.0
    for i in range(1, n):
        th[i], H[i] = th_l[i], float(H_l[i])
        ret = ue[i] * th[i] / nu
        cf[i] = 2.0 * L_l[i] / max(ret, 1.0)
        N += _dn_dx(H[i], th[i], ret) * (s[i] - s[i - 1])
        lam = th_l[i] ** 2 / nu * due[i]
        if ncrit <= N or x[i] >= xtr or (lam < -0.09 and x[i] > 0.002):
            i_tr, x_tr = i, float(x[i])
            break
    th[0] = th[1] if n > 1 else 0.0
    x_sep = 1.0
    h = 1.4
    h1 = _h1_of_h(h)
    separated = False
    t_cur = th[i_tr]
    for i in range(i_tr, n - 1):
        ds = s[i + 1] - s[i]
        nsub = max(1, int(np.ceil(ds / 0.004)))
        d = ds / nsub
        cfi = 0.0
        for _ in range(nsub):
            ret = max(ue[i] * t_cur / nu, 50.0)
            cfi = 0.0 if separated else 0.246 * 10 ** (-0.678 * h) * ret**-0.268
            t_new = max(t_cur + d * (cfi / 2 - (h + 2) * t_cur / ue[i] * due[i]), 1e-7)
            if not separated:
                f_ent = 0.0306 * max(h1 - 3.0, 1e-3) ** -0.6169
                h1 = (ue[i] * t_cur * h1 + d * ue[i] * f_ent) / (ue[i] * t_new)
                h = _h_of_h1(h1)
                if h >= h_sep:
                    separated, x_sep, h = True, float(x[i]), h_sep
            t_cur = t_new
        th[i + 1], H[i + 1] = t_cur, h
        cf[i + 1] = cfi
    if i_tr >= n - 1:
        th[-1] = max(th[-1], th_l[-1])
    return BLResult(x_tr, x_sep, float(th[-1]), float(H[-1]), float(ue[-1]), th * H, cf)


class _Panels:
    """Hess-Smith system with a transpiration right-hand side; factorised once per geometry."""

    def __init__(self, X: FloatArr, Y: FloatArr) -> None:
        self.xc, self.yc, self.th, self.L = _geometry(X, Y)
        n = len(self.xc)
        s, c, lr, beta = _influence(X, Y, self.xc, self.yc, self.th)
        An = (s * lr + c * beta) / (2 * np.pi)
        Bn = (c * lr - s * beta) / (2 * np.pi)
        self.At = (s * beta - c * lr) / (2 * np.pi)
        self.Bt = (s * lr + c * beta) / (2 * np.pi)
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = An
        M[:n, n] = Bn.sum(1)
        M[n, :n] = self.At[0] + self.At[-1]
        M[n, n] = self.Bt[0].sum() + self.Bt[-1].sum()
        self.lu = lu_factor(M)
        self.n = n

    def solve(self, alpha_deg: float, vn: FloatArr | None = None) -> tuple[FloatArr, float, float]:
        """Tangential velocity, c_l and c_m,c/4 with outward transpiration ``vn``."""
        a, th, n = np.deg2rad(alpha_deg), self.th, self.n
        rhs = np.zeros(n + 1)
        rhs[:n] = -np.sin(a - th) + (0.0 if vn is None else vn)
        rhs[n] = -np.cos(th[0] - a) - np.cos(th[-1] - a)
        sol = lu_solve(self.lu, rhs)
        q, gam = sol[:n], float(sol[n])
        vt = self.At @ q + gam * self.Bt.sum(1) + np.cos(th - a)
        cp = 1 - vt**2
        nx, ny = -np.sin(th), np.cos(th)
        cm = float(np.sum(cp * self.L * ((self.xc - 0.25) * ny - self.yc * nx)))
        return vt, 2 * gam * float(self.L.sum()), cm


@dataclass
class VIIResult:
    alpha: float
    cl: float
    cd: float
    cd_friction: float
    cm: float
    x_tr_upper: float
    x_tr_lower: float
    x_sep_upper: float
    converged: bool


def _smooth(v: FloatArr, n: int = 2) -> FloatArr:
    for _ in range(n):
        v = np.r_[v[0], 0.25 * v[:-2] + 0.5 * v[1:-1] + 0.25 * v[2:], v[-1]]
    return v


def analyse(
    X: FloatArr,
    Y: FloatArr,
    alpha: float,
    Re: float,
    ncrit: float = 9.0,
    xtr: float = 1.0,
    passes: int = 12,
    panels: _Panels | None = None,
) -> VIIResult:
    """Viscous analysis of a contour in panel order (TE, lower, LE, upper, TE), chord 1."""
    P = panels or _Panels(X, Y)
    vn = np.zeros(P.n)
    d_prev: FloatArr | None = None
    ok = False
    for _ in range(passes):
        vt, cl, cm = P.solve(alpha, vn)
        flips = np.where(np.sign(vt[:-1]) != np.sign(vt[1:]))[0]
        k = int(flips[np.argmin(P.xc[flips])])  # stagnation: the sign change nearest the nose
        w = abs(vt[k]) / (abs(vt[k]) + abs(vt[k + 1]))
        xs = P.xc[k] + w * (P.xc[k + 1] - P.xc[k])
        ys = P.yc[k] + w * (P.yc[k + 1] - P.yc[k])

        def side(idx: NDArray[np.int64], xs: float = xs, ys: float = ys, vt: FloatArr = vt) -> tuple[NDArray[np.int64], FloatArr, FloatArr, FloatArr]:
            idx = idx[P.xc[idx] <= X_END]
            px, py = np.r_[xs, P.xc[idx]], np.r_[ys, P.yc[idx]]
            return idx, np.r_[0.0, np.cumsum(np.hypot(np.diff(px), np.diff(py)))], np.r_[0.0, np.abs(vt[idx])], px

        iu, su, uu, xu = side(np.arange(k + 1, P.n))
        il, sl, ul, xl = side(np.arange(k, -1, -1))
        bu = march(su, uu, xu, Re, ncrit, xtr)
        bl = march(sl, ul, xl, Re, ncrit, xtr)
        d = np.zeros(P.n)
        d[iu], d[il] = bu.delta_star[1:], bl.delta_star[1:]
        ok = d_prev is not None and float(np.abs(d - d_prev).max()) < 1e-4
        d_prev = d
        new = np.zeros(P.n)
        for idx, s_, u_, b in ((iu, su, uu, bu), (il, sl, ul, bl)):
            m = _smooth(np.minimum(u_ * b.delta_star, 0.08), 3)
            new[idx] = np.gradient(m, s_)[1:]
        vn = 0.7 * vn + 0.3 * np.clip(new, -0.2, 0.2)
        if ok:
            break
    cd = 2.0 * bu.theta_te * bu.ue_te ** (0.5 * (bu.H_te + 5.0)) + 2.0 * bl.theta_te * bl.ue_te ** (0.5 * (bl.H_te + 5.0))
    cdf = float(np.trapezoid(bu.cf, xu) + np.trapezoid(bl.cf, xl))
    f = float(np.clip(bu.x_sep, 0.0, 1.0))
    kir = ((1 + np.sqrt(f)) / 2) ** 2
    return VIIResult(alpha, cl * kir, cd, cdf, cm, bu.x_tr, bl.x_tr, bu.x_sep, ok)


def polar(X: FloatArr, Y: FloatArr, alphas: FloatArr, Re: float, ncrit: float = 9.0, xtr: float = 1.0) -> dict[str, FloatArr]:
    P = _Panels(X, Y)
    rs = [analyse(X, Y, float(a), Re, ncrit, xtr, panels=P) for a in alphas]
    return {
        "alpha": np.asarray(alphas, float),
        "CL": np.array([r.cl for r in rs]),
        "CD": np.array([r.cd for r in rs]),
        "CM": np.array([r.cm for r in rs]),
        "xtr_u": np.array([r.x_tr_upper for r in rs]),
        "xsep_u": np.array([r.x_sep_upper for r in rs]),
        "converged": np.array([r.converged for r in rs]),
    }
