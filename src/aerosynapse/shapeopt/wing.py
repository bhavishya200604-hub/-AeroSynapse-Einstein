r"""3D wing: span, area, taper, twist and winglet, with structure and aeroelastic checks.

Aerodynamics: nonlinear lifting line on a non-planar curve. The wing is a chain of horseshoe
vortices in the Trefftz plane, so the winglet is handled by the same equations and the induced
drag is exact for the computed load (Munk). Section lift and drag come from a viscous polar,
per flight state. Bordered Newton solves circulation and angle of attack for a required C_L.

Structure: sized wing box. Bending caps carry the ultimate bending moment, so a longer or thinner
wing costs real mass, and the mass feeds back into the required lift. This stops the optimiser
from simply extending the span without limit.

Limits: unswept, low subsonic, one section along the span, no fuselage. The structural mass is
scaled so that the baseline wing is ``wing_mass_fraction`` of take-off mass. It is a comparative
model, not a stress analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from aerosynapse.shapeopt.dynamics import Section2DOF, flutter_k, gust_load_factor

FloatArr = NDArray[np.float64]
G0 = 9.81
NU_SL = 1.46e-5
A_SOUND = 340.0


@dataclass
class Vehicle:
    """Generic light-aircraft class of vehicle (values are round numbers, not a real type)."""

    non_wing_mass: float = 950.0
    v_cruise: float = 60.0
    v_dive: float = 84.0
    n_limit: float = 3.8
    safety: float = 1.5
    wing_mass_fraction: float = 0.14
    k_struct: float = 1.0  # set by calibrate()
    sigma_allow: float = 3.0e8
    e_mod: float = 7.0e10
    g_mod: float = 2.7e10
    rho_mat: float = 2800.0
    secondary_kg_m2: float = 4.0
    parasite_area: float = 0.40  # non-wing drag area C_D S, m^2


@dataclass
class Planform:
    area: float
    aspect: float
    taper: float
    twist_tip: float  # deg, negative is washout
    twist_exp: float
    winglet: float  # height as a fraction of semi-span; 0 means no winglet

    def as_array(self) -> FloatArr:
        return np.array([self.area, self.aspect, self.taper, self.twist_tip, self.twist_exp, self.winglet])

    @property
    def span(self) -> float:
        return float(np.sqrt(self.area * self.aspect))

    @property
    def c_root(self) -> float:
        return 2 * self.area / (self.span * (1 + self.taper))

    @property
    def mac(self) -> float:
        t = self.taper
        return 2 / 3 * self.c_root * (1 + t + t * t) / (1 + t)


@dataclass
class Mesh:
    y: FloatArr  # control points
    z: FloatArr
    c: FloatArr  # chord
    ds: FloatArr  # panel length
    phi: FloatArr  # panel dihedral angle
    twist: FloatArr  # deg
    is_wing: NDArray[np.bool_]
    A: FloatArr  # normal wash per unit circulation
    S: float


def build_mesh(p: Planform, n: int = 20, n_w: int = 4) -> Mesh:
    half = p.span / 2
    e = half * np.sin(np.linspace(0, np.pi / 2, n + 1))
    hw = p.winglet * half
    if hw > 1e-3 * half:
        zt = hw * np.sin(np.linspace(0, np.pi / 2, n_w + 1))[1:]
        ny, nz = np.r_[e, np.full(n_w, half)], np.r_[np.zeros(n + 1), zt]
    else:
        ny, nz = e, np.zeros(n + 1)
    y, z = 0.5 * (ny[1:] + ny[:-1]), 0.5 * (nz[1:] + nz[:-1])
    dy, dz = np.diff(ny), np.diff(nz)
    ds = np.hypot(dy, dz)
    phi = np.arctan2(dz, dy)
    is_wing = dz < 1e-9
    eta = np.minimum(y / half, 1.0)
    tip_c = p.c_root * p.taper
    c = np.where(is_wing, p.c_root * (1 - (1 - p.taper) * eta), tip_c * (1 - 0.6 * (z / max(hw, 1e-9))))
    twist = np.where(is_wing, p.twist_tip * eta**p.twist_exp, 0.0)
    # Trefftz-plane influence with the mirror half: a trailing filament leaves each panel edge
    nrm = np.stack([-np.sin(phi), np.cos(phi)], 1)

    def wash(sy: FloatArr, sz: FloatArr) -> FloatArr:
        """Normal velocity at control points from unit vortices at (sy, sz), semi-infinite."""
        ry = y[:, None] - sy[None, :]
        rz = z[:, None] - sz[None, :]
        r2 = ry**2 + rz**2 + 1e-12
        return (-rz / r2 * nrm[:, 0:1] + ry / r2 * nrm[:, 1:2]) / (4 * np.pi)

    # panel j: filament +1 at its outer node, -1 at its inner node (sign fixed so that lift gives downwash)
    wo, wi = wash(ny[1:], nz[1:]), wash(ny[:-1], nz[:-1])
    wmo, wmi = wash(-ny[1:], nz[1:]), wash(-ny[:-1], nz[:-1])
    A = (wo - wi) - (wmo - wmi)
    return Mesh(y, z, c, ds, phi, twist, is_wing, A, p.area)


@dataclass
class LLTResult:
    alpha: FloatArr
    CD: FloatArr
    CDi: FloatArr
    CDp: FloatArr
    cl: FloatArr  # (F, N)
    gamma: FloatArr
    converged: bool


def _table(alpha_grid: FloatArr, T: FloatArr, a: FloatArr) -> tuple[FloatArr, FloatArr]:
    """Row-wise linear interpolation on a uniform grid. T (F, A), a (F, N) -> value and slope."""
    d = alpha_grid[1] - alpha_grid[0]
    t = np.clip((a - alpha_grid[0]) / d, 0, len(alpha_grid) - 1 - 1e-9)
    i = t.astype(int)
    w = t - i
    r = np.arange(T.shape[0])[:, None]
    v0, v1 = T[r, i], T[r, i + 1]
    return v0 + w * (v1 - v0), (v1 - v0) / d


def solve_llt(m: Mesh, CL: FloatArr, ag: FloatArr, cl_tab: FloatArr, cd_tab: FloatArr, iters: int = 14) -> LLTResult:
    """Solve F flight states at once. Tables are (F, A), capped after c_l,max by the caller.

    Circulation is per unit speed: G = Gamma / V. Unknowns: G (N) and the wing angle alpha.
    """
    F, N = len(CL), len(m.y)
    Wn = m.is_wing
    cphi = np.cos(m.phi)
    lift_w = 2 * 2 * m.ds * cphi / m.S  # C_L = sum(lift_w * G)
    # start from the linear solution: a poor guess overshoots the table on small tip panels
    mid = len(ag) // 2
    slope = float((cl_tab[:, mid + 1] - cl_tab[:, mid - 1]).mean() / (ag[mid + 1] - ag[mid - 1]))
    a_l0 = float((ag[mid] - cl_tab[:, mid] / slope).mean())
    _, g1, a1, _ = linear_load(m, 1.0, a_l0, slope)
    _, g0, a0, _ = linear_load(m, 0.0, a_l0, slope)
    G = g0[None, :] + CL[:, None] * (g1 - g0)[None, :]
    a = a0 + CL * (a1 - a0)
    ok = False
    for _ in range(iters):
        w = G @ m.A.T  # normal wash per unit V (positive = upwash on the panel normal)
        a_eff = (a[:, None] + m.twist[None, :]) * cphi[None, :] + np.rad2deg(w)
        cl_w, sl_w = _table(ag, cl_tab, np.where(Wn[None, :], a_eff, 0.0))
        cl = np.where(Wn[None, :], cl_w, 0.1 * a_eff)
        sl = np.where(Wn[None, :], np.maximum(sl_w, 0.012), 0.1)
        R = G - 0.5 * m.c * cl
        Rl = G @ lift_w - CL
        res = max(float(np.abs(R).max()), float(np.abs(Rl).max()))
        if res < 1e-6:
            ok = True
            break
        J = np.zeros((F, N + 1, N + 1))
        J[:, :N, :N] = np.eye(N)[None] - (0.5 * m.c * sl * np.rad2deg(1.0))[:, :, None] * m.A[None]
        J[:, :N, N] = -0.5 * m.c * sl * cphi[None, :]
        J[:, N, :N] = lift_w
        dx = np.linalg.solve(J, -np.c_[R, Rl][..., None])[..., 0]
        G = G + dx[:, :N]
        a = np.clip(a + np.clip(dx[:, N], -4.0, 4.0), -10.0, 30.0)
    w = G @ m.A.T
    a_eff = (a[:, None] + m.twist[None, :]) * cphi[None, :] + np.rad2deg(w)
    cd_w, _ = _table(ag, cd_tab, a_eff)
    cd = np.where(Wn[None, :], cd_w, 0.0085)
    cl_w, _ = _table(ag, cl_tab, np.where(Wn[None, :], a_eff, 0.0))
    cl = np.where(Wn[None, :], cl_w, 0.1 * a_eff)
    CDp = 2 * (cd * m.c * m.ds).sum(1) / m.S
    CDi = -2 * 2 * (G * w * m.ds).sum(1) / m.S
    return LLTResult(a, CDp + CDi, CDi, CDp, cl, G, ok)


def linear_load(m: Mesh, CL: float, a_l0: float, cl_a: float) -> tuple[FloatArr, FloatArr, float, FloatArr]:
    """Linear lifting line at one C_L: returns local c_l, G, alpha and the 'additional' c_l per C_L."""
    N = len(m.y)
    cphi = np.cos(m.phi)
    lift_w = 2 * 2 * m.ds * cphi / m.S
    K = np.eye(N) - (0.5 * m.c * cl_a * np.rad2deg(1.0))[:, None] * m.A
    # G = 0.5 c cl_a ((a + tw) cphi - a_l0 + deg(wash))
    base = 0.5 * m.c * cl_a * (m.twist * cphi - np.where(m.is_wing, a_l0, 0.0))
    add = 0.5 * m.c * cl_a * cphi
    Gb, Ga = np.linalg.solve(K, base), np.linalg.solve(K, add)
    a = (CL - lift_w @ Gb) / (lift_w @ Ga)
    G = Gb + a * Ga
    cl = 2 * G / m.c
    return cl, G, float(a), 2 * Ga / m.c / (lift_w @ Ga)


def wing_clmax(m: Mesh, a_l0: float, cl_a: float, cl_max_sec: FloatArr | float) -> tuple[float, float]:
    """Critical-section C_L,max and the span station that stalls first."""
    c1, *_ = linear_load(m, 0.3, a_l0, cl_a)
    c2, *_ = linear_load(m, 0.9, a_l0, cl_a)
    b = (c2 - c1) / 0.6
    a = c1 - 0.3 * b
    cm = np.broadcast_to(np.asarray(cl_max_sec, float), a.shape)
    cand = np.where(m.is_wing, (cm - a) / np.maximum(b, 1e-6), np.inf)
    i = int(np.argmin(cand))
    return float(cand[i]), float(m.y[i] / m.y[m.is_wing].max())


@dataclass
class Structure:
    wing_mass: float
    takeoff_mass: float
    root_bending: float
    n_gust: float
    v_flutter: float
    v_divergence: float
    k_flutter: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)


def size_structure(
    p: Planform, m: Mesh, t_c: float, a_l0: float, cl_a: float, x_ac_margin: float, v: Vehicle, m_wing_guess: float = 150.0
) -> Structure:
    """Fixed-point sizing of the wing box, then gust, flutter and divergence."""
    cl_a_rad = cl_a * 180 / np.pi
    wing = m.is_wing
    y, c, ds = m.y, m.c, m.ds
    half = p.span / 2
    m_w = m_wing_guess
    n_g, M = v.n_limit, np.zeros_like(y)
    A_cap = np.zeros_like(y)
    for _ in range(4):
        W = (v.non_wing_mass + m_w) * G0
        cl_lin, G, *_ = linear_load(m, 0.5, a_l0, cl_a)
        n_g = gust_load_factor(1.0, v.v_cruise, cl_a_rad * p.aspect / (p.aspect + 2), W / p.area, p.mac)
        n_des = max(v.n_limit, n_g) * v.safety
        l = 2 * G * ds  # relative lift per panel on one half
        l = l / (2 * l.sum()) * (W * n_des)  # scale so that both halves carry n_des * W
        arm = np.where(wing, y, half)
        # bending moment at each wing station from everything outboard
        M = np.array([np.sum(l * np.maximum(arm - yi, 0.0)) for yi in y])
        h = 0.85 * t_c * c
        A_cap = np.where(wing, M / (v.sigma_allow * h), 0.0)
        cap = 2 * 2 * v.rho_mat * float(np.sum(A_cap * ds))  # two caps, two halves
        web = 2 * 2 * v.rho_mat * float(np.sum(np.where(wing, 0.0015 * h, 0.0) * ds))
        sec = v.secondary_kg_m2 * p.area
        wl = 2 * 8.0 * float(np.sum(np.where(wing, 0.0, c * ds)))  # winglets: 8 kg/m^2 plus root fitting
        m_w = v.k_struct * (cap + web) + sec + 3.0 * wl
    # flutter at the 70 % station
    i = int(np.argmin(np.abs(np.where(wing, y, 1e9) - 0.7 * half)))
    j = int(np.argmin(np.abs(np.where(wing, y, 1e9) - 0.35 * half)))
    ci, cj = float(c[i]), float(c[j])
    b = ci / 2
    h_j = 0.85 * t_c * cj
    EI = 2 * float(A_cap[j]) * v.e_mod * (h_j / 2) ** 2
    Ab = 0.5 * cj * h_j
    GJ = 4 * Ab**2 * v.g_mod * 0.0015 / (2 * (0.5 * cj + h_j))
    m_len = m_w / p.span
    I_a = m_len * 0.25 * b**2
    w_h = 1.875**2 * np.sqrt(EI / (m_len * half**4))
    w_a = (np.pi / 2) * np.sqrt(GJ / (I_a * half**2))
    rho = 1.225
    a_ea = -0.2
    sec2 = Section2DOF(mu=m_len / (np.pi * rho * b**2), a=a_ea, x_alpha=0.1, r2=0.25, sigma=float(w_h / w_a))
    u, kf = flutter_k(sec2)
    v_f = float(u * b * w_a)
    e = 0.15 + x_ac_margin  # c/4 aerodynamic centre to the 40 % chord elastic axis
    K_a = float(GJ / half) * (np.pi / 2) ** 2
    q_d = K_a / (e * ci**2 * half * 0.5 * cl_a_rad)
    v_d = float(np.sqrt(2 * q_d / rho))
    return Structure(
        wing_mass=float(m_w),
        takeoff_mass=float(v.non_wing_mass + m_w),
        root_bending=float(M.max()),
        n_gust=float(n_g),
        v_flutter=v_f,
        v_divergence=v_d,
        k_flutter=kf,
        extras={"mu": sec2.mu, "sigma": sec2.sigma, "w_alpha": float(w_a), "w_h": float(w_h)},
    )


def calibrate(p: Planform, t_c: float, a_l0: float, cl_a: float, v: Vehicle) -> Vehicle:
    """Scale the box mass so that the baseline wing matches the assumed mass fraction."""
    m = build_mesh(p)
    target = v.wing_mass_fraction * (v.non_wing_mass / (1 - v.wing_mass_fraction))
    lo, hi = 0.2, 8.0
    for _ in range(30):
        v.k_struct = 0.5 * (lo + hi)
        s = size_structure(p, m, t_c, a_l0, cl_a, 0.0, v, target)
        if s.wing_mass > target:
            hi = v.k_struct
        else:
            lo = v.k_struct
    return v


def loft(p: Planform, xy: FloatArr, n_span: int = 30) -> tuple[FloatArr, NDArray[np.int64]]:
    """Triangulated surface of the half wing plus mirror, in metres, for STL export."""
    ys = p.span / 2 * np.linspace(-1, 1, 2 * n_span + 1)
    n = len(xy)
    V, Fc = [], []
    for k, yk in enumerate(ys):
        eta = abs(yk) / (p.span / 2)
        c = p.c_root * (1 - (1 - p.taper) * eta)
        tw = np.deg2rad(-p.twist_tip * eta**p.twist_exp)
        xs, zs = (xy[:, 0] - 0.25) * c, xy[:, 1] * c
        V.append(np.c_[xs * np.cos(tw) + zs * np.sin(tw) + 0.25 * p.c_root, np.full(n, yk), -xs * np.sin(tw) + zs * np.cos(tw)])
        if k:
            a, b = (k - 1) * n, k * n
            for i in range(n - 1):
                Fc += [(a + i, b + i, b + i + 1), (a + i, b + i + 1, a + i + 1)]
    return np.vstack(V), np.array(Fc, dtype=np.int64)
