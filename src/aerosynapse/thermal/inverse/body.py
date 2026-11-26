"""Dimensional forward models for the Phase 5 inverse problems: transient conduction in a 1D body with a known
volumetric source,

    rho c dT/dt = k (1/r^n) d/dr (r^n dT/dr) + q_vol,   0 < r < L,   t > 0
    dT/dr(0, t) = 0                       (symmetry: centre plane / axis / centre)
    -k dT/dr(L, t) = h (T(L, t) - T_inf)  (convective surface)
    T(r, 0) = T0

with shape n = 0 'slab' (plane wall of half-thickness L), n = 1 'cylinder' (long cylinder of radius L), n = 2 'sphere'
(radius L). Unknown parameters p = (alpha, k, h), alpha = k / (rho c); known: shape, L, T0, T_inf, q_vol, sensors,
times. With theta = (T - T_inf)/(T0 - T_inf), r* = r/L, t* = alpha t / L^2:

    theta_t* = (1/r*^n) (r*^n theta_r*)_r* + Q,   -theta_r*(1) = Bi theta(1),   theta(r*, 0) = 1,
    Bi = h L / k,   Q = q_vol L^2 / (k (T0 - T_inf)),

so without a source (Q = 0) k and h enter only through Bi = hL/k (structural confounding, Step 5.6). Steady state:
theta_ss = Q (1 - r*^2) / (2 (n + 1)) + Q / ((n + 1) Bi).

Forward models:
  'analytic'  eigen-series theta = theta_ss + sum_j C_j exp(-lam_j^2 t*) X(lam_j r*) with X = cos (slab, the closed-form
              Case B series of aerosynapse.thermal.physics.analytic), J0 (cylinder), sin(lam r)/(lam r) (sphere); eigenvalues from
              -X'(1) = Bi X(1); coefficients by Gauss-Legendre quadrature (cylinder, sphere)
  'fdm'       slab: the Phase 1 finite-difference solver (aerosynapse.thermal.numerics.fd1d); cylinder / sphere: a conservative
              cell-centred finite-volume solver (face areas r^n, second-order Robin closure); Crank-Nicolson with
              Rannacher start-up and fixed step counts (smooth in the parameters, Step 5.1 / D83)
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
from scipy.special import j0, j1

from aerosynapse.thermal.numerics import fd1d
from aerosynapse.thermal.numerics.linalg import factorize
from aerosynapse.thermal.numerics.time_integrators import integrate
from aerosynapse.thermal.physics import analytic as an

PARAMS = ("alpha", "k", "h")
SHAPES = {"slab": 0, "cylinder": 1, "sphere": 2}


@dataclass(frozen=True)
class Body:
    shape: str  # slab | cylinder | sphere
    L: float  # half-thickness (slab) or radius [m]
    T0: float  # initial temperature
    T_inf: float  # ambient temperature
    q_vol: float  # known volumetric heat source [W/m^3]

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"body.shape '{self.shape}': one of {list(SHAPES)}")
        if not self.L > 0:
            raise ValueError("body.L must be > 0")
        if self.T0 == self.T_inf:
            raise ValueError("body.T0 must differ from body.T_inf (temperature scale T0 - T_inf)")

    @property
    def n(self) -> int:
        return SHAPES[self.shape]

    @property
    def dT(self) -> float:
        return self.T0 - self.T_inf

    def groups(self, p: dict[str, float]) -> dict[str, float]:
        """Dimensionless groups of parameter set p: Bi, Q and the time scale L^2 / alpha."""
        return {
            "Bi": p["h"] * self.L / p["k"],
            "Q": self.q_vol * self.L**2 / (p["k"] * self.dT),
            "t_scale": self.L**2 / p["alpha"],
        }


def steady(r: np.ndarray, Bi: float, Q: float, n: int) -> np.ndarray:
    return Q * (1.0 - np.asarray(r, float) ** 2) / (2.0 * (n + 1)) + Q / ((n + 1) * Bi)


def _mode(lam: np.ndarray, r: np.ndarray, n: int) -> np.ndarray:
    """X(lam r) for every (lam, r): array (len(lam), len(r))."""
    z = np.outer(lam, r)
    if n == 0:
        return np.cos(z)
    if n == 1:
        return j0(z)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(z == 0.0, 1.0, np.sin(z) / np.where(z == 0.0, 1.0, z))


@functools.lru_cache(maxsize=16)
def _gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [-1, 1], cached by order (recomputing them dominated each evaluation)."""
    g, w = np.polynomial.legendre.leggauss(order)
    g.setflags(write=False)
    w.setflags(write=False)
    return g, w


def eigenvalues(Bi: float, n_terms: int, n: int) -> np.ndarray:
    """First n_terms positive roots of -X'(1) = Bi X(1)."""
    if Bi <= 0:
        raise ValueError("Bi must be > 0")
    if n == 0:
        return an.robin_eigenvalues(Bi, n_terms)
    if n == 1:
        g = lambda lam: lam * j1(lam) - Bi * j0(lam)  # noqa: E731
    else:
        g = lambda lam: lam * np.cos(lam) - (1.0 - Bi) * np.sin(lam)  # noqa: E731  (lam = 0 trivial root excluded)
    grid = np.arange(1e-9, (n_terms + 2) * np.pi, np.pi / 64)
    vals = g(grid)
    idx = np.flatnonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[:n_terms]
    if len(idx) < n_terms:
        raise RuntimeError(f"eigenvalue scan found {len(idx)} of {n_terms} roots")
    # all roots at once: vectorised bisection of the scan brackets (width pi/64; 52 halvings reach machine precision,
    # ~40x faster than one scalar root-finder call per root, which dominated the cylinder / sphere series)
    a, b = grid[idx], grid[idx + 1]
    fa = vals[idx]
    for _ in range(52):
        m = 0.5 * (a + b)
        fm = g(m)
        same = np.sign(fm) == np.sign(fa)
        a, fa, b = np.where(same, m, a), np.where(same, fm, fa), np.where(same, b, m)
    return np.where(np.abs(fa) <= np.abs(g(b)), a, b)  # the bracket end with the smaller residual (as a scalar solver)


def theta_analytic(x: np.ndarray, t: np.ndarray, Bi: float, Q: float, n_terms: int = 100, n: int = 0) -> np.ndarray:
    """Eigen-series at all (t, r*): array (len(t), len(x)). Slab: the closed-form Case B coefficients (Phase 1)."""
    x, t = np.asarray(x, float), np.asarray(t, float)
    if n > 0:  # radial bodies: the quadrature cost grows as n_terms^2; N_TERMS_RADIAL terms are exact for t* >= ~4e-5
        n_terms = min(n_terms, N_TERMS_RADIAL)
    lam = eigenvalues(Bi, n_terms, n)
    if n == 0:
        norm = 0.5 + np.sin(2 * lam) / (4 * lam)
        C = (np.sin(lam) / lam - an._cos_projection_steady(lam, Bi, Q, 0.0)) / norm
    else:  # C_j = int r^n (1 - theta_ss) X_j dr / int r^n X_j^2 dr, Gauss-Legendre on [0, 1]
        g, w = _gauss_legendre(max(400, 8 * n_terms))
        r, w = 0.5 * (g + 1.0), 0.5 * w * (0.5 * (g + 1.0)) ** n
        X = _mode(lam, r, n)
        C = (X * (1.0 - steady(r, Bi, Q, n)) * w).sum(1) / (X**2 * w).sum(1)
    decay = np.exp(-np.outer(t, lam**2)) * C
    return steady(x, Bi, Q, n)[None, :] + decay @ _mode(lam, x, n)


def _segments(t: np.ndarray, dt: float, steps) -> tuple[np.ndarray, np.ndarray]:
    seg = np.diff(np.concatenate([[0.0], t]))
    k = np.asarray(steps, int) if steps is not None else np.maximum(np.ceil(seg / dt - 1e-12).astype(int), 1)
    if len(k) != len(t) or np.any(k < 1):
        raise ValueError("steps: one positive count per measurement interval")
    return seg, k


def theta_fdm(
    x: np.ndarray,
    t: np.ndarray,
    Bi: float,
    Q: float,
    N: int = 200,
    dt: float = 1e-3,
    method: str = "crank_nicolson",
    steps: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Slab: Phase 1 FD solver (vertex grid, N intervals), linear interpolation to x. Time steps: ``steps`` = number of
    uniform steps in each interval (0, t_0], (t_0, t_1], ... (fixed counts make the output a SMOOTH function of the
    parameters, which finite-difference Jacobians and least squares need); without ``steps``, a uniform step dt in
    Fourier time (the step count then jumps with the parameters). Crank–Nicolson starts with Rannacher smoothing:
    its first two steps are replaced by four implicit-Euler half-steps, which damp the stiff modes excited by the
    uniform initial state meeting the Robin boundary (CN alone: 0.04 K error at the surface at t* = 0.05)."""
    prob = fd1d.ConvDiff1D(
        N,
        a=0.0,
        D=1.0,
        source=lambda xx, tt: np.full_like(np.asarray(xx, float), Q),
        left=fd1d.neumann(0.0),
        right=fd1d.robin(Bi),
    )
    t = np.asarray(t, float)
    seg, n = _segments(t, dt, steps)
    y = np.ones(len(prob.x_active))
    out, t_prev = [], 0.0
    for i, tt in enumerate(t):
        h = seg[i] / n[i]
        k = n[i]
        if i == 0 and method == "crank_nicolson" and n[0] >= 3:
            y = integrate(prob, y, 0.0, 2.0 * h, 0.5 * h, "implicit_euler")
            t_prev, k = 2.0 * h, n[0] - 2
        if k > 0 and tt > t_prev:
            y = integrate(prob, y, t_prev, float(tt), (float(tt) - t_prev) / k, method)
        out.append(np.interp(x, prob.x, prob.nodal(y, float(tt))))
        t_prev = float(tt)
    return np.asarray(out)


def theta_fv(
    x: np.ndarray,
    t: np.ndarray,
    Bi: float,
    Q: float,
    n: int,
    N: int = 100,
    dt: float = 1e-3,
    steps: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Radial finite volumes for any shape: N cells, d theta_i/dt = (sum of face fluxes A_f dtheta/dr) / V_i + Q with
    face areas r_f^n, volumes (r_{i+1}^{n+1} - r_i^{n+1})/(n+1), zero flux at r = 0, Robin flux A Bi theta_s at r = 1
    with theta_s = theta_N / (1 + Bi h/2) (second order). Crank-Nicolson with Rannacher start-up; output by linear
    interpolation between r = 0 (zero-slope value), the cell centres and the surface value."""
    h = 1.0 / N
    rf = np.linspace(0.0, 1.0, N + 1)
    rc = 0.5 * (rf[:-1] + rf[1:])
    A = rf**n
    V = (rf[1:] ** (n + 1) - rf[:-1] ** (n + 1)) / (n + 1)
    main = np.zeros(N)
    off = A[1:-1] / h  # coupling through the interior faces
    main[:-1] -= off
    main[1:] -= off
    main[-1] -= A[-1] * Bi / (1.0 + 0.5 * Bi * h)
    M = sps.diags([off / V[1:], main / V, off / V[:-1]], [-1, 0, 1], format="csr")
    I = sps.identity(N, format="csr")
    t = np.asarray(t, float)
    seg, ks = _segments(t, dt, steps)
    y = np.ones(N)
    src = np.full(N, Q)
    cache: dict = {}

    def step(y, d, scheme):
        key = (round(d, 15), scheme)
        if key not in cache:
            cache[key] = factorize(I - d * M if scheme == "ie" else I - 0.5 * d * M, "splu")
        rhs = y + d * src if scheme == "ie" else y + 0.5 * d * (M @ y) + d * src
        return cache[key](rhs)

    out, t_prev = [], 0.0
    for i, tt in enumerate(t):
        d, k = seg[i] / ks[i], ks[i]
        if i == 0 and ks[0] >= 3:
            for _ in range(4):
                y = step(y, 0.5 * d, "ie")
            t_prev, k = 2.0 * d, ks[0] - 2
        dd = (float(tt) - t_prev) / k if k > 0 else 0.0
        for _ in range(k if tt > t_prev else 0):
            y = step(y, dd, "cn")
        surf = y[-1] / (1.0 + 0.5 * Bi * h)
        out.append(np.interp(x, np.concatenate([[0.0], rc, [1.0]]), np.concatenate([[y[0]], y, [surf]])))
        t_prev = float(tt)
    return np.asarray(out)


def fdm_steps(ts: np.ndarray, dt_seconds: float) -> np.ndarray:
    """Step counts per measurement interval from a DIMENSIONAL step (independent of the parameters); >= 4 in the
    first interval (Rannacher start-up)."""
    seg = np.diff(np.concatenate([[0.0], np.asarray(ts, float)]))
    n = np.maximum(np.ceil(seg / dt_seconds - 1e-12).astype(int), 1)
    n[0] = max(n[0], 4)
    return n


def temperatures(
    p: dict[str, float],
    body: Body,
    xs: np.ndarray,
    ts: np.ndarray,
    model: str = "analytic",
    n_terms: int = 100,
    fdm: dict | None = None,
) -> np.ndarray:
    """Temperatures at sensor positions xs [m] and times ts [s] (sorted, > 0): array (len(ts), len(xs)).
    fdm: {N, dt_fourier | dt_seconds, method}; dt_seconds (set by aerosynapse.thermal.inverse.spec.fdm_settings from the TRUE alpha)
    fixes the step counts, so the numerical output is smooth in the parameters."""
    g = body.groups(p)
    xstar, tstar = np.asarray(xs, float) / body.L, np.asarray(ts, float) / g["t_scale"]
    if model == "analytic":
        th = theta_analytic(xstar, tstar, g["Bi"], g["Q"], n_terms, body.n)
    elif model == "fdm":
        f = fdm or {}
        steps = fdm_steps(ts, float(f["dt_seconds"])) if f.get("dt_seconds") else None
        N, dtf = int(f.get("N", 200)), float(f.get("dt_fourier", 1e-3))
        if body.n == 0 and f.get("solver", "fd1d") == "fd1d":
            th = theta_fdm(xstar, tstar, g["Bi"], g["Q"], N, dtf, f.get("method", "crank_nicolson"), steps)
        else:
            th = theta_fv(xstar, tstar, g["Bi"], g["Q"], body.n, N, dtf, steps)
    else:
        raise ValueError(f"forward model '{model}': analytic or fdm")
    return body.T_inf + body.dT * th


N_TERMS_RADIAL = 300  # cylinder / sphere: quadrature cost ~8 n_terms^2; exact for t* >= ~4e-5
N_TERMS_MAX = 4000  # covers t*_min >= ~2e-7; below that the parameters are far outside any physical regime


def truncation_terms(t_min_fourier: float, tol: float = 1e-14) -> int:
    """Smallest eigen-series length with exp(-(n pi)^2 t_min) < tol (the first omitted term's decay), capped at
    N_TERMS_MAX: an optimiser or sampler exploring alpha -> 0 (Step 5.6 profiles) must not request ~1e10 terms."""
    return min(int(np.ceil(np.sqrt(-np.log(tol) / max(t_min_fourier, 1e-300)) / np.pi)) + 1, N_TERMS_MAX)
