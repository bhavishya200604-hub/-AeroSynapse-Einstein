r"""Section aerodynamics for the optimisation loop (NeuralFoil adapter plus compressibility).

NeuralFoil (Sharpe) is a neural surrogate trained on XFOIL. It predicts lift, drag, moment and
transition across laminar, transitional and turbulent boundary layers, and it returns an
``analysis_confidence`` that drops outside its training distribution. That number is used as a
hard constraint, because an optimiser otherwise drifts to wherever the surrogate is wrong.

NeuralFoil is incompressible. Compressibility is added here:

* Prandtl-Glauert: the section must give :math:`c_{l,0}=c_l\beta`, :math:`\beta=\sqrt{1-M^2}`.
* Critical Mach from the peak edge velocity: Karman-Tsien-corrected :math:`C_{p,min}` equals
  :math:`C_p^*(M)`.
* Wave drag from Lock's fourth-power law :math:`c_{dw}=20(M-M_{cr})^4`.

These are engineering estimates, not a transonic solution. The first boundary-layer station is
at x/c = 0.0156, so a sharp leading-edge suction peak is under-resolved and M_cr is optimistic at
high lift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]
ALPHA_GRID = np.arange(-14.0, 22.0, 2.0)
CHUNK = 60000


def nf_eval(
    W: FloatArr, alpha: FloatArr, Re: FloatArr, ncrit: FloatArr, xtu: FloatArr, xtl: FloatArr, model: str = "large"
) -> dict[str, FloatArr]:
    """Row-wise NeuralFoil query. ``W`` is (R, 18) and the rest are (R,)."""
    import neuralfoil as nf  # optional dependency: pip install "aerosynapse-einstein[shapeopt]"

    W = np.asarray(W, float)
    R = W.shape[0]
    out: dict[str, list[FloatArr]] = {k: [] for k in ("CL", "CD", "CM", "conf", "xtr_u", "ue_max")}
    ue_keys = [f"upper_bl_ue/vinf_{i}" for i in range(32)]
    for a in range(0, R, CHUNK):
        s = slice(a, min(a + CHUNK, R))
        k = {
            "upper_weights": W[s, :8].T,
            "lower_weights": W[s, 8:16].T,
            "leading_edge_weight": W[s, 16],
            "TE_thickness": W[s, 17],
        }
        r = nf.get_aero_from_kulfan_parameters(
            k, alpha=alpha[s], Re=Re[s], n_crit=ncrit[s], xtr_upper=xtu[s], xtr_lower=xtl[s], model_size=model
        )
        out["CL"].append(np.asarray(r["CL"]))
        out["CD"].append(np.asarray(r["CD"]))
        out["CM"].append(np.asarray(r["CM"]))
        out["conf"].append(np.asarray(r["analysis_confidence"]))
        out["xtr_u"].append(np.asarray(r["Top_Xtr"]))
        out["ue_max"].append(np.max([np.asarray(r[q]) for q in ue_keys], axis=0))
    return {k: np.concatenate(v) for k, v in out.items()}


@dataclass
class Conditions:
    """A list of P flow states. Fields are (P,) arrays."""

    Re: FloatArr
    mach: FloatArr
    cl: FloatArr
    ncrit: FloatArr
    xtr: FloatArr
    names: list[str] = field(default_factory=list)
    weight: FloatArr | None = None

    def __len__(self) -> int:
        return len(self.Re)

    def take(self, idx: NDArray[np.int64] | slice) -> Conditions:
        return Conditions(self.Re[idx], self.mach[idx], self.cl[idx], self.ncrit[idx], self.xtr[idx])


def critical_mach(cp_min: FloatArr) -> FloatArr:
    """Solve C_p^*(M) = Karman-Tsien(C_p,min, M) for M by bisection (vectorised)."""
    cp0 = np.minimum(cp_min, -1e-3)
    lo, hi = np.full_like(cp0, 0.05), np.full_like(cp0, 0.99)
    for _ in range(40):
        m = 0.5 * (lo + hi)
        b = np.sqrt(1 - m**2)
        kt = cp0 / (b + m**2 / (1 + b) * cp0 / 2)
        star = 2 / (1.4 * m**2) * (((2 + 0.4 * m**2) / 2.4) ** 3.5 - 1)
        over = kt < star  # local flow is already supersonic: M is above critical
        hi = np.where(over, m, hi)
        lo = np.where(over, lo, m)
    return 0.5 * (lo + hi)


def solve_at_cl(W: FloatArr, c: Conditions, model: str = "large", refine: int = 2) -> dict[str, FloatArr]:
    """For designs (D, 18) and P conditions, find the state that gives each target c_l.

    Returns (D, P) arrays. ``cd`` includes wave drag. If the target exceeds c_l,max the design is
    held at c_l,max, and ``cl_short`` records the lift it failed to deliver.
    """
    W = np.atleast_2d(W)
    D, P, A = W.shape[0], len(c), len(ALPHA_GRID)
    beta = np.sqrt(1 - c.mach**2)
    rows = (D, P, A)
    Wr = np.broadcast_to(W[:, None, None, :], (*rows, 18)).reshape(-1, 18)
    br = lambda v: np.broadcast_to(v[None, :, None], rows).ravel()  # noqa: E731
    al = np.broadcast_to(ALPHA_GRID[None, None, :], rows).ravel()
    r = nf_eval(Wr, al, br(c.Re), br(c.ncrit), br(c.xtr), br(c.xtr), model)
    CL = r["CL"].reshape(rows)

    # c_l,max with a parabola through the three points around the discrete peak
    k = np.clip(np.argmax(CL, axis=2), 1, A - 2)
    ii, jj = np.meshgrid(np.arange(D), np.arange(P), indexing="ij")
    y0, y1, y2 = CL[ii, jj, k - 1], CL[ii, jj, k], CL[ii, jj, k + 1]
    den = y0 - 2 * y1 + y2
    d = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
    d = np.clip(d, -1, 1)
    cl_max = np.maximum(y1 - 0.25 * (y0 - y2) * d, y1)
    a_stall = ALPHA_GRID[k] + 2.0 * d

    target = np.broadcast_to((c.cl * beta)[None, :], (D, P))
    tgt = np.minimum(target, 0.995 * cl_max)
    # bracket on the attached branch: last grid point below the target, before the peak
    idx = np.arange(A)[None, None, :]
    below = (tgt[..., None] > CL) & (idx <= k[..., None])
    lo = np.clip(below.sum(2) - 1, 0, A - 2)
    c0, c1 = CL[ii, jj, lo], CL[ii, jj, lo + 1]
    alpha = ALPHA_GRID[lo] + 2.0 * (tgt - c0) / np.where(np.abs(c1 - c0) < 1e-6, 1e-6, c1 - c0)
    alpha = np.clip(alpha, ALPHA_GRID[0], ALPHA_GRID[-1])

    Wd = np.broadcast_to(W[:, None, :], (D, P, 18)).reshape(-1, 18)
    b2 = lambda v: np.broadcast_to(v[None, :], (D, P)).ravel()  # noqa: E731
    args = (b2(c.Re), b2(c.ncrit), b2(c.xtr), b2(c.xtr))
    slope = np.full((D, P), 0.11)
    s = nf_eval(Wd, alpha.ravel(), *args, model)
    for _ in range(refine):
        err = tgt - s["CL"].reshape(D, P)
        alpha = np.clip(alpha + np.clip(err / slope, -2.0, 2.0), ALPHA_GRID[0], ALPHA_GRID[-1])
        s = nf_eval(Wd, alpha.ravel(), *args, model)

    cd = s["CD"].reshape(D, P)
    cp_min = 1 - s["ue_max"].reshape(D, P) ** 2
    m_cr = critical_mach(cp_min)
    cd_w = 20.0 * np.maximum(c.mach[None, :] - m_cr, 0.0) ** 4
    return {
        "cd": cd + cd_w,
        "cd_visc": cd,
        "cd_wave": cd_w,
        "cl": s["CL"].reshape(D, P) / beta[None, :],
        "cm": s["CM"].reshape(D, P) / beta[None, :],
        "alpha": alpha,
        "conf": s["conf"].reshape(D, P),
        "xtr_u": s["xtr_u"].reshape(D, P),
        "cl_max": cl_max,  # incompressible value: real c_l,max falls with Mach, so it is not scaled up
        "alpha_stall": a_stall,
        "cl_short": np.maximum(target - 0.995 * cl_max, 0.0) / beta[None, :],
        "m_cr": m_cr,
    }


def polar(
    W: FloatArr, Re: float, ncrit: float = 9.0, xtr: float = 1.0, alpha: FloatArr | None = None, model: str = "xxxlarge"
) -> dict[str, FloatArr]:
    """Dense polar for one design."""
    a = np.arange(-10.0, 24.01, 0.5) if alpha is None else np.asarray(alpha, float)
    one = np.ones_like(a)
    r = nf_eval(np.tile(np.asarray(W, float), (len(a), 1)), a, Re * one, ncrit * one, xtr * one, xtr * one, model)
    r["alpha"] = a
    return r


def section_table(
    W: FloatArr, Re: FloatArr, ncrit: FloatArr, xtr: FloatArr, alpha: FloatArr, model: str = "large"
) -> dict[str, FloatArr]:
    """c_l and c_d on a grid for designs (D, 18): arrays of shape (D, n_cond, n_alpha)."""
    W = np.atleast_2d(W)
    D, C, A = W.shape[0], len(Re), len(alpha)
    rows = (D, C, A)
    Wr = np.broadcast_to(W[:, None, None, :], (*rows, 18)).reshape(-1, 18)
    br = lambda v: np.broadcast_to(np.asarray(v)[None, :, None], rows).ravel()  # noqa: E731
    al = np.broadcast_to(alpha[None, None, :], rows).ravel()
    r = nf_eval(Wr, al, br(Re), br(ncrit), br(xtr), br(xtr), model)
    return {k: r[k].reshape(rows) for k in ("CL", "CD", "CM", "conf")}
