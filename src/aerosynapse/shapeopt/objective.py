r"""The whole-design objective: section and wing are scored together, in every regime at once.

A design is z in [0, 1]^24: 18 section numbers and 6 planform numbers (area, aspect ratio, taper,
tip twist, twist shape, winglet height). Extending or shrinking the span, tapering, twisting,
reshaping the section and adding a winglet are all moves in this one space.

For each flow state xi the score is the drag relative to the baseline in that same state, so a
laminar state and a turbulent state count equally. The objective is a mean-risk blend,

.. math:: J=\tfrac12\bar J_{2D}+\tfrac12\bar J_{3D}+\rho\sum_i\max(g_i,0)+\rho_2\sum_i\max(g_i,0)^2,
          \quad \bar J=\tfrac12\left(\mathbb E[r]+\mathrm{CVaR}_{0.9}[r]\right)

where CVaR_0.9 is the mean of the worst 10 % of states. The tail term is what makes the design
robust: without it the optimiser sacrifices rare extreme states for ordinary ones.

Guards against optimising model error rather than the physics:

* NeuralFoil's ``analysis_confidence`` is a constraint.
* NeuralFoil misses abrupt stall (see the NACA 0012 check), so any c_l,max gain it reports is
  capped at ``CLMAX_CREDIT`` times the baseline when computing stall speed. The stall margin is
  then re-judged by the independent solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from aerosynapse.optim.robust import cvar
from aerosynapse.shapeopt import aero2d, scenarios
from aerosynapse.shapeopt import wing as wg
from aerosynapse.shapeopt.geometry import SectionLimits, section_metrics, section_violations

FloatArr = NDArray[np.float64]
RHO_SL = 1.225
MU_AIR = 1.79e-5
A_GRID = np.arange(-6.0, 19.0, 1.0)
CLMAX_CREDIT = 1.05
SHORT_PENALTY = 8.0
PLAN_LO = np.array([11.0, 5.0, 0.30, -8.0, 0.5, 0.0])
PLAN_HI = np.array([26.0, 16.0, 1.10, 2.0, 3.0, 0.15])
PLAN_NAMES = ("area_m2", "aspect_ratio", "taper", "tip_twist_deg", "twist_exponent", "winglet_height")
SEC_DELTA = 0.14
A_OPERATING = A_GRID <= 12.0  # confidence is judged where the wing flies, not on post-stall table rows
G_NAMES = (
    "thickness", "area", "nose_radius", "rear_depth", "waviness", "surface_crossing",
    "stall_speed", "stall_location", "flutter", "divergence", "wing_solver", "surrogate_confidence", "pitching_moment",
)  # fmt: skip


@dataclass
class FlightStates:
    V: FloatArr
    rho: FloatArr
    n: FloatArr
    ncrit: FloatArr
    xtr: FloatArr
    names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.V)


NAMED_FLIGHT = [
    ("cruise", 60.0, 0.96, 1.0, 9.0, 1.0),
    ("cruise_turbulent_air", 60.0, 0.96, 1.0, 3.0, 1.0),
    ("cruise_contaminated", 60.0, 0.96, 1.0, 9.0, 0.05),
    ("climb", 38.0, 1.20, 1.0, 9.0, 1.0),
    ("approach_dirty", 32.0, 1.225, 1.0, 6.0, 0.05),
    ("manoeuvre_2.5g", 55.0, 1.10, 2.5, 9.0, 1.0),
    ("rough_air_1.8g", 68.0, 1.00, 1.8, 3.0, 1.0),
]


def flight_states(n_random: int, seed: int, v_stall: float = 27.0) -> FlightStates:
    """Named states plus random ones that stay inside the V-n envelope.

    Speed keeps a 1.25 margin over the stall speed at that density, and load factor stays below
    80 % of what the wing can reach at that speed. A state outside the envelope cannot be flown by
    any aircraft of this class, so scoring it would reward oversized wings for fictional physics.
    """
    r = np.random.default_rng(seed)
    a = np.array([q[1:] for q in NAMED_FLIGHT], float)
    rho = r.uniform(0.80, 1.225, n_random)
    sig = rho / RHO_SL
    V = np.maximum(r.uniform(34.0, 72.0, n_random), 1.25 * v_stall / np.sqrt(sig))
    n_env = 0.8 * sig * (V / v_stall) ** 2
    n = np.where(r.random(n_random) < 0.3, r.uniform(1.3, 2.5, n_random), 1.0)
    n = np.maximum(np.minimum(n, n_env), 1.0)
    nc = np.where(r.random(n_random) < 0.35, r.uniform(1.5, 4.0, n_random), r.uniform(7.0, 11.0, n_random))
    xt = np.where(r.random(n_random) < 0.25, r.uniform(0.03, 0.10, n_random), 1.0)
    names = [q[0] for q in NAMED_FLIGHT] + [f"random_{i}" for i in range(n_random)]
    return FlightStates(np.r_[a[:, 0], V], np.r_[a[:, 1], rho], np.r_[a[:, 2], n], np.r_[a[:, 3], nc], np.r_[a[:, 4], xt], names)


def _lift_line_fit(cl: FloatArr) -> tuple[float, float]:
    """Zero-lift angle (deg) and slope (per deg) from the linear part of a c_l table on A_GRID."""
    m = (A_GRID >= -2) & (A_GRID <= 6)
    s, b = np.polyfit(A_GRID[m], cl[m], 1)
    return float(-b / s), float(s)


@dataclass
class WingEval:
    drag: FloatArr  # N, per flight state
    CD: FloatArr
    CDi: FloatArr
    CL: FloatArr
    conf: float
    cl_max_wing: float
    eta_stall: float
    v_stall: float
    structure: wg.Structure
    cl_section: FloatArr
    y: FloatArr
    chord: FloatArr
    gamma: FloatArr
    converged: bool


def eval_wing(
    plan: wg.Planform, v: wg.Vehicle, fs: FlightStates, tab: dict[str, FloatArr], t_c: float, clmax_ref: float | None, n_mesh: int = 24
) -> WingEval:
    """Evaluate one wing. ``tab`` holds section tables (F, A) for the flight states."""
    m = wg.build_mesh(plan, n=n_mesh)
    # the first state is clean cruise, which defines the linear section properties
    a_l0, cl_a = _lift_line_fit(tab["CL"][0])
    st = wg.size_structure(plan, m, t_c, a_l0, cl_a, 0.0, v)
    W = st.takeoff_mass * wg.G0
    q = 0.5 * fs.rho * fs.V**2
    CL = fs.n * W / (q * plan.area)
    cl_cap = np.maximum.accumulate(tab["CL"], axis=1)  # no descending branch inside Newton
    r = wg.solve_llt(m, CL, A_GRID, cl_cap, tab["CD"])
    D = q * plan.area * r.CD + q * v.parasite_area
    clmax_sec = float(tab["CL"][3].max())  # climb state: low-speed clean section
    if clmax_ref is not None:
        clmax_sec = min(clmax_sec, CLMAX_CREDIT * clmax_ref)
    cl_max_w, eta = wg.wing_clmax(m, a_l0, cl_a, clmax_sec)
    v_s = float(np.sqrt(2 * W / (RHO_SL * plan.area * max(cl_max_w, 0.3))))
    # a state that the wing cannot carry is charged rather than ignored
    over = np.maximum(CL / max(cl_max_w, 0.3) - 0.95, 0.0)
    D = D * (1 + 4.0 * over)
    return WingEval(D, r.CD, r.CDi, CL, float(tab["conf"].min()), cl_max_w, eta, v_s, st, r.cl, m.y, m.c, r.gamma, r.converged)


class Problem:
    """Baseline, scenarios and limits for one optimisation; evaluates batches of designs."""

    def __init__(
        self,
        W0: FloatArr,
        plan0: wg.Planform,
        n_random_2d: int = 40,
        n_random_3d: int = 10,
        seed: int = 11,
        model: str = "large",
        weight_2d: float = 0.5,
    ) -> None:
        self.W0, self.plan0, self.model, self.w2 = np.asarray(W0, float), plan0, model, weight_2d
        self.lim = SectionLimits.from_baseline(self.W0)
        self.t0 = float(section_metrics(self.W0[None])["t"][0])
        self.cond = scenarios.concat(scenarios.named(), scenarios.random_extreme(n_random_2d, seed))
        self.fs = flight_states(n_random_3d, seed + 1)
        self.v = wg.Vehicle()
        self.clmax_ref: float | None = None
        t = self._tables(self.W0[None], [plan0])
        a_l0, cl_a = _lift_line_fit(t["CL"][0, 0])
        self.v = wg.calibrate(plan0, self.t0, a_l0, cl_a, self.v)
        self.clmax_ref = float(t["CL"][0, 3].max())
        o = aero2d.solve_at_cl(self.W0[None], self.cond, model)
        self.cd_ref = o["cd"][0]
        self.conf_min = float(min(0.70, o["conf"].min() - 0.03, t["conf"][..., A_OPERATING].min() - 0.03))
        self.cm_ref = float(o["cm"][0, 3])
        we = self.wing_eval(plan0, {k: t[k][0] for k in t}, self.t0)
        # the envelope depends on this vehicle's real stall speed, so redraw with it
        self.fs = flight_states(n_random_3d, seed + 1, we.v_stall)
        t = self._tables(self.W0[None], [plan0])
        we = self.wing_eval(plan0, {k: t[k][0] for k in t}, self.t0)
        self.drag_ref = we.drag
        self.v_s_ref = we.v_stall
        self.base_wing = we

    # -- mapping between z in [0, 1]^24 and physical variables
    def encode(self, W: FloatArr, plan: wg.Planform) -> FloatArr:
        zs = (np.asarray(W, float) - self.W0) / (2 * SEC_DELTA) + 0.5
        zs[17] = np.clip(W[17] / 0.01, 0, 1)
        return np.r_[zs, (plan.as_array() - PLAN_LO) / (PLAN_HI - PLAN_LO)]

    def decode(self, Z: FloatArr) -> tuple[FloatArr, list[wg.Planform]]:
        Z = np.atleast_2d(Z)
        W = self.W0 + (Z[:, :18] - 0.5) * 2 * SEC_DELTA
        W[:, 17] = Z[:, 17] * 0.01
        P = PLAN_LO + Z[:, 18:] * (PLAN_HI - PLAN_LO)
        return W, [wg.Planform(*row) for row in P]

    def _tables(self, W: FloatArr, plans: list[wg.Planform]) -> dict[str, FloatArr]:
        D, F, A = W.shape[0], len(self.fs), len(A_GRID)
        mac = np.array([p.mac for p in plans])
        Re = self.fs.rho[None, :] * self.fs.V[None, :] * mac[:, None] / MU_AIR  # (D, F)
        rows = (D, F, A)
        Wr = np.broadcast_to(W[:, None, None, :], (*rows, 18)).reshape(-1, 18)
        one = np.ones(rows)
        r = aero2d.nf_eval(
            Wr,
            (one * A_GRID).ravel(),
            (one * Re[:, :, None]).ravel(),
            (one * self.fs.ncrit[None, :, None]).ravel(),
            (one * self.fs.xtr[None, :, None]).ravel(),
            (one * self.fs.xtr[None, :, None]).ravel(),
            self.model,
        )
        return {k: r[k].reshape(rows) for k in ("CL", "CD", "CM", "conf")}

    def wing_eval(self, plan: wg.Planform, tab: dict[str, FloatArr], t_c: float) -> WingEval:
        return eval_wing(plan, self.v, self.fs, tab, t_c, self.clmax_ref)

    def __call__(self, Z: FloatArr) -> FloatArr:
        return self.evaluate(Z)["J"]

    def evaluate(self, Z: FloatArr) -> dict[str, FloatArr]:
        W, plans = self.decode(Z)
        D = W.shape[0]
        met = section_metrics(W)
        g_sec = section_violations(W, self.lim)
        o = aero2d.solve_at_cl(W, self.cond, self.model)
        r2 = o["cd"] / self.cd_ref[None, :] + SHORT_PENALTY * o["cl_short"]
        tabs = self._tables(W, plans)
        J2 = np.empty(D)
        J3 = np.empty(D)
        g_w = np.zeros((D, 5))
        for i in range(D):
            J2[i] = 0.5 * (r2[i].mean() + cvar(r2[i], 0.9))
            we = self.wing_eval(plans[i], {k: tabs[k][i] for k in tabs}, float(met["t"][i]))
            r3 = we.drag / self.drag_ref
            J3[i] = 0.5 * (r3.mean() + cvar(r3, 0.9))
            s = we.structure
            g_w[i] = [
                we.v_stall / self.v_s_ref - 1.0,  # stall speed must not rise
                (we.eta_stall - 0.60) / 0.6,  # stall must start inboard, keeping the ailerons flying
                1.2 * self.v.v_dive / max(s.v_flutter, 1.0) - 1.0,
                1.2 * self.v.v_dive / max(s.v_divergence, 1.0) - 1.0,
                0.0 if we.converged else 0.2,
            ]
        g_conf = (self.conf_min - np.minimum(o["conf"].min(1), tabs["conf"][..., A_OPERATING].min((1, 2)))) / 0.1
        g_cm = (self.cm_ref - 0.03 - o["cm"][:, 3]) / 0.05
        g = np.c_[g_sec, g_w, g_conf, g_cm]
        v = np.maximum(g, 0.0)
        J0 = self.w2 * J2 + (1 - self.w2) * J3
        J = J0 + 2.0 * v.sum(1) + 40.0 * (v**2).sum(1)
        return {"J": J, "J0": J0, "J2": J2, "J3": J3, "g": g, "W": W, "r2": r2}
