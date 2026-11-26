r"""Whole-body optimisation over one design space, from conventional to free-form.

A family is only a choice of which variables are free and over what range. "Freeform" frees every
one, including the superformula, Fourier and Moebius (hyperbolic) section terms, so the search may
leave any conventional layout behind.

The score is mission fuel plus robust off-design drag, both relative to the baseline,

.. math:: J=\tfrac12\frac{m_f}{m_{f,0}}+\tfrac12\cdot\tfrac12\left(\mathbb E[r]+\mathrm{CVaR}_{0.9}[r]\right)+\text{penalties},

where r is drag relative to the baseline over hot days, cold days, contamination, rough air,
off-design speed and altitude. Constraints carry the things that make a shape a real aircraft:
cabin floor, fuel volume, span limit, approach speed, static margin, short-period damping,
directional stability, flutter, divergence, skin temperature, tail closure and where the wing
stalls first. A conventional tail has a minimum size, because trim and rotation at take-off need
authority that a static-margin check alone does not capture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerosynapse.aircraft import analysis as an
from aerosynapse.aircraft import vehicle as vh
from aerosynapse.aircraft.analysis import Flight, Mission
from aerosynapse.aircraft.atmosphere import G0, isa
from aerosynapse.optim.robust import cvar
from aerosynapse.shapeopt.cmaes import cmaes

FloatArr = NDArray[np.float64]
EUCLID_SECTION = ("n1", "f2", "f3")
HYPERBOLIC = ("wi",)
FIXED_TUBE = ("wi", "f2", "f3", "n1")


FAMILIES: dict[str, dict[str, Any]] = {
    "tube_and_wing": {"preset": "tube_and_wing", "fixed": FIXED_TUBE, "bounds": {"W": (1.6, 2.6), "H": (1.6, 2.6), "L": (26.0, 48.0), "Vh": (0.8, 1.3), "Vv": (0.065, 0.12)}, "link": ("W", "H")},
    "blended_wing_body": {"preset": "blended_wing_body", "fixed": ("wi", "f2", "f3"), "bounds": {"W": (5.0, 14.0), "H": (1.2, 2.4), "L": (18.0, 42.0), "Vh": (0.0, 0.0), "Vv": (0.02, 0.09)}},
    "flying_wing": {"preset": "flying_wing", "fixed": ("wi", "f2", "f3"), "bounds": {"W": (7.0, 16.0), "H": (1.05, 1.9), "L": (14.0, 32.0), "Vh": (0.0, 0.0), "Vv": (0.02, 0.07)}},
    "freeform": {"preset": "freeform_lifting_body", "fixed": (), "bounds": {}},
    "freeform_euclidean": {"preset": "freeform_lifting_body", "fixed": HYPERBOLIC, "bounds": {}, "zero": HYPERBOLIC},
    "supersonic_jet": {"preset": "supersonic_jet", "fixed": FIXED_TUBE, "bounds": {"W": (1.0, 2.2), "H": (1.0, 2.2), "L": (28.0, 60.0), "tc": (0.025, 0.07), "sweep": (35.0, 72.0), "AR": (1.6, 4.5), "S": (60.0, 260.0), "Vh": (0.0, 0.7)}, "link": ("W", "H")},
    "supersonic_freeform": {"preset": "supersonic_jet", "fixed": (), "bounds": {"tc": (0.025, 0.07), "sweep": (35.0, 72.0), "AR": (1.6, 4.5), "S": (60.0, 260.0), "L": (28.0, 60.0)}},
}

OFF_DESIGN = [
    Flight("hot day ISA+25", dT=25.0),
    Flight("cold day ISA-25", dT=-25.0),
    Flight("contaminated surface", contaminated=True),
    Flight("rough air, early transition", re_tr=6e5),
    Flight("high altitude +2 km", altitude=None),
    Flight("low altitude -3 km", altitude=None),
    Flight("high speed", mach=None),
    Flight("low speed", mach=None),
    Flight("1.5 g manoeuvre", n=1.5),
]


def off_design(m: Mission) -> list[Flight]:
    fl = [Flight(f.name, f.mach, f.altitude, f.dT, f.n, f.contaminated, f.re_tr) for f in OFF_DESIGN]
    fl[4].altitude = m.altitude + 2000.0
    fl[5].altitude = m.altitude - 3000.0
    fl[6].mach = m.mach * 1.04
    fl[7].mach = m.mach * 0.9
    return fl


def drag_states(v: vh.Vehicle, m: Mission, base: an.Result, fl: list[Flight]) -> FloatArr:
    """Drag in each off-design state at the design's own cruise weight."""
    W = 0.93 * base.mtow * G0
    out = np.empty(len(fl))
    for i, f in enumerate(fl):
        M = f.mach or m.mach
        air = isa(f.altitude or m.altitude, f.dT)
        q = 0.5 * air.rho * (M * air.a) ** 2
        CL = f.n * W / (q * v.S_ref)
        out[i] = an.drag(v, air, M, CL, f, kappa=m.k_cl).CD * q * v.S_ref
    return out


@dataclass
class Space:
    names: list[str]
    lo: FloatArr
    hi: FloatArr
    base: dict[str, float]
    link: tuple[str, str] | None = None

    def decode(self, z: FloatArr) -> dict[str, float]:
        d = dict(self.base)
        for k, v in zip(self.names, self.lo + np.clip(z, 0, 1) * (self.hi - self.lo), strict=False):
            d[k] = float(v)
        if self.link:
            d[self.link[1]] = d[self.link[0]]
        return d

    def encode(self, d: dict[str, float]) -> FloatArr:
        return np.array([(d[k] - lo) / max(hi - lo, 1e-12) for k, lo, hi in zip(self.names, self.lo, self.hi, strict=False)])


def space(family: str) -> Space:
    f = FAMILIES[family]
    base = dict(vh.PRESETS[f["preset"]])
    for k in f.get("zero", ()):
        base[k] = 0.0
    names, lo, hi = [], [], []
    link = f.get("link")
    for k, a, b in vh.VARS:
        if k in f["fixed"] or (link and k == link[1]):
            continue
        a, b = f["bounds"].get(k, (a, b))
        if b <= a:
            base[k] = a
            continue
        names.append(k)
        lo.append(a)
        hi.append(b)
    return Space(names, np.array(lo), np.array(hi), base, link)


class Study:
    """Scores designs against a fixed baseline."""

    def __init__(self, mission: Mission, baseline: dict[str, float], oew_target: float | None = None) -> None:
        self.v0 = vh.build(baseline)
        self.m = an.calibrate(self.v0, mission, oew_target)
        self.fl = off_design(self.m)
        self.r0 = an.evaluate(self.v0, self.m)
        self.d0 = drag_states(self.v0, self.m, self.r0, self.fl)

    def score(self, d: dict[str, float]) -> tuple[float, an.Result, FloatArr]:
        v = vh.build(d)
        r = an.evaluate(v, self.m)
        rr = drag_states(v, self.m, r, self.fl) / self.d0
        g = np.array(list(r.constraints.values()), float)
        g = np.where(np.isfinite(g), g, 5.0)
        p = np.maximum(g, 0.0)
        J = 0.5 * r.fuel / self.r0.fuel + 0.25 * (float(rr.mean()) + cvar(rr, 0.9)) + 2.0 * p.sum() + 25.0 * (p**2).sum()
        return float(J), r, rr


def run(job: dict[str, Any]) -> dict[str, Any]:
    """One family, one seed. Runs in its own process through scripts/shapeopt_worker.py."""
    t0 = time.time()
    m = Mission(**an.SUPERSONIC) if job["mission"] == "supersonic" else Mission()
    st = Study(m, vh.PRESETS[job["baseline"]], job.get("oew"))
    sp = space(job["family"])
    z0 = np.clip(sp.encode(sp.base), 0, 1)

    def f(Z: FloatArr) -> FloatArr:
        out = np.empty(len(Z))
        for i, z in enumerate(Z):
            try:
                out[i] = st.score(sp.decode(z))[0]
            except (FloatingPointError, np.linalg.LinAlgError, ValueError, ZeroDivisionError):
                out[i] = 50.0
        return out

    res = cmaes(f, z0, sigma=job.get("sigma", 0.2), seed=job["seed"], budget=job["budget"], pop=job.get("pop", 16), restarts=job.get("restarts", 2), stall_gens=45)
    d = sp.decode(res.x)
    J, r, rr = st.score(d)
    print(f"[{job['mission']}/{job['family']}/{job['seed']}] J {J:.4f} fuel {r.fuel / st.r0.fuel:.3f} L/D {r.L_over_D:.2f} t {time.time() - t0:.0f}s", flush=True)
    return {
        "family": job["family"], "mission": job["mission"], "seed": job["seed"], "J": J, "design": d, "summary": {k: float(x) for k, x in an.summary(r).items()},
        "constraints": {k: float(x) for k, x in r.constraints.items()}, "off_design": rr.tolist(), "off_names": [q.name for q in st.fl],
        "baseline_summary": {k: float(x) for k, x in an.summary(st.r0).items()}, "baseline_constraints": {k: float(x) for k, x in st.r0.constraints.items()},
        "history": res.history[:: max(1, len(res.history) // 80)], "evals": res.evals, "wall_s": time.time() - t0, "k_struct": st.m.k_wing,
    }  # fmt: skip
