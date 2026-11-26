"""Step 5.1: optimal experiment design — choose the measurement design that minimises the Cramer-Rao bounds of the
unknown parameters, with the number of readings held fixed (same sensor count and number of times as the base design).

Design variables (any subset, `design_optimisation.variables`):
    positions   sensor positions r / L in [0, 1] (one per sensor of the base design)
    t1          record length [s] in bounds.t1
    spacing     time-spacing exponent g in bounds.spacing: t_j = t0 + (t1 - t0) u_j^g, u_j uniform in [0, 1]
                (g < 1: readings crowded early, g > 1: late; g = 1: the linear base design)
    q_vol       known source power [W/m^3] in bounds.q_vol (an experimental choice when the heater is adjustable)

Criteria (on the bounds at identifiability.reference_level):
    max_crlb    minimise the worst relative bound over the unknowns (smoothed by a log-sum-exp, sharpness 50)
    d_optimal   maximise log det F (volume of the confidence ellipsoid)
    a_optimal   minimise the sum of squared relative bounds
Bounded variables are mapped through a logistic function, so L-BFGS-B works on unconstrained coordinates; multi-start
(Sobol starts plus the base design), best result kept.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import minimize

from aerosynapse.thermal.data.design import sobol_unit
from aerosynapse.thermal.inverse import body as B
from aerosynapse.thermal.inverse import identifiability as ID
from aerosynapse.thermal.inverse import spec as IS

VARIABLES = ("positions", "t1", "spacing", "q_vol")
CRITERIA = ("max_crlb", "d_optimal", "a_optimal")
_SHARP = 50.0


def _logistic(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _logit(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return np.log(u / (1 - u))


class Designer:
    def __init__(self, spec: dict):
        self.spec = spec
        d = spec["design_optimisation"]
        self.vars = list(d["variables"])
        self.bounds = {k: [float(v) for v in d["bounds"][k]] for k in ("t1", "spacing", "q_vol") if k in self.vars}
        self.criterion = d["criterion"]
        self.unknown = list(spec["unknown"])
        self.level = float(spec["identifiability"]["reference_level"])
        self.n_sens = len(IS.sensor_positions(spec))
        self.p = {k: float(spec["parameters"][k]) for k in B.PARAMS}
        self.n_eval = 0

    # --- design <-> unconstrained coordinates --------------------------------------------------------------------
    def decode(self, z: np.ndarray) -> dict:
        out, i = {}, 0
        if "positions" in self.vars:
            out["positions"] = np.sort(_logistic(z[: self.n_sens])).tolist()
            i = self.n_sens
        for k in ("t1", "spacing", "q_vol"):
            if k in self.vars:
                lo, hi = self.bounds[k]
                u = float(_logistic(z[i]))
                out[k] = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))  # log-uniform map
                i += 1
        return out

    def encode(self, d: dict) -> np.ndarray:
        z = []
        if "positions" in self.vars:
            z += list(_logit(np.asarray(d["positions"], float)))
        for k in ("t1", "spacing", "q_vol"):
            if k in self.vars:
                lo, hi = self.bounds[k]
                z.append(float(_logit(np.array([(np.log(d[k]) - np.log(lo)) / (np.log(hi) - np.log(lo))]))[0]))
        return np.asarray(z, float)

    def base_design(self) -> dict:
        s, m = self.spec, self.spec["measurement"]
        d = {}
        if "positions" in self.vars:
            d["positions"] = (IS.sensor_positions(s) / float(s["body"]["L"])).tolist()
        if "t1" in self.vars:
            d["t1"] = float(IS.measurement_times(s).max())
        if "spacing" in self.vars:
            d["spacing"] = float(m["times"].get("spacing", 1.0))
        if "q_vol" in self.vars:
            d["q_vol"] = float(s["body"]["q_vol"])
        return d

    def spec_of(self, d: dict) -> dict:
        over: dict = {"measurement": {"sensors": {}, "times": {}}, "body": {}}
        if "positions" in d:
            over["measurement"]["sensors"]["positions"] = [float(v) for v in d["positions"]]
        if "t1" in d:
            over["measurement"]["times"]["t1"] = float(d["t1"])
        if "spacing" in d:
            over["measurement"]["times"]["spacing"] = float(d["spacing"])
        if "q_vol" in d:
            over["body"]["q_vol"] = float(d["q_vol"])
        # a design read from the data file is first written out explicitly (same sensors, same number of times)
        base = IS.deep_merge(self.spec, IS.explicit_design(self.spec, keep_time_grid=not ("t1" in d or "spacing" in d)))
        return IS.deep_merge(base, over)

    # --- evaluation ----------------------------------------------------------------------------------------------
    def evaluate(self, d: dict) -> dict:
        s = self.spec_of(d)
        body, xs, ts = IS.body_of(s), IS.sensor_positions(s), IS.measurement_times(s)
        n = IS.n_terms(s)
        unknown = self.unknown

        def fn(pp: dict) -> np.ndarray:
            return B.temperatures(pp, body, xs, ts, "analytic", n)

        J = ID.jacobian(
            lambda q: fn({**self.p, **q}), {k: self.p[k] for k in unknown}, unknown, float(s["jacobian"]["rel_step"])
        )
        T = fn(self.p)
        r = ID.analyse(J, IS.noise_sigma(s, T, self.level), unknown, float(s["identifiability"]["rank_tol"]))
        self.n_eval += 1
        return r

    def objective(self, z: np.ndarray) -> float:
        r = self.evaluate(self.decode(z))
        c = np.asarray(r["crlb_log"], float)
        if not np.all(np.isfinite(c)):
            return 1e6
        if self.criterion == "max_crlb":
            lc = np.log(c)
            return float(np.log(np.sum(np.exp(_SHARP * (lc - lc.max())))) / _SHARP + lc.max())
        if self.criterion == "a_optimal":
            return float(np.log(np.sum(c**2)))
        sign, logdet = np.linalg.slogdet(np.asarray(r["fisher"], float))
        return float(-logdet) if sign > 0 else 1e6

    def optimise(self, starts: int, seed: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        base = self.base_design()
        z0s = [self.encode(base)]
        dim = len(z0s[0])
        for u in sobol_unit(max(starts - 1, 0), dim, seed) if starts > 1 else []:
            z0s.append(_logit(0.05 + 0.9 * u))
        runs = []
        for z0 in z0s:
            res = minimize(self.objective, z0, method="L-BFGS-B", options={"maxiter": 200, "eps": 1e-6})
            runs.append({"z": res.x, "f": float(res.fun), "nit": int(res.nit), "success": bool(res.success)})
        best = min(runs, key=lambda r: r["f"])
        d_opt = self.decode(best["z"])
        return {
            "criterion": self.criterion,
            "variables": self.vars,
            "base_design": base,
            "optimal_design": d_opt,
            "base": self.evaluate(base),
            "optimal": self.evaluate(d_opt),
            "starts": [{"f": r["f"], "nit": r["nit"], "success": r["success"]} for r in runs],
            "n_evaluations": self.n_eval,
            "wall_s": time.perf_counter() - t0,
        }
