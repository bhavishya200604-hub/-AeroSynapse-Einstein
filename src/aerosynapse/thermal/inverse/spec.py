"""Phase 5 settings: configs/inverse/defaults.yaml + optional user files + --set overrides, validated; and the
measurement design (sensor positions, times, noise standard deviations) built from them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aerosynapse.thermal.data.noise import noise_scale
from aerosynapse.thermal.data.sensors import layout_1d
from aerosynapse.thermal.data.spec import _read, deep_merge, parse_set  # noqa: F401  (parse_set re-exported)
from aerosynapse.thermal.inverse.body import PARAMS, SHAPES, Body, truncation_terms
from aerosynapse.thermal.utils.experiment import ROOT

DEFAULT_PATH = ROOT / "configs" / "inverse" / "defaults.yaml"
LAYOUTS = ("uniform", "random", "clustered_left", "clustered_right")
NOISE_MODELS = ("gaussian", "relative", "gaussian_range")


def _err(path: str, msg: str) -> ValueError:
    return ValueError(f"invalid setting at '{path}': {msg}")


def _pos(v: Any, path: str) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or not np.isfinite(v) or v <= 0:
        raise _err(path, f"must be a positive number, got {v!r}")
    return float(v)


def load_spec(user: str | Path | list | None = None, overrides: dict | None = None) -> dict:
    files = [DEFAULT_PATH] + ([] if user is None else [user] if isinstance(user, str | Path) else list(user))
    spec: dict = {}
    for f in files:
        spec = deep_merge(spec, _read(f))
    if overrides:
        spec = deep_merge(spec, overrides)
    _resolve_auto_level(spec)
    validate(spec)
    return spec


def read_columns(path: str | Path) -> dict[str, np.ndarray]:
    """Columns of a measured-data file: CSV with a header line ('#' comment lines removed first) or NPZ with the same
    array names (x [m], t [s], T, optional sigma). Relative paths are taken from the project root."""
    import io

    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"data.file '{path}' not found")
    if p.suffix.lower() == ".npz":
        z = np.load(p)
        return {k: np.atleast_1d(np.asarray(z[k], float)) for k in z.files}
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    arr = np.genfromtxt(io.StringIO("\n".join(lines)), delimiter=",", names=True, dtype=float)
    return {k: np.atleast_1d(np.asarray(arr[k], float)) for k in arr.dtype.names}


_DESIGN_CACHE: dict = {}


def data_design(spec: dict) -> dict[str, Any]:
    """The measurement design of the data file: distinct sensor positions [m] and times [s] (used when
    measurement.sensors.from_data / measurement.times.from_data are true), and whether the readings form the full
    positions x times grid that the design-based steps assume."""
    f = (spec.get("data") or {}).get("file")
    if not f:
        raise ValueError("measurement ... from_data needs data.file")
    p = Path(f) if Path(f).is_absolute() else ROOT / f
    key = (str(p), p.stat().st_mtime if p.exists() else None)
    if key not in _DESIGN_CACHE:
        c = read_columns(f)
        xs, ts = np.unique(c["x"]), np.unique(c["t"])
        pairs = len({(a, b) for a, b in zip(c["x"].tolist(), c["t"].tolist(), strict=True)})
        _DESIGN_CACHE[key] = {
            "positions": xs,
            "times": ts,
            "n_readings": len(c["x"]),
            "full_grid": pairs == len(xs) * len(ts),
            "sigma_mean": float(np.mean(c["sigma"])) if "sigma" in c else None,
        }
    return _DESIGN_CACHE[key]


def _resolve_auto_level(spec: dict) -> None:
    """data.level: auto -> the noise level that reproduces the file's mean sigma under the configured noise model."""
    d = spec.get("data") or {}
    if d.get("level") != "auto":
        return
    if not d.get("file"):
        raise _err("data.level", "'auto' needs data.file with a sigma column")
    c = read_columns(d["file"])
    if "sigma" not in c:
        raise _err("data.level", "'auto' needs a sigma column in data.file (or give the level as a number)")
    b = spec["body"]
    dT = float(b["T0"]) - float(b["T_inf"])
    theta = (c["T"] - float(b["T_inf"])) / dT
    model = spec["measurement"]["noise"]["model"]
    if model == "gaussian":
        lv = float(np.mean(c["sigma"])) / (float(np.sqrt(np.mean(theta**2))) * abs(dT))
    elif model == "gaussian_range":
        lv = float(np.mean(c["sigma"])) / (float(np.ptp(theta)) * abs(dT))
    else:
        lv = float(np.median(c["sigma"] / np.maximum(np.abs(theta * dT), 1e-12)))
    spec["data"]["level"] = lv
    spec["data"]["level_from_file"] = True


def validate(spec: dict) -> None:
    s = spec["body"]
    if s.get("shape", "slab") not in SHAPES:
        raise _err("body.shape", f"one of {list(SHAPES)}")
    _pos(s["L"], "body.L")
    for k in ("T0", "T_inf", "q_vol"):
        if isinstance(s[k], bool) or not isinstance(s[k], int | float):
            raise _err(f"body.{k}", "must be a number")
    if s["T0"] == s["T_inf"]:
        raise _err("body.T0", "must differ from body.T_inf")
    if s["q_vol"] < 0:
        raise _err("body.q_vol", "must be >= 0")
    for n in PARAMS:
        _pos(spec["parameters"][n], f"parameters.{n}")
    m = spec["measurement"]
    se = m["sensors"]
    if se.get("positions") is not None:
        pos = se["positions"]
        if not pos or not all(isinstance(v, int | float) and 0 <= v <= 1 for v in pos):
            raise _err("measurement.sensors.positions", "a non-empty list of fractions of L in [0, 1]")
    else:
        if se["layout"] not in LAYOUTS:
            raise _err("measurement.sensors.layout", f"one of {LAYOUTS}")
        if not isinstance(se["n"], int) or se["n"] < 1:
            raise _err("measurement.sensors.n", "a positive integer")
    t = m["times"]
    if t["kind"] == "list":
        v = t.get("values") or []
        if not v or any(x <= 0 for x in v) or list(v) != sorted(v):
            raise _err("measurement.times.values", "a non-empty increasing list of positive times")
    elif t["kind"] in ("linear", "geom"):
        if not 0 < float(t["t0"]) < float(t["t1"]) or not isinstance(t["n"], int) or t["n"] < 1:
            raise _err("measurement.times", "need 0 < t0 < t1 and an integer n >= 1")
    else:
        raise _err("measurement.times.kind", "linear, geom or list")
    if t["kind"] == "linear" and not 0 < float(t.get("spacing", 1.0)) <= 10:
        raise _err("measurement.times.spacing", "an exponent in (0, 10] (1 = uniform)")
    nz = m["noise"]
    if nz["model"] not in NOISE_MODELS:
        raise _err("measurement.noise.model", f"one of {NOISE_MODELS}")
    if not nz["levels"] or any(lv <= 0 for lv in nz["levels"]):
        raise _err("measurement.noise.levels", "a non-empty list of levels > 0 (the bound is 0 for noise-free data)")
    f = spec["forward"]
    if f["model"] not in ("analytic", "fdm"):
        raise _err("forward.model", "analytic or fdm")
    if f["n_terms"] != "auto" and (not isinstance(f["n_terms"], int) or f["n_terms"] < 1):
        raise _err("forward.n_terms", "auto or a positive integer")
    _pos(spec["jacobian"]["rel_step"], "jacobian.rel_step")
    _pos(spec["identifiability"]["rank_tol"], "identifiability.rank_tol")
    _pos(spec["identifiability"]["reference_level"], "identifiability.reference_level")
    th = spec["identifiability"].get("thresholds", [10.0, 50.0])
    if len(th) != 2 or not 0 < th[0] < th[1]:
        raise _err("identifiability.thresholds", "[well, weak] with 0 < well < weak (percent)")
    unk = spec.get("unknown") or []
    if not unk or any(k not in PARAMS for k in unk) or len(set(unk)) != len(unk):
        raise _err("unknown", f"a non-empty list of distinct names from {list(PARAMS)}")
    for k in unk:
        _pos((spec.get("initial_guess") or {}).get(k), f"initial_guess.{k}")
    d = spec.get("data") or {}
    if d:
        if d["source"] not in ("synthetic", "file"):
            raise _err("data.source", "synthetic or file")
        if d["source"] == "file" and not d.get("file"):
            raise _err("data.file", "a CSV / NPZ path when data.source = file")
        _pos(d["level"], "data.level")
    m = spec.get("measurement") or {}
    for part in ("sensors", "times"):
        if (m.get(part) or {}).get("from_data") and not d.get("file"):
            raise _err(f"measurement.{part}.from_data", "true needs data.file (the design is read from your data)")
    pn = spec.get("pinn") or {}
    if pn:
        for k in ("n_pde", "n_bc"):
            if not isinstance(pn["collocation"][k], int) or pn["collocation"][k] < 1:
                raise _err(f"pinn.collocation.{k}", "a positive integer")
        for k, v in pn["weights"].items():
            if k not in ("pde", "bc", "data") or not isinstance(v, int | float) or v < 0:
                raise _err(f"pinn.weights.{k}", "keys pde / bc / data with values >= 0")
        if pn["balancing"]["method"] not in ("fixed", "grad_balance"):
            raise _err("pinn.balancing.method", "fixed or grad_balance")
        if pn.get("time_coordinate", "linear") not in ("linear", "sqrt"):
            raise _err("pinn.time_coordinate", "linear or sqrt")
    do = spec.get("design_optimisation") or {}
    if do:
        from aerosynapse.thermal.inverse.design import CRITERIA, VARIABLES  # noqa: PLC0415

        if do["criterion"] not in CRITERIA:
            raise _err("design_optimisation.criterion", f"one of {CRITERIA}")
        if not do["variables"] or any(v not in VARIABLES for v in do["variables"]):
            raise _err("design_optimisation.variables", f"a non-empty list from {VARIABLES}")
        for k in ("t1", "spacing", "q_vol"):
            if k in do["variables"]:
                b = do["bounds"].get(k)
                if not b or len(b) != 2 or not 0 < float(b[0]) < float(b[1]):
                    raise _err(f"design_optimisation.bounds.{k}", "[lo, hi] with 0 < lo < hi")
        if not isinstance(do["starts"], int) or do["starts"] < 1:
            raise _err("design_optimisation.starts", "a positive integer")
    cl = spec.get("classical") or {}
    if cl:
        if cl["model"] not in ("fdm", "analytic"):
            raise _err("classical.model", "fdm or analytic")
        if cl["method"] not in ("trf", "lm", "dogbox"):
            raise _err("classical.method", "trf, lm or dogbox")
        if cl["jac"] not in ("2-point", "3-point"):
            raise _err("classical.jac", "2-point or 3-point")
        _pos(cl["diff_step"], "classical.diff_step")
        ms = cl["multistart"]
        if (
            not isinstance(ms["n"], int)
            or ms["n"] < 1
            or float(ms["spread"]) < 1
            or float(ms.get("chi2_tol", 0.1)) <= 0
        ):
            raise _err("classical.multistart", "n >= 1, spread >= 1 and chi2_tol > 0")
        mc = cl["monte_carlo"]
        if (
            not isinstance(mc["replicates"], int)
            or mc["replicates"] < 0
            or not isinstance(mc["batch"], int)
            or mc["batch"] < 1
        ):
            raise _err("classical.monte_carlo", "replicates >= 0 and batch >= 1 (integers)")
    st = spec.get("studies") or {}
    for c in (st.get("classical") or {}).get("cases") or []:
        if c not in ("base", "optimised_design", "no_source", "slab", "cylinder", "sphere"):
            raise _err(
                "studies.classical.cases", "entries from base, optimised_design, no_source, slab, cylinder, sphere"
            )
    uq = spec.get("uq") or {}
    if uq:
        if uq["model"] not in ("analytic", "fdm"):
            raise _err("uq.model", "analytic or fdm")
        if any(isinstance(v, bool) or not isinstance(v, int | float) or not 0 < v < 1 for v in uq["levels"]):
            raise _err("uq.levels", "interval levels in (0, 1)")
        bs, mc = uq["bootstrap"], uq["mcmc"]
        if not isinstance(bs["replicates"], int) or bs["replicates"] < 2:
            raise _err("uq.bootstrap.replicates", "an integer >= 2")
        if any(k not in ("parametric", "residual") for k in bs["kinds"]):
            raise _err("uq.bootstrap.kinds", "entries from parametric, residual")
        for key in ("steps", "burn", "adapt_every"):
            if not isinstance(mc[key], int) or mc[key] < 1:
                raise _err(f"uq.mcmc.{key}", "a positive integer")
        if float(mc["prior_factor"]) <= 1:
            raise _err("uq.mcmc.prior_factor", "a number > 1")
    su = st.get("uq") or {}
    if su:
        for key in ("datasets", "bootstrap_replicates", "batch"):
            if not isinstance(su[key], int) or su[key] < (
                0 if key == "datasets" else 2 if key == "bootstrap_replicates" else 1
            ):
                raise _err(f"studies.uq.{key}", "an integer (datasets >= 0, bootstrap_replicates >= 2, batch >= 1)")
        if not su.get("cases"):
            raise _err("studies.uq.cases", "at least one case (name: overrides)")
    cf = st.get("confounding") or {}
    if cf:
        _pos(cf["level"], "studies.confounding.level")
        pc = cf["profile"]
        if float(pc["span"]) <= 1 or any(not isinstance(pc[k], int) or pc[k] < 1 for k in ("fine", "coarse")):
            raise _err("studies.confounding.profile", "span > 1 and integer fine, coarse >= 1")
        for key in ("starts", "batch"):
            if not isinstance(cf[key], int) or cf[key] < 1:
                raise _err(f"studies.confounding.{key}", "a positive integer")
        if not isinstance(cf["replicates"], int) or cf["replicates"] < 0:
            raise _err("studies.confounding.replicates", "an integer >= 0")
        for name, unk in (cf.get("known") or {}).items():
            if not unk or any(k not in PARAMS for k in unk):
                raise _err(f"studies.confounding.known.{name}", f"a non-empty list of names from {list(PARAMS)}")
        pr = cf.get("priors")
        if pr and (pr["parameter"] not in PARAMS or any(float(v) <= 0 for v in pr["sd"])):
            raise _err("studies.confounding.priors", f"parameter from {list(PARAMS)} and sd > 0")
        for i, unk in enumerate((cf.get("pinn") or {}).get("unknown_sets") or []):
            if not unk or any(k not in PARAMS for k in unk):
                raise _err(f"studies.confounding.pinn.unknown_sets[{i}]", f"names from {list(PARAMS)}")
    rb = st.get("robustness") or {}
    if rb:
        for key in ("replicates", "batch"):
            if not isinstance(rb[key], int) or rb[key] < (0 if key == "replicates" else 1):
                raise _err(f"studies.robustness.{key}", "an integer (replicates >= 0, batch >= 1)")
        if rb["mc_model"] not in ("analytic", "fdm"):
            raise _err("studies.robustness.mc_model", "analytic or fdm")
        _pos(rb["zero_noise_weight_level"], "studies.robustness.zero_noise_weight_level")
        _pos(rb["stall_sigmas"], "studies.robustness.stall_sigmas")
        levels = list(rb["noise"]["levels"]) + [rb["sensors"]["level"], rb["placement"]["level"]]
        levels += list((rb.get("pinn") or {}).get("levels") or [])
        if any(isinstance(v, bool) or not isinstance(v, int | float) or v < 0 for v in levels):
            raise _err("studies.robustness", "noise levels must be numbers >= 0")
        for d in rb["noise"]["designs"]:
            if d not in ("base", "optimised_design"):
                raise _err("studies.robustness.noise.designs", "entries from base, optimised_design")
        if any(not isinstance(n, int) or n < 1 for n in rb["sensors"]["n"]):
            raise _err("studies.robustness.sensors.n", "positive integers")
        for name, ov in (rb["placement"]["layouts"] or {}).items():
            pos = (ov or {}).get("positions")
            if pos is not None and any(not 0.0 <= float(x) <= 1.0 for x in pos):
                raise _err(f"studies.robustness.placement.layouts.{name}.positions", "fractions of L in [0, 1]")
    pi = st.get("pinn_inverse") or {}
    names = []
    for i, v in enumerate(pi.get("variants") or []):
        if not isinstance(v.get("name"), str) or v.get("mode", "joint") not in ("joint", "staged", "two_stage"):
            raise _err(
                f"studies.pinn_inverse.variants[{i}]",
                "need {name, mode: joint | staged | two_stage[, weights, balancing, time_coordinate]}",
            )
        if (v.get("balancing") or {}).get("method", "fixed") not in ("fixed", "grad_balance"):
            raise _err(f"studies.pinn_inverse.variants[{i}].balancing.method", "fixed or grad_balance")
        names.append(v["name"])
    if len(set(names)) != len(names):
        raise _err("studies.pinn_inverse.variants", "variant names must be unique")
    sv = pi.get("shape_variant", "auto")
    if pi and sv != "auto" and sv not in names:
        raise _err("studies.pinn_inverse.shape_variant", f"auto or one of the variant names {names}")
    for sec in ("robustness", "uq", "confounding"):
        rp = (st.get(sec) or {}).get("pinn")
        if rp and rp.get("variant") not in names:
            raise _err(f"studies.{sec}.pinn.variant", f"one of the studies.pinn_inverse variant names {names}")
    for sh in pi.get("shapes") or []:
        if sh not in SHAPES:
            raise _err("studies.pinn_inverse.shapes", f"entries from {list(SHAPES)}")
    if pi and (not pi.get("seeds") or not all(isinstance(x, int) for x in pi["seeds"])):
        raise _err("studies.pinn_inverse.seeds", "a non-empty list of integers")
    for k in st.get("sweeps") or {}:
        if k not in ("h", "q_vol", "n_sensors", "layout", "t1", "alpha", "k"):
            raise _err(f"studies.sweeps.{k}", "one of h, q_vol, n_sensors, layout, t1, alpha, k")
    mc = st.get("monte_carlo") or {}
    if mc and (not isinstance(mc.get("replicates"), int) or mc["replicates"] < 0):
        raise _err("studies.monte_carlo.replicates", "an integer >= 0")


def body_of(spec: dict) -> Body:
    s = spec["body"]
    return Body(s.get("shape", "slab"), float(s["L"]), float(s["T0"]), float(s["T_inf"]), float(s["q_vol"]))


def sensor_positions(spec: dict) -> np.ndarray:
    """Sensor positions [m], sorted. Precedence: explicit positions (kept as given: coinciding sensors = replicate
    readings, as an optimised design may choose) > the data file's positions (sensors.from_data) > layout and n
    (plus the optional surface / centre sensors; de-duplicated)."""
    se, L = spec["measurement"]["sensors"], float(spec["body"]["L"])
    if se.get("positions") is not None:
        return np.sort(np.asarray(se["positions"], float)) * L
    if se.get("from_data"):
        return data_design(spec)["positions"].copy()
    else:
        frac = layout_1d(se["layout"], int(se["n"]), int(spec["seed"]), float(se.get("margin", 0.0)))
        extra = ([1.0] if se.get("include_surface") else []) + ([0.0] if se.get("include_centre") else [])
        frac = np.concatenate([frac, extra])
    return np.unique(frac) * L


def measurement_times(spec: dict) -> np.ndarray:
    t = spec["measurement"]["times"]
    if t.get("from_data"):
        return data_design(spec)["times"].copy()
    if t["kind"] == "list":
        return np.asarray(t["values"], float)
    if t["kind"] == "geom":
        return np.geomspace(float(t["t0"]), float(t["t1"]), int(t["n"]))
    u = np.linspace(0.0, 1.0, int(t["n"])) ** float(t.get("spacing", 1.0))  # spacing 1 = uniform (the default)
    return float(t["t0"]) + (float(t["t1"]) - float(t["t0"])) * u


def explicit_design(spec: dict, keep_time_grid: bool = True) -> dict:
    """Overrides that replace a design read from the data file (from_data) by the same design written out
    explicitly, so code that varies one part of the design (sweeps, design optimisation) can do so. With
    keep_time_grid=False the times become a linear grid t0..t1 with the same count (for t1 / spacing variations)."""
    m = spec["measurement"]
    over: dict = {"measurement": {}}
    if m["sensors"].get("from_data") and m["sensors"].get("positions") is None:
        frac = (sensor_positions(spec) / float(spec["body"]["L"])).tolist()
        over["measurement"]["sensors"] = {"from_data": False, "positions": frac}
    if m["times"].get("from_data"):
        ts = measurement_times(spec)
        over["measurement"]["times"] = (
            {"from_data": False, "kind": "list", "values": ts.tolist()}
            if keep_time_grid
            else {
                "from_data": False,
                "kind": "linear",
                "t0": float(ts.min()),
                "t1": float(ts.max()),
                "n": len(ts),
                "spacing": 1.0,
            }
        )
    return over


def n_terms(spec: dict) -> int:
    f = spec["forward"]
    if f["n_terms"] != "auto":
        return int(f["n_terms"])
    t_min = float(measurement_times(spec).min()) * float(spec["parameters"]["alpha"]) / float(spec["body"]["L"]) ** 2
    return max(truncation_terms(t_min), 20)


def noise_sigma(spec: dict, T_clean: np.ndarray, level: float) -> np.ndarray:
    """Per-reading noise standard deviation [K] (Phase 2 noise models applied to theta, scaled by T0 - T_inf)."""
    s = spec["body"]
    dT = float(s["T0"]) - float(s["T_inf"])
    theta = (np.asarray(T_clean) - float(s["T_inf"])) / dT
    return noise_scale(theta, level, spec["measurement"]["noise"]["model"]) * abs(dT)


def fdm_settings(spec: dict) -> dict:
    """FDM forward settings with a dimensional step dt_seconds = dt_fourier x L^2 / alpha_true: the step COUNTS are
    then the same for every parameter set (smooth model output)."""
    f = dict(spec["forward"]["fdm"])
    f["dt_seconds"] = float(f["dt_fourier"]) * float(spec["body"]["L"]) ** 2 / float(spec["parameters"]["alpha"])
    return f


def pinn_settings(spec: dict, variant: dict | None = None) -> dict:
    """Inverse-PINN settings: the network merged over the Phase 3/4 network defaults (all architecture options), plus
    a study variant's weights / balancing overrides."""
    from aerosynapse.thermal.pinn import spec as PS  # noqa: PLC0415

    p = deep_merge(
        spec["pinn"],
        {
            k: v
            for k, v in (variant or {}).items()
            if k in ("weights", "balancing", "adam", "lbfgs", "network", "collocation", "time_coordinate", "staged")
        },
    )
    p["network"] = deep_merge(PS.load_spec(None, "smoke")["network"], p["network"])
    return p
