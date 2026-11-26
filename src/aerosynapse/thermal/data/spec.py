"""User-definable conditions: layered spec (defaults <- user YAML <- profile <- --set overrides) with validation.

Layering order (later wins; dictionaries are deep-merged, lists/scalars replaced):
    configs/data/defaults.yaml  <  user file(s)  <  profile block (`profiles.<name>` of defaults AND of user files)
    <  command-line `--set a.b.c=value` overrides.
Every validation problem is reported with the dotted path of the offending key.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

from aerosynapse.thermal.data.design import Dim
from aerosynapse.thermal.utils.experiment import ROOT

DEFAULT_PATH = ROOT / "configs" / "data" / "defaults.yaml"
ALLOWED_PARAMS = {"caseA": ["Pe"], "caseB": ["Bi"], "caseC": ["Re", "Pr", "Bi", "Ri"]}
CASE_REQUIRED_SWEPT = {"caseA": ["Pe"], "caseB": ["Bi"], "caseC": []}
SCHEMES = ["upwind1", "central", "upwind2", "quick", "tvd_minmod", "tvd_vanleer", "tvd_superbee", "tvd_vanalbada"]
VEL_PROFILES = ["zero", "uniform", "cosh", "parabolic", "expr"]
THERMAL_TYPES = ["dirichlet", "neumann", "robin"]
WALLS = ["top", "bottom", "left", "right"]
TIME_KINDS = ["linear", "geom", "list"]


def deep_merge(base: dict, over: dict, replace: tuple[str, ...] = ()) -> dict:
    """Recursive merge (returns a new dict). Dict values under a key listed in ``replace`` REPLACE the base value
    instead of merging (e.g. a user-given parameter range must not inherit the default's explicit ``values``)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and v.get("_replace") is True:
            # a user dict marked `_replace: true` REPLACES the default one (e.g. your own case list instead of the
            # default studies' cases, which a key-by-key merge could only add to)
            out[k] = copy.deepcopy({kk: vv for kk, vv in v.items() if kk != "_replace"})
        elif isinstance(v, dict) and isinstance(out.get(k), dict) and k not in replace:
            out[k] = deep_merge(out[k], v, replace)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_path(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
        if not isinstance(cur, dict):
            raise ValueError(f"--set {dotted}: '{k}' is not a section")
    cur[keys[-1]] = value


def parse_set(items: list[str]) -> dict:
    """['cases.caseC.grid=48', 'noise.levels=[0,0.05]'] -> nested dict (values parsed as YAML)."""
    out: dict = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        set_path(out, k.strip(), coerce_numbers(yaml.safe_load(v)))
    return out


_SCI = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def coerce_numbers(v: Any) -> Any:
    """PyYAML (YAML 1.1) reads 3e-3 / 1e+30 as STRINGS (a dot is required); turn such scientific-notation strings
    into floats, recursively, so users can write the natural form in config files and --set values."""
    if isinstance(v, str) and _SCI.match(v.strip()):
        return float(v)
    if isinstance(v, dict):
        return {k: coerce_numbers(x) for k, x in v.items()}
    if isinstance(v, list):
        return [coerce_numbers(x) for x in v]
    return v


def _read(path: str | Path) -> dict:
    from aerosynapse.thermal._home import resolve

    path = resolve(path)
    with open(path, encoding="utf-8") as fh:
        return coerce_numbers(yaml.safe_load(fh) or {})


def load_spec(user: str | Path | list | None = None, profile: str = "smoke", overrides: dict | None = None) -> dict:
    """Merge defaults, user file(s), profile blocks and overrides, then validate. Returns the resolved spec."""
    files = [DEFAULT_PATH] + ([] if user is None else [user] if isinstance(user, str | Path) else list(user))
    spec: dict = {}
    profiles: list[dict] = []
    for f in files:
        d = _read(f)
        profiles.append(d.pop("profiles", {}) or {})
        spec = deep_merge(spec, d)
    for p in profiles:  # profile blocks apply AFTER all base layers so a user can tune a profile too
        if profile in p:
            spec = deep_merge(spec, p[profile] or {})
        elif p and profile not in p:
            raise ValueError(f"profile {profile!r} not defined; available: {sorted(p)}")
    if overrides:
        spec = deep_merge(spec, overrides)
    spec["profile"] = profile
    validate(spec)
    return spec


# --- validation ---------------------------------------------------------------------------------
def _err(path: str, msg: str) -> ValueError:
    return ValueError(f"invalid condition at '{path}': {msg}")


def _num(v: Any, path: str) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise _err(path, f"expected a number, got {v!r}")
    return float(v)


def case_dims(ccfg: dict, case: str) -> tuple[list[Dim], dict[str, float]]:
    """Split a case's `parameters` into swept dims (dicts) and fixed values (numbers)."""
    dims: list[Dim] = []
    fixed: dict[str, float] = {}
    for name, v in ccfg["parameters"].items():
        p = f"cases.{case}.parameters.{name}"
        if isinstance(v, dict):
            unknown = set(v) - {"lo", "hi", "scale"}
            if unknown:
                raise _err(p, f"unknown keys {sorted(unknown)} (allowed: lo, hi, scale)")
            try:
                dims.append(
                    Dim(name, _num(v.get("lo"), p + ".lo"), _num(v.get("hi"), p + ".hi"), v.get("scale", "log"))
                )
            except ValueError as e:
                if "invalid condition" in str(e):
                    raise
                raise _err(p, str(e)) from None
        else:
            fixed[name] = _num(v, p)
    return dims, fixed


def validate_times(t: dict, path: str) -> None:
    kind = t.get("kind")
    if kind not in TIME_KINDS:
        raise _err(path + ".kind", f"must be one of {TIME_KINDS}, got {kind!r}")
    if kind == "list":
        vals = t.get("values")
        if not isinstance(vals, list) or not vals or sorted(vals) != list(vals) or vals[0] <= 0:
            raise _err(path + ".values", "need a non-empty increasing list of positive times")
        return
    t0, t1, n = _num(t.get("t0"), path + ".t0"), _num(t.get("t1"), path + ".t1"), t.get("n")
    if not (0 < t0 < t1):
        raise _err(path, f"need 0 < t0 < t1, got t0={t0}, t1={t1}")
    if not isinstance(n, int) or n < 1:
        raise _err(path + ".n", "need an integer >= 1")


def validate(spec: dict) -> None:
    d = spec.get("design", {})
    if d.get("method") not in ("sobol", "lhs"):
        raise _err("design.method", f"must be sobol or lhs, got {d.get('method')!r}")
    if not 0.2 <= _num(d.get("unit_train_limit"), "design.unit_train_limit") < 1.0:
        raise _err("design.unit_train_limit", "must be in [0.2, 1)")
    from aerosynapse.thermal.data.noise import MODELS

    if spec["noise"]["model"] not in MODELS:
        raise _err("noise.model", f"must be one of {list(MODELS)}")
    for i, lv in enumerate(spec["noise"]["levels"]):
        if _num(lv, f"noise.levels[{i}]") < 0:
            raise _err(f"noise.levels[{i}]", "must be >= 0")
    s = spec["sensors"]
    for k in ("counts_1d", "counts_2d"):
        if not all(isinstance(c, int) and c > 0 for c in s[k]):
            raise _err(f"sensors.{k}", "need positive integers")
    from aerosynapse.thermal.data.sensors import LAYOUTS_1D, LAYOUTS_2D

    if s["layouts"] is not None:
        bad = [x for x in s["layouts"] if x not in LAYOUTS_1D + LAYOUTS_2D]
        if bad:
            raise _err("sensors.layouts", f"unknown {bad}; 1D: {list(LAYOUTS_1D)}, 2D: {list(LAYOUTS_2D)}")
    for name, c in spec["cases"].items():
        if name not in ALLOWED_PARAMS:
            raise _err(f"cases.{name}", f"unknown case; available: {list(ALLOWED_PARAMS)}")
        if not c.get("enabled", True):
            continue
        base = f"cases.{name}"
        unknown = set(c["parameters"]) - set(ALLOWED_PARAMS[name])
        if unknown:
            raise _err(base + ".parameters", f"unknown {sorted(unknown)}; allowed: {ALLOWED_PARAMS[name]}")
        dims, fixed = case_dims(c, name)
        swept = [x.name for x in dims]
        for req in CASE_REQUIRED_SWEPT[name]:
            if req not in swept:
                raise _err(base + f".parameters.{req}", "this parameter must be swept ({lo, hi, scale}) for this case")
        if not dims:
            raise _err(base + ".parameters", "at least one parameter must be swept")
        for n, v in fixed.items():
            if n in ("Re", "Pr", "Pe", "Bi") and v <= 0:
                raise _err(base + f".parameters.{n}", f"must be > 0, got {v}")
        sz = c["sizes"]
        for k in ("train", "val", "interp", "extrap"):
            if not isinstance(sz.get(k), int) or sz[k] < 0:
                raise _err(base + f".sizes.{k}", "need an integer >= 0")
        if sz["train"] < 1:
            raise _err(base + ".sizes.train", "need at least 1 training point")
        validate_times(c["times"], base + ".times")
        if name == "caseC":
            _validate_case_c(c, base, set(swept) | set(fixed))
        else:
            if not isinstance(c.get("nx"), int) or c["nx"] < 5:
                raise _err(base + ".nx", "need an integer >= 5")
    if not any(c.get("enabled", True) for c in spec["cases"].values()):
        raise _err("cases", "no case enabled")


def _validate_case_c(c: dict, base: str, pnames: set[str]) -> None:
    if not isinstance(c.get("grid"), int) or c["grid"] < 8 or c["grid"] % 2:
        raise _err(base + ".grid", "need an even integer >= 8 (cells per direction)")
    if c.get("scheme") not in SCHEMES:
        raise _err(base + ".scheme", f"must be one of {SCHEMES}")
    if c.get("bc_order") not in (1, 2):
        raise _err(base + ".bc_order", "must be 1 or 2")
    cond = c["conditions"]
    _num(cond.get("initial_theta", 0.0), base + ".conditions.initial_theta")
    for w in WALLS:
        v = cond["velocity"].get(w, {"profile": "zero"})
        p = f"{base}.conditions.velocity.{w}"
        if v.get("profile") not in VEL_PROFILES:
            raise _err(p + ".profile", f"must be one of {VEL_PROFILES}")
        if v["profile"] == "expr":
            _check_expr(v.get("expr"), p + ".expr")
        if v["profile"] == "cosh":
            _num(v.get("C", 50.0), p + ".C")
        if v["profile"] != "zero":
            _num(v.get("amplitude", 1.0), p + ".amplitude")
        t = cond["thermal"].get(w)
        if t is None:
            raise _err(f"{base}.conditions.thermal.{w}", "every wall needs a thermal condition")
        p = f"{base}.conditions.thermal.{w}"
        if t.get("type") not in THERMAL_TYPES:
            raise _err(p + ".type", f"must be one of {THERMAL_TYPES}")
        if t["type"] in ("dirichlet", "neumann"):
            _num(t.get("value", 0.0), p + ".value")
        else:
            b = t.get("Bi")
            if isinstance(b, str):
                if b not in pnames:
                    raise _err(p + ".Bi", f"refers to parameter {b!r} which is not defined in `parameters`")
            else:
                if _num(b, p + ".Bi") <= 0:
                    raise _err(p + ".Bi", "must be > 0")
            _num(t.get("theta_inf", 0.0), p + ".theta_inf")
    thermal_has_hot = (
        any(
            t["type"] == "dirichlet" or t.get("theta_inf", 0) != 0 or t.get("value", 0) != 0
            for t in cond["thermal"].values()
        )
        or cond.get("initial_theta", 0.0) != 0
    )
    if not thermal_has_hot:
        raise _err(
            base + ".conditions.thermal", "all walls adiabatic/zero and initial_theta = 0: the solution is trivially 0"
        )


_SAFE = {"__builtins__": {}}


def _check_expr(expr: Any, path: str) -> None:
    if not isinstance(expr, str):
        raise _err(path, "expr profile needs a string expression in s and t")
    import numpy as np

    try:
        v = eval(expr, _SAFE | _np_names(), {"s": np.linspace(0, 1, 5), "t": 0.0})  # noqa: S307  (local user config)
        np.broadcast_to(np.asarray(v, float), (5,))
    except Exception as e:  # noqa: BLE001
        raise _err(path, f"cannot evaluate {expr!r}: {e}") from None


def _np_names() -> dict:
    import numpy as np

    return {
        k: getattr(np, k)
        for k in (
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "sqrt",
            "abs",
            "tanh",
            "cosh",
            "sinh",
            "pi",
            "where",
            "minimum",
            "maximum",
            "clip",
        )
    }


def eval_profile(vel: dict, s, t: float):
    """Evaluate a wall-velocity profile spec at positions s (0..1 along the wall) and time t."""
    import numpy as np

    kind = vel.get("profile", "zero")
    a = float(vel.get("amplitude", 1.0))
    s = np.asarray(s, float)
    if kind == "zero":
        return np.zeros_like(s)
    if kind == "uniform":
        return a * np.ones_like(s)
    if kind == "cosh":
        C = float(vel.get("C", 50.0))
        return a * (1.0 - np.cosh(C * (s - 0.5)) / np.cosh(0.5 * C))
    if kind == "parabolic":
        return a * 4.0 * s * (1.0 - s)
    return a * np.broadcast_to(np.asarray(eval(vel["expr"], _SAFE | _np_names(), {"s": s, "t": t}), float), s.shape)  # noqa: S307


def describe(spec: dict) -> str:
    """Human-readable one-screen summary of the resolved conditions."""
    lines = [f"profile={spec['profile']} seed={spec['seed']} design={spec['design']} noise={spec['noise']}"]
    for n, c in spec["cases"].items():
        if not c.get("enabled", True):
            lines.append(f"  {n}: disabled")
            continue
        dims, fixed = case_dims(c, n)
        lines.append(
            f"  {n}: swept {[f'{d.name}[{d.lo:g},{d.hi:g}] {d.scale}' for d in dims]} fixed {fixed} "
            f"sizes {c['sizes']} times {c['times']}"
        )
    return "\n".join(lines)
