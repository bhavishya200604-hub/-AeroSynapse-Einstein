"""User-definable PINN settings: layered spec (defaults < user YAML < profile < --set) with validation.

Reuses the layering helpers of the data spec. Problem parameter grids may be given as explicit ``values`` or as a
range ``{lo, hi, n, scale}`` and are expanded by :func:`param_values`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aerosynapse.thermal.data.spec import _read, deep_merge, parse_set  # noqa: F401  (parse_set re-exported)
from aerosynapse.thermal.utils.experiment import ROOT

DEFAULT_PATH = ROOT / "configs" / "pinn" / "defaults.yaml"
ACTIVATIONS = ["tanh", "sin", "silu", "gelu"]
PROBLEMS = {"A_steady": "Pe", "A_pulse": "Pe", "B_slab": "Bi"}
SCHEDULES = ["constant", "cosine", "exp"]
REPLACE = ("parameter",)  # a user's parameter block replaces the default one (values vs lo/hi/n must not mix)


def _err(path: str, msg: str) -> ValueError:
    return ValueError(f"invalid setting at '{path}': {msg}")


def _num(v: Any, path: str) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise _err(path, f"expected a number, got {v!r}")
    return float(v)


def _int(v: Any, path: str, lo: int = 0) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise _err(path, f"expected an integer >= {lo}, got {v!r}")
    return v


def param_values(pcfg: dict) -> list[float]:
    """Explicit list or {lo, hi, n, scale} -> list of floats."""
    if "values" in pcfg:
        return [float(v) for v in pcfg["values"]]
    lo, hi, n = float(pcfg["lo"]), float(pcfg["hi"]), int(pcfg["n"])
    grid = np.geomspace(lo, hi, n) if pcfg.get("scale", "log") == "log" else np.linspace(lo, hi, n)
    return [float(v) for v in grid]


def param_range(pcfg: dict) -> tuple[float, float]:
    v = param_values(pcfg)
    return min(v), max(v)


def load_spec(user: str | Path | list | None = None, profile: str = "smoke", overrides: dict | None = None) -> dict:
    files = [DEFAULT_PATH] + ([] if user is None else [user] if isinstance(user, str | Path) else list(user))
    spec: dict = {}
    profiles: list[dict] = []
    for f in files:
        d = _read(f)
        profiles.append(d.pop("profiles", {}) or {})
        spec = deep_merge(spec, d, REPLACE)
    for p in profiles:
        if profile in p:
            spec = deep_merge(spec, p[profile] or {}, REPLACE)
        elif p and profile not in p:
            raise ValueError(f"profile {profile!r} not defined; available: {sorted(p)}")
    if overrides:
        spec = deep_merge(spec, overrides, REPLACE)
    spec["profile"] = profile
    for name, p in spec.get(
        "problems", {}
    ).items():  # the parameter name is fixed per problem: users need not repeat it
        if name in PROBLEMS and isinstance(p.get("parameter"), dict):
            p["parameter"].setdefault("name", PROBLEMS[name])
    validate(spec)
    return spec


ARCHS = ["mlp", "fourier", "modified_mlp", "resnet", "rwf", "siren"]


def validate_network(n: dict, base: str) -> None:
    if n.get("arch", "mlp") not in ARCHS:
        raise _err(f"{base}.arch", f"must be one of {ARCHS}")
    _int(n["width"], f"{base}.width", 2)
    _int(n["depth"], f"{base}.depth", 1)
    dr = _num(n.get("dropout", 0.0), f"{base}.dropout")
    # studies merge the main network block into their own entries, so the mlp-only rule is checked on `network` only
    if not 0.0 <= dr < 1.0 or (dr > 0 and base == "network" and n.get("arch", "mlp") != "mlp"):
        raise _err(f"{base}.dropout", "a rate in [0, 1); > 0 only with arch mlp (Step 6.2 MC-dropout)")
    if n["activation"] not in ACTIVATIONS:
        raise _err(f"{base}.activation", f"must be one of {ACTIVATIONS}")
    if n["init"] not in ("xavier", "kaiming"):
        raise _err(f"{base}.init", "must be xavier or kaiming")
    f = n["fourier"]
    sig = f["sigma"]
    for i, v in enumerate(sig if isinstance(sig, list) else [sig]):
        if _num(v, f"{base}.fourier.sigma[{i}]") <= 0:
            raise _err(f"{base}.fourier.sigma", "values must be > 0")
    _int(f["n_features"], f"{base}.fourier.n_features", 1)
    for k in ("trainable", "include_input"):
        if not isinstance(f[k], bool):
            raise _err(f"{base}.fourier.{k}", "true or false")
    if n["resnet"]["block_layers"] not in (1, 2):
        raise _err(f"{base}.resnet.block_layers", "must be 1 or 2")
    _num(n["rwf"]["mu"], f"{base}.rwf.mu")
    if _num(n["rwf"]["sigma"], f"{base}.rwf.sigma") < 0:
        raise _err(f"{base}.rwf.sigma", "must be >= 0")
    if _num(n["siren"]["omega0"], f"{base}.siren.omega0") <= 0:
        raise _err(f"{base}.siren.omega0", "must be > 0")


def network_variant(spec: dict, entry: Any, default_label: str | None = None) -> tuple[str, dict]:
    """(label, network block) for an architecture entry: a name, or {name, arch, <overrides>}."""
    base = spec["network"]
    if isinstance(entry, str):
        net = deep_merge(base, {"arch": entry})
        return entry, net
    over = {k: v for k, v in entry.items() if k != "name"}
    net = deep_merge(base, over)
    return str(entry.get("name", entry.get("arch", default_label or "variant"))), net


def validate_arch_study(spec: dict) -> None:
    a = spec["arch_study"]
    known = spec["problems"]

    def case_ok(c: dict, path: str) -> None:
        if c.get("problem") not in known:
            raise _err(path + ".problem", f"must be one of {list(known)}")
        if _num(c.get("value"), path + ".value") <= 0:
            raise _err(path + ".value", "must be > 0")

    if not a["cases"]:
        raise _err("arch_study.cases", "need at least one case")
    for i, c in enumerate(a["cases"]):
        case_ok(c, f"arch_study.cases[{i}]")
        check_case_data(spec, c, f"arch_study.cases[{i}]")
    if not a["architectures"]:
        raise _err("arch_study.architectures", "need at least one architecture")
    for i, e in enumerate(a["architectures"]):
        _, net = network_variant(spec, e)
        validate_network(net, f"arch_study.architectures[{i}]")
    if not a["seeds"] or not all(isinstance(s, int) for s in a["seeds"]):
        raise _err("arch_study.seeds", "need a non-empty list of integers")
    sw = a["sigma_sweep"]
    if sw.get("enabled", False):
        case_ok(sw["case"], "arch_study.sigma_sweep.case")
        if not sw["values"] or any(_num(v, "arch_study.sigma_sweep.values") <= 0 for v in sw["values"]):
            raise _err("arch_study.sigma_sweep.values", "need a non-empty list of positive numbers")


USER_DATA_ROLES = ("evaluate", "train", "both")
USER_DATA_FORMATS = ("auto", "csv", "npz", "h5", "phase2")


def validate_user_data(spec: dict) -> None:
    src = (spec.get("user_data") or {}).get("sources") or {}
    if not isinstance(src, dict):
        raise _err("user_data.sources", "a mapping {name: {file, role, columns, ...}}")
    for n, c in src.items():
        base = f"user_data.sources.{n}"
        if not isinstance(c, dict) or not isinstance(c.get("file"), str):
            raise _err(base + ".file", "need a file path (relative to the project root or absolute)")
        if c.get("format", "auto") not in USER_DATA_FORMATS:
            raise _err(base + ".format", f"one of {list(USER_DATA_FORMATS)}")
        if c.get("role", "evaluate") not in USER_DATA_ROLES:
            raise _err(base + ".role", f"one of {list(USER_DATA_ROLES)}")
        for k in ("columns", "scale", "select"):
            if c.get(k) is not None and not isinstance(c[k], dict):
                raise _err(f"{base}.{k}", "a mapping")
        for k, v in (c.get("columns") or {}).items():
            if not isinstance(v, str):
                raise _err(f"{base}.columns.{k}", "the column name in the file (a string)")
        for k, v in (c.get("scale") or {}).items():
            _num(v, f"{base}.scale.{k}")
        _int(c.get("max_points", 0) or 0, base + ".max_points", 0)


def check_case_data(spec: dict, case: Any, path: str) -> None:
    if not isinstance(case, dict) or not case.get("data"):
        return
    names = case["data"] if isinstance(case["data"], list) else [case["data"]]
    known = (spec.get("user_data") or {}).get("sources") or {}
    for n in names:
        if n not in known:
            raise _err(path + ".data", f"'{n}' is not defined under user_data.sources (defined: {sorted(known)})")


def check_user_files(spec: dict, cases: list) -> None:
    """When a study RUNS, every attached data file must exist (not checked at load time: unused sources never block)."""
    known = (spec.get("user_data") or {}).get("sources") or {}
    for c in cases:
        if not isinstance(c, dict) or not c.get("data"):
            continue
        for n in c["data"] if isinstance(c["data"], list) else [c["data"]]:
            f = Path(known[n]["file"])
            f = f if f.is_absolute() else ROOT / f
            if not f.exists():
                raise _err(f"user_data.sources.{n}.file", f"{f} does not exist")


def validate_execution(spec: dict) -> None:
    e = spec["execution"]
    if not isinstance(e.get("isolate"), bool):
        raise _err("execution.isolate", "true or false")
    if e.get("workers") != "auto":
        _int(e.get("workers"), "execution.workers", 1)
    _int(e.get("threads_per_worker", 0), "execution.threads_per_worker", 0)


def case_name(entry: Any) -> str:
    """Study case entries are a problem name or a mapping {problem, value?, formulation?, data?}."""
    return entry if isinstance(entry, str) else entry["problem"]


PROBLEMS2D = {
    "C_mms": ("Re", "Pr", "Ri"),
    "C_cavity": ("Re", "Pr", "Bi", "Ri"),
    "D_graetz": ("Re", "Pr", "Bi", "Ri"),
    "D_obstacle": ("Re", "Pr", "Bi", "Ri"),
}
CHANNEL = ("D_graetz", "D_obstacle")
FORMULATION_KEYS = ("kind", "hard_bc", "pressure_gauge", "normalise")
RESIDUALS2D = ("mom_x", "mom_y", "cont", "energy")


def formulation_variant(spec: dict, entry: dict) -> tuple[str, dict]:
    """(label, formulation block) for a study variant {name, <formulation overrides>}."""
    over = {k: v for k, v in entry.items() if k != "name"}
    return str(entry.get("name", "variant")), deep_merge(spec["formulation"], over)


def validate_formulation(f: dict, base: str) -> None:
    for k in f:
        if k not in FORMULATION_KEYS:
            raise _err(f"{base}.{k}", f"unknown key; allowed {list(FORMULATION_KEYS)}")
    if f["kind"] not in ("primitive", "streamfunction"):
        raise _err(f"{base}.kind", "must be primitive or streamfunction")
    if not isinstance(f["hard_bc"], bool):
        raise _err(f"{base}.hard_bc", "true or false")
    g = f["pressure_gauge"]
    if g["kind"] not in ("none", "point", "mean"):
        raise _err(f"{base}.pressure_gauge.kind", "must be none, point or mean")
    if g["mode"] not in ("soft", "hard"):
        raise _err(f"{base}.pressure_gauge.mode", "must be soft or hard")
    if g["kind"] == "mean" and g["mode"] == "hard":
        raise _err(f"{base}.pressure_gauge.mode", "a hard gauge is available for kind=point only (use mode=soft)")
    for k in ("x", "y"):
        if not 0.0 <= _num(g[k], f"{base}.pressure_gauge.{k}") <= 1.0:
            raise _err(f"{base}.pressure_gauge.{k}", "must lie in [0, 1]")
    _num(g["value"], f"{base}.pressure_gauge.value")
    nm = f["normalise"]
    if nm["kind"] not in ("none", "init_rms", "manual"):
        raise _err(f"{base}.normalise.kind", "must be none, init_rms or manual")
    for k in RESIDUALS2D:
        if _num(nm["scales"].get(k), f"{base}.normalise.scales.{k}") <= 0:
            raise _err(f"{base}.normalise.scales.{k}", "must be > 0")


def validate_channel_problem(p: dict, base: str) -> None:
    g = p["geometry"]
    Lx, Ly = _num(g.get("Lx"), base + ".geometry.Lx"), _num(g.get("Ly"), base + ".geometry.Ly")
    if Lx <= 0 or Ly <= 0:
        raise _err(base + ".geometry", "Lx, Ly must be > 0")
    o = g.get("obstacle")
    if o:
        x0, x1, y0, y1 = (_num(o.get(k), f"{base}.geometry.obstacle.{k}") for k in ("x0", "x1", "y0", "y1"))
        _num(o.get("theta"), base + ".geometry.obstacle.theta")
        if not (0 < x0 < x1 < Lx and 0 < y0 < y1 < Ly):
            raise _err(base + ".geometry.obstacle", "need 0 < x0 < x1 < Lx and 0 < y0 < y1 < Ly")
        if o.get("psi", "auto") not in ("auto", "symmetric", "learned", "soft"):
            raise _err(
                base + ".geometry.obstacle.psi",
                "auto, symmetric, learned or soft (stream-function value on the obstacle; soft = obstacle not exact)",
            )
    swept = 0
    for k, v in p["parameters"].items():
        if k not in ("Re", "Pr", "Bi", "Ri"):
            raise _err(f"{base}.parameters.{k}", "one of Re, Pr, Bi, Ri")
        if isinstance(v, dict):
            swept += 1
            lo, hi = _num(v.get("lo"), f"{base}.parameters.{k}.lo"), _num(v.get("hi"), f"{base}.parameters.{k}.hi")
            sc = v.get("scale", "log")
            if sc not in ("log", "linear") or not lo < hi or (sc == "log" and lo <= 0):
                raise _err(f"{base}.parameters.{k}", "need lo < hi, scale log (lo > 0) or linear")
        else:
            _num(v, f"{base}.parameters.{k}")
    if "Re" not in p["parameters"] or "Pr" not in p["parameters"]:
        raise _err(base + ".parameters", "need Re and Pr (swept or fixed)")
    if p["inlet"]["velocity"].get("profile") not in ("zero", "uniform", "cosh", "parabolic", "expr"):
        raise _err(base + ".inlet.velocity.profile", "zero, uniform, cosh, parabolic or expr")
    _num(p["inlet"]["theta"], base + ".inlet.theta")
    for w in ("bottom", "top"):
        t = p["thermal"][w]
        if t.get("type") not in ("dirichlet", "neumann", "robin"):
            raise _err(f"{base}.thermal.{w}.type", "dirichlet, neumann or robin")
        if t["type"] == "robin" and isinstance(t.get("Bi"), str) and t["Bi"] not in p["parameters"]:
            raise _err(f"{base}.thermal.{w}.Bi", f"names {t['Bi']!r}, not in parameters")
    r = p["reference"]
    _int(r["cells_per_unit"], base + ".reference.cells_per_unit", 4)
    _int(r["refine"], base + ".reference.refine", 2)
    if _num(r["tol"], base + ".reference.tol") <= 0:
        raise _err(base + ".reference.tol", "must be > 0")
    if r["scheme"] not in ("upwind1", "central", "upwind2", "quick"):
        raise _err(base + ".reference.scheme", "a linear scheme: upwind1, central, upwind2 or quick")
    if o:
        for k in ("x0", "x1", "y0", "y1"):
            for n in (r["cells_per_unit"], r["cells_per_unit"] * r["refine"]):
                if abs(o[k] * n - round(o[k] * n)) > 1e-9:
                    raise _err(f"{base}.geometry.obstacle.{k}", f"must lie on the reference grid lines (h = 1/{n})")


def validate_channel_study(spec: dict) -> None:
    cs = spec["channel_study"]
    if not cs["cases"] or any(c not in CHANNEL or c not in spec["problems2d"] for c in cs["cases"]):
        raise _err("channel_study.cases", f"a non-empty list of {list(CHANNEL)}")
    f = deep_merge(spec["formulation"], cs["formulation"] or {})
    validate_formulation(f, "channel_study.formulation")
    if f["hard_bc"] and f["kind"] != "streamfunction":
        raise _err(
            "channel_study.formulation.hard_bc",
            "hard constraints for channel geometries need kind: streamfunction (exact psi on inlet, walls, obstacle)",
        )
    if f["hard_bc"]:
        from aerosynapse.thermal.pinn.channel2d import obstacle_psi_mode  # noqa: PLC0415  (spec is imported by channel2d's callers)

        for c in cs["cases"]:
            p = spec["problems2d"][c]
            if p["geometry"].get("obstacle"):
                pr = p["parameters"]
                fixed = {k: v for k, v in pr.items() if not isinstance(v, dict)}
                try:
                    obstacle_psi_mode(
                        p["geometry"], p["inlet"], fixed, [k for k, v in pr.items() if isinstance(v, dict)]
                    )
                except ValueError as e:
                    raise _err(f"problems2d.{c}.geometry.obstacle.psi", str(e)) from None
    d = cs["design"]
    for s in ("train", "interp", "extrap"):
        _int(d["sizes"].get(s, 0), f"channel_study.design.sizes.{s}", 0)
    if not 0 < _num(d["u_lim"], "channel_study.design.u_lim") < 1:
        raise _err("channel_study.design.u_lim", "must be in (0, 1)")
    names = []
    for i, v in enumerate(cs["variants"] or []):
        path = f"channel_study.variants[{i}]"
        if not isinstance(v, dict) or not isinstance(v.get("name"), str):
            raise _err(path, "need {name, kind, <sampling / training / loss / network overrides>}")
        for k in v:
            if k not in PARAM_VARIANT_KEYS:
                raise _err(f"{path}.{k}", f"unknown key; allowed {list(PARAM_VARIANT_KEYS)}")
        if v.get("kind") not in ("parametric", "per_point"):
            raise _err(f"{path}.kind", "parametric or per_point")
        names.append(v["name"])
    if not names or len(set(names)) != len(names):
        raise _err("channel_study.variants", f"need unique variant names, got {names}")
    if not cs["seeds"] or not all(isinstance(s, int) for s in cs["seeds"]):
        raise _err("channel_study.seeds", "need a non-empty list of integers")
    for c, srcs in (cs.get("data") or {}).items():
        if c not in cs["cases"]:
            raise _err(f"channel_study.data.{c}", "not one of channel_study.cases")
        check_case_data(spec, {"data": srcs}, f"channel_study.data.{c}")


def validate_2d(spec: dict) -> None:
    validate_formulation(spec["formulation"], "formulation")
    for name, p in spec["problems2d"].items():
        base = f"problems2d.{name}"
        if name not in PROBLEMS2D:
            raise _err(base, f"unknown 2D problem; available {list(PROBLEMS2D)}")
        if name in CHANNEL:
            validate_channel_problem(p, base)
            continue
        for k in PROBLEMS2D[name]:
            v = _num(p["parameters"].get(k), f"{base}.parameters.{k}")
            if k != "Ri" and v <= 0:
                raise _err(f"{base}.parameters.{k}", "must be > 0")
        if name == "C_mms":
            for k in ("psi", "p", "theta"):
                if not isinstance(p["exact"].get(k), str | int | float):
                    raise _err(f"{base}.exact.{k}", "need an expression string in x and y")
            for w in ("left", "right", "bottom", "top"):
                ab = p["thermal_bc"].get(w)
                if not isinstance(ab, list) or len(ab) != 2:
                    raise _err(f"{base}.thermal_bc.{w}", "need [alpha, beta]")
                a, b = (_num(q, f"{base}.thermal_bc.{w}") for q in ab)
                if a == 0 and b == 0:
                    raise _err(f"{base}.thermal_bc.{w}", "alpha and beta cannot both be 0")
        else:
            c = p["conditions"]
            if c != "data" and not (isinstance(c, dict) and "velocity" in c and "thermal" in c):
                raise _err(f"{base}.conditions", "'data' or a block with velocity and thermal entries")
            if isinstance(c, dict):
                for w, t in c["thermal"].items():
                    if t.get("type") not in ("dirichlet", "neumann", "robin"):
                        raise _err(f"{base}.conditions.thermal.{w}.type", "dirichlet, neumann or robin")
                    if t["type"] == "robin" and isinstance(t.get("Bi"), str) and t["Bi"] not in p["parameters"]:
                        raise _err(f"{base}.conditions.thermal.{w}.Bi", f"names {t['Bi']!r}, not in parameters")
            r = p["reference"]
            _int(r["grid"], f"{base}.reference.grid", 8)
            _int(r["refine"], f"{base}.reference.refine", 2)
            if _num(r["tol"], f"{base}.reference.tol") <= 0:
                raise _err(f"{base}.reference.tol", "must be > 0")
    fs = spec["formulation_study"]
    if not fs["cases"]:
        raise _err("formulation_study.cases", f"non-empty list of {list(spec['problems2d'])}")
    for i, c in enumerate(fs["cases"]):
        if not isinstance(c, str | dict) or case_name(c) not in spec["problems2d"]:
            raise _err(
                f"formulation_study.cases[{i}]",
                f"a name or {{problem, data}} with problem in {list(spec['problems2d'])}",
            )
        check_case_data(spec, c, f"formulation_study.cases[{i}]")
    if not fs["variants"]:
        raise _err("formulation_study.variants", "need at least one variant")
    names = []
    for i, e in enumerate(fs["variants"]):
        if not isinstance(e, dict):
            raise _err(f"formulation_study.variants[{i}]", "need a mapping {name: ..., <formulation overrides>}")
        label, f = formulation_variant(spec, e)
        validate_formulation(f, f"formulation_study.variants[{i}]")
        names.append(label)
    if len(set(names)) != len(names):
        raise _err("formulation_study.variants", f"variant names must be unique, got {names}")
    if not fs["seeds"] or not all(isinstance(s, int) for s in fs["seeds"]):
        raise _err("formulation_study.seeds", "need a non-empty list of integers")
    ev = fs["evaluation"]
    _int(ev["n_grid"], "formulation_study.evaluation.n_grid", 9)
    _int(ev["n_val"], "formulation_study.evaluation.n_val", 16)
    _int(ev["n_wall"], "formulation_study.evaluation.n_wall", 8)


BALANCING = ("fixed", "lr_annealing", "gradnorm", "ntk", "softadapt", "relobralo", "uncertainty", "self_adaptive")


def balancing_variant(spec: dict, entry: Any) -> tuple[str, dict]:
    """(label, balancing block) for a study entry: a method name, or {name, method, <method-block overrides>}."""
    base = spec["loss"]["balancing"]
    if isinstance(entry, str):
        return entry, deep_merge(base, {"method": entry})
    over = {k: v for k, v in entry.items() if k != "name"}
    return str(entry.get("name", entry.get("method", "variant"))), deep_merge(base, over)


def validate_balancing(b: dict, base: str, resample_every: int) -> None:
    m = b.get("method")
    if m not in BALANCING:
        raise _err(f"{base}.method", f"must be one of {list(BALANCING)}")
    wb = b["weight_bounds"]
    if not isinstance(wb, list) or len(wb) != 2:
        raise _err(f"{base}.weight_bounds", "need [lo, hi] with 0 < lo < hi")
    if not 0 < _num(wb[0], f"{base}.weight_bounds") < _num(wb[1], f"{base}.weight_bounds"):
        raise _err(f"{base}.weight_bounds", "need [lo, hi] with 0 < lo < hi")
    for k in ("lr_annealing", "gradnorm", "ntk", "softadapt", "relobralo"):
        _int(b[k]["every"], f"{base}.{k}.every", 1)
    la = b["lr_annealing"]
    if not 0 < _num(la["alpha"], f"{base}.lr_annealing.alpha") <= 1:
        raise _err(f"{base}.lr_annealing.alpha", "must be in (0, 1]")
    if la["statistic"] not in ("max", "mean"):
        raise _err(f"{base}.lr_annealing.statistic", "max or mean")
    if la["reference"] not in ("pde", "bc", "ic"):
        raise _err(f"{base}.lr_annealing.reference", "pde, bc or ic")
    if _num(b["gradnorm"]["lr"], f"{base}.gradnorm.lr") <= 0:
        raise _err(f"{base}.gradnorm.lr", "must be > 0")
    _num(b["gradnorm"]["alpha"], f"{base}.gradnorm.alpha")
    _int(b["ntk"]["n_points"], f"{base}.ntk.n_points", 1)
    if b["ntk"]["trace"] not in ("mean", "sum"):
        raise _err(f"{base}.ntk.trace", "mean or sum")
    sa = b["softadapt"]
    _num(sa["beta"], f"{base}.softadapt.beta")
    _int(sa["window"], f"{base}.softadapt.window", 2)
    for k in ("normalized", "loss_weighted"):
        if not isinstance(sa[k], bool):
            raise _err(f"{base}.softadapt.{k}", "true or false")
    rl = b["relobralo"]
    for k in ("alpha", "rho"):
        if not 0 <= _num(rl[k], f"{base}.relobralo.{k}") <= 1:
            raise _err(f"{base}.relobralo.{k}", "must be in [0, 1]")
    if _num(rl["temperature"], f"{base}.relobralo.temperature") <= 0:
        raise _err(f"{base}.relobralo.temperature", "must be > 0")
    if _num(b["uncertainty"]["lr"], f"{base}.uncertainty.lr") <= 0:
        raise _err(f"{base}.uncertainty.lr", "must be > 0")
    _num(b["uncertainty"]["init_log_var"], f"{base}.uncertainty.init_log_var")
    s = b["self_adaptive"]
    if _num(s["lr"], f"{base}.self_adaptive.lr") <= 0:
        raise _err(f"{base}.self_adaptive.lr", "must be > 0")
    init = s["init"]
    vals = init if isinstance(init, list) else [init]
    if len(vals) not in (1, 2) or any(_num(v, f"{base}.self_adaptive.init") < 0 for v in vals):
        raise _err(f"{base}.self_adaptive.init", "a number >= 0 or [lo, hi] with 0 <= lo")
    if s["mask"] not in ("square", "identity", "softplus"):
        raise _err(f"{base}.self_adaptive.mask", "square, identity or softplus")
    if m == "self_adaptive" and resample_every != 0:
        raise _err(
            f"{base}.method",
            "self_adaptive keeps one weight per collocation point, so the points must stay fixed: "
            "set sampling.resample_every=0",
        )


def validate_balance_study(spec: dict) -> None:
    st = spec["balance_study"]
    if not st["cases"]:
        raise _err("balance_study.cases", "need at least one case")
    for i, c in enumerate(st["cases"]):
        path = f"balance_study.cases[{i}]"
        p = c.get("problem")
        if p in PROBLEMS2D:
            if p not in spec["problems2d"]:
                raise _err(path + ".problem", f"{p} has no problems2d entry")
            f = deep_merge(spec["formulation"], c.get("formulation", {}) or {})
            validate_formulation(f, path + ".formulation")
        elif p in spec["problems"]:
            if _num(c.get("value"), path + ".value") <= 0:
                raise _err(path + ".value", "must be > 0")
        else:
            raise _err(path + ".problem", f"must be one of {list(spec['problems']) + list(PROBLEMS2D)}")
        check_case_data(spec, c, path)
    if not st["methods"]:
        raise _err("balance_study.methods", "need at least one method")
    labels = []
    for i, e in enumerate(st["methods"]):
        label, b = balancing_variant(spec, e)
        validate_balancing(b, f"balance_study.methods[{i}]", 0)  # resampling rule: check_balance_study_runnable
        labels.append(label)
    if len(set(labels)) != len(labels):
        raise _err("balance_study.methods", f"labels must be unique, got {labels}")
    if not st["seeds"] or not all(isinstance(s, int) for s in st["seeds"]):
        raise _err("balance_study.seeds", "need a non-empty list of integers")


def check_balance_study_runnable(spec: dict) -> None:
    """Checks that only matter when the Step 4.3 study is actually run (not for every spec that is loaded)."""
    for i, e in enumerate(spec["balance_study"]["methods"]):
        _, b = balancing_variant(spec, e)
        validate_balancing(b, f"balance_study.methods[{i}]", spec["sampling"]["resample_every"])


STRATEGY_KEYS = ("name", "sampling", "training", "loss", "network")


def validate_strategy_settings(spec: dict, base: str = "", cross: bool = True) -> None:
    """Step 4.4 blocks: sampling.adaptive, training.causal / time_marching / curriculum. ``cross`` = also check the
    conflicts with other settings (resampling, self-adaptive weights, n_pde); studies check those only when they run."""
    pre = (base + ".") if base else ""
    a = spec["sampling"]["adaptive"]
    if a["method"] not in ("none", "rad", "rar_d"):
        raise _err(pre + "sampling.adaptive.method", "none, rad or rar_d")
    if a["method"] != "none":
        _int(a["period"], pre + "sampling.adaptive.period", 1)
        _int(a["n_candidates"], pre + "sampling.adaptive.n_candidates", 1)
        if cross and a["n_candidates"] < spec["sampling"]["n_pde"]:
            raise _err(pre + "sampling.adaptive.n_candidates", "must be >= sampling.n_pde")
        for k in ("k", "c"):
            if a.get(k) is not None and _num(a[k], f"{pre}sampling.adaptive.{k}") < 0:
                raise _err(f"{pre}sampling.adaptive.{k}", "must be >= 0 (or null = paper default)")
        if not 0 < _num(a["start_fraction"], pre + "sampling.adaptive.start_fraction") <= 1:
            raise _err(pre + "sampling.adaptive.start_fraction", "must be in (0, 1]")
        if cross and spec["sampling"]["resample_every"] != 0:
            raise _err(
                pre + "sampling.resample_every",
                "must be 0 with adaptive sampling (the adaptive sampler owns the points)",
            )
        if cross and spec["loss"]["balancing"]["method"] == "self_adaptive":
            raise _err(
                pre + "sampling.adaptive.method",
                "cannot be combined with loss.balancing.method=self_adaptive "
                "(its pointwise weights belong to fixed points)",
            )
    tr = spec["training"]
    c = tr["causal"]
    if not isinstance(c["enabled"], bool):
        raise _err(pre + "training.causal.enabled", "true or false")
    _int(c["n_chunks"], pre + "training.causal.n_chunks", 1)
    if not c["eps"] or any(_num(e, pre + "training.causal.eps") < 0 for e in c["eps"]):
        raise _err(pre + "training.causal.eps", "a non-empty list of numbers >= 0")
    if not 0 < _num(c["delta"], pre + "training.causal.delta") <= 1:
        raise _err(pre + "training.causal.delta", "must be in (0, 1]")
    tm = tr["time_marching"]
    if not isinstance(tm["enabled"], bool):
        raise _err(pre + "training.time_marching.enabled", "true or false")
    _int(tm["windows"], pre + "training.time_marching.windows", 1)
    cu = tr["curriculum"]
    if not isinstance(cu["enabled"], bool):
        raise _err(pre + "training.curriculum.enabled", "true or false")
    if not 0 < _num(cu["start_fraction"], pre + "training.curriculum.start_fraction") <= 1:
        raise _err(pre + "training.curriculum.start_fraction", "must be in (0, 1]")
    _int(cu["stages"], pre + "training.curriculum.stages", 1)
    if not isinstance(cu["parameter"], str):
        raise _err(pre + "training.curriculum.parameter", "auto or a parameter name (Pe, Bi, Re, ...)")
    if sum(bool(x) for x in (c["enabled"], tm["enabled"], cu["enabled"])) > 1:
        raise _err(pre + "training", "causal, time_marching and curriculum are separate strategies: enable at most one")


def validate_strategy_study(spec: dict) -> None:
    st = spec["strategy_study"]
    if not st["cases"]:
        raise _err("strategy_study.cases", "need at least one case")
    for i, c in enumerate(st["cases"]):
        path = f"strategy_study.cases[{i}]"
        p = c.get("problem") if isinstance(c, dict) else None
        if p in PROBLEMS2D:
            validate_formulation(deep_merge(spec["formulation"], c.get("formulation", {}) or {}), path + ".formulation")
        elif p in spec["problems"]:
            if _num(c.get("value"), path + ".value") <= 0:
                raise _err(path + ".value", "must be > 0")
        else:
            raise _err(path + ".problem", f"must be one of {list(spec['problems']) + list(PROBLEMS2D)}")
        check_case_data(spec, c, path)
    names = []
    for i, s in enumerate(st["strategies"] or []):
        path = f"strategy_study.strategies[{i}]"
        if not isinstance(s, dict) or not isinstance(s.get("name"), str):
            raise _err(path, "need {name: ..., <sampling / training / loss / network overrides>}")
        for k in s:
            if k not in STRATEGY_KEYS:
                raise _err(f"{path}.{k}", f"unknown key; allowed {list(STRATEGY_KEYS)}")
        merged = deep_merge(spec, {k: v for k, v in s.items() if k != "name"})
        validate_network(merged["network"], path + ".network")
        validate_strategy_settings(merged, path, cross=False)  # conflicts: check_strategy_study_runnable
        names.append(s["name"])
    if not names:
        raise _err("strategy_study.strategies", "need at least one strategy")
    if len(set(names)) != len(names):
        raise _err("strategy_study.strategies", f"names must be unique, got {names}")
    if not st["seeds"] or not all(isinstance(x, int) for x in st["seeds"]):
        raise _err("strategy_study.seeds", "need a non-empty list of integers")


def check_strategy_study_runnable(spec: dict) -> None:
    """Cross-setting checks of every strategy, made when the Step 4.4 study RUNS (not for every loaded spec)."""
    for i, s in enumerate(spec["strategy_study"]["strategies"]):
        merged = deep_merge(spec, {k: v for k, v in s.items() if k != "name"})
        validate_strategy_settings(merged, f"strategy_study.strategies[{i}]", cross=True)


def validate_hybrid_study(spec: dict) -> None:
    hs = spec.get("hybrid_study")
    if not hs:
        return
    if not hs["cases"] or any(c not in CHANNEL or c not in spec["problems2d"] for c in hs["cases"]):
        raise _err("hybrid_study.cases", f"a non-empty list of {list(CHANNEL)}")
    for c in hs.get("problems") or {}:
        if c not in CHANNEL:
            raise _err(f"hybrid_study.problems.{c}", f"one of {list(CHANNEL)}")
    for c in hs["cases"]:
        validate_channel_problem(
            deep_merge(spec["problems2d"][c], (hs.get("problems") or {}).get(c) or {}), f"problems2d.{c}"
        )
    f = deep_merge(spec["formulation"], hs["formulation"] or {})
    validate_formulation(f, "hybrid_study.formulation")
    if f["hard_bc"] and f["kind"] != "streamfunction":
        raise _err(
            "hybrid_study.formulation.hard_bc", "hard constraints for channel geometries need kind: streamfunction"
        )
    modes = hs["modes"]
    if not modes or any(m not in ("data_free", "data_only", "hybrid") for m in modes) or len(set(modes)) != len(modes):
        raise _err("hybrid_study.modes", "distinct entries of data_free, data_only, hybrid")
    d = hs["data"]
    if not d["fields"] or any(x not in ("u", "v", "theta", "p") for x in d["fields"]):
        raise _err("hybrid_study.data.fields", "a non-empty list of u, v, theta, p")
    if d["placement"] not in ("random", "sobol"):
        raise _err("hybrid_study.data.placement", "random or sobol")
    if _num(d["noise"], "hybrid_study.data.noise") < 0:
        raise _err("hybrid_study.data.noise", "must be >= 0")
    _int(d["seed"], "hybrid_study.data.seed", 0)
    pp, pa = hs["per_point"], hs["parametric"]
    if pp["enabled"]:
        _int(pp["point"], "hybrid_study.per_point.point", 0)
        ns = pp["n_data"]
        if not ns or not all(isinstance(n, int) and n > 0 for n in ns):
            raise _err("hybrid_study.per_point.n_data", "a non-empty list of positive integers")
    if pa["enabled"]:
        _int(pa["n_snapshots"], "hybrid_study.parametric.n_snapshots", 1)
        _int(pa["n_data"], "hybrid_study.parametric.n_data", 1)
        if pa["n_snapshots"] > int(spec["channel_study"]["design"]["sizes"].get("train", 0)):
            raise _err(
                "hybrid_study.parametric.n_snapshots",
                "at most channel_study.design.sizes.train (snapshots are taken at the train design points)",
            )
    if not pp["enabled"] and not pa["enabled"]:
        raise _err("hybrid_study", "per_point and parametric are both disabled: nothing to run")
    if not hs["seeds"] or not all(isinstance(s, int) for s in hs["seeds"]):
        raise _err("hybrid_study.seeds", "need a non-empty list of integers")
    for c, srcs in (hs.get("user_data") or {}).items():
        if c not in hs["cases"]:
            raise _err(f"hybrid_study.user_data.{c}", f"not in hybrid_study.cases {hs['cases']}")
        for n in [srcs] if isinstance(srcs, str) else srcs:
            if n not in ((spec.get("user_data") or {}).get("sources") or {}):
                raise _err(f"hybrid_study.user_data.{c}", f"source '{n}' is not defined under user_data.sources")


GATE_KINDS = {"ghia_cavity": ("Re", "Pr"), "graetz_cwt": ("Re", "Pr", "theta_wall", "window"), "dvd": ("Ra", "Pr")}


def validate_gate(spec: dict) -> None:
    g = spec.get("gate")
    if not g:
        return
    for i, c in enumerate(g.get("compile") or []):
        if not isinstance(c, dict) or not all(isinstance(c.get(k), str) for k in ("label", "run", "group")):
            raise _err(f"gate.compile[{i}]", "need {label, run, group[, where, values]}")
    for name, b in (g.get("benchmarks") or {}).items():
        path = f"gate.benchmarks.{name}"
        if b.get("kind") not in GATE_KINDS:
            raise _err(path + ".kind", f"one of {list(GATE_KINDS)}")
        for k in GATE_KINDS[b["kind"]]:
            if k not in b:
                raise _err(f"{path}.{k}", f"required for kind {b['kind']}")
        if b["kind"] == "ghia_cavity" and int(b["Re"]) not in (100, 400, 1000, 3200, 5000, 7500, 10000):
            raise _err(path + ".Re", "a Ghia et al. table Reynolds number (100, 400, 1000, 3200, 5000, 7500, 10000)")
        if b["kind"] == "dvd" and float(b["Ra"]) not in (1e4, 1e5, 1e6):
            raise _err(path + ".Ra", "1e4, 1e5 or 1e6 (the verified de Vahl Davis values)")
        if b["kind"] == "graetz_cwt":
            w = b["window"]
            if not (
                isinstance(w, list)
                and len(w) == 2
                and 0 <= w[0] < w[1] <= float(spec["problems2d"]["D_graetz"]["geometry"]["Lx"])
            ):
                raise _err(path + ".window", "[x_lo, x_hi] inside the channel")
        if b.get("formulation"):
            validate_formulation(deep_merge(spec["formulation"], b["formulation"]), path + ".formulation")
    r = g["reference"]
    _int(r["grid"], "gate.reference.grid", 4)
    _int(r["refine"], "gate.reference.refine", 2)
    if not g["seeds"] or not all(isinstance(s, int) for s in g["seeds"]):
        raise _err("gate.seeds", "a non-empty list of integers")


ABLATION_KEYS = ("formulation", "training", "sampling", "loss", "network", "data")


def validate_ablation_study(spec: dict) -> None:
    ab = spec.get("ablation_study")
    if not ab:
        return
    if ab["case"] not in CHANNEL or ab["case"] not in spec["problems2d"]:
        raise _err("ablation_study.case", f"one of {list(CHANNEL)}")
    if ab["kind"] not in ("parametric", "per_point"):
        raise _err("ablation_study.kind", "parametric or per_point")
    _int(ab["point"], "ablation_study.point", 0)
    comps = ab["components"] or {}
    if not comps:
        raise _err("ablation_study.components", "need at least one component {name: overrides}")
    for name, over in comps.items():
        if not isinstance(over, dict) or not over:
            raise _err(f"ablation_study.components.{name}", "need a non-empty dict of overrides")
        for k in over:
            if k not in ABLATION_KEYS:
                raise _err(f"ablation_study.components.{name}.{k}", f"unknown key; allowed {list(ABLATION_KEYS)}")
        if "data" in over:
            d = over["data"]
            _int(d.get("n_data"), f"ablation_study.components.{name}.data.n_data", 1)
            if ab["kind"] == "parametric":
                _int(d.get("n_snapshots"), f"ablation_study.components.{name}.data.n_snapshots", 1)
                if d["n_snapshots"] > int(spec["channel_study"]["design"]["sizes"].get("train", 0)):
                    raise _err(
                        f"ablation_study.components.{name}.data.n_snapshots", "at most channel_study.design.sizes.train"
                    )
    if sum("data" in o for o in comps.values()) > 1:
        raise _err("ablation_study.components", "at most one component may carry 'data'")
    d = ab["data"]
    if not d["fields"] or any(x not in ("u", "v", "theta", "p") for x in d["fields"]):
        raise _err("ablation_study.data.fields", "a non-empty list of u, v, theta, p")
    if d["placement"] not in ("random", "sobol"):
        raise _err("ablation_study.data.placement", "random or sobol")
    if _num(d["noise"], "ablation_study.data.noise") < 0:
        raise _err("ablation_study.data.noise", "must be >= 0")
    if not ab["designs"] or any(x not in ("leave_one_out", "add_one_in") for x in ab["designs"]):
        raise _err("ablation_study.designs", "a non-empty list of leave_one_out, add_one_in")
    if not ab["seeds"] or not all(isinstance(s, int) for s in ab["seeds"]) or len(set(ab["seeds"])) != len(ab["seeds"]):
        raise _err("ablation_study.seeds", "a non-empty list of distinct integers (>= 3 for mean +- std)")
    for c in ab.get("problems") or {}:
        if c not in CHANNEL:
            raise _err(f"ablation_study.problems.{c}", f"one of {list(CHANNEL)}")


PARAM_VARIANT_KEYS = ("name", "kind", "splits", "sampling", "training", "loss", "network")


def validate_param_study(spec: dict) -> None:
    ps = spec["param_study"]
    if ps["case"] != "C_cavity":
        raise _err("param_study.case", "only C_cavity is implemented (Case D is Step 4.5b)")
    validate_formulation(deep_merge(spec["formulation"], ps["formulation"] or {}), "param_study.formulation")
    par = ps["parameters"]
    if par != "data":
        if not isinstance(par, dict) or not par:
            raise _err("param_study.parameters", "'data' or a block {Re: {lo, hi, scale}, Pr: ..., Bi: ..., Ri: 0.0}")
        swept = 0
        for k, v in par.items():
            if k not in ("Re", "Pr", "Pe", "Bi", "Ri"):
                raise _err(f"param_study.parameters.{k}", "one of Re, Pr, Pe, Bi, Ri")
            if isinstance(v, dict):
                swept += 1
                lo, hi = (
                    _num(v.get("lo"), f"param_study.parameters.{k}.lo"),
                    _num(v.get("hi"), f"param_study.parameters.{k}.hi"),
                )
                sc = v.get("scale", "log")
                if sc not in ("log", "linear") or not lo < hi or (sc == "log" and lo <= 0):
                    raise _err(f"param_study.parameters.{k}", "need lo < hi, scale log (lo > 0) or linear")
            else:
                _num(v, f"param_study.parameters.{k}")
        if swept == 0:
            raise _err("param_study.parameters", "sweep at least one parameter ({lo, hi, scale})")
        if "Re" not in par or not ("Pr" in par or "Pe" in par):
            raise _err("param_study.parameters", "need Re and Pr (or Pe), swept or fixed")
        d = ps["design"]
        for s in ("train", "interp", "extrap"):
            _int(d["sizes"].get(s, 0), f"param_study.design.sizes.{s}", 0)
        _int(d["seed"], "param_study.design.seed", 0)
        if not 0 < _num(d["u_lim"], "param_study.design.u_lim") < 1:
            raise _err("param_study.design.u_lim", "must be in (0, 1)")
    if not ps["test_splits"] or any(s not in ("train", "val", "interp", "extrap") for s in ps["test_splits"]):
        raise _err("param_study.test_splits", "a non-empty subset of train, val, interp, extrap")
    r = ps["reference"]
    _int(r["grid"], "param_study.reference.grid", 8)
    _int(r["refine"], "param_study.reference.refine", 2)
    if _num(r["tol"], "param_study.reference.tol") <= 0:
        raise _err("param_study.reference.tol", "must be > 0")
    if r["scheme"] not in ("upwind1", "central", "upwind2", "quick"):
        raise _err("param_study.reference.scheme", "a linear scheme: upwind1, central, upwind2 or quick")
    names = []
    for i, v in enumerate(ps["variants"] or []):
        path = f"param_study.variants[{i}]"
        if not isinstance(v, dict) or not isinstance(v.get("name"), str):
            raise _err(path, "need {name, kind, <sampling / training / loss / network overrides>}")
        for k in v:
            if k not in PARAM_VARIANT_KEYS:
                raise _err(f"{path}.{k}", f"unknown key; allowed {list(PARAM_VARIANT_KEYS)}")
        if v.get("kind") not in ("parametric", "per_point"):
            raise _err(f"{path}.kind", "parametric or per_point")
        merged = deep_merge(
            deep_merge(spec, ps.get("common") or {}),
            {k: x for k, x in v.items() if k in ("sampling", "training", "loss", "network")},
        )
        validate_network(merged["network"], path + ".network")
        names.append(v["name"])
    if not names:
        raise _err("param_study.variants", "need at least one variant")
    if len(set(names)) != len(names):
        raise _err("param_study.variants", f"names must be unique, got {names}")
    if not ps["seeds"] or not all(isinstance(s, int) for s in ps["seeds"]):
        raise _err("param_study.seeds", "need a non-empty list of integers")
    check_case_data(spec, {"data": ps.get("data")}, "param_study")


def validate_scaling_study(spec: dict) -> None:
    """Step 6.1 settings (scaling_study)."""
    s = spec.get("scaling_study")
    if not s:
        return
    for name, cfg in (s.get("problems") or {}).items():
        if name not in PROBLEMS:
            raise _err(f"scaling_study.problems.{name}", f"one of {sorted(PROBLEMS)}")
        _num((cfg or {}).get("value"), f"scaling_study.problems.{name}.value")
    for k, vals in (s.get("sweeps") or {}).items():
        if k not in ("width", "depth", "n_pde", "adam_epochs", "lbfgs_iterations"):
            raise _err(f"scaling_study.sweeps.{k}", "one of width, depth, n_pde, adam_epochs, lbfgs_iterations")
        if not vals or any(not isinstance(v, int) or v < (0 if k == "lbfgs_iterations" else 1) for v in vals):
            raise _err(f"scaling_study.sweeps.{k}", "a non-empty list of positive integers")
    if not s.get("seeds") or any(not isinstance(v, int) for v in s["seeds"]):
        raise _err("scaling_study.seeds", "a non-empty list of integers")
    if any(not 0 < float(v) < 1 for v in s.get("targets") or []):
        raise _err("scaling_study.targets", "relative L2 targets in (0, 1)")
    so = s.get("solvers") or {}
    for k in ("fd_N", "spectral_N"):
        if any(not isinstance(v, int) or v < 4 for v in so.get(k) or []):
            raise _err(f"scaling_study.solvers.{k}", "integers >= 4")
    if so.get("fd_solver", "splu") not in ("splu", "thomas", "auto"):
        raise _err("scaling_study.solvers.fd_solver", "one of splu, thomas, auto")
    if "fd_dt_per_h" in so:
        _num(so["fd_dt_per_h"], "scaling_study.solvers.fd_dt_per_h")
    _int(s.get("bootstrap", 0), "scaling_study.bootstrap", 0)


def validate_uq_study(spec: dict) -> None:
    """Step 6.2 settings (uq_study)."""
    s = spec.get("uq_study")
    if not s:
        return
    for name, cfg in (s.get("problems") or {}).items():
        if name not in PROBLEMS:
            raise _err(f"uq_study.problems.{name}", f"one of {sorted(PROBLEMS)}")
        r = (cfg or {}).get("range")
        if not isinstance(r, list) or len(r) != 2 or not 0 < float(r[0]) < float(r[1]):
            raise _err(f"uq_study.problems.{name}.range", "[lo, hi] with 0 < lo < hi")
    _int(s.get("n_values", 13), "uq_study.n_values", 3)
    _int(s.get("eval_nx", 201), "uq_study.eval_nx", 3)
    rv = s.get("random_values") or {}
    _int(rv.get("calibration", 48), "uq_study.random_values.calibration", 2)
    _int(rv.get("test", 48), "uq_study.random_values.test", 1)
    _int(rv.get("seed", 0), "uq_study.random_values.seed", 0)
    e = s.get("ensemble") or {}
    m = _int(e.get("members", 8), "uq_study.ensemble.members", 2)
    if any(not isinstance(k, int) or not 2 <= k <= m for k in e.get("sizes") or []):
        raise _err("uq_study.ensemble.sizes", f"integers in [2, members = {m}]")
    d = s.get("mc_dropout") or {}
    if any(not 0 < float(r) < 1 for r in d.get("rates") or []):
        raise _err("uq_study.mc_dropout.rates", "rates in (0, 1)")
    _int(d.get("samples", 100), "uq_study.mc_dropout.samples", 2)
    _int(d.get("adam_epochs", 3000), "uq_study.mc_dropout.adam_epochs", 1)
    if any(not 0 < float(v) < 1 for v in s.get("levels") or []):
        raise _err("uq_study.levels", "nominal levels in (0, 1)")
    if any(not 0 < float(v) < 1 for v in s.get("conformal_alpha") or []):
        raise _err("uq_study.conformal_alpha", "miscoverage levels in (0, 1)")


def validate_extrapolation_study(spec: dict) -> None:
    """Step 6.3 settings (extrapolation_study)."""
    s = spec.get("extrapolation_study")
    if not s:
        return
    for name, cfg in (s.get("problems") or {}).items():
        if name not in PROBLEMS:
            raise _err(f"extrapolation_study.problems.{name}", f"one of {sorted(PROBLEMS)}")
        for k in ("decades_below", "decades_above"):
            if _num((cfg or {}).get(k, 1.0), f"extrapolation_study.problems.{name}.{k}") < 0:
                raise _err(f"extrapolation_study.problems.{name}.{k}", ">= 0")
        _int((cfg or {}).get("n", 41), f"extrapolation_study.problems.{name}.n", 3)
    tm = s.get("time") or {}
    if _num(tm.get("factor", 3.0), "extrapolation_study.time.factor") < 1:
        raise _err("extrapolation_study.time.factor", ">= 1")
    _int(tm.get("n", 30), "extrapolation_study.time.n", 2)
    _int(tm.get("values", 5), "extrapolation_study.time.values", 1)
    _int((s.get("calibration") or {}).get("n", 48), "extrapolation_study.calibration.n", 2)
    if not 0 < float(s.get("threshold_quantile", 0.95)) < 1:
        raise _err("extrapolation_study.threshold_quantile", "in (0, 1)")
    if any(not 0 < float(v) for v in s.get("error_tolerances") or []):
        raise _err("extrapolation_study.error_tolerances", "positive numbers")
    c = s.get("cavity") or {}
    if c.get("enabled", False):
        if not c.get("Re") or any(float(r) <= 0 for r in c["Re"]) or list(c["Re"]) != sorted(c["Re"]):
            raise _err("extrapolation_study.cavity.Re", "an increasing list of positive Reynolds numbers")
        if not c.get("N") or any(not isinstance(n, int) or n < 8 for n in c["N"]):
            raise _err("extrapolation_study.cavity.N", "grid sizes (integers >= 8)")
        _int(c.get("n_eig", 6), "extrapolation_study.cavity.n_eig", 1)


def validate_sensitivity_study(spec: dict) -> None:
    """Step 6.4 settings (sensitivity_study)."""
    s = spec.get("sensitivity_study")
    if not s:
        return
    pow2 = lambda v: isinstance(v, int) and v >= 2 and (v & (v - 1)) == 0  # noqa: E731
    for name, c in (s.get("cases") or {}).items():
        if name not in ("D_graetz", "C_cavity"):
            raise _err(f"sensitivity_study.cases.{name}", "D_graetz or C_cavity")
        r = c.get("ranges", "train_box")
        if r != "train_box":
            for k, v in (r or {}).items():
                if (
                    k not in ("Re", "Pr", "Bi")
                    or not isinstance(v, list)
                    or len(v) != 2
                    or not 0 < float(v[0]) < float(v[1])
                ):
                    raise _err(f"sensitivity_study.cases.{name}.ranges.{k}", "Re / Pr / Bi: [lo, hi] with 0 < lo < hi")
        for blk in ("solver", "pinn"):
            if not pow2((c.get(blk) or {}).get("n")):
                raise _err(f"sensitivity_study.cases.{name}.{blk}.n", "a power of two >= 2")
        _int(c.get("pce_degree", 3), f"sensitivity_study.cases.{name}.pce_degree", 1)
    if any(not pow2(v) for v in s.get("convergence") or []):
        raise _err("sensitivity_study.convergence", "powers of two")
    _int(s.get("bootstrap", 1000), "sensitivity_study.bootstrap", 0)
    _int(s.get("chunk", 32), "sensitivity_study.chunk", 1)
    g = s.get("grid_study") or {}
    for name, c in (g.get("cases") or {}).items():
        gr = c.get("grids") or []
        if not gr or any(not isinstance(v, int) or v < 4 for v in gr) or list(gr) != sorted(gr):
            raise _err(f"sensitivity_study.grid_study.cases.{name}.grids", "an increasing list of integers >= 4")
        _int(c.get("target", 1024), f"sensitivity_study.grid_study.cases.{name}.target", 4)
    if not 0 < float(g.get("max_memory_fraction", 0.6)) <= 1:
        raise _err("sensitivity_study.grid_study.max_memory_fraction", "in (0, 1]")
    if g.get("run_target", "auto") not in ("auto", "never", "force"):
        raise _err("sensitivity_study.grid_study.run_target", "auto, never or force")
    pl = s.get("plots") or {}
    _int(pl.get("resolution", 61), "sensitivity_study.plots.resolution", 5)
    sp = pl.get("surface_pair", ["Re", "Bi"])
    if len(sp) != 2 or len(set(sp)) != 2 or any(k not in ("Re", "Pr", "Bi") for k in sp):
        raise _err("sensitivity_study.plots.surface_pair", "two different inputs of Re, Pr, Bi")


def validate_diagnostics_study(spec: dict) -> None:
    """Step 6.5 settings (diagnostics_study)."""
    s = spec.get("diagnostics_study")
    if not s:
        return
    for name, c in (s.get("cases") or {}).items():
        if (c or {}).get("problem") not in ("A_steady", "B_slab", "A_pulse"):
            raise _err(f"diagnostics_study.cases.{name}.problem", "A_steady, B_slab or A_pulse")
        if _num(c.get("value"), f"diagnostics_study.cases.{name}.value") <= 0:
            raise _err(f"diagnostics_study.cases.{name}.value", "> 0")
    if not s.get("variants"):
        raise _err("diagnostics_study.variants", "at least one variant")
    _int(s.get("snapshot_every", 50), "diagnostics_study.snapshot_every", 1)
    _int(s.get("ntk_snapshots", 8), "diagnostics_study.ntk_snapshots", 2)
    if (s.get("landscape") or {}).get("directions", "pca") not in ("pca", "random"):
        raise _err("diagnostics_study.landscape.directions", "pca or random")
    _int((s.get("landscape") or {}).get("n", 31), "diagnostics_study.landscape.n", 3)
    w = s.get("width_study") or {}
    if w and w.get("case") not in (s.get("cases") or {}):
        raise _err("diagnostics_study.width_study.case", "one of diagnostics_study.cases")
    sc = s.get("sigma_scan") or {}
    if sc and (sc.get("case") not in (s.get("cases") or {}) or any(float(v) <= 0 for v in sc.get("sigmas") or [])):
        raise _err("diagnostics_study.sigma_scan", "case from diagnostics_study.cases and positive sigmas")
    ty = s.get("toy") or {}
    if any(not isinstance(k, int) or k < 1 for k in ty.get("frequencies") or [1]):
        raise _err("diagnostics_study.toy.frequencies", "positive integers")


def validate_validation_study(spec: dict) -> None:
    """Step 6.6 settings (validation_study)."""
    s = spec.get("validation_study")
    if not s:
        return
    g = s.get("graetz") or {}
    for i, pr in enumerate(g.get("pairs") or []):
        if not isinstance(pr, list) or len(pr) != 2 or float(pr[0]) <= 0 or float(pr[1]) <= 0:
            raise _err(f"validation_study.graetz.pairs[{i}]", "[Re_H, Pr] with positive numbers")
    Lx = _num(g.get("Lx", 8.0), "validation_study.graetz.Lx")
    if Lx <= 0:
        raise _err("validation_study.graetz.Lx", "> 0")
    if not 0 <= _num(g.get("exclude_outlet", 1.0), "validation_study.graetz.exclude_outlet") < Lx:
        raise _err("validation_study.graetz.exclude_outlet", "in [0, Lx)")
    _int(g.get("cells_per_unit", 32), "validation_study.graetz.cells_per_unit", 4)
    n = s.get("natural") or {}
    if any(float(r) <= 0 for r in n.get("Ra") or []) or list(n.get("Ra") or []) != sorted(n.get("Ra") or []):
        raise _err("validation_study.natural.Ra", "an increasing list of positive Rayleigh numbers")
    if any(float(v) <= 0 for v in n.get("Pr") or []):
        raise _err("validation_study.natural.Pr", "positive")
    gr = n.get("grids") or []
    if any(not isinstance(v, int) or v < 8 for v in gr) or list(gr) != sorted(gr):
        raise _err("validation_study.natural.grids", "an increasing list of integers >= 8")


def validate_twod_study(spec: dict) -> None:
    """Step 6.6b settings (twod_study)."""
    s = spec.get("twod_study")
    if not s:
        return
    for c, cfg in (s.get("cases") or {}).items():
        if c not in ("D_graetz", "C_cavity"):
            raise _err(f"twod_study.cases.{c}", "D_graetz or C_cavity (the 2D adapters of Step 6.4)")
        if not (cfg or {}).get("surrogate", {}).get("run"):
            raise _err(f"twod_study.cases.{c}.surrogate.run", "a results directory of a saved parametric run")
    u = s.get("uq") or {}
    _int(u.get("n_points", 128), "twod_study.uq.n_points", 4)
    if any(not 0 < float(v) < 1 for v in u.get("levels") or []):
        raise _err("twod_study.uq.levels", "in (0, 1)")
    if not 0 < float(u.get("conformal_alpha", 0.1)) < 1:
        raise _err("twod_study.uq.conformal_alpha", "in (0, 1)")
    x = s.get("extrapolation") or {}
    ds = [float(v) for v in x.get("distances") or [0.0]]
    if any(v < 0 for v in ds) or ds != sorted(ds):
        raise _err("twod_study.extrapolation.distances", "an increasing list of distances >= 0 (decades)")
    if not 0 < float(x.get("quantile", 0.95)) < 1:
        raise _err("twod_study.extrapolation.quantile", "in (0, 1)")
    ls = (s.get("diagnostics") or {}).get("landscape") or {}
    if int(ls.get("n", 25)) % 2 == 0 or (int(ls.get("n_line", 57)) - 1) % 7:
        raise _err(
            "twod_study.diagnostics.landscape", "n odd and n_line - 1 a multiple of 7 (the end points on the grids)"
        )
    sl = x.get("slice") or {}
    pr = sl.get("pair", ["Re", "Pr"])
    if len(pr) != 2 or pr[0] == pr[1] or any(p not in ("Re", "Pr", "Bi") for p in pr):
        raise _err("twod_study.extrapolation.slice.pair", "two different names of Re, Pr, Bi")


def validate_cost_study(spec: dict) -> None:
    """Step 6.7 settings (cost_study)."""
    s = spec.get("cost_study")
    if not s:
        return
    for c, cfg in (s.get("cases") or {}).items():
        dim = (cfg or {}).get("dim")
        if dim not in (1, 2):
            raise _err(f"cost_study.cases.{c}.dim", "1 (a Step 6.2 ensemble) or 2 (a Step 6.4 adapter case)")
        if dim == 2 and c not in ("D_graetz", "C_cavity"):
            raise _err(f"cost_study.cases.{c}", "2D cases: D_graetz or C_cavity")
        if dim == 1 and c not in ("A_steady", "B_slab"):
            raise _err(f"cost_study.cases.{c}", "1D cases: A_steady or B_slab")
        if dim == 2 and not (cfg.get("surrogate") or {}).get("run"):
            raise _err(f"cost_study.cases.{c}.surrogate.run", "a results directory of a saved parametric run")
        if dim == 1 and not cfg.get("run"):
            raise _err(f"cost_study.cases.{c}.run", "a Step 6.2 results directory")
    lat = s.get("latency") or {}
    b = [int(v) for v in lat.get("batches") or [1]]
    if any(v < 1 for v in b) or b != sorted(b):
        raise _err("cost_study.latency.batches", "an increasing list of positive batch sizes")
    if any(int(v) < 0 for v in lat.get("threads") or [1]):
        raise _err("cost_study.latency.threads", "integers >= 0 (0 = torch default)")
    _int(lat.get("repeats", 5), "cost_study.latency.repeats", 1)
    if (s.get("execution") or {}).get("workers", 1) not in (1, "1"):
        raise _err("cost_study.execution.workers", "1 (timings must not share the machine)")


def validate(spec: dict) -> None:
    if spec["dtype"] not in ("float64", "float32"):
        raise _err("dtype", "must be float64 or float32")
    if spec["device"] not in ("auto", "cpu", "cuda", "mps"):
        raise _err("device", "must be auto, cpu, cuda or mps")
    validate_network(spec["network"], "network")
    validate_scaling_study(spec)
    validate_uq_study(spec)
    validate_extrapolation_study(spec)
    validate_sensitivity_study(spec)
    validate_diagnostics_study(spec)
    validate_validation_study(spec)
    validate_twod_study(spec)
    validate_cost_study(spec)
    s = spec["sampling"]
    if s["method"] not in ("sobol", "random", "lhs"):
        raise _err("sampling.method", "must be sobol, random or lhs")
    for k in ("n_pde", "n_ic", "n_bc"):
        _int(s[k], f"sampling.{k}", 1)
    _int(s["resample_every"], "sampling.resample_every", 0)
    for k, v in spec["loss"]["weights"].items():
        if k not in ("pde", "ic", "bc", "gauge", "data") or _num(v, f"loss.weights.{k}") < 0:
            raise _err(f"loss.weights.{k}", "keys pde/ic/bc/gauge/data with values >= 0")
    t = spec["training"]
    a = t["adam"]
    _int(a["epochs"], "training.adam.epochs", 0)
    if _num(a["lr"], "training.adam.lr") <= 0 or _num(a["lr_min"], "training.adam.lr_min") < 0:
        raise _err("training.adam.lr", "lr > 0 and lr_min >= 0 required")
    if a["schedule"] not in SCHEDULES:
        raise _err("training.adam.schedule", f"must be one of {SCHEDULES}")
    _num(a["grad_clip"], "training.adam.grad_clip")
    lb = t["lbfgs"]
    _int(lb["iterations"], "training.lbfgs.iterations", 0)
    _int(lb["inner"], "training.lbfgs.inner", 1)
    _int(lb["history"], "training.lbfgs.history", 1)
    _num(lb["lr"], "training.lbfgs.lr")
    for k in ("log_every", "eval_every", "checkpoint_every"):
        _int(t[k], f"training.{k}", 1)
    e = spec["evaluation"]
    if _num(e["target_rel_l2"], "evaluation.target_rel_l2") <= 0:
        raise _err("evaluation.target_rel_l2", "must be > 0")
    _int(e["nx"], "evaluation.nx", 11)
    _int(e["nt"], "evaluation.nt", 2)
    rs = e["reference_solvers"]
    _int(rs["fd_N"], "evaluation.reference_solvers.fd_N", 8)
    _num(rs["fd_dt"], "evaluation.reference_solvers.fd_dt")
    _int(rs["spectral_N"], "evaluation.reference_solvers.spectral_N", 8)
    if not any(p.get("enabled", True) for p in spec["problems"].values()):
        raise _err("problems", "no problem enabled")
    for name, p in spec["problems"].items():
        base = f"problems.{name}"
        if name not in PROBLEMS:
            raise _err(base, f"unknown problem; available {list(PROBLEMS)}")
        if not p.get("enabled", True):
            continue
        pc = p["parameter"]
        if pc.get("name") != PROBLEMS[name]:
            raise _err(base + ".parameter.name", f"must be {PROBLEMS[name]!r} for {name}")
        if "values" in pc:
            vals = pc["values"]
            if not isinstance(vals, list) or not vals:
                raise _err(base + ".parameter.values", "need a non-empty list")
            for i, v in enumerate(vals):
                if _num(v, f"{base}.parameter.values[{i}]") <= 0:
                    raise _err(f"{base}.parameter.values[{i}]", "must be > 0")
        else:
            lo, hi = _num(pc.get("lo"), base + ".parameter.lo"), _num(pc.get("hi"), base + ".parameter.hi")
            if not 0 < lo < hi:
                raise _err(base + ".parameter", "need 0 < lo < hi")
            _int(pc.get("n"), base + ".parameter.n", 1)
            if pc.get("scale", "log") not in ("log", "linear"):
                raise _err(base + ".parameter.scale", "log or linear")
        if not isinstance(p.get("parametric", False), bool):
            raise _err(base + ".parametric", "true or false")
        if name != "A_steady" and _num(p["T"], base + ".T") <= 0:
            raise _err(base + ".T", "must be > 0")
        c = p["conditions"]
        if name == "A_steady":
            for k in ("c", "theta_left", "theta_right"):
                _num(c[k], f"{base}.conditions.{k}")
        elif name == "A_pulse":
            pu = c["pulse"]
            for k in ("sigma0", "x0", "c", "amplitude", "baseline"):
                _num(pu[k], f"{base}.conditions.pulse.{k}")
            if pu["sigma0"] <= 0:
                raise _err(base + ".conditions.pulse.sigma0", "must be > 0")
        else:
            for k in ("Q", "m", "theta0"):
                _num(c[k], f"{base}.conditions.{k}")
    st = spec["studies"]
    lw = st["loss_weights"]
    if lw.get("enabled", False):
        if lw["problem"] not in spec["problems"] or not spec["problems"][lw["problem"]].get("enabled", True):
            raise _err(
                "studies.loss_weights.problem",
                f"{lw['problem']!r} is not an enabled problem; enable it, pick another problem, or set "
                "studies.loss_weights.enabled=false",
            )
        pv = _num(lw["parameter_value"], "studies.loss_weights.parameter_value")
        if pv <= 0:
            raise _err("studies.loss_weights.parameter_value", "must be > 0")
        if not lw["bc_weights"]:
            raise _err("studies.loss_weights.bc_weights", "need a non-empty list")
    _num(st["error_spectrum"]["k_cut"], "studies.error_spectrum.k_cut")
    validate_user_data(spec)
    validate_arch_study(spec)
    validate_2d(spec)
    validate_execution(spec)
    validate_balancing(spec["loss"]["balancing"], "loss.balancing", spec["sampling"]["resample_every"])
    validate_balance_study(spec)
    validate_strategy_settings(spec)
    validate_strategy_study(spec)
    validate_param_study(spec)
    validate_channel_study(spec)
    validate_hybrid_study(spec)
    validate_ablation_study(spec)
    validate_gate(spec)


def describe(spec: dict) -> str:
    n, t = spec["network"], spec["training"]
    lines = [
        f"profile={spec['profile']} seed={spec['seed']} device={spec['device']} dtype={spec['dtype']} "
        f"net={n.get('arch', 'mlp')} {n['depth']}x{n['width']} {n['activation']} "
        f"adam={t['adam']['epochs']} lbfgs={t['lbfgs']['iterations']}"
    ]
    for name, p in spec["problems"].items():
        if not p.get("enabled", True):
            lines.append(f"  {name}: disabled")
            continue
        lines.append(
            f"  {name}: {p['parameter']['name']} in {param_values(p['parameter'])} "
            f"{'(one parametric network)' if p.get('parametric') else '(one network per value)'}"
        )
    return "\n".join(lines)
