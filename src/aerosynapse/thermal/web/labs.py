"""Phase 10.4-10.6: the laboratories THERMAL LAB gained after the first public version (D149).

  flow(q)            transient lid-driven / buoyant cavity flow with heat transfer (the Phase 2 Case C generator:
                     MAC projection Navier-Stokes + energy, verified in Phase 1): frames of temperature, velocity,
                     stream function and vorticity; wall Nusselt numbers and kinetic energy in time
  dataset(q)         the Phase 2 training-data pipeline on demand: Sobol / LHS design with train / val / interp / extrap
                     splits and the leakage check, ground-truth fields, sensor layouts and noise; kept for download
  train_start(q)     a PINN trained on REAL soil temperatures (src/inverse/realpinn.py) as a background job, with live
  train_status(q)    progress; the classical estimator on the same sensors for comparison
  export(q)          a ZIP for any result: request, result (JSON), tables (CSV), figures (PNG) and reproduce.py, which
                     re-runs the request with the project's code and redraws the same figures
  code_zip()         the project's code (no data, no saved networks) as a ZIP

Everything runs the project's own verified code; inputs are validated and bounded (tighter on a public server).
"""

from __future__ import annotations

import copy
import csv
import io
import json
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import yaml

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
DATA_CFG = ROOT / "configs" / "data" / "defaults.yaml"

# (local, public) limits of the new laboratories
LIMITS = {
    "flow_grid": (64, 32),
    "flow_frames": (60, 30),
    "flow_t_end": (40.0, 16.0),
    "points_1d": (96, 24),  # design points of a 1D training-data set
    "points_2d": (16, 4),  # design points of a cavity training-data set (one simulation each)
    "train_adam": (20000, 2500),
    "train_lbfgs": (2000, 200),
    "train_colloc": (5000, 1500),
    "train_running": (2, 1),  # training jobs at the same time
}
PUBLIC = {"on": False}


def lim(name: str):
    return LIMITS[name][1 if PUBLIC["on"] else 0]


def _num(q: dict, key: str, default: float, lo: float, hi: float) -> float:
    v = float(q.get(key, default))
    if not lo <= v <= hi:
        raise ValueError(f"{key} = {v:g} outside [{lo:g}, {hi:g}]")
    return v


def _int(q: dict, key: str, default: int, lo: int, hi: int) -> int:
    return int(_num(q, key, default, lo, hi))


def _data_cfg() -> dict:
    return yaml.safe_load(DATA_CFG.read_text(encoding="utf-8"))


# ================================================================================================= flow lab
HEATING = {
    # thermal wall conditions of the cavity; theta = (T - T_cold) / (T_hot - T_cold)
    "lid_hot": {  # the project's default Case C: hot moving lid, convective (Robin) bottom, adiabatic sides
        "top": {"type": "dirichlet", "value": 1.0},
        "bottom": {"type": "robin", "Bi": "Bi", "theta_inf": 0.0},
        "left": {"type": "neumann", "value": 0.0},
        "right": {"type": "neumann", "value": 0.0},
    },
    "bottom_heated": {  # heated from below, cooled from above (Rayleigh-Benard-like when Ri > 0)
        "top": {"type": "dirichlet", "value": 0.0},
        "bottom": {"type": "dirichlet", "value": 1.0},
        "left": {"type": "neumann", "value": 0.0},
        "right": {"type": "neumann", "value": 0.0},
    },
    "side_heated": {  # differentially heated cavity (natural convection when Ri > 0)
        "top": {"type": "neumann", "value": 0.0},
        "bottom": {"type": "neumann", "value": 0.0},
        "left": {"type": "dirichlet", "value": 1.0},
        "right": {"type": "dirichlet", "value": 0.0},
    },
}
LIDS = {
    "cosh": {"profile": "cosh", "C": 50.0},  # regularised lid (the project's default: no corner singularity)
    "uniform": {"profile": "uniform"},
    "parabolic": {"profile": "parabolic"},
    "none": {"profile": "zero"},
}


def flow(q: dict) -> dict:
    """Transient cavity: Navier-Stokes (MAC projection) + energy equation, frames in time."""
    from aerosynapse.thermal.data import generators as G

    p = {
        "Re": _num(q, "Re", 400.0, 1.0, 3000.0),
        "Pr": _num(q, "Pr", 0.71, 0.01, 100.0),
        "Bi": _num(q, "Bi", 1.0, 1e-3, 1e3),
        "Ri": _num(q, "Ri", 0.0, -10.0, 100.0),
    }
    heating, lid = str(q.get("heating", "lid_hot")), str(q.get("lid", "cosh"))
    if heating not in HEATING or lid not in LIDS:
        raise ValueError(f"heating must be one of {list(HEATING)} and lid one of {list(LIDS)}")
    amp = _num(q, "lid_speed", 1.0, -3.0, 3.0)
    grid = _int(q, "grid", 32, 8, lim("flow_grid"))
    frames = _int(q, "frames", 24, 2, lim("flow_frames"))
    t_end = _num(q, "t_end", 8.0, 0.1, lim("flow_t_end"))
    cfg = copy.deepcopy(_data_cfg()["cases"]["caseC"])
    cfg["conditions"]["thermal"] = copy.deepcopy(HEATING[heating])
    cfg["conditions"]["velocity"]["top"] = dict(LIDS[lid], amplitude=amp)
    cfg["conditions"]["initial_theta"] = _num(q, "theta0", 0.0, -1.0, 2.0)
    if "scheme" in q:
        cfg["scheme"] = str(q["scheme"])
    times = np.linspace(t_end / frames, t_end, frames)
    t0 = time.perf_counter()
    r = G.gen_case_c_point(p, times, cfg, grid)
    x, y = r["xc"], r["yc"]
    dx, dy = x[1] - x[0], y[1] - y[0]
    # stream function from the cell velocities: psi(y) = integral of u dy from the bottom wall (trapezoid), psi_wall = 0
    psi = np.concatenate([np.zeros((frames, 1, grid)), np.cumsum(r["u"], axis=1) * dy], axis=1)
    psi = 0.5 * (psi[:, 1:, :] + psi[:, :-1, :])
    vort = np.gradient(r["v"], dx, axis=2) - np.gradient(r["u"], dy, axis=1)
    step = max(1, grid // 32)  # velocity at <= 32 x 32 for arrows and particles
    out = {
        "params": p,
        "heating": heating,
        "lid": lid,
        "lid_speed": amp,
        "grid": grid,
        "t": times,
        "x": x,
        "y": y,
        "theta": np.round(r["theta"], 5),
        "psi": np.round(psi, 6),
        "vorticity": np.round(vort, 4),
        "xv": x[::step],
        "yv": y[::step],
        "u": np.round(r["u"][:, ::step, ::step], 5),
        "v": np.round(r["v"][:, ::step, ::step], 5),
        "nu_top": r["nu_top"],
        "nu_bottom": r["nu_bottom"],
        "nu_left": r["nu_left"],
        "nu_right": r["nu_right"],
        "ke": r["ke"],
        "steps": int(r["steps"]),
        "max_div": float(r["max_div"]),
        "wall_s": time.perf_counter() - t0,
        "solver": "MAC projection Navier-Stokes + energy (Phase 1.6 / 2, verified); Pe = Re Pr; buoyancy Ri along +y",
    }
    return out


# ================================================================================================= training data
DATASETS: OrderedDict[str, dict] = OrderedDict()
_DS_LOCK = threading.Lock()


def dataset(q: dict) -> dict:
    """The Phase 2 pipeline (design, ground truth, sensors, noise) for one case, with the user's choices."""
    from aerosynapse.thermal.experiments.p2_data import generate
    from aerosynapse.thermal.data import design as D
    from aerosynapse.thermal.data import sensors as S
    from aerosynapse.thermal.data.noise import add_noise

    spec = _data_cfg()
    case = str(q.get("case", "caseB"))
    if case not in spec["cases"]:
        raise ValueError(f"case must be one of {list(spec['cases'])}")
    ccfg = copy.deepcopy(spec["cases"][case])
    is2d = case == "caseC"
    dims, fixed = [], {}
    for name, v in ccfg["parameters"].items():
        if isinstance(v, dict):
            o = (q.get("ranges") or {}).get(name, {})
            dims.append(D.Dim(name, float(o.get("lo", v["lo"])), float(o.get("hi", v["hi"])), v.get("scale", "log")))
        else:
            fixed[name] = float((q.get("fixed") or {}).get(name, v))
    sizes = {k: int((q.get("sizes") or {}).get(k, ccfg["sizes"][k])) for k in ("train", "val", "interp", "extrap")}
    n_max = lim("points_2d") if is2d else lim("points_1d")
    if sum(sizes.values()) > n_max or sizes["train"] < 1 or min(sizes.values()) < 0:
        raise ValueError(f"at most {n_max} design points in total for {case} (train >= 1)")
    if is2d:
        ccfg["grid"] = _int(q, "grid", min(int(ccfg["grid"]), 24), 8, lim("flow_grid"))
    else:
        ccfg["nx"] = _int(q, "nx", int(ccfg["nx"]), 17, 2049)
    ccfg["times"]["n"] = _int(q, "n_times", int(ccfg["times"]["n"]), 1, 40)
    seed = _int(q, "seed", int(spec["seed"]), 0, 10**6)
    method = str(q.get("method", spec["design"]["method"]))
    if method not in ("sobol", "lhs"):
        raise ValueError("method must be sobol or lhs")
    u_lim = _num(q, "u_lim", float(spec["design"]["unit_train_limit"]), 0.3, 0.95)
    noise_model = str(q.get("noise_model", spec["noise"]["model"]))
    levels = [float(v) for v in q.get("noise_levels", spec["noise"]["levels"])][:8]
    layout = str(q.get("layout", "grid" if is2d else "uniform"))
    n_sens = _int(q, "n_sensors", 16 if is2d else 8, 1, 400 if is2d else 200)
    t0 = time.perf_counter()
    des = D.design_case(case, dims, sizes, seed, u_lim, method, float(spec["design"]["min_dist_factor"]))
    coords, fields, extra, timing = generate(case, des, fixed, ccfg)
    leak = D.check_leakage(des)
    idx = des.idx("interp")
    pi = int(idx[0]) if len(idx) else int(des.idx("train")[0])
    if is2d:
        pts = S.layout_2d(layout, n_sens, seed)
        clean = S.sample_2d(coords["x"], coords["y"], fields["theta"][pi], pts)
    else:
        pts = S.layout_1d(layout, n_sens, seed)
        clean = S.sample_1d(coords["x"], fields["theta"][pi], pts)
    noisy = np.stack([add_noise(clean, lv, noise_model, seed + 1000 * k) for k, lv in enumerate(levels)])
    did = uuid.uuid4().hex[:12]
    rec = {
        "id": did,
        "case": case,
        "request": q,
        "dims": [{"name": d.name, "lo": d.lo, "hi": d.hi, "scale": d.scale} for d in dims],
        "fixed": fixed,
        "split_names": list(D.SPLITS),
        "split": des.split,
        "u": des.u,
        "params": des.params,
        "coords": coords,
        "fields": fields,
        "extra": extra,
        "sensors": {
            "point": pi,
            "layout": layout,
            "positions": pts,
            "clean": clean,
            "levels": levels,
            "noise_model": noise_model,
            "noisy": noisy,
        },
        "leakage": leak,
        "seed": seed,
        "method": method,
        "u_lim": u_lim,
        "wall_s": time.perf_counter() - t0,
    }
    with _DS_LOCK:
        DATASETS[did] = rec
        while len(DATASETS) > (3 if PUBLIC["on"] else 8):
            DATASETS.popitem(last=False)
    nbytes = int(sum(np.asarray(v).nbytes for v in fields.values()))
    # preview: the first train point's field(s), the sensors of one interpolation point
    tr = int(des.idx("train")[0])
    preview = (
        {
            "x": coords["x"],
            "y": coords["y"],
            "theta": fields["theta"][tr][-1],
            "u": fields["u"][tr][-1],
            "v": fields["v"][tr][-1],
        }
        if is2d
        else {"x": coords["x"], "t": coords["t"], "theta": fields["theta"][tr]}
    )
    return {
        "id": did,
        "case": case,
        "dims": rec["dims"],
        "fixed": fixed,
        "design": [
            {"split": D.SPLITS[int(s)], "params": dict(zip([d.name for d in dims], map(float, row), strict=True))}
            for s, row in zip(des.split, des.params, strict=True)
        ],
        "leakage": leak,
        "shapes": {k: list(np.shape(v)) for k, v in fields.items()},
        "bytes": nbytes,
        "preview_point": {"index": tr, "params": {d.name: float(des.params[tr, k]) for k, d in enumerate(dims)}},
        "preview": preview,
        "sensors": {
            "point": pi,
            "layout": layout,
            "positions": pts,
            "clean": clean,
            "levels": levels,
            "noisy": noisy,
            "params": {d.name: float(des.params[pi, k]) for k, d in enumerate(dims)},
        },
        "timing": timing,
        "wall_s": rec["wall_s"],
    }


# ================================================================================================= real-data training
JOBS: OrderedDict[str, dict] = OrderedDict()
_JOB_LOCK = threading.Lock()


class _Cancelled(Exception):
    pass


def _thin(hist: list[dict], n: int = 400) -> list[dict]:
    k = max(1, int(np.ceil(len(hist) / n)))
    return hist[::k] + ([hist[-1]] if hist and (len(hist) - 1) % k else [])


def train_start(q: dict, window_of, gate=None) -> dict:
    """Start a background training job. window_of(q) -> (window dict, description) builds the data window exactly as
    the Real-data tab does (a NOAA station or an uploaded table); gate: the server's computation semaphore."""
    from aerosynapse.thermal.inverse import realpinn as P

    win, desc = window_of(q)
    z = sorted(float(v) for v in win["z"])
    top, bot = float(q.get("top_m", z[0])), float(q.get("bottom_m", z[-1]))
    inner = [float(v) for v in q.get("train_inner", [v for v in z if top < v < bot][1:2] or [])]
    holdout = q.get("holdout")
    holdout = float(holdout) if holdout not in (None, "") else None
    if holdout is not None and (holdout in inner or not top < holdout < bot):
        raise ValueError("the held-out depth must lie between the boundary sensors and not be a training sensor")
    hp = {
        "seed": _int(q, "seed", 0, 0, 10**6),
        "width": _int(q, "width", 32, 4, 128),
        "depth": _int(q, "depth", 3, 1, 8),
        "harmonics": _int(q, "harmonics", 2, 0, 6),
        "n_colloc": _int(q, "n_colloc", 1500, 100, lim("train_colloc")),
        "warmup": _int(q, "warmup", 500, 0, 10000),
        "adam_steps": _int(q, "adam_steps", 2500 if PUBLIC["on"] else 3000, 10, lim("train_adam")),
        "lr": _num(q, "lr", 3e-3, 1e-5, 0.1),
        "lbfgs_steps": _int(q, "lbfgs_steps", 200 if PUBLIC["on"] else 300, 0, lim("train_lbfgs")),
        "w_pde": _num(q, "w_pde", 1.0, 0.0, 1e4),
        "alpha_init": _num(q, "alpha_init", 1e-6, 1e-9, 1e-4),
    }
    with _JOB_LOCK:
        now = time.time()
        for k in [k for k, j in JOBS.items() if now - j["created"] > 3600 and j["state"] not in ("queued", "running")]:
            JOBS.pop(k)
        running = sum(j["state"] in ("queued", "running") for j in JOBS.values())
        if running >= lim("train_running"):
            raise ValueError("a training is already running on this server: wait for it or cancel it")
        jid = uuid.uuid4().hex[:12]
        job = {
            "id": jid,
            "state": "queued",
            "created": time.time(),
            "history": [],
            "request": q,
            "data": desc,
            "hp": hp,
            "top": top,
            "bottom": bot,
            "inner": inner,
            "holdout": holdout,
            "cancel": False,
            "total": hp["adam_steps"] + hp["lbfgs_steps"],
        }
        JOBS[jid] = job

    def progress(rec):
        job["history"].append(rec)
        if job["cancel"]:
            raise _Cancelled()

    def work():
        acquired = gate is None or gate.acquire(timeout=120)
        if not acquired:
            job.update(state="error", error="the server is busy with other visitors' computations: try again shortly")
            return
        try:
            job["state"] = "running"
            job["started"] = time.time()
            r = P.train(win, [top, *inner, bot], top, bot, progress=progress, every=25, **hp)
            res = {
                "alpha": r["alpha"],
                "wall_s": r["wall_s"],
                "n_parameters": r["n_parameters"],
                "scales": r["scales"],
                "config": r["config"],
            }
            spin = 24 * 3600.0
            res["rmse"] = {
                f"{zz:g}": P.rmse_after(win["t"], win["T"][:, j], r["prediction"][:, j], spin)
                for j, zz in enumerate(win["z"])
                if top - 1e-9 <= zz <= bot + 1e-9
            }
            series = {"t_h": (win["t"] - win["t"][0]) / 3600, "z": win["z"], "obs": win["T"], "pinn": r["prediction"]}
            if holdout is not None:
                from aerosynapse.thermal.web.server import soil_cfg

                m = {"top_m": top, "bottom_m": bot, "dz_m": 0.0025, "dt_s": 600, "spinup_h": 24}
                cl = P.classical_loo(win, inner, holdout, m, soil_cfg()["fit"])
                k = int(np.argmin(np.abs(win["z"] - holdout)))
                res["classical"] = {
                    "alpha": cl["alpha"],
                    "converged": cl["converged"],
                    "rmse_holdout": P.rmse_after(win["t"], win["T"][:, k], cl["prediction"], spin),
                }
                series["classical_holdout"] = cl["prediction"]
            job["result"], job["series"] = res, series
            job["state"] = "done"
        except _Cancelled:
            job["state"] = "cancelled"
        except Exception as err:  # noqa: BLE001 - reported to the page
            job.update(state="error", error=f"{type(err).__name__}: {err}")
        finally:
            job["finished"] = time.time()
            if gate is not None:
                gate.release()

    threading.Thread(target=work, daemon=True, name=f"train-{jid}").start()
    return {
        "id": jid,
        "state": job["state"],
        "total": job["total"],
        "data": desc,
        "hp": hp,
        "inner": inner,
        "holdout": holdout,
        "top": top,
        "bottom": bot,
    }


def train_status(q: dict) -> dict:
    j = JOBS.get(str(q.get("id", "")))
    if j is None:
        raise ValueError("no such training job (finished jobs are kept for one hour)")
    if q.get("cancel"):
        j["cancel"] = True
    h = j["history"]
    out = {k: j.get(k) for k in ("id", "state", "error", "data", "hp", "inner", "holdout", "top", "bottom", "total")}
    out["step"] = h[-1]["step"] if h else 0
    out["history"] = _thin(h)
    if j["state"] == "done":
        s = j["series"]
        k = max(1, int(np.ceil(len(s["t_h"]) / 700)))
        out["result"] = j["result"]
        out["series"] = {key: (v[::k] if key != "z" else v) for key, v in s.items()}
    return out


# ================================================================================================= downloads
README = """THERMAL LAB export: {kind}
Created {when} by the project's code (git {git}).

  request.json     the inputs of this result (edit them and re-run)
  result.json      the full result as returned by the server
  tables/*.csv     every series and field of the result as tables
  figures/*.png    the figures (drawn by src/web/labs.py: figures_for)
  reproduce.py     re-runs request.json with the project's code and redraws the figures

To reproduce: put reproduce.py and request.json in the project root (or in the code downloaded with "Download code"
from THERMAL LAB, with its requirements installed) and run:  python reproduce.py
"""

REPRODUCE = '''"""Reproduce a THERMAL LAB result ({kind}) with the project's code: python reproduce.py [request.json]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aerosynapse.thermal.web import labs, server  # noqa: E402

req = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "request.json").read_text(encoding="utf-8"))
kind = {kind!r}
result = labs.run_kind(kind, req)
Path("reproduced").mkdir(exist_ok=True)
Path("reproduced/result.json").write_text(json.dumps(server.clean(result)), encoding="utf-8")
for name, png in labs.figures_for(kind, req, result).items():
    Path(f"reproduced/{{name}}.png").write_bytes(png)
print("written: reproduced/result.json and", len(labs.figures_for(kind, req, result)), "figures")
'''


def run_kind(kind: str, req: dict) -> dict:
    """Re-run one kind of computation exactly as the server does (used by export and by reproduce.py)."""
    from aerosynapse.thermal.web import server as W

    routes = {
        "solve": "/api/solve",
        "converge": "/api/converge",
        "sweep": "/api/sweep",
        "inverse": "/api/inverse",
        "real_fit": "/api/real/fit",
        "lumped": "/api/real/lumped",
        "flow": "/api/flow",
        "dataset": "/api/dataset",
    }
    if kind == "train":
        raise ValueError("a training is exported from its job (it is not re-run on export)")
    if kind not in routes:
        raise ValueError(f"unknown kind {kind!r}")
    return W.ROUTES[routes[kind]](req)


def _tables(obj, prefix: str = "") -> dict[str, str]:
    """Every array of a (nested) result as CSV text: 1D arrays of equal length side by side, 2D arrays as matrices,
    higher dimensions flattened to rows (index columns first)."""
    flat: dict[str, np.ndarray] = {}

    def walk(o, p):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{p}{k}." if p or k else k)
        elif isinstance(o, (list, tuple, np.ndarray)):
            try:
                a = np.asarray(o, dtype=float)
            except (TypeError, ValueError):
                return
            if a.size > 1:
                flat[p.rstrip(".")] = a

    walk(obj, prefix)
    out: dict[str, str] = {}
    one = [(k, a) for k, a in flat.items() if a.ndim == 1]
    by_len: dict[int, list] = {}
    for k, a in one:
        by_len.setdefault(len(a), []).append((k, a))
    for n, cols in by_len.items():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow([k for k, _ in cols])
        for i in range(n):
            w.writerow([f"{a[i]:.10g}" for _, a in cols])
        out[f"series_{n}.csv"] = buf.getvalue()
    for k, a in flat.items():
        if a.ndim < 2:
            continue
        buf = io.StringIO()
        w = csv.writer(buf)
        if a.ndim == 2:
            for row in a:
                w.writerow([f"{v:.10g}" for v in row])
        else:
            idx = np.indices(a.shape).reshape(a.ndim, -1).T
            w.writerow([f"i{d}" for d in range(a.ndim)] + ["value"])
            for ii, v in zip(idx, a.ravel(), strict=True):
                w.writerow([*ii.tolist(), f"{v:.10g}"])
        out[f"{k.replace('/', '_')}.csv"] = buf.getvalue()
    return out


def _png(fig) -> bytes:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def figures_for(kind: str, req: dict, r: dict) -> dict[str, bytes]:
    """The figures of one result as PNG bytes (matplotlib, the project's style)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aerosynapse.thermal.viz.style import DATA, PINN, SOLVER, apply_style

    apply_style()
    figs: dict[str, bytes] = {}
    A = np.asarray
    if kind == "solve":
        s = r.get("solver", {})
        if "y" in s:
            fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
            im = axs[0].pcolormesh(A(s["x"]), A(s["y"]), A(s["theta"]), cmap="inferno", shading="auto")
            fig.colorbar(im, ax=axs[0], label="θ")
            axs[0].set(title="temperature", aspect="equal")
            sp = np.hypot(A(s["u"]), A(s["v"]))
            im = axs[1].pcolormesh(A(s["x"]), A(s["y"]), sp, cmap="viridis", shading="auto")
            axs[1].streamplot(A(s["x"]), A(s["y"]), A(s["u"]), A(s["v"]), color="w", linewidth=0.6, density=1.1)
            fig.colorbar(im, ax=axs[1], label="|u|")
            axs[1].set(title="speed and streamlines", aspect="equal")
            figs["fields"] = _png(fig)
        else:
            fig, ax = plt.subplots(figsize=(6.4, 3.6))
            th = A(s["theta"])
            for i, row in enumerate(np.atleast_2d(th)):
                ax.plot(
                    A(s["x"]),
                    row,
                    color=SOLVER,
                    lw=1.2,
                    alpha=0.4 + 0.6 * (i + 1) / len(np.atleast_2d(th)),
                    label="verified solver" if i == 0 else None,
                )
            if "pinn" in r:
                for i, row in enumerate(np.atleast_2d(A(r["pinn"]["mean"]))):
                    ax.plot(
                        A(r["pinn"]["x"]), row, color=PINN, lw=1.0, ls="--", label="PINN ensemble" if i == 0 else None
                    )
            ax.set(xlabel="x", ylabel="θ", title=f"{req.get('case', '')} {r.get('params', {})}")
            ax.legend(fontsize=8)
            figs["profiles"] = _png(fig)
    elif kind == "flow":
        t, k = A(r["t"]), -1
        fig, axs = plt.subplots(2, 2, figsize=(9.5, 7.6))
        axs = axs.ravel()
        im = axs[0].pcolormesh(A(r["x"]), A(r["y"]), A(r["theta"])[k], cmap="inferno", shading="auto")
        fig.colorbar(im, ax=axs[0], label="θ")
        axs[0].streamplot(A(r["xv"]), A(r["yv"]), A(r["u"])[k], A(r["v"])[k], color="w", linewidth=0.6, density=1.0)
        axs[0].set(title=f"temperature and streamlines, t = {t[k]:.2f}", aspect="equal")
        im = axs[1].pcolormesh(A(r["x"]), A(r["y"]), A(r["vorticity"])[k], cmap="RdBu_r", shading="auto")
        fig.colorbar(im, ax=axs[1], label="ω")
        axs[1].set(title="vorticity", aspect="equal")
        axs[2].plot(t, A(r["nu_top"]), color=SOLVER, label="Nu top")
        axs[2].plot(t, A(r["nu_bottom"]), color=PINN, label="Nu bottom")
        if r.get("heating") == "side_heated":
            axs[2].plot(t, A(r["nu_left"]), color="#CC79A7", label="Nu left")
            axs[2].plot(t, A(r["nu_right"]), color="#F0E442", label="Nu right")
        axs[2].set(xlabel="t", ylabel="Nu", title="wall heat transfer")
        axs[2].legend(fontsize=7)
        axs[3].plot(t, A(r["ke"]), color=DATA)
        axs[3].set(xlabel="t", ylabel="kinetic energy", title="kinetic energy of the flow")
        figs["flow_final"] = _png(fig)
    elif kind == "dataset":
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
        names = [d["name"] for d in r["dims"]]
        colors = {"train": SOLVER, "val": DATA, "interp": PINN, "extrap": "#CC79A7"}
        for split in colors:
            pts = [d["params"] for d in r["design"] if d["split"] == split]
            if not pts:
                continue
            xs = [p[names[0]] for p in pts]
            ys = [p[names[1]] for p in pts] if len(names) > 1 else [0] * len(pts)
            axs[0].scatter(xs, ys, s=18, color=colors[split], label=split)
        axs[0].set(
            xscale="log",
            xlabel=names[0],
            ylabel=names[1] if len(names) > 1 else "",
            title="parameter design (splits never share a point)",
        )
        if len(names) > 1:
            axs[0].set_yscale("log")
        axs[0].legend(fontsize=7)
        s = r["sensors"]
        pos = A(s["positions"])
        if pos.ndim == 1:
            for lv, row in zip(s["levels"], A(s["noisy"]), strict=True):
                axs[1].plot(pos, row[-1] if row.ndim > 1 else row, "o", ms=3, label=f"noise {lv:g}")
            axs[1].set(xlabel="x", ylabel="θ at the last snapshot", title="sensor readings, one interpolation point")
            axs[1].legend(fontsize=7)
        else:
            axs[1].scatter(pos[:, 0], pos[:, 1], c=A(s["clean"])[-1], cmap="inferno", s=24)
            axs[1].set(aspect="equal", title="sensor layout (colour: clean reading)")
        figs["design_sensors"] = _png(fig)
    elif kind == "train":
        h, s, res = r["history"], r["series"], r["result"]
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
        st = [x["step"] for x in h]
        axs[0].semilogy(st, [x["data"] for x in h], color=DATA, label="data misfit")
        axs[0].semilogy(st, [x["pde"] for x in h], color=PINN, label="heat-equation residual")
        axs[0].set(xlabel="step", ylabel="loss (scaled)", title="training")
        axs[0].legend(fontsize=7)
        axs[1].plot(st, [x["alpha"] for x in h], color=PINN, label="PINN")
        if "classical" in res:
            axs[1].axhline(res["classical"]["alpha"], color=SOLVER, ls="--", label="classical estimator")
        axs[1].set(xlabel="step", ylabel="α [m²/s]", title="thermal diffusivity")
        axs[1].legend(fontsize=7)
        figs["training"] = _png(fig)
        z = A(s["z"])
        fig, axs = plt.subplots(len(z), 1, figsize=(8, 1.7 * len(z)), sharex=True, squeeze=False)
        for j, zz in enumerate(z):
            ax = axs[j, 0]
            ax.plot(A(s["t_h"]), A(s["obs"])[:, j], color=DATA, lw=1.0, label="measured")
            ax.plot(A(s["t_h"]), A(s["pinn"])[:, j], color=PINN, lw=0.9, label="PINN")
            if r.get("holdout") is not None and abs(zz - r["holdout"]) < 1e-9 and "classical_holdout" in s:
                ax.plot(A(s["t_h"]), A(s["classical_holdout"]), color=SOLVER, ls="--", lw=0.9, label="classical")
            tag = " (held out)" if r.get("holdout") is not None and abs(zz - r["holdout"]) < 1e-9 else ""
            ax.set_ylabel(f"{zz * 100:.0f} cm{tag}", fontsize=8)
        axs[0, 0].legend(fontsize=7, ncol=3)
        axs[-1, 0].set(xlabel="hours")
        figs["sensors"] = _png(fig)
    elif kind == "real_fit":
        s = r["series"]
        fig, ax = plt.subplots(figsize=(8, 3.4))
        T, P_ = A(s["T"]), A(s["pred"])
        for j in range(T.shape[1]):
            ax.plot(A(s["t_h"]), T[:, j], lw=0.8, color=DATA, alpha=0.5 + 0.5 * j / max(1, T.shape[1] - 1))
        for j in range(P_.shape[1]):
            ax.plot(A(s["t_h"]), P_[:, j], lw=1.0, color=PINN, ls="--")
        ax.set(xlabel="hours", ylabel="T [°C]", title="measured (green) and model (dashed) at the inner sensors")
        figs["fit"] = _png(fig)
    elif kind == "converge":
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        for name, sc in r["schemes"].items():
            ax.loglog(r["h"], sc["err_inf"], "o-", label=f"{name}: order {sc['observed_order']:.2f}")
        ax.set(xlabel="h", ylabel="max |error|", title="grid convergence (verification)")
        ax.legend(fontsize=8)
        figs["convergence"] = _png(fig)
    elif kind == "inverse":
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        f = A(r["fitted"])
        for j in range(f.shape[1]):
            ax.plot(A(r["times"]), f[:, j], color=PINN, lw=0.9)
        ax.set(xlabel="t", ylabel="T", title=f"fitted sensor histories (identifiable: {r['identifiable']})")
        figs["fit"] = _png(fig)
    if not figs:  # generic: every 1D series of equal length against its index
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.text(0.5, 0.5, f"{kind}: see tables/*.csv", ha="center")
        figs["summary"] = _png(fig)
    return figs


def export(q: dict) -> tuple[bytes, str]:
    """ZIP of one result: request, result, tables, figures, reproduce.py, README."""
    from aerosynapse.thermal.utils.experiment import git_hash
    from aerosynapse.thermal.web import server as W

    kind = str(q.get("kind", ""))
    req = q.get("request") or {}
    if kind == "train":
        j = JOBS.get(str(q.get("id", "")))
        if j is None or j["state"] != "done":
            raise ValueError("export a training once it has finished")
        res = train_status({"id": j["id"]})
        req = j["request"]
    elif kind == "dataset" and q.get("id") in DATASETS:
        d = DATASETS[q["id"]]
        req = d["request"]
        res = {k: d[k] for k in ("case", "dims", "fixed", "leakage", "seed", "method", "u_lim")}
        res["design"] = [
            {
                "split": d["split_names"][int(s)],
                "params": dict(zip([x["name"] for x in d["dims"]], map(float, row), strict=True)),
            }
            for s, row in zip(d["split"], d["params"], strict=True)
        ]
        res["sensors"] = d["sensors"]
    else:
        res = run_kind(kind, req)
    when = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", README.format(kind=kind, when=when, git=git_hash()))
        z.writestr("request.json", json.dumps(req, indent=1))
        z.writestr("result.json", json.dumps(W.clean(res)))
        if kind != "train":
            z.writestr("reproduce.py", REPRODUCE.format(kind=kind))
        for name, text in _tables(W.clean(res)).items():
            z.writestr(f"tables/{name}", text)
        for name, png in figures_for(kind, req, res).items():
            z.writestr(f"figures/{name}.png", png)
        if kind == "dataset" and q.get("id") in DATASETS:
            d = DATASETS[q["id"]]
            arr = io.BytesIO()
            np.savez_compressed(
                arr,
                u=d["u"],
                params=d["params"],
                split=d["split"],
                **{f"coord_{k}": v for k, v in d["coords"].items()},
                **{f"field_{k}": v for k, v in d["fields"].items()},
                sensor_positions=d["sensors"]["positions"],
                sensor_clean=d["sensors"]["clean"],
                sensor_noisy=d["sensors"]["noisy"],
            )
            z.writestr("dataset.npz", arr.getvalue())
            z.writestr(
                "dataset_card.json",
                json.dumps(
                    W.clean(
                        {
                            "case": d["case"],
                            "dims": d["dims"],
                            "fixed": d["fixed"],
                            "splits": d["split_names"],
                            "seed": d["seed"],
                            "method": d["method"],
                            "u_lim": d["u_lim"],
                            "leakage": d["leakage"],
                            "shapes": {k: list(np.shape(v)) for k, v in d["fields"].items()},
                            "sensors": {k: d["sensors"][k] for k in ("point", "layout", "levels", "noise_model")},
                        }
                    ),
                    indent=1,
                ),
            )
    return buf.getvalue(), f"thermal_lab_{kind}_{time.strftime('%Y%m%d_%H%M%S')}.zip"


CODE_INCLUDE = [
    "src",
    "experiments",
    "configs",
    "deploy",
    "tasks.py",
    "pyproject.toml",
    "LICENSE",
    "README.md",
    "Makefile",
    "docs/WEB.md",
    "docs/REAL_DATA.md",
]


def code_zip() -> tuple[bytes, str]:
    """The project's code and configurations (no data, no saved networks, no results) as a ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for item in CODE_INCLUDE:
            p = ROOT / item
            if not p.exists():
                continue
            for f in [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file()):
                rel = f.relative_to(ROOT)
                if "__pycache__" in rel.parts or f.suffix in (".pyc", ".pt", ".npz", ".h5"):
                    continue
                z.write(f, f"thermal_lab_code/{rel.as_posix()}")
        z.writestr(
            "thermal_lab_code/HOW_TO_RUN.txt",
            "pip install numpy scipy sympy matplotlib pyyaml h5py pandas plotly torch\n"
            "python tasks.py web            # THERMAL LAB on http://127.0.0.1:8765\n"
            "python tasks.py pipeline       # the automated end-to-end run (data, studies, training, tests)\n"
            "Saved PINN networks and the NOAA data are not included: python -m aerosynapse.thermal.experiments.p10_fetch_real\n"
            "downloads the data; the networks are re-trained by the phase scripts (docs/ROADMAP.md).\n",
        )
    return buf.getvalue(), "thermal_lab_code.zip"
