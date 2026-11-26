"""Phase 10.2: the web demo (THERMAL LAB) — one page that drives the whole project.

    python tasks.py web                 # or: python -m aerosynapse.thermal.web.server [--port 8765] [--no-browser]

A local server (Python standard library only, bound to 127.0.0.1) serves src/web/static and a JSON API over the
project's own code, so the page shows exactly what the verified solvers, the saved PINN ensembles and the
estimators compute; nothing is re-implemented in JavaScript (training: only the bounded jobs of /api/train, 10.5):

  GET  /api/info              cases (parameters, ranges, PINN training boxes, conditions), PINN availability,
                              stations on disk, key findings (from the papers' generated numbers, with registry ids)
  POST /api/solve             verified solver of a case            (src/app/backend.solve)
  POST /api/compare           solver + PINN ensemble + error       (backend.compare; solver only if no PINN)
  POST /api/converge          grid convergence, observed order     (backend.convergence)
  POST /api/sweep             PINN error / spread across a parameter range (backend.sweep_1d)
  POST /api/inverse           synthetic slab identification with the rank check (backend.inverse); also your CSV
  POST /api/real/stations     NOAA USCRN station files on disk and their sensor coverage
  POST /api/real/fetch        download one USCRN station-year from NOAA (validated name, fixed host)
  POST /api/real/fit          conduction model fitted to a station window or to your table (src/inverse/realdata)
  POST /api/real/lumped       Newton-cooling fit of one temperature series (your table)
  POST /api/flow              transient cavity flow + heat transfer (src/web/labs.flow, Case C generator)
  POST /api/dataset           the training-data pipeline on demand (labs.dataset)
  POST /api/train/start       a PINN trained on real soil temperatures as a background job (labs.train_start)
  POST /api/train/status      its progress and result (labs.train_status; cancel: true stops it)
  POST /api/export            any result as a ZIP: request, result, CSV, PNG, reproduce.py (labs.export)
  GET  /api/download/code     the project's code as a ZIP (labs.code_zip)

Every request is validated; uploads stay in memory (a temporary file only for the slab CSV reader) and are limited
in size; the server never reads a path sent by the client.

Public deployment (the Hugging Face Space of deploy/hf_space): `--public` listens on all interfaces (port from $PORT
or --port), applies the tighter column of LIMITS, runs at most --max-jobs computations at a time (others wait, then
get 503) and keeps at most LIMITS["fetch_files"] downloaded station files.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import tempfile
import threading
import time
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import yaml

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
STATIC = Path(__file__).resolve().parent / "static"
SOIL_CFG = ROOT / "configs" / "real" / "soil.yaml"
MAX_BODY = 25 * 2**20
# Limits of one interactive request, (local, public): on a public host one visitor must not be able to occupy the CPU
# or the memory of the machine; locally you may try finer grids on your own computer.
LIMITS = {
    "N": (2000, 1000),  # grid cells of the 1D cases (grid convergence included)
    "N_cavity": (256, 96),  # cells per side of the lid-driven cavity
    "cells_per_unit": (64, 32),  # Graetz channel
    "n_out": (200, 50),  # output times of the slab
    "T_end": (100.0, 20.0),  # end time of the slab (dimensionless)
    "levels": (10, 8),  # grids of one convergence study
    "fetch_files": (None, 30),  # station-year files kept on disk
}
STRATEGIES = ("identifiability", "multistart", "continuation", "regularise", "robust")
STATE: dict = {"public": False, "jobs": None, "wait_s": 30.0}


def lim(name: str):
    return LIMITS[name][1 if STATE["public"] else 0]


STATION_RE = re.compile(r"^[A-Z]{2}_[A-Za-z0-9_.\-]{1,60}$")
FINDINGS = {  # macro of the papers' generated numbers -> what it is (the value and registry ids come from numbers.tex)
    "paper1": [
        ("SlabChebRatio", "1D slab: Chebyshev solver reaches the PINN's error this many times faster"),
        ("SteadyChebRatioHi", "1D steady problem: Chebyshev speed advantage at equal error (largest)"),
        ("GThreeSlabFactor", "full training budget lowers the slab PINN error by this factor"),
        ("GThreeSlabWallH", "median full-budget training time of the slab PINN [h]"),
        ("GThreeFdMs", "finite-difference solve that is at least as accurate [ms]"),
        ("GThreeGrWorse", "2D channel: error grew by this factor at the full budget (collocation overfitting)"),
        ("GOneEstablished", "ablation effects established (of 12) at five seeds"),
        ("GTwoOneFail", "mean gradient cosine (PDE vs boundary loss) in failing 1D runs"),
        ("GrCovNinety", "coverage of nominal 90 % intervals, three-member 2D ensemble"),
        ("BreakGr", "queries after which the Graetz surrogate pays off"),
    ],
    "paper2": [
        ("VerPass", "manufactured-solution order checks passed (of 69)"),
        ("DVDmax", "largest deviation from de Vahl Davis' benchmark"),
        ("MCStdK", "Monte-Carlo sd of k / Cramer-Rao bound (efficient estimator)"),
        ("UQMedErr", "median error of the Graetz PINN surrogate inside its range"),
        ("CavOpposite", "cavity PINN: test points with the wrong sign of the Nusselt number"),
        ("PinnKmin", "PINN inversion: smallest error of k over variants"),
    ],
}


# ============================================================================ helpers
def clean(o):
    """JSON-safe: arrays -> lists, non-finite floats -> None, complex -> real part."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(np.real(o).tolist())
    if isinstance(o, np.generic):
        return clean(o.item())
    if isinstance(o, complex):
        return clean(o.real)
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    return o


def thin(a, n: int):
    """Every k-th row so that at most n rows remain (plots do not need every hourly sample of a long window)."""
    a = np.asarray(a)
    k = max(1, int(np.ceil(len(a) / n)))
    return a[::k]


def soil_cfg() -> dict:
    return yaml.safe_load(SOIL_CFG.read_text(encoding="utf-8"))


def findings() -> list[dict]:
    out = []
    for paper, items in FINDINGS.items():
        f = ROOT / "paper" / paper / "numbers.tex"
        if not f.exists():
            continue
        txt = f.read_text(encoding="utf-8")
        for name, what in items:
            m = re.search(r"\\newcommand\{\\" + name + r"\}\{(.*?)\}%\s*(.*)$", txt, re.M)
            if not m:
                continue
            val = m.group(1).replace("\\,", "\u2009").replace("\\ensuremath{-}", "\u2212").replace("\\%", "%")
            ids = re.findall(r"\bR(?:\d|B)[\w.\-]*\w", m.group(2))
            out.append({"paper": paper, "macro": name, "value": val, "what": what, "registry": ids})
    return out


def stations_on_disk() -> list[dict]:
    cfg = soil_cfg()["data"]
    folder = ROOT / cfg["folder"]
    out = []
    for p in sorted(folder.glob("*/CRNH0203-*.txt")):
        m = re.match(r"CRNH0203-(\d{4})-(.+)\.txt$", p.name)
        if m:
            out.append({"year": int(m.group(1)), "station": m.group(2), "default": m.group(2) == cfg["stations"][0]})
    return out


def station_path(station: str, year: int) -> Path:
    if not STATION_RE.match(station) or not 2000 <= int(year) <= 2100:
        raise ValueError("invalid station name or year")
    cfg = soil_cfg()["data"]
    return ROOT / cfg["folder"] / str(int(year)) / f"CRNH0203-{int(year)}-{station}.txt"


# ============================================================================ API
def api_info(_: dict) -> dict:
    from aerosynapse.thermal.app import backend as B

    cases = []
    for k, c in B.CASES.items():
        ok, why = B.pinn_available(k)
        cases.append(
            {
                "key": k,
                "title": c.title,
                "dim": c.dim,
                "transient": c.transient,
                "equation": c.equation,
                "params": [
                    {
                        "name": p.name,
                        "default": p.default,
                        "lo": p.lo,
                        "hi": p.hi,
                        "train": p.train,
                        "log": p.log,
                        "help": p.help,
                    }
                    for p in c.params
                ],
                "conditions": c.conditions,
                "pinn": ok,
                "pinn_note": why if not ok else c.pinn,
                "pinn_detail": c.pinn_note,
                "members": B.member_names(k) if ok else [],
            }
        )
    from aerosynapse.thermal.web import labs

    cfg = soil_cfg()
    dcfg = labs._data_cfg()
    build = ROOT / "BUILD_INFO.json"
    if build.exists():
        git = json.loads(build.read_text(encoding="utf-8")).get("git", "")
    else:
        from aerosynapse.thermal.utils.experiment import git_hash

        git = git_hash()[:7]
    return {
        "cases": cases,
        "stations": stations_on_disk(),
        "findings": findings(),
        "soil_defaults": cfg,
        "mode": "public" if STATE["public"] else "local",
        "build": git,
        "data_cases": {
            k: {
                "parameters": v["parameters"],
                "sizes": v["sizes"],
                "times": v["times"],
                "resolution": {"grid": v["grid"]} if "grid" in v else {"nx": v["nx"]},
            }
            for k, v in dcfg["cases"].items()
        },
        "data_defaults": {"seed": dcfg["seed"], "design": dcfg["design"], "noise": dcfg["noise"]},
        "labs": {k: labs.lim(k) for k in labs.LIMITS} | {"heating": list(labs.HEATING), "lids": list(labs.LIDS)},
    }


def api_solve(q: dict) -> dict:
    from aerosynapse.thermal.app import backend as B

    key = q.get("case", "B_slab")
    c = B.case(key)
    params = {p.name: float(q.get("params", {}).get(p.name, p.default)) for p in c.params}
    for p in c.params:
        if not p.lo <= params[p.name] <= p.hi:
            raise ValueError(f"{p.name} = {params[p.name]:g} outside the accepted range [{p.lo:g}, {p.hi:g}]")
    opts = {k: (float(v) if isinstance(v, int | float) else v) for k, v in (q.get("opts") or {}).items()}
    for k in ("N", "n_out", "cells_per_unit"):
        if k in opts:
            opts[k] = int(opts[k])
    n_max = lim("N_cavity") if key == "cavity" else lim("N")
    if opts.get("N", 0) > n_max or opts.get("cells_per_unit", 0) > lim("cells_per_unit"):
        raise ValueError(
            f"grid too fine for an interactive request (N <= {n_max}, cells_per_unit <= {lim('cells_per_unit')})"
        )
    if opts.get("n_out", 1) > lim("n_out") or not 0 < float(opts.get("T", 1.0)) <= lim("T_end"):
        raise ValueError(f"output times must be <= {lim('n_out')} and the end time in (0, {lim('T_end'):g}]")
    mode = q.get("mode", "compare")
    t0 = time.perf_counter()
    if mode == "compare" and B.pinn_available(key)[0]:
        r = B.compare(key, params, opts)
        r["mode"] = "compare"
    else:
        r = {"solver": B.solve(key, params, opts), "mode": "solver", "validity": B.validity(key, params, opts)}
    r["params"] = params
    r["request_s"] = time.perf_counter() - t0
    return r


def api_converge(q: dict) -> dict:
    from aerosynapse.thermal.app import backend as B

    key = q.get("case", "A_steady")
    c = B.case(key)
    params = {p.name: float(q.get("params", {}).get(p.name, p.default)) for p in c.params}
    ns = [int(n) for n in q.get("Ns") or []] or None
    if ns and (len(ns) > lim("levels") or not all(4 <= n <= lim("N") for n in ns)):
        raise ValueError(f"at most {lim('levels')} grids, each with 4 to {lim('N')} cells")
    schemes = list(dict.fromkeys(str(s) for s in q.get("schemes") or [])) or None
    if schemes and len(schemes) > 3:
        raise ValueError("at most three schemes")
    return B.convergence(key, params, ns, schemes, q.get("opts"))


def api_sweep(q: dict) -> dict:
    from aerosynapse.thermal.app import backend as B

    key = q.get("case", "B_slab")
    if key not in ("A_steady", "B_slab"):
        raise ValueError("the PINN parameter sweep is available for the 1D cases (A_steady, B_slab)")
    if not B.pinn_available(key)[0]:
        raise ValueError(f"no saved PINN ensemble for {key} on this machine")
    p = B.case(key).params[0]
    lo, hi = float(q.get("lo", p.lo)), float(q.get("hi", p.hi))
    vals = np.geomspace(max(lo, p.lo), min(hi, p.hi), int(min(q.get("n", 25), 60)))
    r = B.sweep_1d(key, vals)
    r["train"] = p.train
    return r


def api_inverse(q: dict) -> dict:
    from aerosynapse.thermal.app import backend as B

    un = tuple(x for x in q.get("unknown", ["alpha", "k", "h"]) if x in ("alpha", "k", "h"))
    kw = {
        "q_vol": float(q.get("q_vol", 1e5)),
        "noise": float(q.get("noise", 0.02)),
        "n_sensors": int(q.get("n_sensors", 8)),
        "unknown": un or ("alpha", "k", "h"),
        "seed": int(q.get("seed", 0)),
    }
    if not 0 <= kw["noise"] <= 0.5 or not 1 <= kw["n_sensors"] <= 64:
        raise ValueError("noise must be in [0, 0.5] and sensors in [1, 64]")
    csv = q.get("csv")
    if csv:
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as fh:
            fh.write(csv)
            path = fh.name
        try:
            return B.inverse(**kw, data_file=path)
        finally:
            os.unlink(path)
    return B.inverse(**kw)


def api_real_stations(_: dict) -> dict:
    from aerosynapse.thermal.inverse import realdata as R

    out = []
    for s in stations_on_disk():
        p = station_path(s["station"], s["year"])
        ser = R.load_uscrn(p)
        miss = np.isnan(ser.T).mean(0)
        out.append(
            s
            | {
                "depths_m": ser.z.tolist(),
                "missing_fraction": miss.tolist(),
                "latitude": ser.meta["latitude"],
                "longitude": ser.meta["longitude"],
            }
        )
    return {"stations": out}


def api_real_fetch(q: dict) -> dict:
    from aerosynapse.thermal.experiments.p10_fetch_real import fetch, sha256

    st, yr = str(q.get("station", "")), int(q.get("year", 2023))
    dest = station_path(st, yr)
    base = soil_cfg()["data"]["base_url"].rstrip("/")
    url = f"{base}/{yr}/{dest.name}"
    if dest.exists():
        return {"station": st, "year": yr, "downloaded": False, "bytes": dest.stat().st_size, "sha256": sha256(dest)}
    if lim("fetch_files") is not None and len(stations_on_disk()) >= lim("fetch_files"):
        raise ValueError(
            f"this public server keeps at most {lim('fetch_files')} station files; run THERMAL LAB locally"
        )
    try:
        info = fetch(url, dest)
    except Exception as err:  # noqa: BLE001 - the message goes back to the page
        if dest.exists():
            dest.unlink()
        raise ValueError(f"download failed from {url}: {err}") from err
    return {
        "station": st,
        "year": yr,
        "downloaded": True,
        "bytes": dest.stat().st_size,
        "sha256": sha256(dest),
        "url": url,
        **info,
    }


def _real_series(q: dict):
    from aerosynapse.thermal.inverse import realdata as R

    if q.get("source", "station") == "station":
        s = R.load_uscrn(station_path(q.get("station", soil_cfg()["data"]["stations"][0]), int(q.get("year", 2023))))
        return s, R.iso_to_seconds(s, q.get("start", "2023-07-01"))
    csv = q.get("csv") or ""
    if not csv.strip():
        raise ValueError("no table uploaded")
    pos = {k: float(v) for k, v in (q.get("positions") or {}).items()}
    if len(pos) < 1:
        raise ValueError("give the position (m) of every temperature column you want to use")
    s = R.load_table(csv, q.get("time_col", "time"), pos, q.get("time_unit", "auto"), q.get("ambient_col") or None)
    return s, float(s.t[0]) + float(q.get("start_offset_h", 0)) * 3600


def real_window(q: dict) -> tuple[dict, str]:
    """The data window of a Real-data request (station or uploaded table, sensors between top and bottom), checked for
    gaps: used by the PINN training of the Train tab."""
    from aerosynapse.thermal.inverse import realdata as R

    s, t0 = _real_series(q)
    days = float(q.get("days", 14))
    if not 1 <= days <= (21 if STATE["public"] else 120):
        raise ValueError("window must be 1-21 days on the public server (1-120 locally)")
    gap = int(q.get("max_gap_h", 3))
    zs = sorted(s.z.tolist())
    top, bot = float(q.get("top_m", zs[0])), float(q.get("bottom_m", zs[-1]))
    cols = [i for i, z in enumerate(s.z) if top - 1e-9 <= z <= bot + 1e-9]
    win = R.window(s, t0, days, gap, cols=cols)
    if len(win["t"]) < 10 or np.isnan(win["T"]).any():
        raise ValueError(f"the window has fewer than ten samples or gaps longer than {gap} h: choose another window")
    src = q.get("station", soil_cfg()["data"]["stations"][0]) if q.get("source", "station") == "station" else "table"
    return win, f"{src}: {days:g} days, sensors {', '.join(f'{z:g}' for z in win['z'])} m"


def api_flow(q: dict) -> dict:
    from aerosynapse.thermal.web import labs

    return labs.flow(q)


def api_dataset(q: dict) -> dict:
    from aerosynapse.thermal.web import labs

    return labs.dataset(q)


def api_train_start(q: dict) -> dict:
    from aerosynapse.thermal.web import labs

    return labs.train_start(q, real_window, STATE["jobs"])


def api_train_status(q: dict) -> dict:
    from aerosynapse.thermal.web import labs

    return labs.train_status(q)


class Binary:
    """A file answer (a ZIP) instead of JSON."""

    def __init__(self, data: bytes, filename: str, ctype: str = "application/zip"):
        self.data, self.filename, self.ctype = data, filename, ctype


def api_export(q: dict) -> Binary:
    from aerosynapse.thermal.web import labs

    return Binary(*labs.export(q))


def api_real_fit(q: dict) -> dict:
    from aerosynapse.thermal.inverse import realdata as R

    s, t0 = _real_series(q)
    days = float(q.get("days", 14))
    if not 1 <= days <= 120:
        raise ValueError("window must be 1-120 days")
    gap = int(q.get("max_gap_h", 3))
    zs = sorted(s.z.tolist())
    top, bot = float(q.get("top_m", zs[0])), float(q.get("bottom_m", zs[-1]))
    cols = [i for i, z in enumerate(s.z) if top - 1e-9 <= z <= bot + 1e-9]  # only the sensors the model uses
    win = R.window(s, t0, days, gap, cols=cols)
    if len(win["t"]) < 10:
        raise ValueError("fewer than ten samples in the window: check the start date / time column")
    if np.isnan(win["T"]).any():
        frac = np.isnan(s.T[:, cols]).mean(0)
        bad = [
            f"{z:g} m" + (" (sensor missing most of the record)" if f > 0.5 else "")
            for z, b, f in zip(win["z"], np.isnan(win["T"]).any(0), frac, strict=True)
            if b
        ]
        raise ValueError(
            f"gaps longer than {gap} h at {', '.join(bad)} in this window: choose another window "
            "or other boundary sensors"
        )
    zs = sorted(win["z"].tolist())
    mcfg = {
        "top_m": top,
        "bottom_m": bot,
        "inner_m": [float(v) for v in q.get("inner_m", zs[1:-1])],
        "dz_m": float(q.get("dz_m", 0.0025)),
        "dt_s": float(q.get("dt_s", 600)),
        "spinup_h": float(q.get("spinup_h", 24)),
    }
    if not mcfg["inner_m"]:
        raise ValueError("at least three sensor positions are needed (two boundaries and one inner sensor)")
    if not (
        5e-4 <= mcfg["dz_m"] <= 0.05
        and 30 <= mcfg["dt_s"] <= 3600
        and 0 <= mcfg["spinup_h"] <= 240
        and len(mcfg["inner_m"]) <= 8
        and len(q.get("layer_edges_m") or []) <= 5
    ):
        raise ValueError(
            "model grid out of range: dz 0.5 mm-5 cm, dt 30 s-1 h, spin-up 0-240 h, <= 8 inner sensors, <= 6 layers"
        )
    if "strategies" in q:
        q["strategies"] = [s for s in dict.fromkeys(map(str, q["strategies"] or [])) if s in STRATEGIES]
    fcfg = {
        **soil_cfg()["fit"],
        **{
            k: q[k]
            for k in ("layers", "offsets", "depths", "robust_loss", "alpha_init", "strategies", "tikhonov")
            if k in q
        },
    }
    edges = q.get("layer_edges_m")
    fcfg["layer_edges_m"] = [float(v) for v in edges] if edges else None
    tic = time.perf_counter()
    r = R.fit_conduction(win, mcfg, fcfg)
    out = {
        "rows": r["rows"],
        "fit": r["fit"],
        "attempts": [{k: v for k, v in a.items() if k != "x"} for a in r["attempts"]],
        "strategy_used": r["strategy_used"],
        "identifiability": r["identifiability"],
        "covariance": r["covariance"],
        "rmse": r["rmse_per_sensor"],
        "harmonic": r["harmonic"],
        "model": mcfg,
        "series": {
            "t_h": (win["t"] - win["t"][0]) / 3600,
            "z": win["z"],
            "T": win["T"],
            "pred": r["prediction"],
            "inner": mcfg["inner_m"],
        },
        "stamp0": win["stamp"][0] if win["stamp"] else None,
        "meta": {k: v for k, v in s.meta.items() if k not in ("air_temperature", "ambient")},
    }
    if q.get("holdout", True):
        te = R.window(s, t0 + days * R.DAY, days, gap)
        if len(te["t"]) >= 10 and not np.isnan(te["T"]).any():
            h = R.holdout(win, te, mcfg, fcfg, r["x"])
            out["holdout"] = {k: h[k] for k in ("rmse_model", "rmse_linear_baseline", "skill")} | {
                "t_h": (te["t"] - win["t"][0]) / 3600,
                "pred": h["prediction"],
                "obs": h["observed"],
                "baseline": h["baseline"],
            }
        else:
            out["holdout_note"] = "no complete hold-out window after the training window"
    out["wall_s"] = time.perf_counter() - tic
    return out


def api_real_lumped(q: dict) -> dict:
    from aerosynapse.thermal.inverse import realdata as R

    csv = q.get("csv") or ""
    col = q.get("temp_col")
    if not csv.strip() or not col:
        raise ValueError("upload a table and name its temperature column")
    s = R.load_table(
        csv, q.get("time_col", "time"), {col: 0.0}, q.get("time_unit", "auto"), q.get("ambient_col") or None
    )
    ok = np.isfinite(s.T[:, 0])
    amb = s.meta.get("ambient")
    m = R.Lumped(s.t[ok], s.T[ok, 0], None if amb is None else amb[ok], bool(q.get("fit_source", False)))
    T = s.T[ok, 0]
    x0 = [np.log(max((s.t[ok][-1] - s.t[ok][0]) / 3, 1.0))]
    lb, ub = [np.log(1e-3)], [np.log(1e9)]
    if amb is None:
        x0.append(float(T[-1]))
        lb.append(-100.0)
        ub.append(1000.0)
    x0.append(float(T[0]))
    lb.append(-100.0)
    ub.append(2000.0)
    if q.get("fit_source"):
        x0.append(0.0)
        lb.append(-1e3)
        ub.append(1e3)
    prob = R.Problem(m, m.names, np.array(x0), np.array(lb), np.array(ub))
    r = R.fit(prob, {**soil_cfg()["fit"], "strategies": ["identifiability", "multistart", "robust"]})
    cov = R.covariance(prob, r["x"], int(q.get("hac_lag", 10)))
    tau = float(np.exp(r["x"][0]))
    rows = [
        {
            "parameter": "tau [s]",
            "estimate": tau,
            "ci95_hac": [tau * float(np.exp(-1.96 * cov["sd_hac"][0])), tau * float(np.exp(1.96 * cov["sd_hac"][0]))],
        }
    ]
    for j, n in enumerate(m.names[1:], start=1):
        rows.append(
            {
                "parameter": n,
                "estimate": float(r["x"][j]),
                "ci95_hac": [float(r["x"][j] - 1.96 * cov["sd_hac"][j]), float(r["x"][j] + 1.96 * cov["sd_hac"][j])],
            }
        )
    out = {
        "rows": rows,
        "fit": r["result"],
        "strategy_used": r["strategy_used"],
        "identifiability": r["identifiability"],
        "attempts": [{k: v for k, v in a.items() if k != "x"} for a in r["attempts"]],
        "series": {"t_s": m.t, "T": T, "pred": prob.predict(r["x"]).real[:, 0]},
    }
    rho, c, va = (q.get(k) for k in ("rho", "c", "V_over_A"))
    if rho and c and va:
        out["h"] = float(rho) * float(c) * float(va) / tau
        out["h_note"] = "h = rho c (V/A) / tau, valid for a lumped body (Biot number hL/k << 0.1)"
    return out


def api_real_example(q: dict) -> dict:
    """Example tables for the upload path: a REAL 7-day soil profile cut from the default NOAA station, or a
    SYNTHETIC cooling curve (tau = 900 s, labelled as such) to check that the lumped fit recovers a known value."""
    from aerosynapse.thermal.inverse import realdata as R

    if q.get("kind") == "cooling":
        rng = np.random.default_rng(0)
        t = np.arange(0, 3600 * 2 + 1, 30.0)
        amb = 22 + 0.5 * np.sin(2 * np.pi * t / 3600)
        m = R.Lumped(t, np.zeros_like(t), amb)
        T = m.run([np.log(900.0), 85.0]).real[:, 0] + rng.normal(0, 0.15, t.size)
        lines = ["time_s,T_object_C,T_ambient_C"] + [
            f"{a:g},{b:.3f},{c:.3f}" for a, b, c in zip(t, T, amb, strict=True)
        ]
        return {
            "csv": "\n".join(lines),
            "name": "cooling_synthetic.csv",
            "note": "synthetic cooling curve with tau = 900 s and 0.15 K noise: the fit should recover tau",
        }
    cfg = soil_cfg()
    st = cfg["data"]["stations"][0]
    s = R.load_uscrn(station_path(st, int(cfg["data"]["year"])))
    w = R.window(s, R.iso_to_seconds(s, cfg["fit"]["start"]), 7, int(cfg["data"]["max_gap_h"]))
    cols = [0, 1, 2, 3]
    names = [f"T_{round(z * 100)}cm" for z in s.z[cols]]
    lines = ["time," + ",".join(names)] + [
        w["stamp"][i] + "," + ",".join(f"{w['T'][i, j]:.2f}" for j in cols) for i in range(len(w["t"]))
    ]
    return {
        "csv": "\n".join(lines),
        "name": f"{st}_7days.csv",
        "note": f"real data: NOAA USCRN {st.replace('_', ' ')}, 7 days from {cfg['fit']['start']}",
    }


ROUTES = {
    "/api/info": api_info,
    "/api/real/example": api_real_example,
    "/api/solve": api_solve,
    "/api/converge": api_converge,
    "/api/sweep": api_sweep,
    "/api/inverse": api_inverse,
    "/api/real/stations": api_real_stations,
    "/api/real/fetch": api_real_fetch,
    "/api/real/fit": api_real_fit,
    "/api/real/lumped": api_real_lumped,
    "/api/flow": api_flow,
    "/api/dataset": api_dataset,
    "/api/train/start": api_train_start,
    "/api/train/status": api_train_status,
    "/api/export": api_export,
}


# ============================================================================ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "ThermalLab/1.0"

    def log_message(self, fmt, *args):  # quieter console: one line per API call
        if self.path.startswith("/api"):
            super().log_message(fmt, *args)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(clean(obj), separators=(",", ":")).encode("utf-8"), "application/json")

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/info":
            return self._dispatch(path, {})
        if path == "/api/download/code":
            from aerosynapse.thermal.web import labs

            return self._file(Binary(*labs.code_zip()))
        if path == "/vendor/plotly.min.js":
            import plotly

            f = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
            return self._send(200, f.read_bytes(), "application/javascript")
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        f = (STATIC / rel).resolve()
        if STATIC not in f.parents or not f.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if f.suffix == ".js":
            ctype = "application/javascript"
        return self._send(200, f.read_bytes(), ctype)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return self._json(413, {"error": f"request larger than {MAX_BODY // 2**20} MB"})
        try:
            q = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "body is not JSON"})
        return self._dispatch(path, q)

    def _file(self, b: Binary) -> None:
        self.send_response(200)
        self.send_header("Content-Type", b.ctype)
        self.send_header("Content-Length", str(len(b.data)))
        self.send_header("Content-Disposition", f'attachment; filename="{b.filename}"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(b.data)

    def _dispatch(self, path: str, q: dict) -> None:
        code, out = dispatch(path, q)
        return self._file(out) if isinstance(out, Binary) else self._json(code, out)


def dispatch(path: str, q: dict) -> tuple[int, object]:
    """Run one API route: (HTTP status, JSON-able result or Binary). Shared by this server and by the Gradio-Space entry
    point (deploy/hf_space/app.py), so both answer identically: validation errors 400, busy 503, defects 500."""
    fn = ROUTES.get(path)
    if fn is None:
        return 404, {"error": f"no route {path}"}
    gate = STATE["jobs"] if path not in LIGHT_ROUTES else None
    if gate is not None and not gate.acquire(timeout=STATE["wait_s"]):
        return 503, {"error": "the server is busy with other visitors' computations: try again shortly"}
    try:
        return 200, fn(q)
    except (ValueError, KeyError, RuntimeError) as err:
        return 400, {"error": f"{type(err).__name__}: {err}"}
    except Exception as err:  # noqa: BLE001 - reported to the page, logged here
        traceback.print_exc()
        return int(HTTPStatus.INTERNAL_SERVER_ERROR), {"error": f"{type(err).__name__}: {err}"}
    finally:
        if gate is not None:
            gate.release()


# cheap routes are never queued behind computations; a training job takes a computation slot in its own thread
LIGHT_ROUTES = {"/api/info", "/api/real/stations", "/api/real/example", "/api/train/start", "/api/train/status"}


def make_server(port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, int(port)), Handler)


def configure(public: bool, max_jobs: int | None) -> None:
    """Public mode (tighter LIMITS) and the number of computations that may run at the same time (None: no limit)."""
    from aerosynapse.thermal.web import labs

    STATE["public"] = bool(public)
    labs.PUBLIC["on"] = bool(public)
    STATE["jobs"] = threading.BoundedSemaphore(max_jobs) if max_jobs else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    ap.add_argument("--host", default=None, help="interface to listen on (default 127.0.0.1; 0.0.0.0 with --public)")
    ap.add_argument("--public", action="store_true", help="public deployment: tighter limits, queued computations")
    ap.add_argument("--max-jobs", type=int, default=None, help="computations at the same time (default: 2 if public)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    configure(a.public, a.max_jobs if a.max_jobs is not None else (2 if a.public else None))
    host = a.host or ("0.0.0.0" if a.public else "127.0.0.1")
    srv = make_server(a.port, host)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"THERMAL LAB on {url}  (listening on {host}; public={a.public}; Ctrl+C to stop)", flush=True)
    if not a.no_browser and not a.public:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
