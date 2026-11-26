"""Running the apps' pages for Studio.

``python -m aerosynapse.nativelab list OUT.json``             every app and its pages
``python -m aerosynapse.nativelab run REQUEST.json OUT_DIR``   one page, once
``python -m aerosynapse.nativelab serve DIR``                   a worker: Studio drops ``req_<n>.json`` into DIR and
                                                              reads ``res_<n>/`` (it keeps imports, caches and each
                                                              app's session state between pages, so a page re-runs fast)

A request: ``{"app": key, "page": name, "values": {widget key: value}, "press": button key or null,
"context": the design and assembly (see nativelab.context)}``. A result folder holds ``ui.json`` (the page's inputs:
key, kind, label, value, default, options, range, help, where the value came from), ``blocks.json`` (what the page
shows, in order), the figures (PNG), tables (CSV), downloads, ``summary.txt`` and ``done.json``.
"""

from __future__ import annotations

import importlib
import json
import os
import runpy
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "aerosynapse"

APPS: dict[str, dict[str, Any]] = {
    "thermal_lab": {
        "title": "THERMAL LAB",
        "about": "The convection-diffusion PINN project: solvers, PINNs, inverse problems, real soil data, cavity flow, datasets, training.",
        "kind": "pages",
        "module": "aerosynapse.nativelab.thermal_lab_pages",
    },
    "airfrans": {
        "title": "AirfRANS (Kaggle data)",
        "about": "Flow-field surrogates trained on the 1000 AirfRANS airfoil simulations in the project folder: dataset, results, predictions, training.",
        "kind": "pages",
        "module": "aerosynapse.nativelab.airfrans_pages",
    },
    "thermal_app": {
        "title": "PINNs for heat transfer",
        "about": "The heat-transfer PINN app: solver, PINN, comparison, inverse, convergence, training monitor, uncertainty, export.",
        "kind": "script",
        "path": SRC / "thermal" / "app" / "main.py",
        "page_key": "section",
        "pages": [
            "Overview",
            "Solver",
            "PINN",
            "Comparison",
            "Inverse",
            "Convergence",
            "Training monitor",
            "Uncertainty & validity",
            "Export",
        ],
    },
    "designer": {
        "title": "Designer",
        "about": "The aircraft and fighter designer: design, mission, analysis, optimisation, studies, shapes, data, workflow, materials, data store.",
        "kind": "call",
        "module": "aerosynapse.ui.designer",
        "func": "main",
        "page_key": "page",
    },
    "workbench": {
        "title": "Research workbench",
        "about": "Physics, simulation, ML, optimisation and research workspaces.",
        "kind": "script",
        "path": SRC / "ui" / "app.py",
        "page_key": "workspace",
        "pages": ["Physics", "Simulation", "ML", "Optimization", "Research"],
    },
}
SESSIONS: dict[str, Any] = {}


def _pages(app: str) -> list[str]:
    a = APPS[app]
    if a["kind"] == "pages":
        return list(importlib.import_module(a["module"]).PAGES)
    if "pages" in a:
        return list(a["pages"])
    from aerosynapse.nativelab import shim

    shim.install()
    mod = importlib.import_module(a["module"])
    return list(getattr(mod, "PAGES", {}))


def listing() -> dict[str, Any]:
    out = []
    for k, a in APPS.items():
        try:
            pages = _pages(k)
            err = ""
        except Exception as e:  # noqa: BLE001
            pages, err = [], f"{type(e).__name__}: {e}"
        out.append({"key": k, "title": a["title"], "about": a["about"], "pages": pages, "error": err})
    return {"apps": out}


def run_page(req: dict[str, Any], out: Path, progress: Path | None = None) -> dict[str, Any]:
    from aerosynapse.nativelab import context as C
    from aerosynapse.nativelab import shim

    shim.install()
    app, page = req["app"], req.get("page")
    a = APPS[app]
    ctx = req.get("context") or {}
    defaults = C.label_defaults(ctx) if req.get("use_design", True) else {}
    values = dict(req.get("values") or {})
    if a.get("page_key") and page:
        values[a["page_key"]] = page
    sess = SESSIONS.setdefault(app, shim._State())
    sess["aerosynapse_context"] = ctx
    press = req.get("press")
    t0 = time.perf_counter()
    rec = None
    for _attempt in range(4):
        rec = shim.Recorder(out, values, press, defaults, progress)
        shim.begin(rec, sess)
        try:
            if a["kind"] == "pages":
                T = importlib.reload(importlib.import_module(a["module"]))
                T.PAGES[page or next(iter(T.PAGES))]()
            elif a["kind"] == "script":
                cwd = os.getcwd()
                os.chdir(ROOT)
                try:
                    runpy.run_path(str(a["path"]), run_name="__main__")
                finally:
                    os.chdir(cwd)
            else:
                mod = importlib.import_module(a["module"])
                cwd = os.getcwd()
                os.chdir(ROOT)
                try:
                    getattr(mod, a["func"])()
                finally:
                    os.chdir(cwd)
        except shim.StopPage:
            pass
        except shim.RerunPage:
            press = None
            continue
        except Exception as e:  # noqa: BLE001
            rec.block("error", text=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:])
        break
    assert rec is not None
    if ctx:
        rec.blocks.insert(0, {"type": "context", "text": C.summary(ctx)})
    (out / "ui.json").write_text(
        json.dumps({"app": app, "page": page, "widgets": rec.widgets}, indent=1, default=str), encoding="utf-8"
    )
    (out / "blocks.json").write_text(
        json.dumps(
            {"app": app, "page": page, "blocks": rec.blocks, "seconds": time.perf_counter() - t0}, indent=1, default=str
        ),
        encoding="utf-8",
    )
    (out / "summary.txt").write_text(_summary(app, page, rec.blocks), encoding="utf-8")
    done = {
        "ok": not any(b["type"] == "error" for b in rec.blocks),
        "seconds": time.perf_counter() - t0,
        "widgets": len(rec.widgets),
        "blocks": len(rec.blocks),
    }
    (out / "done.json").write_text(json.dumps(done), encoding="utf-8")
    return done


def _summary(app: str, page: str | None, blocks: list[dict[str, Any]]) -> str:
    lines = [f"{APPS[app]['title']}: {page or ''}".upper(), "=" * 60]
    for b in blocks:
        t = b["type"]
        if t in ("title", "header"):
            lines += ["", b.get("text", "").upper()]
        elif t == "subheader":
            lines += ["", b.get("text", ""), "-" * min(60, len(b.get("text", "")))]
        elif t == "metric":
            lines.append(f"  {b['label']}: {b['value']}" + (f"  ({b['delta']})" if b.get("delta") else ""))
        elif t in ("figure", "table", "download"):
            lines.append(f"  [{t}: {b.get('file')}]")
        elif t != "progress" and b.get("text"):
            lines.append(("  " if t not in ("markdown", "text", "context") else "") + str(b.get("text"))[:3000])
    return "\n".join(lines) + "\n"


def serve(d: Path) -> None:
    """The worker loop: one request at a time, until Studio leaves or two idle hours pass."""
    d.mkdir(parents=True, exist_ok=True)
    import shutil

    for old in list(d.glob("res_*")) + list(d.glob("progress_*")):  # the answers of earlier sessions
        if time.time() - old.stat().st_mtime > 60:
            if old.is_dir():
                shutil.rmtree(old, ignore_errors=True)
            else:
                old.unlink(missing_ok=True)
    (d / "worker.json").write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
    last = time.time()
    while time.time() - last < 7200:
        reqs = sorted(d.glob("req_*.json"), key=lambda p: p.stat().st_mtime)
        if not reqs:
            time.sleep(0.1)
            continue
        p = reqs[0]
        try:
            req = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(0.1)
            continue
        p.unlink(missing_ok=True)
        rid = p.stem[4:]
        out = d / f"res_{rid}"
        try:
            run_page(req, out, d / f"progress_{rid}.json")
        except Exception as e:  # noqa: BLE001
            out.mkdir(parents=True, exist_ok=True)
            (out / "done.json").write_text(
                json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]})
            )
        last = time.time()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    import matplotlib

    matplotlib.use("Agg")
    if not argv or argv[0] == "list":
        js = json.dumps(listing(), indent=1)
        if len(argv) > 1:
            Path(argv[1]).write_text(js, encoding="utf-8")
        else:
            print(js)
        return 0
    if argv[0] == "run":
        req = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        print(json.dumps(run_page(req, Path(argv[2]))))
        return 0
    if argv[0] == "serve":
        serve(Path(argv[1]))
        return 0
    print(__doc__)
    return 2
