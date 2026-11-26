"""Phase 10.3: the public Hugging Face Space of THERMAL LAB, assembled and tested end to end on this machine.

    python -m aerosynapse.thermal.experiments.p10_3_space            # writes results/p10_3_space_default/

1. assemble the Space (deploy/hf_space/build.py) into build/hf_space;
2. copy it to a temporary folder and run what the Docker image runs at build time (fetch_data.py: the NOAA files,
   checked against the SHA-256 recorded by Step 10.1);
3. start the server from that copy exactly as the image does (--public, $PORT) and run the smoke checks
   (deploy/hf_space/smoke.py) with the saved networks required;
4. record the bundle (files, size, checkpoints, project commit), the NOAA provenance and every check.

The Docker image itself is built and smoke-tested on Linux by .github/workflows/space.yml (no Docker here).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash

SPACE = ROOT / "deploy" / "hf_space"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(f"hf_space_{name}", SPACE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(tag: str) -> dict:
    build, smoke = _module("build"), _module("smoke")
    t0 = time.time()
    info = build.assemble()
    out = {
        "bundle": {k: info[k] for k in ("git", "files", "bytes", "checkpoints")} | {"assemble_s": time.time() - t0},
    }
    # ignore_cleanup_errors: on Windows the venv's python.exe is a launcher whose child interpreter (the server) ends a
    # moment after the launcher is terminated and may still hold server.log when the folder is removed
    with tempfile.TemporaryDirectory(prefix="thermal_lab_space_", ignore_cleanup_errors=True) as tmp:
        app = Path(tmp) / "app"
        shutil.copytree(build.OUT, app)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"MPLBACKEND": "Agg"}
        t0 = time.time()
        r = subprocess.run([sys.executable, "fetch_data.py"], cwd=app, env=env, capture_output=True, text=True)
        prov = json.loads((app / "data/real/uscrn/PROVENANCE.json").read_text(encoding="utf-8"))
        out["noaa"] = {k: prov[k] for k in ("files", "downloaded", "hash_matches")} | {
            "fetch_s": time.time() - t0,
            "exit": r.returncode,
        }
        port = _free_port()
        log = open(app / "server.log", "w", encoding="utf-8")
        srv = subprocess.Popen(
            [sys.executable, "-m", "aerosynapse.thermal.web.server", "--public", "--no-browser"],
            cwd=app,
            env=env | {"PORT": str(port)},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            t0 = time.time()
            up = smoke.wait_up(base, 180)
            out["server"] = {"up": up, "start_s": time.time() - t0, "public": True}
            checks = smoke.run_checks(base, pinn_required=True) if up else []
        finally:
            srv.terminate()
            srv.wait(timeout=30)
            log.close()
            t1 = time.time()  # wait until the server itself (not only the launcher) has released its port
            while time.time() - t1 < 30:
                with socket.socket() as s:
                    if s.connect_ex(("127.0.0.1", port)) != 0:
                        break
                time.sleep(0.5)
        text = (app / "server.log").read_text(encoding="utf-8", errors="replace")
        out["server"]["tracebacks"] = text.count("Traceback")
        out["server"]["listening_on_all_interfaces"] = "listening on 0.0.0.0" in text
    out["checks"] = checks
    out["passed"] = sum(c["ok"] for c in checks)
    out["total"] = len(checks)
    out["pass"] = bool(
        checks
        and out["passed"] == out["total"]
        and out["noaa"]["downloaded"] == out["noaa"]["files"]
        and out["server"]["tracebacks"] == 0
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="default")
    a = ap.parse_args()
    with Experiment(f"p10_3_space_{a.tag}", {"seed": 0, "space": "deploy/hf_space", "port": "free"}) as ex:
        ex.metrics.update(run(a.tag) | {"git": git_hash()})
    m = ex.metrics
    print(
        f"Space bundle: {m['bundle']['files']} files, {m['bundle']['bytes'] / 2**20:.1f} MB, "
        f"{m['bundle']['checkpoints']} checkpoints; NOAA {m['noaa']['hash_matches']}/{m['noaa']['files']} hashes "
        f"match; smoke {m['passed']}/{m['total']}; server tracebacks {m['server']['tracebacks']}; "
        f"{'PASS' if m['pass'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
