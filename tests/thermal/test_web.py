"""Phase 10.2 web demo: the local server (src/web/server.py) over real HTTP on an ephemeral port. Uses only what a
clean clone has (no PINN checkpoints, no downloaded data): solver mode, the synthetic inverse, uploaded tables."""

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from aerosynapse.thermal.web import server as W


@pytest.fixture(scope="module")
def base():
    srv = W.make_server(0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def call(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            raw, code, ctype = r.read(), r.status, r.headers.get("Content-Type")
    except urllib.error.HTTPError as e:
        raw, code, ctype = e.read(), e.code, e.headers.get("Content-Type")
    return code, (json.loads(raw) if ctype == "application/json" else raw)


def test_static_page_and_security(base):
    code, html = call(base, "/")
    assert code == 200 and b"THERMAL" in html and b"app.js" in html
    for p in ("/app.js", "/style.css"):
        assert call(base, p)[0] == 200
    assert call(base, "/../server.py")[0] == 404  # no path traversal out of the static folder
    assert call(base, "/%2e%2e/server.py")[0] == 404
    code, j = call(base, "/api/real/fetch", {"station": "../../etc", "year": 2023})
    assert code == 400 and "invalid station" in j["error"]
    assert call(base, "/api/nothing", {})[0] == 404


def test_info_lists_cases_and_findings(base):
    code, j = call(base, "/api/info")
    assert code == 200
    keys = {c["key"] for c in j["cases"]}
    assert {"A_steady", "B_slab", "D_graetz", "cavity"} <= keys
    assert all("registry" in f for f in j["findings"])
    assert j["soil_defaults"]["model"]["kind"] == "conduction1d"


def test_solve_validates_and_answers(base):
    code, j = call(base, "/api/solve", {"case": "A_steady", "params": {"Pe": 10}, "mode": "solver"})
    assert code == 200 and j["mode"] == "solver" and j["solver"]["err_inf"] < 1e-2
    code, j = call(base, "/api/solve", {"case": "B_slab", "params": {"Bi": 1e6}})
    assert code == 400 and "outside the accepted range" in j["error"]


def test_inverse_identifiability_switch(base):
    code, j = call(base, "/api/inverse", {"q_vol": 0})
    assert code == 200 and j["identifiable"] is False and j["rank"] == 2
    code, j = call(base, "/api/inverse", {})
    assert code == 200 and j["identifiable"] is True and len(j["rows"]) == 3


def wave_upload(a=4e-7, days=4):
    """A synthetic damped diurnal wave at four depths (diffusivity a), as a user would upload it."""
    w = 2 * np.pi / 86400
    d = np.sqrt(2 * a / w)
    t = np.arange(0, days * 86400, 3600.0)
    zs = [0.05, 0.10, 0.20, 0.50]
    rows = ["time_s," + ",".join(f"T_{round(z * 100)}cm" for z in zs)]
    for tt in t:
        rows.append(f"{tt:g}," + ",".join(f"{15 + 8 * np.exp(-z / d) * np.sin(w * tt - z / d):.5f}" for z in zs))
    return {
        "source": "upload",
        "csv": "\n".join(rows),
        "time_col": "time_s",
        "time_unit": "s",
        "days": days,
        "positions": {f"T_{round(z * 100)}cm": z for z in zs},
        "top_m": 0.05,
        "bottom_m": 0.5,
        "inner_m": [0.1, 0.2],
        "holdout": False,
    }


def test_uploaded_profile_and_cooling_curve(base):
    a = 4e-7
    code, j = call(base, "/api/real/fit", wave_upload(a))
    assert code == 200, j
    assert j["fit"]["converged"] and j["rows"][0]["estimate"] == pytest.approx(a, rel=0.02)
    code, j = call(base, "/api/real/example", {"kind": "cooling"})
    assert code == 200 and "synthetic" in j["note"]
    code, j = call(
        base,
        "/api/real/lumped",
        {
            "csv": j["csv"],
            "time_col": "time_s",
            "temp_col": "T_object_C",
            "ambient_col": "T_ambient_C",
            "rho": 7800,
            "c": 500,
            "V_over_A": 0.005,
        },
    )
    assert code == 200 and j["fit"]["converged"]
    assert j["rows"][0]["estimate"] == pytest.approx(900.0, rel=0.02) and j["h"] > 0


def test_request_limits_public_mode_and_job_queue(base):
    # an uploaded "table" that names a file on the server is parsed as text, never opened
    code, j = call(base, "/api/real/lumped", {"csv": str(W.ROOT / "pyproject.toml"), "temp_col": "T"})
    assert code == 400 and "not in the table" in j["error"] and "[project]" not in j["error"]
    # model grids a request may not ask for, in any mode
    code, j = call(base, "/api/real/fit", wave_upload(days=2) | {"dz_m": 1e-5})
    assert code == 400 and "model grid out of range" in j["error"]
    code, j = call(base, "/api/converge", {"case": "A_steady", "Ns": [10, 10**6]})
    assert code == 400 and "grids" in j["error"]
    code, j = call(base, "/api/sweep", {"case": "D_graetz"})  # was a 500 (AttributeError) before the check
    assert code == 400 and "1D cases" in j["error"]
    try:
        W.configure(public=True, max_jobs=1)
        code, j = call(base, "/api/solve", {"case": "D_graetz", "opts": {"cells_per_unit": 48}, "mode": "solver"})
        assert code == 400 and "cells_per_unit <= 32" in j["error"]  # allowed locally (<= 64), not in public
        code, j = call(base, "/api/solve", {"case": "cavity", "opts": {"N": 200}, "mode": "solver"})
        assert code == 400 and "N <= 96" in j["error"]
        wait = W.STATE["wait_s"]
        W.STATE["wait_s"] = 0.2
        assert W.STATE["jobs"].acquire(timeout=1)  # another visitor's computation holds the only slot
        try:
            code, j = call(base, "/api/solve", {"case": "A_steady", "mode": "solver"})
            assert code == 503 and "busy" in j["error"]
            assert call(base, "/api/info")[0] == 200  # light routes are never queued
        finally:
            W.STATE["jobs"].release()
            W.STATE["wait_s"] = wait
        assert call(base, "/api/solve", {"case": "A_steady", "mode": "solver"})[0] == 200
    finally:
        W.configure(public=False, max_jobs=None)
