"""Phase 10.4-10.6: flow lab, data factory, real-data PINN training jobs and downloads, over real HTTP (src/web/labs.py
through src/web/server.py). Uses only what a clean clone has: the training reads an uploaded synthetic table."""

import io
import json
import threading
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pytest

from aerosynapse.thermal.inverse import realpinn as P
from aerosynapse.thermal.web import labs
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
        with urllib.request.urlopen(req, timeout=600) as r:
            raw, code, ctype, disp = (
                r.read(),
                r.status,
                r.headers.get("Content-Type"),
                r.headers.get("Content-Disposition"),
            )
    except urllib.error.HTTPError as e:
        raw, code, ctype, disp = e.read(), e.code, e.headers.get("Content-Type"), None
    return code, (json.loads(raw) if ctype == "application/json" else raw), disp


def wave_table(a=4e-7, days=3):
    w = 2 * np.pi / 86400
    d = np.sqrt(2 * a / w)
    zs = [0.05, 0.10, 0.20, 0.50]
    rows = ["time_s," + ",".join(f"T_{round(z * 100)}cm" for z in zs)]
    for tt in np.arange(0, days * 86400, 3600.0):
        rows.append(f"{tt:g}," + ",".join(f"{15 + 8 * np.exp(-z / d) * np.sin(w * tt - z / d):.5f}" for z in zs))
    return {
        "source": "upload",
        "csv": "\n".join(rows),
        "time_col": "time_s",
        "time_unit": "s",
        "days": days,
        "positions": {f"T_{round(z * 100)}cm": z for z in zs},
    }


def test_flow_lab_runs_and_validates(base):
    code, r, _ = call(base, "/api/flow", {"grid": 12, "frames": 3, "t_end": 1.0})
    assert code == 200, r
    assert np.shape(r["theta"]) == (3, 12, 12) and np.shape(r["u"]) == (3, 12, 12) and len(r["nu_top"]) == 3
    assert r["max_div"] < 1e-10  # the projection keeps the velocity divergence-free
    code, r, _ = call(
        base, "/api/flow", {"heating": "side_heated", "lid": "none", "Ri": 5, "grid": 12, "frames": 2, "t_end": 1.0}
    )
    assert code == 200 and len(r["nu_left"]) == 2 and abs(r["nu_top"][-1]) < 1e-12  # adiabatic top wall
    code, r, _ = call(base, "/api/flow", {"heating": "nonsense"})
    assert code == 400 and "heating must be one of" in r["error"]
    code, r, _ = call(base, "/api/flow", {"grid": 500})
    assert code == 400 and "grid" in r["error"]


def test_data_factory_design_and_leakage(base):
    body = {
        "case": "caseB",
        "sizes": {"train": 4, "val": 1, "interp": 2, "extrap": 2},
        "nx": 65,
        "n_times": 3,
        "n_sensors": 5,
        "noise_levels": [0, 0.05],
    }
    code, r, _ = call(base, "/api/dataset", body)
    assert code == 200, r
    assert len(r["design"]) == 9 and r["leakage"]["all_ok"] and r["shapes"]["theta"] == [9, 3, 65]
    assert np.shape(r["sensors"]["noisy"]) == (2, 3, 5)
    noisy = np.asarray(r["sensors"]["noisy"])
    assert np.allclose(noisy[0], r["sensors"]["clean"]) and not np.allclose(noisy[1], r["sensors"]["clean"])
    code, z, disp = call(base, "/api/export", {"kind": "dataset", "id": r["id"]})
    assert code == 200 and "attachment" in disp
    names = zipfile.ZipFile(io.BytesIO(z)).namelist()
    assert {"dataset.npz", "dataset_card.json", "reproduce.py", "figures/design_sensors.png"} <= set(names)
    npz = np.load(io.BytesIO(zipfile.ZipFile(io.BytesIO(z)).read("dataset.npz")))
    assert npz["field_theta"].shape == (9, 3, 65)


def test_exports_hold_data_figures_and_a_working_script(base):
    code, z, _ = call(base, "/api/export", {"kind": "converge", "request": {"case": "A_steady"}})
    assert code == 200
    zf = zipfile.ZipFile(io.BytesIO(z))
    names = set(zf.namelist())
    assert {"README.txt", "request.json", "result.json", "reproduce.py", "figures/convergence.png"} <= names
    assert any(n.startswith("tables/") and n.endswith(".csv") for n in names)
    assert zf.read("figures/convergence.png")[:8] == b"\x89PNG\r\n\x1a\n"
    # the script re-runs the request with the project's code
    out = labs.run_kind("converge", json.loads(zf.read("request.json")))
    assert out["schemes"]["central"]["observed_order"] == pytest.approx(
        json.loads(zf.read("result.json"))["schemes"]["central"]["observed_order"]
    )
    code, r, _ = call(base, "/api/export", {"kind": "unknown", "request": {}})
    assert code == 400


def test_code_download_has_no_data_or_networks(base):
    code, z, disp = call(base, "/api/download/code")
    assert code == 200 and "thermal_lab_code.zip" in disp
    names = zipfile.ZipFile(io.BytesIO(z)).namelist()
    assert "thermal_lab_code/src/web/server.py" in names and "thermal_lab_code/HOW_TO_RUN.txt" in names
    assert not [n for n in names if n.startswith("thermal_lab_code/data/") or n.endswith((".pt", ".npz", ".h5"))]


def test_training_job_on_an_uploaded_table(base):
    body = wave_table() | {
        "train_inner": [0.2],
        "holdout": 0.1,
        "adam_steps": 60,
        "lbfgs_steps": 0,
        "warmup": 20,
        "n_colloc": 200,
        "width": 8,
        "depth": 2,
    }
    code, r, _ = call(base, "/api/train/start", body)
    assert code == 200, r
    for _ in range(600):
        code, st, _ = call(base, "/api/train/status", {"id": r["id"]})
        if st["state"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.5)
    assert st["state"] == "done", st
    res = st["result"]
    assert res["alpha"] > 0 and "classical" in res and np.isfinite(res["classical"]["rmse_holdout"])
    assert set(res["rmse"]) == {"0.05", "0.1", "0.2", "0.5"} and st["history"][-1]["step"] == 59
    code, z, _ = call(base, "/api/export", {"kind": "train", "id": r["id"]})
    assert code == 200 and "figures/training.png" in zipfile.ZipFile(io.BytesIO(z)).namelist()
    code, r2, _ = call(base, "/api/train/start", body | {"holdout": 0.2})  # 0.2 is a training sensor
    assert code == 400 and "held-out" in r2["error"]


def test_realpinn_recovers_a_known_diffusivity():
    # synthetic damped diurnal wave with alpha = 4e-7: training on 5, 20, 50 cm recovers alpha and predicts 10 cm
    a = 4e-7
    w = 2 * np.pi / 86400
    d = np.sqrt(2 * a / w)
    t = np.arange(0, 4 * 86400, 3600.0)
    z = np.array([0.05, 0.10, 0.20, 0.50])
    T = 15 + 8 * np.exp(-z[None, :] / d) * np.sin(w * t[:, None] - z[None, :] / d)
    win = {"t": t, "z": z, "T": T}
    r = P.train(win, [0.05, 0.2, 0.5], 0.05, 0.5, seed=0, adam_steps=1500, lbfgs_steps=200, warmup=300, n_colloc=800)
    assert r["alpha"] == pytest.approx(a, rel=0.15)
    assert P.rmse_after(t, T[:, 1], r["prediction"][:, 1], 86400) < 0.3
