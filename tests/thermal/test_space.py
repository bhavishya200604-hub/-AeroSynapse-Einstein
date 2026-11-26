"""Phase 10.3: the Hugging Face Space bundle (deploy/hf_space/build.py) holds what the page needs and nothing that
must stay private (data/), and its Docker files agree with each other (port, public mode)."""

import hashlib
import importlib.util
import json
import re
from pathlib import Path

from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as ROOT  # noqa: E402, F401
SPACE = ROOT / "deploy" / "hf_space"


def _build():
    spec = importlib.util.spec_from_file_location("hf_space_build", SPACE / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bundle_contents_and_record(tmp_path):
    out = tmp_path / "space"
    info = _build().assemble(out, allow_missing_checkpoints=True)  # default: the free Gradio / ZeroGPU Space
    for f in ("app.py", "README.md", "requirements.txt", "fetch_data.py", "smoke.py", "BUILD_INFO.json"):
        assert (out / f).is_file(), f
    assert not (out / "Dockerfile").exists() and info["sdk"] == "gradio"
    for f in (
        "src/web/server.py",
        "src/web/static/index.html",
        "src/inverse/realdata.py",
        "experiments/p10_fetch_real.py",
        "configs/real/soil.yaml",
        "paper/paper1/numbers.tex",
        "results/p6_2_uq_default/config.yaml",
        "results/p10_fetch_default/metrics.json",
    ):
        assert (out / f).is_file(), f
    files = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()]
    assert not [f for f in files if f.startswith("data/") or "__pycache__" in f or f.endswith((".npz", ".h5"))]
    assert not (out / "docs").exists() and not (out / "examples").exists()
    rec = json.loads((out / "BUILD_INFO.json").read_text(encoding="utf-8"))
    assert rec["files"] == info["files"] == len(files) - 1  # every file but BUILD_INFO.json itself
    f = "src/web/server.py"
    assert rec["sha256"][f] == hashlib.sha256((out / f).read_bytes()).hexdigest()


def test_docker_files_agree():
    docker = (SPACE / "Dockerfile").read_text(encoding="utf-8")
    card = (SPACE / "README.md").read_text(encoding="utf-8")
    front = card.split("---")[1]
    port = re.search(r"^app_port:\s*(\d+)", front, re.M).group(1)
    assert re.search(r"^sdk:\s*docker", front, re.M) and f"PORT={port}" in docker and f"EXPOSE {port}" in docker
    assert '"--public"' in docker and "fetch_data.py" in docker and "useradd -m -u 1000" in docker
    assert len(re.search(r"^short_description:\s*(.+)$", front, re.M).group(1)) <= 60
    pins = dict(line.split("==") for line in (SPACE / "requirements.txt").read_text().splitlines() if "==" in line)
    assert {"numpy", "scipy", "pandas", "plotly", "pyyaml"} <= set(pins)


def test_docker_bundle_and_gradio_card(tmp_path):
    docker = _build().assemble(tmp_path / "d", allow_missing_checkpoints=True, sdk="docker")
    assert (tmp_path / "d" / "Dockerfile").is_file() and not (tmp_path / "d" / "app.py").exists()
    assert docker["sdk"] == "docker"
    card = (SPACE / "README_gradio.md").read_text(encoding="utf-8").split("---")[1]
    assert re.search(r"^sdk:\s*gradio", card, re.M) and re.search(r"^app_file:\s*app.py", card, re.M)
    assert re.search(r'^python_version:\s*"3\.12"', card, re.M)  # ZeroGPU runs Python 3.12 / 3.10
    req = (SPACE / "requirements_gradio.txt").read_text(encoding="utf-8")
    lines = [x.split("#")[0].strip() for x in req.splitlines()]
    assert not [x for x in lines if x.startswith("spaces")]  # the platform pins spaces itself
    torch_pin = next(x for x in lines if x.startswith("torch=="))
    assert torch_pin.split("==")[1] in {"2.8.0", "2.9.1", "2.10.0", "2.11.0", "2.12.1", "2.13.0"}  # ZeroGPU versions
    app = (SPACE / "app.py").read_text(encoding="utf-8")
    assert "@spaces.GPU" in app and 'setdefault("PINN_DEVICE", "cpu")' in app and "W.dispatch" in app
