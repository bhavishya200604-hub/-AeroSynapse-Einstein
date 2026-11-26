"""Python access to the C++ programs: ``aerokernel`` (geometry), ``aeroflow`` (3D flow) and ``AeroSynapseStudio``.

A design is a JSON feature tree (see ``cpp/common/sdf.hpp`` for every feature type and modifier). The programs
are built on first use with the MinGW g++ found on the PATH (or the ``AEROSYNAPSE_CXX`` compiler), and rebuilt
when a source file is newer than the program.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerosynapse.exceptions import ConfigurationError

ROOT = Path(__file__).resolve().parents[3]
CPP = ROOT / "cpp"
BIN = ROOT / "bin"
FLAGS = ["-std=gnu++14", "-O3", "-march=native", "-mfpmath=sse", "-msse2", "-fno-math-errno", "-static"]
PROGRAMS = {
    "aerokernel": (["kernel/aerokernel.cpp"], []),
    "aeroflow": (["flow/aeroflow.cpp"], []),
    "AeroSynapseStudio": (["studio/studio.cpp", "studio/studio.res.o"], ["-mwindows", "-lopengl32", "-lglu32", "-lgdi32", "-lcomctl32", "-lcomdlg32", "-lshell32", "-lole32", "-lgdiplus"]),
}
Design = dict[str, Any]


def compiler() -> str:
    c = os.environ.get("AEROSYNAPSE_CXX") or shutil.which("g++")
    if not c:
        raise ConfigurationError("no C++ compiler: install MinGW g++ or set AEROSYNAPSE_CXX")
    return c


def build(name: str, force: bool = False) -> Path:
    """Compile one program if it is missing or older than its sources."""
    srcs, extra = PROGRAMS[name]
    exe = BIN / (name + (".exe" if os.name == "nt" else ""))
    deps = [CPP / s for s in srcs if not s.endswith(".o")] + list((CPP / "common").glob("*.hpp"))
    if name == "AeroSynapseStudio":
        deps += list((CPP / "studio").glob("*"))
    if exe.exists() and not force and all(d.stat().st_mtime <= exe.stat().st_mtime for d in deps if d.exists()):
        return exe
    BIN.mkdir(exist_ok=True)
    if name == "AeroSynapseStudio":
        rc = CPP / "studio" / "studio.rc"
        windres = Path(compiler()).with_name("windres.exe")
        subprocess.run([str(windres), str(rc), "-O", "coff", "-o", str(CPP / "studio" / "studio.res.o")], check=True, cwd=CPP / "studio")
    cmd = [compiler(), *FLAGS, "-o", str(exe), *[str(CPP / s) for s in srcs], *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise ConfigurationError(f"building {name} failed:\n{r.stderr[-3000:]}")
    return exe


def exe(name: str) -> Path:
    return build(name)


def _with_file(design: Design | str | Path) -> tuple[Path, Any]:
    if isinstance(design, (str, Path)) and Path(design).exists():
        return Path(design), None
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")  # noqa: SIM115  (closed below; deleted by the caller)
    json.dump(design, tmp)
    tmp.close()
    return Path(tmp.name), tmp.name


def _run(args: list[str], timeout: float = 600) -> dict[str, Any]:
    r = subprocess.run([str(exe("aerokernel")), *args], capture_output=True, text=True, timeout=timeout)
    try:
        out: dict[str, Any] = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"aerokernel {args[0]} failed: {r.stdout[-500:]} {r.stderr[-500:]}") from e
    if out.get("ok") is False and args[0] != "check":
        raise ConfigurationError(f"aerokernel {args[0]}: {out.get('error')}")
    return out


def _call(cmd: str, design: Design | str | Path, *rest: str) -> dict[str, Any]:
    path, tmp = _with_file(design)
    try:
        return _run([cmd, str(path), *rest])
    finally:
        if tmp:
            os.unlink(tmp)


def check(design: Design | str | Path) -> dict[str, Any]:
    """Validate a design: {'ok', 'errors', 'features', 'bbox_lo', 'bbox_hi'}."""
    return _call("check", design)


def info(design: Design | str | Path, h: float = 0.0, nx: int = 120) -> dict[str, Any]:
    """Volume, wetted / frontal / planform / side areas, centroid, inertia (unit density) and A(x)."""
    return _call("info", design, "--h", str(h), "--nx", str(nx))


def mesh(design: Design | str | Path, out: str | Path, h: float = 0.0, feature: str = "") -> dict[str, Any]:
    """Write an STL (or .obj) of the design, or of one feature, and return its properties."""
    extra = ["--feature", feature] if feature else []
    return _call("mesh", design, str(out), "--h", str(h), *extra)


def section(design: Design | str | Path, axis: str = "x", at: float = 0.0, h: float = 0.0) -> NDArray[np.float64]:
    """Section contour on a plane as segments (n, 4): u0, v0, u1, v1."""
    with tempfile.TemporaryDirectory() as d:
        o = Path(d) / "s.json"
        _call("slice", design, str(o), "--axis", axis, "--at", str(at), "--h", str(h))
        return np.array(json.loads(o.read_text())["segments"], float).reshape(-1, 4)


def distance(design: Design | str | Path, pts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Signed distance at points (n, 3): negative inside."""
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "p.bin", Path(d) / "d.bin"
        np.ascontiguousarray(pts, float).tofile(a)
        _call("eval", design, str(a), str(b))
        return np.fromfile(b, float)


def templates() -> dict[str, Any]:
    out: dict[str, Any] = json.loads(subprocess.run([str(exe("aerokernel")), "templates"], capture_output=True, text=True).stdout)
    return out
