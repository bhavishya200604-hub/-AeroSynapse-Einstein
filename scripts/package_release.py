"""Build the distributable package: dist/AeroSynapse-<version>-win64.zip with a SHA-256 manifest.

    python scripts/package_release.py [--with-project falcon]

Contents: the three programs in bin/ (Studio, geometry kernel, flow solver), the Python package (src/), the C++
sources (cpp/), scripts, tests, docs, example designs (projects/*.json), the launcher (AeroSynapse.bat), README,
CHANGELOG, LICENSE, pyproject.toml and, optionally, one finished workflow project with its report and exports.
Large result folders, the virtual environment and the data store are left out: they are rebuilt or regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".venv", "node_modules"}
TOP = ["bin", "src", "cpp", "scripts", "tests", "docs", "configs", "media", "assets"]
FILES = ["AeroSynapse.bat", "README.md", "CHANGELOG.md", "LICENSE", "pyproject.toml", "constraints.txt", "CITATION.cff", "SECURITY.md", "CONTRIBUTING.md", "mkdocs.yml"]
SKIP_SUFFIX = {".pyc", ".o.tmp", ".mp4"}


def version() -> str:
    for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version"):
            return line.split("=")[1].strip().strip('"')
    return "0"


def files(project: str | None) -> list[Path]:
    out = [ROOT / f for f in FILES if (ROOT / f).exists()]
    for top in TOP:
        for f in (ROOT / top).rglob("*"):
            if f.is_file() and not (set(f.relative_to(ROOT).parts) & SKIP_DIRS) and f.suffix not in SKIP_SUFFIX:
                if top == "bin" and f.suffix.lower() not in (".exe", ".dll"):
                    continue
                out.append(f)
    out += sorted((ROOT / "projects").glob("*.json"))
    if project:
        pd = ROOT / "projects" / project
        out += [f for f in pd.rglob("*") if f.is_file() and not f.name.endswith(".bin") and not f.name.endswith(".zip")]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-project", default=None, help="include this finished workflow project (projects/NAME)")
    a = ap.parse_args()
    name = f"AeroSynapse-{version()}-win64"
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    zp = dist / f"{name}.zip"
    manifest = {"package": name, "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": {}}
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files(a.with_project):
            rel = f.relative_to(ROOT).as_posix()
            data = f.read_bytes()
            manifest["files"][rel] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            z.writestr(f"{name}/{rel}", data)
        z.writestr(f"{name}/MANIFEST.json", json.dumps(manifest, indent=1))
        z.writestr(f"{name}/INSTALL.txt", INSTALL)
    h = hashlib.sha256(zp.read_bytes()).hexdigest()
    (dist / f"{name}.zip.sha256").write_text(f"{h}  {zp.name}\n", encoding="ascii")
    print(f"{zp} ({zp.stat().st_size / 1e6:.1f} MB, {len(manifest['files'])} files)\nSHA-256 {h}")


INSTALL = """AeroSynapse: install and first run
=================================

1. Python 3.12 with the project's dependencies:
       python -m venv .venv
       .venv\\Scripts\\python -m pip install -e .[designer,shapeopt] -c constraints.txt
   (XFOIL is optional; put xfoil.exe in .venv\\tools\\xfoil\\ for the airfoil checks.)
2. Double-click AeroSynapse.bat. Studio opens; the programs in bin\\ are ready to run.
   If bin\\ is empty, the launcher builds them with g++ (MinGW, C++14).
3. In Studio: Workflow > Design workflow (F9) > New project > Run all remaining.
   The stages run in order (basic design, design, wing and body optimisation, structure, PINN skin,
   3D flow, export); each finished stage's design is loaded into the editor, and Export writes the
   STEP, glTF, STL, DXF and PDF drawing, the bill of materials and the report.
4. The same from a terminal:
       .venv\\Scripts\\python -m aerosynapse.cli workflow new myjet
       .venv\\Scripts\\python -m aerosynapse.cli workflow run myjet --profile quick

Research use only: preliminary-design methods with stated assumptions, not certification analysis.
"""

if __name__ == "__main__":
    main()
