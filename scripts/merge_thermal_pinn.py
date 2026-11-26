"""Bring the parametric PINN heat-transfer project into AeroSynapse as ``aerosynapse.thermal``.

Every module is copied, not wrapped, so all of its features (verified solvers, parametric PINNs,
adaptive balancing, inverse identification, real-data study, UQ, validity checks, figures, THERMAL LAB
web app) live in one package. Imports ``src.x`` become ``aerosynapse.thermal.x``, ``experiments.x``
becomes ``aerosynapse.thermal.experiments.x``, and every ``ROOT = Path(__file__).parents[n]`` points
at one home folder, ``projects/thermal_pinn``, which holds that project's configs, data, results and docs.

The original repository is only read. Re-running the script refreshes the copy.

    python scripts/merge_thermal_pinn.py --source E:/PINNs_Heat_Transfer
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PKG = HERE / "src" / "aerosynapse" / "thermal"
HOME = HERE / "projects" / "thermal_pinn"
SUBS = "pinn|viz|utils|numerics|inverse|data|physics|web|app|analysis"
ROOT_RE = re.compile(r"^(\s*)(ROOT|ROOT_DIR)\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[\d\]\s*$", re.M)
SKIP = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}


def rewrite(text: str) -> str:
    text = re.sub(rf"\bsrc\.({SUBS})\b", r"aerosynapse.thermal.\1", text)
    text = re.sub(r"(?<![\w.])experiments\.(?=[\w{])", "aerosynapse.thermal.experiments.", text)
    text = re.sub(r"\bpaper\.tools\b", "aerosynapse.thermal.paper_tools", text)
    text = re.sub(r"\bpaper\.common\b", "aerosynapse.thermal.paper_common", text)
    text = text.replace('"tests.unit.isolation_probe"', '"aerosynapse.thermal._isolation_probe"')
    text = re.sub(r"\bfrom experiments import\b", "from aerosynapse.thermal.experiments import", text)
    text = re.sub(r"\bimport experiments\b(?!\.)", "import aerosynapse.thermal.experiments as experiments", text)
    # paths into the old source tree now point at the package (CODE) or at tests/thermal (TESTS)
    for a, b in PATHS:
        text = text.replace(a, b)
    if "_compat." in text and "import _compat" not in text:
        text = text.replace("\nimport pickle\n", "\nimport pickle\n\nfrom aerosynapse.thermal import _compat\n", 1)
    if "def _read(path: str | Path) -> dict:" in text:  # config paths were relative to the old repository root
        text = text.replace("def _read(path: str | Path) -> dict:\n", "def _read(path: str | Path) -> dict:\n    from aerosynapse.thermal._home import resolve\n\n    path = resolve(path)\n", 1)
    return ROOT_RE.sub(r"\1from aerosynapse.thermal._home import CODE, THERMAL_TESTS, HOME as \2  # noqa: E402, F401", text)


_H = '__import__("aerosynapse.thermal._home", fromlist=["HOME"])'  # explicit, so no file depends on an import it lacks
PATHS = [
    ('ROOT / "src" / "app" / "main.py"', f'{_H}.CODE / "app" / "main.py"'),
    ('ROOT / "src/app/main.py"', f'{_H}.CODE / "app/main.py"'),
    ('ROOT / "src" / "viz"', f'{_H}.CODE / "viz"'),
    ('ROOT / "experiments"', f'{_H}.CODE / "experiments"'),
    ('ROOT / "tests" / "unit"', f"{_H}.THERMAL_TESTS"),
    ('(ROOT / "tests").rglob', f"{_H}.THERMAL_TESTS.rglob"),
    ("return pickle.loads(blob)", "return _compat.loads(blob)"),
    ('importlib.import_module(entry["module"])', 'importlib.import_module(_compat.modname(entry["module"]))'),
]


def rewrite_name(n: str) -> str:
    if n.startswith("src."):
        return "aerosynapse.thermal." + n[4:]
    if n.startswith("experiments."):
        return "aerosynapse.thermal.experiments." + n[len("experiments.") :]
    return n


def copy_tree(src: Path, dst: Path, code: bool, exclude: set[str] = frozenset()) -> int:  # type: ignore[assignment]
    n = 0
    for f in src.rglob("*"):
        rel = f.relative_to(src)
        if any(p in SKIP or p in exclude for p in rel.parts) or f.is_dir():
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if code and f.suffix == ".py":
            out.write_text(rewrite(f.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copy2(f, out)
        n += 1
    return n


HOME_PY = '''"""Home folder of the thermal PINN project: its configs, data, results, docs and caches.

Set AEROSYNAPSE_THERMAL_HOME to move it, for example to a larger drive.
"""

from __future__ import annotations

import os
from pathlib import Path

HOME = Path(os.environ.get("AEROSYNAPSE_THERMAL_HOME") or Path(__file__).resolve().parents[3] / "projects" / "thermal_pinn")
CODE = Path(__file__).resolve().parent  # the package itself (src/ in the original repository)
THERMAL_TESTS = Path(__file__).resolve().parents[3] / "tests" / "thermal"  # tests/unit in the original repository


def resolve(path: str | Path) -> Path:
    """A path the original project wrote relative to its root: try it as given, then under HOME."""
    p = Path(path)
    return p if p.exists() or p.is_absolute() else HOME / p
'''

COMPAT_PY = '''"""Read files written by the original repository, whose modules were called ``src.*`` and ``experiments.*``."""

from __future__ import annotations

import io
import pickle
from typing import Any


def modname(name: str) -> str:
    if name.startswith("src."):
        return "aerosynapse.thermal." + name[4:]
    if name.startswith("experiments."):
        return "aerosynapse.thermal.experiments." + name[len("experiments.") :]
    if name.startswith("paper.tools"):
        return "aerosynapse.thermal.paper_tools" + name[len("paper.tools") :]
    return name


class Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(modname(module), name)


def loads(blob: bytes) -> Any:
    return Unpickler(io.BytesIO(blob)).load()
'''

INIT_PY = '''"""Adaptive parametric PINNs for convection-diffusion heat transfer, merged from the author's
``adaptive-pinn-heat-transfer`` project.

Subpackages: ``numerics`` (verified FD / FVM / spectral / Navier-Stokes solvers), ``physics`` (analytic,
Graetz, manufactured solutions), ``pinn`` (architectures, adaptive balancing, strategies, parametric 2D,
diagnostics, UQ, validity), ``inverse`` (identification of alpha, k, h, real data), ``data``, ``viz``,
``web`` (THERMAL LAB) and ``experiments`` (every study script). ``bridge`` connects them to the aircraft tools.
"""
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="E:/PINNs_Heat_Transfer")
    ap.add_argument("--with-results", action="store_true", help="also copy saved results (large)")
    a = ap.parse_args()
    src = Path(a.source)
    if not (src / "src" / "pinn").is_dir():
        raise SystemExit(f"{src} does not look like the PINN project")
    bridge = PKG / "bridge.py"
    keep = bridge.read_text(encoding="utf-8") if bridge.exists() else None
    if PKG.exists():
        shutil.rmtree(PKG)
    n = copy_tree(src / "src", PKG, code=True)
    n += copy_tree(src / "experiments", PKG / "experiments", code=True)
    (PKG / "experiments" / "__init__.py").touch()
    n += copy_tree(src / "paper" / "tools", PKG / "paper_tools", code=True)
    n += copy_tree(src / "paper" / "common", PKG / "paper_common", code=True)
    if (src / "tests" / "unit" / "isolation_probe.py").exists():
        (PKG / "_isolation_probe.py").write_text(rewrite((src / "tests" / "unit" / "isolation_probe.py").read_text(encoding="utf-8")), encoding="utf-8")
    # small saved results and the figure catalogue (metrics, tables, configs): tests and the app read them
    small = (".json", ".csv", ".yaml", ".md", ".txt")
    for f in (src / "results").rglob("*") if (src / "results").is_dir() and not a.with_results else []:
        if f.is_file() and f.suffix in small and f.stat().st_size < 2_000_000:
            out = HOME / "results" / f.relative_to(src / "results")
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
            n += 1
    for f in (src / "docs" / "figures").rglob("*.json") if (src / "docs" / "figures").is_dir() else []:
        out = HOME / "docs" / "figures" / f.relative_to(src / "docs" / "figures")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        n += 1
    n += copy_tree(src / "paper", HOME / "paper", code=False, exclude={"dist", "tools"})
    (PKG / "__init__.py").write_text(INIT_PY, encoding="utf-8")
    (PKG / "_home.py").write_text(HOME_PY, encoding="utf-8")
    (PKG / "_compat.py").write_text(COMPAT_PY, encoding="utf-8")
    if keep is not None:
        bridge.write_text(keep, encoding="utf-8")
    tests = HERE / "tests" / "thermal"
    if tests.exists():
        shutil.rmtree(tests)
    n += copy_tree(src / "tests" / "unit", tests, code=True)
    (tests / "__init__.py").unlink(missing_ok=True)
    for f in tests.glob("test_*.py"):  # pytest needs unique basenames across tests/ and tests/thermal/
        if (HERE / "tests" / f.name).exists():
            f.rename(f.with_name("test_thermal_" + f.name[len("test_"):]))
    p = tests / "test_viz_anim.py"
    if p.exists():  # the one test that finds the project root by itself
        p.write_text(p.read_text(encoding="utf-8").replace('(Path(__file__).resolve().parents[2] / "docs/tables/phase5_gate.md")', '(__import__("aerosynapse.thermal._home", fromlist=["HOME"]).HOME / "docs/tables/phase5_gate.md")'), encoding="utf-8")
    for d in ("configs", "examples", "deploy", "data"):
        if (src / d).is_dir():
            n += copy_tree(src / d, HOME / d, code=False)
    n += copy_tree(src / "tests" / "fixtures" / "figures", HOME / "tests" / "fixtures" / "figures", code=False)
    n += copy_tree(src / "docs", HOME / "docs", code=False, exclude={"figures", "animations", "media", "linkedin"})
    man = HOME / "tests" / "fixtures" / "figures" / "MANIFEST.json"
    if man.exists():  # figure fixtures are keyed by module names, which changed
        m = json.loads(man.read_text(encoding="utf-8"))
        out = {}
        for k, e in m.items():
            e["module"] = rewrite_name(e["module"])
            head, _, tail = k.partition("@")
            label, _, rest = tail.partition(".")
            out[f"{head}@{label}.{rewrite_name(rest)}" if tail else k] = e
        man.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for f in ("README.md", "CLAUDE.md", "LICENSE", "CITATION.cff", "tasks.py", "Makefile"):
        if (src / f).exists():
            HOME.mkdir(parents=True, exist_ok=True)
            out = HOME / f
            out.write_text(rewrite(f_text) if (f_text := (src / f).read_text(encoding="utf-8", errors="ignore")) and f.endswith(".py") else f_text, encoding="utf-8")
            n += 1
    if a.with_results and (src / "results").is_dir():
        n += copy_tree(src / "results", HOME / "results", code=False)
    left = [str(p) for p in PKG.rglob("*.py") if re.search(r"\bfrom src\.|\bimport src\.|parents\[\d\]\s*$", p.read_text(encoding="utf-8"), re.M)]
    print(f"copied {n} files into {PKG} and {HOME}")
    print("unconverted:", left or "none")


if __name__ == "__main__":
    main()
