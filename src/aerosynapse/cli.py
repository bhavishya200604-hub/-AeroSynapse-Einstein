"""Command-line interface: ``aerosynapse run | train | optimize | validate | report | reproduce | figures | doctor``.

Every command prints the research-use banner and the validation tier first (spec Section 16).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from aerosynapse.device import hardware_report
from aerosynapse.safety import banner

ROOT = Path(__file__).resolve().parents[2]


def _doctor(_: argparse.Namespace) -> int:
    hw = hardware_report()
    print(json.dumps(hw, indent=2))
    checks = {"numpy": hw.get("numpy"), "scipy": hw.get("scipy"), "torch (optional, [torch])": hw.get("torch")}
    for name, ver in checks.items():
        print(f"[{'ok' if ver else '--'}] {name}: {ver or 'not installed'}")
    try:
        import streamlit  # noqa: F401

        print("[ok] streamlit (optional, [ui])")
    except ImportError:
        print("[--] streamlit (optional, [ui]): not installed")
    return 0


def _script(name: str, extra: list[str]) -> int:
    path = ROOT / "scripts" / name
    if not path.is_file():
        print(f"script {path} not found (run from a source checkout)", file=sys.stderr)
        return 2
    return subprocess.call([sys.executable, str(path), *extra])


def _run(ns: argparse.Namespace) -> int:
    from aerosynapse.experiments.runner import run_config

    out = run_config(ns.config, profile=ns.profile, root=ns.results)
    print(json.dumps(out, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aerosynapse", description="AeroSynapse-Einstein research platform")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check environment and report hardware").set_defaults(fn=_doctor)
    for name, script, hlp in (
        ("train", "run_phase2.py", "train PINN / baseline studies"),
        ("optimize", "run_phase5.py", "run optimisation and inverse-problem studies"),
        ("validate", "run_phase7.py", "build trust reports and fallback checks"),
        ("report", "make_report.py", "generate HTML/Markdown reports from results/"),
        ("reproduce", "reproduce_all.py", "reproduce every reported result"),
        ("figures", "make_figures.py", "generate paper figures from results/"),
    ):
        sp = sub.add_parser(name, help=hlp)
        sp.add_argument("--profile", default="quick", choices=["quick", "standard", "full"])
        sp.set_defaults(fn=lambda ns, s=script: _script(s, ["--profile", ns.profile]))
    rp = sub.add_parser("run", help="run a single experiment from a YAML config")
    rp.add_argument("config")
    rp.add_argument("--profile", default=None, choices=["quick", "standard", "full"])
    rp.add_argument("--results", default="results")
    rp.set_defaults(fn=_run)
    ns = p.parse_args(argv)
    print(banner())
    return int(ns.fn(ns))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
