"""Whole-design shape optimisation: give it a shape, get back an optimised one and a comparison.

Examples:
    python scripts/run_shapeopt.py --shape naca2412 --profile standard
    python scripts/run_shapeopt.py --shape my_airfoil.dat --plan 16.2,7.5,1,0,1,0
    python scripts/run_shapeopt.py --shape square --stage topology

A shape is ``nacaXXXX``, ``circle``, ``square``, ``ellipse:0.5`` or a coordinate file (Selig or
Lednicer .dat, or x,y .csv). Airfoil-like shapes go through the whole-design search. Shapes that
CST cannot represent, such as a square, go through topology optimisation.

Needs the optional dependency: pip install "aerosynapse-einstein[shapeopt]"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    r = subprocess.call([sys.executable, str(HERE / script), *args], cwd=HERE)
    if r:
        raise SystemExit(f"{script} failed with exit code {r}")


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--shape", default="naca2412")
    a.add_argument("--profile", default="quick", choices=["quick", "standard", "full"])
    a.add_argument("--stage", default="auto", choices=["auto", "search", "topology", "verify"])
    a.add_argument("--plan", default="16.2,7.5,1.0,0.0,1.0,0.0", help="area, AR, taper, tip twist, twist exponent, winglet")
    n = a.parse_args()
    sys.path.insert(0, str(HERE.parent / "src"))
    from aerosynapse.shapeopt import geometry as G

    stage = n.stage
    if stage == "auto":
        rms = G.fit_cst(G.load_shape(n.shape))[1]
        stage = "search" if rms <= 3e-3 else "topology"
        print(f"CST fit error {rms:.5f} chord: {'airfoil-like, whole-design search' if stage == 'search' else 'not airfoil-like, topology optimisation'}")
    ns_level = {"quick": "quick", "standard": "standard", "full": "standard"}[n.profile]
    if stage == "search":
        run("shapeopt_search.py", "--shape", n.shape, "--profile", n.profile, "--plan", n.plan)
        run("shapeopt_verify.py", "--shape", n.shape, "--ns", ns_level, "--profile", n.profile)
    elif stage == "topology":
        run("shapeopt_topology.py", "--shape", n.shape, "--start", "shape", "--profile", n.profile)
    else:
        run("shapeopt_verify.py", "--shape", n.shape, "--ns", ns_level)


if __name__ == "__main__":
    main()
