"""Write an example 'measured data' file for the Phase 5 inverse problems (default: examples/inverse/slab_measured_example.csv).

The file is SYNTHETIC: the analytic forward model of a configured problem at its `parameters`, plus Gaussian noise
(Phase 2 model at --level, seed --seed, a different realisation from data.seed). It shows the format for your own
data (columns x [m], t [s], T, sigma; '#' comment lines allowed). The true parameters are written in the header
comments only, so an example can be analysed as if they were unknown (data.compare_to_parameters: false).

    python experiments/make_inverse_example.py                       # the slab example (default problem)
    python experiments/make_inverse_example.py --set body.shape=cylinder ... --out examples/inverse/my_example.csv
"""

from __future__ import annotations

import argparse

import numpy as np

from aerosynapse.thermal.inverse import data as D
from aerosynapse.thermal.inverse import spec as IS
from aerosynapse.thermal.utils.experiment import ROOT, git_hash


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--out", default="examples/inverse/slab_measured_example.csv")
    ap.add_argument("--level", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()
    spec = IS.load_spec(a.config or None, IS.parse_set(a.sets) | {"data": {"source": "synthetic"}})
    d = D.synthetic(spec, a.level, a.seed)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    p, b = spec["parameters"], spec["body"]
    head = [
        "# SYNTHETIC example data (not a measurement): experiments/make_inverse_example.py at commit " + git_hash()[:7],
        f"# body {b}; true parameters {p}",
        f"# noise: Phase 2 {spec['measurement']['noise']['model']} model, level {a.level:g}, seed {a.seed}; sigma = per-reading noise standard deviation",
        "# columns: x [m] (0 = centre plane / axis / centre, L = surface), t [s], T [deg C], sigma [K]",
    ]
    rows = np.stack([d["r"], d["t"], d["T"], d["sigma"]], 1)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(head) + "\nx,t,T,sigma\n")
        np.savetxt(fh, rows, delimiter=",", fmt="%.10g")
    print(f"wrote {out} ({len(rows)} readings)")


if __name__ == "__main__":
    main()
