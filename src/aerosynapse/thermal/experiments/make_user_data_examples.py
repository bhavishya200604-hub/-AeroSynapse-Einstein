"""Write the example user-data files in examples/user_data/ (synthetic, fully reproducible; see the README there).

    python experiments/make_user_data_examples.py

slab_bi10_noisy.csv   B_slab (Bi = 10, default conditions): analytic theta at 200 Sobol points (x, t) plus Gaussian noise
                      sigma = 0.01 (seed 0); columns x, t, T_meas (deliberately not named 'theta': shows `columns:` mapping)
cavity_probes.npz     C_cavity (default parameters and Case C walls): u, v, theta of the steady Newton FV solution on 64^2
                      cells (reference grid 32, refine 2) at a 7 x 7 probe array; no noise
"""

from __future__ import annotations

import csv
import json

import numpy as np

from aerosynapse.thermal.data.design import sobol_unit
from aerosynapse.thermal.physics import analytic as an
from aerosynapse.thermal.pinn import problems2d as P2
from aerosynapse.thermal.pinn import spec as SP
from aerosynapse.thermal.pinn.ref2d import cavity_reference
from aerosynapse.thermal.utils.experiment import ROOT

OUT = ROOT / "examples" / "user_data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = SP.load_spec(None, "smoke")
    pb = spec["problems"]["B_slab"]
    c, T, Bi = pb["conditions"], float(pb["T"]), 10.0
    u = sobol_unit(200, 2, 0)
    x, t = u[:, 0], T * u[:, 1]
    theta = np.array(
        [
            an.slab_transient(np.array([xi]), float(ti), Bi, c["Q"], c["m"], c["theta0"], n_terms=400)[0]
            for xi, ti in zip(x, t, strict=True)
        ]
    )
    noisy = theta + 0.01 * np.random.default_rng(0).standard_normal(theta.size)
    with open(OUT / "slab_bi10_noisy.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "t", "T_meas"])
        w.writerows(zip(np.round(x, 12), np.round(t, 12), np.round(noisy, 12), strict=True))
    pc = spec["problems2d"]["C_cavity"]
    par = {k: float(v) for k, v in pc["parameters"].items()}
    ref = cavity_reference(par, P2.cavity_conditions(pc), 32, 2, 1e-10)
    ii = np.linspace(4, len(ref["x"]) - 5, 7).round().astype(int)
    J, I = np.meshgrid(ii, ii, indexing="ij")  # J: y index, I: x index
    np.savez(
        OUT / "cavity_probes.npz",
        x=ref["x"][I].ravel(),
        y=ref["y"][J].ravel(),
        u=ref["u"][J, I].ravel(),
        v=ref["v"][J, I].ravel(),
        theta=ref["theta"][J, I].ravel(),
    )
    info = {
        "slab_bi10_noisy.csv": {"points": 200, "noise_sigma": 0.01, "Bi": Bi, "T": T, "conditions": c},
        "cavity_probes.npz": {
            "points": 49,
            "parameters": par,
            "grid": ref["n_fine"],
            "coarse_fine_diff": ref["coarse_fine_diff"],
        },
    }
    (OUT / "provenance.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
