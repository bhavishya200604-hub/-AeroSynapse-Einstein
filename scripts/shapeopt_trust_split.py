"""How much of a shape-optimisation gain lies where the surrogate is trusted?

Re-evaluates the baseline and the verified design on the verification hold-out states (NeuralFoil "xxxlarge",
the same 311 named + random states as ``shapeopt_verify.py``) and splits the drag ratios by NeuralFoil's own
``analysis_confidence``: states where both designs are at or above the problem's confidence floor, and the rest.
A gain that only appears in low-confidence states is a gain of the surrogate, not of the airfoil.

    python scripts/shapeopt_trust_split.py --shape naca2412 [--profile standard]

Writes ``results/shapeopt/<shape>/trust_split_<profile>.json``.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from _common import RESULTS  # noqa: E402

from aerosynapse.optim.robust import cvar  # noqa: E402
from aerosynapse.shapeopt import aero2d  # noqa: E402
from aerosynapse.shapeopt import wing as wg  # noqa: E402
from aerosynapse.shapeopt.search import holdout_problem, pick_best  # noqa: E402


def stats(r: np.ndarray) -> dict[str, float]:
    if len(r) == 0:
        return {"n": 0}
    return {"n": int(len(r)), "mean": float(r.mean()), "median": float(np.median(r)), "cvar90": float(cvar(r, 0.9)),
            "worst": float(r.max()), "frac_better": float((r < 1).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="naca2412")
    ap.add_argument("--profile", default="standard")
    n = ap.parse_args()
    out = RESULTS / "shapeopt" / n.shape.replace(":", "_")
    d = json.loads((out / f"search_{n.profile}.json").read_text())
    W0, plan0 = np.array(d["W0"]), wg.Planform(*d["plan0"])
    H = holdout_problem(W0, plan0, 300, 60, model="xxxlarge")
    Wb = H.decode(np.array(pick_best(d["runs"])["z"]))[0][0]
    a = aero2d.solve_at_cl(W0[None], H.cond, "xxxlarge")
    b = aero2d.solve_at_cl(Wb[None], H.cond, "xxxlarge")
    ratio = b["cd"][0] / a["cd"][0]
    ca, cb = a["conf"][0], b["conf"][0]
    floor = H.conf_min
    trusted = (ca >= floor) & (cb >= floor)
    res = {"shape": n.shape, "profile": n.profile, "confidence_floor": floor, "n_states": int(len(ratio)),
           "all": stats(ratio), "trusted": stats(ratio[trusted]), "untrusted": stats(ratio[~trusted]),
           "untrusted_because": {"baseline_only": int(((ca < floor) & (cb >= floor)).sum()), "design_only": int(((ca >= floor) & (cb < floor)).sum()),
                                 "both": int(((ca < floor) & (cb < floor)).sum())},
           "Re_untrusted": np.asarray(H.cond.Re)[~trusted].tolist(), "mach_untrusted": np.asarray(H.cond.mach)[~trusted].tolist()}
    (out / f"trust_split_{n.profile}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("untrusted") or isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
