"""AirfRANS study: derived coefficient table, surrogate on official splits (with convergence check),
panel-vs-RANS model discrepancy. Requires the downloaded dataset and ``pip install airfrans``.

    python scripts/run_airfrans.py --root /path/to/Dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import RESULTS, ROOT

from aerosynapse.experiments import ExperimentRecord, ExperimentRun
from aerosynapse.ml import airfrans as A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="AirfRANS Dataset directory (contains manifest.json)")
    ap.add_argument("--profile", default="quick")
    ap.add_argument(
        "--max-sims",
        type=int,
        default=None,
        help="deterministic subsample per manifest split (fast first run); omit for all simulations",
    )
    a = ap.parse_args()
    table = ROOT / "data_derived" / "airfrans_coefficients.csv"
    table.parent.mkdir(exist_ok=True)
    rec = ExperimentRecord(
        experiment_id="airfrans_coefficients",
        research_question="How well can force coefficients of RANS airfoil flows be learned, and how large is the "
        "inviscid panel method's discrepancy against RANS?",
        hypothesis="H-A1 the MLP surrogate converges (validation plateau) on full/scarce/reynolds/aoa splits. H-A2 "
        "panel C_L exceeds RANS C_L on average (no boundary-layer decambering).",
        mathematical_formulation="(log10 Re, AoA, NACA params) -> (C_D, C_L); C_L panel vs RANS",
        seed=0,
        dataset_description=A.LICENCE_NOTE,
        data_provenance=str(a.root),
        reproducibility="python scripts/run_airfrans.py --root <Dataset>",
        evaluation_metrics=["MAE", "convergence"],
        limitations=["scalar coefficients only (fields not modelled)", "RANS reference has its own model error"],
    )
    with ExperimentRun(rec, RESULTS) as run:
        names = None
        if a.max_sims:
            man_all = json.loads((Path(a.root) / "manifest.json").read_text(encoding="utf-8"))
            names = sorted({n for v in man_all.values() for n in sorted(v)[: a.max_sims]})
            run.record.results["max_sims_per_split"] = a.max_sims
        rows = A.extract_table(a.root, names=names, out_csv=table)
        man = f"{a.root}/manifest.json"
        res = {"n_simulations": len(rows), "table": str(table.relative_to(ROOT)), "splits": {}}
        keys = json.loads(Path(man).read_text(encoding="utf-8"))
        for t in ("full", "scarce", "reynolds", "aoa"):
            tr, te = f"{t}_train", f"{t}_test"
            if tr in keys and (te in keys or "full_test" in keys):
                res["splits"][t] = A.evaluate_on_splits(table, man, tr, te if te in keys else "full_test")
        res["panel_vs_rans"] = A.panel_vs_rans(rows)
        run.record.results.update(res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
