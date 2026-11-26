"""Phase 6: fluid mechanics and aerospace (see aerosynapse.experiments.studies.phase6)."""

from __future__ import annotations

import json

from _common import RESULTS, parse_profile

from aerosynapse.experiments import ExperimentRecord, ExperimentRun
from aerosynapse.experiments.studies import phase6 as S


def main() -> None:
    a = parse_profile(__doc__ or "")
    q = a.profile == "quick"
    rec = ExperimentRecord(
        experiment_id="phase6_fluids_aero",
        research_question="Does the MAC projection solver reproduce analytic flows and published laminar benchmarks, "
        "and does the panel method support a legitimate aerodynamic optimisation?",
        hypothesis="H6.1 analytic channel flows reproduced with O(h^2) error. H6.2 cavity centre-line RMS vs Ghia < 0.02 "
        "at Re=100 and 400 on 64^2. H6.3 cylinder C_D (DFG 2D-1) within 10% with the staircase boundary. H6.4 "
        "marching BL matches Blasius within 1%. H6.5 the optimised airfoil remains feasible at higher panel resolution.",
        mathematical_formulation="incompressible NS (MAC, Chorin projection); Prandtl BL equations; Hess-Smith panels",
        seed=0,
        solver_configuration={"profile": a.profile},
        evaluation_metrics=["RMS vs benchmark", "C_D error", "observed order", "C_L slope", "trusted feasibility"],
        dataset_description="Ghia et al. 1982 tables and DFG 2D-1 reference values (fetched; src/aerosynapse/data)",
        reproducibility=f"python scripts/run_phase6.py --profile {a.profile}",
        limitations=["laminar only", "staircase immersed boundary", "inviscid airfoil model (no drag)"],
    )
    with ExperimentRun(rec, RESULTS) as run:
        res = {
            "analytic": S.analytic_flows(),
            "cavity": S.cavity([100, 400] if q else [100, 400, 1000], [32, 64] if q else [32, 64, 128]),
            "cylinder": S.cylinder([0.01] if q else [0.01, 0.005]),
            "boundary_layer": S.boundary_layer(),
            "panel": S.panel(),
            "aero_optimisation": S.aero_optimisation(),
            "cp_surrogate": S.cp_surrogate(),
            "misc": S.misc(),
        }
        run.record.results.update(res)

    def strip(d):
        if isinstance(d, dict):
            return {
                k: strip(v)
                for k, v in d.items()
                if k not in ("profile", "field", "history", "trace", "naca2412_cp", "naca2412_cp_alpha4")
            }
        return d

    print(json.dumps(strip(res), indent=1, default=str)[:7000])


if __name__ == "__main__":
    main()
