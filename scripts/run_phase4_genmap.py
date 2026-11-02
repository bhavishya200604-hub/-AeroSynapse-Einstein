"""Phase 4 generalisation map (Fig. G): the trained FNO from phase4_operators_uq evaluated over (Pe, input bandwidth)."""

from __future__ import annotations

import json

import torch
from _common import RESULTS, parse_profile

from aerosynapse.experiments import ExperimentRecord, ExperimentRun
from aerosynapse.experiments.record import file_sha256
from aerosynapse.experiments.studies import phase4 as S
from aerosynapse.ml.fno import FNO1d
from aerosynapse.ml.io import load_model


def main() -> None:
    a = parse_profile(__doc__ or "")
    ckpt = RESULTS / "phase4_operators_uq" / "fno.pt"
    torch.set_num_threads(4)
    rec = ExperimentRecord(
        experiment_id="phase4_generalisation_map",
        research_question="Where in (Pe, input bandwidth) does the FNO trained on Pe in [5, 50], k_max = 4 stay accurate?",
        hypothesis="H4.8 error grows by more than 10x outside the training Pe range at fixed bandwidth, and grows with "
        "input bandwidth beyond k_max = 4.",
        mathematical_formulation="median per-sample rel. L2 of the FNO on 12 fresh FV samples per (Pe, k_max) cell",
        seed=500,
        data_provenance=f"model {ckpt.name} sha256={file_sha256(ckpt)}",
        reproducibility=f"python scripts/run_phase4_genmap.py --profile {a.profile} (after run_phase4.py)",
        evaluation_metrics=["median rel. L2 per cell"],
        limitations=["12 samples per cell"],
    )
    with ExperimentRun(rec, RESULTS) as run:
        m = FNO1d(2, 1, 32, 12, 4)
        load_model(m, ckpt)
        run.record.results["generalisation_map"] = S.generalisation_map(m)
    print(json.dumps(run.record.results["generalisation_map"])[:800])


if __name__ == "__main__":
    main()
