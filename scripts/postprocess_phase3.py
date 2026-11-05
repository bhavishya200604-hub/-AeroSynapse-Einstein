"""Post-hoc analysis (added after the run, labelled as such): Holm correction across the six ablation tests,
and the across-seed spread at Pe = 1 (failure-mode diagnosis). Appends to the Phase 3 metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aerosynapse.experiments import stats as ST

P = Path(__file__).resolve().parents[1] / "results" / "phase3_adaptive_collocation" / "metrics.json"


def main() -> None:
    m = json.loads(P.read_text())
    ab = m["ablation"]
    m["posthoc"] = {
        "note": "post-hoc (not pre-registered): Holm over the 6 ablation comparisons",
        "ablation_holm": ST.holm({k: v["p_wilcoxon_vs_baseline"] for k, v in ab.items()}),
        "ablation_holm_early_t": ST.holm({k: v["p_early_vs_baseline"] for k, v in ab.items()}),
    }
    rows = m["rows"]
    for pe in (1.0, 1000.0):
        v = [r["rel_l2"] for r in rows if r["pe"] == pe and r["method"] == "uniform" and "ablation" not in r]
        e = [r["rel_l2_early_t"] for r in rows if r["pe"] == pe and r["method"] == "uniform" and "ablation" not in r]
        lt = [r["rel_l2_late_t"] for r in rows if r["pe"] == pe and r["method"] == "uniform" and "ablation" not in r]
        m["posthoc"][f"pe{pe:g}_uniform_spread"] = {
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "early_t_median": float(np.median(e)),
            "late_t_median": float(np.median(lt)),
        }
    P.write_text(json.dumps(m, indent=2))
    print(json.dumps(m["posthoc"], indent=1))


if __name__ == "__main__":
    main()
