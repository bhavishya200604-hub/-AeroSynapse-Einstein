# Phase 7 summary — validation engine, trust and fallback, resources, safety
<!-- sources: phase7_validation_engine -->

Source: `results/phase7_validation_engine/` (`python scripts/run_phase7.py --profile quick`, 100 s).

**Built:** `validation/trust.py` (TrustPolicy: hard parameter-range check, Mahalanobis input novelty, ensemble
spread, physics residual; JSON-lines decision log; Markdown trust reports), `validation/resources.py` (resource
estimator calibrated by micro-benchmarks), notebook generator (execution mode C), external-solver adapter (mode
D, Phase 6), checkpoint/resume, dataset catalogue (`data/datasets.json`), and the five-tier safety ladder.

## Trust and fallback (FNO ensemble, 3 members, thresholds at the 99th percentile of a separate calibration split)
Thresholds: Mahalanobis 2.8898, relative spread 0.005128, mass residual 0.003140.

| Query set (100 each) | Fallback rate | Raw surrogate error median / max | Guarded system median / max |
|---|---|---|---|
| In-distribution | 0.09 | 0.00278 / 0.02695 | 0.00245 / 0.01259 |
| OOD Pe ∈ [200, 1000] | 1.0 | 0.15265 / 0.23389 | 0.00064 / 0.00101 |
| OOD high-bandwidth inputs | 1.0 | 0.0066 / 0.05578 | 0.00012 / 0.00046 |

Fallback reasons are logged per query (`fallback_decisions.jsonl`, 300 decisions). OOD-Pe queries trip all four
checks. OOD-input queries trip input novelty in every case, the spread check in 36 and the residual check in 13.
A fallback costs one classical solve (mean 0.0366 s).

**Hypotheses.** H7.1 *partly*: the OOD fallback rate is 100 %, but the in-distribution false-fallback rate is 9 %,
above the 5 % target. Three separate 99th-percentile thresholds combine to about 3 % by design; the observed 9 %
shows the 100-sample calibration split is too small. A larger calibration split or a joint threshold is needed.
**H7.2 supported:** the guarded worst-case error is far below the raw surrogate's on both OOD sets, and also lower
in-distribution (0.01259 vs 0.02695), because the worst ID queries are the ones rejected.

## Resource estimator (calibrated on this machine)
| Workload | Predicted s | Measured s | Relative error |
|---|---|---|---|
| PINN, 500 iterations × 1000 points × 2273 parameters | 3.888 | 3.664 | +0.0613 |
| FV van Leer, 512 cells × 5243 steps | 0.8057 | 0.5249 | +0.5348 |

**H7.3 supported (within a factor of 2).** The first version, a single-point proportional PINN model, over-predicted
several-fold (superseded run). It was replaced by an affine model (per-iteration overhead plus slope) fitted from two
calibration runs. Both versions are logged in ADR-0016.

## Other gate items
* **Checkpoint/resume:** an interrupted and resumed PINN run is bit-identical to an uninterrupted run
  (`tests/test_validation_engine.py::test_checkpoint_resume_is_bit_identical`).
* **Execution modes:** A (CPU) and B (automatic CUDA selection, not exercised: no GPU) via `device.py`. C:
  notebooks in `notebooks/`, generated, not executed. D: the OpenFOAM adapter, whose pre-flight reports
  OpenFOAM missing.
* **Trust reports** (Markdown, with banner and tier): `trust_report_fno_operator.md`, `trust_report_pinn_convdiff.md`,
  `trust_report_optimisation_surrogates.md` in the results folder.
* **Datasets:** PDEBench (data CC BY 4.0), AirfRANS (ODbL-1.0) and The Well (licence not verified) are catalogued
  with source, licence, system, format, units, split, leakage risk, limitations and access. None is used in a
  reported result.
* **Safety:** the platform tier is 1/5 (research simulation); classical solvers are tier 3 (code-verified); the
  ML surrogates are tier 1. The banner appears in the CLI, the UI and every report.
