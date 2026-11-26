# Research roadmap

Ranked by research value per hour of this machine (2 cores, 6 GB RAM, no GPU; see
`HARDWARE_AND_STORAGE_REPORT.md`). Each item has a measurable stopping criterion. Status as of 2026-11-11.

| ID | Item | Why | Stopping criterion | Cost here | Status |
|---|---|---|---|---|---|
| R-1 | Experiment integrity: registry, no overwrites, feasibility reporting | a reported result was the wrong experiment (KI-67) and infeasible rounds could win (KI-69) | every optimisation writes a per-run record, feasibility and violated constraints; registry up to date | hours | **done** for shape optimisation; CFD runs still write into a named folder (R-9) |
| R-2 | 3D CFD validation on a public wind-tunnel case | `aeroflow` has verification (Sod, Taylor-Green) but no 3D validation; every CFD optimisation depends on it | C_p at the measured span stations within stated tolerances on a grid shown to be converged; result published with its error | ~1 day of compute per grid level | next; dataset to be acquired (below) |
| R-3 | Mesh independence of the forces used in optimisation, and the half/full discrepancy (KI-68) | C_L doubles from draft to standard on the Falcon; half and full models disagree by 1.6x / 2.8x | three grids in the asymptotic range (R = e21/e32 < 1) with a GCI for C_L and C_D; half and full models agree within the GCI | 8 grids done 2026-11-11 (4.5 h): half/full explained (agree within 4 % in C_L from 34 cells per length); **not converged** up to 68 (C_L -13 % from 48 to 68, KI-77). 96 cells per length (2 h) still oscillated; the cause was the thin-surface treatment (KI-78 to KI-80, fixed 2026-11-11): with the fixed solver C_L is within 1.3 % on 34-68 (0.140-0.144). 96 cells per length with the fixed solver (70 min): **drag converges** (48-96 monotonic, p 2.7, GCI 4.4 %) but **lift falls 18 %** (0.1167): thin-cell steps in the wing sheet (KI-84); two fixes tried and rejected. Next: a sub-cell sheet position (cut faces) for thin surfaces, then repeat 68/96 | in progress (drag converged; lift open, KI-84) |
| R-4 | Independent judge where the surrogate's confidence drops | the shape optimum leaves NeuralFoil's trusted region at 2 of 1,584 points (KI-73) | the flagged states evaluated with XFOIL or the panel + boundary-layer solver; constraint redefined from a single minimum to a stated quantile if the independent judge agrees | hours | open |
| R-5 | Multi-objective wing study: drag against wing mass and flutter margin | the single score hides a +47 % mass trade, with aspect ratio and taper on their bounds | Pareto front (NSGA-II or a scalarised sweep) over the same model, with the bound-limited variables widened or justified | hours to a day | open |
| R-6 | Controlled CMA-ES restart comparison (current vs IPOP) | the restart scheme does not match its description (KI-72) | same budget, 10+ seeds, benchmark functions (sphere, Rosenbrock, Rastrigin in 24-D) and the real problem; report median and spread | hours | open |
| R-7 | CFD-in-the-loop optimisation, redone | the first study is unreliable (KI-68) | after R-3: full model, grid with more than 120 wall cells, its GCI reported; best wing confirmed on the next finer grid | 2-3 h per 14-run study | waits for KI-84: lift is not grid-converged (-18 % from 68 to 96); drag has a converged estimate only from 96 cells per length (70 min per run) |
| R-8 | Full test-suite baseline | the last full run was cut off | complete run with totals recorded; failures triaged | 1-2 h | open |
| R-9 | Runs on the storage root with free-space checks | `C:\AeroSynapse_Data` exists but is unused | configurable root; a run refuses to start below 10 GB free | hours | open |
| R-10 | ML surrogate of the 3D CFD (operator learning on the saved volumes) | would make CFD-informed optimisation cheaper | only after R-2/R-3 provide validated, converged training data; held-out designs, calibrated uncertainty | days of CPU training | later |

## Data acquisition (phases 4-6 of the brief): not started

No dataset has been downloaded in this audit. Candidate public datasets for the items above, to be checked for
licence, size and access before anything is claimed:

| Purpose | Candidate | Status |
|---|---|---|
| R-2 3D validation | ONERA M6 wing, transonic pressure data (Schmitt and Charpin, AGARD AR-138, 1979), widely redistributed by NASA validation archives | source URL, licence and file format to be verified |
| R-2 3D validation (larger) | NASA Common Research Model (Drag Prediction Workshop) geometry and wind-tunnel data | to be checked; geometry is large for this machine |
| R-4 2D airfoil truth | NACA 0012 measurements of Ladson (NASA TM-4074) as used by NASA's turbulence-modelling resource | to be checked |
| R-10 ML | AirfRANS (2D RANS airfoil fields, NeurIPS 2022 datasets track); a loader already exists (`ml/airfrans.py`) | not on disk; size and licence to be checked |

When a dataset cannot be downloaded automatically, it will be listed in `MANUAL_DATA_REQUIRED.md` with its exact
official page, files, size, licence and target folder under `C:\AeroSynapse_Data\raw_data\`; only the dependent
item stops.

## Deliverables of the brief and their state

| Document | State |
|---|---|
| PROJECT_AUDIT.md, BASELINE_RESULTS.md, HARDWARE_AND_STORAGE_REPORT.md, RESEARCH_ROADMAP.md, EXPERIMENT_REGISTRY.csv | written (this folder) |
| KNOWN_ISSUES.md | the project's register `docs/KNOWN_ISSUES.md`, KI-67 to KI-73 added |
| DATASET_CATALOG.md, MANUAL_DATA_REQUIRED.md | with the first data acquisition (R-2) |
| MATHEMATICAL_FORMULATION.md, OPTIMIZATION_METHODOLOGY.md | to consolidate from `docs/theory/` and `docs/SHAPE_OPTIMISATION.md` when R-5/R-6 change the methods |
| MODEL_ARCHITECTURE.md, VALIDATION_REPORT.md, LIMITATIONS.md, FINAL_RESEARCH_REPORT.md | existing material in `docs/VV_REPORT.md`, `docs/ARCHITECTURE.md`; to be updated after R-2/R-3 |
