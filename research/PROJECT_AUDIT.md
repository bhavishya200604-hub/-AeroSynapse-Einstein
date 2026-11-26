# Project audit

Audit of 2026-11-11, before any major change. Every statement is labelled:
**[verified]** checked in code or by running it; **[observed]** measured in an experiment of record;
**[hypothesis]** a likely explanation not yet tested; **[missing]** information that does not exist yet.

## 1. What the repository is

| Part | Language | Size | Role |
|---|---|---|---|
| `src/aerosynapse/` | Python 3.12 | 342 files | analysis, optimisation, ML, workflow, reports, Designer web app |
| `cpp/` | C++14 (MinGW 32-bit) | 23 files | geometry kernel (SDF, meshing, exact tessellation), 3D flow solver `aeroflow`, Studio desktop app |
| `tests/` | pytest | 92 files | unit, regression and verification tests |
| `scripts/` | Python, PowerShell | 59 files | experiment drivers, benchmarks, report builders, demo recorder |
| `PINNs_Heat_Transfer/` | Python | separate project with its own `.venv` (26,000 files) | the earlier thermal-PINN project, imported into `src/aerosynapse/thermal` |
| `results/` | data | many GB of runs | outputs of experiments (no registry until this audit) |
| `docs/` | Markdown | 20 top-level documents | guides, V&V report, known issues (KI-1 to KI-66 before this audit) |

Python packages by size (lines): shapeopt 2,732; sim 2,419; aircraft 1,922; numerics 1,891; workflow 1,847;
ml 1,791; optim 1,419; fighter 1,235; ui 1,159; structures 775; physics 624. **[verified]**

## 2. Simulation engines

| Engine | Method | Verification evidence | Validation evidence |
|---|---|---|---|
| `aeroflow` (C++) | compressible finite volume, Cartesian cut cells with ghost-cell walls, SSP-RK3, local time steps, WALE LES, wall model | Sod tube, Taylor-Green vortex (`results/cfd_verification`) | **[missing]** none in 3D: no comparison with a wind-tunnel case |
| 2D Navier-Stokes (`numerics`) | MAC projection | Couette, Poiseuille (order 2.004), DFG cylinder C_D 5.464 vs 5.580 | Ghia cavity |
| Panel method, lifting line, VLM | potential flow | cylinder C_p exact | NACA data via XFOIL/NeuralFoil only |
| NeuralFoil (external) | neural surrogate of XFOIL | n/a | trained on XFOIL; 4 % drag error quoted by its authors |
| Structures (FEM, flutter) | beam FEM, Theodorsen k-method | analytic beams | **[missing]** none experimental |
| PINN / FNO / DeepONet | ML surrogates | autograd, MMS residuals | against exact solutions; OOD failure documented |

The V&V report's own conclusion stands: the platform has **no experimental data**, so the highest validation
tier reached is 3 (verified software) for the classical 2D solvers; the 3D CFD solver has verification cases but
no validation. **[verified]** (`docs/VV_REPORT.md`)

## 3. Optimisation algorithms

| Algorithm | Where | Used for |
|---|---|---|
| CMA-ES (own implementation, Hansen 2016 tutorial) | `shapeopt/cmaes.py` | whole-design airfoil + wing search (24 variables) |
| Differential evolution (SciPy) | `shapeopt/search.py` | control on the same budget |
| Gaussian-process Bayesian optimisation (Expected Improvement) | `sim/optimise.py` (added 2026-11-10) | main-wing design with 3D CFD in the loop |
| NSGA-style, SLSQP, robust (CVaR) | `optim/`, `aircraft/optimise.py`, `fighter/optimise.py` | sizing and fighter studies |

## 4. The CMA-ES experiment examined (section requested in the brief)

Problem `shapeopt.objective.Problem`, NACA 2412 baseline, 16.2 m2 wing of aspect ratio 7.5. **[verified]**

| Item | Value |
|---|---|
| Design variables | z in [0, 1]^24: 18 CST weights (baseline +/- 0.14, the last one, a leading-edge term, 0 to 0.01) and 6 planform numbers (area 11-26 m2, AR 5-16, taper 0.3-1.1, tip twist -8 to 2 deg, twist exponent 0.5-3, winglet 0-0.15) |
| Objective | J = 0.5 J2D + 0.5 J3D + 2 sum(max(g,0)) + 40 sum(max(g,0)^2); each J-bar = (mean + CVaR0.9)/2 of drag relative to the baseline in the same state |
| Constraints | 13: thickness, area, nose radius, rear depth, waviness, surface crossing, stall speed, stall location, flutter, divergence, wing solver, surrogate confidence, pitching moment - all soft (penalty) |
| Normalisation | per-state drag ratio, so the baseline scores exactly J = 1 when feasible; all constraints scaled to O(1) |
| Population, budget | standard profile: lambda 16, 2,400 evaluations per round, up to 6 rounds, 3 CMA-ES seeds + 1 DE control; about 12,000 evaluations per seed |
| Seeds | CMA-ES 0, 1, 2; DE 7; flow states re-drawn every round (seed 1000(s+1)+round); hold-out seed 20240 |
| Hold-out | 150 2D + 40 3D states on NeuralFoil "xlarge" (validation, used to pick rounds); verification uses 300 + 60 states on "xxxlarge" (test) |

Findings:

1. **The objective is mathematically consistent** **[verified]**: ratios to the baseline in the same state make
   regimes comparable, CVaR protects the tail, and the baseline maps to J = 1 exactly (checked: all baseline
   constraints are <= 0 and every ratio is 1).
2. **The record of the standard experiment had been overwritten** **[verified]**. A 2-minute quick-profile run
   started from the Designer's "Airfoil" tile (`ui/fighter_pages.py`, `--profile quick`) on 2026-11-09 wrote
   `search.json`, `verify.json` and the exported design over the 59-minute standard search of 2026-11-07.
   The per-seed results survived in `jobs/standard/` and the record was rebuilt from them (KI-67).
3. **The quick run found no improvement** **[observed]**: its best feasible design scored J = 1.028 on the search
   hold-out, worse than the baseline's 1.000; on the test hold-out it violated the surrogate-confidence
   constraint (J = 32.2 with penalties, unpenalised 0.961).
4. **Infeasible designs could be reported as best** **[verified]**: rounds were ranked by penalised J only, so for
   seeds 0 and 2 of the standard search the recorded best round was infeasible on the hold-out. Fixed:
   feasibility first, then score (KI-69).
5. **Surrogate exploitation tested, not found** **[observed]** (KI-73): the standard design is infeasible on the
   test hold-out because its NeuralFoil confidence falls below 0.70 at 2 of 1,584 evaluations (baseline minimum
   0.773). Restricted to the 310 trusted 2D states its mean drag ratio is still 0.934, with 93 % of states
   improved, so the gain does not come from low-confidence states. It is still judged by the same surrogate
   family; XFOIL (254 of 311 states, mean ratio 0.930; KI-16) agrees but is not independent, since NeuralFoil was
   trained on XFOIL.
5b. **The wing optimum is bound-limited** **[observed]**: aspect ratio 15.99 (bound 16) and taper 0.300 (bound
   0.30). The 3D drag gain of 16 % costs 47 % more wing mass; the result is set by the bounds and the mass model.
6. **CMA-ES restarts do not do what the docs say** **[verified]**: restarts begin at the best point with a 0.8x
   smaller step and a 2x larger population; the documentation calls this "increasing-population restarts" that
   search "more globally" (IPOP-CMA-ES restarts from a random point with the original step). It is a local
   refinement, not a global restart (KI-72). Not changed yet: changing it changes results, so it needs a
   controlled comparison first (roadmap R-6).
7. **Sampling** **[observed]**: standard 12,000 evaluations in 24 dimensions; three CMA-ES seeds agree to 0.0006 in
   hold-out J (0.9250 to 0.9256) and DE reaches 0.9290 on the same budget, so the optimum found is reproducible
   across seeds for this model. That says nothing about the model's error.

## 5. The 3D CFD and its optimisation (added this week)

1. **Half and full models disagree** **[observed]** (KI-68). Same flight (Mach 0.85, 2 deg, 11 km), same grid
   spacing (30 cells per length), same stopping rule: full aircraft C_L 0.0435, C_D 0.0107 (132,600 cells, 124 +
   130 wall cells); half model with symmetry plane C_L 0.068, C_D 0.0296 (67,600 cells, 48 + 60 wall cells).
   At zero sideslip these must agree. The half model has no wall cells within 0.15 m of the centre plane where
   the full model has 79, so the fuselage spine and fin on the plane are resolved differently **[hypothesis]**.
   **Update 2026-11-11 (mesh study R-3):** explained by grid resolution **[measured]**. From 34 cells per length on,
   half and full agree within 4 % in C_L (1.012 at 68); C_D within 7 % at 68. But the forces are not grid-converged
   up to 68 cells per length: C_L still drops 13 % from 48 to 68 (KI-77, `BASELINE_RESULTS.md` section 2.1).
   **Update 2026-11-11 (later):** the lift scatter came from the thin-surface treatment (KI-78 to KI-80). With it fixed
   the half-model C_L is 0.140-0.144 on 34-68 cells per length (vortex lattice 0.157); C_D still changes by 11 %
   (`BASELINE_RESULTS.md` section 2.2). **Update (96 cells per length):** C_D converges (GCI 4.4 %) but C_L falls
   18 % to 0.117: the thin-surface representation still depends on the grid (KI-84).
2. **The CFD wing optimisation of 2026-11-10 is not reliable** **[verified]**: it used half models below the
   120-wall-cell threshold. Its "+22.6 % L/D" is flagged in its own summary. The optimiser now uses the full
   model and warns on coarse grids.
3. **Draft steady runs were stopped before their forces settled** **[observed]** (KI-71): at 900 steps C_D read
   0.030 against a settled 0.0107. Fixed: steady runs get up to 2,400 steps and stop on settled window-mean
   forces; on the benchmark this is 1.85x faster than running every step with C_L and C_D within 0.1 %.
4. **Grid sequencing gave no speed-up** **[observed]** (KI-70): starting the draft grid from a coarse pass jolts
   the fine field and the slow force oscillation sets the run time either way. Kept as an option, not default.

## 6. Data

* No experimental dataset is on disk. **[verified]** The XFOIL check used NASA airfoil data quoted in KI-16
  (source files to be catalogued, roadmap R-2).
* `data_store/` (7 MB): fighter design evaluations generated by this platform (synthetic by construction, labelled
  as such in their manifests).
* AirfRANS loader exists (`ml/airfrans.py`, needs the dataset downloaded); the dataset is **not** on disk.

## 7. Tests

The last full run of record (`results/full_tests.log`, 2026-11-08) was cut off before its totals; it lists 8
failures, mostly packaging and figure-catalogue tests in the thermal sub-project (`test_space`, `test_viz_catalogue`,
`test_phase9`, `test_labs`) and `test_stats_logging::test_parallel_and_logging`. A complete run is part of the
baseline (`BASELINE_RESULTS.md`).

## 8. Highest-value improvements (ranked)

1. **Experiment integrity** - never overwrite a run, report feasibility, registry of experiments. Cheap, removes
   the risk of reporting the wrong result. Done first (this audit).
2. **3D CFD validation** against a public wind-tunnel case (ONERA M6 wing is the usual first case) and the
   half/full-model discrepancy - without these no CFD optimisation result can be trusted.
3. **Mesh independence of the forces used in optimisation** - draft grids give C_D with unknown error.
4. **Surrogate-trust handling in the shape optimiser** - judge candidate designs with an independent solver
   where NeuralFoil's confidence drops, rather than only penalising.
5. **Controlled CMA-ES restart comparison** (current vs IPOP) on benchmark functions and on the real problem.
