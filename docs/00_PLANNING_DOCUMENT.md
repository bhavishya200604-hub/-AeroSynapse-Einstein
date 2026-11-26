# AeroSynapse–Einstein — Planning Document (Phase 0 deliverable)

Date: 2026-11-02. Status: written before implementation; later deviations are logged in
`docs/DECISIONS.md`. Literature cited here is verified in `docs/LITERATURE_AND_NOVELTY.md`
(each entry fetched from arXiv or another source during this run).

---

## A. Project definition

**AeroSynapse–Einstein** is an open-source Python research platform. A user can (1) define a PDE
problem, (2) solve it with *verified* classical solvers, (3) train scientific-ML surrogates (PINNs,
FNO, DeepONet, hybrids), (4) check those surrogates with a validation engine that is independent of
training loss, and (5) use them in constrained design optimisation. The final candidate design is
always re-checked with the trusted solver.

**What it is:** a laptop-scale (CPU-first) testbed for reproducible studies of *where scientific ML
helps and where it fails*. Every claim traces to a `results/<id>/` artifact.

**What it is not:** a production CFD code, a turbulence solver, a certified engineering tool, or a
general "AI that solves physics". Laminar, low-dimensional, mostly 1D/2D problems only.

**Ultimate purpose:** to connect mathematics → physics → numerical simulation → scientific ML →
optimisation → engineering decisions through executed experiments that say plainly when a learned
model should be trusted and when to fall back to a conventional solver.

## B. Feasibility assessment

| Item | Feasible for one student on a CPU laptop? | Notes |
|---|---|---|
| Verified FD/FV/FEM solvers for 5 Level-1 PDEs | Yes | Fully implemented in Phase 1 |
| PINNs for 1D/2D problems | Yes | Thousands of Adam iterations take seconds to minutes on a CPU |
| Multi-seed adaptive-collocation study | Yes, at reduced scale | Budget: ~10³ collocation points, ~2–5k iterations, ≥5 seeds |
| FNO / DeepONet on 1D parametric data | Yes | Datasets of 10²–10³ samples from our own solvers |
| 2D incompressible NS (cavity, channel) | Yes, low Re and coarse grids | Re ≤ 1000, ≤ 64² cells |
| Panel-method aerodynamics | Yes | Inviscid; no profile drag |
| RANS/LES, turbulent aerodynamics | No | Needs external CFD (OpenFOAM, Mode D) and cluster time |
| 3D geometry, unstructured meshes, elasticity, EM, QM | Long-term | Plugin interface only (Level 4) |
| Experimental validation | No | No lab data, so validation tier ≤ 3 |

## C. Research direction

Five candidate starting questions were scored 1–5 on the nine Section 17 criteria. The evidence for
each score is in the "basis" row; references are verified in `LITERATURE_AND_NOVELTY.md`.

| Criterion | Q1 PINN + adaptive collocation, transient conv-diff vs Pe | Q2 PINN transient heat | Q3 Neural operators, parametric PDEs | Q4 FNO flow prediction | Q5 Hybrid numerical-ML | Q6 AI aerodynamic optimisation |
|---|---|---|---|---|---|---|
| Mathematical depth | 4 | 2 | 4 | 3 | 4 | 3 |
| Novelty potential | 2 | 1 | 2 | 1 | 3 | 2 |
| Computational feasibility | 5 | 5 | 4 | 2 | 4 | 4 |
| Data availability (self-generated) | 5 | 5 | 4 | 3 | 4 | 4 |
| Validation difficulty (5 = easy) | 5 (exact solution) | 5 | 4 | 2 | 4 | 3 |
| Publication potential | 2 | 1 | 2 | 1 | 3 | 2 |
| Aerospace / sci-comp relevance | 4 | 2 | 4 | 5 | 4 | 5 |
| Fit for an ME student | 5 | 5 | 4 | 3 | 4 | 4 |
| Fit for constrained resources | 5 | 5 | 4 | 2 | 4 | 4 |
| **Total / 45** | **37** | 31 | 32 | 22 | 34 | 31 |

Basis: residual-based adaptive sampling for PINNs has been benchmarked extensively already (Lu et
al. 2021, RAR; Wu et al. 2023, RAD/RAR-D over more than 6000 PINN runs; Daw et al. 2023, R3). PINN
failure on convection-dominated problems is documented (Krishnapriyan et al. 2021), and so are its
remedies (Wang et al. 2021 gradient pathologies; Wang et al. 2022 causal training). Q1 is therefore
**feasible and well-posed but low in novelty**. Its value is as a controlled, reproducible
re-examination with exact reference solutions, equal compute budgets and multiple seeds, focused on
the Péclet dependence. Q5 has slightly higher novelty potential but needs the Q1 and Q3
infrastructure first.

**Decision (ADR-0003):** start with Q1 as the core study. The novelty bar is a careful, honest
benchmark on a precisely defined transient conv-diff family, not a new method. Q3 and Q5 follow as
Phases 4–5 and reuse the same infrastructure.

## D. Mathematical foundation (initial problem)

Transient 1D convection–diffusion on the periodic unit interval, non-dimensionalised with length L
and speed v:

  u_t + u_x = Pe⁻¹ u_xx,  x ∈ [0, 1) periodic,  t ∈ (0, T],  u(x, 0) = exp(−(x − x₀)²/(2σ₀²)).

The Péclet number is Pe = vL/D. The exact solution (method of images) is

  u(x, t) = Σ_k (σ₀/σ(t)) exp(−(x − x₀ − t − k)²/(2σ(t)²)),  σ(t)² = σ₀² + 2t/Pe.

Assumptions: constant velocity and diffusivity, no source, linear. As Pe grows, the pulse stays sharp
(σ(T) → σ₀), so the solution's space-time "frequency content" grows. That is the regime where PINNs
are known to struggle. Boundary conditions are periodic. A second, steady boundary-layer problem
(Dirichlet u(0)=0, u(1)=1; exact u = (e^{Pe x} − 1)/(e^{Pe} − 1)) stresses thin layers of thickness
O(1/Pe).

PINN loss: L = λ_r L_r + λ_i L_i + λ_b L_b (+ λ_d L_d). Here L_r is the mean squared residual
r = u_t + u_x − Pe⁻¹ u_xx at collocation points, L_i the IC misfit, and L_b the periodicity misfit of
u and u_x (or hard enforcement through the periodic embedding (cos 2πx, sin 2πx)).

Optimisation formulation (Phase 5): min_x f(x) s.t. g_i(x) ≤ 0, h_j(x) = 0, x_min ≤ x ≤ x_max. The
dimensionless thermal objective is T_max/ΔT_ref + λ P/P_ref, with λ stated together with its unit
conversion (Section H).

## E. System architecture

```
Layer F  ui/ (Streamlit workbench)  ── calls only the public API ──┐
Layer E  validation/ (metrics, trust report, fallback)  <──────────┤
Layer D  optim/ (variables, problems, algorithms, KKT, robust, 8G protocol)
Layer C  ml/ (MLP, Fourier features, PINN, samplers, FNO, DeepONet, POD, hybrids) + uq/
Layer B  numerics/ (FD, FV, FEM, NS projection, panel method, verification)
Layer A  physics/ (PDEProblem, Parameter, BCs, dimensionless numbers, library, plugins)
Cross-cutting: config (pydantic), experiments (records, runs, provenance), registry (plugins),
exceptions, logging, device, safety (validation ladder)
```

Interfaces: a `PDEProblem` is consumed by both solvers and PINNs. Solvers return arrays plus
diagnostics and raise `SolverDivergenceError`/`StabilityError` on failure. Models implement
`fit / predict` (operators) or `train / evaluate` (PINNs). Optimisers implement
`minimize(problem, budget) → OptimizationResult` with a full evaluation history. The validation
engine takes predictions and references and never sees the training loss.

## F. Technology stack

| Tool | Why | Rejected alternatives |
|---|---|---|
| Python ≥ 3.10 | ecosystem, Windows support | Julia (smaller ML ecosystem for the user) |
| NumPy / SciPy (sparse LU, optimize, qmc, stats) | classical solvers, SQP/L-BFGS, Sobol/LHS, statistics | PETSc (overkill) |
| PyTorch (CPU, optional CUDA) | autograd for PINNs, FNO/DeepONet | JAX (Windows support weaker); evaluated, not used |
| pydantic + YAML | validated configs | Hydra (extra complexity) |
| Matplotlib (+ Plotly for interactive HTML) | figures | ParaView (optional for 2D/3D CFD output, not required) |
| Streamlit | workbench, testable with `AppTest` | Dash (more boilerplate) |
| pytest + hypothesis, ruff, mypy | quality gates | — |
| OpenFOAM | optional Mode D adapter only | not required |

## G. Complete requirements specification

Priorities: P0 scientific correctness, P1 first useful release, P2 research expansion, P3 long-term.

| ID | Requirement | Pri | Scientific justification | Implementation | Verification | Acceptance criterion |
|---|---|---|---|---|---|---|
| A1 | Problem API N(u;μ)=f, B(u;μ)=g, IC | P0 | single source of truth for solver and ML | `physics.problem.PDEProblem` | unit tests | invalid domains/BCs/params raise `ConfigurationError` |
| A2 | Parameters with units and ranges | P0 | dimensional consistency | `Parameter` | unit test | out-of-range raises |
| A3 | Dimensionless numbers Pe, Re, Fo, Bi, CFL | P1 | regime identification | `dimensionless_numbers` | unit test | matches hand calculation |
| A4 | Dirichlet/Neumann/Robin/periodic/initial/interface BCs | P0 | Section 6 | `physics.bc` | MMS per type | observed order 2 ± 0.15 |
| A5 | Five Level-1 PDE definitions with exact solutions | P0 | verification truth | `physics.library` | residual of exact solution | FD residual < 1e-3 |
| B1 | FD heat 1D/2D, θ-method | P0 | Stage 1 | `numerics.heat` | convergence | order 2 (CN), 1 (BE) |
| B2 | FV conv-diff: upwind, central, ≥1 higher-order | P0 | conservative transport | `numerics.convdiff` | Gaussian exact | orders 1/2/2/2; mass drift < 1e-12 |
| B3 | Burgers, wave, Poisson solvers | P0 | Level 1 | `numerics.burgers/wave/poisson` | Cole–Hopf, standing wave, MMS | order ≈ 2 |
| B4 | P1 FEM cross-check | P1 | independent method | `numerics.fem` | MMS | L2 order 2, H1 order 1 |
| B5 | Adaptive time stepping | P1 | efficiency, error control | step doubling, CFL-adaptive | test | dt varies, error < tol scale |
| B6 | Mesh refinement | P1 | boundary layers | `numerics.amr` | test | AMR error < 0.1 × uniform at equal nodes |
| B7 | Verification tools: order, Richardson, GCI, von Neumann | P0 | V&V | `numerics.verification` | analytic sequences | exact p recovered |
| B8 | Divergence and stability guards | P0 | Section 16 | `guards`, `StabilityError` | tests | NaN/blow-up raises |
| B9 | 2D incompressible NS projection solver | P1 | Level 2 | `numerics.ns2d` | Poiseuille, Couette, Ghia cavity | analytic error < 1e-3; cavity RMS vs Ghia < 0.02 (Re=100) |
| B10 | Fluid BCs: no-slip, slip, inlet velocity/pressure, outlet, symmetry, periodic, far-field | P1 | Section 6 | `numerics.ns2d` BC specs | one test each | tests pass |
| B11 | Regime guard (laminar/transitional/turbulent) | P0 | rule 5 | `physics.regime` | tests | Re > limit raises `RegimeError` unless `extrapolate=True` |
| B12 | Panel method, NACA 4-digit, CST | P1 | Level 3 | `numerics.panel`, `physics.airfoil` | thin-airfoil theory | C_L slope within 10% of 2π for 12%-thick airfoil |
| B13 | OpenFOAM adapter (optional) | P3 | Mode D | `numerics.external` | pre-flight test | missing binary reported, not crash |
| C1 | MLP + Fourier features | P0 | PINN backbone | `ml.nets` | shape/grad tests | autograd vs FD gradient rel err < 1e-6 |
| C2 | PINN residual/IC/BC/data/conservation/interface/regularisation losses | P0 | Section 7 | `ml.pinn` | exact-solution residual ≈ 0 | residual of exact solution < 1e-10 |
| C3 | Hard BC by trial functions; penalty; domain decomposition; SDF | P1 | Section 6 | `ml.bc_strategies` | comparison experiment | results reported |
| C4 | Samplers: uniform, LHS, Sobol, RAR, RAD, (R3) | P0 | core study | `ml.sampling` | unit tests | budgets respected |
| C5 | Adaptive loss weighting, only after a demonstrated failure | P1 | rule 12 | `ml.weighting` | ablation | evidence logged |
| C6 | Causal training, if long-time failure is demonstrated | P2 | Section 7 | `ml.pinn` causal weights | experiment | evidence logged |
| C7 | Supervised-MLP and classical-solver baselines | P0 | rule 4 | `ml.baselines` | experiment | reported |
| C8 | FNO 1D/2D, DeepONet | P1 | Layer C | `ml.fno`, `ml.deeponet` | held-out params | error distribution + worst case reported |
| C9 | POD-ROM, small CNN baseline, multi-fidelity, physics-aware operator | P2 | baselines | `ml.rom`, `ml.cnn` | experiment | reported |
| C10 | Hybrid: solver + neural correction, residual correction, data assimilation | P2 | Section 7 | `ml.hybrid` | vs conventional baseline | reported |
| C11 | GNN only if justified | P3 | rule 12 | interface | decision log | ADR recorded |
| C12 | Serialisation, determinism, inference reproducibility | P0 | ML verification | `ml.io` | round-trip test | bitwise-equal predictions |
| D1 | Design variables (continuous/integer/categorical, normalisation) | P1 | Section 8A | `optim.variables` | tests | encode/decode identity |
| D2 | Objective/constraint API with provenance | P1 | Section 8B/C | `optim.problem` | tests | every evaluation recorded |
| D3 | Algorithms: GD, L-BFGS, Adam, SQP, GP-BO, cBO, EHVI, GA, DE, NSGA-II | P1 | Section 8D | `optim.algorithms` | analytic benchmarks | known optima found within tolerance |
| D4 | Capability table per algorithm | P1 | Section 8D | `optim.capabilities` | doc test | complete |
| D5 | KKT / second-order / conditioning diagnostics | P1 | Section 8E | `optim.kkt` | analytic KKT problem | KKT residual < 1e-6 |
| D6 | Robust: expectation, CVaR, chance constraints, worst case | P2 | Section 8F | `optim.robust` | analytic test | MC estimate within 3 standard errors |
| D7 | 12-step protocol enforced in code | P0 | Section 8G | `optim.protocol` | tests | missing step raises |
| D8 | Surrogate-assisted loop with trust region; trusted re-evaluation | P1 | rule 11 | `optim.surrogate` | test | final candidate re-verified |
| D9 | Inverse problem: PINN route and adjoint route | P1 | Sections 1, 3 | `optim.inverse` | synthetic data | parameter recovered within noise-consistent CI |
| E1 | Metrics: rel/abs L2, L∞, MAE, RMSE, conservation, BC error, energy drift, speed-up, memory | P0 | Section 10 | `validation.metrics` | unit tests | match hand calculations |
| E2 | Error distributions + worst case | P0 | Section 10 | `validation.metrics.error_distribution` | test | quantiles reported |
| E3 | Trust report + fallback decision | P1 | Section 3 | `validation.trust` | OOD inputs | fallback triggers |
| E4 | UQ: ensembles, GP, MC, Sobol, intervals, calibration | P1 | Section 9 | `uq.*` | coverage tests | coverage within binomial CI on synthetic data |
| E5 | Resource estimator calibrated by micro-benchmarks | P2 | Section 12 | `validation.resources` | measured vs predicted | error reported |
| E6 | Validation ladder + banner in CLI/UI/reports | P0 | Section 16 | `safety` | tests | banner present |
| F1 | Workbench with 5 workspaces | P1 | Section 14 | `ui/app.py` | AppTest smoke | all pages render |
| X1 | Experiment records with all Section 13 fields | P0 | reproducibility | `experiments.record` | test | fields present, hypothesis written before run |
| X2 | Config validation | P0 | | `config` | test | unknown keys rejected |
| X3 | Plugin registry | P1 | extensibility | `registry` | test | register/get |
| X4 | CLI (run/train/optimize/validate/report/reproduce/figures/doctor) | P1 | | `cli` | test | each command exits 0 on quick profile |
| X5 | CI, pre-commit, Docker, devcontainer, lock file | P1 | | `.github/` etc. | CI run | green (if CI available) |
| X6 | Docs site, README, summaries, paper package, claims audit | P1 | | `docs/`, `paper/` | `scripts/audit_claims.py` | zero unmatched claims |
| L4 | Level-4 physics plugin interface + one demo plugin | P3 | Section 5 | `plugins/oscillator.py` | symplectic energy test | bounded energy error |

## H. Optimisation specification

- **Design variables:** continuous (scaled to [0, 1] by (x − lo)/(hi − lo)), integer (rounded after
  decoding; handled natively by GA/DE), categorical (one-hot for BO, index for GA).
- **Objectives:** single (scalar f), weighted (Σ w_k f_k/f_k,ref with every f_k,ref stated, so
  that the sum is dimensionless), Pareto (vector f), robust (E_ξ f, CVaR_β), and risk-sensitive.
- **Constraints:** written as g_i(x) ≤ 0 after normalisation by a reference value. Equalities are
  h_j(x) = 0 with tolerance ε_h. Bounds are handled by the encoding.
- **Feasibility:** max_i g_i ≤ 0 and max_j |h_j| ≤ ε_h. Feasibility is re-checked with the trusted
  solver.
- **Algorithms:** gradient descent, L-BFGS-B, Adam, SLSQP (SQP), GP-EI BO, constrained BO (EI × PoF),
  EHVI (2–3 objectives, Monte Carlo), real-coded GA, DE/rand/1/bin, NSGA-II.
- **Stopping:** an evaluation budget is mandatory. Optional stops: ‖∇L‖ tolerance, stall
  generations, or hypervolume stagnation.
- **Uncertainty:** the GP posterior σ for surrogate points, Monte Carlo over ξ for robust
  objectives, and ensemble spread for neural surrogates.
- **Final validation (Section 8G steps 11–12):** the best candidate is re-evaluated with the trusted
  solver. It is accepted only if it is feasible there. The surrogate–solver discrepancy is reported.

## I. Verification and validation plan

1. **Code verification:** MMS and exact solutions for every solver. Observed order must agree with
   the formal order within 0.15 on the finest pair. Conservation to round-off for conservative
   schemes. Stability limits enforced by tests.
2. **ML verification:** gradients by autograd vs finite differences; the residual of an exact
   solution is zero; determinism with fixed seeds; serialisation round-trip.
3. **Solution verification:** GCI for the reference solutions used in ML studies. Exact solutions
   are used wherever they exist, so the reference error is zero apart from quadrature.
4. **Validation:** NS cavity vs Ghia et al. (1982) tables (fetched); panel method vs thin-airfoil
   theory. No experimental data are used, so the maximum tier is 3 (verified software) for
   classical solvers and 1 for ML surrogates.
5. **Scientific validation of ML:** held-out parameters, out-of-distribution regimes, error
   distributions with worst case, calibration curves.

## J. Repository structure

```
src/aerosynapse/            library (import name)
  physics/                  Layer A: problem, BCs, library, regime, airfoil
  numerics/                 Layer B: fd1d, heat, convdiff, burgers, wave, poisson, fem, amr, ns2d, panel,
                            verification, cards, guards, timestep
  ml/                       Layer C: nets, pinn, sampling, weighting, bc_strategies, fno, deeponet, rom, hybrid
  uq/                       ensembles, gp, sobol, calibration
  optim/                    Layer D: variables, problem, algorithms, kkt, robust, protocol, surrogate, inverse
  validation/               Layer E: metrics, trust, resources
  experiments/              records, runner, studies (one module per phase study)
  viz/                      plotting helpers, colour-blind-safe palette
  plugins/                  Level-4 demo plugin(s)
  ui/                       Streamlit workbench (Layer F)
  data/                     fetched benchmark data with provenance
  cli.py, config.py, registry.py, exceptions.py, safety.py, device.py
tests/                      pytest (markers unit/verification/validation/slow/gpu)
configs/                    YAML experiment configs (quick/standard/full)
scripts/                    run_phaseN.py, reproduce_all.py, audit_claims.py, make_figures.py
results/<experiment_id>/    record.json, config.yaml, metrics.json, history.csv
docs/                       planning, decisions, phases, V&V report, literature, guides
paper/                      LaTeX, figures, figure_scripts, claims audit, readiness
notebooks/                  Colab/Kaggle notebooks for full-profile runs
```

## K. Development roadmap

| Phase | Milestone (measurable) |
|---|---|
| 0 | install + lint + type-check + tests green on skeleton; this document |
| 1 | all Level-1 solvers with observed orders within ±0.15; MMS for every BC type |
| 2 | PINN exact-solution residual ≈ 0; heat/Poisson PINN errors reported vs exact |
| 3 | Pe-sweep × sampler × ≥5 seeds study with CIs, Wilcoxon tests, ablations |
| 4 | FNO/DeepONet held-out error distributions; resolution transfer; calibration; break-even queries |
| 5 | analytic optimisation benchmarks solved; 8G protocol enforced; inverse problem recovered |
| 6 | Ghia comparison; analytic flows; fluid BC tests; regime guard; airfoil optimum re-verified |
| 7 | trust report per model family; fallback on OOD inputs; resume test; estimator error |
| 8 | workbench smoke tests for 5 workspaces; end-to-end quick workflow |
| 9 | docs, figures, paper, claims audit with zero unmatched claims, clean-env install |

## L. Risk register

| Risk | Type | Likelihood | Mitigation |
|---|---|---|---|
| PINN fails at high Pe | scientific | high (documented) | report as a negative result; ablate Fourier features / causal weights |
| No novelty | research | high | honest verdict; position the work as a careful benchmark |
| CPU budget too small for the multi-seed study | computational | medium | small networks, measured wall-clock, quick/full profiles |
| Reference solution error contaminates ML error | scientific | low | exact solutions; GCI otherwise |
| Laminar solver misused at high Re | scientific | medium | `RegimeError` guard |
| Fabricated benchmark values | integrity | — | data fetched with provenance in `data/` |
| Scope explosion | software | high | P0 → P3 ordering; DEFERRED entries with reasons |
| Windows-specific failures | software | medium | pathlib, CI matrix includes Windows (if CI runs) |

## M. Research potential

- **Established methods (no novelty):** FD/FV/FEM solvers, PINNs, RAR/RAD sampling, FNO, DeepONet,
  GP-BO, NSGA-II, panel methods.
- **Engineering integration (useful, not novel):** one platform linking verified solvers, an
  independent validation engine, trust/fallback logic and the 8G optimisation protocol with trusted
  re-verification.
- **Potentially novel (to be tested, not claimed):** (i) a quantitative map of *when* residual-based
  adaptive collocation stops helping as Pe grows under equal compute, with exact references and
  paired statistics; (ii) the break-even query count of surrogates vs the verified solver, measured
  on the same machine. Both may already exist in the literature; Phase 3 runs the audit.

## N. Immediate next action

First implementation task: the Layer A problem/BC API and the 1D FD assembler with Dirichlet,
Neumann, Robin and periodic ends, verified by MMS.

Acceptance criteria: observed order 2 ± 0.15 for −u'' = f with each BC combination (DD, NR, RD, PP,
NN); a pure-Neumann problem with incompatible data raises `ConfigurationError`; ruff and mypy
clean.

Tests required before moving on: `tests/test_physics.py`, and `tests/test_numerics.py::test_poisson1d_all_bcs`
and `::test_pure_neumann_incompatible_data_raises`.
