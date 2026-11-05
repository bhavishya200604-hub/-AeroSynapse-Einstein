# Changelog
All notable changes are documented here (Keep a Changelog; semantic versioning).

## [Unreleased]
### Added
- Phase 0: package scaffold (`src/aerosynapse`), pydantic configs, experiment records (Section 13
  fields), plugin registry, exception hierarchy, device/hardware report, safety ladder, CLI `doctor`,
  CI/pre-commit/Docker/devcontainer, planning document A–N.
- Phase 1: physics definition API, BC framework (Dirichlet/Neumann/Robin/periodic/initial/interface),
  Level-1 problem library with exact solutions; FD heat 1D/2D, FV conv-diff (upwind/central/QUICK/
  van Leer), Burgers FV, wave leapfrog, Poisson FD 1D/2D, P1 FEM, adaptive dt, 1D AMR, verification
  toolkit (order, Richardson, GCI, von Neumann), solver cards.
- Phase 2: PINN framework (MLP, Fourier features, periodic embedding, hard/penalty BCs, causal weighting,
  annealing, L-BFGS), 7 collocation samplers, domain decomposition, SDF hard constraints, supervised baseline,
  validation metrics.
- Phase 3: pre-registered equal-budget sampler study (7 samplers × 4 Pe × 8 seeds), ablations, post-hoc Holm,
  budget sweep, field snapshots.
- Phase 4: dataset generator with cards and leakage checks; FNO 1D/2D, DeepONet, POD, CNN/MLP baselines;
  mass-conserving FNO; neural correction, CG initial guess, Kalman data assimilation; GP, deep ensembles, MC,
  Sobol, calibration; generalisation map.
- Phase 5: design space, recorded/budgeted problems, 11 optimisers + capability table, KKT diagnostics, robust
  measures, 12-step protocol, GP trust-region surrogate, discrete adjoint and PINN inverse routes, engineering demos.
- Phase 6: MAC projection NS solver with 8 fluid BC types and immersed bodies, Blasius/BL marching, regime and
  Mach guards, NACA/CST airfoils, Hess–Smith panel method, OpenFOAM adapter, oscillator plugin, airfoil optimisation.
- Phase 7: trust policy + fallback + trust reports, affine resource estimator, notebooks, dataset catalogue.
- Phase 8: Streamlit workbench (5 workspaces, background jobs), end-to-end workflow runner (`aerosynapse run`).
- Phase 9: figures A–G (PDF/SVG/600-dpi PNG/HTML), paper package (LaTeX, generated tables, verified bib, claims map,
  readiness), source-scoped claims audit with negative control, reproduce_all, quality gate, docs site.
- AirfRANS integration (loader, derived table, surrogate, Kaggle notebook) — not run (network policy).
### Fixed
- Cylinder force accounting (pressure contribution), panel-method sign convention, CST validity check,
  airfoil crossing constraint, annealing α, resource-estimator model, 2D Poisson assembly vectorised.
