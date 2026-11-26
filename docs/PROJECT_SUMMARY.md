# Project summary (technical)
<!-- sources: phase1_verification, phase2_pinn_benchmarks, phase3_adaptive_collocation, phase3_extras, phase4_operators_uq, phase4_generalisation_map, phase5_optimization, phase6_fluids_aero, phase7_validation_engine -->

Per-phase details, all with real numbers: `docs/phases/PHASE_N_SUMMARY.md`. Auto-generated tables:
`docs/RESULTS_TABLES.md`. Status: `PROGRESS.md`.

| Phase | What was built | Key evidence |
|---|---|---|
| 0 | scaffold, configs, records, registry, safety, CLI, CI, planning document A–N | gate: install, lint, types, tests |
| 1 | physics API, BCs, FD/FV/FEM solvers, AMR, verification tools | observed orders match theory (e.g. van Leer 1.996, FEM H1 0.999); AMR 2.92e-3 vs uniform 7.51e-2 at equal nodes |
| 2 | PINN framework, samplers, BC strategies, baselines, metrics | hard BCs: Poisson 1.76e-2 vs penalty 2.03e-1; annealing diverged (negative) |
| 3 | equal-budget sampler study, 7 samplers × 4 Pe × 8 seeds | U-shaped error in Pe; no significant gain from RAR/RAD/R3 (negative); budget barely matters |
| 4 | FNO, DeepONet, POD, CNN/MLP, hybrids, UQ | FNO ID 0.0016, OOD-Pe 0.0885; resolution transfer holds; CG initial guess worse (12 → 96 iterations); break-even 955 queries |
| 5 | optimisation engine, KKT, robust, protocol, adjoint inverse | Rosenbrock to 1.41e-10 (L-BFGS); QP μ = 1; α̂ = 0.1004 ± 0.00083 (adjoint) vs 0.0756 (PINN) |
| 6 | NS projection solver, all fluid BCs, BL, panel method, airfoil optimisation | Ghia RMS ≤ 0.0044 (64²); DFG C_D 5.4636 vs 5.5795; airfoil C_L 1.3777 (ACCEPT) |
| 7 | trust/fallback, resource estimator, notebooks, datasets, safety | OOD fallback 1.0; ID false fallback 0.09; estimator error +0.0613 (PINN) |
| 8 | Streamlit workbench (5 workspaces), end-to-end workflow | AppTest smoke tests; screenshots in `docs/screenshots/` |
| 9 | docs, figures A–G, paper package, claims audit, packaging | claims audit, wheel + clean-venv install |

**Mathematics covered:** θ-methods and von Neumann stability; conservative FV with TVD limiters; Galerkin FEM;
Richardson/GCI; Chorin projection on MAC grids; potential-flow panels with the Kutta condition; PINN residuals by
autodiff; Fourier operators; Gaussian processes; Sobol indices; KKT and second-order conditions; CVaR and chance
constraints; discrete adjoints; Kalman filtering.

**Principal negative results** (kept, not hidden): no sampler effect at equal budget (Phase 3); learning-rate
annealing diverges (Phase 2); the multi-fidelity FNO and learned CG initial guess did not help (Phase 4); the
PINN inverse estimate is 24 % off (Phase 5); 9 % false fallbacks (Phase 7).
