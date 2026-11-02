# Progress
<!-- sources: quality_gate -->

Final gate (`python scripts/run_quality_gate.py` → `results/quality_gate/metrics.json`): 117 tests, 117 passed,
0 failed; ruff clean; mypy (strict, core packages) clean; coverage 82.66 % overall. Coverage by package: numerics
96.77 %, physics 98.88 %, validation 99.32 %, ml 94.23 %, optim 93.5 %, uq 98.57 % (targets: core ≥ 90 %, overall
≥ 80 %, both met). The `experiments` package is 26.91 %: its study drivers are exercised by the phase scripts,
not unit tests.

| Phase | Status | Gate evidence | Notes |
|---|---|---|---|
| 0 Planning + scaffold | PASSED | install, lint, types, tests; planning A–N | GitHub Actions green on Ubuntu + Windows + docs (run 14) |
| 1 Physics + classical solvers | PASSED | observed orders, MMS for every BC type, conservation, guards | Burgers pre-asymptotic (KI-1) |
| 2 PINNs + validation core | PASSED | autograd/FD, exact residual ≈ 0, determinism, serialisation | annealing negative (KI-6) |
| 3 Adaptive collocation study | PASSED WITH LIMITATIONS | 272 runs, stats, ablations, extras | core hypothesis not supported (KI-8, KI-9) |
| 4 Operators, hybrids, UQ | PASSED WITH LIMITATIONS | held-out/OOD distributions, resolution transfer, calibration, break-even | CG initial guess and multi-fidelity negative |
| 5 Optimisation + inverse | PASSED WITH LIMITATIONS | analytic optima, KKT, budgets, trusted re-verification | GD/Adam fail; corner optima (KI-10, KI-11, KI-12) |
| 6 Fluids + aerospace | PASSED | Ghia, DFG, analytic flows, every fluid BC tested, regime guards, ACCEPTed airfoil | OpenFOAM not installed |
| 7 Validation engine | PASSED WITH LIMITATIONS | fallback on OOD, trust reports, resume bit-identical, estimator within 2× | 9 % false fallbacks |
| 8 Workbench | PASSED | AppTest for 5 workspaces, background job, CLI and UI workflow | screenshots captured |
| 9 Docs, figures, paper, audit | PASSED WITH LIMITATIONS | claims audit 0 unmatched; wheel + twine + clean-venv smoke | LaTeX not compiled (no TeX); AirfRANS not run (KI-13) |

All phases were completed on 2026-11-02 in one run (see `docs/REPRODUCIBILITY.md` for measured wall-clock times).
