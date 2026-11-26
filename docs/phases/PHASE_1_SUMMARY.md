# Phase 1 summary — classical solvers and verification
<!-- sources: phase1_verification -->

**Built:** a problem-definition API and a BC framework; seven solvers (FD heat 1D/2D, FV conv-diff
with four face schemes, FV Burgers, FD wave, FD Poisson 1D/2D, P1 FEM Poisson); adaptive time
stepping (step doubling); 1D adaptive mesh refinement; verification tools; solver cards
(`numerics/cards.py`).

**Real results** (`results/phase1_verification/metrics.json`, `python scripts/run_phase1.py --profile quick`).
Observed orders on the finest pair:

| Solver | Theory | Observed |
|---|---|---|
| heat 1D CN | 2 | 2.000 |
| heat 2D CN | 2 | 2.000 |
| conv-diff FV upwind, Pe=50 | 1 | 0.955 |
| conv-diff FV central | 2 | 1.999 |
| conv-diff FV QUICK | 2 | 1.988 |
| conv-diff FV van Leer | 2 | 1.996 |
| Burgers FV (ν=0.1) | 2 | 1.818 (pre-asymptotic) |
| wave leapfrog | 2 | 2.000 |
| Poisson 2D FD | 2 | 2.002 |
| P1 FEM L2 / H1 | 2 / 1 | 1.997 / 0.999 |
| Poisson 1D DD / NR / RD / PP | 2 | 2.000 / 2.000 / 2.000 / 2.001 |

Other results: FV mass drift 1.18e-14; leapfrog relative energy drift 4.3e-15. At Pe = 100, AMR
with 70 nodes gives a max error of 2.92e-3, against 7.51e-2 on a uniform grid with the same number
of nodes.

**Failure modes demonstrated:** at cell Péclet 50 the central scheme undershoots below −0.1, while
upwind stays monotone (test `test_central_oscillates_at_high_cell_peclet`). Explicit schemes beyond
their stability limits raise `StabilityError`.

**Limitations:** uniform grids (except AMR in 1D); 2D heat is Dirichlet-only; Burgers is periodic
only.

**Next phase depends on:** `physics.library.convdiff_exact` (reference for the PINN study) and the FV
solver (data generation for neural operators).
