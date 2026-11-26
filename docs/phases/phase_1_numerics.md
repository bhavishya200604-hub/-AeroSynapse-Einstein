# Phase 1 — physics definition engine and classical solvers
- **Deliverables:** `physics/` (problem, BCs, library); `numerics/` (fd1d, heat, convdiff, burgers,
  wave, poisson, fem, amr, timestep, verification, guards, cards); `scripts/run_phase1.py`.
- **Dependencies:** Phase 0.
- **Mathematics:** Taylor truncation analysis, θ-method, von Neumann stability, conservative FV, TVD
  limiters, Galerkin FEM, Richardson/GCI.
- **Programming:** scipy.sparse assembly and LU, vectorised stencils.
- **Effort (basis: this run):** about 1 session.
- **Risks:** wrong ghost-node signs (caught by MMS); pre-asymptotic orders (documented).
- **Acceptance tests:** `tests/test_numerics.py`, `tests/test_physics.py`.
- **Definition of done:** orders within ±0.15; MMS for every BC type; conservation; divergence raises.
