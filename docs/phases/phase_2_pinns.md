# Phase 2 — PINN framework and validation-engine core
- **Deliverables:** `ml/nets.py` (MLP, Fourier features, periodic embedding), `ml/autodiff.py`,
  `ml/problems.py` (heat, Poisson, Burgers, conv-diff, boundary layer, disk), `ml/pinn.py` (trainer,
  causal weighting, annealing, L-BFGS), `ml/sampling.py`, `ml/weighting.py`, `ml/bc_strategies.py`,
  `ml/baselines.py`, `ml/io.py`, `validation/metrics.py`, `scripts/run_phase2.py`.
- **Dependencies:** Phase 1 (exact solutions, classical reference timing).
- **Mathematics:** automatic differentiation, trial-function constraints, approximate distance
  functions, gradient-statistics weighting.
- **Programming:** PyTorch autograd with `create_graph`, deterministic seeding, process-parallel seeds.
- **Effort (basis: this run):** about 1 session plus 4 min of compute.
- **Risks:** loss imbalance (demonstrated), weak conv-diff accuracy (demonstrated).
- **Acceptance tests:** `tests/test_ml.py` (autograd vs FD, exact residual ≈ 0, hard BCs exact,
  determinism, serialisation, samplers, divergence).
- **Definition of done:** tests green; accuracy table vs exact solutions recorded.
