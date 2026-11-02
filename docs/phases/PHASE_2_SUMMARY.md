# Phase 2 summary — PINNs and validation core
<!-- sources: phase2_pinn_benchmarks -->

Source: `results/phase2_pinn_benchmarks/metrics.json`, produced by `python scripts/run_phase2.py --profile quick`
(3 seeds, MLP 3×32 tanh, 3000 Adam iterations, 1000 collocation points, float32).

| Problem | Strategy | rel. L2 mean ± sd |
|---|---|---|
| heat 1D | penalty | 8.22e-4 ± 9.6e-5 |
| heat 1D | hard (trial function) | 3.16e-4 ± 1.2e-4 |
| Poisson 2D | penalty | 2.03e-1 ± 2.8e-2 |
| Poisson 2D | hard | 1.76e-2 ± 1.6e-3 |
| Poisson 2D | penalty + annealing (α = 0.9) | 9.73e-1 ± 2.4e-2 (**negative**) |
| Burgers ν = 0.1 | penalty IC, periodic embedding | 1.96e-2 ± 1.4e-2 |
| conv-diff Pe = 2 (σ₀ = 0.05) | penalty IC | 3.32e-1 ± 3.9e-3 (**failure**) |
| conv-diff Pe = 10 (σ₀ = 0.05) | penalty IC | 1.58e-1 ± 5.6e-2 |
| boundary layer Pe = 20 | penalty | 4.97e-1 ± 8.5e-1 (one seed failed) |
| boundary layer Pe = 20 | hard | 6.14e-4 ± 1.7e-4 |
| boundary layer Pe = 20 | domain decomposition (2 subdomains) | 5.38e-2 ± 5.7e-2 |
| disk Poisson | penalty | 2.26e-3 ± 1.5e-4 |
| disk Poisson | hard (distance function φ = 1 − r²) | 4.05e-4 ± 2.2e-4 |

The baseline, a supervised MLP with the same architecture trained on exact labels, reached rel. L2
of 3.82e-3 (heat), 2.63e-2 (Poisson), 1.17e-2 (Burgers) and 6.81e-2 (conv-diff Pe 10). The classical
solvers reached rel. L2 of 2.42e-5 in 0.0096 s (heat FD, n = 128) and 6.83e-4 in 0.084 s (Poisson <!-- audit:ignore (grid sizes are inputs) -->
FD, n = 64). The PINN mean wall time was 21.2 s. <!-- audit:ignore (grid sizes are inputs) -->

**Hypotheses.** H2.1 (hard beats penalty) is supported for every problem tested. H2.2 (supervised
beats PINN) is *not* supported for heat (the PINN was better) or for hard-BC Poisson; it is supported
for conv-diff and Burgers. H2.3 is supported: the classical solvers are more accurate and 2–3 orders
of magnitude faster.

**Demonstrated failure: loss imbalance.** For penalty-BC Poisson, mean|∇L_r| / mean|∇L_bc| = 111 at
initialisation (seed 0). Learning-rate annealing as specified by Wang, Teng & Perdikaris (2021),
λ̂ = max|∇L_r| / mean|∇L_bc| with α = 0.9 (verified from the paper), made the error *worse*. The
diagnosis is positive feedback: as L_bc → 0 its gradient shrinks, so λ grows without bound (final λ_bc
between 4.15e8 and 2.03e9 over the 3 seeds). Proposed fix (not run): cap λ, or balance weighted gradient norms (NTK-style). The effective
remedy at this scale is hard enforcement.

**Demonstrated failure: conv-diff.** PINN error with a sharp initial pulse is large even at low Pe.
This motivates Phase 3.

**Limitations:** 3 seeds; small budgets; one architecture.
