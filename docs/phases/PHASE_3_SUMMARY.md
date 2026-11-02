# Phase 3 summary — adaptive collocation across Péclet numbers (core study)
<!-- sources: phase3_adaptive_collocation -->

Source: `results/phase3_adaptive_collocation/` (`python scripts/run_phase3.py --profile quick`; post-hoc
analysis by `scripts/postprocess_phase3.py`). Total wall clock 4836 s on 4 CPU cores (shared with other runs).

**Design.** u_t + u_x = Pe⁻¹u_xx on periodic [0,1) × [0,1], Gaussian pulse σ₀ = 0.1, exact reference by
the method of images. MLP 3×32 tanh with periodic embedding, Adam 4000 iterations, 2000 collocation points
(the same budget for every sampler), resampling every 250 iterations, 8 seeds; a seed fixes the network
initialisation, so comparisons are paired. The hypotheses were written into `record.json` before the run.
Statistics: paired two-sided exact Wilcoxon signed-rank vs uniform, Holm-corrected within each Pe (6
comparisons); Friedman across all 7 samplers; percentile-bootstrap 95 % CIs of the median.

## Main result (median rel. L2 [95 % CI], 8 seeds)
| Pe | uniform | best other sampler | worst other sampler | Friedman p |
|---|---|---|---|---|
| 1 | 0.1503 [0.1500, 0.1505] | LHS 0.1502 | R3 0.1516 | < 1e-4 |
| 10 | 0.0180 [0.0104, 0.0375] | Sobol 0.0137 | RAR 0.0246 | 0.0001 |
| 100 | 0.0504 [0.0333, 0.1014] | LHS 0.0415 | RAR 0.0602 | 0.0323 |
| 1000 | 0.1793 [0.1308, 0.2825] | R3 0.1658 | RAD 0.1888 | 0.5461 |

Full table with all samplers, Holm-adjusted p-values and wall times: `docs/RESULTS_TABLES.md` and
`paper/tables/phase3.tex`. Every sampler used the same 2000 points and 4000 iterations. Residual
evaluations per run were 7.29e6 (RAR, fewer points early on), 8.12e6 (RAD) and 8.03e6 (R3), against
8.00e6 for uniform. Wall time was not equalised (machine load varied); it is reported per cell.

## Hypotheses (pre-registered) — outcomes
* **H3.1 supported.** Error is U-shaped in Pe: the minimum is at Pe = 10 (0.0180); Pe = 1 gives 0.1503 and
  Pe = 1000 gives 0.1793.
* **H3.2 not supported (negative result).** At Pe = 100 and 1000 no adaptive sampler (RAR, RAD, R3) differs
  significantly from fixed uniform sampling after Holm correction. The smallest adjusted p is 0.2344 (RAD,
  Pe = 1000), and its median is *higher* than uniform. At this budget and architecture, residual-based
  adaptivity does not reduce error for a smoothly transported pulse.
* **H3.3 partly contradicted.** At Pe = 10 no sampler differs (all Holm p ≥ 0.0938). At Pe = 1 RAR, RAD and
  R3 are *significantly worse* than uniform (Holm p = 0.0469 each), but by a practically negligible amount
  (median log10 ratio 0.002 to 0.004, i.e. < 1 %). The significance comes from the extremely small
  seed-to-seed spread at Pe = 1 (CI width about 5e-4).

## Ablations (paired vs the same configuration without the component; Holm over 6 tests, post-hoc)
| Ablation | median | baseline median | Holm p |
|---|---|---|---|
| Fourier features (16, σ = 1), Pe = 1000, uniform | 0.7894 | 0.1793 | 0.0469 (**much worse**) |
| Fourier features, Pe = 1000, RAD | 0.7913 | 0.1888 | 0.0469 (**much worse**) |
| causal weighting (ε = 1), Pe = 1000 | 0.1134 | 0.1793 | 0.0469 (better) |
| causal weighting, Pe = 1 | 0.1496 | 0.1503 | 0.0469 (negligible) |
| hard initial condition, Pe = 1000 | 0.0706 | 0.1793 | 0.0469 (better) |
| hard initial condition, Pe = 1 | 0.2394 | 0.1503 | 0.0469 (worse overall; early-time error 0.2703 vs 0.5384) |

**H3.4 partly supported.** Causal weighting helps at Pe = 1000 but does not reduce the early-time error at
Pe = 1 (0.5359 vs 0.5384). Fourier features at this scale *hurt*: an untuned σ injects high frequencies
that the smooth pulse does not need. The minimum attainable p with 8 paired seeds is 0.0078, so 0.0469 is
the smallest Holm-adjusted p possible for six tests; larger studies are needed for finer conclusions.

## Failure diagnosis (Pe = 1)
At Pe = 1 every sampler and every seed converges to nearly the same wrong solution (rel. L2 0.1500–0.1516).
The error is concentrated at early times: the median early-time (t ≤ 0.05) error is 0.5384. The pulse
diffuses on a time scale σ₀²·Pe/2 = 0.005, about 200× shorter than the domain, so this is an initial-layer <!-- audit:ignore (derived from inputs: 1/0.005) -->
stiffness failure. Hard enforcement of the initial condition halves the early-time error but worsens the
global error. Since the failure is localised in time, the natural remedies are time-marching / causal
training with a much finer early-time resolution, or a change of variables (e.g. similarity scaling).
These are proposed, not run.

## Novelty consequence
See `docs/LITERATURE_AND_NOVELTY.md`. The equal-budget negative result for RAD/RAR/R3 on this family is
informative, but it is a benchmark finding, not a new method.
