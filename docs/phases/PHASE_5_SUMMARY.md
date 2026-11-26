# Phase 5 summary — optimisation engine and inverse problems
<!-- sources: phase5_optimization -->

Source: `results/phase5_optimization/metrics.json` (`python scripts/run_phase5.py --profile quick`, 3 seeds).

**Built:** design space (continuous/integer/categorical), recorded and budget-enforced problem API, 11
optimisers behind one interface with a capability table, KKT/second-order diagnostics, robust measures
(expectation, CVaR, chance, worst case), the 12-step protocol (`OptimizationStudy`), a GP trust-region
surrogate loop, a discrete-adjoint inverse solver and a PINN inverse route.

## Analytic benchmarks
| Problem | Method | Result | Known optimum |
|---|---|---|---|
| Rosenbrock, 400 evals | L-BFGS / SQP | f = 1.41e-10 / 2.77e-10 | 0 |
| Rosenbrock, 3000 evals | DE | median f = 9.53e-19 | 0 |
| Rosenbrock, 3000 evals | GA | median f = 0.00118 | 0 |
| Rosenbrock, 3000 evals | Adam (FD gradients) | f = 0.0315 (**not converged**) | 0 |
| Rosenbrock, 3000 evals | gradient descent | f = 101 (**failed**: fixed step in the scaled cube) | 0 |
| Convex QP | SQP | f = 0.5, μ = 1, KKT satisfied | 0.5, μ = 1 |
| Convex QP | DE | f = 0.5000000000000369, KKT satisfied | |
| Convex QP | GA / cBO / L-BFGS (penalty) | f = 0.5002 / 0.5002 / 0.5028; KKT check **fails** (constraint not active to tolerance) | |
| Branin, 40 evals | BO / GA / DE | median gap to the computed global minimum 0.39788735772973816: 0.00241 / 0.255 / 2.00 | |
| ZDT1 | NSGA-II, 4000 evals | HV 0.8755 vs analytic-front HV 0.8767 | |
| ZDT1, 40 evals | EHVI / NSGA-II | HV 0.688 / 0.264 | |

## Engineering demonstrations (protocol 8G; final candidates re-evaluated with the finer trusted model)
| Study | Baseline design f | Optimiser f (trusted) | Baseline optimiser f (trusted) | Verdict |
|---|---|---|---|---|
| Thermal rod (DE vs GA, 600 evals) | 6.344 | 0.6412 | 0.6412 | ACCEPT |
| Source placement (SQP vs DE) | 9.996 | 0.5748 | 0.5748 | ACCEPT |
| Cantilever mass [kg] (DE vs GA, 1500 evals) | 58.875 | 4.850 | 4.851 | ACCEPT |

Surrogate-assisted vs direct, thermal rod with 40 trusted evaluations (median of 3 seeds, re-verified at
4× resolution): GP trust region 0.6412, cBO 0.6422, DE 0.6866. Every re-verified candidate was feasible.
Multi-objective: the 3-objective beam search with NSGA-II gave a feasible non-dominated set of 1229 points.
On the 2-objective thermal problem at 40 evaluations, EHVI reached HV 25.46 and NSGA-II 6.11.

## Inverse problem (α of the heat equation, 40 noisy observations, σ = 0.01, true α = 0.1)
* Discrete-adjoint least squares: α̂ = 0.1004, standard error 0.00083. Over 20 noise replicates the mean
  was 0.1004 with standard deviation 0.00097, so the Gauss–Newton SE is consistent with the replicate spread.
* PINN with trainable α (3000 Adam iterations): α̂ = 0.0756 (**24 % error**), wall time 1078 s. That run
  shared 4 cores with other studies, so the time is not a clean benchmark.

## Hypotheses
H5.1 supported for L-BFGS, SQP and DE; GD and Adam failed within budget. H5.2 supported (μ = 1). H5.3
supported. H5.4 supported: every final candidate was feasible under the trusted solver. **H5.5 not
demonstrated**: the deterministic and robust optima coincide (m = 4, x_s = 0.1, both at bounds) and both
satisfy the chance constraint, because the cooling-cost weight λ = 0.1 makes the problem bound-dominated.
This is an uninformative demonstration, not evidence for robust design. H5.6 supported.

## Limitations
Corner optima in the thermal and source demos (objectives too weakly traded off); illustrative material
values; KKT diagnostics use finite differences.
