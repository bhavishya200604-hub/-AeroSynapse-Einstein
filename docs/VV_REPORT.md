# Verification and Validation report
<!-- sources: phase1_verification, phase2_pinn_benchmarks, phase3_adaptive_collocation, phase4_operators_uq, phase6_fluids_aero, phase7_validation_engine, quality_gate -->

**Verification** asks whether the equations are solved correctly (code and solution verification). **Validation**
asks whether the model matches reality or an accepted benchmark. This platform has **no experimental data**: its
validation evidence is agreement with published high-resolution numerical benchmarks, so the highest tier
reached is 3 (verified software) for the classical solvers. The ML surrogates are tier 1 (research simulation).
Nothing is certified.

## 1. Code verification of classical solvers (`results/phase1_verification`)
Observed order on the finest pair (theory in brackets): heat 1D CN 2.000 (2); heat 2D CN 2.000 (2); conv-diff FV
upwind 0.955 (1), central 1.999 (2), QUICK 1.988 (2), van Leer 1.996 (2); Burgers 1.818 (2, pre-asymptotic);
wave leapfrog 2.000 (2); Poisson 2D FD 2.002 (2); P1 FEM 1.997 in L2 (2) and 0.999 in H1 (1); Poisson 1D with
Dirichlet–Dirichlet 2.000, Neumann–Robin 2.000, Robin–Dirichlet 2.000, periodic 2.001. Conservation: FV mass
drift 1.18e-14; leapfrog energy drift 4.3e-15. Tests: `tests/test_numerics.py` (MMS for each BC type, stability
guards, divergence guard, von Neumann property test).

NS solver (`results/phase6_fluids_aero`): Couette 4.16e-10; Poiseuille observed order 2.004; divergence at
round-off. Panel method: cylinder C_p equals the exact 1 − 4 sin²θ within the asserted test tolerance 1e-10. <!-- audit:ignore (test threshold, not a measurement) -->

## 2. ML verification (`tests/test_ml.py`, `tests/test_operators_uq.py`)
Autograd matches finite differences within the asserted tolerances (first derivatives rel. 1e-6, second 1e-4), parameter gradcheck <!-- audit:ignore (test thresholds) -->
passes, exact solutions give PDE residuals below the asserted 1e-9, hard constraints <!-- audit:ignore (test threshold) --> hold exactly on the boundary, training is
deterministic for a fixed seed, serialisation round-trips are bit-identical, and checkpoint/resume is
bit-identical.

## 3. Validation against benchmarks (`results/phase6_fluids_aero`)
| Benchmark | Quantity | This platform | Reference |
|---|---|---|---|
| Ghia et al. (1982) cavity Re = 100, 64² | RMS u / RMS v on centre-lines | 0.0018 / 0.0043 | tables (transcription) |
| Ghia et al. (1982) cavity Re = 400, 64² | RMS u / RMS v | 0.0036 / 0.0044 | |
| DFG 2D-1 cylinder Re = 20, h = 0.01 | C_D | 5.4636 | 5.5795 |
| Blasius boundary layer | max profile deviation | 0.00976 | computed by shooting (f''(0) = 0.33206) |

## 4. Scientific validation of ML (accuracy is against an independent reference, never the training loss)
* PINNs vs exact solutions (`results/phase2_pinn_benchmarks`, `results/phase3_adaptive_collocation`): see the
  Phase 2/3 summaries. Error distributions and worst cases are reported per run.
* Neural operators (`results/phase4_operators_uq`): FNO median/max per-sample error 0.0016 / 0.0348
  in-distribution, 0.0885 / 0.1431 out-of-distribution in Pe. The reference's own error on that OOD split is
  0.0682 (the accuracy floor).
* Calibration: deep-ensemble 95 % coverage 0.93 (ID) and 0.899 (OOD-Pe); GP 0.944 on a noisy heat QoI.
* Trust/fallback (`results/phase7_validation_engine`): OOD fallback rate 1.0; guarded worst-case error 0.00101
  vs 0.23389 for the raw surrogate on OOD-Pe queries.

## 5. Software quality (`results/quality_gate`)
See `results/quality_gate/metrics.json`: tests passed/failed, coverage overall and per package, ruff and mypy
status. Quoted in `PROGRESS.md`.

## 6. Validation tier
| Component | Tier | Evidence |
|---|---|---|
| FD/FV/FEM solvers, NS solver, panel method | 3 — verified software | observed orders, MMS, analytic flows, benchmark agreement |
| Optimisation engine | 2 — engineering analysis | analytic optima/KKT; illustrative engineering models |
| PINNs, FNO/DeepONet, GP/ensemble surrogates | 1 — research simulation | errors vs references, OOD failures documented |
| Whole platform | 1 | minimum over components |
| Experimental validation / certification | not reached | no experimental data; no certification |
