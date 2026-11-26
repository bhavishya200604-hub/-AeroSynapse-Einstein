# Phase 4 summary — neural operators, hybrids, uncertainty quantification
<!-- sources: phase4_operators_uq, phase4_generalisation_map -->

Source: `results/phase4_operators_uq/` (`python scripts/run_phase4.py --profile quick`, 597 s) and
`results/phase4_generalisation_map/`.

**Data.** The operator is G: (u₀, Pe) ↦ u(·, T = 0.5) for periodic conv-diff. Data come from the FV van Leer
solver: 400 training samples with Pe ∈ [5, 50] (log-uniform), generated in 27.0 s. Tests: 100 in-distribution
(ID), 100 with Pe ∈ [200, 1000] (OOD-Pe) and 100 with higher-bandwidth inputs (OOD-input). Leakage check:
0 shared samples between training and every test split. The reference's own error, measured against the exact
spectral solution (max norm), is 0.00473 on the ID test set but **0.0682 on OOD-Pe**. OOD-Pe errors below
that level are therefore not meaningful (the reference uncertainty is the floor).

## Operator accuracy (per-sample relative L2: median / worst case)
| Model | ID | OOD-Pe | OOD-input | train s |
|---|---|---|---|---|
| FNO | 0.0016 / 0.0348 | 0.0885 / 0.1431 | 0.0045 / 0.0393 | 28.5 |
| FNO, mass-conserving (physics-aware) | 0.0005 / 0.0039 | 0.0927 / 0.1575 | 0.0016 / 0.0088 | 28.0 |
| DeepONet | 0.0093 / 0.0800 | 0.2332 / 0.5074 | 0.0660 / 0.2874 | 6.5 |
| CNN | 0.0145 / 0.1132 | 0.2578 / 0.4265 | 0.4290 / 0.9329 | 11.6 |
| MLP (flattened grid) | 0.0036 / 0.0333 | 0.1490 / 0.2690 | 0.0130 / 0.0460 | 4.3 |
| POD + polynomial regression | 0.0005 / 0.0048 | 0.7307 / 1.9805 | 0.0009 / 0.0056 | 0.1 |

The mass-conserving FNO has a max mass error of 1.19e-7 (float32 round-off), against 0.00467 for the plain FNO.

**Resolution transfer** (trained at n = 64, evaluated on the same fields regenerated at higher resolution):
FNO median 0.0016 → 0.0017 (n = 128) → 0.0017 (n = 256); CNN 0.0145 → 0.0905 → 0.1214. DeepONet, with inputs
resampled to its 64 sensors, stays at 0.0093. The flattened MLP cannot be applied (fixed input layer).

**Hypotheses.** H4.1 *partly*: the FNO beats CNN, MLP and DeepONet ID, but a linear POD regressor matches the
mass-conserving FNO ID (median 0.0005). This is expected because the operator is linear in u₀. POD then fails
badly OOD-Pe (0.7307), because its parameter dependence is a polynomial fit. H4.2 supported: every model degrades
by more than 2× OOD-Pe (FNO 55×). H4.3 supported. H4.4 supported.

## Uncertainty quantification
* **Deep ensemble (5 FNOs).** 95 % coverage is 0.93 ID and 0.899 OOD-Pe. The mean absolute coverage error is
  0.0570 ID and 0.0255 OOD-Pe. The Spearman correlation between ensemble spread and error is 0.90 ID but only
  0.33 OOD-Pe, so the spread is a weak error indicator exactly where it is needed. **H4.5 partly supported.**
* **GP** (heat QoI, 30 noisy points): estimated noise 0.00954 (true 0.01), 95 % coverage 0.944 on 500 test
  points, mean absolute coverage error 0.0102.
* **Monte Carlo** through the FD heat solver (α ~ U[0.05, 0.15], 400 samples): mean 0.6078 ± 0.0045 (SE) vs
  analytic 0.6167 (z = −1.99; the difference includes FD discretisation error).
* **Sobol (Ishigami, 81920 evaluations):** S₁ = (0.3145, 0.4421, 0.0001) vs analytic (0.3139, 0.4424, 0);
  S_T = (0.5576, 0.4424, 0.2437) vs (0.5576, 0.4424, 0.2437).
* **Kalman filter** (8 sensors, wrong initial state): final RMSE 0.0012 vs 0.2383 for the free run; 2σ coverage
  1.0. That is over-coverage, which means the filter covariance is conservative.

## Hybrids (each against its conventional baseline)
* **Neural correction of a coarse first-order solver:** median 0.0045, against 0.0788 for the coarse solver
  alone and 0.0140 for the same CNN without the solver input. **H4.6 supported.**
* **Multi-fidelity (40 high-fidelity samples):** the FNO with the low-fidelity input reached 0.0121, the FNO on
  high-fidelity data only 0.0096. **Negative:** low-fidelity data did not help at this size.
* **FNO initial guess for CG (Poisson 2D):** median iterations to 1e-8 rose from 12 (zero initial guess) to **96**
  (FNO initial guess), although the FNO alone has 0.0678 relative error. **H4.7 refuted.** Diagnosis: the
  test sources contain only 9 Laplacian eigenmodes, so CG from zero converges in about that many iterations,
  while the FNO's error is spread over all modes. Learned initial guesses help only when the residual
  spectrum is broader than the guess error. Both runs reach the same final accuracy (relative error ~1e-9),
  because CG keeps the classical solver in charge.

## Speed-up and break-even (this machine, batch size 1)
FNO query 0.00168 s vs FV solve 0.0598 s, i.e. speed-up 35.6. Including FNO training (28.5 s) and data
generation (27.0 s), the surrogate pays off after **955 queries**, and only for in-distribution queries:
OOD-Pe errors are about 55× larger.

## Generalisation map (Fig. G; `results/phase4_generalisation_map`)
The trained FNO was evaluated on fresh samples over Pe ∈ [1, 3000] × input bandwidth k_max ∈ [2, 10] (training:
Pe ∈ [5, 50], k_max = 4). Median errors inside the training Pe range are 0.0011–0.0067. They rise to 0.3247–0.444
at Pe = 1 and to 0.1113–0.1351 at Pe = 3000. At fixed Pe the error barely depends on bandwidth (e.g. Pe = 10:
0.0011–0.0016 for k_max from 2 to 10), because diffusion damps the high-k modes before T = 0.5. **H4.8 half
supported:** extrapolation in Pe fails in both directions; extrapolation in bandwidth is benign for this operator.
