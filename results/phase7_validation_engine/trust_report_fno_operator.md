# Trust report — FNO conv-diff operator

> RESEARCH USE ONLY. AeroSynapse-Einstein results are not certified for aircraft, pressure vessels, industrial machinery or any safety-critical decision. Platform validation tier: 1/5 (Research simulation: exploratory, no accuracy guarantee.) Classical solvers: tier 3 (code-verified). ML surrogates: tier 1.

**Validation tier of this model:** 1/5 research simulation

## Accuracy (in-distribution)

| quantity | value |
|---|---|
| rel_l2_median | 0.002776 |
| rel_l2_max | 0.02695 |

## Generalisation (out-of-distribution Pe 200-1000)

| quantity | value |
|---|---|
| rel_l2_median | 0.1526 |
| rel_l2_max | 0.2339 |

## Generalisation (high-frequency inputs)

| quantity | value |
|---|---|
| rel_l2_median | 0.006603 |
| rel_l2_max | 0.05578 |

## Fallback behaviour

| quantity | value |
|---|---|
| in_distribution_fallback_rate | 0.09 |
| ood_pe_fallback_rate | 1 |
| ood_input_fallback_rate | 1 |

## Physical consistency

| quantity | value |
|---|---|
| mass_residual_threshold | 0.00314 |

## Calibration (Phase 4 ensemble)

| quantity | value |
|---|---|
| test_id_coverage_95 | 0.93 |
| ood_pe_coverage_95 | 0.8994 |

## Performance (Phase 4)

| quantity | value |
|---|---|
| speedup | 35.56 |
| break_even_queries | 955.1 |
