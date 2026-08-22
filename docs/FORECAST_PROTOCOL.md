# FORECAST PROTOCOL

## Information-set discipline

A forecast uses only information available at its origin. Later-vintage capital extensions are forecasts/current-information inputs when evaluating an earlier origin; they are never backfilled as if known historically.

## E1 — pre-2016 rolling pseudo-OOS

- Purpose: all tuning, lag choice, empirical prior calibration, forecast-combination weights, ML hyperparameters.
- Minimum history: 20 annual observations where the model permits.
- Horizons: h=1,3,5; h=10 where feasible.
- Formal model-comparison inference (DM/MCS) is based on this long error history with dependence-aware aggregation/bootstrap.

## E2 — sacred holdout

- Train through 2015 only.
- Evaluate 2016–2019.
- No tuning, model selection, empirical-prior construction using holdout observations, or averaging-weight estimation on this period.
- Run once after candidate specifications are frozen.
- Report 2015-origin h=1..4 paths and pooled/country loss distributions.
- Do not use low-power country-level DM tests based on only four annual errors.

## E3 — historical stress

- Freeze information at 2019.
- Report 2020 separately as shock-year performance.
- Report 2021–2023/25 separately as recovery/persistence performance.
- Keep this methodological/historical exercise separate from current policy probabilities.

## Current-information policy origin

- Use the latest vetted information set available at the final policy-origin date (~2025/26).
- Forecast Sixteenth Plan horizon, 2030, and Vision 2100/2043 scenarios.
- Every post-2019 input must carry an observed/extended/forecast provenance flag.

## Structural forecast modes

1. Conditional: realized future G/K/H supplied. Label as **performance given realized future inputs**.
2. Unconditional: future inputs forecast from the origin and uncertainty propagated. This is the headline structural forecasting mode.

## Metrics

Point: RMSE, MAE, MAPE.
Probabilistic: CRPS, log predictive score, 50/80/95% coverage and interval width.
ML enters probabilistic metrics only if a defensible predictive distribution is implemented.

## Forecast fairness

Compared models must share the same target definition, origin, vintage, and exogenous information set. Nested tuning is performed inside the training history only. BMA/stacking/DMA weights for E2 are learned pre-2016 only.
