# Variable-treatment policy

This policy applies before modeling. Any exception must be explained in a notebook and recorded in the treatment log.

| Category | Examples | Default treatment |
|---|---|---|
| Dependent variable | GDP growth, GDP-per-capita growth | Never interpolate or impute; use an unbalanced panel and report sample changes. |
| Flow / rate | Inflation, investment, trade, R&D, patents | Do not interpolate by default. A theory-supported lagged or multi-year average belongs in a separate specification. |
| Stock / slowly changing index | Capital stock, population, human-capital index | Preserve missingness by default. Internal interpolation is permitted only after source review, with original values and a flag retained. |
| Categorical / institutional | Regime or policy indicator | No statistical imputation. |

The pipeline does not impute values. `interpolate_stock_with_flag` is deliberately opt-in and does not fill leading or trailing gaps.
