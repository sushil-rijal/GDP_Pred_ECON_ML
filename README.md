# Global Economic Growth Analysis

Reproducible cross-country research on economic growth, combining transparent panel-data construction, descriptive analysis, panel econometrics, and a clearly separated machine-learning extension.

## Research scope

The project asks how capital, labor, human capital, technology, and macroeconomic conditions relate to cross-country growth outcomes. It prioritizes measurement, missing-data transparency, and econometric diagnostics over predictive accuracy alone.

## Reproducibility principles

- Raw data and the original student project are preserved without modification.
- Dependent variables are never interpolated or imputed.
- Flow variables (for example, inflation and annual growth) are not interpolated by default.
- Interpolation is allowed only for documented stock/index variables, and every affected value receives a flag.
- All missing-data decisions are recorded in `docs/variable_treatment.md` and exported to `data/audit/`.

## Project layout

```text
archive/original_project/     Original notebook and source data, unchanged
data/raw/                     Immutable source files
data/intermediate/            Harmonized inputs and merge stages
data/processed/               Analysis-ready panel (not committed by default)
data/audit/                   Missingness and treatment outputs
docs/                         Data dictionary and methodology
notebooks/                    Research narrative and results
src/                          Reusable pipeline code
results/{tables,figures}/     Generated research outputs
```

## Quick start

1. Put the supplied original files in `archive/original_project/` unchanged.
2. Copy data used by the pipeline to `data/raw/` (or configure the input file in `src/config.py`).
3. Install dependencies: `python -m pip install -r requirements.txt`
4. Build an auditable panel: `python -m src.pipeline --input data/raw/manual_merge_before_processing.csv`
5. Open the notebooks in order. `06_final_results.ipynb` is the reader-facing entry point after analysis is completed.

## Workflow

1. **01 Data engineering:** ingest, harmonize, merge, and audit; no modeling.
2. **02 Exploratory analysis:** panel coverage, moments, distributions, and correlations.
3. **03 Panel econometrics:** theory-led specifications, FE/RE, diagnostics, and model comparison.
4. **04 Dynamic panel GMM:** only after the panel and identification strategy are documented.
5. **05 Machine learning extension:** out-of-sample prediction as a complement, not causal evidence.
6. **06 Final results:** concise rerunnable tables and figures for readers.

## Current results

The data-construction pipeline has been run on the original merged dataset, producing an 11,040-observation panel spanning 184 countries and 42 analysis columns. The primary panel comparison uses a common observed-data sample of 7,065 country-years from 135 countries and includes full year effects. It reports pooled OLS, random effects, fixed effects, and specification diagnostics. The technology extension is estimated on its own observed-data sample, so it does not silently narrow the main analysis.

## Public data policy

The public repository contains the reusable code, notebooks, documentation, and generated tables. The raw source data and the unchanged original student notebook remain local so that data licensing and redistribution conditions are respected.
