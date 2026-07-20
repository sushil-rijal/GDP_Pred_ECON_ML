# Global Economic Growth Analysis

A reproducible cross-country study of economic growth that combines data engineering, panel econometrics, technology indicators, and machine-learning prediction.

## Research contribution

This project turns a substantial original GDP-prediction notebook into a transparent empirical research workflow. It brings together physical capital, labor, human capital, technology, demographics, and macroeconomic information to study cross-country growth patterns.

The workflow is designed to make each stage reviewable: how the panel is constructed, which observations support each analysis, how economic feature blocks are summarized, and how econometric and predictive results compare.

## Methodological strengths

- **Transparent data construction:** country-year inputs, coverage, and variable definitions are documented and auditable.
- **Economically coherent measurement:** capital, labor, technology, and demographic indicators can be summarized using clearly defined PCA blocks.
- **Panel-data focus:** the analysis progresses from descriptive evidence to theory-led pooled, fixed-effects, random-effects, and dynamic-panel specifications.
- **Prediction as a complement:** machine learning is used for out-of-sample assessment and nonlinear patterns, alongside—not instead of—econometric analysis.
- **Reproducible communication:** reusable Python modules produce the data layer; staged notebooks explain the research decisions and results.

## Project layout

```text
data/                         Local source and generated research data
docs/                         Data dictionary, methodology, and measurement notes
notebooks/                    Staged research analysis and final results
src/                          Reusable data and analysis functions
results/{tables,figures}/     Generated research outputs
```

## Workflow

1. **Data engineering:** ingest, harmonize, construct, and audit the country-year panel.
2. **Exploratory analysis:** describe coverage, distributions, moments, and correlations.
3. **Panel econometrics:** estimate and diagnose theory-led panel specifications.
4. **Dynamic panel analysis:** document and evaluate a parsimonious GMM design.
5. **Machine-learning extension:** assess predictive performance using time-aware validation.
6. **Final results:** produce reader-facing tables, figures, and a PDF research report.

## Current status

The data-construction layer has been run against the original merged dataset, producing a 11,040-observation panel spanning 184 countries and 39 analysis columns. The next stage is to finalize the econometric specifications and generate the research tables, figures, and final PDF report.

## Data access

The public repository contains code and documentation. Source data and the archived original notebook remain local so that data licensing and redistribution conditions are respected.
