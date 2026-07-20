# Methodological evolution

The original GDP-prediction notebook contains a substantial multi-source merge, economically meaningful feature blocks, PCA, iterative VIF diagnostics, panel fixed- and random-effects models, and machine-learning exercises. This project builds on that work by organizing it as a reproducible empirical research workflow.

## What is carried forward

- Multi-source country-year data construction and the associated variable knowledge.
- Economic feature blocks covering capital, labor, technology, and demographics.
- PCA as a transparent measurement tool for correlated indicators.
- VIF diagnostics as part of specification review.
- Panel models and machine-learning comparison as complementary analytical tools.

## Strengthened research design

The refactored workflow adds a documented data-construction stage before modeling. It keeps observed values and data-availability flags, records panel coverage and missingness, and links each variable to an explicit treatment policy.

PCA and model specifications are built on clearly stated analytical samples. Panel estimation, dynamic-panel design, and out-of-sample prediction are presented in distinct notebooks so that interpretation, diagnostics, and prediction are not conflated.

## Result

The project now provides a traceable path from authorized source data to final tables, figures, and a reader-facing research report.
