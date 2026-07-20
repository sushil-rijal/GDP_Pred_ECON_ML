# Methodology overview

The project separates construction from inference. Notebook 01 produces an auditable country-year panel without automated imputation. Notebook 02 describes coverage, moments (including skewness and excess kurtosis), distributions, and Pearson/Spearman correlations. PCA may summarize pre-specified economically coherent blocks such as technology; it does not replace a theory-led specification.

The core analysis compares pooled OLS, country/time fixed effects, and random effects where appropriate. VIF and correlations diagnose collinearity but do not identify causal effects. Dynamic panel GMM is deferred until the endogenous regressors, lag structure, instrument count, and diagnostics are explicitly documented.

Machine learning is an out-of-sample prediction extension; feature importance and SHAP values are not causal estimates.

## Baseline log-linear panel model

The baseline outcome is the natural logarithm of observed PPP GDP per capita. Private and government capital are converted to PPP dollars per person before logging; human-capital and TFP indices are also entered in natural logs. The baseline model has no global intercept. Country effects absorb country-specific baselines and year effects absorb common time shocks. Standard errors are clustered by country.

This specification is an association model. It documents the empirical relationships in the observed sample and provides a benchmark for later dynamic-panel analysis of simultaneity and persistence.
