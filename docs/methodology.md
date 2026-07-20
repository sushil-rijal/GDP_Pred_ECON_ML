# Methodology overview

The project separates construction from inference. Notebook 01 produces an auditable country-year panel without automated imputation. Notebook 02 describes coverage, moments (including skewness and excess kurtosis), distributions, and Pearson/Spearman correlations. PCA may summarize pre-specified economically coherent blocks such as technology; it does not replace a theory-led specification.

The core analysis compares pooled OLS, random effects, and country/year fixed effects using a common observed-data sample. The specification uses full year indicators rather than a linear time trend. VIF and correlations diagnose collinearity but do not identify causal effects. Dynamic panel GMM is deferred until the endogenous regressors, lag structure, instrument count, and diagnostics are explicitly documented.

Machine learning is an out-of-sample prediction extension; feature importance and SHAP values are not causal estimates.

## Primary log-linear panel specification

The primary outcome is the natural logarithm of observed PPP GDP per capita. Private and government capital are converted to PPP dollars per person before logging; the human-capital index is also entered in natural logs. The primary specification is:

`ln GDP per capita_it = alpha + beta_1 ln private capital per capita_it + beta_2 ln government capital per capita_it + beta_3 ln human capital_it + year effects_t + error_it`.

The pooled and random-effects estimators include an intercept and full year effects. The fixed-effects estimator adds country effects, allowing each country to have its own time-invariant baseline. The intercept is a baseline term; it is not total factor productivity. PWT's constructed TFP series is retained for a separate growth-accounting extension rather than treated as a primary causal-style regressor.

The model sequence is: pooled OLS with year effects; an F test for country effects; random effects with year effects; an unbalanced-panel variance-component LM test; a conventional Hausman diagnostic; and the fixed-effects benchmark. Standard errors in the reported coefficient tables are clustered by country.

Technology enters a separately reported extension using researchers per million people. It therefore has its own observed-data sample and does not silently constrain the primary specification.

These specifications document conditional associations in the observed sample. They do not resolve simultaneity between income and capital; that question belongs to a later dynamic-panel design with an explicit instrument strategy.

## Dynamic-panel diagnostic stage

The dynamic-panel notebook estimates collapsed-instrument Difference GMM and System GMM designs in first differences. GDP per capita, private capital, and government capital are classified as endogenous. Human capital is estimated as predetermined and, separately, as endogenous. Instruments are restricted to a 2?4 lag window and collapsed; all specifications retain full year effects.

The annual diagnostic versions use 6,795 difference equations from 135 countries, with 71 or 76 instruments. Although the Hansen J test is not rejected, the AR(2) residual-correlation screen is significant in every annual version. The results are therefore treated as diagnostic evidence, not final causal estimates. A subsequent design must address serial correlation before dynamic coefficients are interpreted.

