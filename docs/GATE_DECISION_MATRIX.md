# GATE DECISION MATRIX

The gate sequence is pre-committed so downstream model choice is not made after inspecting favorable results.

| Gate | Primary evidence | Decision rule | Downstream consequence |
|---|---|---|---|
| 0 Measurement/provenance | Source/vintage/unit/missingness flags for Y, KG, KP, H | Stop if headline variables cannot be traced to immutable source values or if hidden upstream imputation remains | Rebuild data before inference |
| 1 Cross-sectional dependence | Raw CD for description; residual CD after two-way FE/CRE is decisive | Material residual CD -> common-factor/CCE treatment required | CCE/CS-DL/CS-ARDL + Bayesian common/regional factor |
| 2 Integration | Second-generation panel unit-root tests on total log Y, KG, KP, H | Predominantly I(1) -> no long-run levels interpretation before Gate 3; I(2) -> standard VECM suspended | Determines admissible transformations |
| 3 Cointegration | Westerlund EC-style panel test with dependence-robust/bootstrap inference where feasible | Supported -> long-run relation survives; rejected -> growth/difference structural core; ambiguous -> retain both | Fixes VAR/VECM, BVAR/BVECM and structural long-run branch |

## Implementation principles

- Do not run Gates 1–3 on the legacy WB-HCI effective-labor variable for the headline design.
- Balanced cores are robustness tools only; they must be assessed for representativeness rather than accepted by an arbitrary N/T threshold.
- Panel cointegration evidence does not by itself establish Johansen rank exactly equal to one.
- Johansen/country-specific evidence is supplementary for focal countries with sufficiently long series.
- Static levels FE/CRE coefficients before Gate 3 are labelled conditional within-country associations, not long-run elasticities.
- Research Design v1.0 is empirically frozen only after Gate 3 is populated.
