# EXECUTION PLAN — GDP Production-Function Extension Project

**Version:** 1.0-candidate (governance frozen; empirical specification becomes v1.0 after Gates 1–3)
**Reference point:** 2018 thesis (R0) — original production-function + inverse-target question, not the empirical foundation of the modern extension
**Destination:** probabilistic 2030/2043 forecasts + inverse-feasibility frontier, Nepal focal

## 1. Frozen design decisions

### D1. Data system

- Primary modern Y, KG, KP, KPPP and sectoral GFCF: official IMF FAD ICSD 1.0.0 system, using one internally consistent modern benchmark over the full historical window. Never splice old and new PPP benchmarks.
- Thesis-era file: replication/audit reference only. R0 is the historical research reference point; the main extension is rebuilt from the modern data system.
- PWT 11.0: `emp`, `hc`, `rgdpna`; human-capital construction, growth validation, and post-2019 outcome checks.
- Population history: fixed WDI/PWT vintage. Population projection: UN WPP 2024 probabilistic paths.
- Post-2019 capital: World Bank 2025 reproducibility-package extensions, explicitly labelled model-based/current-information inputs rather than 2019-known observations.
- Infrastructure efficiency: World Bank IEI and Nepal PIMA/budget-execution evidence as measurement/scenario inputs, not automatic causal regressors.
- Headline production equation uses **total quantities** and leaves returns to scale unrestricted.
- Cross-vintage differences are audited as revisions. The old-data pipeline is rerun for replication; the modern historical analysis is rebuilt entirely in the new benchmark.
- Depreciation recovered from the ICSD stock-flow identity is an **implied consistency schedule / prior center**, not an independently identified estimate, because the capital stocks themselves are PIM-constructed.

### D2. Human capital, missingness, and TFP

- Structural baseline A: `H = PWT emp × PWT hc`, observed only.
- Forecasting panel B: missing/future H may be predicted with origin-safe ML/Bayesian models and evaluated OOS.
- Structural robustness C: latent/multiple-imputation H only if imputation uncertainty is propagated; point-filled H is not admissible for headline elasticities.
- TFP is never an ordinary RHS production regressor. PWT TFP is used to validate/measure the latent productivity state and as a robustness forecast signal.

### D3. Sample and grouping architecture

- A1 structural panel: developing economies, 1980–2019 annual, unbalanced; >=25 observed years for Y/G/K/H; >=15 usable pre-2016 years; population >=1M baseline, >=5M sensitivity; no Y imputation.
- Structural grouping uses **initial-state development status**, not end-of-sample status, plus geographic region. Where an official historical classification is unavailable at the exact start date, use the earliest predetermined classification/initial-income information and document the rule; do not infer group membership from end-state outcomes.
- Competing hierarchical structures: global-only; development group; region; crossed region + initial group. Validate using leave-one-country-out, leave-future-out, Nepal-specific predictive performance, and variance decomposition rather than observation-level LOO/WAIC alone.
- Core-5 (BGD, IND, NPL, PAK, LKA): focal reporting/Nepal-validation group, not an imposed prior class.
- SAARC-6 may be the structural data reality; SAARC-8 remains the policy/reporting universe when data permit. Do not impute Bhutan/Afghanistan merely to preserve the label.
- SDG policy reporting: all LDCs with sufficient information regardless of structural population screen; distinguish estimated+reported, hierarchy-prediction-only, and insufficient-data countries.
- Phillips–Sul is descriptive convergence analysis only, not validation of elasticity-pooling structure.

### D4. Track separation

Structural inference, forecasting, policy reporting, and descriptive analysis have separate admissibility rules. In particular:

- observed-only H for the headline structural elasticity model;
- predicted H permitted in forecasting;
- 2019-origin exercises are methodological/historical validation, not current policy probabilities;
- current-information origin is the headline policy exercise;
- small-economy screens can apply to estimation but not automatically to treaty/policy reporting.

### D5. Forecast protocol

- E1 pre-2016 rolling pseudo-OOS: all tuning, empirical prior calibration, lag choice and combination weights. h=1,3,5 and 10 where feasible.
- E2 sacred holdout: train <=2015, evaluate 2016–2019. Run only after choices are frozen. No tuning/weights use this period.
- E3 historical stress: train <=2019; report 2020 separately and 2021–2023/25 recovery separately.
- Current-information policy origin (~2025/26): 2028/29, 2030 and 2043 policy/scenario outputs under explicitly vetted post-2019 data rules.
- Conditional structural forecasts are labelled **performance given realized future inputs**; they do not identify the structural mapping free of endogeneity.
- Metrics: RMSE, MAE, MAPE; CRPS, log score, interval coverage/width where predictive distributions exist.
- Formal DM/MCS inference is performed on the long E1 rolling forecast-error history with dependence-aware aggregation/bootstrap. E2 is a final design-based holdout and is not used for low-power country-level DM tests.

### D6. Model families

- F1 structural frequentist: pooled/RE legacy comparison -> FE -> CRE/Mundlak -> CCE/CS-DL or CS-ARDL; GMM remains diagnostic robustness; convergence-growth specification is parallel.
- F2 dynamic frequentist: VAR or VECM according to Gates 2–3.
- F3 dynamic Bayesian: BVAR/BVECM/hierarchical panel BVAR, with SV if warranted.
- F4 structural Bayesian: hierarchical production model -> regional/common productivity state -> persistent nonnegative country inefficiency; frontier-VECM unification is Tier 3 only.
- F5 ML predictive benchmark: elastic net, RF, boosting with time-aware CV; no structural interpretation.
- Literature-centered prior regime for elasticities will be documented after the literature synthesis; exact prior means are not frozen before that review. Weakly informative and R0-informed sensitivity regimes remain planned.

### D7. Inverse feasibility

Scalarized quantities:

- Δi_G: public-investment/GDP increment (policy lever);
- Δi_P: required private capital-formation condition (not a direct government lever);
- Δg_H: human-capital growth improvement;
- Δg_A: productivity-growth improvement/condition;
- Δθ: infrastructure-efficiency improvement.

For q in {0.50, 0.80, 0.90}, compute

`F(q) = {admissible lever/condition combinations: P(target attained | path, D) >= q}`

using sample-average approximation. Headline figures show the (Δi_G, Δi_P) plane at baseline H/A/θ and how that frontier shifts with efficiency, H and A. Empty F(q) is a substantive result.

### D8. Nepal target ladder

- Sixteenth Plan: preserve the official 2028/29 growth and GNI-per-capita targets in their official concepts.
- 2030: 7% real-GDP-growth benchmark and legacy GNIpc >= $2,500 target.
- Vision 2100 / ~2043: preserve official long-horizon metrics and label the exercise a scenario/projection, not an ordinary forecast.
- GDP-per-capita is a supplementary common comparison column; official GNI/GDP concepts are never silently replaced.
- GNI bridge uses `GNI = GDP + net primary income from abroad`; total remittances are never mechanically added to GDP.
- A Nepal NPI/GDP (or GNI/GDP) auxiliary model may be used. Remittance scenarios enter only through economically valid channels (e.g. compensation-of-employees component, savings/investment/external-income dynamics), not as an accounting shortcut.
- Federalism/post-2017 and PIMA/budget-execution effects are **candidate break/mechanism tests**, not automatically imposed headline coefficients.

## 2. Phased execution

### Phase 0 — Repository governance

1. Work on a research-design branch descended from `agent/econometric-refactor`; do not merge to `main` before empirical gates are complete.
2. Commit: `EXECUTION_PLAN_v1.0.md`, `TRACK_SEPARATION.md`, `RESOLVED_QUESTIONS.md`, `GATE_DECISION_MATRIX.md`, `FORECAST_PROTOCOL.md`, `PRICE_CURRENCY_CONVENTIONS.md`, `INVERSE_FEASIBILITY_PROTOCOL.md`.
3. Reuse old ETL reshaping and feature-block ideas only after provenance review. Old GDP imputation and random train/test results are superseded.

### Phase 1 — Data foundation and Gate 0

1. Trace every headline Y/G/K/H value to immutable source data and rebuild provenance flags. No Gate 1–3 testing on an upstream panel whose primary fields may already have been interpolated/imputed.
2. Ingest current ICSD, PWT 11.0 and population vintages into an A0 audit panel.
3. Produce `DATA_VINTAGES.csv` and `SAMPLE_MANIFEST.csv` with actual counts.
4. Vintage audit: SAARC + random audit countries; compare old/new levels **and within-country time variation in the log revision**. A one-year ratio is not enough to distinguish rebasing from path revision.
5. Recover/verify ICSD-implied depreciation schedules using the correct stock-flow identity; use them for accounting consistency and prior centering, not as new empirical depreciation estimates.
6. Build initial-state development/region codes and population-screen flags.
7. Build observed `emp × hc`; reconcile against the legacy WB-HCI labor measure.

### Phase 2 — R0 replication and bridge ladder

- R0a: original specification + original data.
- R0b1: same common country/year intersection, new Y/G/K + original L.
- R0b2: new Y/G/K/L on the same intersection.
- R0c: extend modern consistent data through 2019.
- R1: replace L with modern H.
- R2: modern estimator.

Report exact country intersections/exclusions. State explicitly that the sequential bridge is order-dependent and diagnostic, not a unique decomposition.

### Phase 3 — Gates 1–3 and empirical freeze

- Gate 1: raw CD for description; residual CD after two-way FE/CRE is the key common-factor diagnostic.
- Gate 2: second-generation panel unit-root testing on revised total Y/G/K/H.
- Gate 3: Westerlund error-correction-based panel cointegration with dependence-robust/bootstrap inference where feasible; balanced-core results are robustness only and must be assessed for representativeness. Country/Johansen evidence is supplementary for sufficiently long focal series rather than an automatic fallback for every country.
- Cointegration supported -> VECM/BVECM long-run branch.
- No cointegration -> growth/difference structural branch; levels relation loses long-run inferential status.
- Ambiguous -> carry both and call a conclusion robust only if it survives both.

Only after Gates 1–3 are populated does this candidate become empirical **Research Design v1.0**.

### Phase 4 — Frequentist core

FE -> CRE -> CCE/CS-DL/CS-ARDL as Gates warrant; convergence-growth robustness; restricted GMM diagnostics; VAR/VECM dynamics. Begin E1 rolling forecasts.

### Phase 5 — Bayesian core

After Gate 3 only: hierarchical CRE; BVAR/BVECM/hierarchical BVAR; structural dynamic production/frontier model; latent-H robustness. Compete region/group/crossed pooling using dependence-aware country/future-block prediction.

### Phase 6 — ML and forecasting exam

Time-aware elastic net/RF/boosting; E1 tuning and model-combination weights; run E2 once; E3 stress; dependence-aware forecast comparison. Produce the accuracy-versus-structure frontier.

### Phase 7 — Policy track

Current-information origin is the headline policy posterior. Report LDC 7% probabilities using the three-tier data-availability rule; construct feasibility frontiers; compare against LTGM-PC and the recent World Bank public-capital benchmark. Keep 2019-origin science results clearly separate.

### Phase 8 — Nepal module and writing

Model the GNI/GDP/NPI bridge correctly; test Nepal-specific institutional breaks/mechanisms rather than automatically imposing them; produce the four-target dashboard and feasibility frontiers; assemble manuscript from generated tables/figures only.

## 3. Critical-path rules

1. No Bayesian production/frontier coding before Gate 3 determines the admissible long-run/dynamic form.
2. E2 is sacred and run only after all tuning/model-choice rules are frozen; any accidental reuse is logged explicitly.
3. Structural, forecast, policy and descriptive tracks obey separate admissibility rules.
4. Every item in `RESOLVED_QUESTIONS.md` must cite its evidence; reversals are logged.
5. Nulls, failed cointegration, ML dominance and infeasible policy targets are results, not implementation failures.
6. The one-paper versus two-paper publication decision waits until the forecasting exam reveals the actual contribution.
