"""Construct the analysis panel from the original merged data without imputation."""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_TO_ANALYSIS = {
    "country": "country",
    "country_code": "country_code",
    "income_group": "income_group",
    "year": "year",
    "GDP_rppp": "gdp_rppp",
    "GDP_n": "gdp_n",
    "real_gdp_ppp_output": "real_gdp_ppp_output",
    "kppp_rppp": "ppp_capital",
    "kpriv_rppp": "priv_capital",
    "kgov_rppp": "gov_capital",
    "Labor force, total_x": "labor_force",
    "Human capital index (HCI) (scale 0-1)": "human_capital_index",
    "Unemployment, total (% of total labor force) (modeled ILO estimate)_x": "unemployment_pct",
    "tfp_index": "tfp_index",
    "Researchers in R&D (per million people)": "researchers_per_million",
    "Patent applications, residents": "patents_residents",
    "Patent applications, nonresidents": "patents_nonresidents",
    "Energy use (kg of oil equivalent per capita)": "energy_use_per_capita",
    "Population, total": "population_total",
    "Age dependency ratio (% of working-age population)": "age_dependency_ratio",
    "Urban population (% of total population)": "urban_population_pct",
}

DEPENDENT_VARIABLES = ("gdp_per_capita_ppp", "log_gdp_per_capita_ppp")
GDP_RPPP_TO_DOLLARS = 1_000_000_000  # GDP_rppp is recorded in billions of PPP dollars.
FLOW_OR_RATE_VARIABLES = ("unemployment_pct", "researchers_per_million", "patents_residents", "patents_nonresidents", "energy_use_per_capita")
STOCK_OR_INDEX_VARIABLES = ("ppp_capital", "priv_capital", "gov_capital", "labor_force", "human_capital_index", "tfp_index", "population_total")


def construct_analysis_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Select documented columns, preserve all missing values, and add provenance flags.

    This function deliberately does not drop countries by an arbitrary void threshold,
    replace zeros with missing values, interpolate, median-impute, or create a target
    from other GDP measures. Those decisions must be explicit sensitivity analyses.
    """
    available = {raw_name: analysis_name for raw_name, analysis_name in RAW_TO_ANALYSIS.items() if raw_name in raw.columns}
    panel = raw.loc[:, list(available)].rename(columns=available).copy()
    required = {"country", "country_code", "year", "gdp_rppp"}
    absent = required.difference(panel.columns)
    if absent:
        raise ValueError(f"Required columns unavailable: {sorted(absent)}")
    panel["country_code"] = panel["country_code"].astype("string").str.strip().str.upper()
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    for variable in panel.select_dtypes(include="number"):
        panel[f"{variable}_observed_flag"] = panel[variable].notna().astype("int8")

    # The main outcome uses observed PPP GDP and observed population only. No
    # interpolation, target reconstruction, or zero substitution is performed.
    valid_per_capita = (
        panel["gdp_rppp"].notna()
        & panel["population_total"].notna()
        & panel["gdp_rppp"].gt(0)
        & panel["population_total"].gt(0)
    )
    panel["gdp_per_capita_ppp"] = np.where(
        valid_per_capita,
        panel["gdp_rppp"] * GDP_RPPP_TO_DOLLARS / panel["population_total"],
        np.nan,
    )
    panel["log_gdp_per_capita_ppp"] = np.log(panel["gdp_per_capita_ppp"])
    panel["gdp_per_capita_observed_flag"] = valid_per_capita.astype("int8")
    return panel.sort_values(["country_code", "year"]).reset_index(drop=True)


def treatment_log() -> pd.DataFrame:
    rows = []
    for variable in DEPENDENT_VARIABLES:
        rows.append((variable, "dependent variable", "No interpolation or imputation", "Original observed values only"))
    for variable in FLOW_OR_RATE_VARIABLES:
        rows.append((variable, "flow or rate", "No interpolation by default", "Retain missingness; justify any aggregation"))
    for variable in STOCK_OR_INDEX_VARIABLES:
        rows.append((variable, "stock or index", "No automatic interpolation", "May use internal interpolation only with original values and flag"))
    return pd.DataFrame(rows, columns=["variable", "category", "default_treatment", "audit_requirement"])
