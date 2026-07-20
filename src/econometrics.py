"""Theory-led panel estimators for the growth analysis."""
from __future__ import annotations

from math import erfc, sqrt
import numpy as np
import pandas as pd

CAPITAL_TO_DOLLARS = 1_000_000_000
BASELINE_LOG_VARIABLES = (
    "log_private_capital_per_capita",
    "log_government_capital_per_capita",
    "log_human_capital_index",
    "log_tfp_index",
)


def build_baseline_log_sample(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the observed-data sample for the baseline log-linear specification.

    Capital measures are converted from billions of PPP dollars to PPP dollars per
    person. All logarithms are natural logs and require positive observed values.
    """
    result = panel.copy()
    result["private_capital_per_capita"] = (
        result["priv_capital"] * CAPITAL_TO_DOLLARS / result["population_total"]
    )
    result["government_capital_per_capita"] = (
        result["gov_capital"] * CAPITAL_TO_DOLLARS / result["population_total"]
    )
    transforms = {
        "private_capital_per_capita": "log_private_capital_per_capita",
        "government_capital_per_capita": "log_government_capital_per_capita",
        "human_capital_index": "log_human_capital_index",
        "tfp_index": "log_tfp_index",
    }
    for source, target in transforms.items():
        result[target] = np.nan
        positive = result[source].gt(0)
        result.loc[positive, target] = np.log(result.loc[positive, source])
    required = ["log_gdp_per_capita_ppp", *BASELINE_LOG_VARIABLES, "country_code", "year"]
    sample = result.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    return sample.sort_values(["country_code", "year"]).reset_index(drop=True)


def fit_no_constant_two_way_fe(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Estimate a two-way FE log-linear model without a global intercept.

    Country dummies absorb country-specific baselines and year dummies absorb common
    time shocks. The design deliberately contains no all-ones constant column.
    Standard errors use a country-clustered sandwich covariance estimator.
    """
    y = sample["log_gdp_per_capita_ppp"].to_numpy(dtype=float)
    x_main = sample.loc[:, BASELINE_LOG_VARIABLES].astype(float)
    country_effects = pd.get_dummies(sample["country_code"], prefix="country", dtype=float)
    year_effects = pd.get_dummies(sample["year"], prefix="year", drop_first=True, dtype=float)
    design = pd.concat([x_main, country_effects, year_effects], axis=1)
    if "const" in design.columns or (design.sum(axis=0) == len(design)).any():
        raise ValueError("The no-constant design unexpectedly contains a global intercept.")
    x = design.to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coefficients
    n_obs, n_parameters = x.shape
    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((n_parameters, n_parameters))
    for _, positions in sample.groupby("country_code", sort=False).indices.items():
        x_group = x[positions, :]
        u_group = residuals[positions]
        score = x_group.T @ u_group
        meat += np.outer(score, score)
    n_clusters = sample["country_code"].nunique()
    correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - rank))
    covariance = correction * bread @ meat @ bread
    std_error = np.sqrt(np.clip(np.diag(covariance), 0, None))
    t_stat = coefficients / std_error
    p_value = np.array([erfc(abs(value) / sqrt(2)) for value in t_stat])
    result = pd.DataFrame(
        {
            "variable": design.columns,
            "coefficient": coefficients,
            "clustered_std_error": std_error,
            "t_statistic": t_stat,
            "p_value_normal": p_value,
        }
    )
    result = result[result["variable"].isin(BASELINE_LOG_VARIABLES)].reset_index(drop=True)
    total_ss = ((y - y.mean()) ** 2).sum()
    summary = {
        "observations": int(n_obs),
        "countries": int(n_clusters),
        "years": int(sample["year"].nunique()),
        "parameters": int(rank),
        "r_squared_including_fixed_effects": float(1 - residuals.dot(residuals) / total_ss),
        "residual_rmse": float(np.sqrt(residuals.dot(residuals) / (n_obs - rank))),
    }
    return result, summary
