"""Transparent pooled, random-effects, and fixed-effects panel estimators.

The functions use only NumPy and pandas so the baseline workflow can run in a
minimal Python environment. The primary specification uses full year indicators,
not a linear time trend.
"""
from __future__ import annotations

from math import erfc, exp, lgamma, log, sqrt
from typing import Any

import numpy as np
import pandas as pd

CAPITAL_TO_DOLLARS = 1_000_000_000
PRIMARY_LOG_VARIABLES = (
    "log_private_capital_per_capita",
    "log_government_capital_per_capita",
    "log_effective_labor_per_capita",
)
TECHNOLOGY_LOG_VARIABLE = "log_researchers_per_million"


def _positive_log(frame: pd.DataFrame, source: str, target: str) -> None:
    frame[target] = np.nan
    available = frame[source].gt(0)
    frame.loc[available, target] = np.log(frame.loc[available, source])


def build_baseline_log_sample(panel: pd.DataFrame, *, include_technology: bool = False) -> pd.DataFrame:
    """Create an observed-data sample for the primary log-linear specification.

    GDP per capita is the outcome. PWT's constructed TFP series is retained for a
    later growth-accounting extension rather than used in the primary regression.
    """
    result = panel.copy()
    result["private_capital_per_capita"] = result["priv_capital"] * CAPITAL_TO_DOLLARS / result["population_total"]
    result["government_capital_per_capita"] = result["gov_capital"] * CAPITAL_TO_DOLLARS / result["population_total"]
    _positive_log(result, "private_capital_per_capita", "log_private_capital_per_capita")
    _positive_log(result, "government_capital_per_capita", "log_government_capital_per_capita")
    # HCI is labor quality. Following the original notebook, its primary use is
    # to scale labor quantity: effective_labor = labor_force * HCI.
    result["effective_labor_per_capita"] = result["effective_labor"] / result["population_total"]
    _positive_log(result, "effective_labor_per_capita", "log_effective_labor_per_capita")
    _positive_log(result, "researchers_per_million", TECHNOLOGY_LOG_VARIABLE)
    variables = list(PRIMARY_LOG_VARIABLES)
    if include_technology:
        variables.append(TECHNOLOGY_LOG_VARIABLE)
    required = ["log_gdp_per_capita_ppp", *variables, "country_code", "year"]
    return (result.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
            .sort_values(["country_code", "year"]).reset_index(drop=True))


def _normal_two_sided_p(value: float) -> float:
    return erfc(abs(value) / sqrt(2))


def _regularized_gamma_q(shape: float, value: float) -> float:
    """Upper regularized gamma for chi-square tail probabilities."""
    if value <= 0:
        return 1.0
    if value < shape + 1:
        term = 1 / shape
        total = term
        current = shape
        while abs(term) > abs(total) * 1e-14:
            current += 1
            term *= value / current
            total += term
        lower = total * exp(-value + shape * log(value) - lgamma(shape))
        return max(0.0, min(1.0, 1 - lower))
    tiny = 1e-300
    b = value + 1 - shape
    c = 1 / tiny
    d = 1 / max(b, tiny)
    fraction = d
    for iteration in range(1, 500):
        an = -iteration * (iteration - shape)
        b += 2
        d = max(an * d + b, tiny)
        c = max(b + an / c, tiny)
        d = 1 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1) < 1e-14:
            break
    return max(0.0, min(1.0, exp(-value + shape * log(value) - lgamma(shape)) * fraction))


def _chi_square_p(statistic: float, degrees_freedom: int) -> float:
    return _regularized_gamma_q(degrees_freedom / 2, statistic / 2)


def _regularized_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta for F-distribution tail probabilities."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    def fraction(xx: float, aa: float, bb: float) -> float:
        tiny = 1e-300
        qab, qap, qam = aa + bb, aa + 1, aa - 1
        c = 1.0
        d = 1 - qab * xx / qap
        d = 1 / max(d, tiny)
        value = d
        for m in range(1, 500):
            m2 = 2 * m
            step = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
            d = 1 / max(1 + step * d, tiny)
            c = max(1 + step / c, tiny)
            value *= d * c
            step = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
            d = 1 / max(1 + step * d, tiny)
            c = max(1 + step / c, tiny)
            delta = d * c
            value *= delta
            if abs(delta - 1) < 1e-14:
                break
        return value
    logarithm = a * log(x) + b * log(1 - x) - lgamma(a) - lgamma(b) + lgamma(a + b)
    if x < (a + 1) / (a + b + 2):
        return exp(logarithm) * fraction(x, a, b) / a
    return 1 - exp(logarithm) * fraction(1 - x, b, a) / b


def _f_upper_p(statistic: float, numerator_df: int, denominator_df: int) -> float:
    x = numerator_df * statistic / (numerator_df * statistic + denominator_df)
    return max(0.0, min(1.0, 1 - _regularized_beta(x, numerator_df / 2, denominator_df / 2)))


def _ols(y: np.ndarray, x: np.ndarray, clusters: np.ndarray) -> dict[str, Any]:
    coefficient, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ coefficient
    n_obs = len(y)
    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for cluster in np.unique(clusters):
        position = np.where(clusters == cluster)[0]
        score = x[position].T @ residual[position]
        meat += np.outer(score, score)
    cluster_count = len(np.unique(clusters))
    correction = (cluster_count / (cluster_count - 1)) * ((n_obs - 1) / (n_obs - rank))
    return {
        "coefficient": coefficient, "residual": residual, "rank": int(rank), "ssr": float(residual.dot(residual)),
        "cluster_covariance": correction * bread @ meat @ bread,
        "homoskedastic_covariance": residual.dot(residual) / (n_obs - rank) * bread,
    }


def _design(sample: pd.DataFrame, variables: list[str], *, include_entity_effects: bool) -> tuple[np.ndarray, list[str]]:
    pieces = [pd.Series(1.0, index=sample.index, name="constant"), sample.loc[:, variables].astype(float)]
    if include_entity_effects:
        pieces.append(pd.get_dummies(sample["country_code"], prefix="country", drop_first=True, dtype=float))
    pieces.append(pd.get_dummies(sample["year"], prefix="year", drop_first=True, dtype=float))
    design = pd.concat(pieces, axis=1)
    return design.to_numpy(float), design.columns.tolist()


def _coefficient_rows(model: str, names: list[str], fit: dict[str, Any], variables: list[str]) -> pd.DataFrame:
    standard_error = np.sqrt(np.clip(np.diag(fit["cluster_covariance"]), 0, None))
    result = pd.DataFrame({"variable": names, "coefficient": fit["coefficient"], "clustered_std_error": standard_error})
    result["t_statistic"] = result["coefficient"] / result["clustered_std_error"]
    result["p_value_normal"] = result["t_statistic"].map(_normal_two_sided_p)
    result.insert(0, "model", model)
    return result[result["variable"].isin(variables)].reset_index(drop=True)


def _random_effects(sample: pd.DataFrame, pooled: dict[str, Any], fe: dict[str, Any], pooled_x: np.ndarray, names: list[str]) -> dict[str, Any]:
    """Feasible GLS random-intercept estimator with full year indicators."""
    groups = sample["country_code"].to_numpy()
    sigma_e2 = fe["ssr"] / (len(sample) - fe["rank"])
    mean_residual = pd.Series(pooled["residual"]).groupby(groups).mean()
    group_size = pd.Series(groups).value_counts().sort_index()
    sigma_u2 = max(0.0, float((mean_residual.pow(2) - sigma_e2 / group_size).mean()))
    theta_by_group = (1 - np.sqrt(sigma_e2 / (sigma_e2 + group_size * sigma_u2))).to_dict()
    transformed_y = sample["log_gdp_per_capita_ppp"].to_numpy(float).copy()
    transformed_x = pooled_x.copy()
    for country in np.unique(groups):
        position = np.where(groups == country)[0]
        theta = theta_by_group[country]
        transformed_y[position] -= theta * transformed_y[position].mean()
        transformed_x[position] -= theta * transformed_x[position].mean(axis=0)
    fit = _ols(transformed_y, transformed_x, groups)
    fit.update({"sigma_e2": float(sigma_e2), "sigma_u2": float(sigma_u2), "mean_theta": float(np.mean(list(theta_by_group.values()))), "names": names})
    return fit


def _lm_random_effects_test(pooled: dict[str, Any], pooled_x: np.ndarray, groups: np.ndarray) -> dict[str, float | int | str]:
    """One-sided, unbalanced-panel score test for a random intercept."""
    residual = pooled["residual"]
    sigma2 = pooled["ssr"] / (len(residual) - pooled["rank"])
    indicators = pd.get_dummies(groups, dtype=float).to_numpy(float)
    residualized = indicators - pooled_x @ np.linalg.pinv(pooled_x.T @ pooled_x) @ (pooled_x.T @ indicators)
    information_matrix = indicators.T @ residualized
    residual_sum = indicators.T @ residual
    score = 0.5 * (residual_sum.dot(residual_sum) / sigma2**2 - np.trace(information_matrix) / sigma2)
    information = 0.5 * np.trace(information_matrix @ information_matrix) / sigma2**2
    statistic = max(0.0, float(score**2 / information)) if score > 0 and information > 0 else 0.0
    p_value = 0.5 * erfc(sqrt(statistic / 2)) if score > 0 else 1.0
    return {"test": "LM test: pooled OLS vs random effects", "statistic": statistic, "df": 1, "p_value": float(p_value), "decision_rule": "one-sided variance-component score test"}


def _hausman_test(fe: dict[str, Any], re: dict[str, Any], names: list[str], variables: list[str]) -> dict[str, float | int | str]:
    location = [names.index(variable) for variable in variables]
    difference = fe["coefficient"][location] - re["coefficient"][location]
    covariance_difference = fe["homoskedastic_covariance"][np.ix_(location, location)] - re["homoskedastic_covariance"][np.ix_(location, location)]
    eigenvalue, eigenvector = np.linalg.eigh((covariance_difference + covariance_difference.T) / 2)
    keep = eigenvalue > 1e-10
    if not keep.any():
        return {"test": "Hausman test: random effects vs fixed effects", "statistic": np.nan, "df": 0, "p_value": np.nan, "decision_rule": "covariance difference not positive definite; diagnostic unavailable"}
    inverse = eigenvector[:, keep] @ np.diag(1 / eigenvalue[keep]) @ eigenvector[:, keep].T
    statistic = max(0.0, float(difference @ inverse @ difference))
    df = int(keep.sum())
    return {"test": "Hausman test: random effects vs fixed effects", "statistic": statistic, "df": df, "p_value": _chi_square_p(statistic, df), "decision_rule": "conventional covariance; clustered standard errors are reported separately"}


def fit_panel_model_sequence(sample: pd.DataFrame, *, include_technology: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate pooled OLS, random effects, and two-way fixed effects."""
    variables = list(PRIMARY_LOG_VARIABLES)
    if include_technology:
        variables.append(TECHNOLOGY_LOG_VARIABLE)
    y = sample["log_gdp_per_capita_ppp"].to_numpy(float)
    groups = sample["country_code"].to_numpy()
    pooled_x, pooled_names = _design(sample, variables, include_entity_effects=False)
    fe_x, fe_names = _design(sample, variables, include_entity_effects=True)
    pooled = _ols(y, pooled_x, groups)
    fe = _ols(y, fe_x, groups)
    re = _random_effects(sample, pooled, fe, pooled_x, pooled_names)
    coefficients = pd.concat([
        _coefficient_rows("Pooled OLS + year effects", pooled_names, pooled, variables),
        _coefficient_rows("Random effects + year effects", pooled_names, re, variables),
        _coefficient_rows("Fixed effects + year effects", fe_names, fe, variables),
    ], ignore_index=True)
    entity_df = fe["rank"] - pooled["rank"]
    denominator_df = len(sample) - fe["rank"]
    f_stat = ((pooled["ssr"] - fe["ssr"]) / entity_df) / (fe["ssr"] / denominator_df)
    f_test = {"test": "F test: pooled OLS vs fixed effects", "statistic": float(f_stat), "df": f"{entity_df}, {denominator_df}", "p_value": _f_upper_p(float(f_stat), entity_df, denominator_df), "decision_rule": "tests whether country fixed effects are jointly zero"}
    diagnostics = pd.DataFrame([f_test, _lm_random_effects_test(pooled, pooled_x, groups), _hausman_test(fe, re, pooled_names, variables)])
    total_ss = float(((y - y.mean()) ** 2).sum())
    summaries = []
    for label, fit, design in [("Pooled OLS + year effects", pooled, pooled_x), ("Random effects + year effects", re, pooled_x), ("Fixed effects + year effects", fe, fe_x)]:
        original_residual = y - design @ fit["coefficient"]
        original_ssr = float(original_residual.dot(original_residual))
        summaries.append({"model": label, "observations": len(sample), "countries": int(sample["country_code"].nunique()), "years": int(sample["year"].nunique()), "parameters": fit["rank"], "overall_r_squared": 1 - original_ssr / total_ss, "overall_residual_rmse": sqrt(original_ssr / (len(sample) - fit["rank"])), "sigma_entity_squared": fit.get("sigma_u2", np.nan), "sigma_idiosyncratic_squared": fit.get("sigma_e2", np.nan)})
    return coefficients, diagnostics, pd.DataFrame(summaries)
