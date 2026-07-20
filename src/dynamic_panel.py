"""Collapsed-instrument Difference and System GMM diagnostics for the growth panel.

This implementation is intentionally parsimonious: GDP, private capital, and
public capital are endogenous; human capital is estimated once as predetermined
and once as endogenous.  Instruments are restricted to a 2--4 lag window.
"""
from __future__ import annotations

from math import erfc, sqrt
from typing import Any

import numpy as np
import pandas as pd

from src.econometrics import PRIMARY_LOG_VARIABLES, build_baseline_log_sample, _chi_square_p

CORE_NAMES = (
    "lag_log_gdp_per_capita",
    "log_private_capital_per_capita",
    "log_government_capital_per_capita",
    "log_human_capital_index",
)


def _normal_p(z: float) -> float:
    return erfc(abs(z) / sqrt(2))


def _year_vector(year: int, base_year: int, year_columns: list[int]) -> np.ndarray:
    vector = np.zeros(len(year_columns), dtype=float)
    if year != base_year:
        vector[year_columns.index(year)] = 1.0
    return vector


def _lookup(frame: pd.DataFrame, country: str, year: int, variable: str) -> float | None:
    try:
        value = frame.loc[(country, year), variable]
    except KeyError:
        return None
    return float(value) if pd.notna(value) else None


def _is_available(*values: float | None) -> bool:
    return all(value is not None and np.isfinite(value) for value in values)


def _dynamic_rows(sample: pd.DataFrame, human_treatment: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create transformed equations and a collapsed, lag-limited instrument matrix."""
    if human_treatment not in {"predetermined", "endogenous"}:
        raise ValueError("human_treatment must be 'predetermined' or 'endogenous'.")
    variables = {
        "y": "log_gdp_per_capita_ppp",
        "private": "log_private_capital_per_capita",
        "government": "log_government_capital_per_capita",
        "human": "log_human_capital_index",
    }
    lookup_table = {
        (str(row.country_code), int(row.year)): {
            "y": float(row.log_gdp_per_capita_ppp),
            "private": float(row.log_private_capital_per_capita),
            "government": float(row.log_government_capital_per_capita),
            "human": float(row.log_human_capital_index),
        }
        for row in sample.itertuples(index=False)
    }
    def value(country: str, year: int, name: str) -> float | None:
        row = lookup_table.get((str(country), int(year)))
        return row.get(name) if row is not None else None
    years = sorted(sample["year"].unique().tolist())
    base_year = years[0]
    year_columns = [year for year in years if year != base_year]
    endog_lags = [2, 3, 4]
    human_lags = [1, 2, 3] if human_treatment == "predetermined" else [2, 3, 4]
    diff_instrument_names = [f"D_{name}_lag{lag}" for name, lags in (("y", endog_lags), ("private", endog_lags), ("government", endog_lags), ("human", human_lags)) for lag in lags]
    level_instrument_names = [f"L_delta_{name}_lag1" for name in ("y", "private", "government", "human")]
    diff_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    for country, country_frame in sample.groupby("country_code", sort=False):
        available_years = sorted(country_frame["year"].unique().tolist())
        for year in available_years:
            # Difference equation at t requires t, t-1, and t-2.
            values = {name: {lag: value(country, year - lag, name) for lag in range(5)} for name in variables}
            if _is_available(values["y"][0], values["y"][1], values["y"][2], values["private"][0], values["private"][1], values["government"][0], values["government"][1], values["human"][0], values["human"][1]):
                required_lags = {"y": 2, "private": 2, "government": 2, "human": human_lags[0]}
                if all(values[name][lag] is not None for name, lag in required_lags.items()):
                    d_year = _year_vector(year, base_year, year_columns) - _year_vector(year - 1, base_year, year_columns)
                    d_instruments = []
                    for name, lags in (("y", endog_lags), ("private", endog_lags), ("government", endog_lags), ("human", human_lags)):
                        d_instruments.extend([values[name][lag] if values[name][lag] is not None else 0.0 for lag in lags])
                    diff_rows.append({
                        "country": country, "year": year,
                        "y": values["y"][0] - values["y"][1],
                        "x": [values["y"][1] - values["y"][2], values["private"][0] - values["private"][1], values["government"][0] - values["government"][1], values["human"][0] - values["human"][1], *d_year, 0.0],
                        "z_difference": d_instruments,
                    })
            # Levels equation uses lagged first differences as instruments.
            if _is_available(values["y"][0], values["y"][1], values["y"][2], values["private"][0], values["private"][1], values["private"][2], values["government"][0], values["government"][1], values["government"][2], values["human"][0], values["human"][1], values["human"][2]):
                l_year = _year_vector(year, base_year, year_columns)
                level_rows.append({
                    "country": country, "year": year,
                    "y": values["y"][0],
                    "x": [values["y"][1], values["private"][0], values["government"][0], values["human"][0], *l_year, 1.0],
                    "z_level": [values["y"][1] - values["y"][2], values["private"][1] - values["private"][2], values["government"][1] - values["government"][2], values["human"][1] - values["human"][2]],
                })
    metadata = {"year_columns": year_columns, "diff_instrument_names": diff_instrument_names, "level_instrument_names": level_instrument_names}
    return {"difference": diff_rows, "levels": level_rows}, metadata


def _two_step_gmm(y: np.ndarray, x: np.ndarray, z: np.ndarray, cluster: np.ndarray) -> dict[str, Any]:
    """Two-step GMM with country-clustered moment covariance and limited instruments."""
    n_obs = len(y)
    ztz = z.T @ z / n_obs
    first_weight = np.linalg.pinv(ztz)
    cross = x.T @ z / n_obs
    first_beta = np.linalg.pinv(cross @ first_weight @ cross.T) @ (cross @ first_weight @ (z.T @ y / n_obs))
    first_residual = y - x @ first_beta
    def moment_covariance(residual: np.ndarray) -> np.ndarray:
        value = np.zeros((z.shape[1], z.shape[1]))
        for country in np.unique(cluster):
            position = np.where(cluster == country)[0]
            moment = z[position].T @ residual[position]
            value += np.outer(moment, moment)
        return value / n_obs
    robust_moments = moment_covariance(first_residual)
    weight = np.linalg.pinv(robust_moments)
    beta = np.linalg.pinv(cross @ weight @ cross.T) @ (cross @ weight @ (z.T @ y / n_obs))
    residual = y - x @ beta
    robust_moments = moment_covariance(residual)
    bread = np.linalg.pinv(cross @ weight @ cross.T)
    middle = cross @ weight @ robust_moments @ weight @ cross.T
    covariance = bread @ middle @ bread / n_obs
    country_count = len(np.unique(cluster))
    covariance *= country_count / (country_count - 1)
    mean_moment = z.T @ residual / n_obs
    j_statistic = float(n_obs * mean_moment @ weight @ mean_moment)
    return {"beta": beta, "covariance": covariance, "residual": residual, "j_statistic": j_statistic, "instrument_count": z.shape[1], "observations": n_obs, "countries": country_count}


def _ar_screen(rows: list[dict[str, Any]], residual: np.ndarray, lag: int) -> tuple[float, float, int]:
    values = pd.DataFrame({"country": [row["country"] for row in rows], "year": [row["year"] for row in rows], "residual": residual})
    # Exact country-year matching is clearer than relying on row order in an unbalanced panel.
    lookup = {(row.country, int(row.year)): float(row.residual) for row in values.itertuples(index=False)}
    products = [(error, lookup[(country, year - lag)]) for (country, year), error in lookup.items() if (country, year - lag) in lookup]
    if len(products) < 3:
        return np.nan, np.nan, len(products)
    current = np.array([pair[0] for pair in products])
    previous = np.array([pair[1] for pair in products])
    correlation = float(np.corrcoef(current, previous)[0, 1])
    z_value = correlation * sqrt(len(products))
    return correlation, _normal_p(z_value), len(products)


def _estimate_difference(rows: dict[str, Any], metadata: dict[str, Any], human_treatment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    difference = rows["difference"]
    y = np.array([row["y"] for row in difference], dtype=float)
    x = np.array([row["x"] for row in difference], dtype=float)[:, :-1]
    z = np.array([row["z_difference"] + [*row["x"][4:-1]] for row in difference], dtype=float)
    cluster = np.array([row["country"] for row in difference])
    fit = _two_step_gmm(y, x, z, cluster)
    names = [*CORE_NAMES, *[f"year_{year}" for year in metadata["year_columns"]]]
    standard_error = np.sqrt(np.clip(np.diag(fit["covariance"]), 0, None))
    table = pd.DataFrame({"model": f"Difference GMM ({human_treatment})", "variable": names, "coefficient": fit["beta"], "clustered_std_error": standard_error})
    table["z_statistic"] = table["coefficient"] / table["clustered_std_error"]
    table["p_value_normal"] = table["z_statistic"].map(_normal_p)
    ar1, ar1_p, ar1_n = _ar_screen(difference, fit["residual"], 1)
    ar2, ar2_p, ar2_n = _ar_screen(difference, fit["residual"], 2)
    diagnostics = pd.DataFrame([
        {"model": f"Difference GMM ({human_treatment})", "test": "Hansen J overidentification test", "statistic": fit["j_statistic"], "df": fit["instrument_count"] - len(names), "p_value": _chi_square_p(fit["j_statistic"], fit["instrument_count"] - len(names)), "notes": "Two-step cluster-robust moment covariance; interpret with instrument count."},
        {"model": f"Difference GMM ({human_treatment})", "test": "AR(1) residual-correlation screen", "statistic": ar1, "df": ar1_n, "p_value": ar1_p, "notes": "Normal approximation screen using exact country-year residual pairs."},
        {"model": f"Difference GMM ({human_treatment})", "test": "AR(2) residual-correlation screen", "statistic": ar2, "df": ar2_n, "p_value": ar2_p, "notes": "No AR(2) is the key differenced-error diagnostic."},
        {"model": f"Difference GMM ({human_treatment})", "test": "Instrument count", "statistic": fit["instrument_count"], "df": fit["countries"], "p_value": np.nan, "notes": "Instrument count should remain below the number of countries."},
    ])
    return table[table["variable"].isin(CORE_NAMES)].reset_index(drop=True), diagnostics


def _estimate_system(rows: dict[str, Any], metadata: dict[str, Any], human_treatment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    difference, levels = rows["difference"], rows["levels"]
    n_d, n_l = len(difference), len(levels)
    year_count = len(metadata["year_columns"])
    d_count = len(metadata["diff_instrument_names"])
    l_count = len(metadata["level_instrument_names"])
    diff_y = np.array([row["y"] for row in difference], dtype=float)
    level_y = np.array([row["y"] for row in levels], dtype=float)
    diff_x = np.array([row["x"] for row in difference], dtype=float)
    level_x = np.array([row["x"] for row in levels], dtype=float)
    x = np.vstack([diff_x, level_x])
    y = np.concatenate([diff_y, level_y])
    z_diff = np.zeros((n_d, d_count + l_count + year_count + 1))
    z_diff[:, :d_count] = np.array([row["z_difference"] for row in difference], dtype=float)
    z_diff[:, d_count + l_count:d_count + l_count + year_count] = diff_x[:, 4:-1]
    z_level = np.zeros((n_l, d_count + l_count + year_count + 1))
    z_level[:, d_count:d_count + l_count] = np.array([row["z_level"] for row in levels], dtype=float)
    z_level[:, d_count + l_count:d_count + l_count + year_count] = level_x[:, 4:-1]
    z_level[:, -1] = 1.0
    z = np.vstack([z_diff, z_level])
    cluster = np.array([row["country"] for row in difference] + [row["country"] for row in levels])
    fit = _two_step_gmm(y, x, z, cluster)
    names = [*CORE_NAMES, *[f"year_{year}" for year in metadata["year_columns"]], "constant"]
    standard_error = np.sqrt(np.clip(np.diag(fit["covariance"]), 0, None))
    table = pd.DataFrame({"model": f"System GMM ({human_treatment})", "variable": names, "coefficient": fit["beta"], "clustered_std_error": standard_error})
    table["z_statistic"] = table["coefficient"] / table["clustered_std_error"]
    table["p_value_normal"] = table["z_statistic"].map(_normal_p)
    # Serial correlation is assessed on the differenced-equation subset.
    diff_residual = fit["residual"][:n_d]
    ar1, ar1_p, ar1_n = _ar_screen(difference, diff_residual, 1)
    ar2, ar2_p, ar2_n = _ar_screen(difference, diff_residual, 2)
    diagnostics = pd.DataFrame([
        {"model": f"System GMM ({human_treatment})", "test": "Hansen J overidentification test", "statistic": fit["j_statistic"], "df": fit["instrument_count"] - len(names), "p_value": _chi_square_p(fit["j_statistic"], fit["instrument_count"] - len(names)), "notes": "Two-step cluster-robust moment covariance; level moments require mean-stationarity assumptions."},
        {"model": f"System GMM ({human_treatment})", "test": "AR(1) residual-correlation screen", "statistic": ar1, "df": ar1_n, "p_value": ar1_p, "notes": "Normal approximation screen using differenced-equation residuals."},
        {"model": f"System GMM ({human_treatment})", "test": "AR(2) residual-correlation screen", "statistic": ar2, "df": ar2_n, "p_value": ar2_p, "notes": "No AR(2) is the key differenced-error diagnostic."},
        {"model": f"System GMM ({human_treatment})", "test": "Instrument count", "statistic": fit["instrument_count"], "df": fit["countries"], "p_value": np.nan, "notes": "Instrument count should remain below the number of countries."},
    ])
    return table[table["variable"].isin(CORE_NAMES)].reset_index(drop=True), diagnostics


def run_gmm_versions(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Difference and System GMM with two human-capital classifications."""
    sample = build_baseline_log_sample(panel)
    result_tables, diagnostic_tables, samples = [], [], []
    for treatment in ("predetermined", "endogenous"):
        rows, metadata = _dynamic_rows(sample, treatment)
        difference_table, difference_diagnostics = _estimate_difference(rows, metadata, treatment)
        system_table, system_diagnostics = _estimate_system(rows, metadata, treatment)
        result_tables.extend([difference_table, system_table])
        diagnostic_tables.extend([difference_diagnostics, system_diagnostics])
        samples.append({"human_capital_treatment": treatment, "common_observed_panel_rows": len(sample), "difference_equations": len(rows["difference"]), "level_equations": len(rows["levels"]), "countries_in_difference": len({row["country"] for row in rows["difference"]}), "year_effects": len(metadata["year_columns"]), "difference_lag_window": "2-4 for endogenous variables; 1-3 for predetermined human capital", "collapsed_instruments": True})
    return pd.concat(result_tables, ignore_index=True), pd.concat(diagnostic_tables, ignore_index=True), pd.DataFrame(samples)
