"""Non-destructive data quality checks."""
import pandas as pd

def harmonize_panel_keys(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {"Country Code":"country_code", "Country":"country_code", "country":"country_code", "iso3":"country_code", "Year":"year"}
    out = frame.rename(columns={key:value for key,value in aliases.items() if key in frame.columns}).copy()
    missing = {"country_code", "year"}.difference(out.columns)
    if missing: raise ValueError(f"Missing panel keys: {sorted(missing)}")
    out["country_code"] = out["country_code"].astype("string").str.strip().str.upper()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out

def validate_unique_panel_keys(frame: pd.DataFrame) -> None:
    if frame.duplicated(["country_code", "year"]).any():
        raise ValueError("Duplicate country-year observations require resolution before analysis.")

def missingness_report(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame({"variable":frame.columns, "missing_observations":[int(frame[c].isna().sum()) for c in frame]})
    result["missing_percent"] = (100 * result.missing_observations / len(frame)).round(2) if len(frame) else 0.0
    return result.sort_values(["missing_percent", "variable"], ascending=[False, True])

def panel_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("country_code", dropna=False).agg(first_year=("year","min"), last_year=("year","max"), observations=("year","count")).reset_index()
