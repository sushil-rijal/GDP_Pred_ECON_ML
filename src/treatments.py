"""Explicit, conservative treatment of missing stock variables."""
import pandas as pd

def interpolate_stock_with_flag(frame: pd.DataFrame, variable: str, *, enabled: bool=False) -> pd.DataFrame:
    if variable not in frame: raise KeyError(variable)
    out = frame.sort_values(["country_code", "year"]).copy()
    out[f"{variable}_original"] = out[variable]
    out[f"{variable}_interpolated_flag"] = 0
    if enabled:
        filled = out.groupby("country_code", group_keys=False)[variable].transform(lambda s: s.interpolate(method="linear", limit_area="inside"))
        changed = out[variable].isna() & filled.notna()
        out.loc[changed, variable] = filled.loc[changed]
        out.loc[changed, f"{variable}_interpolated_flag"] = 1
    return out
