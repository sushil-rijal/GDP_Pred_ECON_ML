import pandas as pd

def distribution_moments(frame: pd.DataFrame, variables: list[str]|None=None) -> pd.DataFrame:
    data = frame.select_dtypes(include="number") if variables is None else frame[variables]
    return pd.DataFrame({"n":data.count(), "mean":data.mean(), "sd":data.std(), "min":data.min(), "q1":data.quantile(.25), "median":data.median(), "q3":data.quantile(.75), "max":data.max(), "skewness":data.skew(), "kurtosis_excess":data.kurt()}).reset_index(names="variable").round(4)

def correlation_matrix(frame: pd.DataFrame, variables: list[str]|None=None, method: str="pearson") -> pd.DataFrame:
    data = frame.select_dtypes(include="number") if variables is None else frame[variables]
    return data.corr(method=method)
