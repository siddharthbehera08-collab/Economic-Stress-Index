"""
transforms.py  –  All cleaning, filtering, merging, and normalisation logic.
"""
import pandas as pd
import numpy as np
from typing import Optional
from src.config import COUNTRY, COUNTRY_CODE, START_YEAR, END_YEAR, N_LAG_FEATURES


def filter_country(df: pd.DataFrame, value_col: str,
                   country: str = COUNTRY, start_year: int = START_YEAR,
                   end_year: int = END_YEAR) -> pd.DataFrame:
    """Extract one country's time series, reindex to full year range, interpolate gaps."""
    mask = (df["Country"] == country) | (df["Country_Code"] == COUNTRY_CODE)
    series = df[mask][["Year", value_col]].copy()
    if series.empty:
        raise ValueError(f"Country '{country}' not found. Available: {df['Country'].unique()[:10].tolist()}")

    series = series[(series["Year"] >= start_year) & (series["Year"] <= end_year)]
    full_years = pd.DataFrame({"Year": range(start_year, end_year + 1)})
    series = full_years.merge(series, on="Year", how="left")
    series[value_col] = (series[value_col]
                         .interpolate(method="linear", limit=3, limit_direction="both")
                         .ffill().bfill())
    return series.reset_index(drop=True)


def build_lag_features(df: pd.DataFrame, value_col: str,
                        n_lags: int = N_LAG_FEATURES) -> pd.DataFrame:
    """Add lag, rolling mean/std, and YoY change features. Drops first n_lags rows."""
    df = df.copy().sort_values("Year").reset_index(drop=True)
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df[value_col].shift(i)
    df["rolling_mean_3"] = df[value_col].shift(1).rolling(3, min_periods=1).mean()
    df["rolling_std_3"]  = df[value_col].shift(1).rolling(3, min_periods=1).std().fillna(0)
    df["yoy_change"]     = df[value_col].diff()
    return df.dropna().reset_index(drop=True)


def merge_indicators(filtered_dfs: dict) -> pd.DataFrame:
    """Outer-join multiple [Year, value_col] DataFrames on Year, then interpolate."""
    merged = None
    for value_col, df in filtered_dfs.items():
        subset = df[["Year", value_col]].copy()
        merged = subset if merged is None else merged.merge(subset, on="Year", how="outer")

    merged = merged.sort_values("Year").reset_index(drop=True)
    vcols = [c for c in merged.columns if c != "Year"]
    merged[vcols] = (merged[vcols]
                     .interpolate(method="linear", limit_direction="both")
                     .ffill().bfill())
    return merged


def minmax_normalise(df: pd.DataFrame, cols: Optional[list] = None,
                     exclude: Optional[list] = None) -> pd.DataFrame:
    """Min-Max scale columns to [0, 1]. Skips constant columns."""
    df = df.copy()
    exclude = exclude or []
    if cols is None:
        cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]
    for col in cols:
        lo, hi = df[col].min(), df[col].max()
        df[col] = 0.0 if hi == lo else (df[col] - lo) / (hi - lo)
    return df


def apply_stress_direction(norm_df: pd.DataFrame,
                            positive_stressors: list,
                            negative_stressors: list) -> pd.DataFrame:
    """Invert negative stressors so higher always means more stress."""
    df = norm_df.copy()
    for col in negative_stressors:
        if col in df.columns:
            df[col] = 1.0 - df[col]
    return df
