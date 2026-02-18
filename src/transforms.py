"""
transforms.py – Data wrangling utilities for the ESI pipeline.

Key responsibilities:
  • Filter a wide-format World Bank DataFrame to a single country
  • Reshape from wide (year columns) to long (Year | value)
  • Standardise column names and data types
  • Align multiple long DataFrames on a shared Year axis
  • Merge all indicators into one combined DataFrame
  • Handle missing values
"""

from __future__ import annotations

import pandas as pd


# ── Low-level helpers ─────────────────────────────────────────────────────────

def filter_country(df: pd.DataFrame, country_name: str) -> pd.DataFrame:
    """Return rows matching *country_name* in the 'Country Name' column."""
    mask = df["Country Name"].str.strip().str.lower() == country_name.strip().lower()
    result = df[mask].copy()
    if result.empty:
        raise ValueError(
            f"Country '{country_name}' not found. "
            f"Available: {df['Country Name'].unique()[:10].tolist()} …"
        )
    return result


def _year_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that look like 4-digit years."""
    return [c for c in df.columns if c.isdigit() and len(c) == 4]


def wide_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Melt a wide-format (country row, year columns) DataFrame into long format.

    Returns a DataFrame with columns: Year (int), <value_name> (float).
    """
    year_cols = _year_columns(df)
    long = df[year_cols].melt(var_name="Year", value_name=value_name)
    long["Year"] = long["Year"].astype(int)
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    return long.reset_index(drop=True)


def select_year_range(
    df: pd.DataFrame,
    start: int,
    end: int,
    year_col: str = "Year",
) -> pd.DataFrame:
    """Keep only rows where *year_col* is between *start* and *end*."""
    return df[(df[year_col] >= start) & (df[year_col] <= end)].copy()


def clean_data_types(
    df: pd.DataFrame,
    year_col: str = "Year",
    value_col: str | None = None,
) -> pd.DataFrame:
    """Ensure Year is int and an optional value column is float."""
    df = df.copy()
    df[year_col] = df[year_col].astype(int)
    if value_col:
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df


# ── High-level pipeline helpers ───────────────────────────────────────────────

def prepare_indicator(
    raw_df: pd.DataFrame,
    country: str,
    value_name: str,
    start_year: int = 1991,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Full preparation for one indicator:
      filter → wide-to-long → year range → clean types.

    Returns a two-column DataFrame: Year | <value_name>.
    """
    country_df = filter_country(raw_df, country)
    long = wide_to_long(country_df, value_name)
    long = select_year_range(long, start_year, end_year)
    long = clean_data_types(long, value_col=value_name)
    return long.sort_values("Year").reset_index(drop=True)


def merge_indicators(indicator_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge a dict of {value_name: long_df} on the 'Year' column using an outer join,
    then forward-fill and back-fill missing values.

    Returns a single DataFrame with columns: Year, <indicator1>, <indicator2>, …
    """
    merged: pd.DataFrame | None = None
    for value_name, df in indicator_frames.items():
        df = df[["Year", value_name]].copy()
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="Year", how="outer")

    if merged is None:
        raise ValueError("No indicator frames provided.")

    merged = merged.sort_values("Year").reset_index(drop=True)

    # Fill gaps: forward-fill then back-fill so edge years are covered
    value_cols = [c for c in merged.columns if c != "Year"]
    merged[value_cols] = (
        merged[value_cols].ffill().bfill()
    )
    return merged


def normalise_indicators(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Min-max normalise all numeric columns except those in *exclude_cols*.
    Returns a new DataFrame with the same column names.
    """
    exclude_cols = exclude_cols or ["Year"]
    result = df.copy()
    for col in result.columns:
        if col in exclude_cols:
            continue
        col_min = result[col].min()
        col_max = result[col].max()
        denom = col_max - col_min
        if denom == 0:
            result[col] = 0.0
        else:
            result[col] = (result[col] - col_min) / denom
    return result


def calculate_esi(
    df_norm: pd.DataFrame,
    positive_stressors: list[str],
    inverse_stressors: list[str],
) -> pd.Series:
    """
    Calculate the composite Economic Stress Index (ESI) as the mean of weighted components.
    
    Formula:
      Sum(Stressors) + Sum(1 - InverseStressors) / TotalComponents
    
    Assumes equal weighting for simplicity, consistent with a robust index strategy.
    
    Returns a pandas Series of ESI scores.
    """
    total_components = len(positive_stressors) + len(inverse_stressors)
    if total_components == 0:
        raise ValueError("No stressors provided for ESI calculation.")

    weighted_sum = 0.0
    
    for col in positive_stressors:
        weighted_sum += df_norm[col]
        
    for col in inverse_stressors:
        weighted_sum += (1.0 - df_norm[col])
        
    return weighted_sum / total_components


def classify_stress_regimes(
    df: pd.DataFrame,
    col_name: str,
    n_bins: int = 3,
    labels: list[str] | None = None
) -> pd.Series:
    """
    Classify a continuous column into N stress regimes based on quantiles.
    e.g. Low (0-33%), Moderate (33-66%), High (66-100%).
    
    Returns a pandas Series of regime labels.
    """
    if labels is None:
        labels = ["Low Stress", "Moderate Stress", "High Stress"]
        
    try:
        return pd.qcut(df[col_name], q=n_bins, labels=labels)
    except ValueError:
        # Fallback if bin edges are not unique (rare but possible with constant data)
        return pd.cut(df[col_name], bins=n_bins, labels=labels)


def detect_change_points(
    df: pd.DataFrame,
    col_name: str,
    threshold_factor: float = 1.5,
    year_col: str = "Year"
) -> list[tuple[int, str]]:
    """
    Detect structural breaks using First-Difference Spike Detection.
    
    Logic:
      1. Calculate absolute year-over-year difference.
      2. Threshold = Mean(Diff) + threshold_factor * Std(Diff).
      3. Flag years where Diff > Threshold.
      
    Returns a list of (Year, Description) tuples.
    """
    series = df[col_name]
    diffs = series.diff().abs()
    
    mu = diffs.mean()
    sigma = diffs.std()
    threshold = mu + (threshold_factor * sigma)
    
    change_points = []
    
    for idx, val in diffs.items():
        if pd.isna(val):
            continue
            
        if val > threshold:
            year = int(df.loc[idx, year_col])
            direction = "Spike" if series[idx] > series[idx-1] else "Drop"
            desc = f"Significant {direction} ({val:.2f} delta)"
            change_points.append((year, desc))
            
    return change_points

