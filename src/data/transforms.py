"""
transforms.py
-------------
Data wrangling and preprocessing utilities for the ESI pipeline.
Handles filtering, reshaping, merging, and normalizing raw data.
"""

import pandas as pd

def filter_country(df: pd.DataFrame, country_name: str) -> pd.DataFrame:
    """
    Filters a DataFrame to only include rows for a specific country.
    
    Args:
        df: Raw World Bank DataFrame.
        country_name: Name of the country to filter by.
        
    Returns:
        DataFrame containing only the specified country.
    """
    mask = df["Country Name"].str.strip().str.lower() == country_name.strip().lower()
    result = df[mask].copy()
    
    if result.empty:
        raise ValueError(
            f"Country '{country_name}' not found. "
            f"Available: {df['Country Name'].unique()[:10].tolist()} …"
        )
        
    return result

def _year_columns(df: pd.DataFrame) -> list[str]:
    """Helper to find all 4-digit year columns."""
    return [c for c in df.columns if c.isdigit() and len(c) == 4]

def wide_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Converts a wide-format DataFrame (years as columns) into long format.
    
    Args:
        df: Wide-format DataFrame.
        value_name: Name to assign to the new value column.
        
    Returns:
        Long-format DataFrame with 'Year' and `value_name` columns.
    """
    year_cols = _year_columns(df)
    
    # Melt reshapes the dataframe from wide to long
    long_df = df[year_cols].melt(var_name="Year", value_name=value_name)
    
    # Ensure proper data types
    long_df["Year"] = long_df["Year"].astype(int)
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    
    return long_df.reset_index(drop=True)

def select_year_range(df: pd.DataFrame, start: int, end: int, year_col: str = "Year") -> pd.DataFrame:
    """Filters the DataFrame to a specific range of years."""
    return df[(df[year_col] >= start) & (df[year_col] <= end)].copy()

def clean_data_types(df: pd.DataFrame, year_col: str = "Year", value_col: str | None = None) -> pd.DataFrame:
    """Ensures year column is integer and value column is float."""
    df = df.copy()
    df[year_col] = df[year_col].astype(int)
    
    if value_col:
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        
    return df

def prepare_indicator(raw_df: pd.DataFrame, country: str, value_name: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Full preparation pipeline for a single indicator:
    Filter by country -> Reshape to long format -> Filter by year range -> Clean types.
    
    Args:
        raw_df: The raw indicator DataFrame.
        country: Target country name.
        value_name: Name for the indicator value column.
        start_year: Start year (inclusive).
        end_year: End year (inclusive).
        
    Returns:
        Prepared two-column DataFrame (Year, value).
    """
    country_df = filter_country(raw_df, country)
    long_df = wide_to_long(country_df, value_name)
    ranged_df = select_year_range(long_df, start_year, end_year)
    clean_df = clean_data_types(ranged_df, value_col=value_name)
    
    return clean_df.sort_values("Year").reset_index(drop=True)

def merge_indicators(indicator_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merges multiple indicator DataFrames on the 'Year' column using an outer join.
    Forward-fills and back-fills any missing values.
    
    Args:
        indicator_frames: Dict mapping indicator names to their DataFrames.
        
    Returns:
        Merged DataFrame with 'Year' and all indicator columns.
    """
    merged_df = None
    
    # Outer join all dataframes iteratively
    for value_name, df in indicator_frames.items():
        subset = df[["Year", value_name]].copy()
        
        if merged_df is None:
            merged_df = subset
        else:
            merged_df = pd.merge(merged_df, subset, on="Year", how="outer")

    if merged_df is None:
        raise ValueError("No indicator frames provided to merge.")

    merged_df = merged_df.sort_values("Year").reset_index(drop=True)

    # Clean up missing data (forward fill, then backward fill)
    value_cols = [c for c in merged_df.columns if c != "Year"]
    merged_df[value_cols] = merged_df[value_cols].ffill().bfill()
    
    return merged_df

def normalise_indicators(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Min-max normalizes all numeric columns to a 0-1 scale.
    
    Args:
        df: Input DataFrame.
        exclude_cols: List of columns to skip (e.g., 'Year').
        
    Returns:
        Normalized DataFrame.
    """
    exclude_cols = exclude_cols or ["Year"]
    result = df.copy()
    
    for col in result.columns:
        if col in exclude_cols:
            continue
            
        col_min = result[col].min()
        col_max = result[col].max()
        denom = col_max - col_min
        
        # Avoid division by zero if column values are constant
        if denom == 0:
            result[col] = 0.0
        else:
            result[col] = (result[col] - col_min) / denom
            
    return result
