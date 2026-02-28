"""
external.py
-----------
Logic for integrating external data sources, such as Brent Crude Oil prices.
"""

from pathlib import Path
import pandas as pd

def merge_oil_data(macro_df: pd.DataFrame, oil_csv_path: str | Path, year_col: str = "Year") -> pd.DataFrame:
    """
    Downsamples daily Brent Crude oil prices to annual averages and merges
    the result into the macroeconomic DataFrame.
    
    Adds:
      - oil_price_avg: Mean annual closing price.
      - oil_volatility: Standard deviation of daily closing prices.
      
    Args:
        macro_df: The baseline macroeconomic DataFrame.
        oil_csv_path: Path to the daily Brent Crude CSV.
        year_col: Name of the year column to align on.
        
    Returns:
        DataFrame with the added oil features.
    """
    # 1. Load the daily oil data
    oil_daily = pd.read_csv(oil_csv_path, parse_dates=["Date"], index_col="Date")

    if "Close" not in oil_daily.columns:
        raise KeyError(
            f"Expected a 'Close' column in {oil_csv_path}. "
            f"Found: {list(oil_daily.columns)}"
        )

    # 2. Resample daily data to annual frequency
    oil_annual = oil_daily["Close"].resample("YE").agg(
        oil_price_avg="mean",
        oil_volatility="std",
    )

    # Convert the DatetimeIndex to an integer year index
    oil_annual.index = oil_annual.index.year.astype(int)
    oil_annual.index.name = year_col

    # 3. Align the macro dataframe on the integer year index
    macro_indexed = macro_df.copy()
    macro_indexed.index = macro_indexed[year_col].astype(int)

    # 4. Perform a left join to keep all macro rows
    merged = macro_indexed.join(oil_annual, how="left")

    # 5. Handle any missing oil data (e.g., forward fill price, zero fill volatility)
    merged["oil_price_avg"] = merged["oil_price_avg"].ffill()
    merged["oil_volatility"] = merged["oil_volatility"].fillna(0.0)

    # 6. Restore the original integer index
    merged = merged.reset_index(drop=True)

    return merged
