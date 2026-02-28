"""
fetch.py
--------
Responsible for downloading and saving raw data from external APIs.
"""

import yfinance as yf
import pandas as pd
from src.config import BRENT_TICKER, BRENT_START_DATE, RAW_DATA_DIR, BRENT_CSV_FILENAME

def fetch_brent_crude() -> None:
    """
    Downloads historical Brent Crude oil price data from Yahoo Finance
    and saves it to the raw data directory as a CSV file.
    
    Returns:
        None. Data is saved directly to disk.
    """
    print(f"Downloading Brent Crude data ({BRENT_TICKER}) from {BRENT_START_DATE} …")
    
    # Download data from Yahoo Finance without the progress bar
    raw = yf.download(BRENT_TICKER, start=BRENT_START_DATE, auto_adjust=True, progress=False)
    
    if raw.empty:
        raise RuntimeError(
            f"No data returned for {BRENT_TICKER}. "
            "Check your internet connection or the ticker symbol."
        )
        
    # Flatten MultiIndex columns if present (yfinance >= 0.2 returns a MultiIndex)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
        
    # Filter to the standard OHLCV columns (ignoring any others like Adj Close)
    desired_columns = ["Open", "High", "Low", "Close", "Volume"]
    available_columns = [col for col in desired_columns if col in raw.columns]
    df = raw[available_columns].copy()
    
    # Reset index to make 'Date' a standard column rather than the index
    df.index.name = "Date"
    df.reset_index(inplace=True)
    
    # Save the cleaned dataframe to the raw data directory
    output_path = RAW_DATA_DIR / BRENT_CSV_FILENAME
    df.to_csv(output_path, index=False)
    
    print(f"✓ Saved {len(df):,} rows → {output_path}")
