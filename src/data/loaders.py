"""
loaders.py
----------
Raw data ingestion logic for the ESI pipeline.
Reads CSV files and safely handles metadata rows.
"""

import pandas as pd
from src.config import RAW_DATA_DIR

def _load_world_bank_csv(filename: str) -> pd.DataFrame:
    """
    Generic loader for World Bank-format CSVs.
    Skips the 4 metadata rows and drops trailing unnamed columns.
    
    Args:
        filename: Name of the CSV file in the raw data directory.
        
    Returns:
        A clean pandas DataFrame.
    """
    path = RAW_DATA_DIR / filename
    
    # World Bank CSVs have 4 rows of metadata before the header
    df = pd.read_csv(path, skiprows=4)
    
    # Drop any empty trailing columns pandas reads as "Unnamed: ..."
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    
    return df

def load_inflation_data() -> pd.DataFrame:
    """Loads consumer price inflation data."""
    return _load_world_bank_csv("inflation.csv")

def load_unemployment_data() -> pd.DataFrame:
    """Loads unemployment rate data."""
    return _load_world_bank_csv("unemployment.csv")

def load_gdp_growth_data() -> pd.DataFrame:
    """Loads GDP growth data."""
    return _load_world_bank_csv("gdp_growth.csv")

def load_food_inflation_data() -> pd.DataFrame:
    """Loads food inflation data."""
    return _load_world_bank_csv("food_inflation.csv")

def load_interest_rate_data() -> pd.DataFrame:
    """Loads interest rate data."""
    return _load_world_bank_csv("interest_rates.csv")
