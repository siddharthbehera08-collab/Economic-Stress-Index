"""
loaders.py – Raw data ingestion for the ESI pipeline.

Each loader:
  • reads a CSV from data/raw/ using a path relative to the project root
  • skips the 4-row World Bank metadata header
  • drops unnamed trailing columns
  • returns a clean DataFrame in wide format
    (Country Name | Country Code | Indicator Name | Indicator Code | 1991 … 2024)
"""

from pathlib import Path
import pandas as pd

# Project root is two levels above this file (ESI/src/loaders.py → ESI/)
_PROJECT_ROOT = Path(__file__).parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"


def _load_world_bank_csv(filename: str) -> pd.DataFrame:
    """Generic loader for World Bank-format CSVs stored in data/raw/."""
    path = _RAW_DIR / filename
    df = pd.read_csv(path, skiprows=4)
    # Drop any empty trailing columns that pandas names "Unnamed: …"
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    return df


# ── Individual loaders ────────────────────────────────────────────────────────

def load_inflation_data() -> pd.DataFrame:
    """Consumer price inflation (annual %) – World Bank FP.CPI.TOTL.ZG."""
    return _load_world_bank_csv("inflation.csv")


def load_unemployment_data() -> pd.DataFrame:
    """Unemployment rate (% of total labour force) – World Bank SL.UEM.TOTL.ZS."""
    return _load_world_bank_csv("unemployment.csv")


def load_gdp_growth_data() -> pd.DataFrame:
    """GDP growth (annual %) – World Bank NY.GDP.MKTP.KD.ZG."""
    return _load_world_bank_csv("gdp_growth.csv")


def load_food_inflation_data() -> pd.DataFrame:
    """Food inflation (annual %) – World Bank FP.CPI.FOOD.ZG proxy."""
    return _load_world_bank_csv("food_inflation.csv")


def load_interest_rate_data() -> pd.DataFrame:
    """Lending interest rate (%) – World Bank FR.INR.LEND."""
    return _load_world_bank_csv("interest_rates.csv")
