"""
loaders.py  –  Reads raw World Bank CSVs and returns tidy long-format DataFrames.
Each returned DataFrame has columns: Year | Country | Country_Code | <value_col>
Nothing is cleaned here – that is transforms.py's job.
"""
import pandas as pd
from pathlib import Path
from src.config import RAW_DIR, CSV_REGISTRY


def _parse_wb_csv(filepath: Path, value_col: str) -> pd.DataFrame:
    """Parses World Bank wide-format CSV (4-row metadata header) into long format."""
    if not filepath.exists():
        raise FileNotFoundError(f"Raw file not found: {filepath}")

    df = pd.read_csv(filepath, skiprows=4, encoding="utf-8-sig")
    df = df.dropna(axis=1, how="all")
    df.columns = df.columns.str.strip()

    meta_cols = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
    year_cols = [c for c in df.columns if c.isdigit()]

    long_df = df[meta_cols + year_cols].melt(
        id_vars=meta_cols, value_vars=year_cols,
        var_name="Year", value_name=value_col,
    )
    long_df["Year"]    = long_df["Year"].astype(int)
    long_df[value_col] = pd.to_numeric(long_df[value_col], errors="coerce")
    long_df = long_df.rename(columns={"Country Name": "Country", "Country Code": "Country_Code"})
    return long_df[["Year", "Country", "Country_Code", value_col]]


def load_inflation()      -> pd.DataFrame: return _parse_wb_csv(RAW_DIR / CSV_REGISTRY["inflation"][0],      "inflation_rate")
def load_food_inflation() -> pd.DataFrame: return _parse_wb_csv(RAW_DIR / CSV_REGISTRY["food_inflation"][0], "food_inflation_rate")
def load_unemployment()   -> pd.DataFrame: return _parse_wb_csv(RAW_DIR / CSV_REGISTRY["unemployment"][0],   "unemployment_rate")
def load_interest_rate()  -> pd.DataFrame: return _parse_wb_csv(RAW_DIR / CSV_REGISTRY["interest_rate"][0],  "interest_rate")
def load_gdp_growth()     -> pd.DataFrame: return _parse_wb_csv(RAW_DIR / CSV_REGISTRY["gdp_growth"][0],     "gdp_growth_rate")
def load_oil()            -> pd.DataFrame: return _parse_wb_csv(RAW_DIR / CSV_REGISTRY["oil"][0],            "oil_rents")

LOADER_MAP = {
    "inflation":      load_inflation,
    "food_inflation": load_food_inflation,
    "unemployment":   load_unemployment,
    "interest_rate":  load_interest_rate,
    "gdp_growth":     load_gdp_growth,
    "oil":            load_oil,
}

def load_all() -> dict:
    result = {}
    for key, fn in LOADER_MAP.items():
        try:
            result[key] = fn()
            print(f"  + Loaded  [{key}]  shape={result[key].shape}")
        except FileNotFoundError as e:
            print(f"  x SKIPPED [{key}]: {e}")
    return result
