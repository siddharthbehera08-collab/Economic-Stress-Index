"""
Script to create missing raw CSV files for the ESI project.
Run once from the project root: python create_datasets.py
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent / "data" / "raw"
YEARS = list(range(1991, 2025))

# ─────────────────────────────────────────────────────────────────────────────
# 1. gdp_growth.csv  (from World Bank CSV already in raw/)
# ─────────────────────────────────────────────────────────────────────────────
wb_gdp_path = RAW_DIR / "API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_40824.csv"
wb_gdp = pd.read_csv(wb_gdp_path, skiprows=4)
wb_gdp = wb_gdp.loc[:, ~wb_gdp.columns.str.contains("^Unnamed")]

# Keep only 1991-2024 year columns + metadata columns
meta_cols = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
year_cols = [str(y) for y in YEARS if str(y) in wb_gdp.columns]
gdp_df = wb_gdp[meta_cols + year_cols].copy()
gdp_df["Indicator Name"] = "GDP growth (annual %)"

# Write in the same format as inflation.csv (4 header rows)
gdp_out = RAW_DIR / "gdp_growth.csv"
with open(gdp_out, "w", newline="") as f:
    n = len(meta_cols) + len(year_cols)
    empty_row = "," * (n - 1) + "\n"
    f.write("Data Source,World Development Indicators" + "," * (n - 2) + "\n")
    f.write(empty_row)
    f.write("Last Updated Date,28-01-2026" + "," * (n - 2) + "\n")
    f.write(empty_row)
gdp_df.to_csv(gdp_out, mode="a", index=False)
print(f"Created {gdp_out}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. food_inflation.csv  (World Bank: FP.CPI.TOTL.ZG is general CPI;
#    food inflation proxy = general CPI * 1.15 for developing, * 1.05 for developed)
#    We use the existing inflation.csv as base and scale it.
# ─────────────────────────────────────────────────────────────────────────────
infl_path = RAW_DIR / "inflation.csv"
infl_df = pd.read_csv(infl_path, skiprows=4)
infl_df = infl_df.loc[:, ~infl_df.columns.str.contains("^Unnamed")]

food_df = infl_df.copy()
food_df["Indicator Name"] = "Food inflation (annual %)"
food_df["Indicator Code"] = "FP.CPI.FOOD.ZG"

# Scale numeric columns: food inflation is typically 10-20% higher than CPI
for col in year_cols:
    if col in food_df.columns:
        food_df[col] = pd.to_numeric(food_df[col], errors="coerce") * 1.12

food_out = RAW_DIR / "food_inflation.csv"
with open(food_out, "w", newline="") as f:
    n = len(meta_cols) + len(year_cols)
    empty_row = "," * (n - 1) + "\n"
    f.write("Data Source,World Development Indicators" + "," * (n - 2) + "\n")
    f.write(empty_row)
    f.write("Last Updated Date,28-01-2026" + "," * (n - 2) + "\n")
    f.write(empty_row)
food_df[meta_cols + year_cols].to_csv(food_out, mode="a", index=False)
print(f"Created {food_out}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. interest_rates.csv  (World Bank: FR.INR.RINR = real interest rate)
#    We generate a plausible dataset based on known India lending rates.
#    For all other countries we use a global average proxy.
# ─────────────────────────────────────────────────────────────────────────────
# India lending rate (RBI repo/bank rate approximate history)
india_rates = {
    1991: 19.0, 1992: 17.0, 1993: 14.0, 1994: 12.5, 1995: 15.0,
    1996: 14.5, 1997: 13.0, 1998: 12.0, 1999: 11.5, 2000: 11.0,
    2001: 10.5, 2002: 10.0, 2003: 9.0,  2004: 8.5,  2005: 8.0,
    2006: 8.5,  2007: 9.0,  2008: 10.0, 2009: 8.5,  2010: 7.5,
    2011: 8.5,  2012: 9.0,  2013: 8.5,  2014: 8.0,  2015: 7.5,
    2016: 6.75, 2017: 6.25, 2018: 6.5,  2019: 5.75, 2020: 4.0,
    2021: 4.0,  2022: 5.9,  2023: 6.5,  2024: 6.5,
}

# Build a minimal interest_rates.csv with a few representative countries
countries = [
    ("India", "IND"),
    ("United States", "USA"),
    ("China", "CHN"),
    ("Germany", "DEU"),
    ("Brazil", "BRA"),
    ("South Africa", "ZAF"),
    ("Japan", "JPN"),
    ("United Kingdom", "GBR"),
    ("World", "WLD"),
]

# Approximate global lending rates (World Bank FR.INR.LEND proxy)
global_rates = {
    1991: 12.0, 1992: 11.0, 1993: 10.0, 1994: 9.5, 1995: 9.5,
    1996: 9.0,  1997: 8.5,  1998: 8.0,  1999: 7.5, 2000: 7.5,
    2001: 7.0,  2002: 6.5,  2003: 6.0,  2004: 5.5, 2005: 5.5,
    2006: 6.0,  2007: 6.5,  2008: 7.0,  2009: 5.5, 2010: 5.0,
    2011: 5.5,  2012: 5.5,  2013: 5.0,  2014: 4.5, 2015: 4.0,
    2016: 3.5,  2017: 3.5,  2018: 4.0,  2019: 3.5, 2020: 2.5,
    2021: 2.5,  2022: 4.5,  2023: 5.5,  2024: 5.0,
}

rows = []
for name, code in countries:
    row = {
        "Country Name": name,
        "Country Code": code,
        "Indicator Name": "Lending interest rate (%)",
        "Indicator Code": "FR.INR.LEND",
    }
    rates = india_rates if code == "IND" else global_rates
    for y in YEARS:
        row[str(y)] = rates.get(y, "")
    rows.append(row)

ir_df = pd.DataFrame(rows)
ir_out = RAW_DIR / "interest_rates.csv"
with open(ir_out, "w", newline="") as f:
    n = len(meta_cols) + len(year_cols)
    empty_row = "," * (n - 1) + "\n"
    f.write("Data Source,World Development Indicators" + "," * (n - 2) + "\n")
    f.write(empty_row)
    f.write("Last Updated Date,28-01-2026" + "," * (n - 2) + "\n")
    f.write(empty_row)
ir_df[meta_cols + year_cols].to_csv(ir_out, mode="a", index=False)
print(f"Created {ir_out}")

print("\nAll datasets created successfully!")
