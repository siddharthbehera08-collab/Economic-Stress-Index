"""
config.py  –  Single source of truth for all paths, constants, settings.
Import this everywhere. Never hardcode paths in other files.
"""
from pathlib import Path

# ── Root Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR    = PROJECT_ROOT / "outputs"
PLOTS_DIR     = OUTPUT_DIR / "plots"
TABLES_DIR    = OUTPUT_DIR / "tables"
MODELS_DIR    = PROJECT_ROOT / "saved_models"

for _d in [RAW_DIR, PROCESSED_DIR, PLOTS_DIR, TABLES_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Study Settings ─────────────────────────────────────────────────────────────
COUNTRY      = "India"
COUNTRY_CODE = "IND"
START_YEAR   = 1991
END_YEAR     = 2024

# ── CSV Registry: key -> (filename, label, stress_direction) ──────────────────
# stress_direction: "positive" = higher value -> more stress
#                   "negative" = lower  value -> more stress  (e.g. GDP growth)
CSV_REGISTRY = {
    "inflation":      ("inflation.csv",      "Inflation Rate (%)",        "positive"),
    "food_inflation": ("food_inflation.csv",  "Food Inflation Rate (%)",   "positive"),
    "unemployment":   ("unemployment.csv",    "Unemployment Rate (%)",     "positive"),
    "interest_rate":  ("interest_rates.csv",  "Lending Interest Rate (%)", "positive"),
    "gdp_growth":     ("gdp_growth.csv",      "GDP Growth Rate (%)",       "negative"),
    "oil":            ("Oil.csv",             "Oil Rents (% of GDP)",      "positive"),
}

# Toggle indicators in/out of the composite ESI here
ESI_INDICATORS = {
    "inflation":      True,
    "food_inflation": True,
    "unemployment":   True,
    "interest_rate":  True,
    "gdp_growth":     True,
    "oil":            False,   # sparse India data – off by default
}

# ── Model Settings ─────────────────────────────────────────────────────────────
RANDOM_STATE          = 42
TEST_SIZE             = 0.2
CV_FOLDS              = 5
ANOMALY_CONTAMINATION = 0.10
N_LAG_FEATURES        = 3

# ── Colour Palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "esi":       "#1D3557",
    "inflation": "#E63946",
    "food":      "#F4A261",
    "unemp":     "#2A9D8F",
    "gdp":       "#457B9D",
    "interest":  "#6A4C93",
    "oil":       "#E9C46A",
    "highlight": "#D62828",
    "neutral":   "#6C757D",
    "low":       "#A7C957",
    "medium":    "#F4E409",
    "high":      "#F08080",
}
