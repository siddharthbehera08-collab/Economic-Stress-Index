"""
config.py
---------
Central configuration for the ESI project.
Defines data paths, output paths, constants, and settings.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"

# Ensure directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Project Constants ─────────────────────────────────────────────────────────
COUNTRY = "India"
START_YEAR = 1991
END_YEAR = 2024

# ── External Data (Brent Crude) ───────────────────────────────────────────────
BRENT_TICKER = "BZ=F"
BRENT_START_DATE = "1991-01-01"
BRENT_CSV_FILENAME = "brent_crude.csv"
