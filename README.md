# Economic Stress Index (ESI)

A macroeconomic project designed to fetch, process, standardise, and analyze country-level economic stress. 
This pipeline calculates a composite Economic Stress Index (ESI) based on underlying macroeconomic indicators such as inflation, unemployment, GDP growth, and lending rates, and employs statistical anomaly detection, basic regression models, and an overall regime classification to evaluate economic health.

## Project Goal
To provide a clean, modular, scalable data pipeline that evaluates the economic stress trajectory of a given country over a 30+ year period. It uses an equal-weighted aggregation approach to fuse multiple datasets into a single index.

## Directory Structure
The architecture of this project focuses on clear separation of concerns:

```
ESI/
├── data/
│   ├── raw/                   # Unprocessed input data (e.g., World Bank CSVs)
│   └── processed/             # Cleaned intermediate data
├── outputs/
│   ├── plots/                 # Generated visualizations (.png)
│   └── tables/                # Analytical outputs (.csv)
├── src/
│   ├── data/
│   │   ├── fetch.py           # External API caller scripts
│   │   ├── loaders.py         # Handles ingest of local csv files
│   │   └── transforms.py      # Cleans, interpolates, merges DataFrames
│   ├── features/
│   │   ├── esi.py             # Feature logic to generate score calculations
│   │   └── external.py        # Logic combining alternate frequencies (like Oil)
│   ├── models/
│   │   ├── classification.py  # ML predicting Low/Medium/High stress class
│   │   ├── anomaly.py         # Statistical / Isolation forest anomaly finder
│   │   └── regression.py      # Checks linear consistency between models
│   ├── analysis/
│   │   └── regime_analysis.py # Creates rolling statistics and structural plots
│   ├── config.py              # Central constants and directory configurations
│   └── main.py                # Main orchestrator script
├── requirements.txt           # Project dependencies
└── README.md                  # This document
```

## How to Run
Ensure you are at the project root (`ESI/`). 

1. Create a virtual environment and install the required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or `venv\\Scripts\\activate` on Windows
   pip install -r requirements.txt
   ```

2. Generate/Download required raw datasets (if not provided):
   ```bash
   python create_datasets.py
   ```

3. Run the orchestration pipeline:
   ```bash
   python -m src.main
   ```

All generated tabular data will output to `outputs/tables/` and graphs to `outputs/plots/`.
