# Economic Stress Index (ESI)

Python-based analysis of India's key economic stress indicators from 1991 to 2024 using World Bank data. The project demonstrates clean data engineering with modular code, standardised pipelines, and professional visualisations.

## 📊 Project Overview

This project analyses five economic indicators for India and combines them into a **Composite Economic Stress Index**:

| Indicator | Source | Column |
|-----------|--------|--------|
| Consumer Price Inflation | World Bank FP.CPI.TOTL.ZG | `inflation_rate` |
| Food Inflation (proxy) | Derived from CPI | `food_inflation_rate` |
| Unemployment Rate | World Bank SL.UEM.TOTL.ZS | `unemployment_rate` |
| GDP Growth Rate | World Bank NY.GDP.MKTP.KD.ZG | `gdp_growth_rate` |
| Lending Interest Rate | World Bank FR.INR.LEND | `interest_rate` |

### Why These Indicators Matter

- **Inflation & Food Inflation**: High inflation erodes purchasing power, with food inflation disproportionately affecting lower-income households.
- **Unemployment**: A key measure of labor market distress and economic underperformance.
- **GDP Growth**: Serves as the baseline for economic health; negative growth indicates recessionary pressure.
- **Interest Rates**: High rates increase borrowing costs, slowing down investment and consumption.

Together, these metrics provide a comprehensive view of the economic pressure facing the Indian economy.

## 🗂️ Project Structure

```
ESI/
├─ data/
│  └─ raw/                     # Raw CSV data files (World Bank format)
│     ├─ inflation.csv
│     ├─ unemployment.csv
│     ├─ gdp_growth.csv
│     ├─ food_inflation.csv
│     └─ interest_rates.csv
├─ outputs/
│  ├─ plots/                   # Generated visualisations (PNG)
│  └─ tables/                  # Merged indicator table (CSV)
├─ src/
│  ├─ __init__.py
│  ├─ main.py                  # Pipeline orchestrator
│  ├─ loaders.py               # Raw data ingestion
│  ├─ transforms.py            # Data wrangling & merging
│  └─ plots.py                 # Visualisation functions
├─ .gitignore
├─ README.md
└─ requirements.txt
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/siddharthbehera08-collab/Economic-Stress-Index.git
   cd ESI
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv .venv

   # Windows:
   .venv\Scripts\activate

   # macOS / Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Weighted Composite Economic Stress Index (ESI)

The project calculates a single ESI score (0–1) for each year using an equal-weighted average of the normalized indicators.

**Formula:**
`ESI = Mean(Norm(Inflation), Norm(Food Inflation), Norm(Unemployment), Norm(Interest Rates), (1 - Norm(GDP Growth)))`

- **Stressors** (Inflation, Unemployment, etc.): Higher values increase stress.
- **Inverse Stressors** (GDP Growth): Higher values decrease stress.

**Interpretation:**
- **ESI → 1.0**: Indicates severe economic distress (e.g., 1991 crisis).
- **ESI → 0.0**: Indicates robust economic stability and growth.

The final index is saved to `outputs/tables/esi_index.csv`.

### Analytical Intelligence
The project now includes statistical methods to extract deeper insights without black-box ML:
1. **Stress Regimes**: Classification of years into Low, Moderate, and High stress based on data quantiles.
2. **Structural Breaks**: Detection of significant shifts in the stress baseline using first-difference thresholds.
3. **Sensitivity Analysis**: Verification that the ESI is robust to weight perturbations (±10%).

### Why Machine Learning is Not Used Yet
1. **Data Volume**: With only ~34 annual data points, complex ML models would severely overfit.
2. **Interpretability**: Policy analysis requires transparent, defensible logic (e.g., "stress increased because inflation spiked") rather than opaque predictions.
3. **Baselines First**: We must establish a robust statistical baseline before attempting predictive modeling.

### Generated Outputs

**`outputs/plots/`**

| File | Description |
|------|-------------|
| `india_inflation.png` | Inflation rate time series |
| `india_food_inflation.png` | Food inflation rate time series |
| `india_unemployment.png` | Unemployment rate time series |
| `india_gdp_growth.png` | GDP growth rate time series |
| `india_gdp_growth_bar.png` | GDP growth bar chart |
| `india_interest_rate.png` | Lending interest rate time series |
| `india_combined_indicators.png` | Multi-line overlay of all key rates |
| `india_composite_stress_stacked.png` | Stacked area chart of stress components |
| `india_esi_score.png` | Composite ESI score over time |
| `india_esi_high_stress.png` | ESI bar chart highlighting top 5 stress years |
| `india_esi_vs_indicators.png` | Dual-axis comparison of ESI vs Inflation/Unemployment |
| `india_esi_regimes.png` | **NEW**: ESI with color-coded stress regimes |
| `india_esi_change_points.png` | **NEW**: ESI with markers for structural breaks |

**`outputs/tables/`**

| File | Description |
|------|-------------|
| `esi_india_1991_2024.csv` | Merged table with all five indicators |
| `esi_index.csv` | Final ESI scores indexed by year |

## 📦 Module Descriptions

| File | Description |
|------|-------------|
| `esi_india_1991_2024.csv` | Merged table with all five indicators |
| `esi_index.csv` | Final ESI scores indexed by year |

## 📦 Module Descriptions

| Module | Purpose |
|--------|---------|
| `src/loaders.py` | Loads World Bank CSVs from `data/raw/`, skips 4-row metadata header, drops unnamed columns |
| `src/transforms.py` | `filter_country`, `wide_to_long`, `merge_indicators`, `normalise_indicators` and more |
| `src/plots.py` | `plot_time_series`, `plot_bar_chart`, `plot_multi_line`, `plot_composite_stress` |
| `src/main.py` | Orchestrates the 7-step pipeline end-to-end |

## 🔧 Extending the Project

**Adding a new indicator:**
1. Place the World Bank CSV in `data/raw/`
2. Add a loader in `src/loaders.py` (one-liner using `_load_world_bank_csv`)
3. Call `transforms.prepare_indicator()` in `src/main.py`
4. Add the column to `transforms.merge_indicators()`

## 📈 Data Sources

All data sourced from [World Bank Open Data](https://data.worldbank.org/).

## 📝 License

Open source – available for educational and research purposes.

## 👤 Author

**Siddharth Behera**  
GitHub: [@siddharthbehera08-collab](https://github.com/siddharthbehera08-collab)

---

**Last Updated:** February 2026
