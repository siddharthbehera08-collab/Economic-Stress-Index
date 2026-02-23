# Economic Stress Index (ESI) – India (1991–2024)

A data science project that constructs, analyses, and models a **Composite Economic Stress Index** for India using World Bank macroeconomic data spanning 33 years.

---

## Key Results

- **1991** is India's highest-stress year (ESI: **0.90**) — correctly aligned with the Balance of Payments crisis and IMF bailout.
- **2020** is flagged as a crisis year (ESI spike from 0.24 in 2019 to **0.53**) — confirmed as COVID-19 economic shock.
- **1998** ranks as the third-highest stress year (ESI: **0.72**) — consistent with the Asian contagion and Pokhran-II sanctions.
- The ESI is **robust to weight perturbations**: correlation > 0.99 when inflation weight is increased by 10%, with 5/5 top-stress years unchanged.
- **5-Fold CV classification accuracy**: Logistic Regression ~0.70–0.85 (varies by fold), confirming the ESI-derived labels are learnable from raw indicators.
- Linear Regression reconstructs ESI with R²≈1.0 (expected — ESI is a deterministic formula of these indicators; serves as a consistency check).

---

## Overview

The ESI is a composite index computed from five key economic indicators:

| Indicator | Direction | Rationale |
|---|---|---|
| Inflation Rate | ↑ Increases stress | Higher prices → reduced purchasing power |
| Food Inflation | ↑ Increases stress | Disproportionately impacts lower-income households |
| Unemployment Rate | ↑ Increases stress | Direct measure of labour market distress |
| Interest Rate | ↑ Increases stress | Higher borrowing costs → investment slowdown |
| GDP Growth | ↓ Decreases stress (inverted) | Growth reduces economic pressure |

All indicators are Min-Max normalised to [0, 1] and averaged with equal weights.

---

## Data Sources

All data sourced from [World Bank Open Data](https://data.worldbank.org/):

| Indicator | World Bank Series Code |
|---|---|
| Consumer Price Inflation (annual %) | FP.CPI.TOTL.ZG |
| Food Price Inflation (annual %) | FP.CPI.FOOD.ZG |
| Unemployment (% of labour force) | SL.UEM.TOTL.ZS |
| GDP Growth (annual %) | NY.GDP.MKTP.KD.ZG |
| Lending Interest Rate (%) | FR.INR.LEND |

---

## Setup

1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv`
3. Activate it: `.venv\Scripts\Activate`
4. Install dependencies: `pip install -r requirements.txt`

---

## Usage

Run the full pipeline from the project root:

```bash
python -m src.main
```

All outputs are auto-generated in `outputs/`.

---

## Project Structure

```
ESI/
├── data/
│   └── raw/                   # World Bank CSVs (5 indicators)
├── src/
│   ├── loaders.py             # Data ingestion (World Bank CSV format)
│   ├── transforms.py          # Feature engineering & ESI calculation
│   ├── plots.py               # All visualisation functions
│   ├── main.py                # Pipeline orchestrator
│   └── models/
│       ├── regression.py      # ESI reconstruction (explanatory)
│       ├── classification.py  # Stress level classification (5-fold CV)
│       └── anomaly.py         # Crisis year detection (IF + Z-score)
├── outputs/
│   ├── plots/                 # All generated charts (.png)
│   └── tables/                # ESI index, model metrics, anomalies (.csv)
├── requirements.txt
└── README.md
```

---

## Outputs

| File | Description |
|---|---|
| `outputs/tables/esi_india_1991_2024.csv` | Merged yearly indicators |
| `outputs/tables/esi_index.csv` | ESI score per year |
| `outputs/tables/anomaly_detection.csv` | Crisis year flags + Z-scores |
| `outputs/tables/model_metrics.csv` | Regression model comparison |
| `outputs/tables/classification_metrics.csv` | CV accuracy + split metrics |
| `outputs/plots/india_esi_score.png` | ESI timeline with event annotations |
| `outputs/plots/india_esi_regimes.png` | Stress regime shading |
| `outputs/plots/anomaly_detection_timeline.png` | Crisis year markers |

---

## Advanced Analytics

### Phase 2: Regression Analysis (Explanatory)

Fits Linear Regression, Random Forest, and Gradient Boosting to explain ESI from raw indicators.

- **Models**: Linear Regression, Random Forest, Gradient Boosting
- **Note**: Linear Regression achieves R²≈1.0 because ESI is a deterministic weighted mean of the same indicators — this validates formula correctness, not predictive power. Tree models provide genuine comparisons.
- **Metrics**: RMSE, R² (saved to `outputs/tables/model_metrics.csv`)

### Phase 3: Classification & Anomaly Detection

#### A. Stress Level Classification

Classifies each year as **Low / Medium / High** stress based on ESI quantiles.

- **Models**: Logistic Regression, Random Forest
- **Evaluation**: 5-fold cross-validation (primary) + stratified split for confusion matrices
- **Output**: CV accuracy ± std, confusion matrix, feature importance

#### B. Crisis Year Detection

Identifies anomalously high-stress years using two independent methods:

- **Isolation Forest**: Unsupervised; detects global outliers in ESI distribution
- **Z-Score (directional)**: Flags years where ESI > mean + 2σ (high stress only — low-stress years are not treated as crises)
- A year is flagged as a crisis only if **at least one method** identifies it as a high-stress outlier

---

### Analytical Intelligence (Phase 2A)

Applies statistical smoothing and regime-aware analysis to the ESI time series — no ML models, pure analytics.

#### Rolling Averages

3-year and 5-year centred rolling means are computed and overlaid on the ESI line chart (`india_esi_rolling_avg.png`). Rolling averages smooth out annual noise and reveal medium-term stress trends — the 5-year average in particular highlights structural economic phases such as the post-liberalisation adjustment (1991–1996) and the post-GFC stabilisation.

#### Stress Regime Classification

Each year is classified into one of three stress regimes using **tertile thresholds** (data-driven):

| Regime | Threshold |
|---|---|
| Low Stress | Bottom 33 % of ESI scores |
| Medium Stress | Middle 33 % |
| High Stress | Top 33 % |

The color-coded ESI timeline (`india_esi_regime_timeline.png`) renders each year's dot in the regime's colour, making structural shifts immediately visible.

#### Regime Transition Detection

Year-over-year regime changes are detected and printed to the console (e.g., `2008: Low Stress → High Stress`). This surfaces the speed and frequency of economic regime shifts — a critical signal for policy analysis.

#### Console Insights

Two analytical facts are printed at the end of the pipeline:

- **Longest High-Stress Streak** — the maximum number of consecutive years classified as High Stress (with the year range).
- **Most Volatile Decade** — the 10-year window with the highest ESI standard deviation, indicating where economic conditions were most unstable.

#### New Output Files

| File | Description |
|---|---|
| `outputs/plots/india_esi_rolling_avg.png` | ESI with 3-yr and 5-yr rolling averages overlaid |
| `outputs/plots/india_esi_regime_timeline.png` | Color-coded ESI timeline by stress regime |
