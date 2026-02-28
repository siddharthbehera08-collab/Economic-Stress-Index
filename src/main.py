"""
main.py
-------
Orchestrator for the Economic Stress Index (ESI) pipeline.
Imports cleanly refactored modules to download data, engineer features,
train models, and analyze regimes.

Run from the project root:
    python -m src.main
"""

from src.config import COUNTRY, START_YEAR, END_YEAR, TABLES_DIR
from src.data import fetch, loaders, transforms
from src.features import esi, external
from src.models import classification, anomaly, regression
from src.analysis import regime_analysis

import pandas as pd

def _prep(raw_df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Shortcut function to prep a single indicator by standardizing structure."""
    return transforms.prepare_indicator(raw_df, COUNTRY, value_name, START_YEAR, END_YEAR)

def main() -> None:
    """Main execution pipeline."""
    print("=" * 60)
    print(f"  Economic Stress Index (ESI) – {COUNTRY} Pipeline")
    print("=" * 60)

    # ── 1. Data Pipeline ──────────────────────────────────────────────────────
    print("\n[1/4] Data Pipeline: Fetching & Loading Data …")
    
    # Example external data fetch execution
    # fetch.fetch_brent_crude()
    
    raw_inflation = loaders.load_inflation_data()
    raw_unemployment = loaders.load_unemployment_data()
    raw_gdp = loaders.load_gdp_growth_data()
    raw_food = loaders.load_food_inflation_data()
    raw_interest = loaders.load_interest_rate_data()
    print("      ✓ Raw datasets loaded.")

    # ── 2. Feature Engineering ────────────────────────────────────────────────
    print(f"\n[2/4] Feature Engineering: Preparing & Calculating ESI ({START_YEAR}–{END_YEAR}) …")
    
    # Standardize individual datasets
    indicators = {
        "inflation_rate": _prep(raw_inflation, "inflation_rate"),
        "unemployment_rate": _prep(raw_unemployment, "unemployment_rate"),
        "gdp_growth_rate": _prep(raw_gdp, "gdp_growth_rate"),
        "food_inflation_rate": _prep(raw_food, "food_inflation_rate"),
        "interest_rate": _prep(raw_interest, "interest_rate")
    }
    
    # Merge and interpolate
    merged_df = transforms.merge_indicators(indicators)
    print(f"      ✓ Indicators merged into shape {merged_df.shape}.")
    
    # Note: Global min-max scaling across the entire timeline prior to train/test split 
    # introduces data leakage because the 'min' and 'max' values of the entire dataset 
    # (including the future test set) are used to scale the training set. 
    # However, since the ESI itself is the target variable constructed from these indicators, 
    # and we want a consistent historical index scale (0-1), we normalize first. 
    # In a pure predictive setup where we don't know the future bounds, scaler should be fit on train only.
    norm_df = transforms.normalise_indicators(merged_df, exclude_cols=["Year"])
    
    positive_stressors = ["inflation_rate", "food_inflation_rate", "unemployment_rate", "interest_rate"]
    inverse_stressors = ["gdp_growth_rate"]
    
    merged_df["esi_score"] = esi.calculate_esi(norm_df, positive_stressors, inverse_stressors)
    
    # Generate basic feature plots utilizing analysis plotting logic
    regime_analysis.plot_time_series(
        df=merged_df, x_col="Year", y_col="esi_score",
        title=f"{COUNTRY} – Composite Economic Stress Index",
        xlabel="Year", ylabel="ESI Score",
        output_filename="india_esi_score.png"
    )
    
    # Save processed table
    table_path = TABLES_DIR / f"esi_{COUNTRY.lower()}_{START_YEAR}_{END_YEAR}.csv"
    merged_df.to_csv(table_path, index=False)
    print(f"      ✓ ESI calculated and saved to {table_path}.")

    # ── 3. Modeling ───────────────────────────────────────────────────────────
    print("\n[3/4] Modeling: Classification, Anomaly, Regression …")
    
    merged_df = classification.run_classification_pipeline(merged_df)
    merged_df = anomaly.run_anomaly_detection_pipeline(merged_df)
    
    best_mod, rmse, r2 = regression.run_regression_pipeline(merged_df)
    print(f"      ✓ Regression Best Model: {best_mod} (RMSE: {rmse:.4f}, R²: {r2:.4f})")

    # ── 4. Analysis ───────────────────────────────────────────────────────────
    print("\n[4/4] Analysis: Regime & Structural Change Evaluation …")
    
    merged_df = regime_analysis.run_analysis_pipeline(merged_df)

    print("\nPipeline complete ✓")

if __name__ == "__main__":
    main()
