"""
main.py – ESI pipeline orchestrator.

Run from the project root:
    python -m src.main

Pipeline steps:
  1. Load all five raw datasets
  2. Prepare (filter → wide-to-long → year range → clean) each indicator for India
  3. Merge all indicators into one combined DataFrame
  4. Save the merged table to outputs/tables/
  5. Calculate Weighted Composite Economic Stress Index (ESI)
  6. Save ESI index to outputs/tables/
  7. Generate individual indicator plots
  8. Generate combined multi-line plot
  9. Generate composite economic stress (normalised) plot and ESI score plot
  10. Generate refinement plots (High Stress Bar, ESI Comparison)
  11. Perform Analytical Intelligence (Regimes, Change-Points, Sensitivity)
  12. Print insights
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import loaders, transforms, plots

# ── Configuration ─────────────────────────────────────────────────────────────
COUNTRY = "India"
START_YEAR = 1991
END_YEAR = 2024

_PROJECT_ROOT = Path(__file__).parent.parent
_TABLES_DIR = _PROJECT_ROOT / "outputs" / "tables"
_TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prep(raw_df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Shortcut: prepare one indicator for COUNTRY over [START_YEAR, END_YEAR]."""
    return transforms.prepare_indicator(
        raw_df, COUNTRY, value_name, START_YEAR, END_YEAR
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Economic Stress Index (ESI) – India Pipeline")
    print("=" * 60)

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    print("\n[1/7] Loading raw datasets …")
    raw_inflation     = loaders.load_inflation_data()
    raw_unemployment  = loaders.load_unemployment_data()
    raw_gdp           = loaders.load_gdp_growth_data()
    raw_food          = loaders.load_food_inflation_data()
    raw_interest      = loaders.load_interest_rate_data()
    print("      ✓ All datasets loaded.")

    # ── 2. Prepare indicators for India ──────────────────────────────────────
    print(f"\n[2/7] Preparing indicators for '{COUNTRY}' ({START_YEAR}–{END_YEAR}) …")
    inflation_df     = _prep(raw_inflation,    "inflation_rate")
    unemployment_df  = _prep(raw_unemployment, "unemployment_rate")
    gdp_df           = _prep(raw_gdp,          "gdp_growth_rate")
    food_df          = _prep(raw_food,          "food_inflation_rate")
    interest_df      = _prep(raw_interest,      "interest_rate")
    print("      ✓ All indicators prepared.")

    # ── 3. Merge into one DataFrame ───────────────────────────────────────────
    print("\n[3/7] Merging indicators …")
    merged = transforms.merge_indicators({
        "inflation_rate":      inflation_df,
        "food_inflation_rate": food_df,
        "unemployment_rate":   unemployment_df,
        "gdp_growth_rate":     gdp_df,
        "interest_rate":       interest_df,
    })
    print(f"      ✓ Merged shape: {merged.shape}")

    # ── 4. Save merged table ──────────────────────────────────────────────────
    print("\n[4/7] Saving merged table …")
    table_path = _TABLES_DIR / f"esi_{COUNTRY.lower()}_{START_YEAR}_{END_YEAR}.csv"
    merged.to_csv(table_path, index=False)
    print(f"      ✓ Saved → {table_path}")

    # ── 5. Calculate Weighted ESI ─────────────────────────────────────────────
    print("\n[5/7] Calculating Weighted Composite ESI …")
    
    # Normalize all columns (0-1)
    norm_df = transforms.normalise_indicators(merged, exclude_cols=["Year"])
    
    # Calculate ESI: Mean of (Stressors + (1 - InverseStressors))
    stressors = ["inflation_rate", "food_inflation_rate", 
                 "unemployment_rate", "interest_rate"]
    inverse = ["gdp_growth_rate"]
    
    esi_scores = transforms.calculate_esi(norm_df, stressors, inverse)
    merged["esi_score"] = esi_scores
    
    # Save ESI index
    esi_path = _TABLES_DIR / "esi_index.csv"
    merged[["Year", "esi_score"]].to_csv(esi_path, index=False)
    print(f"      ✓ ESI calculated. Saved → {esi_path}")

    # ── 6. Individual indicator plots ─────────────────────────────────────────
    print("\n[6/7] Generating individual plots …")

    _individual_plots = [
        (inflation_df,    "inflation_rate",      "Inflation Rate (%)",
         "india_inflation.png"),
        (unemployment_df, "unemployment_rate",   "Unemployment Rate (%)",
         "india_unemployment.png"),
        (gdp_df,          "gdp_growth_rate",     "GDP Growth Rate (%)",
         "india_gdp_growth.png"),
        (food_df,         "food_inflation_rate", "Food Inflation Rate (%)",
         "india_food_inflation.png"),
        (interest_df,     "interest_rate",       "Lending Interest Rate (%)",
         "india_interest_rate.png"),
    ]

    for df, col, ylabel, fname in _individual_plots:
        plots.plot_time_series(
            df=df,
            x_col="Year",
            y_col=col,
            title=f"{COUNTRY} – {ylabel} ({START_YEAR}–{END_YEAR})",
            xlabel="Year",
            ylabel=ylabel,
            output_filename=fname,
        )

    # Bar chart for GDP growth
    plots.plot_bar_chart(
        df=gdp_df,
        x_col="Year",
        y_col="gdp_growth_rate",
        title=f"{COUNTRY} – GDP Growth Rate (Bar) ({START_YEAR}–{END_YEAR})",
        xlabel="Year",
        ylabel="GDP Growth Rate (%)",
        output_filename="india_gdp_growth_bar.png",
    )
    
    # ESI Score Plot
    plots.plot_time_series(
        df=merged,
        x_col="Year",
        y_col="esi_score",
        title=f"{COUNTRY} – Composite Economic Stress Index ({START_YEAR}–{END_YEAR})",
        xlabel="Year",
        ylabel="ESI Score (0 = Low Stress, 1 = High Stress)",
        output_filename="india_esi_score.png",
    )
    print("      ✓ Individual and ESI plots saved.")

    # ── 7. Combined & Composite plots ─────────────────────────────────────────
    print("\n[7/7] Generating combined visualizations & insights …")
    
    # Multi-line plot (Indicators only)
    plots.plot_multi_line(
        df=merged,
        x_col="Year",
        y_cols=["inflation_rate", "food_inflation_rate",
                "unemployment_rate", "interest_rate", "gdp_growth_rate"],
        title=f"{COUNTRY} – Key Economic Indicators ({START_YEAR}–{END_YEAR})",
        xlabel="Year",
        ylabel="Rate (%)",
        output_filename="india_combined_indicators.png",
    )

    # Composite Stress Stacked Area
    stress_contrib = norm_df[["Year"]].copy()
    for col in stressors:
        stress_contrib[col] = norm_df[col]
    for col in inverse:
        stress_contrib[col] = 1.0 - norm_df[col]
        
    plots.plot_composite_stress(
        df_norm=stress_contrib,
        x_col="Year",
        y_cols=stressors + inverse,
        title=f"{COUNTRY} – Economic Stress Components (Stacked) ({START_YEAR}–{END_YEAR})",
        output_filename="india_composite_stress_stacked.png",
    )
    
    # Refinement Plots
    # 1. Bar plot highlighting high-stress years
    plots.plot_bar_stress_years(
        df=merged,
        x_col="Year",
        y_col="esi_score",
        title=f"{COUNTRY} – ESI (High Stress Highlighted)",
        top_n=5,
        output_filename="india_esi_high_stress.png"
    )
    
    # 2. ESI vs Inflation & Unemployment comparison
    plots.plot_esi_comparison(
        df=merged,
        x_col="Year",
        esi_col="esi_score",
        comp_cols=["inflation_rate", "unemployment_rate"],
        title=f"{COUNTRY} – ESI vs Macro Indicators",
        output_filename="india_esi_vs_indicators.png"
    )
    print("      ✓ Refinement plots saved.")

    # ── 8. Analytical Intelligence ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  🧠 Analytical Intelligence")
    print("=" * 60)
    
    # A. Stress Regime Classification
    merged["regime"] = transforms.classify_stress_regimes(merged, "esi_score")
    print("\n[A] Stress Regime Classification (Years per Regime):")
    print(merged["regime"].value_counts().sort_index().to_string())
    
    plots.plot_stress_regimes(
        df=merged,
        x_col="Year",
        y_col="esi_score",
        regime_col="regime",
        title=f"{COUNTRY} – Stress Regimes ({START_YEAR}–{END_YEAR})",
        output_filename="india_esi_regimes.png"
    )
    
    # B. Change-Point Detection
    change_points = transforms.detect_change_points(merged, "esi_score")
    print("\n[B] Structural Break Detection (Change-Points):")
    if change_points:
        for year, desc in change_points:
            print(f"    • {year}: {desc}")
    else:
        print("    No significant structural breaks detected.")
        
    plots.plot_change_points(
        df=merged,
        x_col="Year",
        y_col="esi_score",
        change_points=change_points,
        title=f"{COUNTRY} – ESI Structural Breaks ({START_YEAR}–{END_YEAR})",
        output_filename="india_esi_change_points.png"
    )
    print("      ✓ Analytical plots saved.")
    
    # C. Sensitivity Analysis
    print("\n[C] Sensitivity Analysis (+10% Inflation Weight):")
    # Original: Equal weights (0.2 each for 5 components)
    # Perturbed: Infl=0.22 (+10%), others remain 0.2? Total > 1, but division handles relative magnitude.
    # Manual weighted sum for sensitivity
    
    # Stressors: Infl, Food, Unemp, Int. Inverse: GDP.
    # Perturbed: Infl * 1.1. Others * 1.0. 
    # Formula: (1.1*Infl + Food + Unemp + Int + (1-GDP)) / 5.1 (normalized by total weight)
    
    w_infl = 1.1
    sens_score = (
        (norm_df["inflation_rate"] * w_infl) +
        norm_df["food_inflation_rate"] +
        norm_df["unemployment_rate"] + 
        norm_df["interest_rate"] +
        (1.0 - norm_df["gdp_growth_rate"])
    ) / (4.0 + w_infl) # 4 other components + w_infl
    
    correlation = merged["esi_score"].corr(sens_score)
    
    top_5_orig = set(merged.nlargest(5, "esi_score")["Year"])
    top_5_sens = set(pd.DataFrame({"Year": merged["Year"], "score": sens_score})
                     .nlargest(5, "score")["Year"])
    overlap = top_5_orig.intersection(top_5_sens)
    
    print(f"    • Correlation with Original ESI: {correlation:.4f}")
    print(f"    • Top 5 Years Overlap: {len(overlap)}/5 years match {sorted(list(overlap))}")
    if correlation > 0.95:
        print("    • Interpretation: ESI is robust to weight perturbations.")
    else:
        print("    • Interpretation: ESI is sensitive to inflation weights.")
        
    # ── Insights ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📊 Summary Insights")
    print("=" * 60)
    
    top_5 = merged.nlargest(5, "esi_score")[["Year", "esi_score"]]
    print("\nTop 5 Highest Stress Years:")
    print(top_5.to_string(index=False))

    print("\nPipeline complete ✓")


if __name__ == "__main__":
    main()
