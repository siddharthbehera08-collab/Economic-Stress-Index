"""
main.py  –  The ONE file you run. Everything else is called from here.

Run from the project root:
    python -m src.main
"""
import pandas as pd
from src.config import (COUNTRY, START_YEAR, END_YEAR, CSV_REGISTRY,
                          ESI_INDICATORS, TABLES_DIR, PLOTS_DIR)
from src.data.loaders import load_all
from src.data.transforms import (filter_country, build_lag_features,
                                    merge_indicators, minmax_normalise,
                                    apply_stress_direction)
from src.features.esi import calculate_esi
from src.models.individual_trainer import train_individual_models
from src.models.combined_trainer import train_combined_models
from src.analysis.plotter import plot_esi_overview

_VALUE_COL_MAP = {
    "inflation":      "inflation_rate",
    "food_inflation": "food_inflation_rate",
    "unemployment":   "unemployment_rate",
    "interest_rate":  "interest_rate",
    "gdp_growth":     "gdp_growth_rate",
    "oil":            "oil_rents",
}

def main() -> None:
    print("=" * 60)
    print(f"  ESI Pipeline – {COUNTRY}  ({START_YEAR}-{END_YEAR})")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────
    print("\n[1/5] Loading raw CSVs ...")
    raw_data = load_all()

    # ── 2. Filter + feature engineering ─────────────────────────
    print(f"\n[2/5] Filtering to {COUNTRY} + building features ...")
    india_raw, india_featured = {}, {}
    for key, raw_df in raw_data.items():
        vcol = _VALUE_COL_MAP[key]
        try:
            ts = filter_country(raw_df, vcol)
        except ValueError as e:
            print(f"  SKIP [{key}]: {e}"); continue
        india_raw[vcol] = ts
        india_featured[key] = build_lag_features(ts, vcol)
        print(f"  + {key:<20} years={len(ts)}  feat_rows={len(india_featured[key])}")

    # ── 3. Individual model per CSV ──────────────────────────────
    print(f"\n[3/5] Training INDIVIDUAL models ...")
    ind_results = {k: train_individual_models(k, df) for k, df in india_featured.items()}

    # ── 4. Compute ESI ───────────────────────────────────────────
    print(f"\n[4/5] Computing composite ESI ...")
    to_merge = {vc: india_raw[vc] for k, vc in _VALUE_COL_MAP.items()
                if ESI_INDICATORS.get(k, False) and vc in india_raw}
    merged_df = merge_indicators(to_merge)

    pos_stress = [_VALUE_COL_MAP[k] for k,(_, _,d) in CSV_REGISTRY.items()
                  if d=="positive" and _VALUE_COL_MAP.get(k) in merged_df.columns]
    neg_stress = [_VALUE_COL_MAP[k] for k,(_, _,d) in CSV_REGISTRY.items()
                  if d=="negative" and _VALUE_COL_MAP.get(k) in merged_df.columns]

    ind_cols = [c for c in merged_df.columns if c != "Year"]
    norm_df  = minmax_normalise(merged_df, cols=ind_cols)
    stress_df= apply_stress_direction(norm_df, pos_stress, neg_stress)
    merged_df["esi_score"] = calculate_esi(stress_df, ind_cols)

    print(f"  ESI  min={merged_df['esi_score'].min():.4f}  "
          f"max={merged_df['esi_score'].max():.4f}  "
          f"mean={merged_df['esi_score'].mean():.4f}")

    path = TABLES_DIR / f"esi_{COUNTRY.lower()}_{START_YEAR}_{END_YEAR}.csv"
    merged_df.to_csv(path, index=False)
    print(f"  ESI dataset saved -> {path}")
    plot_esi_overview(merged_df)

    # ── 5. Combined model ────────────────────────────────────────
    print(f"\n[5/5] Training COMBINED models ...")
    enriched = train_combined_models(merged_df)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"\n  Individual Models ({len(ind_results)} indicators):")
    for key, res in ind_results.items():
        rb = res.get("regression", {}).get("best_model","—")
        cb = res.get("classification", {}).get("best_model","—")
        an = len(res.get("anomaly", {}).get("anomaly_years",[]))
        print(f"  {key:<20} Reg={rb:<22} Clf={cb:<22} Anomalies={an}")

    n_crisis = int(enriched["crisis_year"].sum()) if "crisis_year" in enriched.columns else "—"
    pk_yr = int(enriched.loc[enriched["esi_score"].idxmax(),"Year"])
    print(f"\n  Combined ESI: {len(enriched)} years | Peak={pk_yr} | Crisis years={n_crisis}")
    print(f"  Plots  -> {PLOTS_DIR}")
    print(f"  Tables -> {TABLES_DIR}")
    print(f"\n  Pipeline complete.\n")

if __name__ == "__main__":
    main()
