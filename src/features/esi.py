"""
esi.py  –  Computes the composite ESI score and stress-level labels.
All inputs must be normalised (0-1) and direction-adjusted (higher = more stress).
"""
import pandas as pd


def calculate_esi(stress_df: pd.DataFrame, indicator_cols: list,
                  weights: dict = None) -> pd.Series:
    """Weighted average of stress indicators. Equal weights if none provided."""
    available = [c for c in indicator_cols if c in stress_df.columns]
    missing   = [c for c in indicator_cols if c not in stress_df.columns]
    if missing:
        print(f"  ! ESI: skipping missing columns: {missing}")
    if not available:
        raise ValueError("No valid indicator columns found to compute ESI.")

    if weights is None:
        w = {col: 1.0 / len(available) for col in available}
    else:
        raw = {col: weights.get(col, 1.0) for col in available}
        total = sum(raw.values())
        w = {col: v / total for col, v in raw.items()}

    esi = sum(stress_df[col] * w[col] for col in available)
    esi.name = "esi_score"
    return esi


def label_stress_levels(esi_series: pd.Series) -> pd.Series:
    """Tertile-based Low / Medium / High labels."""
    p33, p67 = esi_series.quantile(0.33), esi_series.quantile(0.67)
    return esi_series.map(lambda v: "Low" if v <= p33 else ("Medium" if v <= p67 else "High"))
