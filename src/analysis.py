"""
analysis.py – Phase 2A: ESI Analytical Intelligence

Adds rolling statistics, stress-regime classification, transition detection,
and targeted visualisations on top of the already-computed ESI scores.

Run via:
    python -m src.main

Public API (called from main.py):
    run_analysis_pipeline(df)  →  enriched DataFrame

All plots are saved to  outputs/plots/.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import pandas as pd

# ── Output directory ──────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
_PLOTS_DIR = _PROJECT_ROOT / "outputs" / "plots"
_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour constants (consistent with plots.py palette) ───────────────────────
_COL_ESI       = "#1D3557"   # dark blue
_COL_3YR       = "#E63946"   # red
_COL_5YR       = "#F4A261"   # orange
_COL_LOW       = "#A7C957"   # green
_COL_MED       = "#F4E409"   # yellow
_COL_HIGH      = "#F08080"   # light coral

_REGIME_COLORS = {
    "Low Stress":    _COL_LOW,
    "Medium Stress": _COL_MED,
    "High Stress":   _COL_HIGH,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rolling Averages
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_averages(
    df: pd.DataFrame,
    col: str = "esi_score",
    windows: Tuple[int, int] = (3, 5),
) -> pd.DataFrame:
    """
    Compute rolling mean windows and append them as new columns.

    New columns added:
        esi_3yr_avg  –  3-year centred rolling mean
        esi_5yr_avg  –  5-year centred rolling mean

    Uses  min_periods=1  so edge years are not dropped.
    """
    df = df.copy()
    for w in windows:
        col_name = f"esi_{w}yr_avg"
        df[col_name] = (
            df[col]
            .rolling(window=w, center=True, min_periods=1)
            .mean()
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stress Regime Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_regimes(
    df: pd.DataFrame,
    col: str = "esi_score",
    labels: List[str] | None = None,
) -> pd.DataFrame:
    """
    Classify each year into a stress regime based on tertiles (bottom/middle/top 33%).

    New column added:  analysis_regime
        Low Stress    → bottom 33 %
        Medium Stress → middle 33 %
        High Stress   → top    33 %
    """
    if labels is None:
        labels = ["Low Stress", "Medium Stress", "High Stress"]

    df = df.copy()
    try:
        df["analysis_regime"] = pd.qcut(df[col], q=3, labels=labels)
    except ValueError:
        # Fallback if bin edges are non-unique (constant data edge-case)
        df["analysis_regime"] = pd.cut(df[col], bins=3, labels=labels)

    df["analysis_regime"] = df["analysis_regime"].astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Regime Transition Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_regime_transitions(
    df: pd.DataFrame,
    regime_col: str = "analysis_regime",
    year_col: str = "Year",
) -> List[Tuple[int, str, str]]:
    """
    Detect year-over-year stress regime changes.

    Returns a list of (year, from_regime, to_regime) tuples representing the
    year in which the transition *to* the new regime occurred.
    """
    transitions: List[Tuple[int, str, str]] = []
    regimes = df[regime_col].tolist()
    years   = df[year_col].tolist()

    for i in range(1, len(regimes)):
        prev, curr = regimes[i - 1], regimes[i]
        if prev != curr:
            transitions.append((int(years[i]), prev, curr))

    return transitions


# ─────────────────────────────────────────────────────────────────────────────
# 4. Console Insight Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _longest_high_stress_streak(
    df: pd.DataFrame,
    regime_col: str = "analysis_regime",
    year_col: str = "Year",
    high_label: str = "High Stress",
) -> Tuple[int, int, int]:
    """
    Find the longest consecutive run of High Stress years.

    Returns (length, start_year, end_year).
    """
    best_len = best_start = best_end = 0
    cur_len  = cur_start  = 0

    for _, row in df.iterrows():
        if row[regime_col] == high_label:
            if cur_len == 0:
                cur_start = int(row[year_col])
            cur_len += 1
            cur_end = int(row[year_col])
            if cur_len > best_len:
                best_len   = cur_len
                best_start = cur_start
                best_end   = cur_end
        else:
            cur_len = 0

    return best_len, best_start, best_end


def _most_volatile_decade(
    df: pd.DataFrame,
    col: str = "esi_score",
    year_col: str = "Year",
) -> Tuple[int, float]:
    """
    Identify the 10-year window with the highest ESI standard deviation.

    Returns (decade_start_year, std_dev).
    """
    df = df.copy().sort_values(year_col)
    years = df[year_col].tolist()
    values = df[col].tolist()

    best_decade = int(years[0])
    best_std    = 0.0

    for i in range(len(years) - 9):
        window_vals = values[i : i + 10]
        std = pd.Series(window_vals).std()
        if std > best_std:
            best_std    = std
            best_decade = int(years[i])

    return best_decade, best_std


# ─────────────────────────────────────────────────────────────────────────────
# 5. Visualisation Functions
# ─────────────────────────────────────────────────────────────────────────────

def _format_year_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    ax.tick_params(axis="x", rotation=45)


def _save(filename: str) -> None:
    out = _PLOTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"    Saved → {out}")
    plt.close()


def plot_esi_with_rolling_averages(
    df: pd.DataFrame,
    esi_col: str = "esi_score",
    avg_3yr_col: str = "esi_3yr_avg",
    avg_5yr_col: str = "esi_5yr_avg",
    year_col: str = "Year",
    title: str = "India – ESI with Rolling Averages",
    output_filename: str = "india_esi_rolling_avg.png",
) -> None:
    """
    Line plot of ESI overlaid with 3-year and 5-year rolling averages.

    Output: outputs/plots/<output_filename>
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    # Raw ESI (faint, thin)
    ax.plot(
        df[year_col], df[esi_col],
        linewidth=1.5, color=_COL_ESI, alpha=0.45,
        marker="o", markersize=3.5,
        markerfacecolor="white", markeredgewidth=1.2,
        label="ESI Score (annual)",
    )

    # 3-year rolling mean
    ax.plot(
        df[year_col], df[avg_3yr_col],
        linewidth=2.2, color=_COL_3YR, linestyle="-",
        label="3-Year Rolling Avg",
    )

    # 5-year rolling mean
    ax.plot(
        df[year_col], df[avg_5yr_col],
        linewidth=2.5, color=_COL_5YR, linestyle="--",
        label="5-Year Rolling Avg",
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("ESI Score (0 = Low Stress, 1 = High Stress)", fontsize=11)

    _format_year_axis(ax)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    _save(output_filename)


def plot_esi_regime_timeline(
    df: pd.DataFrame,
    esi_col: str = "esi_score",
    regime_col: str = "analysis_regime",
    year_col: str = "Year",
    title: str = "India – ESI Timeline Colored by Stress Regime",
    output_filename: str = "india_esi_regime_timeline.png",
) -> None:
    """
    Color-coded ESI timeline: each year's dot is colored by its stress regime,
    with a shaded background band behind the line.

    Output: outputs/plots/<output_filename>
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    # Background shading per year
    for _, row in df.iterrows():
        color = _REGIME_COLORS.get(str(row[regime_col]), "#DDDDDD")
        ax.axvspan(row[year_col] - 0.5, row[year_col] + 0.5,
                   facecolor=color, alpha=0.25, edgecolor=None)

    # Base line (neutral)
    ax.plot(df[year_col], df[esi_col],
            linewidth=2.0, color=_COL_ESI, alpha=0.6, zorder=4)

    # Colored scatter dots
    for regime, color in _REGIME_COLORS.items():
        mask = df[regime_col].astype(str) == regime
        ax.scatter(
            df.loc[mask, year_col], df.loc[mask, esi_col],
            color=color, s=55, zorder=5,
            edgecolors=_COL_ESI, linewidths=0.6,
            label=regime,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("ESI Score (0 = Low Stress, 1 = High Stress)", fontsize=11)

    _format_year_axis(ax)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    _save(output_filename)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 2A orchestrator.

    Steps:
      1. Compute 3-year and 5-year rolling averages of ESI.
      2. Classify years into Low / Medium / High stress regimes (tertiles).
      3. Detect regime transitions.
      4. Generate plots.
      5. Print concise console insights.

    Args:
        df: Merged DataFrame that already contains an 'esi_score' column.

    Returns:
        Enriched DataFrame with columns: esi_3yr_avg, esi_5yr_avg, analysis_regime.
    """
    print("\n" + "=" * 60)
    print("  📈 Phase 2A – Analytical Intelligence")
    print("=" * 60)

    # ── Step 1: Rolling averages ──────────────────────────────────────────────
    df = compute_rolling_averages(df, col="esi_score", windows=(3, 5))
    print("\n[2A-1] Rolling averages computed (3-yr and 5-yr).")

    # ── Step 2: Regime classification ─────────────────────────────────────────
    df = classify_regimes(df, col="esi_score")
    regime_counts = df["analysis_regime"].value_counts().sort_index()
    print("\n[2A-2] Stress Regime Classification (tertile-based):")
    for regime, count in regime_counts.items():
        print(f"    • {regime:<20} {count} years")

    # ── Step 3: Transition detection ──────────────────────────────────────────
    transitions = detect_regime_transitions(df)
    print(f"\n[2A-3] Regime Transitions Detected: {len(transitions)}")
    for year, from_r, to_r in transitions:
        print(f"    {year}: {from_r}  →  {to_r}")

    # ── Step 4: Plots ─────────────────────────────────────────────────────────
    print("\n[2A-4] Generating analytical plots …")
    plot_esi_with_rolling_averages(
        df,
        title="India – ESI with 3-Year & 5-Year Rolling Averages (1991–2024)",
        output_filename="india_esi_rolling_avg.png",
    )
    plot_esi_regime_timeline(
        df,
        title="India – ESI Timeline Colored by Stress Regime (1991–2024)",
        output_filename="india_esi_regime_timeline.png",
    )
    print("      ✓ Analytical plots saved.")

    # ── Step 5: Console insights ──────────────────────────────────────────────
    print("\n[2A-5] Key Insights:")

    streak_len, streak_start, streak_end = _longest_high_stress_streak(df)
    if streak_len > 0:
        print(
            f"    • Longest High-Stress Streak : {streak_len} consecutive year(s)"
            f"  ({streak_start}–{streak_end})"
        )
    else:
        print("    • Longest High-Stress Streak : none detected")

    decade_start, decade_std = _most_volatile_decade(df)
    decade_end = decade_start + 9
    print(
        f"    • Most Volatile Decade       : {decade_start}–{decade_end}"
        f"  (ESI σ = {decade_std:.4f})"
    )

    return df
