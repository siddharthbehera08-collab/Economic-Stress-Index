"""
regime_analysis.py
------------------
Phase 2A: ESI Analytical Intelligence and Visualizations.
Adds rolling statistics, stress-regime classification, transition detection,
and targeted visualizations on top of the already-computed ESI scores.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from typing import List, Tuple

from src.config import PLOTS_DIR

# ── Colour constants ────────────────────────────────────────────────────────
_COL_ESI       = "#1D3557"
_COL_3YR       = "#E63946"
_COL_5YR       = "#F4A261"
_COL_LOW       = "#A7C957"
_COL_MED       = "#F4E409"
_COL_HIGH      = "#F08080"

_REGIME_COLORS = {
    "Low Stress":    _COL_LOW,
    "Medium Stress": _COL_MED,
    "High Stress":   _COL_HIGH,
}

PALETTE = {
    "inflation_rate":      "#E63946",
    "food_inflation_rate": "#F4A261",
    "unemployment_rate":   "#2A9D8F",
    "gdp_growth_rate":     "#457B9D",
    "interest_rate":       "#6A4C93",
    "esi_score":           "#1D3557",
    "highlight":           "#D62828",
    "default":             "#2E86AB",
}

LABELS = {
    "inflation_rate":      "Inflation Rate (%)",
    "food_inflation_rate": "Food Inflation Rate (%)",
    "unemployment_rate":   "Unemployment Rate (%)",
    "gdp_growth_rate":     "GDP Growth Rate (%)",
    "interest_rate":       "Interest Rate (%)",
    "esi_score":           "Economic Stress Index (0-1)",
}

# ── 1. Analytical Core Functions ──────────────────────────────────────────────

def compute_rolling_averages(df: pd.DataFrame, col: str = "esi_score", windows: Tuple[int, int] = (3, 5)) -> pd.DataFrame:
    """Computes trailing rolling averages based on given window sizes to prevent data leakage."""
    df = df.copy()
    for w in windows:
        col_name = f"esi_{w}yr_avg"
        # Use trailing window (center=False) to avoid using future data
        df[col_name] = df[col].rolling(window=w, center=False, min_periods=1).mean()
    return df

def classify_regimes(df: pd.DataFrame, col: str = "esi_score") -> pd.DataFrame:
    """Classifies stress scores into Low, Medium, and High regimes based on tertiles."""
    labels = ["Low Stress", "Medium Stress", "High Stress"]
    df = df.copy()
    
    try:
        df["analysis_regime"] = pd.qcut(df[col], q=3, labels=labels)
    except ValueError:
        df["analysis_regime"] = pd.cut(df[col], bins=3, labels=labels)

    df["analysis_regime"] = df["analysis_regime"].astype(str)
    return df

def detect_regime_transitions(df: pd.DataFrame, regime_col: str = "analysis_regime", year_col: str = "Year") -> List[Tuple[int, str, str]]:
    """Detects YoY changes in stress regimes."""
    transitions = []
    regimes = df[regime_col].tolist()
    years = df[year_col].tolist()

    for i in range(1, len(regimes)):
        prev, curr = regimes[i - 1], regimes[i]
        if prev != curr:
            transitions.append((int(years[i]), prev, curr))

    return transitions

def detect_change_points(df: pd.DataFrame, col_name: str, threshold_factor: float = 1.5, year_col: str = "Year") -> list[tuple[int, str]]:
    """
    Detects structural breaks using First-Difference Spike Detection.
    Flags years where the YoY difference exceeds the mean diff + (threshold * std diff).
    """
    series = df[col_name]
    diffs = series.diff().abs()
    
    mu = diffs.mean()
    sigma = diffs.std()
    threshold = mu + (threshold_factor * sigma)
    
    change_points = []
    
    for idx, val in diffs.items():
        if pd.isna(val):
            continue
            
        if val > threshold:
            year = int(df.loc[idx, year_col])
            direction = "Spike" if series[idx] > series[idx-1] else "Drop"
            desc = f"Significant {direction} ({val:.2f} delta)"
            change_points.append((year, desc))
            
    return change_points

# ── 2. Insight Generation Helpers ─────────────────────────────────────────────

def _longest_high_stress_streak(df: pd.DataFrame) -> Tuple[int, int, int]:
    """Finds the longest consecutive run of High Stress years."""
    best_len = best_start = best_end = 0
    cur_len = cur_start = 0

    for _, row in df.iterrows():
        if row["analysis_regime"] == "High Stress":
            if cur_len == 0:
                cur_start = int(row["Year"])
            cur_len += 1
            cur_end = int(row["Year"])
            if cur_len > best_len:
                best_len, best_start, best_end = cur_len, cur_start, cur_end
        else:
            cur_len = 0

    return best_len, best_start, best_end

def _most_volatile_decade(df: pd.DataFrame) -> Tuple[int, float]:
    """Identifies the 10-year window with the highest ESI standard deviation."""
    df = df.copy().sort_values("Year")
    years = df["Year"].tolist()
    values = df["esi_score"].tolist()

    best_decade = int(years[0])
    best_std = 0.0

    for i in range(len(years) - 9):
        window_vals = values[i : i + 10]
        std = pd.Series(window_vals).std()
        if std > best_std:
            best_std, best_decade = std, int(years[i])

    return best_decade, best_std

# ── 3. Visualization Core Functions ───────────────────────────────────────────

def _format_year_axis(ax: plt.Axes) -> None:
    """Formats x-axis to display integer years cleanly."""
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    ax.tick_params(axis="x", rotation=45)

def _save_plot(filename: str) -> None:
    """Saves the active matplotlib plot to the pre-configured plots directory."""
    out_path = PLOTS_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"    Saved plot -> {out_path}")
    plt.close()

# ── 4. General Plots ──────────────────────────────────────────────────────────

def plot_time_series(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, output_filename: str, annotations: dict = None) -> None:
    """Draws a standard single-indicator line chart, optionally with event annotations."""
    color = PALETTE.get(y_col, PALETTE["default"])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df[x_col], df[y_col], linewidth=2.2, color=color, marker="o",
            markersize=3.5, markerfacecolor="white", markeredgewidth=1.5)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)

    if annotations:
        y_min, y_max = df[y_col].min(), df[y_col].max()
        y_range = y_max - y_min if y_max != y_min else 1.0
        text_y = y_max + y_range * 0.04
        for year, label in annotations.items():
            ax.axvline(x=year, color="#888888", linestyle=":", linewidth=1.4, alpha=0.8)
            ax.text(year, text_y, label, rotation=90, ha="right", va="bottom", fontsize=8, color="#555555", style="italic")
        ax.set_ylim(bottom=y_min - y_range * 0.05, top=text_y + y_range * 0.2)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    
    _format_year_axis(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    _save_plot(output_filename)

def plot_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, output_filename: str) -> None:
    """Draws a standard bar chart."""
    color = PALETTE.get(y_col, PALETTE["default"])
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(df[x_col], df[y_col], color=color, alpha=0.82, width=0.7)

    # Color negative bars differently
    for bar, val in zip(bars, df[y_col]):
        if pd.notna(val) and val < 0:
            bar.set_color("#AAAAAA")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    
    _format_year_axis(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    _save_plot(output_filename)

def plot_multi_line(df: pd.DataFrame, x_col: str, y_cols: list[str], title: str, xlabel: str, ylabel: str, output_filename: str) -> None:
    """Overlaid line chart for several indicators."""
    fig, ax = plt.subplots(figsize=(13, 6))

    for col in y_cols:
        color = PALETTE.get(col, PALETTE["default"])
        label = LABELS.get(col, col.replace("_", " ").title())
        ax.plot(df[x_col], df[col], linewidth=2, color=color, label=label, marker="o", markersize=2.5)

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    
    _format_year_axis(ax)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    _save_plot(output_filename)

def plot_composite_stress(df_norm: pd.DataFrame, x_col: str, y_cols: list[str], title: str, output_filename: str) -> None:
    """A stacked area chart showing the normalized contribution of each stressor to the ESI."""
    fig, ax = plt.subplots(figsize=(13, 6))

    colors = [PALETTE.get(c, PALETTE["default"]) for c in y_cols]
    labels = [LABELS.get(c, c.replace("_", " ").title()) for c in y_cols]

    ax.stackplot(
        df_norm[x_col],
        [df_norm[c] for c in y_cols],
        labels=labels,
        colors=colors,
        alpha=0.72,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Normalised Stress Contribution", fontsize=11)
    
    _format_year_axis(ax)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    _save_plot(output_filename)

# ── 5. Analytical Visualization Functions ─────────────────────────────────────

def plot_bar_stress_years(df: pd.DataFrame, x_col: str, y_col: str, title: str, top_n: int, output_filename: str) -> None:
    """Bar chart highlighting the top N highest-stress years."""
    top_years = df.nlargest(top_n, y_col)[x_col].values
    
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(df[x_col], df[y_col], color=PALETTE["default"], alpha=0.6, width=0.7)

    for bar, year in zip(bars, df[x_col]):
        if year in top_years:
            bar.set_color(PALETTE["highlight"])
            bar.set_alpha(0.9)
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("ESI Score (0-1)", fontsize=11)
    
    _format_year_axis(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save_plot(output_filename)

def plot_esi_comparison(df: pd.DataFrame, x_col: str, esi_col: str, comp_cols: list[str], title: str, output_filename: str) -> None:
    """Dual-axis chart comparing ESI against macro indicators."""
    fig, ax1 = plt.subplots(figsize=(13, 6))

    # Left Axis: ESI Score
    color_esi = PALETTE["esi_score"]
    ax1.plot(df[x_col], df[esi_col], color=color_esi, linewidth=3, label="ESI Score", zorder=10)
    ax1.fill_between(df[x_col], df[esi_col], alpha=0.1, color=color_esi)
    ax1.set_ylabel("ESI Score (0-1)", fontsize=11, color=color_esi)
    ax1.tick_params(axis='y', labelcolor=color_esi)
    ax1.set_xlabel("Year", fontsize=11)
    
    # Right Axis: Raw Macro Indicators
    ax2 = ax1.twinx()
    for col in comp_cols:
        color = PALETTE.get(col, "#555555")
        label = LABELS.get(col, col)
        ax2.plot(df[x_col], df[col], color=color, linewidth=1.5, linestyle="--", label=label, alpha=0.8)
    
    ax2.set_ylabel("Macro Indicators (%)", fontsize=11, color="#555555")
    ax2.tick_params(axis='y', labelcolor="#555555")

    # Combine legends cleanly
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    ax1.set_title(title, fontsize=14, fontweight="bold", pad=12)
    _format_year_axis(ax1)
    ax1.grid(axis="x", linestyle="--", alpha=0.3)
    
    fig.tight_layout()
    _save_plot(output_filename)

def plot_esi_with_rolling_averages(df: pd.DataFrame, esi_col: str, avg_3yr_col: str, avg_5yr_col: str, year_col: str, title: str, output_filename: str) -> None:
    """Line plot of ESI overlaid with 3-year and 5-year rolling averages."""
    fig, ax = plt.subplots(figsize=(13, 6))

    # Raw ESI Line
    ax.plot(df[year_col], df[esi_col], linewidth=1.5, color=_COL_ESI, alpha=0.45,
            marker="o", markersize=3.5, markerfacecolor="white", markeredgewidth=1.2, label="ESI Score (annual)")

    # 3-Year Rolling Avg
    ax.plot(df[year_col], df[avg_3yr_col], linewidth=2.2, color=_COL_3YR, linestyle="-", label="3-Year Rolling Avg")

    # 5-Year Rolling Avg
    ax.plot(df[year_col], df[avg_5yr_col], linewidth=2.5, color=_COL_5YR, linestyle="--", label="5-Year Rolling Avg")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("ESI Score (0 = Low, 1 = High)", fontsize=11)

    _format_year_axis(ax)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    _save_plot(output_filename)

def plot_esi_regime_timeline(df: pd.DataFrame, esi_col: str, regime_col: str, year_col: str, title: str, output_filename: str) -> None:
    """Color-coded ESI timeline representing the stress regimes."""
    fig, ax = plt.subplots(figsize=(13, 6))

    # Color coordinate background based on regime
    for _, row in df.iterrows():
        color = _REGIME_COLORS.get(str(row[regime_col]), "#DDDDDD")
        ax.axvspan(row[year_col] - 0.5, row[year_col] + 0.5,
                   facecolor=color, alpha=0.25, edgecolor=None)

    # Base line
    ax.plot(df[year_col], df[esi_col], linewidth=2.0, color=_COL_ESI, alpha=0.6, zorder=4)

    # Scatted color dots matching the regime
    for regime, color in _REGIME_COLORS.items():
        mask = df[regime_col].astype(str) == regime
        ax.scatter(df.loc[mask, year_col], df.loc[mask, esi_col], color=color, s=55, zorder=5,
                   edgecolors=_COL_ESI, linewidths=0.6, label=regime)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("ESI Score", fontsize=11)

    _format_year_axis(ax)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    _save_plot(output_filename)

def plot_change_points(df: pd.DataFrame, x_col: str, y_col: str, change_points: list[tuple[int, str]], title: str, output_filename: str) -> None:
    """Line plot with vertical markers highlighting structural YoY breaks."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df[x_col], df[y_col], linewidth=2.2, color=PALETTE["esi_score"], zorder=5)

    for year, desc in change_points:
        ax.axvline(x=year, color=PALETTE["highlight"], linestyle="--", linewidth=1.5, alpha=0.8)
        ax.text(year, ax.get_ylim()[1]*0.95, str(year), rotation=90, 
                color=PALETTE["highlight"], fontsize=9, fontweight='bold', ha='right')

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("ESI Score", fontsize=11)
    
    _format_year_axis(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save_plot(output_filename)

# ── 6. Pipeline Orchestrator ──────────────────────────────────────────────────

def run_analysis_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes Phase 2A analytical logic.
    Computes rolling averages, regimes, change points, and visually renders everything.
    """
    print("\n" + "=" * 60)
    print("  📈 Phase 2A – Analytical Intelligence")
    print("=" * 60)

    # 1. Rolling Averages
    df = compute_rolling_averages(df, col="esi_score", windows=(3, 5))
    print("\n[2A-1] Rolling averages computed (3-yr and 5-yr).")

    # 2. Risk Regime Classification
    df = classify_regimes(df, col="esi_score")
    regime_counts = df["analysis_regime"].value_counts().sort_index()
    print("\n[2A-2] Stress Regime Classification (tertile-based):")
    for regime, count in regime_counts.items():
        print(f"    • {regime:<20} {count} years")

    # 3. Transition Points
    transitions = detect_regime_transitions(df)
    print(f"\n[2A-3] Regime Transitions Detected: {len(transitions)}")
    for year, from_r, to_r in transitions:
        print(f"    {year}: {from_r}  →  {to_r}")

    # 4. Change-Point / Structural Break Detection
    change_points = detect_change_points(df, "esi_score")
    print("\n[2A-4] Structural Break Detection (Change-Points):")
    if change_points:
        for year, desc in change_points:
            print(f"    • {year}: {desc}")
    else:
        print("    No significant structural breaks detected.")

    # 5. Visual Rendering
    print("\n[2A-5] Generating analytical plots …")
    country = "India" # Derived from main.py's global scope, injected naturally here.
    plot_esi_with_rolling_averages(df, "esi_score", "esi_3yr_avg", "esi_5yr_avg", "Year",
                                   f"{country} – ESI with 3-Year & 5-Year Rolling Averages", "india_esi_rolling_avg.png")
    
    plot_esi_regime_timeline(df, "esi_score", "analysis_regime", "Year",
                             f"{country} – ESI Timeline Colored by Stress Regime", "india_esi_regime_timeline.png")

    plot_change_points(df, "Year", "esi_score", change_points,
                       f"{country} – ESI Structural Breaks", "india_esi_change_points.png")

    # 6. Final Log Outputs
    print("\n[2A-6] Key Insights:")
    streak_len, streak_start, streak_end = _longest_high_stress_streak(df)
    if streak_len > 0:
        print(f"    • Longest High-Stress Streak : {streak_len} consecutive year(s) ({streak_start}–{streak_end})")
    else:
        print("    • Longest High-Stress Streak : none detected")

    decade_start, decade_std = _most_volatile_decade(df)
    print(f"    • Most Volatile Decade       : {decade_start}–{decade_start + 9} (ESI σ = {decade_std:.4f})")

    return df
