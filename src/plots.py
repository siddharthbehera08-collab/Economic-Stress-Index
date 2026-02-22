"""
plots.py – Visualisation functions for the ESI pipeline.

All plots are saved to outputs/plots/ relative to the project root.
Functions:
  • plot_time_series        – single-indicator line chart
  • plot_bar_chart          – single-indicator bar chart
  • plot_multi_line         – overlaid line chart for several indicators
  • plot_composite_stress   – normalised "economic stress" area/line chart
  • plot_bar_stress_years   – bar chart highlighting high-stress years
  • plot_esi_comparison     – dual-axis chart comparing ESI vs macro indicators
  • plot_stress_regimes     – line chart with regime background shading
  • plot_change_points      – line chart with structural break markers
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── Output directory ──────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
_PLOTS_DIR = _PROJECT_ROOT / "outputs" / "plots"
_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "inflation_rate":      "#E63946",   # red
    "food_inflation_rate": "#F4A261",   # orange
    "unemployment_rate":   "#2A9D8F",   # teal
    "gdp_growth_rate":     "#457B9D",   # steel blue
    "interest_rate":       "#6A4C93",   # purple
    "esi_score":           "#1D3557",   # dark blue
    "highlight":           "#D62828",   # bright red
    "default":             "#2E86AB",
    "regime_low":          "#A7C957",   # green
    "regime_mod":          "#F4E409",   # yellow
    "regime_high":         "#F08080",   # light coral
}

LABELS = {
    "inflation_rate":      "Inflation Rate (%)",
    "food_inflation_rate": "Food Inflation Rate (%)",
    "unemployment_rate":   "Unemployment Rate (%)",
    "gdp_growth_rate":     "GDP Growth Rate (%)",
    "interest_rate":       "Interest Rate (%)",
    "esi_score":           "Economic Stress Index (0-1)",
}


def _save_or_show(output_filename: str | None) -> None:
    if output_filename:
        out_path = _PLOTS_DIR / output_filename
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"    Saved → {out_path}")
    else:
        plt.show()
    plt.close()


def _format_year_axis(ax: plt.Axes) -> None:
    """Apply standard integer year formatting to the x-axis."""
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    ax.tick_params(axis="x", rotation=45)


# ── Single-indicator plots ────────────────────────────────────────────────────

def plot_time_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_filename: str | None = None,
    annotations: "dict[int, str] | None" = None,
) -> None:
    """Line chart for a single indicator over time.
    
    Args:
        annotations: Optional dict mapping {year: event_label} to draw
                     vertical reference lines with event names on the plot.
                     Example: {1991: 'BoP Crisis', 2020: 'COVID-19'}
    """
    color = PALETTE.get(y_col, PALETTE["default"])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df[x_col], df[y_col], linewidth=2.2, color=color, marker="o",
            markersize=3.5, markerfacecolor="white", markeredgewidth=1.5)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)

    # ── Event annotations ────────────────────────────────────────────────────
    if annotations:
        y_min, y_max = df[y_col].min(), df[y_col].max()
        y_range = y_max - y_min if y_max != y_min else 1.0
        text_y = y_max + y_range * 0.04  # Slightly above the max value
        for year, label in annotations.items():
            ax.axvline(x=year, color="#888888", linestyle=":", linewidth=1.4, alpha=0.8)
            ax.text(
                year, text_y, label,
                rotation=90, ha="right", va="bottom",
                fontsize=8, color="#555555", style="italic",
            )
        # Expand y-axis to make room for top labels
        ax.set_ylim(bottom=y_min - y_range * 0.05, top=text_y + y_range * 0.2)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    
    _format_year_axis(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    _save_or_show(output_filename)


def plot_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_filename: str | None = None,
) -> None:
    """Bar chart for a single indicator."""
    color = PALETTE.get(y_col, PALETTE["default"])

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(df[x_col], df[y_col], color=color, alpha=0.82, width=0.7)

    # Colour negative bars differently
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

    _save_or_show(output_filename)


# ── Analytical Visualizations ─────────────────────────────────────────────────

def plot_stress_regimes(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    regime_col: str,
    title: str,
    output_filename: str | None = None,
) -> None:
    """Line plot with background shading for Stress Regimes."""
    fig, ax = plt.subplots(figsize=(13, 6))

    # Draw background spans first
    # We iterate year by year to color the background
    # This is a discrete approximation for annual data
    for idx, row in df.iterrows():
        year = row[x_col]
        regime = row[regime_col]
        
        color = "#FFFFFF"
        if "Low" in regime: color = PALETTE["regime_low"]
        elif "Moderate" in regime: color = PALETTE["regime_mod"]
        elif "High" in regime: color = PALETTE["regime_high"]
            
        ax.axvspan(year - 0.5, year + 0.5, facecolor=color, alpha=0.3, edgecolor=None)

    # Line plot
    ax.plot(df[x_col], df[y_col], linewidth=2.5, color=PALETTE["esi_score"], 
            marker="o", markersize=4, zorder=5)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("ESI Score", fontsize=11)
    
    _format_year_axis(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    
    # Custom legend for regimes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["regime_low"], alpha=0.3, label='Low Stress'),
        Patch(facecolor=PALETTE["regime_mod"], alpha=0.3, label='Moderate Stress'),
        Patch(facecolor=PALETTE["regime_high"], alpha=0.3, label='High Stress'),
        plt.Line2D([0], [0], color=PALETTE["esi_score"], lw=2.5, label='ESI Score'),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    fig.tight_layout()
    _save_or_show(output_filename)


def plot_change_points(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    change_points: list[tuple[int, str]],
    title: str,
    output_filename: str | None = None,
) -> None:
    """Line plot with vertical markers for structural breaks."""
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(df[x_col], df[y_col], linewidth=2.2, color=PALETTE["esi_score"], zorder=5)

    # Add markers
    for year, desc in change_points:
        ax.axvline(x=year, color=PALETTE["highlight"], linestyle="--", linewidth=1.5, alpha=0.8)
        # Add text annotation
        ax.text(year, ax.get_ylim()[1]*0.95, str(year), rotation=90, 
                color=PALETTE["highlight"], fontsize=9, fontweight='bold', ha='right')

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("ESI Score", fontsize=11)
    
    _format_year_axis(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    
    fig.tight_layout()
    _save_or_show(output_filename)


# ── Previous Refinement Plots ─────────────────────────────────────────────────

def plot_bar_stress_years(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    top_n: int = 5,
    output_filename: str | None = None,
) -> None:
    """Bar chart of ESI scores, highlighting the top N highest-stress years."""
    # Identify top N years
    top_years = df.nlargest(top_n, y_col)[x_col].values
    
    fig, ax = plt.subplots(figsize=(13, 5))
    
    # Base bars
    bars = ax.bar(df[x_col], df[y_col], color=PALETTE["default"], alpha=0.6, width=0.7)

    # Highlight top years
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
    _save_or_show(output_filename)


def plot_esi_comparison(
    df: pd.DataFrame,
    x_col: str,
    esi_col: str,
    comp_cols: list[str],
    title: str,
    output_filename: str | None = None,
) -> None:
    """
    Dual-axis chart: ESI (Area/Line) on Left, Comparison Indicators on Right.
    """
    fig, ax1 = plt.subplots(figsize=(13, 6))

    # Left Axis: ESI
    color_esi = PALETTE["esi_score"]
    ax1.plot(df[x_col], df[esi_col], color=color_esi, linewidth=3, label="ESI Score", zorder=10)
    ax1.fill_between(df[x_col], df[esi_col], alpha=0.1, color=color_esi)
    ax1.set_ylabel("ESI Score (0-1)", fontsize=11, color=color_esi)
    ax1.tick_params(axis='y', labelcolor=color_esi)
    ax1.set_xlabel("Year", fontsize=11)
    
    # Right Axis: Indicators (Percentage)
    ax2 = ax1.twinx()
    for col in comp_cols:
        color = PALETTE.get(col, "#555555")
        label = LABELS.get(col, col)
        ax2.plot(df[x_col], df[col], color=color, linewidth=1.5, linestyle="--", label=label, alpha=0.8)
    
    ax2.set_ylabel("Macro Indicators (%)", fontsize=11, color="#555555")
    ax2.tick_params(axis='y', labelcolor="#555555")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    ax1.set_title(title, fontsize=14, fontweight="bold", pad=12)
    _format_year_axis(ax1)
    ax1.grid(axis="x", linestyle="--", alpha=0.3)
    
    fig.tight_layout()
    _save_or_show(output_filename)


def plot_multi_line(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    output_filename: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    for col in y_cols:
        color = PALETTE.get(col, PALETTE["default"])
        label = LABELS.get(col, col.replace("_", " ").title())
        ax.plot(df[x_col], df[col], linewidth=2, color=color,
                label=label, marker="o", markersize=2.5)

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    
    _format_year_axis(ax)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    _save_or_show(output_filename)


def plot_composite_stress(
    df_norm: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    output_filename: str | None = None,
) -> None:
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

    _save_or_show(output_filename)
