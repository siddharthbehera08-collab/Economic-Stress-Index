"""
plotter.py  –  ESI overview chart.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from src.config import PLOTS_DIR, PALETTE, COUNTRY

def plot_esi_overview(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["Year"], df["esi_score"], color=PALETTE["esi"], lw=2.5, label="ESI Score")
    ax.fill_between(df["Year"], df["esi_score"], alpha=0.12, color=PALETTE["esi"])
    p67 = df["esi_score"].quantile(0.67)
    p33 = df["esi_score"].quantile(0.33)
    ax.axhspan(p67, 1.0,  color=PALETTE["high"], alpha=0.12, label="High Stress Zone")
    ax.axhspan(0,   p33,  color=PALETTE["low"],  alpha=0.12, label="Low Stress Zone")
    ax.axhline(p67, color=PALETTE["high"], lw=1.0, linestyle="--", alpha=0.6)
    ax.axhline(p33, color=PALETTE["low"],  lw=1.0, linestyle="--", alpha=0.6)
    ax.set_title(f"{COUNTRY} – Composite Economic Stress Index (ESI)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=11); ax.set_ylabel("ESI Score  (0=Low, 1=High)", fontsize=11)
    ax.set_ylim(0, 1); ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=12))
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    out = PLOTS_DIR / "india_esi_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  ESI overview plot saved -> {out}")
    plt.close()
