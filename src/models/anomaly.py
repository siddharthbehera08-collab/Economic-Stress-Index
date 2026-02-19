import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.ensemble import IsolationForest

# Output paths
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_TABLES_DIR = _PROJECT_ROOT / "outputs" / "tables"
_PLOTS_DIR = _PROJECT_ROOT / "outputs" / "plots"

def run_anomaly_detection_pipeline(df: pd.DataFrame):
    """
    Executes Part B: Anomaly Detection (Crisis Years).
    Strategies:
      - Isolation Forest (Global anomaly detection)
      - Z-Score (Deviation from mean)
    """
    print("\n" + "=" * 60)
    print("  🚨 Part B: Anomaly Detection (Crisis Years)")
    print("=" * 60)
    
    df = df.sort_values("Year").reset_index(drop=True)
    
    # 1. Isolation Forest
    X = df[["esi_score"]]
    iso = IsolationForest(contamination=0.1, random_state=42)
    df["anomaly_iso"] = iso.fit_predict(X)
    # -1 is anomaly, 1 is normal -> Convert to boolean True for anomaly
    df["is_anomaly_iso"] = df["anomaly_iso"] == -1
    
    # 2. Z-Score
    mean_esi = df["esi_score"].mean()
    std_esi = df["esi_score"].std()
    df["z_score"] = (df["esi_score"] - mean_esi) / std_esi
    # Flag if absolute Z-score > 2
    df["is_anomaly_z"] = df["z_score"].abs() > 2.0
    
    # 3. Combined Flag (Crisis Year) if EITHER method flags it
    df["crisis_year"] = df["is_anomaly_iso"] | df["is_anomaly_z"]
    
    crisis_years = df[df["crisis_year"]]["Year"].tolist()
    print(f"\n    ⚠️  Detected Crisis Years ({len(crisis_years)}): {crisis_years}")
    
    # 4. Plots
    _plot_anomalies(df)
    _plot_anomaly_frequency(df)

    # Save details
    anomaly_path = _TABLES_DIR / "anomaly_detection.csv"
    df[["Year", "esi_score", "z_score", "crisis_year"]].to_csv(anomaly_path, index=False)
    print(f"    ✓ Anomalies saved -> {anomaly_path}")

    return df

def _plot_anomalies(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot ESI Line
    ax.plot(df["Year"], df["esi_score"], color="#1D3557", linewidth=2, label="ESI Score")
    
    # Highlight Anomalies
    anomalies = df[df["crisis_year"]]
    ax.scatter(anomalies["Year"], anomalies["esi_score"], color="#D62828", 
               s=100, label="Detected Crisis", zorder=5, marker="X")
    
    for _, row in anomalies.iterrows():
        ax.text(row["Year"], row["esi_score"] + 0.02, str(int(row["Year"])), 
                ha='center', fontsize=9, color="#D62828", fontweight="bold")
    
    ax.set_title("Economic Crisis Years Detection (Isolation Forest & Z-Score)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("ESI Score")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    path = _PLOTS_DIR / "anomaly_detection_timeline.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Plot saved -> {path}")

def _plot_anomaly_frequency(df):
    # Just a simple bar chart showing Anomaly vs Normal count
    counts = df["crisis_year"].value_counts()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Normal", "Crisis"], [counts.get(False, 0), counts.get(True, 0)], 
                  color=["#457B9D", "#D62828"], alpha=0.8)
    
    ax.bar_label(bars)
    ax.set_title("Frequency of Crisis Years", fontweight="bold")
    ax.set_ylabel("Number of Years")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    path = _PLOTS_DIR / "anomaly_frequency.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Plot saved -> {path}")
