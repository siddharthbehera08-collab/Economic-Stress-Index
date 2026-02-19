import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Output paths
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_TABLES_DIR = _PROJECT_ROOT / "outputs" / "tables"
_PLOTS_DIR = _PROJECT_ROOT / "outputs" / "plots"

def run_regression_pipeline(df: pd.DataFrame):
    """
    Executes Phase 2A: Regression-based prediction of ESI.
    """
    print("\n" + "=" * 60)
    print("  📈 Phase 2A: Regression Analysis")
    print("=" * 60)

    # 1. Feature Engineering
    # Features: Inflation, Unemployment, GDP Growth, Interest Rate
    # Target: ESI Score
    feature_cols = ["inflation_rate", "unemployment_rate", "gdp_growth_rate", "interest_rate"]
    target_col = "esi_score"
    
    # Ensure year is sorted for time-series split
    df = df.sort_values("Year").reset_index(drop=True)
    
    X = df[feature_cols]
    y = df[target_col]
    years = df["Year"]

    # 2. Train-Test Split (80/20 Time-Aware)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    years_train, years_test = years.iloc[:split_idx], years.iloc[split_idx:]
    
    print(f"    Data Split: Train ({len(X_train)}), Test ({len(X_test)})")
    
    # 3. Model Training
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    metrics = []
    trained_models = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"    Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics.append({
            "Model": name,
            "RMSE": rmse,
            "R2": r2
        })
        trained_models[name] = model
        predictions[name] = y_pred

    metrics_df = pd.DataFrame(metrics).sort_values("RMSE")
    best_model_name = metrics_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_preds = predictions[best_model_name]
    
    print("\n[Evaluation Metrics]")
    print(metrics_df.to_string(index=False))

    # Save metrics
    metrics_path = _TABLES_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n    ✓ Metrics saved -> {metrics_path}")
    
    # 4. Plots
    
    # Actual vs Predicted (Best Model)
    _plot_actual_vs_predicted(years_test, y_test, best_preds, best_model_name)
    
    # Residuals vs Year
    residuals = y_test - best_preds
    _plot_residuals(years_test, residuals, best_model_name)
    
    # Feature Importance (if applicable)
    if hasattr(best_model, "feature_importances_"):
        _plot_feature_importance(best_model.feature_importances_, feature_cols, best_model_name)

    return best_model_name, metrics_df.iloc[0]["RMSE"], metrics_df.iloc[0]["R2"]

def _plot_actual_vs_predicted(years, y_true, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, y_true, "o-", label="Actual ESI", color="#1D3557", linewidth=2)
    ax.plot(years, y_pred, "x--", label=f"Predicted ({model_name})", color="#E63946", linewidth=2)
    
    ax.set_title(f"Actual vs Predicted ESI ({model_name})", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("ESI Score")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Integer years
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    path = _PLOTS_DIR / "regression_actual_vs_predicted.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Plot saved -> {path}")

def _plot_residuals(years, residuals, model_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(years, residuals, color="#457B9D", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    
    ax.set_title(f"Residuals over Time ({model_name})", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    path = _PLOTS_DIR / "regression_residuals.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Plot saved -> {path}")

def _plot_feature_importance(importances, feature_names, model_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Sort
    indices = np.argsort(importances)
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    ax.barh(sorted_features, sorted_importances, color="#2A9D8F")
    ax.set_title(f"Feature Importance ({model_name})", fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    path = _PLOTS_DIR / "regression_feature_importance.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Plot saved -> {path}")
