"""
classification.py
-----------------
Part A: Stress Level Classification.
Classifies the economy into Low, Medium, and High stress regimes using machine learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

from src.config import PLOTS_DIR, TABLES_DIR

def run_classification_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the classification pipeline to predict stress regimes.
    
    Args:
        df: The normalized macroeconomic DataFrame.
        
    Returns:
        The DataFrame with added 'stress_level' predictions.
    """
    print("\n" + "=" * 60)
    print("  🚦 Part A: Stress Level Classification")
    print("=" * 60)

    df = df.sort_values("Year").reset_index(drop=True)
    
    # 1. Create Target Labels (Ternary Stress Levels)
    df["stress_level"] = pd.qcut(df["esi_score"], q=3, labels=["Low", "Medium", "High"])
    
    # 2. Define Features and Target
    feature_cols = ["inflation_rate", "unemployment_rate", "gdp_growth_rate", "interest_rate"]
    target_col = "stress_level"
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Create an 80/20 train-test split for confusion matrix plotting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"    Classes: {y.unique().tolist()}")
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"    Note: 5-fold CV is the primary metric.")
    
    # 3. Model Training
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    metrics = []
    
    for name, model in models.items():
        print(f"\n    Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Primary evaluation using 5-Fold Cross-Validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        print(f"    ✅ 5-Fold CV Accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
        
        # Collect metrics
        metrics.append({
            "Model": name,
            "CV_Mean_Accuracy": round(cv_mean, 4),
            "CV_Std": round(cv_std, 4),
            "Split_Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Split_Precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "Split_Recall": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        })
        
        # Visualizations
        _plot_confusion_matrix(y_test, y_pred, model.classes_, name)
        
        if name == "RandomForest":
            _plot_feature_importance(model.feature_importances_, feature_cols)
            
    # 4. Save Metrics
    metrics_df = pd.DataFrame(metrics)
    print("\n[Classification Metrics]")
    print(metrics_df.to_string(index=False))
    
    metrics_path = TABLES_DIR / "classification_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"    ✓ Metrics saved -> {metrics_path}")
    
    # 5. Additional Visualizations and Insights
    _print_insights(df)
    _plot_stress_distribution(df)

    return df

def _plot_confusion_matrix(y_true, y_pred, classes, model_name) -> None:
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    ax.set_title(f"Confusion Matrix - {model_name}", fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    fig.tight_layout()

    path = PLOTS_DIR / f"classification_confusion_{model_name.lower()}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def _plot_feature_importance(importances, feature_names) -> None:
    """Generates and saves a feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    indices = np.argsort(importances)
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    ax.barh(sorted_features, sorted_importances, color="#2A9D8F")
    ax.set_title("Feature Importance (Random Forest)", fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    path = PLOTS_DIR / "classification_feature_importance.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def _plot_stress_distribution(df: pd.DataFrame) -> None:
    """Generates and saves a bar chart of the stress level distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["Low", "Medium", "High"]
    colors = ["#A7C957", "#F4E409", "#F08080"]
    counts = [df["stress_level"].value_counts().get(lbl, 0) for lbl in order]
    
    ax.bar(order, counts, color=colors, alpha=0.85, width=0.5)
    ax.set_title("Distribution of Stress Levels", fontweight="bold")
    ax.set_xlabel("Stress Level")
    ax.set_ylabel("Count of Years")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    path = PLOTS_DIR / "classification_stress_distribution.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def _print_insights(df: pd.DataFrame) -> None:
    """Prints simple console insights for the classification."""
    print("\n[Insights]")
    high_stress = df[df["stress_level"] == "High"]
    print(f"    • Total High Stress Years: {len(high_stress)}")
    if not high_stress.empty:
        print(f"    • Recent High Stress Years: {high_stress['Year'].tail(5).tolist()}")
        
    means = df.groupby("stress_level")[["inflation_rate", "unemployment_rate", "gdp_growth_rate", "interest_rate"]].mean()
    print("\n    • Average Indicators by Stress Level:")
    print(means.to_string())
