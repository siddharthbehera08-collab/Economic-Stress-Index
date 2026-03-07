"""
individual_trainer.py
---------------------
Trains Regression + Classification + Anomaly Detection for EACH indicator CSV.
Saves models to saved_models/, metrics to outputs/tables/, plots to outputs/plots/.
"""
import pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import cross_val_score

from src.config import PLOTS_DIR, TABLES_DIR, MODELS_DIR, PALETTE, RANDOM_STATE, CV_FOLDS, ANOMALY_CONTAMINATION, CSV_REGISTRY
from src.data.transforms import minmax_normalise
warnings.filterwarnings("ignore")


def train_individual_models(indicator_key: str, df: pd.DataFrame) -> dict:
    _, label, _ = CSV_REGISTRY[indicator_key]
    value_col = _infer_value_col(df)
    print(f"\n{'='*60}\n  Individual Models: {label}\n{'='*60}")
    results = {}
    results["regression"]     = _regression(indicator_key, df, value_col, label)
    results["classification"] = _classification(indicator_key, df, value_col, label)
    results["anomaly"]        = _anomaly(indicator_key, df, value_col, label)
    _save_metrics(indicator_key, results)
    return results


def _regression(key, df, value_col, label):
    print(f"\n  [REG] Predicting {value_col}")
    fc = [c for c in df.columns if c not in ("Year", value_col)]
    X, y, years = df[fc].values, df[value_col].values, df["Year"].values
    sp = int(len(df) * 0.8)
    Xtr, Xte, ytr, yte, yrs = X[:sp], X[sp:], y[:sp], y[sp:], years[sp:]

    models = {"Ridge": Ridge(alpha=1.0),
               "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
               "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE)}
    records, best_rmse, best_name, best_preds = [], np.inf, None, None

    for name, m in models.items():
        m.fit(Xtr, ytr); preds = m.predict(Xte)
        rmse = np.sqrt(mean_squared_error(yte, preds)); r2 = r2_score(yte, preds)
        cv = -cross_val_score(m, X, y, cv=min(CV_FOLDS, len(df)//2), scoring="neg_root_mean_squared_error").mean()
        records.append({"Model": name, "RMSE": round(rmse,4), "R2": round(r2,4), "CV_RMSE": round(cv,4)})
        print(f"    {name:<22} RMSE={rmse:.4f}  R2={r2:.4f}  CV={cv:.4f}")
        _save_model(m, f"ind_{key}_reg_{name.lower()}")
        if rmse < best_rmse: best_rmse, best_name, best_preds = rmse, name, preds

    print(f"    Best: {best_name} (RMSE={best_rmse:.4f})")
    _plot_reg(yrs, yte, best_preds, best_name, key, label)
    _plot_resid(yrs, yte - best_preds, best_name, key, label)
    return {"metrics": pd.DataFrame(records), "best_model": best_name}


def _classification(key, df, value_col, label):
    print(f"\n  [CLF] Classifying stress level for {value_col}")
    norm = minmax_normalise(df[[value_col]], cols=[value_col])
    p33, p67 = norm[value_col].quantile(0.33), norm[value_col].quantile(0.67)
    df = df.copy()
    df["stress_label"] = norm[value_col].map(lambda v: "Low" if v<=p33 else ("Medium" if v<=p67 else "High"))
    fc = [c for c in df.columns if c not in ("Year", value_col, "stress_label")]
    if not fc or len(set(df["stress_label"])) < 2:
        print("    Skipping - not enough features or classes.")
        return {"metrics": pd.DataFrame(), "best_model": "N/A"}

    X, y = df[fc].values, df["stress_label"].values
    sp = int(len(df) * 0.8)
    Xtr, Xte, ytr, yte = X[:sp], X[sp:], y[:sp], y[sp:]
    models = {"LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
               "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)}
    records, best_cv, best_name = [], 0, None

    for name, m in models.items():
        m.fit(Xtr, ytr)
        acc = accuracy_score(yte, m.predict(Xte))
        cv  = cross_val_score(m, X, y, cv=min(CV_FOLDS, len(df)//2), scoring="accuracy").mean()
        records.append({"Model": name, "Test_Acc": round(acc,4), "CV_Acc": round(cv,4)})
        print(f"    {name:<22} Acc={acc:.4f}  CV={cv:.4f}")
        _save_model(m, f"ind_{key}_clf_{name.lower()}")
        if cv > best_cv: best_cv, best_name = cv, name

    print(f"    Best: {best_name} (CV={best_cv:.4f})")
    _plot_dist(df, key, label)
    return {"metrics": pd.DataFrame(records), "best_model": best_name}


def _anomaly(key, df, value_col, label):
    print(f"\n  [ANO] Anomaly detection for {value_col}")
    df = df.copy().sort_values("Year").reset_index(drop=True)
    mu, sigma = df[value_col].mean(), df[value_col].std()
    df["z_score"]   = (df[value_col] - mu) / sigma
    df["anomaly_z"] = df["z_score"].abs() > 2.0
    iso = IsolationForest(contamination=ANOMALY_CONTAMINATION, random_state=RANDOM_STATE)
    df["anomaly_iso"] = iso.fit_predict(df[[value_col]]) == -1
    _save_model(iso, f"ind_{key}_anomaly")
    df["is_anomaly"] = df["anomaly_z"] | df["anomaly_iso"]
    years = df[df["is_anomaly"]]["Year"].tolist()
    print(f"    Flagged: {years}")
    _plot_anomaly(df, value_col, key, label)
    return {"metrics": df[["Year", value_col, "z_score", "is_anomaly"]], "anomaly_years": years}


# -- Plots ----------------------------------------------------------------------
def _plot_reg(years, ytrue, ypred, model, key, label):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(years, ytrue, "o-", label="Actual",    color="#1D3557", lw=2)
    ax.plot(years, ypred, "x--",label=f"Pred ({model})", color="#E63946", lw=2)
    ax.set_title(f"{label} � Actual vs Predicted", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel(label)
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout(); plt.savefig(PLOTS_DIR / f"ind_{key}_regression.png", dpi=130, bbox_inches="tight"); plt.close()

def _plot_resid(years, resid, model, key, label):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(years, resid, color="#457B9D", alpha=0.75); ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{label} � Residuals ({model})", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Residual"); ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout(); plt.savefig(PLOTS_DIR / f"ind_{key}_residuals.png", dpi=130, bbox_inches="tight"); plt.close()

def _plot_dist(df, key, label):
    order = ["Low","Medium","High"]; colors = [PALETTE["low"],PALETTE["medium"],PALETTE["high"]]
    counts = [df["stress_label"].value_counts().get(l, 0) for l in order]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(order, counts, color=colors, alpha=0.85, width=0.5); ax.bar_label(bars)
    ax.set_title(f"{label} � Stress Distribution", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4); plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"ind_{key}_stress_dist.png", dpi=130, bbox_inches="tight"); plt.close()

def _plot_anomaly(df, value_col, key, label):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["Year"], df[value_col], color="#1D3557", lw=2, label=label)
    anoms = df[df["is_anomaly"]]
    ax.scatter(anoms["Year"], anoms[value_col], color="#D62828", s=100, zorder=5, marker="X", label="Anomaly")
    for _, row in anoms.iterrows():
        ax.text(row["Year"], row[value_col], f" {int(row['Year'])}", fontsize=8, color="#D62828", fontweight="bold", va="bottom")
    ax.set_title(f"{label} � Anomaly Detection", fontweight="bold")
    ax.set_xlabel("Year"); ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout(); plt.savefig(PLOTS_DIR / f"ind_{key}_anomalies.png", dpi=130, bbox_inches="tight"); plt.close()


# -- Utils ----------------------------------------------------------------------
def _infer_value_col(df):
    skip = {"Year"}; lag_like = {"lag_","rolling_","yoy_"}
    for col in df.columns:
        if col in skip or any(col.startswith(p) for p in lag_like): continue
        return col
    raise ValueError(f"Cannot infer value column from: {df.columns.tolist()}")

def _save_model(model, name):
    with open(MODELS_DIR / f"{name}.pkl", "wb") as f: pickle.dump(model, f)

def _save_metrics(key, results):
    rows = []
    for phase, res in results.items():
        if isinstance(res.get("metrics"), pd.DataFrame) and not res["metrics"].empty:
            df = res["metrics"].copy()
            df.insert(0, "Phase", phase.capitalize()); df.insert(1, "Indicator", key)
            rows.append(df)
    if rows:
        path = TABLES_DIR / f"ind_{key}_metrics.csv"
        pd.concat(rows, ignore_index=True).to_csv(path, index=False)
        print(f"\n  Saved metrics -> {path}")

