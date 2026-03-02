import pandas as pd
import numpy as np

# Adjust path to import from src
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.features.esi import rolling_zscore

def test_rolling_zscore():
    # 1. Create a dummy time series with 10 periods
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    
    # Generate random data (e.g., simulated market returns)
    data = np.random.normal(loc=0.001, scale=0.02, size=10)
    series = pd.Series(data, index=dates, name="Market_Returns")

    # 2. Apply rolling_zscore with a window of 3
    window = 3
    z = rolling_zscore(series, window=window)

    # 3. Compute rolling mean and rolling std manually for display
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()

    # 4. Create a DataFrame to visually compare
    df = pd.DataFrame({
        "Original": series,
        "Rolling_Mean": rolling_mean,
        "Rolling_Std": rolling_std,
        "Z_Score": z
    })

    print("=== Testing rolling_zscore(window=3) ===")
    print(df.to_string())

if __name__ == "__main__":
    test_rolling_zscore()
