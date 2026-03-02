import pandas as pd

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Computes a trailing rolling z-score for a given pandas Series.
    
    This function standardizes a time series using strictly historical data 
    (a trailing window) to prevent lookahead bias. It calculates the rolling 
    mean and rolling standard deviation over the specified window and uses 
    them to compute the z-score at each point in time. Global means and 
    standard deviations are explicitly avoided.
    
    Parameters
    ----------
    series : pd.Series
        The input time series data to be standardized.
    window : int
        The size of the trailing rolling window (e.g., 252 for roughly 
        one trading year).
        
    Returns
    -------
    pd.Series
        A pandas Series containing the rolling z-scores, aligned with the 
        original index. The first `window - 1` values will be NaN, reflecting 
        the minimum periods required to construct the initial window.
    """
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()

    z_score = (series - rolling_mean) / rolling_std
    return z_score
