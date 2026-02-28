"""
esi.py
------
Logic for calculating the composite Economic Stress Index (ESI).
"""

import pandas as pd

def calculate_esi(df_norm: pd.DataFrame, positive_stressors: list[str], inverse_stressors: list[str]) -> pd.Series:
    """
    Calculates the composite Economic Stress Index (ESI) as an equal-weighted mean.
    
    Formula:
        (Sum of positive stressors + Sum of (1 - inverse stressors)) / Total components
        
    Args:
        df_norm: Min-max normalized DataFrame.
        positive_stressors: List of column names that directly increase stress (e.g. inflation).
        inverse_stressors: List of column names that decrease stress (e.g. GDP growth).
        
    Returns:
        A pandas Series containing the calculated ESI scores (0 to 1).
    """
    total_components = len(positive_stressors) + len(inverse_stressors)
    
    if total_components == 0:
        raise ValueError("No stressors provided for ESI calculation.")

    weighted_sum = 0.0
    
    # Add direct stressors (higher value = more stress)
    for col in positive_stressors:
        weighted_sum += df_norm[col]
        
    # Add inverse stressors (lower value = more stress)
    for col in inverse_stressors:
        weighted_sum += (1.0 - df_norm[col])
        
    # Equal-weighted average
    return weighted_sum / total_components
