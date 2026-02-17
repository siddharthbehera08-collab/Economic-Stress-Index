"""
Data loading module for Economic Stress Index project.
Handles loading and preprocessing of raw data from CSV files.
"""

import pandas as pd
from pathlib import Path


def load_unemployment_data():
    """
    Load and preprocess unemployment data from CSV file.
    
    Returns:
        pd.DataFrame: Preprocessed unemployment data with cleaned columns
    """
    # Use relative path from project root
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "raw" / "unemployment.csv"
    
    # Load dataset, skipping metadata rows
    df = pd.read_csv(csv_path, skiprows=4)
    
    # Remove empty trailing columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    return df


def load_inflation_data():
    """
    Load and preprocess inflation data from CSV file.
    
    Returns:
        pd.DataFrame: Preprocessed inflation data with cleaned columns
    """
    # Use relative path from project root
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "raw" / "inflation.csv"
    
    # Load dataset, skipping metadata rows
    df = pd.read_csv(csv_path, skiprows=4)
    
    # Remove empty trailing columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    return df
