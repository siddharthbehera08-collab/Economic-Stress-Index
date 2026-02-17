"""
Data transformation module for Economic Stress Index project.
Handles filtering, selecting, and transforming data.
"""

import pandas as pd


def filter_country(df, country_name):
    """
    Filter DataFrame by country name.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'Country Name' column
        country_name (str): Name of the country to filter
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing only the specified country
    """
    return df[df["Country Name"] == country_name]


def select_year_range(df, start_year, end_year):
    """
    Select specific year columns from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame with year columns
        start_year (int): Starting year (inclusive)
        end_year (int): Ending year (inclusive)
        
    Returns:
        pd.DataFrame: DataFrame with only the selected year columns
    """
    years = [str(year) for year in range(start_year, end_year + 1)]
    return df[years]


def wide_to_long(df, value_name):
    """
    Convert wide format DataFrame to long format.
    
    Args:
        df (pd.DataFrame): Input DataFrame in wide format
        value_name (str): Name for the value column in long format
        
    Returns:
        pd.DataFrame: DataFrame in long format with 'Year' and value columns
    """
    return df.melt(var_name="Year", value_name=value_name)


def clean_data_types(df, year_col="Year", value_col=None):
    """
    Convert columns to appropriate data types.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        year_col (str): Name of the year column
        value_col (str): Name of the value column to convert to float
        
    Returns:
        pd.DataFrame: DataFrame with cleaned data types
    """
    df = df.copy()
    df[year_col] = df[year_col].astype(int)
    
    if value_col:
        df[value_col] = df[value_col].astype(float)
    
    return df
