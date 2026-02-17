"""
Visualization module for Economic Stress Index project.
Handles creating and saving plots.
"""

import matplotlib.pyplot as plt
from pathlib import Path


def plot_time_series(df, x_col, y_col, title, xlabel, ylabel, output_filename=None):
    """
    Create a time series line plot.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        x_col (str): Column name for x-axis
        y_col (str): Column name for y-axis
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        output_filename (str, optional): Filename to save plot. If None, displays plot.
        
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col], linewidth=2, color='#2E86AB')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if output_filename:
        # Save to outputs/plots/ directory
        project_root = Path(__file__).parent.parent
        output_path = project_root / "outputs" / "plots" / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_bar_chart(df, x_col, y_col, title, xlabel, ylabel, output_filename=None):
    """
    Create a bar chart.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        x_col (str): Column name for x-axis
        y_col (str): Column name for y-axis
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        output_filename (str, optional): Filename to save plot. If None, displays plot.
        
    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    plt.bar(df[x_col], df[y_col], color='#A23B72', alpha=0.8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    
    if output_filename:
        # Save to outputs/plots/ directory
        project_root = Path(__file__).parent.parent
        output_path = project_root / "outputs" / "plots" / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
