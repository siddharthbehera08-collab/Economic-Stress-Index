"""
Main analysis script for Economic Stress Index project.
Orchestrates data loading, transformation, analysis, and visualization.
"""

from src import loaders, transforms, plots


def main():
    """Main analysis pipeline."""
    
    print("=" * 60)
    print("Economic Stress Index - India Unemployment Analysis")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/5] Loading unemployment data...")
    df = loaders.load_unemployment_data()
    print(f"✓ Loaded {len(df)} countries")
    
    # 2. Filter for India
    print("\n[2/5] Filtering data for India...")
    india = transforms.filter_country(df, "India")
    print("✓ India data extracted")
    
    # 3. Select year range and transform
    print("\n[3/5] Selecting years 1991-2024 and transforming data...")
    india_years = transforms.select_year_range(india, 1991, 2024)
    india_long = transforms.wide_to_long(india_years, "Unemployment Rate")
    india_long = transforms.clean_data_types(india_long, value_col="Unemployment Rate")
    india_long = india_long.sort_values("Year").reset_index(drop=True)
    print(f"✓ Transformed to {len(india_long)} data points")
    
    # 4. Display data and statistics
    print("\n[4/5] Analyzing unemployment trends...")
    print("\nFull India Unemployment Data (1991–2024):")
    print(india_long.to_string(index=False))
    
    # Find max and min unemployment rates
    max_row = india_long.loc[india_long["Unemployment Rate"].idxmax()]
    min_row = india_long.loc[india_long["Unemployment Rate"].idxmin()]
    
    print("\n" + "=" * 60)
    print("KEY STATISTICS")
    print("=" * 60)
    print(f"\n📈 Maximum Unemployment:")
    print(f"   Year: {int(max_row['Year'])}, Rate: {max_row['Unemployment Rate']:.2f}%")
    
    print(f"\n📉 Minimum Unemployment:")
    print(f"   Year: {int(min_row['Year'])}, Rate: {min_row['Unemployment Rate']:.2f}%")
    
    # 5. Create visualizations
    print("\n[5/5] Generating visualizations...")
    
    # Line plot
    plots.plot_time_series(
        df=india_long,
        x_col="Year",
        y_col="Unemployment Rate",
        title="India Unemployment Rate (1991–2024)",
        xlabel="Year",
        ylabel="Unemployment Rate (%)",
        output_filename="india_unemployment_line.png"
    )
    
    # Bar chart
    plots.plot_bar_chart(
        df=india_long,
        x_col="Year",
        y_col="Unemployment Rate",
        title="India Unemployment Rate (1991–2024) - Bar Chart",
        xlabel="Year",
        ylabel="Unemployment Rate (%)",
        output_filename="india_unemployment_bar.png"
    )
    
    print("\n" + "=" * 60)
    print("✓ Analysis complete! Check outputs/plots/ for visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
