import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Locate CSV file
csv_path = Path(__file__).parent / "data" / "raw" / "unemployment.csv"

# 2. Load dataset
df = pd.read_csv(csv_path, skiprows=4)

# 3. Remove empty trailing columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# 4. Filter India
india = df[df["Country Name"] == "India"]

# 5. Select years 1991–2024
years = [str(year) for year in range(1991, 2025)]
india_years = india[years]

# 6. Convert wide → long format
india_long = india_years.melt(var_name="Year", value_name="Unemployment Rate")

# 7. Fix data types
india_long["Year"] = india_long["Year"].astype(int)
india_long["Unemployment Rate"] = india_long["Unemployment Rate"].astype(float)

# 8. Sort
india_long = india_long.sort_values("Year")

# -------------------------------
# SHOW ALL YEARS
# -------------------------------
print("\nFull India Unemployment Data (1991–2024):")
print(india_long)

# -------------------------------
# FIND MAX & MIN
# -------------------------------
max_row = india_long.loc[india_long["Unemployment Rate"].idxmax()]
min_row = india_long.loc[india_long["Unemployment Rate"].idxmin()]

print("\nMaximum Unemployment:")
print(f"Year: {max_row['Year']}, Rate: {max_row['Unemployment Rate']}%")

print("\nMinimum Unemployment:")
print(f"Year: {min_row['Year']}, Rate: {min_row['Unemployment Rate']}%")

# -------------------------------
# VISUALIZATION
# -------------------------------
plt.figure()
plt.plot(india_long["Year"], india_long["Unemployment Rate"])
plt.title("India Unemployment Rate (1991–2024)")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
