# Economic Stress Index (ESI)

Python-based analysis of India's unemployment trends from 1991 to 2024 using World Bank data. This project demonstrates clean data engineering practices with modular code organization, standardized data pipelines, and professional visualizations.

## 📊 Project Overview

This project analyzes economic stress indicators for India, focusing on unemployment rates over a 33-year period. The analysis includes:
- Data loading and preprocessing
- Time-series transformation and filtering
- Statistical analysis (min/max identification)
- Professional visualizations (line plots and bar charts)

## 🗂️ Project Structure

```
ESI/
├─ data/
│  ├─ raw/               # Raw CSV data files
│  │  ├─ inflation.csv
│  │  ├─ unemployment.csv
│  ├─ processed/         # Cleaned/processed data outputs
├─ outputs/
│  ├─ plots/             # Generated visualizations
│  ├─ tables/            # Generated tables/CSVs
├─ src/
│  ├─ __init__.py        # Package initialization
│  ├─ main.py            # Main entry point - orchestrates pipeline
│  ├─ loaders.py         # Data loading functions
│  ├─ transforms.py      # Data transformation utilities
│  ├─ plots.py           # Visualization functions
├─ .gitignore
├─ README.md
├─ requirements.txt
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/siddharthbehera08-collab/Economic-Stress-Index.git
   cd ESI
   ```

2. **Create and activate virtual environment (recommended):**
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run the main analysis pipeline:

```bash
python -m src.main
```

This will:
1. Load unemployment data from `data/raw/unemployment.csv`
2. Filter and transform data for India (1991-2024)
3. Display statistical analysis in the console
4. Generate visualizations in `outputs/plots/`

### Output

**Console Output:**
- Full unemployment data table (1991-2024)
- Maximum unemployment year and rate
- Minimum unemployment year and rate

**Generated Files:**
- `outputs/plots/india_unemployment_line.png` - Time series line plot
- `outputs/plots/india_unemployment_bar.png` - Bar chart visualization

## 📦 Module Descriptions

| Module | Purpose |
|--------|---------|
| `src/loaders.py` | Loads and preprocesses CSV files from `data/raw/` |
| `src/transforms.py` | Provides reusable data transformation functions (filtering, reshaping, type conversion) |
| `src/plots.py` | Creates and saves matplotlib visualizations to `outputs/plots/` |
| `src/main.py` | Orchestrates the complete analysis pipeline |

## 🔧 Development

### Code Style

- **Modular Design**: Each module has a single, well-defined responsibility
- **Relative Paths**: All paths are relative to project root for portability
- **Type Safety**: Functions include type hints and docstrings
- **Clean Separation**: Data loading, transformation, and visualization are separated

### Extending the Project

**Adding New Data Sources:**
1. Add new CSV to `data/raw/`
2. Create loader function in `src/loaders.py`
3. Integrate into `src/main.py` pipeline

**Adding New Visualizations:**
1. Create new plot function in `src/plots.py`
2. Call from `src/main.py` with appropriate data

**Adding New Transformations:**
1. Add transformation function to `src/transforms.py`
2. Use in `src/main.py` pipeline

## 📈 Data Source

Unemployment data sourced from [World Bank Open Data](https://data.worldbank.org/).

## 📝 License

This project is open source and available for educational and research purposes.

## 👤 Author

**Siddharth Behera**
- GitHub: [@siddharthbehera08-collab](https://github.com/siddharthbehera08-collab)

---

**Last Updated:** February 2026
