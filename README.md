# Economic Stress Index (ESI) - India

This project calculates and visualizes the Economic Stress Index (ESI) for India from 1991 to 2024.

## Overview
The ESI is a composite index derived from key economic indicators:
- Inflation Rate
- Unemployment Rate
- GDP Growth Rate (Inverse contributor)
- Food Inflation
- Interest Rates

## Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv`
3. Activate it: `.venv\Scripts\Activate`
4. Install dependencies: `pip install -r requirements.txt`

## Usage
Run the main pipeline:
```bash
python -m src.main
```

## Outputs
- `outputs/tables/`: CSV files containing the calculated index and merged data.
- `outputs/plots/`: Visualizations of the ESI and its components.
