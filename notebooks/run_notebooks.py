import json
import os

def run_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']
    
    global_env = {}
    
    print(f"--- Running {filepath} ---")
    for i, cell in enumerate(code_cells):
        source = "".join(cell['source'])
        
        # apply quick fixes
        if "%matplotlib inline" in source:
            source = source.replace("%matplotlib inline", "")
            
        source = source.replace("Data csvs/interest_rates.csv", "../data/raw/interest_rates.csv")
        source = source.replace("Data csvs/Oil.csv", "../data/raw/oil.csv")
        
        # FIX THE SKIPROWS LOGIC
        source = source.replace(
            "if 'Country Name' not in df_raw.columns and 'Country Code' not in df_raw.columns and df_raw.shape[1] < 3:",
            "if 'Country Name' not in df_raw.columns and 'Country Code' not in df_raw.columns:"
        )
        
        # Don't show plots, just run the code
        if "plt.show()" in source:
            source = source.replace("plt.show()", "pass # plt.show()")
            
        print(f"Executing cell {i+1}/{len(code_cells)}...")
        try:
            exec(source, global_env)
        except Exception as e:
            print(f"Error in cell {i+1} of {filepath}: {e}")
            raise e

if __name__ == "__main__":
    run_notebook("interest_rates_arima.ipynb")
    run_notebook("oil_arima.ipynb")
    print("Done!")
