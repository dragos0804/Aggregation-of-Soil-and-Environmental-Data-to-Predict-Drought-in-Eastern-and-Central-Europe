# src/check_years.py
import pandas as pd
import numpy as np

FEATURES_DATA_PATH = '../data/intermediary_steps/complete_spei_benchmark.csv'

print(f"--- Analyzing Unique Years in: {FEATURES_DATA_PATH} ---")

try:
    # Load only the 'YEAR' column for maximum efficiency
    df = pd.read_csv(FEATURES_DATA_PATH, usecols=['YEAR'])
    print("File loaded successfully.")
    
    # Get the unique years and sort them
    unique_years = np.sort(df['YEAR'].unique())
    
    print(f"\nFound {len(unique_years)} unique year(s).")
    print(f"Unique year(s) found in the file: {unique_years}")

    if len(unique_years) > 1:
        print(f"The data spans from {unique_years[0]} to {unique_years[-1]}.")
        print("\nConclusion: Your dataset contains multiple years. The cross-validation strategy is appropriate.")
    else:
        print(f"\n[CRITICAL FINDING] Your dataset ONLY contains data for the year {unique_years[0]}.")
        print("This is the root cause of the validation issues.")
        print("Time-series cross-validation is NOT possible with only one year of data.")

except FileNotFoundError:
    print(f"ERROR: The file was not found at the specified path: {FEATURES_DATA_PATH}")
except KeyError:
    print("ERROR: The file does not contain a 'YEAR' column. Please check the CSV.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")