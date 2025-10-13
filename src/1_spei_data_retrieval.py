import xarray as xr
import pandas as pd
import os
import requests

def download_file(url, folder):
    """Downloads a file from a URL to a specified folder."""
    local_filename = os.path.join(folder, url.split('/')[-1])
    if os.path.exists(local_filename):
        print(f"File '{local_filename}' already exists. Skipping download.")
        return local_filename
    
    print(f"Downloading from {url} to {local_filename}...")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
        return local_filename
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not download the benchmark file. Details: {e}")
        # Clean up partially downloaded file on error
        if os.path.exists(local_filename):
            os.remove(local_filename)
        raise e

def get_benchmark_dataset():
    """
    Downloads and processes the SPEI global drought dataset to create a benchmark.
    It filters the data to the specified European region and saves it as a CSV.
    """
    print("--- Task 1: Acquiring Benchmark Drought Dataset (SPEI) ---")
    
    # --- Configuration ---
    REGION_BOTTOM_LEFT_LAT = 41.18
    REGION_BOTTOM_LEFT_LON = 12.09
    REGION_UPPER_RIGHT_LAT = 70.0
    REGION_UPPER_RIGHT_LON = 66.11
    
    # CORRECTED and VERIFIED URL for the SPEI dataset
    SPEI_URL = "https://spei.csic.es/spei_database/nc/spei01.nc"
    
    # --- File Paths ---
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, '..', 'data', 'intermediary_steps')
    output_path = os.path.join(data_dir, 'complete_spei_benchmark.csv')
    
    os.makedirs(data_dir, exist_ok=True)

    # --- Download ---
    try:
        netcdf_path = download_file(SPEI_URL, data_dir)
    except requests.exceptions.RequestException:
        print("\nHalting script due to download failure. Please check your internet connection or the provided URL.")
        return

    # --- Process ---
    print("Processing NetCDF file to create benchmark CSV...")
    try:
        # Open the NetCDF file
        ds = xr.open_dataset(netcdf_path)
    except Exception as e:
        print(f"ERROR: Could not open the downloaded NetCDF file. It might be corrupt. Details: {e}")
        return
    
    # Select the region of interest
    region_slice = {
        'lat': slice(REGION_BOTTOM_LEFT_LAT, REGION_UPPER_RIGHT_LAT),
        'lon': slice(REGION_BOTTOM_LEFT_LON, REGION_UPPER_RIGHT_LON),
        # Let's take data from the last few decades for relevance
        'time': slice('1990-01-01', '2023-12-31')
    }
    regional_ds = ds.sel(**region_slice)
    
    # Convert to a pandas DataFrame for easier use
    df = regional_ds.to_dataframe().reset_index()
    
    # Clean up the dataframe
    df = df.rename(columns={'lat': 'LAT', 'lon': 'LON', 'spei': 'DROUGHT_INDEX'})
    df['time'] = pd.to_datetime(df['time'])
    df['YEAR'] = df['time'].dt.year
    df['MO'] = df['time'].dt.month
    df['DY'] = df['time'].dt.day
    df.drop(columns=['time'], inplace=True)
    
    # Drop rows with NaN, which represent ocean or no-data areas
    df.dropna(subset=['DROUGHT_INDEX'], inplace=True)
    
    if df.empty:
        print("ERROR: No data found for the specified region and time. Please check the coordinates.")
        return

    # --- Save ---
    df.to_csv(output_path, index=False)
    print(f"✅ Benchmark dataset created successfully and saved to '{output_path}'")

if __name__ == '__main__':
    get_benchmark_dataset()