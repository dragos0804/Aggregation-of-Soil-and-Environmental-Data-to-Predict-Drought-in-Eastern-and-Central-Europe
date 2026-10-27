import xarray as xr
import pandas as pd
import os
import requests
import urllib3
from datetime import datetime, timezone


SPEI_SCALE = os.getenv('SPEI_SCALE', '01').zfill(2)
if SPEI_SCALE not in {'01', '03', '06'}:
    raise ValueError('SPEI_SCALE must be one of 01, 03, or 06')
SPEI_END_DATE = os.getenv('SPEI_END_DATE', '2024-12-31')
SPEI_DATABASE_VERSION = os.getenv('SPEI_DATABASE_VERSION', '2_11')
SPEI_VERIFY_TLS = os.getenv('SPEI_VERIFY_TLS', 'true').lower() == 'true'

def download_file(url, folder, verify_tls):
    """Download a NetCDF file, resuming a prior partial download when possible."""
    local_filename = os.path.join(folder, url.split('/')[-1])
    partial_filename = f'{local_filename}.part'
    if os.path.exists(local_filename) and os.path.getsize(local_filename) > 0:
        print(f"File '{local_filename}' already exists. Skipping download.")
        return local_filename, {
            'status': 'reused existing local artifact',
            'retrieved_at_utc': None,
            'source_last_modified': None,
            'source_etag': None,
        }

    downloaded_bytes = os.path.getsize(partial_filename) if os.path.exists(partial_filename) else 0
    headers = {'Range': f'bytes={downloaded_bytes}-'} if downloaded_bytes else {}
    mode = 'ab' if downloaded_bytes else 'wb'
    print(f"Downloading from {url} to {local_filename}...")
    try:
        if not verify_tls:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        with requests.get(url, stream=True, timeout=60, headers=headers, verify=verify_tls) as r:
            if downloaded_bytes and r.status_code == 200:
                downloaded_bytes = 0
                mode = 'wb'
            r.raise_for_status()
            with open(partial_filename, mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            source_last_modified = r.headers.get('Last-Modified')
            source_etag = r.headers.get('ETag')
        os.replace(partial_filename, local_filename)
        print("Download complete.")
        return local_filename, {
            'status': 'downloaded from source',
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_last_modified': source_last_modified,
            'source_etag': source_etag,
        }
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not download the benchmark file. Details: {e}")
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
    
    SPEI_URL = (
        f"https://spei.csic.es/spei_database_{SPEI_DATABASE_VERSION}"
        f"/nc/spei{SPEI_SCALE}.nc"
    )
    
    # --- File Paths ---
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, '..', 'data', 'intermediary_steps')
    output_name = 'complete_spei_benchmark.csv' if SPEI_SCALE == '01' else f'complete_spei{SPEI_SCALE}_benchmark.csv'
    output_path = os.path.join(data_dir, output_name)
    
    os.makedirs(data_dir, exist_ok=True)

    # --- Download ---
    try:
        netcdf_path, retrieval_metadata = download_file(SPEI_URL, data_dir, SPEI_VERIFY_TLS)
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
        'time': slice('1990-01-01', SPEI_END_DATE)
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
    provenance_path = os.path.join(data_dir, f'spei{SPEI_SCALE}_provenance.txt')
    netcdf_stat = os.stat(netcdf_path)
    with open(provenance_path, 'w', encoding='utf-8') as provenance_file:
        provenance_file.write(f'Source URL: {SPEI_URL}\n')
        provenance_file.write(f'SPEIbase version: {SPEI_DATABASE_VERSION}\n')
        provenance_file.write(f'SPEI accumulation scale: {SPEI_SCALE} month(s)\n')
        provenance_file.write('Region: latitude 41.18--70.0; longitude 12.09--66.11\n')
        provenance_file.write(f'Time window: 1990-01-01 through {SPEI_END_DATE}\n')
        provenance_file.write(f'Run recorded at (UTC): {datetime.now(timezone.utc).isoformat()}\n')
        provenance_file.write(f'Retrieval status: {retrieval_metadata["status"]}\n')
        if retrieval_metadata['retrieved_at_utc']:
            provenance_file.write(
                f'Source retrieval completed at (UTC): {retrieval_metadata["retrieved_at_utc"]}\n'
            )
        provenance_file.write(
            f'Local NetCDF modified at (UTC): '
            f'{datetime.fromtimestamp(netcdf_stat.st_mtime, timezone.utc).isoformat()}\n'
        )
        if retrieval_metadata['source_last_modified']:
            provenance_file.write(
                f'Source Last-Modified header: {retrieval_metadata["source_last_modified"]}\n'
            )
        if retrieval_metadata['source_etag']:
            provenance_file.write(f'Source ETag: {retrieval_metadata["source_etag"]}\n')
    print(f"Saved provenance metadata to '{provenance_path}'")

if __name__ == '__main__':
    get_benchmark_dataset()