import requests
import pandas as pd
from io import StringIO
import numpy as np
import time
import concurrent.futures
from tqdm import tqdm # For the progress bar

def parse_power_response(response_text):
    """
    Parses the text from a NASA POWER API response, handling different header formats.
    It reliably finds the start of the data by looking for '-END HEADER-'.
    """
    header_end_marker = "-END HEADER-"
    header_end_index = response_text.find(header_end_marker)
    
    if header_end_index == -1:
        return None, "Could not find the '-END HEADER-' marker in the response."

    data_text = response_text[header_end_index + len(header_end_marker):].strip()
    
    if not data_text:
        return None, "No data found after the header."

    df = pd.read_csv(StringIO(data_text))
    
    if 'DOY' in df.columns and 'YEAR' in df.columns:
        # --- FIX WAS HERE --- 
        # Changed df['DOO'] to the correct column name df['DOY']
        temp_date = pd.to_datetime(df['YEAR'].astype(str) + ' ' + df['DOY'].astype(str), format='%Y %j')
        df['MO'] = temp_date.dt.month
        df['DY'] = temp_date.dt.day
        df = df.drop(columns=['DOY'])
        
    return df, None


def fetch_power_data_for_tile(args):
    """
    Fetches data for a single tile and parameter. 
    This version is designed to be called by the thread pool executor.
    It accepts a single 'args' tuple to easily map over it.
    """
    # Unpack the arguments
    parameter, tile, start_date, end_date, tile_index, year = args
    
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/regional"
    params = {
        "start": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "latitude-min": tile["lat_min"],
        "longitude-min": tile["lon_min"],
        "latitude-max": tile["lat_max"],
        "longitude-max": tile["lon_max"],
        "community": "AG",
        "parameters": parameter,
        "format": "CSV"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        content = response.text
        
        if response.headers.get('Content-Type') == 'application/json':
            # API returned a JSON error, which is common for invalid requests
            return None

        df, error_message = parse_power_response(content)

        if error_message:
            # Parsing failed
            return None
            
        df.replace(-999, np.nan, inplace=True)
        # Return the successful DataFrame along with its identifiers
        return (year, tile_index, parameter, df)

    except requests.exceptions.RequestException:
        # Network-related errors
        return None

def generate_tiles(lat_min, lon_min, lat_max, lon_max, step=9.9):
    tiles = []
    lat_steps = np.arange(lat_min, lat_max, step)
    lon_steps = np.arange(lon_min, lon_max, step)
    for lat_start in lat_steps:
        for lon_start in lon_steps:
            lat_end = min(lat_start + step, lat_max)
            lon_end = min(lon_start + step, lon_max)
            if lat_end > lat_start and lon_end > lon_start:
                tiles.append({
                    "lat_min": lat_start, "lon_min": lon_start,
                    "lat_max": lat_end, "lon_max": lon_end
                })
    return tiles

def main():
    """Main function to define region, fetch, and merge data concurrently."""
    # --- User defined variables ---
    REGION_BOTTOM_LEFT_LAT = 41.188865661621
    REGION_BOTTOM_LEFT_LON = 12.0961952209472
    REGION_UPPER_RIGHT_LAT = 70
    REGION_UPPER_RIGHT_LON = 66.1155319213867

    START_YEAR = 1990
    END_YEAR = 2013
    
    PARAMETERS = ["GWETPROF", "GWETROOT", "GWETTOP"]
    OUTPUT_FILE = f"complete_nasa_power_data_{START_YEAR}_to_{END_YEAR}.csv"
    TILE_SIZE_DEGREES = 9.9
    
    MAX_WORKERS = 10 
    # --- End of user defined variables ---

    print("Step 1: Generating tiles...")
    tiles = generate_tiles(
        REGION_BOTTOM_LEFT_LAT, REGION_BOTTOM_LEFT_LON, 
        REGION_UPPER_RIGHT_LAT, REGION_UPPER_RIGHT_LON, 
        step=TILE_SIZE_DEGREES
    )
    print(f"Generated {len(tiles)} tiles using a {TILE_SIZE_DEGREES}-degree step.\n")

    tasks = []
    for year in range(START_YEAR, END_YEAR + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        for i, tile in enumerate(tiles):
            for param in PARAMETERS:
                tasks.append((param, tile, start_date, end_date, i, year))

    print(f"Step 2: Fetching data for {len(tasks)} total requests using up to {MAX_WORKERS} parallel workers...")
    
    results = {year: {i: [] for i in range(len(tiles))} for year in range(START_YEAR, END_YEAR + 1)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(fetch_power_data_for_tile, task): task for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Downloading data"):
            result = future.result()
            if result:
                year, tile_index, parameter, df = result
                if df is not None:
                    merge_keys = [col for col in ['YEAR', 'MO', 'DY', 'HR', 'LAT', 'LON'] if col in df.columns]
                    results[year][tile_index].append(df.set_index(merge_keys))
    
    print("\nStep 3: Merging all downloaded data and saving to CSV...")
    
    all_dataframes = []
    for year in sorted(results.keys()):
        for tile_index in sorted(results[year].keys()):
            tile_data_by_param = results[year][tile_index]
            if tile_data_by_param:
                merged_tile_df = pd.concat(tile_data_by_param, axis=1).reset_index()
                all_dataframes.append(merged_tile_df)

    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)
        
        merge_keys = [col for col in ['YEAR', 'MO', 'DY', 'HR', 'LAT', 'LON'] if col in final_df.columns]
        final_df.drop_duplicates(subset=merge_keys, inplace=True)
        
        cols_order = merge_keys + sorted([param for param in PARAMETERS if param in final_df.columns])
        final_df = final_df[cols_order]

        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Success! All data from {START_YEAR} to {END_YEAR} has been merged and saved to '{OUTPUT_FILE}'")
    else:
        print("❌ Failure. No data was successfully downloaded. Please review any potential errors.")

if __name__ == "__main__":
    main()