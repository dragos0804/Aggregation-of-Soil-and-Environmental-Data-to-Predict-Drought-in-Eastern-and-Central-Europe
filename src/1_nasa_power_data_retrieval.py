import requests
import pandas as pd
from io import StringIO
import numpy as np
import time
import random
import concurrent.futures
import os
import shutil
from datetime import datetime, timezone
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
    
    max_retries = 6
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, timeout=60)

            # Retry transient/rate-limit errors with exponential backoff.
            if response.status_code == 429 or 500 <= response.status_code < 600:
                sleep_s = min(60, (2 ** attempt) + random.uniform(0.25, 1.25))
                time.sleep(sleep_s)
                continue

            response.raise_for_status()
            content = response.text

            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/json' in content_type:
                return None, f"Unexpected JSON response for year={year}, tile={tile_index}, param={parameter}"

            df, error_message = parse_power_response(content)
            if error_message:
                return None, f"Parse error for year={year}, tile={tile_index}, param={parameter}: {error_message}"

            df.replace(-999, np.nan, inplace=True)
            return (year, tile_index, parameter, df), None

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return None, f"Request failed after retries for year={year}, tile={tile_index}, param={parameter}: {e}"
            sleep_s = min(60, (2 ** attempt) + random.uniform(0.25, 1.25))
            time.sleep(sleep_s)

    return None, f"Exhausted retries for year={year}, tile={tile_index}, param={parameter}"

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

    START_YEAR = int(os.getenv("NASA_START_YEAR", "1990"))
    END_YEAR = int(os.getenv("NASA_END_YEAR", "2024"))
    if END_YEAR < START_YEAR:
        raise ValueError("NASA_END_YEAR must be greater than or equal to NASA_START_YEAR")
    
    PARAMETERS = ["GWETPROF", "GWETROOT", "GWETTOP"]
    TILE_SIZE_DEGREES = 9.9
    
    MAX_WORKERS = int(os.getenv("NASA_MAX_WORKERS", "2"))
    # --- End of user defined variables ---

    script_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "intermediary_steps"))
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(
        data_dir,
        f"complete_nasa_power_data_{START_YEAR}_to_{END_YEAR}.csv",
    )
    partial_output_file = f"{output_file}.part"
    provenance_file = os.path.join(
        data_dir,
        f"nasa_power_{START_YEAR}_to_{END_YEAR}_provenance.txt",
    )

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"Output already exists: '{output_file}'. Delete it to force a new retrieval.")
        return

    base_end_year = int(os.getenv("NASA_BASE_END_YEAR", "2013"))
    base_file = os.getenv(
        "NASA_BASE_FILE",
        os.path.join(data_dir, f"complete_nasa_power_data_{START_YEAR}_to_{base_end_year}.csv"),
    )
    reuse_base = (
        START_YEAR == 1990
        and END_YEAR > base_end_year
        and os.path.exists(base_file)
        and os.path.getsize(base_file) > 0
    )
    retrieval_start_year = base_end_year + 1 if reuse_base else START_YEAR
    retrieval_started_at = datetime.now(timezone.utc)

    if os.path.exists(partial_output_file):
        os.remove(partial_output_file)
    if reuse_base:
        print(f"Reusing verified base artifact through {base_end_year}: '{base_file}'")
        shutil.copyfile(base_file, partial_output_file)

    print("Step 1: Generating tiles...")
    tiles = generate_tiles(
        REGION_BOTTOM_LEFT_LAT, REGION_BOTTOM_LEFT_LON, 
        REGION_UPPER_RIGHT_LAT, REGION_UPPER_RIGHT_LON, 
        step=TILE_SIZE_DEGREES
    )
    print(f"Generated {len(tiles)} tiles using a {TILE_SIZE_DEGREES}-degree step.\n")

    total_requests = max(0, END_YEAR - retrieval_start_year + 1) * len(tiles) * len(PARAMETERS)
    print(
        f"Step 2: Fetching {total_requests} total requests for "
        f"{retrieval_start_year}-{END_YEAR} using up to {MAX_WORKERS} parallel workers..."
    )

    try:
        for year in range(retrieval_start_year, END_YEAR + 1):
            tasks = []
            year_results = {i: [] for i in range(len(tiles))}
            failure_messages = []
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            for i, tile in enumerate(tiles):
                for param in PARAMETERS:
                    tasks.append((param, tile, start_date, end_date, i, year))

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_task = {executor.submit(fetch_power_data_for_tile, task): task for task in tasks}

                for future in tqdm(
                    concurrent.futures.as_completed(future_to_task),
                    total=len(tasks),
                    desc=f"Downloading {year}",
                ):
                    result, error_message = future.result()
                    if result:
                        _, tile_index, _, df = result
                        if df is not None:
                            merge_keys = [
                                col for col in ['YEAR', 'MO', 'DY', 'HR', 'LAT', 'LON']
                                if col in df.columns
                            ]
                            year_results[tile_index].append(df.set_index(merge_keys))
                    elif error_message:
                        failure_messages.append(error_message)

            if failure_messages:
                sample = "\n".join(f"  - {message}" for message in failure_messages[:10])
                raise RuntimeError(
                    f"NASA POWER retrieval for {year} had {len(failure_messages)} failed requests; "
                    f"the combined artifact was not published.\n{sample}"
                )

            year_dataframes = []
            for tile_index in sorted(year_results):
                tile_data_by_param = year_results[tile_index]
                if len(tile_data_by_param) != len(PARAMETERS):
                    raise RuntimeError(
                        f"NASA POWER retrieval for {year}, tile {tile_index} returned "
                        f"{len(tile_data_by_param)} of {len(PARAMETERS)} parameters"
                    )
                year_dataframes.append(pd.concat(tile_data_by_param, axis=1).reset_index())

            year_df = pd.concat(year_dataframes, ignore_index=True)
            merge_keys = [
                col for col in ['YEAR', 'MO', 'DY', 'HR', 'LAT', 'LON']
                if col in year_df.columns
            ]
            year_df.drop_duplicates(subset=merge_keys, inplace=True)
            cols_order = merge_keys + sorted(param for param in PARAMETERS if param in year_df.columns)
            year_df = year_df[cols_order]
            append_to_existing = os.path.exists(partial_output_file) and os.path.getsize(partial_output_file) > 0
            year_df.to_csv(
                partial_output_file,
                mode="a" if append_to_existing else "w",
                header=not append_to_existing,
                index=False,
            )
            print(f"Saved {len(year_df):,} rows for {year}.")

        os.replace(partial_output_file, output_file)
    except Exception:
        if os.path.exists(partial_output_file):
            os.remove(partial_output_file)
        raise

    retrieval_completed_at = datetime.now(timezone.utc)
    with open(provenance_file, "w", encoding="utf-8") as handle:
        handle.write("Source: NASA POWER Daily Regional API\n")
        handle.write("Endpoint: https://power.larc.nasa.gov/api/temporal/daily/regional\n")
        handle.write(f"Parameters: {', '.join(PARAMETERS)}\n")
        handle.write(f"Time window: {START_YEAR}-01-01 through {END_YEAR}-12-31\n")
        handle.write(f"Retrieval started at (UTC): {retrieval_started_at.isoformat()}\n")
        handle.write(f"Retrieval completed at (UTC): {retrieval_completed_at.isoformat()}\n")
        handle.write(f"Reused base artifact: {base_file if reuse_base else 'none'}\n")
        handle.write(f"Freshly retrieved years: {retrieval_start_year}-{END_YEAR}\n")
        handle.write(f"Output artifact: {output_file}\n")
    print(f"Success! NASA POWER data from {START_YEAR} to {END_YEAR} saved to '{output_file}'")
    print(f"Saved provenance metadata to '{provenance_file}'")

if __name__ == "__main__":
    main()