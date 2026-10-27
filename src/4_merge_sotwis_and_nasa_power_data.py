import pandas as pd
import numpy as np
import os

def merge_datasets():
    """
    Merges the synthetically enhanced SOTWIS soil data with the time-series
    NASA POWER soil moisture data based on geographic location.
    
    For each NASA POWER data point, it finds the corresponding SOTWIS bounding
    box and appends its soil properties.
    """
    print("--- Task 3: Merging SOTWIS and NASA POWER Datasets ---")
    
    # --- File Paths ---
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, '..', 'data', 'intermediary_steps')
    final_data_dir = os.path.join(script_dir, '..', 'data')
    start_year = int(os.getenv('DATA_START_YEAR', '1990'))
    end_year = int(os.getenv('DATA_END_YEAR', '2024'))
    sotwis_path = os.path.join(data_dir, 'synthetically_enhanced_sotwis.csv')
    nasa_path = os.getenv(
        'NASA_DATA_PATH',
        os.path.join(data_dir, f'complete_nasa_power_data_{start_year}_to_{end_year}.csv'),
    )
    output_path = os.getenv(
        'FUSED_DATA_PATH',
        os.path.join(final_data_dir, f'complete_integrated_dataset_{start_year}_to_{end_year}.csv'),
    )
    partial_output_path = f'{output_path}.part'
    chunk_size = int(os.getenv('NASA_MERGE_CHUNK_SIZE', '500000'))
    
    # --- Load Data ---
    print("Loading datasets...")
    try:
        sotwis_df = pd.read_csv(sotwis_path)
    except FileNotFoundError as e:
        print(f"ERROR: Could not find input file. Make sure previous steps have run successfully. Details: {e}")
        return
    if not os.path.exists(nasa_path):
        print(f"ERROR: Could not find NASA POWER input file: {nasa_path}")
        return
        
    # --- Prepare for Merge ---
    # Compute assignments only for unique coordinate pairs, then map back.
    sotwis_bounds = sotwis_df[['BOTTOM_LEFT_LAT', 'UPPER_RIGHT_LAT', 'BOTTOM_LEFT_LON', 'UPPER_RIGHT_LON']].values

    sotwis_to_merge = sotwis_df.drop(columns=['BOTTOM_LEFT_LAT', 'UPPER_RIGHT_LAT', 'BOTTOM_LEFT_LON', 'UPPER_RIGHT_LON'])
    coordinate_map = {}
    input_rows = 0
    matched_rows = 0
    wrote_header = False
    if os.path.exists(partial_output_path):
        os.remove(partial_output_path)

    print(f"Streaming NASA POWER data in chunks of {chunk_size:,} rows...")
    try:
        for chunk_number, nasa_chunk in enumerate(pd.read_csv(nasa_path, chunksize=chunk_size), start=1):
            input_rows += len(nasa_chunk)
            unique_coords = nasa_chunk[['LAT', 'LON']].drop_duplicates().itertuples(index=False, name=None)
            for lat, lon in unique_coords:
                key = (lat, lon)
                if key in coordinate_map:
                    continue
                matches = (
                    (lat >= sotwis_bounds[:, 0]) & (lat <= sotwis_bounds[:, 1]) &
                    (lon >= sotwis_bounds[:, 2]) & (lon <= sotwis_bounds[:, 3])
                )
                match_indices = np.where(matches)[0]
                coordinate_map[key] = match_indices[0] if len(match_indices) > 0 else np.nan

            nasa_chunk['sotwis_index'] = [
                coordinate_map[(lat, lon)]
                for lat, lon in nasa_chunk[['LAT', 'LON']].itertuples(index=False, name=None)
            ]
            nasa_chunk.dropna(subset=['sotwis_index'], inplace=True)
            nasa_chunk['sotwis_index'] = nasa_chunk['sotwis_index'].astype(int)
            matched_rows += len(nasa_chunk)

            merged_chunk = nasa_chunk.merge(
                sotwis_to_merge,
                left_on='sotwis_index',
                right_index=True,
            )
            merged_chunk.drop(columns=['sotwis_index'], inplace=True)
            merged_chunk.to_csv(
                partial_output_path,
                mode='a' if wrote_header else 'w',
                header=not wrote_header,
                index=False,
            )
            wrote_header = True
            print(f"Processed chunk {chunk_number}: {input_rows:,} input rows so far.")

        os.replace(partial_output_path, output_path)
    except Exception:
        if os.path.exists(partial_output_path):
            os.remove(partial_output_path)
        raise

    print(f"Join complete. {matched_rows:,} of {input_rows:,} records matched.")
    print(f"Integrated dataset created successfully and saved to '{output_path}'")

if __name__ == '__main__':
    merge_datasets()