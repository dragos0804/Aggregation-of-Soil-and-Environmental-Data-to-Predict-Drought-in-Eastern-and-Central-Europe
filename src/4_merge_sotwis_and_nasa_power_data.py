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
    sotwis_path = os.path.join(data_dir, 'synthetically_enhanced_sotwis.csv')
    nasa_path = os.path.join(data_dir, 'complete_nasa_power_data_1990_to_2013.csv') # Using the output from our first task
    output_path = os.path.join(final_data_dir, 'complete_integrated_dataset_1990_to_2013.csv')
    
    # --- Load Data ---
    print("Loading datasets...")
    try:
        sotwis_df = pd.read_csv(sotwis_path)
        nasa_df = pd.read_csv(nasa_path)
    except FileNotFoundError as e:
        print(f"ERROR: Could not find input file. Make sure previous steps have run successfully. Details: {e}")
        return
        
    # --- Prepare for Merge ---
    # We will create an 'assignment' column in nasa_df to store the index of the matching sotwis row
    nasa_df['sotwis_index'] = -1
    
    # Convert to NumPy for highly efficient comparison
    sotwis_bounds = sotwis_df[['BOTTOM_LEFT_LAT', 'UPPER_RIGHT_LAT', 'BOTTOM_LEFT_LON', 'UPPER_RIGHT_LON']].values
    nasa_coords = nasa_df[['LAT', 'LON']].values
    
    # --- Perform Spatial Join ---
    print(f"Performing spatial join for {len(nasa_df)} NASA POWER records...")
    sotwis_indices = []
    
    # This loop finds which SOTWIS box each NASA point belongs to
    for i in range(len(nasa_coords)):
        lat, lon = nasa_coords[i]
        
        # Use a vectorized boolean mask to find the matching box
        # This is much faster than a nested loop
        matches = (
            (lat >= sotwis_bounds[:, 0]) & (lat <= sotwis_bounds[:, 1]) &
            (lon >= sotwis_bounds[:, 2]) & (lon <= sotwis_bounds[:, 3])
        )
        
        # Find the index of the first match
        match_indices = np.where(matches)[0]
        
        if len(match_indices) > 0:
            sotwis_indices.append(match_indices[0])
        else:
            sotwis_indices.append(np.nan) # Mark as NaN if no box found

    nasa_df['sotwis_index'] = sotwis_indices
    
    # Drop NASA points that didn't fall into any SOTWIS box
    initial_rows = len(nasa_df)
    nasa_df.dropna(subset=['sotwis_index'], inplace=True)
    nasa_df['sotwis_index'] = nasa_df['sotwis_index'].astype(int)
    print(f"Join complete. {len(nasa_df)} of {initial_rows} records matched.")
    
    # --- Merge DataFrames ---
    # Merge using the found indices
    # We drop the SOTWIS lat/lon columns to avoid redundancy
    sotwis_to_merge = sotwis_df.drop(columns=['BOTTOM_LEFT_LAT', 'UPPER_RIGHT_LAT', 'BOTTOM_LEFT_LON', 'UPPER_RIGHT_LON'])
    merged_df = nasa_df.merge(sotwis_to_merge, left_on='sotwis_index', right_index=True)
    
    # Clean up final dataframe
    merged_df.drop(columns=['sotwis_index'], inplace=True)

    # --- Save ---
    merged_df.to_csv(output_path, index=False)
    print(f"✅ Integrated dataset created successfully and saved to '{output_path}'")

if __name__ == '__main__':
    merge_datasets()