import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import os
import warnings

# Point to the public API location for the warning in modern pandas versions.
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

def spatial_imputation_advanced():
    """
    Performs an advanced two-pass spatial imputation on the SOTWIS dataset.

    Methodology:
    1.  Calculates centroids and performs DBSCAN clustering.
    2.  Pass 1 (Within Clusters): Imputes missing values by **randomly sampling** from other
        non-null values within the same cluster. This preserves local variance.
    3.  Pass 2 (Remaining Nulls): Imputes values using an **Inverse Distance Weighted (IDW)**
        average of the k-nearest neighbors. This creates realistic spatial gradients.
    """
    print("--- Starting Advanced Spatial Imputation Process ---")

    # --- 0. Configuration ---
    DBSCAN_RADIUS_KM = 75
    DBSCAN_MIN_SAMPLES = 10
    N_NEIGHBORS_FOR_IDW = 5

    # --- 1. File Paths and Data Loading ---
    print("Step 1: Loading and preparing data...")
    try:
        script_dir = os.path.dirname(__file__)
        input_path = os.path.join(script_dir, '..', 'data', 'intermediary_steps', 'complete_sotwis_preprocessed.csv')
        output_path = os.path.join(script_dir, '..', 'data', 'intermediary_steps', 'synthetically_enhanced_sotwis.csv')
        
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{input_path}'.")
        return

    df['LAT_CENTER'] = (df['BOTTOM_LEFT_LAT'] + df['UPPER_RIGHT_LAT']) / 2
    df['LON_CENTER'] = (df['BOTTOM_LEFT_LON'] + df['UPPER_RIGHT_LON']) / 2
    
    feature_cols = df.columns.drop([
        'BOTTOM_LEFT_LAT', 'BOTTOM_LEFT_LON', 'UPPER_RIGHT_LAT', 'UPPER_RIGHT_LON',
        'LAT_CENTER', 'LON_CENTER'
    ]).tolist()
    print(f"Data loaded successfully with {len(df)} records.")
    
    # --- 2. DBSCAN Spatial Clustering ---
    print("\nStep 2: Performing DBSCAN spatial clustering...")
    coords = np.radians(df[['LAT_CENTER', 'LON_CENTER']].values)
    kms_per_radian = 6371
    epsilon = DBSCAN_RADIUS_KM / kms_per_radian 
    db = DBSCAN(eps=epsilon, min_samples=DBSCAN_MIN_SAMPLES, algorithm='ball_tree', metric='haversine').fit(coords)
    df['cluster'] = db.labels_
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"DBSCAN found {n_clusters} clusters.")

    # --- 3. Pass 1: Random Sampling Imputation ---
    print(f"\nStep 3: Imputation Pass 1 (Random Sampling within clusters)...")
    cols_with_missing = df[feature_cols].columns[df[feature_cols].isnull().any()].tolist()
    print(f"Columns to impute: {cols_with_missing}")
    
    def random_impute(series):
        valid_samples = series.dropna()
        if valid_samples.empty:
            return series
        is_null = series.isnull()
        imputed_values = np.random.choice(valid_samples, size=is_null.sum())
        series[is_null] = imputed_values
        return series

    if cols_with_missing:
        imputable_df = df[df['cluster'] != -1]
        imputed_df = imputable_df.groupby('cluster')[cols_with_missing].transform(random_impute)
        df.update(imputed_df)
    print("Imputation Pass 1 complete.")

    # --- 4. Pass 2: Inverse Distance Weighting (IDW) Imputation ---
    print(f"\nStep 4: Imputation Pass 2 (IDW with {N_NEIGHBORS_FOR_IDW} nearest neighbors)...")
    
    cols_still_missing = df.columns[df.isnull().any()].tolist()
    
    if not any(c in feature_cols for c in cols_still_missing):
        print("  - No remaining missing values in feature columns. Skipping Pass 2.")
    else:
        all_coords = df[['LAT_CENTER', 'LON_CENTER']].values
        for col in cols_still_missing:
            if col not in feature_cols: continue

            print(f"  - Processing column '{col}'...")
            has_value_mask = df[col].notna()
            missing_value_mask = df[col].isnull()

            if not has_value_mask.any(): continue
            if not missing_value_mask.any(): continue

            tree = cKDTree(all_coords[has_value_mask])
            distances, nn_indices = tree.query(all_coords[missing_value_mask], k=N_NEIGHBORS_FOR_IDW)
            
            distances += 1e-9
            weights = 1.0 / distances
            
            # --- THE FIX IS HERE ---
            # 1. Get all possible values into a simple 1D NumPy array.
            source_values = df.loc[has_value_mask, col].values
            # 2. Use NumPy's advanced indexing to select the neighbor values. This is robust and fast.
            neighbor_values = source_values[nn_indices]
            # --- END OF FIX ---
            
            imputed_values = np.sum(neighbor_values * weights, axis=1) / np.sum(weights, axis=1)
            df.loc[missing_value_mask, col] = imputed_values
            
        print("Imputation Pass 2 complete.")

    # --- 5. Finalization and Saving ---
    print("\nStep 5: Finalizing and saving the dataset...")
    df_final = df.drop(columns=['LAT_CENTER', 'LON_CENTER', 'cluster'])
    remaining_nulls = df_final[feature_cols].isnull().sum().sum()
    if remaining_nulls == 0:
        print("Verification successful: No missing values remain.")
    else:
        print(f"WARNING: {remaining_nulls} missing values still remain.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"\n✅ Process complete. Synthetically enhanced dataset saved to:\n{output_path}")

if __name__ == '__main__':
    spatial_imputation_advanced()