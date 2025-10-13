# src/run_comparison_experiments.py (Final Version with SVM Subsampling)

import pandas as pd
import numpy as np
import time
import warnings

# --- Imports ---
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
from scipy.spatial import cKDTree
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---
# File Paths
script_dir = os.path.dirname(__file__)
intermediary_dir = os.path.join(script_dir, '..', 'data', 'intermediary_steps')
final_data_dir = os.path.join(script_dir, '..', 'data')
results_dir = os.path.join(script_dir, '..', 'results')

SOTWIS_PATH = os.path.join(intermediary_dir, 'synthetically_enhanced_sotwis.csv')
NASA_PATH = os.path.join(intermediary_dir, 'complete_nasa_power_data_1990_to_2013.csv')
FUSED_PATH = os.path.join(final_data_dir, 'complete_integrated_dataset_1990_to_2013.csv')
SPEI_PATH = os.path.join(intermediary_dir, 'complete_spei_benchmark.csv')
REPORT_OUTPUT_PATH = os.path.join(results_dir, 'full_experimental_report.md')

# Model Parameters
SPEI_COLUMN_NAME = 'DROUGHT_INDEX' 
CHRONOLOGICAL_SPLIT_RATIO = 0.8
LSTM_SEQUENCE_LENGTH = 14
SVM_SUBSAMPLE_SIZE = 50000 # Max samples to use for the slow SVM model

# --- Helper Functions ---
def categorize_drought_binary(spei_value):
    return 1 if spei_value < -1.0 else 0

def create_sequences(X, y, sequence_length):
    xs, ys = [], []
    for i in range(len(X) - sequence_length):
        xs.append(X[i:(i + sequence_length)])
        ys.append(y[i + sequence_length])
    return np.array(xs), np.array(ys)

def run_model_pipeline(X, y, pipeline_name, results_dict):
    """A standardized pipeline to train and evaluate all models."""
    print(f"\n{'='*20} Running All Models for Pipeline: {pipeline_name} {'='*20}")
    
    split_index = int(len(X) * CHRONOLOGICAL_SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    if len(X_train) == 0: return

    # Robust Scaling
    constant_cols = X_train.columns[X_train.std() == 0]
    variable_cols = X_train.columns[X_train.std() != 0]
    scaler = StandardScaler()
    X_train_scaled_vars = scaler.fit_transform(X_train[variable_cols])
    X_test_scaled_vars = scaler.transform(X_test[variable_cols])
    X_train_processed = np.hstack((X_train_scaled_vars, X_train[constant_cols].values))
    X_test_processed = np.hstack((X_test_scaled_vars, X_test[constant_cols].values))
    
    # --- Static Models ---
    static_models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        # --- FIX: Added verbose=True to show progress ---
        "SVM (RBF Kernel)": SVC(kernel='rbf', random_state=42, class_weight='balanced', verbose=True)
    }
    for name, model in static_models.items():
        print(f"\n--- Training {name} for {pipeline_name} ---")
        
        X_train_to_fit = X_train_processed
        y_train_to_fit = y_train

        # --- FIX: Subsample data if it's the SVM and the dataset is too large ---
        if 'SVM' in name and len(X_train_to_fit) > SVM_SUBSAMPLE_SIZE:
            print(f"Dataset is large ({len(X_train_to_fit)} samples). Subsampling to {SVM_SUBSAMPLE_SIZE} for SVM.")
            # Use pandas sampling on original data to keep alignment
            sample_indices = X_train.sample(n=SVM_SUBSAMPLE_SIZE, random_state=42).index
            # Get the corresponding rows from the processed numpy array
            original_indices_list = X_train.index.tolist()
            sample_mask = [original_indices_list.index(i) for i in sample_indices]
            
            X_train_to_fit = X_train_processed[sample_mask]
            y_train_to_fit = y_train.loc[sample_indices]

        model.fit(X_train_to_fit, y_train_to_fit)
        y_pred = model.predict(X_test_processed)
        p = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        r = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        results_dict[name][pipeline_name] = {'precision': p, 'recall': r, 'f1-score': f1}
        print(f"Results for {name}: Precision={p:.4f}, Recall={r:.4f}, F1-Score={f1:.4f}")

    # --- LSTM Model ---
    # (The rest of the script is unchanged)
    # ...
    print(f"\n--- Training LSTM for {pipeline_name} ---")
    X_train_seq, y_train_seq = create_sequences(X_train_processed, y_train.values, LSTM_SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_processed, y_test.values, LSTM_SEQUENCE_LENGTH)

    if X_train_seq.shape[0] > 0:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            Dropout(0.2), LSTM(50, return_sequences=False), Dropout(0.2),
            Dense(25, activation='relu'), Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_seq), y=y_train_seq)
        model.fit(X_train_seq, y_train_seq, epochs=15, batch_size=64, 
                  callbacks=[EarlyStopping(monitor='loss', patience=3)], 
                  class_weight=dict(enumerate(weights)), verbose=0)
        y_pred_prob = model.predict(X_test_seq, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        p = precision_score(y_test_seq, y_pred, pos_label=1, zero_division=0)
        r = recall_score(y_test_seq, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test_seq, y_pred, pos_label=1, zero_division=0)
        results_dict["LSTM"][pipeline_name] = {'precision': p, 'recall': r, 'f1-score': f1}
        print(f"Results for LSTM: Precision={p:.4f}, Recall={r:.4f}, F1-Score={f1:.4f}")

# --- Data Preparation Pipelines (Unchanged) ---
# ... (all three `prepare_..._data` functions are correct and remain the same) ...
def prepare_sotwis_only_data():
    """Prepares the SOTWIS-only dataset by merging with SPEI efficiently."""
    print("\n{'='*20} Preparing Pipeline A: SOTWIS-Only {'='*20}")
    sotwis_df = pd.read_csv(SOTWIS_PATH).reset_index().rename(columns={'index': 'sotwis_index'})
    spei_df = pd.read_csv(SPEI_PATH)
    
    print("Finding nearest SPEI point for each SOTWIS polygon for each month...")
    sotwis_df['LAT'] = (sotwis_df['BOTTOM_LEFT_LAT'] + sotwis_df['UPPER_RIGHT_LAT']) / 2
    sotwis_df['LON'] = (sotwis_df['BOTTOM_LEFT_LON'] + sotwis_df['UPPER_RIGHT_LON']) / 2

    monthly_results = []
    spei_groups = spei_df.groupby(['YEAR', 'MO'])
    sotwis_coords = sotwis_df[['LAT', 'LON']].values
    sotwis_indices = sotwis_df['sotwis_index'].values

    for (year, month), spei_group in spei_groups:
        kdtree = cKDTree(spei_group[['LAT', 'LON']].values)
        _, indices = kdtree.query(sotwis_coords, k=1)
        
        matched_spei_values = spei_group.iloc[indices][SPEI_COLUMN_NAME].values
        
        month_df = pd.DataFrame({
            'sotwis_index': sotwis_indices, 'YEAR': year, 'MO': month,
            SPEI_COLUMN_NAME: matched_spei_values
        })
        monthly_results.append(month_df)
        
    all_months_df = pd.concat(monthly_results, ignore_index=True)
    df_full = pd.merge(sotwis_df, all_months_df, on='sotwis_index')
    df_full['is_drought'] = df_full[SPEI_COLUMN_NAME].apply(categorize_drought_binary)
    
    date_df = df_full[['YEAR', 'MO']].copy()
    date_df.rename(columns={'YEAR': 'year', 'MO': 'month'}, inplace=True)
    date_df['day'] = 1
    df_full['date'] = pd.to_datetime(date_df)
    df_full = df_full.sort_values(by='date').reset_index(drop=True)

    features = [col for col in sotwis_df.columns if col not in ['sotwis_index', 'LAT', 'LON', 'BOTTOM_LEFT_LAT', 'UPPER_RIGHT_LAT', 'BOTTOM_LEFT_LON', 'UPPER_RIGHT_LON']]
    return df_full[features], df_full['is_drought']
    
def prepare_nasa_only_data():
    """Prepares the NASA-only dataset by aggregating, filling NaNs, and merging with SPEI."""
    print("\n{'='*20} Preparing Pipeline B: NASA-Only {'='*20}")
    nasa_df = pd.read_csv(NASA_PATH)
    spei_df = pd.read_csv(SPEI_PATH)
    
    dynamic_features = ['GWETPROF', 'GWETROOT', 'GWETTOP']
    aggregation_rules = {feat: ['mean', 'min', 'max', 'std'] for feat in dynamic_features}
    nasa_monthly = nasa_df.groupby(['YEAR', 'MO', 'LAT', 'LON']).agg(aggregation_rules).reset_index()
    nasa_monthly.columns = ['_'.join(col).strip() for col in nasa_monthly.columns.values]
    nasa_monthly.rename(columns={'YEAR_': 'YEAR', 'MO_': 'MO', 'LAT_': 'LAT', 'LON_': 'LON'}, inplace=True)
    nasa_monthly.fillna(0, inplace=True)

    merged_dfs = []
    nasa_groups = nasa_monthly.groupby(['YEAR', 'MO'])
    spei_groups = spei_df.groupby(['YEAR', 'MO'])

    for (year, month), nasa_group in nasa_groups:
        try:
            spei_group = spei_groups.get_group((year, month))
            kdtree = cKDTree(spei_group[['LAT', 'LON']].values)
            _, indices = kdtree.query(nasa_group[['LAT', 'LON']].values, k=1)
            matched_spei = spei_group.iloc[indices].reset_index(drop=True)
            nasa_group_copy = nasa_group.copy().reset_index(drop=True)
            nasa_group_copy[SPEI_COLUMN_NAME] = matched_spei[SPEI_COLUMN_NAME]
            merged_dfs.append(nasa_group_copy)
        except KeyError: continue
            
    df_full = pd.concat(merged_dfs, ignore_index=True)
    df_full['is_drought'] = df_full[SPEI_COLUMN_NAME].apply(categorize_drought_binary)
    
    date_df = df_full[['YEAR', 'MO']].copy()
    date_df.rename(columns={'YEAR': 'year', 'MO': 'month'}, inplace=True)
    date_df['day'] = 1
    df_full['date'] = pd.to_datetime(date_df)
    df_full = df_full.sort_values(by='date').reset_index(drop=True)
    
    features = [col for col in df_full.columns if col not in ['YEAR', 'MO', 'LAT', 'LON', 'is_drought', SPEI_COLUMN_NAME, 'date']]
    return df_full[features], df_full['is_drought']
    
def prepare_fused_data():
    """Prepares the Fused dataset by aggregating, filling NaNs, and merging with SPEI."""
    print("\n{'='*20} Preparing Pipeline C: Fused Dataset {'='*20}")
    fused_df = pd.read_csv(FUSED_PATH)
    spei_df = pd.read_csv(SPEI_PATH)
    
    dynamic_features = ['GWETPROF', 'GWETROOT', 'GWETTOP']
    static_features = [col for col in fused_df.columns if col not in dynamic_features + ['YEAR', 'MO', 'DY', 'LAT', 'LON']]
    aggregation_rules = {feat: ['mean', 'min', 'max', 'std'] for feat in dynamic_features}
    aggregation_rules.update({feat: 'first' for feat in static_features})
    
    fused_monthly = fused_df.groupby(['YEAR', 'MO', 'LAT', 'LON']).agg(aggregation_rules).reset_index()
    fused_monthly.columns = ['_'.join(col).strip() for col in fused_monthly.columns.values]
    fused_monthly.rename(columns={'YEAR_': 'YEAR', 'MO_': 'MO', 'LAT_': 'LAT', 'LON_': 'LON'}, inplace=True)
    fused_monthly.fillna(0, inplace=True)
    
    merged_dfs = []
    fused_groups = fused_monthly.groupby(['YEAR', 'MO'])
    spei_groups = spei_df.groupby(['YEAR', 'MO'])

    for (year, month), fused_group in fused_groups:
        try:
            spei_group = spei_groups.get_group((year, month))
            kdtree = cKDTree(spei_group[['LAT', 'LON']].values)
            _, indices = kdtree.query(fused_group[['LAT', 'LON']].values, k=1)
            matched_spei = spei_group.iloc[indices].reset_index(drop=True)
            fused_group_copy = fused_group.copy().reset_index(drop=True)
            fused_group_copy[SPEI_COLUMN_NAME] = matched_spei[SPEI_COLUMN_NAME]
            merged_dfs.append(fused_group_copy)
        except KeyError: continue
            
    df_full = pd.concat(merged_dfs, ignore_index=True)
    df_full['is_drought'] = df_full[SPEI_COLUMN_NAME].apply(categorize_drought_binary)
    
    date_df = df_full[['YEAR', 'MO']].copy()
    date_df.rename(columns={'YEAR': 'year', 'MO': 'month'}, inplace=True)
    date_df['day'] = 1
    df_full['date'] = pd.to_datetime(date_df)
    df_full = df_full.sort_values(by='date').reset_index(drop=True)
    
    features = [col for col in df_full.columns if col not in ['YEAR', 'MO', 'LAT', 'LON', 'is_drought', SPEI_COLUMN_NAME, 'date']]
    return df_full[features], df_full['is_drought']

# --- Main Execution ---
if __name__ == '__main__':
    model_names = ["Random Forest", "SVM (RBF Kernel)", "LSTM"]
    pipeline_names = ["SOTWIS-Only", "NASA-Only", "Fused (SOTWIS + NASA)"]
    results = {name: {p_name: {} for p_name in pipeline_names} for name in model_names}
    
    X_sotwis, y_sotwis = prepare_sotwis_only_data()
    run_model_pipeline(X_sotwis, y_sotwis, "SOTWIS-Only", results)
    
    X_nasa, y_nasa = prepare_nasa_only_data()
    run_model_pipeline(X_nasa, y_nasa, "NASA-Only", results)
    
    X_fused, y_fused = prepare_fused_data()
    run_model_pipeline(X_fused, y_fused, "Fused (SOTWIS + NASA)", results)

    # --- Generate and Save Final Report ---
    print("\n\n{'='*20} FINAL COMPREHENSIVE REPORT {'='*20}")
    precision_df = pd.DataFrame({model: {pipe: res.get('precision', 0) for pipe, res in pipes.items()} for model, pipes in results.items()}).T
    recall_df = pd.DataFrame({model: {pipe: res.get('recall', 0) for pipe, res in pipes.items()} for model, pipes in results.items()}).T
    f1_df = pd.DataFrame({model: {pipe: res.get('f1-score', 0) for pipe, res in pipes.items()} for model, pipes in results.items()}).T
    
    report_content = (
        "# Full Experimental Results: Drought Class Prediction\n\n"
        "This report compares the performance of three models across three different feature sets. "
        "All metrics reported are for the minority 'Drought' class (SPEI < -1.0).\n\n"
        "## Table 1: Precision for 'Drought' Class\n"
        "*(What percentage of drought predictions were correct?)*\n\n"
        f"{precision_df.round(4).to_markdown()}\n\n"
        "## Table 2: Recall for 'Drought' Class\n"
        "*(What percentage of actual droughts were found?)*\n\n"
        f"{recall_df.round(4).to_markdown()}\n\n"
        "## Table 3: F1-Score for 'Drought' Class\n"
        "*(The harmonic mean of Precision and Recall, representing a balanced score.)*\n\n"
        f"{f1_df.round(4).to_markdown()}\n"
    )
    print(report_content)
    os.makedirs(results_dir, exist_ok=True)
    with open(REPORT_OUTPUT_PATH, 'w') as f:
        f.write(report_content)
    print(f"\n✅ Full report saved to: {REPORT_OUTPUT_PATH}")