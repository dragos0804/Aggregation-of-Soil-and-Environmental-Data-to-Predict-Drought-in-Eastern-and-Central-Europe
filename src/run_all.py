# src/run_all.py (FINAL, MULTI-CLASS VERSION)

import pandas as pd
import numpy as np
import time
import warnings

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils import class_weight

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

# SciPy for spatial merge
from scipy.spatial import cKDTree

# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning)


# --- Configuration ---
FEATURES_DATA_PATH = '../data/complete_integrated_dataset_1990_to_2013.csv'
TARGET_DATA_PATH = '../data/intermediary_steps/complete_spei_benchmark.csv'
SPEI_COLUMN_NAME = 'DROUGHT_INDEX' 
CHRONOLOGICAL_SPLIT_RATIO = 0.8
LSTM_SEQUENCE_LENGTH = 14
MULTICLASS_REPORT_PATH = '../results/multiclass_experimental_report.md'

# --- Main Script Functions ---

# --- CHANGE 1: Define the multi-class categorization function ---
def categorize_drought(spei_value):
    if spei_value < -2.0:
        return 3  # Extreme Drought
    elif spei_value < -1.5:
        return 2  # Severe Drought
    elif spei_value < -1.0:
        return 1  # Moderate Drought
    else:
        return 0  # No Drought

def load_and_prepare_data():
    """Performs a nearest-neighbor spatial merge and creates a multi-class target."""
    print("--- 1. Data Loading and Preparation (Multi-Class) ---")
    try:
        df_features = pd.read_csv(FEATURES_DATA_PATH)
        df_target_source = pd.read_csv(TARGET_DATA_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure data files are in the correct paths.")
        return None

    print("\nAggregating daily feature data into monthly statistics...")
    dynamic_features = ['GWETPROF', 'GWETROOT', 'GWETTOP']
    static_features = [
        'DRAIN', 'CFRAG', 'SDTO', 'STPC', 'CLPC', 'PSCL', 'BULK', 'TAWC', 'CECS', 
        'BSAT', 'CECC', 'PHAQ', 'TCEQ', 'GYPS', 'ELCO', 'TOTC', 'TOTN', 'ECEC', 
        'ALSA', 'ESP'
    ]
    aggregation_rules = {feat: ['mean', 'min', 'max', 'std'] for feat in dynamic_features}
    aggregation_rules.update({feat: 'first' for feat in static_features})
    
    df_features_monthly = df_features.groupby(['YEAR', 'MO', 'LAT', 'LON']).agg(aggregation_rules).reset_index()
    df_features_monthly.columns = ['_'.join(col).strip() for col in df_features_monthly.columns.values]
    df_features_monthly.rename(columns={'YEAR_': 'YEAR', 'MO_': 'MO', 'LAT_': 'LAT', 'LON_': 'LON'}, inplace=True)
    
    print("\nPerforming nearest neighbor spatial merge...")
    df_features_monthly = df_features_monthly.astype({'YEAR': int, 'MO': int})
    df_target_source = df_target_source.astype({'YEAR': int, 'MO': int})

    merged_dfs = []
    feature_groups = df_features_monthly.groupby(['YEAR', 'MO'])
    target_groups = df_target_source.groupby(['YEAR', 'MO'])

    for (year, month), feature_group in feature_groups:
        try:
            target_group = target_groups.get_group((year, month))
            kdtree = cKDTree(target_group[['LAT', 'LON']].values)
            distances, indices = kdtree.query(feature_group[['LAT', 'LON']].values, k=1)
            matched_targets = target_group.iloc[indices].reset_index(drop=True)
            feature_group_copy = feature_group.copy().reset_index(drop=True)
            feature_group_copy[SPEI_COLUMN_NAME] = matched_targets[SPEI_COLUMN_NAME]
            merged_dfs.append(feature_group_copy)
        except KeyError:
            continue
    
    if not merged_dfs:
        print("CRITICAL: Spatial merge failed.")
        return None

    df_full = pd.concat(merged_dfs, ignore_index=True)
    print(f"Final aligned dataset created with {df_full.shape[0]} rows.")

    # --- CHANGE 2: Apply the categorization to create 'drought_class' ---
    df_full['drought_class'] = df_full[SPEI_COLUMN_NAME].apply(categorize_drought)
    print("\nCreated multi-class target variable 'drought_class'. Distribution:")
    print(df_full['drought_class'].value_counts().sort_index())

    date_df = df_full[['YEAR', 'MO']].copy()
    date_df.rename(columns={'YEAR': 'year', 'MO': 'month'}, inplace=True)
    date_df['day'] = 1
    df_full['date'] = pd.to_datetime(date_df)
    df_full = df_full.sort_values(by=['date', 'LAT', 'LON']).reset_index(drop=True)
    
    return df_full

def get_features_and_target(df):
    """Separates the dataframe into features (X) and multi-class target (y)."""
    print("\n--- 2. Defining Features (X) and Target (y) ---")
    if df is None or df.empty: return None, None
    
    target_col = 'drought_class' # <-- New target column
    id_cols = ['YEAR', 'MO', 'LAT', 'LON', SPEI_COLUMN_NAME, 'date']
    features_list = [col for col in df.columns if col not in id_cols and col != target_col]
    
    print(f"Identified {len(features_list)} aggregated features for the models.")
    X = df[features_list]
    y = df[target_col]

    if X.isnull().values.any():
        print("Warning: Missing values found. Dropping rows with NaNs.")
        na_mask = X.isnull().any(axis=1)
        X = X[~na_mask].reset_index(drop=True)
        y = y[~na_mask].reset_index(drop=True)
    
    return X, y

def build_multiclass_evaluation(y_true, y_pred, target_names):
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))
    return report_dict, matrix

def multiclass_matrix_to_markdown(matrix, target_names):
    matrix_df = pd.DataFrame(
        matrix,
        index=[f'Actual {name}' for name in target_names],
        columns=[f'Pred {name}' for name in target_names]
    )
    return matrix_df.to_markdown()

def multiclass_report_to_markdown(report_dict, target_names):
    rows = []
    for class_name in target_names:
        rows.append({
            'Class': class_name,
            'Precision': report_dict[class_name]['precision'],
            'Recall': report_dict[class_name]['recall'],
            'F1-Score': report_dict[class_name]['f1-score'],
            'Support': int(report_dict[class_name]['support'])
        })
    return pd.DataFrame(rows).round(4).to_markdown(index=False)

def run_static_models(X, y):
    """Trains static models for multi-class classification with balanced weights."""
    print("\n--- 3. Training Static Models (Multi-Class) ---")
    split_index = int(len(X) * CHRONOLOGICAL_SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    target_names = ['No Drought', 'Moderate', 'Severe', 'Extreme']
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        "SVM (RBF Kernel)": SVC(kernel='rbf', random_state=42, class_weight='balanced')
    }
    
    for name, model in models.items():
        print(f"\nTraining {name} with balanced class weights...")
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        end_time = time.time()
        print(f"Evaluation for {name}:")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        # Weighted average is best for imbalanced multi-class summary
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        report_dict, matrix = build_multiclass_evaluation(y_test, y_pred, target_names)
        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Training Time (s)": end_time - start_time,
            "class_report": report_dict,
            "confusion_matrix": matrix.tolist()
        }
    return results

def create_sequences(X, y, sequence_length):
    """Transforms data into sequences."""
    xs, ys = [], []
    for i in range(len(X) - sequence_length):
        xs.append(X[i:(i + sequence_length)])
        ys.append(y[i + sequence_length])
    return np.array(xs), np.array(ys)

def run_lstm_model(X, y, sequence_length):
    """Prepares and evaluates the LSTM using a stratified validation set."""
    print("\n--- 4. Training Temporal Model (Multi-Class with Stratified Validation) ---")
    num_classes = len(np.unique(y))
    target_names = ['No Drought', 'Moderate', 'Severe', 'Extreme']

    # 1. Main Chronological Split (same as before)
    split_index = int(len(X) * CHRONOLOGICAL_SPLIT_RATIO)
    X_train_main, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train_main, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    scaler = StandardScaler()
    X_train_main_scaled = scaler.fit_transform(X_train_main)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Create Sequences from the Main Training and Test Sets
    X_train_main_seq, y_train_main_seq = create_sequences(X_train_main_scaled, y_train_main.values, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, sequence_length)
    
    if X_train_main_seq.shape[0] == 0: return {}

    # --- ENHANCEMENT: Create a Stratified Validation Set from the Main Training Sequences ---
    print(f"\nCreating a stratified validation set from the training data...")
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_main_seq, 
        y_train_main_seq, 
        test_size=0.15,  # e.g., use 15% of the training data for validation
        stratify=y_train_main_seq, # This is the magic!
        random_state=42
    )
    print(f"Sub-training set size: {len(X_train_sub)}")
    print(f"Stratified validation set size: {len(X_val)}")
    # --- END OF ENHANCEMENT ---
        
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train_sub.shape[1], X_train_sub.shape[2])),
        Dropout(0.2), LSTM(50, return_sequences=False), Dropout(0.2),
        Dense(25, activation='relu'), Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_sub), y=y_train_sub)
    class_weight_dict = dict(enumerate(weights))
    print(f"\nUsing class weights for LSTM: {class_weight_dict}")

    print("\nTraining LSTM with stratified validation set...")
    start_time = time.time()
    # Pass the new validation data manually
    model.fit(
        X_train_sub, y_train_sub, 
        epochs=50, batch_size=32,
        validation_data=(X_val, y_val), # <-- Pass the stratified set
        callbacks=[early_stopping],
        class_weight=class_weight_dict,
        verbose=1
    )
    end_time = time.time()
    
    print("\nFinal Evaluation on the Chronological (unseen) Test Set:")
    y_pred_prob = model.predict(X_test_seq)
    y_pred = np.argmax(y_pred_prob, axis=1)
    print(classification_report(y_test_seq, y_pred, target_names=target_names, zero_division=0))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_seq, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test_seq, y_pred)
    report_dict, matrix = build_multiclass_evaluation(y_test_seq, y_pred, target_names)
    return {
        "LSTM": {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Training Time (s)": end_time - start_time,
            "class_report": report_dict,
            "confusion_matrix": matrix.tolist()
        }
    }

# --- Main Execution ---
if __name__ == "__main__":
    full_dataset = load_and_prepare_data()
    X, y = get_features_and_target(full_dataset)

    if X is not None and y is not None:
        static_results = run_static_models(X, y)
        lstm_results = run_lstm_model(X, y, LSTM_SEQUENCE_LENGTH)

        print("\n--- 5. Consolidated Experiment Results ---")
        all_results = {**static_results, **lstm_results}
        summary_results = {
            model_name: {k: v for k, v in metrics.items() if k not in ['class_report', 'confusion_matrix']}
            for model_name, metrics in all_results.items()
        }
        results_df = pd.DataFrame.from_dict(summary_results, orient='index')
        
        if not results_df.empty:
            results_df = results_df.round(4)
            print("\nThe following table can be used in your paper's results section:")
            print(results_df.to_markdown(index=True))

            detailed_sections = []
            target_names = ['No Drought', 'Moderate', 'Severe', 'Extreme']
            for model_name, metrics in all_results.items():
                matrix_md = multiclass_matrix_to_markdown(np.array(metrics['confusion_matrix']), target_names)
                report_md = multiclass_report_to_markdown(metrics['class_report'], target_names)
                detailed_sections.append(
                    f"## {model_name}\n\n"
                    f"Confusion matrix:\n\n{matrix_md}\n\n"
                    f"Per-class metrics:\n\n{report_md}\n"
                )

            report_content = (
                "# Multi-Class Experimental Results\n\n"
                "The table below summarizes the chronological-test performance of the three classifiers for multi-class drought severity prediction.\n\n"
                f"{results_df.to_markdown(index=True)}\n\n"
                "# Detailed Confusion Matrices and Per-Class Metrics\n\n"
                + "\n".join(detailed_sections)
            )
            with open(MULTICLASS_REPORT_PATH, 'w') as report_file:
                report_file.write(report_content)
            print(f"\nDetailed multi-class report saved to: {MULTICLASS_REPORT_PATH}")
    else:
        print("\nHalting execution due to errors in data preparation.")