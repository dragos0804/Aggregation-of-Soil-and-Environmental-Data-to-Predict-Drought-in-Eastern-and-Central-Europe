# src/run_comparison_experiments.py (Final Version with SVM Subsampling)

import pandas as pd
import numpy as np
import time
import warnings

# --- Imports ---
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils import class_weight
from sklearn.inspection import permutation_importance
from scipy.spatial import cKDTree
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sequence_utils import create_location_sequences
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
DATA_START_YEAR = int(os.getenv('DATA_START_YEAR', '1990'))
DATA_END_YEAR = int(os.getenv('DATA_END_YEAR', '2024'))

SOTWIS_PATH = os.path.join(intermediary_dir, 'synthetically_enhanced_sotwis.csv')
NASA_PERIOD_PATH = os.path.join(
    intermediary_dir, f'complete_nasa_power_data_{DATA_START_YEAR}_to_{DATA_END_YEAR}.csv'
)
NASA_EXTENDED_PATH = os.path.join(intermediary_dir, 'complete_nasa_power_data_1990_to_2024.csv')
NASA_PATH = os.getenv(
    'NASA_DATA_PATH',
    NASA_PERIOD_PATH if os.path.exists(NASA_PERIOD_PATH) else NASA_EXTENDED_PATH,
)
FUSED_PERIOD_PATH = os.path.join(
    final_data_dir, f'complete_integrated_dataset_{DATA_START_YEAR}_to_{DATA_END_YEAR}.csv'
)
FUSED_EXTENDED_PATH = os.path.join(final_data_dir, 'complete_integrated_dataset_1990_to_2024.csv')
FUSED_PATH = os.getenv(
    'FUSED_DATA_PATH',
    FUSED_PERIOD_PATH if os.path.exists(FUSED_PERIOD_PATH) else FUSED_EXTENDED_PATH,
)
SPEI_SCALE = os.getenv('SPEI_SCALE', '01').zfill(2)
SPEI_FILENAME = 'complete_spei_benchmark.csv' if SPEI_SCALE == '01' else f'complete_spei{SPEI_SCALE}_benchmark.csv'
SPEI_PATH = os.path.join(intermediary_dir, SPEI_FILENAME)
REPORT_OUTPUT_PATH = os.getenv(
    'COMPARISON_REPORT_OUTPUT_PATH',
    os.path.join(results_dir, 'full_experimental_report.md'),
)
COMMON_SUPPORT_REPORT_PATH = os.getenv(
    'COMMON_SUPPORT_OUTPUT_PATH',
    os.path.join(results_dir, 'common_support_summary.csv'),
)
COMPARISON_METRICS_PATH = os.getenv(
    'COMPARISON_METRICS_OUTPUT_PATH',
    os.path.join(results_dir, 'comparison_metrics.csv'),
)
PERMUTATION_IMPORTANCE_PATH = os.getenv(
    'PERMUTATION_IMPORTANCE_OUTPUT_PATH',
    os.path.join(results_dir, 'rf_permutation_importance.csv'),
)
PERMUTATION_IMPORTANCE_ROWS = []

# Model Parameters
SPEI_COLUMN_NAME = 'DROUGHT_INDEX'
SPEI_THRESHOLD = float(os.getenv('SPEI_THRESHOLD', '-1.0'))
CHRONOLOGICAL_SPLIT_RATIO = 0.8
LSTM_SEQUENCE_LENGTH = int(os.getenv('LSTM_SEQUENCE_LENGTH', '14'))
LSTM_VALIDATION_FRACTION = float(os.getenv('LSTM_VALIDATION_FRACTION', '0.1'))
SVM_SUBSAMPLE_SIZE = int(os.getenv('SVM_SUBSAMPLE_SIZE', '50000'))
SVM_EVALUATION_SIZE = int(os.getenv('SVM_EVALUATION_SIZE', '100000'))
PERMUTATION_IMPORTANCE_SIZE = int(os.getenv('PERMUTATION_IMPORTANCE_SIZE', '10000'))
IMPUTATION_STRATEGY = os.getenv('NASA_IMPUTATION_STRATEGY', 'median_indicators')
EXPERIMENT_SEED = int(os.getenv('EXPERIMENT_SEED', '42'))
ENABLED_MODELS = [name.strip() for name in os.getenv(
    'EXPERIMENT_MODELS', 'Random Forest,SVM (RBF Kernel),LSTM'
).split(',')]
RUN_SOTWIS_ONLY = os.getenv('RUN_SOTWIS_ONLY', 'true').lower() == 'true'
NASA_FEATURE_SET = os.getenv('NASA_FEATURE_SET', 'full')

# --- Helper Functions ---
def categorize_drought_binary(spei_value):
    return 1 if spei_value < SPEI_THRESHOLD else 0

def create_sequences(X, y, sequence_length, dates=None, latitudes=None, longitudes=None):
    """Create sequences only when location and monthly metadata are supplied."""
    if dates is None or latitudes is None or longitudes is None:
        raise ValueError('Location-aware sequences require dates, latitudes, and longitudes')
    sequences = create_location_sequences(
        X, y, dates, latitudes, longitudes, sequence_length
    )
    return sequences['features'], sequences['targets'], sequences

def build_binary_evaluation(y_true, y_pred):
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=['No Drought', 'Drought'],
        zero_division=0,
        output_dict=True
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return report_dict, matrix


def calculate_additional_metrics(y_true, y_pred, scores):
    """Return metrics that remain informative under drought-class imbalance."""
    return {
        'balanced-accuracy': balanced_accuracy_score(y_true, y_pred),
        'pr-auc': average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan,
    }

def matrix_to_markdown(matrix):
    matrix_df = pd.DataFrame(
        matrix,
        index=['Actual No Drought', 'Actual Drought'],
        columns=['Pred No Drought', 'Pred Drought']
    )
    return matrix_df.to_markdown()

def class_report_to_markdown(report_dict):
    class_rows = []
    for label in ['No Drought', 'Drought']:
        class_rows.append({
            'Class': label,
            'Precision': report_dict[label]['precision'],
            'Recall': report_dict[label]['recall'],
            'F1-Score': report_dict[label]['f1-score'],
            'Support': int(report_dict[label]['support'])
        })
    class_df = pd.DataFrame(class_rows)
    return class_df.round(4).to_markdown(index=False)

def impute_train_median_with_indicators(X_train, X_test):
    """Fit imputation on the training period and preserve missingness signals."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    feature_columns = list(X_train.columns)
    missing_columns = [column for column in feature_columns if X_train[column].isna().any() or X_test[column].isna().any()]
    for column in missing_columns:
        indicator = f'{column}__missing'
        X_train[indicator] = X_train[column].isna().astype(np.int8)
        X_test[indicator] = X_test[column].isna().astype(np.int8)
    medians = X_train[feature_columns].median().fillna(0.0)
    X_train[feature_columns] = X_train[feature_columns].fillna(medians)
    X_test[feature_columns] = X_test[feature_columns].fillna(medians)
    return X_train, X_test

def apply_imputation_strategy(X_train, X_test):
    if IMPUTATION_STRATEGY == 'zero':
        return X_train.fillna(0.0), X_test.fillna(0.0)
    if IMPUTATION_STRATEGY == 'median_indicators':
        return impute_train_median_with_indicators(X_train, X_test)
    raise ValueError(f'Unsupported NASA_IMPUTATION_STRATEGY: {IMPUTATION_STRATEGY}')


def filter_data_period(frame):
    """Restrict records with a YEAR column to the configured study window."""
    if 'YEAR' not in frame.columns:
        return frame
    return frame.loc[frame['YEAR'].between(DATA_START_YEAR, DATA_END_YEAR)].copy()


def aggregate_daily_csv_monthly(path, dynamic_features, static_features=None):
    """Aggregate a large daily CSV without retaining all daily rows in memory."""
    static_features = static_features or []
    group_columns = ['YEAR', 'MO', 'LAT', 'LON']
    chunk_size = int(os.getenv('MONTHLY_AGGREGATION_CHUNK_SIZE', '500000'))
    cache_enabled = os.getenv('MONTHLY_AGGREGATION_CACHE', 'true').lower() == 'true'
    cache_dir = os.getenv(
        'MONTHLY_AGGREGATION_CACHE_DIR',
        os.path.join(intermediary_dir, 'monthly_cache'),
    )
    source_stat = os.stat(path)
    cache_path = os.path.join(cache_dir, f'{os.path.basename(path)}.monthly.pkl')
    expected_metadata = {
        'cache_version': 2,
        'source_size': source_stat.st_size,
        'source_mtime_ns': source_stat.st_mtime_ns,
        'data_start_year': DATA_START_YEAR,
        'data_end_year': DATA_END_YEAR,
        'dynamic_features': list(dynamic_features),
        'static_features': list(static_features),
    }
    if cache_enabled and os.path.exists(cache_path):
        payload = pd.read_pickle(cache_path)
        if all(payload.get(key) == value for key, value in expected_metadata.items()):
            print(f'Loaded monthly aggregation cache for {os.path.basename(path)}')
            return payload['frame']

    partial_frames = []

    for chunk_number, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size), start=1):
        chunk = filter_data_period(chunk)
        if chunk.empty:
            continue
        grouped = chunk.groupby(group_columns, sort=False, dropna=False)
        partial = grouped[dynamic_features].agg(['count', 'sum', 'min', 'max'])
        partial.columns = [f'{feature}__{stat}' for feature, stat in partial.columns]
        for feature in dynamic_features:
            squared = chunk[feature].pow(2)
            partial[f'{feature}__sum_squared'] = squared.groupby(
                [chunk[column] for column in group_columns],
                sort=False,
                dropna=False,
            ).sum().values
        if static_features:
            static_partial = grouped[static_features].first()
            static_partial.columns = [f'{column}__first' for column in static_partial.columns]
            partial = partial.join(static_partial)
        partial_frames.append(partial.reset_index())
        print(f'Aggregated daily chunk {chunk_number} from {os.path.basename(path)}')

    if not partial_frames:
        return pd.DataFrame(columns=group_columns)

    partial_data = pd.concat(partial_frames, ignore_index=True)
    combined_groups = partial_data.groupby(group_columns, sort=False, dropna=False)
    monthly = combined_groups.size().to_frame('_partial_count')

    for feature in dynamic_features:
        count = combined_groups[f'{feature}__count'].sum()
        total = combined_groups[f'{feature}__sum'].sum()
        sum_squared = combined_groups[f'{feature}__sum_squared'].sum()
        monthly[f'{feature}_mean'] = total / count.replace(0, np.nan)
        monthly[f'{feature}_min'] = combined_groups[f'{feature}__min'].min()
        monthly[f'{feature}_max'] = combined_groups[f'{feature}__max'].max()
        variance_numerator = sum_squared - total.pow(2) / count.replace(0, np.nan)
        monthly[f'{feature}_std'] = np.sqrt(
            variance_numerator.clip(lower=0) / (count - 1).where(count > 1)
        )

    for feature in static_features:
        monthly[feature] = combined_groups[f'{feature}__first'].first()

    monthly = monthly.drop(columns=['_partial_count']).reset_index()
    if cache_enabled:
        os.makedirs(cache_dir, exist_ok=True)
        temporary_path = f'{cache_path}.{os.getpid()}.tmp'
        pd.to_pickle({**expected_metadata, 'frame': monthly}, temporary_path)
        os.replace(temporary_path, cache_path)
        print(f'Saved monthly aggregation cache for {os.path.basename(path)}')
    return monthly


def load_spei_data():
    """Load only benchmark months inside the configured study window."""
    spei_df = pd.read_csv(SPEI_PATH)
    return filter_data_period(spei_df)

def add_next_month_targets(frame):
    """Attach the next-month drought label without crossing locations."""
    frame = frame.copy()
    frame = frame.sort_values(by=['LAT', 'LON', 'date']).reset_index(drop=True)
    location_groups = frame.groupby(['LAT', 'LON'], sort=False)
    frame['target_drought'] = location_groups['is_drought'].shift(-1)
    frame['target_date'] = location_groups['date'].shift(-1)
    expected_target_date = frame['date'] + pd.offsets.MonthBegin(1)
    valid_target = frame['target_date'].eq(expected_target_date)
    frame.loc[~valid_target, 'target_drought'] = np.nan
    frame.loc[~valid_target, 'target_date'] = pd.NaT
    return frame


def run_model_pipeline(frame, feature_columns, pipeline_name, results_dict):
    """A standardized pipeline to train and evaluate all models."""
    print(f"\n{'='*20} Running All Models for Pipeline: {pipeline_name} {'='*20}")

    frame = add_next_month_targets(frame)
    valid_rows = frame['target_drought'].notna()
    frame = frame.loc[valid_rows].copy()
    if frame.empty:
        return

    unique_target_dates = np.sort(frame['target_date'].unique())
    split_index = max(1, int(len(unique_target_dates) * CHRONOLOGICAL_SPLIT_RATIO))
    split_index = min(split_index, len(unique_target_dates) - 1)
    split_date = unique_target_dates[split_index]
    train_rows = frame['target_date'] < split_date
    test_rows = ~train_rows
    X_train = frame.loc[train_rows, feature_columns]
    X_test = frame.loc[test_rows, feature_columns]
    y_train = frame.loc[train_rows, 'target_drought'].astype(int)
    y_test = frame.loc[test_rows, 'target_drought'].astype(int)
    
    if len(X_train) == 0: return

    X_train, X_test = apply_imputation_strategy(X_train, X_test)

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
        if name not in ENABLED_MODELS:
            continue
        print(f"\n--- Training {name} for {pipeline_name} ---")
        
        X_train_to_fit = X_train_processed
        y_train_to_fit = y_train

        # --- FIX: Subsample data if it's the SVM and the dataset is too large ---
        if 'SVM' in name and len(X_train_to_fit) > SVM_SUBSAMPLE_SIZE:
            print(f"Dataset is large ({len(X_train_to_fit)} samples). Subsampling to {SVM_SUBSAMPLE_SIZE} for SVM.")
            # Use pandas sampling on original data to keep alignment
            sample_indices = X_train.sample(n=SVM_SUBSAMPLE_SIZE, random_state=42).index
            # Map sampled labels to processed-array positions without quadratic lookup.
            sample_mask = X_train.index.get_indexer(sample_indices)
            
            X_train_to_fit = X_train_processed[sample_mask]
            y_train_to_fit = y_train.loc[sample_indices]

        model.fit(X_train_to_fit, y_train_to_fit)
        X_eval = X_test_processed
        y_eval = y_test
        if 'SVM' in name and len(X_eval) > SVM_EVALUATION_SIZE:
            eval_indices = y_test.sample(n=SVM_EVALUATION_SIZE, random_state=42).index
            eval_mask = y_test.index.get_indexer(eval_indices)
            X_eval = X_test_processed[eval_mask]
            y_eval = y_test.loc[eval_indices]
            print(f"Evaluating SVM on deterministic sample of {SVM_EVALUATION_SIZE} test rows.")
        y_pred = model.predict(X_eval)
        if hasattr(model, 'decision_function'):
            prediction_scores = model.decision_function(X_eval)
        elif hasattr(model, 'predict_proba'):
            prediction_scores = model.predict_proba(X_eval)[:, 1]
        else:
            prediction_scores = y_pred
        p = precision_score(y_eval, y_pred, pos_label=1, zero_division=0)
        r = recall_score(y_eval, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_eval, y_pred, pos_label=1, zero_division=0)
        report_dict, matrix = build_binary_evaluation(y_eval, y_pred)
        results_dict[name][pipeline_name] = {
            'precision': p,
            'recall': r,
            'f1-score': f1,
            'weighted-f1': report_dict['weighted avg']['f1-score'],
            **calculate_additional_metrics(y_eval, y_pred, prediction_scores),
            'class_report': report_dict,
            'confusion_matrix': matrix.tolist()
        }
        if name == 'Random Forest':
            importance_size = min(PERMUTATION_IMPORTANCE_SIZE, len(X_test_processed))
            importance_rng = np.random.default_rng(EXPERIMENT_SEED)
            importance_indices = importance_rng.choice(
                len(X_test_processed),
                size=importance_size,
                replace=False,
            )
            importance = permutation_importance(
                model,
                X_test_processed[importance_indices],
                y_test.iloc[importance_indices],
                scoring='f1',
                n_repeats=5,
                random_state=42,
                n_jobs=1,
            )
            processed_feature_names = list(variable_cols) + list(constant_cols)
            for feature_name, mean_importance, std_importance in zip(
                processed_feature_names,
                importance.importances_mean,
                importance.importances_std,
            ):
                PERMUTATION_IMPORTANCE_ROWS.append({
                    'pipeline': pipeline_name,
                    'model': name,
                    'feature': feature_name,
                    'importance_mean': mean_importance,
                    'importance_std': std_importance,
                })
        print(f"Results for {name}: Precision={p:.4f}, Recall={r:.4f}, F1-Score={f1:.4f}")

    # --- Location-aware LSTM Model ---
    if "LSTM" not in ENABLED_MODELS:
        return

    print(f"\n--- Training LSTM for {pipeline_name} ---")
    all_train_features, all_test_features = apply_imputation_strategy(
        frame.loc[train_rows, feature_columns],
        frame.loc[test_rows, feature_columns],
    )
    all_imputed = pd.concat([all_train_features, all_test_features]).loc[frame.index]
    all_processed = np.hstack(
        (
            scaler.transform(all_imputed[variable_cols]),
            all_imputed[constant_cols].values,
        )
    )
    all_dates = frame['date'].to_numpy(dtype='datetime64[M]')
    sequence_result = create_location_sequences(
        all_processed,
        frame['is_drought'].to_numpy(),
        all_dates,
        frame['LAT'].to_numpy(),
        frame['LON'].to_numpy(),
        LSTM_SEQUENCE_LENGTH,
    )
    sequence_train = sequence_result['target_dates'] < split_date
    sequence_test = ~sequence_train
    X_train_seq = sequence_result['features'][sequence_train]
    y_train_seq = sequence_result['targets'][sequence_train]
    X_test_seq = sequence_result['features'][sequence_test]
    y_test_seq = sequence_result['targets'][sequence_test]
    train_target_dates = sequence_result['target_dates'][sequence_train]

    if X_train_seq.shape[0] > 0:
        # Last training months form a chronological validation segment for early stopping.
        unique_train_dates = np.unique(train_target_dates)
        validation_start = unique_train_dates[int(len(unique_train_dates) * (1 - LSTM_VALIDATION_FRACTION))]
        fit_rows = train_target_dates < validation_start
        X_fit_seq, y_fit_seq = X_train_seq[fit_rows], y_train_seq[fit_rows]
        X_val_seq, y_val_seq = X_train_seq[~fit_rows], y_train_seq[~fit_rows]
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(EXPERIMENT_SEED)
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_fit_seq.shape[1], X_fit_seq.shape[2])),
            Dropout(0.2), LSTM(50, return_sequences=False), Dropout(0.2),
            Dense(25, activation='relu'), Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_fit_seq), y=y_fit_seq)
        history = model.fit(X_fit_seq, y_fit_seq, epochs=15, batch_size=64,
                  validation_data=(X_val_seq, y_val_seq),
                  callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
                  class_weight=dict(enumerate(weights)), verbose=0)
        print(f"LSTM stopped after {len(history.history['loss'])} epochs "
              f"(best val_loss {min(history.history['val_loss']):.4f}).")
        y_pred_prob = model.predict(X_test_seq, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        p = precision_score(y_test_seq, y_pred, pos_label=1, zero_division=0)
        r = recall_score(y_test_seq, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test_seq, y_pred, pos_label=1, zero_division=0)
        report_dict, matrix = build_binary_evaluation(y_test_seq, y_pred)
        results_dict["LSTM"][pipeline_name] = {
            'precision': p,
            'recall': r,
            'f1-score': f1,
            'weighted-f1': report_dict['weighted avg']['f1-score'],
            **calculate_additional_metrics(y_test_seq, y_pred, y_pred_prob.flatten()),
            'class_report': report_dict,
            'confusion_matrix': matrix.tolist()
        }
        print(f"Results for LSTM: Precision={p:.4f}, Recall={r:.4f}, F1-Score={f1:.4f}")

# --- Data Preparation Pipelines (Unchanged) ---
# ... (all three `prepare_..._data` functions are correct and remain the same) ...
def prepare_sotwis_only_data():
    """Prepares the SOTWIS-only dataset by merging with SPEI efficiently."""
    print("\n{'='*20} Preparing Pipeline A: SOTWIS-Only {'='*20}")
    sotwis_df = pd.read_csv(SOTWIS_PATH).reset_index().rename(columns={'index': 'sotwis_index'})
    spei_df = load_spei_data()
    
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
    return df_full, features
    
def prepare_nasa_only_data():
    """Prepares the NASA-only dataset by aggregating, filling NaNs, and merging with SPEI."""
    print("\n{'='*20} Preparing Pipeline B: NASA-Only {'='*20}")
    spei_df = load_spei_data()
    
    dynamic_features = ['GWETPROF', 'GWETROOT', 'GWETTOP']
    nasa_monthly = aggregate_daily_csv_monthly(NASA_PATH, dynamic_features)
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
    if NASA_FEATURE_SET == 'means_only':
        features = [f'{feature}_mean' for feature in dynamic_features]
    elif NASA_FEATURE_SET != 'full':
        raise ValueError(f'Unsupported NASA_FEATURE_SET: {NASA_FEATURE_SET}')
    return df_full, features
    
def prepare_fused_data():
    """Prepares the Fused dataset by aggregating, filling NaNs, and merging with SPEI."""
    print("\n{'='*20} Preparing Pipeline C: Fused Dataset {'='*20}")
    spei_df = load_spei_data()
    
    dynamic_features = ['GWETPROF', 'GWETROOT', 'GWETTOP']
    fused_columns = pd.read_csv(FUSED_PATH, nrows=0).columns.tolist()
    static_features = [
        column for column in fused_columns
        if column not in dynamic_features + ['YEAR', 'MO', 'DY', 'LAT', 'LON']
    ]
    fused_monthly = aggregate_daily_csv_monthly(
        FUSED_PATH,
        dynamic_features,
        static_features,
    )
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
    if NASA_FEATURE_SET == 'means_only':
        dynamic_summary_features = [
            f'{feature}_{stat}'
            for feature in dynamic_features
            for stat in ['mean', 'min', 'max', 'std']
        ]
        static_model_features = [
            feature for feature in features if feature not in dynamic_summary_features
        ]
        features = static_model_features + [f'{feature}_mean' for feature in dynamic_features]
    elif NASA_FEATURE_SET != 'full':
        raise ValueError(f'Unsupported NASA_FEATURE_SET: {NASA_FEATURE_SET}')
    return df_full, features


def restrict_to_common_support(nasa_frame, fused_frame):
    """Restrict the primary dynamic comparison to identical location-month keys."""
    key_columns = ['YEAR', 'MO', 'LAT', 'LON']
    common_keys = nasa_frame[key_columns].drop_duplicates().merge(
        fused_frame[key_columns].drop_duplicates(),
        on=key_columns,
        how='inner',
    )
    common_nasa = nasa_frame.merge(common_keys, on=key_columns, how='inner')
    common_fused = fused_frame.merge(common_keys, on=key_columns, how='inner')

    summary_rows = []
    for name, original, common in [
        ('NASA-Only', nasa_frame, common_nasa),
        ('Fused', fused_frame, common_fused),
    ]:
        summary_rows.append({
            'pipeline': name,
            'original_rows': len(original),
            'common_support_rows': len(common),
            'original_locations': original[['LAT', 'LON']].drop_duplicates().shape[0],
            'common_support_locations': common[['LAT', 'LON']].drop_duplicates().shape[0],
            'drought_prevalence_original': original['is_drought'].mean(),
            'drought_prevalence_common_support': common['is_drought'].mean(),
        })
    pd.DataFrame(summary_rows).to_csv(COMMON_SUPPORT_REPORT_PATH, index=False)
    return common_nasa, common_fused

# --- Main Execution ---
if __name__ == '__main__':
    tf.keras.utils.set_random_seed(EXPERIMENT_SEED)
    tf.config.experimental.enable_op_determinism()
    model_names = ENABLED_MODELS
    pipeline_names = [
        "SOTWIS-Only (separate static susceptibility experiment)",
        "NASA-Only (common support)",
        "Fused (SOTWIS + NASA, common support)",
    ]
    results = {name: {p_name: {} for p_name in pipeline_names} for name in model_names}
    
    if RUN_SOTWIS_ONLY:
        sotwis_frame, sotwis_features = prepare_sotwis_only_data()
        run_model_pipeline(
            sotwis_frame,
            sotwis_features,
            "SOTWIS-Only (separate static susceptibility experiment)",
            results,
        )

    nasa_frame, nasa_features = prepare_nasa_only_data()
    fused_frame, fused_features = prepare_fused_data()
    nasa_frame, fused_frame = restrict_to_common_support(nasa_frame, fused_frame)
    run_model_pipeline(nasa_frame, nasa_features, "NASA-Only (common support)", results)
    run_model_pipeline(
        fused_frame,
        fused_features,
        "Fused (SOTWIS + NASA, common support)",
        results,
    )

    # --- Generate and Save Final Report ---
    print("\n\n{'='*20} FINAL COMPREHENSIVE REPORT {'='*20}")
    precision_df = pd.DataFrame({model: {pipe: res.get('precision', 0) for pipe, res in pipes.items()} for model, pipes in results.items()}).T
    recall_df = pd.DataFrame({model: {pipe: res.get('recall', 0) for pipe, res in pipes.items()} for model, pipes in results.items()}).T
    f1_df = pd.DataFrame({model: {pipe: res.get('f1-score', 0) for pipe, res in pipes.items()} for model, pipes in results.items()}).T
    weighted_f1_df = pd.DataFrame({model: {pipe: res.get('weighted-f1', 0) for pipe, res in pipes.items()} for model, pipes in results.items()}).T
    avg_drought_f1_df = pd.DataFrame({'Average Drought F1-Score': f1_df.mean(axis=0).round(4)})
    avg_weighted_f1_df = pd.DataFrame({'Average Weighted F1-Score': weighted_f1_df.mean(axis=0).round(4)})
    metric_rows = []
    for model_name, pipeline_results in results.items():
        for pipeline_name, metrics in pipeline_results.items():
            if metrics:
                metric_rows.append({
                    'spei_scale': SPEI_SCALE,
                    'spei_threshold': SPEI_THRESHOLD,
                    'imputation_strategy': IMPUTATION_STRATEGY,
                    'nasa_feature_set': NASA_FEATURE_SET,
                    'lstm_sequence_length': LSTM_SEQUENCE_LENGTH,
                    'model': model_name,
                    'pipeline': pipeline_name,
                    'precision_drought': metrics['precision'],
                    'recall_drought': metrics['recall'],
                    'f1_drought': metrics['f1-score'],
                    'pr_auc': metrics['pr-auc'],
                    'balanced_accuracy': metrics['balanced-accuracy'],
                    'weighted_f1': metrics['weighted-f1'],
                })
    pd.DataFrame(metric_rows).to_csv(COMPARISON_METRICS_PATH, index=False)
    if PERMUTATION_IMPORTANCE_ROWS:
        pd.DataFrame(PERMUTATION_IMPORTANCE_ROWS).sort_values(
            ['pipeline', 'importance_mean'], ascending=[True, False]
        ).to_csv(PERMUTATION_IMPORTANCE_PATH, index=False)

    detailed_sections = []
    for model_name, pipeline_results in results.items():
        for pipeline_name, metrics in pipeline_results.items():
            if not metrics:
                continue
            matrix = np.array(metrics['confusion_matrix'])
            class_report_md = class_report_to_markdown(metrics['class_report'])
            matrix_md = matrix_to_markdown(matrix)
            detailed_sections.append(
                f"### {model_name} on {pipeline_name}\n\n"
                f"Confusion matrix:\n\n{matrix_md}\n\n"
                f"Per-class metrics:\n\n{class_report_md}\n"
            )
    
    report_content = (
        "# Full Experimental Results: Drought Class Prediction\n\n"
        f"NASA monthly missing values use the `{IMPUTATION_STRATEGY}` strategy; medians and missingness indicators, when selected, are fitted on the chronological training partition only.\n\n"
        "This report compares the performance of three models across three different feature sets. "
        f"Drought-class precision, recall, F1, and PR-AUC are reported for the minority class (SPEI-{SPEI_SCALE} < {SPEI_THRESHOLD:g}); weighted F1 is a complementary class-imbalance-aware summary.\n\n"
        f"All models predict the drought class for the next month using features available through the current month. "
        "LSTM sequences are restricted to one location and consecutive monthly observations.\n\n"
        "## Table 1: Precision for 'Drought' Class\n"
        "*(What percentage of drought predictions were correct?)*\n\n"
        f"{precision_df.round(4).to_markdown()}\n\n"
        "## Table 2: Recall for 'Drought' Class\n"
        "*(What percentage of actual droughts were found?)*\n\n"
        f"{recall_df.round(4).to_markdown()}\n\n"
        "## Table 3: F1-Score for 'Drought' Class\n"
        "*(The harmonic mean of Precision and Recall, representing a balanced score.)*\n\n"
        f"{f1_df.round(4).to_markdown()}\n\n"
        "## Table 4: Weighted F1-Score\n"
        "*(Overall F1 weighted by class support, useful for operational reliability under imbalance.)*\n\n"
        f"{weighted_f1_df.round(4).to_markdown()}\n\n"
        "## Table 5: Average 'Drought' F1-Score by Dataset\n\n"
        f"{avg_drought_f1_df.to_markdown()}\n\n"
        "## Table 6: Average Weighted F1-Score by Dataset\n\n"
        f"{avg_weighted_f1_df.to_markdown()}\n\n"
        "## Detailed Confusion Matrices and Per-Class Metrics\n\n"
        + "\n".join(detailed_sections)
    )
    print(report_content)
    os.makedirs(results_dir, exist_ok=True)
    with open(REPORT_OUTPUT_PATH, 'w') as f:
        f.write(report_content)
    print(f"\nFull report saved to: {REPORT_OUTPUT_PATH}")