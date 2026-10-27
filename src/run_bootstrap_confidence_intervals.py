"""Location-block bootstrap confidence intervals for the primary drought-F1/weighted-F1 metrics.

Unlike the existing three-fold spatial/temporal extrapolation intervals (which use a
normal approximation across only three folds), this resamples the *unique test
locations* with replacement so that within-location month-to-month dependence is
preserved, then recomputes drought-F1 and weighted-F1 on each resample.
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import run_comparison_experiments as primary

RESULTS_DIR = primary.results_dir
OUTPUT_CSV = os.getenv(
    'BOOTSTRAP_OUTPUT_CSV',
    os.path.join(RESULTS_DIR, 'bootstrap_confidence_intervals.csv'),
)
OUTPUT_MD = os.getenv(
    'BOOTSTRAP_OUTPUT_MD',
    os.path.join(RESULTS_DIR, 'bootstrap_confidence_intervals.md'),
)

N_BOOTSTRAP = int(os.getenv('BOOTSTRAP_ITERATIONS', '1000'))
RANDOM_STATE = int(os.getenv('BOOTSTRAP_SEED', '42'))
MODEL_NAMES = tuple(
    model.strip()
    for model in os.getenv('BOOTSTRAP_MODELS', 'Random Forest,SVM (RBF Kernel)').split(',')
    if model.strip()
)


def fit_and_predict(frame, feature_columns, model_name):
    """Reproduce the primary train/test split, imputation, and scaling for one pipeline/model.

    Returns held-out (y_true, y_pred, location_codes) using the same configuration as
    run_comparison_experiments.run_model_pipeline, so point estimates match the primary results.
    """
    frame = primary.add_next_month_targets(frame)
    valid_rows = frame['target_drought'].notna()
    frame = frame.loc[valid_rows].copy()

    unique_target_dates = np.sort(frame['target_date'].unique())
    split_index = max(1, int(len(unique_target_dates) * primary.CHRONOLOGICAL_SPLIT_RATIO))
    split_index = min(split_index, len(unique_target_dates) - 1)
    split_date = unique_target_dates[split_index]
    train_rows = frame['target_date'] < split_date
    test_rows = ~train_rows

    X_train = frame.loc[train_rows, feature_columns]
    X_test = frame.loc[test_rows, feature_columns]
    y_train = frame.loc[train_rows, 'target_drought'].astype(int)
    y_test = frame.loc[test_rows, 'target_drought'].astype(int)

    X_train, X_test = primary.apply_imputation_strategy(X_train, X_test)

    constant_cols = X_train.columns[X_train.std() == 0]
    variable_cols = X_train.columns[X_train.std() != 0]
    scaler = StandardScaler()
    X_train_scaled_vars = scaler.fit_transform(X_train[variable_cols])
    X_test_scaled_vars = scaler.transform(X_test[variable_cols])
    X_train_processed = np.hstack((X_train_scaled_vars, X_train[constant_cols].values))
    X_test_processed = np.hstack((X_test_scaled_vars, X_test[constant_cols].values))

    if model_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        eval_frame = frame.loc[test_rows]
        return y_test.to_numpy(), y_pred, eval_frame[['LAT', 'LON']]

    if model_name == 'SVM (RBF Kernel)':
        model = SVC(kernel='rbf', random_state=42, class_weight='balanced')
        X_train_to_fit = X_train_processed
        y_train_to_fit = y_train
        if len(X_train_to_fit) > primary.SVM_SUBSAMPLE_SIZE:
            sample_indices = X_train.sample(n=primary.SVM_SUBSAMPLE_SIZE, random_state=42).index
            sample_mask = X_train.index.get_indexer(sample_indices)
            X_train_to_fit = X_train_processed[sample_mask]
            y_train_to_fit = y_train.loc[sample_indices]
        model.fit(X_train_to_fit, y_train_to_fit)

        X_eval = X_test_processed
        y_eval = y_test
        eval_index = y_test.index
        if len(X_eval) > primary.SVM_EVALUATION_SIZE:
            eval_indices = y_test.sample(n=primary.SVM_EVALUATION_SIZE, random_state=42).index
            eval_mask = y_test.index.get_indexer(eval_indices)
            X_eval = X_test_processed[eval_mask]
            y_eval = y_test.loc[eval_indices]
            eval_index = eval_indices
        y_pred = model.predict(X_eval)
        eval_frame = frame.loc[eval_index]
        return y_eval.to_numpy(), y_pred, eval_frame[['LAT', 'LON']]

    raise ValueError(f'Unsupported model for bootstrap CI: {model_name}')


def block_bootstrap_ci(y_true, y_pred, location_frame, n_bootstrap, random_state):
    """Resample unique (LAT, LON) test locations with replacement to preserve within-location dependence."""
    location_codes = location_frame.groupby(['LAT', 'LON'], sort=False).ngroup().to_numpy()
    unique_codes = np.unique(location_codes)
    code_to_indices = {code: np.where(location_codes == code)[0] for code in unique_codes}

    rng = np.random.default_rng(random_state)
    drought_f1_samples = np.empty(n_bootstrap)
    weighted_f1_samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled_codes = rng.choice(unique_codes, size=len(unique_codes), replace=True)
        sampled_indices = np.concatenate([code_to_indices[code] for code in sampled_codes])
        y_true_sample = y_true[sampled_indices]
        y_pred_sample = y_pred[sampled_indices]
        drought_f1_samples[i] = f1_score(y_true_sample, y_pred_sample, pos_label=1, zero_division=0)
        weighted_f1_samples[i] = f1_score(y_true_sample, y_pred_sample, average='weighted', zero_division=0)
    return drought_f1_samples, weighted_f1_samples, len(unique_codes)


def main():
    nasa_frame, nasa_features = primary.prepare_nasa_only_data()
    fused_frame, fused_features = primary.prepare_fused_data()
    nasa_frame, fused_frame = primary.restrict_to_common_support(nasa_frame, fused_frame)
    pipelines = {
        'NASA-Only (common support)': (nasa_frame, nasa_features),
        'Fused (SOTWIS + NASA, common support)': (fused_frame, fused_features),
    }

    rows = []
    for pipeline_name, (frame, feature_columns) in pipelines.items():
        for model_name in MODEL_NAMES:
            print(f'Bootstrapping {model_name} on {pipeline_name}...')
            y_true, y_pred, location_frame = fit_and_predict(frame, feature_columns, model_name)
            point_drought_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            point_weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            drought_samples, weighted_samples, n_locations = block_bootstrap_ci(
                y_true, y_pred, location_frame, N_BOOTSTRAP, RANDOM_STATE
            )
            for metric_name, point_estimate, samples in (
                ('drought_f1', point_drought_f1, drought_samples),
                ('weighted_f1', point_weighted_f1, weighted_samples),
            ):
                rows.append({
                    'data_start_year': primary.DATA_START_YEAR,
                    'data_end_year': primary.DATA_END_YEAR,
                    'svm_subsample_size': primary.SVM_SUBSAMPLE_SIZE,
                    'pipeline': pipeline_name,
                    'model': model_name,
                    'metric': metric_name,
                    'point_estimate': point_estimate,
                    'ci_lower_95': np.percentile(samples, 2.5),
                    'ci_upper_95': np.percentile(samples, 97.5),
                    'n_bootstrap': N_BOOTSTRAP,
                    'n_locations': n_locations,
                    'n_rows': len(y_true),
                })

            pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_MD, 'w', encoding='utf-8') as handle:
        handle.write('# Location-Block Bootstrap Confidence Intervals\n\n')
        handle.write(
            f'Location-block bootstrap ({N_BOOTSTRAP} resamples of unique (LAT, LON) test '
            'locations, sampled with replacement) applied to the primary SPEI-01, threshold '
            '-1.0, common-support RF/SVM held-out predictions. Resampling whole locations '
            'rather than individual rows preserves within-location month-to-month '
            'dependence. These intervals are distinct from the three-fold spatial/temporal '
            'extrapolation normal-approximation intervals reported elsewhere in the '
            'manuscript, which summarize variability across geographic/temporal blocks '
            'rather than sampling uncertainty of a single held-out set.\n\n'
        )
        handle.write(results_df.round(4).to_markdown(index=False))
        handle.write('\n')
    print(f'Saved bootstrap confidence intervals to {OUTPUT_CSV} and {OUTPUT_MD}')


if __name__ == '__main__':
    main()
