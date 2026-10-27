"""Training-free reference baselines on the primary chronological split.

Persistence predicts next-month drought from the current-month label. Site
climatology scores each location by its training-period drought frequency.
"""

import os
import tempfile

os.environ.setdefault(
    'COMMON_SUPPORT_OUTPUT_PATH',
    os.path.join(tempfile.gettempdir(), 'baseline_common_support_summary.csv'),
)

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import run_comparison_experiments as primary

OUTPUT_CSV = os.path.join(primary.results_dir, 'reference_baselines.csv')


def chronological_split(frame):
    frame = primary.add_next_month_targets(frame)
    frame = frame.loc[frame['target_drought'].notna()].copy()
    target_dates = np.sort(frame['target_date'].unique())
    split_index = max(1, int(len(target_dates) * primary.CHRONOLOGICAL_SPLIT_RATIO))
    split_index = min(split_index, len(target_dates) - 1)
    train_rows = frame['target_date'] < target_dates[split_index]
    return frame.loc[train_rows], frame.loc[~train_rows]


def score_row(pipeline, baseline, y_true, y_pred, y_score):
    return {
        'data_start_year': primary.DATA_START_YEAR,
        'data_end_year': primary.DATA_END_YEAR,
        'spei_threshold': primary.SPEI_THRESHOLD,
        'pipeline': pipeline,
        'baseline': baseline,
        'n_test': len(y_true),
        'test_drought_prevalence': float(np.mean(y_true)),
        'precision_drought': precision_score(y_true, y_pred, zero_division=0),
        'recall_drought': recall_score(y_true, y_pred, zero_division=0),
        'f1_drought': f1_score(y_true, y_pred, zero_division=0),
        'pr_auc': average_precision_score(y_true, y_score),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }


def evaluate_baselines(frame, pipeline):
    train, test = chronological_split(frame)
    y_true = test['target_drought'].astype(int).to_numpy()
    rows = []

    persistence = test['is_drought'].astype(int).to_numpy()
    rows.append(score_row(pipeline, 'Persistence (label at t)', y_true, persistence, persistence))

    site_rate = train.groupby(['LAT', 'LON'])['target_drought'].mean().rename('site_rate')
    overall_rate = float(train['target_drought'].mean())
    scores = (
        test[['LAT', 'LON']]
        .merge(site_rate, left_on=['LAT', 'LON'], right_index=True, how='left')['site_rate']
        .fillna(overall_rate)
        .to_numpy()
    )
    # Flag sites whose training drought frequency exceeds the pooled training frequency.
    climatology = (scores > overall_rate).astype(int)
    rows.append(score_row(pipeline, 'Site climatology (training frequency)', y_true, climatology, scores))
    return rows


def main():
    rows = []
    sotwis_frame, _ = primary.prepare_sotwis_only_data()
    rows += evaluate_baselines(sotwis_frame, 'SOTWIS-Only (separate static susceptibility experiment)')
    del sotwis_frame

    nasa_frame, _ = primary.prepare_nasa_only_data()
    fused_frame, _ = primary.prepare_fused_data()
    nasa_frame, _ = primary.restrict_to_common_support(nasa_frame, fused_frame)
    del fused_frame
    # NASA-Only and Fused share identical location-month keys and labels.
    rows += evaluate_baselines(nasa_frame, 'NASA-Only / Fused (common support)')

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_CSV, index=False)
    print(result.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
