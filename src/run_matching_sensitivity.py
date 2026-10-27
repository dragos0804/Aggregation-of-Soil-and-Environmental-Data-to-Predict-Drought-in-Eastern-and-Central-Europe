"""Sensitivity of the fused Random Forest result to ambiguous SOTWIS bounding-box matches.

Static soil attributes are reassigned per coordinate in memory under alternative
candidate-selection rules, so the 35M-row fused CSV does not need rebuilding.
"""

import os
import tempfile

os.environ.setdefault(
    'COMMON_SUPPORT_OUTPUT_PATH',
    os.path.join(tempfile.gettempdir(), 'matching_common_support_summary.csv'),
)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

import run_comparison_experiments as primary

OUTPUT_CSV = os.path.join(primary.results_dir, 'matching_ambiguity_sensitivity.csv')
DISAGREEMENT_CSV = os.path.join(primary.results_dir, 'matching_ambiguity_attribute_disagreement.csv')
BOUND_COLUMNS = ['BOTTOM_LEFT_LAT', 'UPPER_RIGHT_LAT', 'BOTTOM_LEFT_LON', 'UPPER_RIGHT_LON']


def candidate_map(coordinates, sotwis):
    bounds = sotwis[BOUND_COLUMNS].to_numpy()
    candidates = {}
    for lat, lon in coordinates:
        mask = (
            (lat >= bounds[:, 0]) & (lat <= bounds[:, 1])
            & (lon >= bounds[:, 2]) & (lon <= bounds[:, 3])
        )
        candidates[(lat, lon)] = np.where(mask)[0]
    return candidates


def choose(candidates, sotwis, rule, rng):
    areas = (
        (sotwis['UPPER_RIGHT_LAT'] - sotwis['BOTTOM_LEFT_LAT'])
        * (sotwis['UPPER_RIGHT_LON'] - sotwis['BOTTOM_LEFT_LON'])
    ).to_numpy()
    chosen = {}
    for key, idx in candidates.items():
        if rule == 'first':
            chosen[key] = idx[0]
        elif rule == 'last':
            chosen[key] = idx[-1]
        elif rule == 'smallest_box':
            chosen[key] = idx[np.argmin(areas[idx])]
        elif rule == 'random':
            chosen[key] = rng.choice(idx)
        else:
            raise ValueError(rule)
    return chosen


def reassign_static(frame, static_columns, sotwis, chosen):
    lookup = pd.DataFrame(
        [(lat, lon, index) for (lat, lon), index in chosen.items()],
        columns=['LAT', 'LON', '_sotwis_index'],
    )
    soil = sotwis[static_columns].reset_index().rename(columns={'index': '_sotwis_index'})
    lookup = lookup.merge(soil, on='_sotwis_index').drop(columns='_sotwis_index')
    return frame.drop(columns=static_columns).merge(lookup, on=['LAT', 'LON'], how='left')


def fit_rf(frame, features):
    frame = primary.add_next_month_targets(frame)
    frame = frame.loc[frame['target_drought'].notna()].copy()
    dates = np.sort(frame['target_date'].unique())
    split_index = min(max(1, int(len(dates) * primary.CHRONOLOGICAL_SPLIT_RATIO)), len(dates) - 1)
    train_rows = frame['target_date'] < dates[split_index]
    X_train, X_test = primary.apply_imputation_strategy(
        frame.loc[train_rows, features], frame.loc[~train_rows, features]
    )
    y_train = frame.loc[train_rows, 'target_drought'].astype(int)
    y_test = frame.loc[~train_rows, 'target_drought'].astype(int)
    variable = X_train.columns[X_train.std() != 0]
    constant = X_train.columns[X_train.std() == 0]
    scaler = StandardScaler()
    X_train_p = np.hstack((scaler.fit_transform(X_train[variable]), X_train[constant].values))
    X_test_p = np.hstack((scaler.transform(X_test[variable]), X_test[constant].values))
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train_p, y_train)
    y_pred = model.predict(X_test_p)
    y_score = model.predict_proba(X_test_p)[:, 1]
    return {
        'n_test': len(y_test),
        'n_locations': frame[['LAT', 'LON']].drop_duplicates().shape[0],
        'test_drought_prevalence': float(y_test.mean()),
        'precision_drought': precision_score(y_test, y_pred, zero_division=0),
        'recall_drought': recall_score(y_test, y_pred, zero_division=0),
        'f1_drought': f1_score(y_test, y_pred, zero_division=0),
        'pr_auc': average_precision_score(y_test, y_score),
    }


def main():
    sotwis = pd.read_csv(primary.SOTWIS_PATH)
    nasa_frame, nasa_features = primary.prepare_nasa_only_data()
    fused_frame, fused_features = primary.prepare_fused_data()
    nasa_frame, fused_frame = primary.restrict_to_common_support(nasa_frame, fused_frame)
    static_columns = [c for c in sotwis.columns if c not in BOUND_COLUMNS and c in fused_frame.columns]

    coordinates = list(fused_frame[['LAT', 'LON']].drop_duplicates().itertuples(index=False, name=None))
    candidates = candidate_map(coordinates, sotwis)
    counts = pd.Series({key: len(idx) for key, idx in candidates.items()})
    ambiguous = [key for key, idx in candidates.items() if len(idx) > 1]
    unambiguous = [key for key, idx in candidates.items() if len(idx) == 1]
    print(f'{len(coordinates)} coordinates; {len(ambiguous)} ambiguous; {len(unambiguous)} unique; '
          f'median candidates among ambiguous {counts[counts > 1].median():.0f}')

    # How different are the candidate soil records at ambiguous coordinates?
    overall_sd = sotwis[static_columns].std().replace(0, np.nan)
    disagreement_rows = []
    for column in static_columns:
        spreads = [np.ptp(sotwis[column].to_numpy()[candidates[key]]) for key in ambiguous]
        spreads = np.asarray(spreads)
        disagreement_rows.append({
            'attribute': column,
            'share_ambiguous_coords_with_differing_values': float(np.mean(spreads > 1e-9)),
            'median_candidate_range_in_sd_units': float(np.median(spreads) / overall_sd[column]),
        })
    pd.DataFrame(disagreement_rows).to_csv(DISAGREEMENT_CSV, index=False)
    print(pd.DataFrame(disagreement_rows).round(3).to_string(index=False))

    rows = []
    rng = np.random.default_rng(42)

    def record(scenario, frame, features, pipeline):
        print(f'Fitting RF: {scenario} / {pipeline}')
        rows.append({'scenario': scenario, 'pipeline': pipeline, **fit_rf(frame, features)})
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    for rule in ['first', 'last', 'smallest_box', 'random']:
        chosen = choose(candidates, sotwis, rule, rng)
        frame = fused_frame if rule == 'first' else reassign_static(fused_frame, static_columns, sotwis, chosen)
        record(f'All matched coordinates, {rule} candidate', frame, fused_features, 'Fused')

    unique_keys = pd.DataFrame(unambiguous, columns=['LAT', 'LON'])
    record('Unambiguous coordinates only', nasa_frame.merge(unique_keys, on=['LAT', 'LON']), nasa_features, 'NASA-Only')
    record('Unambiguous coordinates only', fused_frame.merge(unique_keys, on=['LAT', 'LON']), fused_features, 'Fused')

    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
