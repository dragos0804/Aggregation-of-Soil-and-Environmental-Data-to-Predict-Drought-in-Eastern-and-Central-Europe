import math
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from sequence_utils import create_location_sequences
import run_comparison_experiments as primary


SPEI_COLUMN_NAME = "DROUGHT_INDEX"
SPEI_SCALE = os.getenv("SPEI_SCALE", "01").zfill(2)
SPEI_THRESHOLD = float(os.getenv("SPEI_THRESHOLD", "-1.0"))
SVM_SUBSAMPLE_SIZE = 5000
SVM_EVALUATION_SIZE = 100000
TEMPORAL_SPLITS = 3
# 16 covers every cell of the 4x4 quantile grid; empty cells are skipped.
SPATIAL_HOLDOUT_BLOCKS = int(os.getenv("SPATIAL_HOLDOUT_BLOCKS", "16"))
SPATIAL_BUFFERS_DEG = [float(v) for v in os.getenv("SPATIAL_BUFFERS_DEG", "0").split(",")]
RANDOM_FOREST_JOBS = int(os.getenv("EXTRAPOLATION_RF_JOBS", "1"))
LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH", "14"))
IMPUTATION_STRATEGY = os.getenv("NASA_IMPUTATION_STRATEGY", "median_indicators")
VALIDATION_MODELS = [
    model.strip()
    for model in os.getenv("EXTRAPOLATION_MODELS", "Random Forest,LSTM").split(",")
    if model.strip()
]
RESUME_CHECKPOINT = os.getenv("EXTRAPOLATION_RESUME", "1") == "1"


def categorize_drought_binary(spei_value):
    return 1 if spei_value < SPEI_THRESHOLD else 0


def ci95(std, n):
    if n <= 1 or pd.isna(std):
        return np.nan
    return 1.96 * std / math.sqrt(n)


def load_paths():
    script_dir = os.path.dirname(__file__)
    intermediary_dir = os.path.join(script_dir, "..", "data", "intermediary_steps")
    final_data_dir = os.path.join(script_dir, "..", "data")
    results_dir = os.path.join(script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    return {
        "SOTWIS_PATH": os.path.join(intermediary_dir, "synthetically_enhanced_sotwis.csv"),
        "NASA_PATH": primary.NASA_PATH,
        "FUSED_PATH": primary.FUSED_PATH,
        "SPEI_PATH": os.path.join(
            intermediary_dir,
            "complete_spei_benchmark.csv" if SPEI_SCALE == "01" else f"complete_spei{SPEI_SCALE}_benchmark.csv",
        ),
        "RESULTS_DIR": results_dir,
    }


def merge_with_spei(monthly_df, spei_df):
    merged_dfs = []
    monthly_groups = monthly_df.groupby(["YEAR", "MO"])
    spei_groups = spei_df.groupby(["YEAR", "MO"])

    for (year, month), group in monthly_groups:
        try:
            spei_group = spei_groups.get_group((year, month))
        except KeyError:
            continue
        kdtree = cKDTree(spei_group[["LAT", "LON"]].values)
        _, indices = kdtree.query(group[["LAT", "LON"]].values, k=1)
        matched_spei = spei_group.iloc[indices].reset_index(drop=True)
        group_copy = group.copy().reset_index(drop=True)
        group_copy[SPEI_COLUMN_NAME] = matched_spei[SPEI_COLUMN_NAME]
        merged_dfs.append(group_copy)

    if not merged_dfs:
        return pd.DataFrame()

    df = pd.concat(merged_dfs, ignore_index=True)
    df["is_drought"] = df[SPEI_COLUMN_NAME].apply(categorize_drought_binary)
    date_df = df[["YEAR", "MO"]].copy()
    date_df.rename(columns={"YEAR": "year", "MO": "month"}, inplace=True)
    date_df["day"] = 1
    df["date"] = pd.to_datetime(date_df)
    df = df.sort_values(by=["date", "LAT", "LON"]).reset_index(drop=True)
    return df


def add_next_month_targets(df):
    """Attach a next-month target while preserving one-location continuity."""
    df = df.sort_values(by=["LAT", "LON", "date"]).reset_index(drop=True).copy()
    location_groups = df.groupby(["LAT", "LON"], sort=False)
    df["target_drought"] = location_groups["is_drought"].shift(-1)
    df["target_date"] = location_groups["date"].shift(-1)
    expected_target_date = df["date"] + pd.offsets.MonthBegin(1)
    valid_target = df["target_date"].eq(expected_target_date)
    df.loc[~valid_target, "target_drought"] = np.nan
    df.loc[~valid_target, "target_date"] = pd.NaT
    return df.loc[df["target_drought"].notna()].reset_index(drop=True)


def prepare_nasa_df(paths):
    nasa_df = pd.read_csv(paths["NASA_PATH"])
    spei_df = pd.read_csv(paths["SPEI_PATH"])

    dynamic_features = ["GWETPROF", "GWETROOT", "GWETTOP"]
    aggregation_rules = {feat: ["mean", "min", "max", "std"] for feat in dynamic_features}

    nasa_monthly = nasa_df.groupby(["YEAR", "MO", "LAT", "LON"]).agg(aggregation_rules).reset_index()
    nasa_monthly.columns = ["_".join(col).strip() for col in nasa_monthly.columns.values]
    nasa_monthly.rename(columns={"YEAR_": "YEAR", "MO_": "MO", "LAT_": "LAT", "LON_": "LON"}, inplace=True)
    df = merge_with_spei(nasa_monthly, spei_df)
    features = [c for c in df.columns if c not in ["YEAR", "MO", "LAT", "LON", SPEI_COLUMN_NAME, "is_drought", "date"]]
    return df, features


def prepare_fused_df(paths):
    fused_df = pd.read_csv(paths["FUSED_PATH"])
    spei_df = pd.read_csv(paths["SPEI_PATH"])

    dynamic_features = ["GWETPROF", "GWETROOT", "GWETTOP"]
    static_features = [c for c in fused_df.columns if c not in dynamic_features + ["YEAR", "MO", "DY", "LAT", "LON"]]
    aggregation_rules = {feat: ["mean", "min", "max", "std"] for feat in dynamic_features}
    aggregation_rules.update({feat: "first" for feat in static_features})

    fused_monthly = fused_df.groupby(["YEAR", "MO", "LAT", "LON"]).agg(aggregation_rules).reset_index()
    fused_monthly.columns = ["_".join(col).strip() for col in fused_monthly.columns.values]
    fused_monthly.rename(columns={"YEAR_": "YEAR", "MO_": "MO", "LAT_": "LAT", "LON_": "LON"}, inplace=True)
    df = merge_with_spei(fused_monthly, spei_df)
    features = [c for c in df.columns if c not in ["YEAR", "MO", "LAT", "LON", SPEI_COLUMN_NAME, "is_drought", "date"]]
    return df, features


def scale_train_test(X_train, X_test):
    constant_cols = X_train.columns[X_train.std() == 0]
    variable_cols = X_train.columns[X_train.std() != 0]

    scaler = StandardScaler()
    X_train_scaled_vars = scaler.fit_transform(X_train[variable_cols])
    X_test_scaled_vars = scaler.transform(X_test[variable_cols])

    X_train_processed = np.hstack((X_train_scaled_vars, X_train[constant_cols].values))
    X_test_processed = np.hstack((X_test_scaled_vars, X_test[constant_cols].values))
    return X_train_processed, X_test_processed


def impute_train_median_with_indicators(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()
    feature_columns = list(X_train.columns)
    missing_columns = [column for column in feature_columns if X_train[column].isna().any() or X_test[column].isna().any()]
    for column in missing_columns:
        indicator = f"{column}__missing"
        X_train[indicator] = X_train[column].isna().astype(np.int8)
        X_test[indicator] = X_test[column].isna().astype(np.int8)
    medians = X_train[feature_columns].median().fillna(0.0)
    X_train[feature_columns] = X_train[feature_columns].fillna(medians)
    X_test[feature_columns] = X_test[feature_columns].fillna(medians)
    return X_train, X_test


def apply_imputation_strategy(X_train, X_test):
    if IMPUTATION_STRATEGY == "zero":
        return X_train.fillna(0.0), X_test.fillna(0.0)
    if IMPUTATION_STRATEGY == "median_indicators":
        return impute_train_median_with_indicators(X_train, X_test)
    raise ValueError(f"Unsupported NASA_IMPUTATION_STRATEGY: {IMPUTATION_STRATEGY}")


def create_sequences(X, y, sequence_length):
    raise ValueError("Use location-aware sequence construction with dates and coordinates")


def fit_lstm_sequences(frame, features, train_mask, test_mask):
    X_train = frame.loc[train_mask, features]
    X_test = frame.loc[test_mask, features]
    if X_train.empty or X_test.empty:
        return np.array([]), np.array([]), np.array([])
    sequence_frame = frame.loc[train_mask | test_mask]
    X_train, X_test = apply_imputation_strategy(X_train, X_test)
    constant_cols = X_train.columns[X_train.std() == 0]
    variable_cols = X_train.columns[X_train.std() != 0]
    scaler = StandardScaler()
    scaler.fit(X_train[variable_cols])
    all_train, all_test = apply_imputation_strategy(
        frame.loc[train_mask, features], frame.loc[test_mask, features]
    )
    all_imputed = pd.concat([all_train, all_test]).loc[sequence_frame.index]
    all_processed = np.hstack(
        (scaler.transform(all_imputed[variable_cols]), all_imputed[constant_cols].values)
    )
    sequences = create_location_sequences(
        all_processed,
        sequence_frame["is_drought"].to_numpy(),
        sequence_frame["date"].to_numpy(dtype="datetime64[M]"),
        sequence_frame["LAT"].to_numpy(),
        sequence_frame["LON"].to_numpy(),
        LSTM_SEQUENCE_LENGTH,
    )
    train_keys = {
        (latitude, longitude, str(target_date.astype("datetime64[M]")))
        for latitude, longitude, target_date in zip(
            frame.loc[train_mask, "LAT"],
            frame.loc[train_mask, "LON"],
            frame.loc[train_mask, "target_date"].to_numpy(dtype="datetime64[M]"),
        )
    }
    test_keys = {
        (latitude, longitude, str(target_date.astype("datetime64[M]")))
        for latitude, longitude, target_date in zip(
            frame.loc[test_mask, "LAT"],
            frame.loc[test_mask, "LON"],
            frame.loc[test_mask, "target_date"].to_numpy(dtype="datetime64[M]"),
        )
    }
    sequence_keys = [
        (latitude, longitude, str(target_date))
        for latitude, longitude, target_date in zip(
            sequences["target_latitudes"],
            sequences["target_longitudes"],
            sequences["target_dates"],
        )
    ]
    sequence_train = np.array([key in train_keys for key in sequence_keys])
    sequence_test = np.array([key in test_keys for key in sequence_keys])
    X_train_seq = sequences["features"][sequence_train]
    y_train_seq = sequences["targets"][sequence_train]
    X_test_seq = sequences["features"][sequence_test]
    y_test_seq = sequences["targets"][sequence_test]
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        return np.array([]), np.array([]), np.array([])

    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            Dropout(0.2), LSTM(50, return_sequences=False), Dropout(0.2),
            Dense(25, activation="relu"), Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train_seq), y=y_train_seq
    )
    model.fit(
        X_train_seq, y_train_seq, epochs=15, batch_size=64,
        callbacks=[EarlyStopping(monitor="loss", patience=3)],
        class_weight=dict(enumerate(weights)), verbose=0,
    )
    y_score = model.predict(X_test_seq, verbose=0).flatten()
    y_pred = (y_score > 0.5).astype(int)
    return y_test_seq, y_pred, y_score


def fit_predict(model_name, X_train, y_train, X_test):
    if model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=RANDOM_FOREST_JOBS,
        )
        model.fit(X_train, y_train)
        return model.predict(X_test), model.predict_proba(X_test)[:, 1]

    if model_name == "SVM (RBF Kernel)":
        X_fit = X_train
        y_fit = y_train
        if len(X_fit) > SVM_SUBSAMPLE_SIZE:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_fit), SVM_SUBSAMPLE_SIZE, replace=False)
            X_fit = X_fit[idx]
            y_fit = y_fit[idx]
        model = SVC(kernel="rbf", random_state=42, class_weight="balanced")
        model.fit(X_fit, y_fit)
        return model.predict(X_test), model.decision_function(X_test)

    if model_name == "LSTM":
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, LSTM_SEQUENCE_LENGTH)
        X_test_seq, _ = create_sequences(X_test, np.zeros(len(X_test), dtype=int), LSTM_SEQUENCE_LENGTH)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            return np.array([])

        model = Sequential(
            [
                LSTM(50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train_seq), y=y_train_seq
        )
        class_weight_dict = dict(enumerate(weights))

        model.fit(
            X_train_seq,
            y_train_seq,
            epochs=15,
            batch_size=64,
            callbacks=[EarlyStopping(monitor="loss", patience=3)],
            class_weight=class_weight_dict,
            verbose=0,
        )

        y_score = model.predict(X_test_seq, verbose=0).flatten()
        y_pred = (y_score > 0.5).astype(int)
        return y_pred, y_score

    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_fold(y_true, y_pred, y_score=None):
    if y_score is None:
        y_score = y_pred
    return {
        "precision_drought": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_drought": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "drought_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def temporal_extrapolation(
    df, features, dataset_name, model_name, completed_keys=None, on_row=None
):
    df = add_next_month_targets(df)
    unique_dates = np.sort(df["target_date"].unique())
    splits = TimeSeriesSplit(n_splits=TEMPORAL_SPLITS)
    rows = []
    completed_keys = completed_keys or set()

    for fold_idx, (train_idx, test_idx) in enumerate(splits.split(unique_dates), start=1):
        fold_key = (dataset_name, model_name, "Temporal Block (rolling-origin)", fold_idx)
        if fold_key in completed_keys:
            print(f"  Skipping completed temporal fold {fold_idx}")
            continue
        print(f"  Fitting temporal fold {fold_idx}/{TEMPORAL_SPLITS}")
        train_dates = unique_dates[train_idx]
        test_dates = unique_dates[test_idx]
        train_mask = df["target_date"].isin(train_dates)
        test_mask = df["target_date"].isin(test_dates)
        X_train = df.loc[train_mask, features]
        X_test = df.loc[test_mask, features]
        y_train = df.loc[train_mask, "target_drought"].astype(int).values
        y_test = df.loc[test_mask, "target_drought"].astype(int).values

        if model_name == "LSTM":
            y_eval, y_pred, y_score = fit_lstm_sequences(df, features, train_mask, test_mask)
            if len(y_pred) == 0:
                continue
            metrics = evaluate_fold(y_eval, y_pred, y_score)
            train_n = int(train_mask.sum())
            test_n = len(y_eval)
        else:
            X_train, X_test = apply_imputation_strategy(X_train, X_test)
            X_train_p, X_test_p = scale_train_test(X_train, X_test)
            y_eval = y_test
            X_eval = X_test_p
            if model_name == "SVM (RBF Kernel)" and len(X_eval) > SVM_EVALUATION_SIZE:
                eval_positions = np.random.default_rng(42).choice(
                    len(X_eval), SVM_EVALUATION_SIZE, replace=False
                )
                X_eval = X_eval[eval_positions]
                y_eval = y_test[eval_positions]
            y_pred, y_score = fit_predict(model_name, X_train_p, y_train, X_eval)
            metrics = evaluate_fold(y_eval, y_pred, y_score)
            train_n = len(y_train)
            test_n = len(y_eval)

        row = {
            "dataset": dataset_name,
            "model": model_name,
            "scheme": "Temporal Block (rolling-origin)",
            "fold": fold_idx,
            "train_n": train_n,
            "test_n": test_n,
            "train_drought_prevalence": float(y_train.mean()),
            "test_drought_prevalence": float(y_test.mean()),
            **metrics,
        }
        rows.append(row)
        if on_row is not None:
            on_row(row)

    return rows


def build_spatial_blocks(df, q=4):
    lat_bin = pd.qcut(df["LAT"], q=q, labels=False, duplicates="drop")
    lon_bin = pd.qcut(df["LON"], q=q, labels=False, duplicates="drop")
    block = lat_bin.astype(str) + "_" + lon_bin.astype(str)
    return block


def spatial_extrapolation(
    df, features, dataset_name, model_name, completed_keys=None, on_row=None, buffer_deg=0.0
):
    work = add_next_month_targets(df)
    work["block"] = build_spatial_blocks(work, q=4)
    scheme = "Spatial Block (leave-block-out)"
    if buffer_deg > 0:
        scheme = f"Spatial Block (leave-block-out, {buffer_deg:g} deg buffer)"

    block_counts = work["block"].value_counts().head(SPATIAL_HOLDOUT_BLOCKS)
    holdout_blocks = list(block_counts.index)

    rows = []
    completed_keys = completed_keys or set()
    for fold_idx, block in enumerate(holdout_blocks, start=1):
        fold_key = (dataset_name, model_name, scheme, fold_idx)
        if fold_key in completed_keys:
            print(f"  Skipping completed spatial fold {fold_idx}")
            continue
        print(f"  Fitting spatial fold {fold_idx}/{len(holdout_blocks)} (block {block}, buffer {buffer_deg:g})")
        test_mask = work["block"] == block
        train_mask = ~test_mask
        if buffer_deg > 0:
            lat_min, lat_max = work.loc[test_mask, "LAT"].min(), work.loc[test_mask, "LAT"].max()
            lon_min, lon_max = work.loc[test_mask, "LON"].min(), work.loc[test_mask, "LON"].max()
            near_block = (
                work["LAT"].between(lat_min - buffer_deg, lat_max + buffer_deg)
                & work["LON"].between(lon_min - buffer_deg, lon_max + buffer_deg)
            )
            train_mask = train_mask & ~near_block

        X_train = work.loc[train_mask, features]
        X_test = work.loc[test_mask, features]
        y_train = work.loc[train_mask, "target_drought"].astype(int).values
        y_test = work.loc[test_mask, "target_drought"].astype(int).values

        if len(X_test) == 0 or len(X_train) == 0:
            continue

        if model_name == "LSTM":
            y_eval, y_pred, y_score = fit_lstm_sequences(work, features, train_mask, test_mask)
            if len(y_pred) == 0:
                continue
            metrics = evaluate_fold(y_eval, y_pred, y_score)
            train_n = int(train_mask.sum())
            test_n = len(y_eval)
        else:
            X_train, X_test = apply_imputation_strategy(X_train, X_test)
            X_train_p, X_test_p = scale_train_test(X_train, X_test)
            y_eval = y_test
            X_eval = X_test_p
            if model_name == "SVM (RBF Kernel)" and len(X_eval) > SVM_EVALUATION_SIZE:
                eval_positions = np.random.default_rng(42).choice(
                    len(X_eval), SVM_EVALUATION_SIZE, replace=False
                )
                X_eval = X_eval[eval_positions]
                y_eval = y_test[eval_positions]
            y_pred, y_score = fit_predict(model_name, X_train_p, y_train, X_eval)
            metrics = evaluate_fold(y_eval, y_pred, y_score)
            train_n = int(train_mask.sum())
            test_n = len(y_eval)

        row = {
            "dataset": dataset_name,
            "model": model_name,
            "scheme": scheme,
            "fold": fold_idx,
            "holdout_block": block,
            "buffer_deg": buffer_deg,
            "train_n": train_n,
            "test_n": test_n,
            "train_drought_prevalence": float(y_train.mean()),
            "test_drought_prevalence": float(y_test.mean()),
            "holdout_lat_min": float(work.loc[test_mask, "LAT"].min()),
            "holdout_lat_max": float(work.loc[test_mask, "LAT"].max()),
            "holdout_lon_min": float(work.loc[test_mask, "LON"].min()),
            "holdout_lon_max": float(work.loc[test_mask, "LON"].max()),
            **metrics,
        }
        rows.append(row)
        if on_row is not None:
            on_row(row)

    return rows


def summarize_results(detail_df):
    grouped = detail_df.groupby(["dataset", "model", "scheme"], dropna=False)
    summary = grouped.agg(
        folds=("fold", "count"),
        precision_drought_mean=("precision_drought", "mean"),
        precision_drought_std=("precision_drought", "std"),
        recall_drought_mean=("recall_drought", "mean"),
        recall_drought_std=("recall_drought", "std"),
        drought_f1_mean=("drought_f1", "mean"),
        drought_f1_std=("drought_f1", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),
    ).reset_index()

    summary["test_drought_prevalence_mean"] = grouped["test_drought_prevalence"].mean().values
    summary["drought_f1_ci95"] = summary.apply(lambda r: ci95(r["drought_f1_std"], int(r["folds"])), axis=1)
    summary["weighted_f1_ci95"] = summary.apply(lambda r: ci95(r["weighted_f1_std"], int(r["folds"])), axis=1)
    return summary


def restrict_common_support(left, right):
    key_columns = ["YEAR", "MO", "LAT", "LON"]
    common_keys = left[key_columns].drop_duplicates().merge(
        right[key_columns].drop_duplicates(), on=key_columns, how="inner"
    )
    return (
        left.merge(common_keys, on=key_columns, how="inner"),
        right.merge(common_keys, on=key_columns, how="inner"),
    )


def write_checkpoint(rows, checkpoint_path):
    checkpoint_df = pd.DataFrame(rows)
    checkpoint_df.insert(0, "data_start_year", primary.DATA_START_YEAR)
    checkpoint_df.insert(1, "data_end_year", primary.DATA_END_YEAR)
    temporary_path = f"{checkpoint_path}.tmp"
    checkpoint_df.to_csv(temporary_path, index=False)
    os.replace(temporary_path, checkpoint_path)


def load_checkpoint(checkpoint_path):
    if not RESUME_CHECKPOINT or not os.path.exists(checkpoint_path):
        return []
    checkpoint_df = pd.read_csv(checkpoint_path)
    if checkpoint_df.empty:
        return []
    expected_period = {(primary.DATA_START_YEAR, primary.DATA_END_YEAR)}
    actual_period = set(zip(checkpoint_df["data_start_year"], checkpoint_df["data_end_year"]))
    if actual_period != expected_period:
        raise ValueError(
            f"Checkpoint period {actual_period} does not match {expected_period}; "
            "delete the checkpoint or set EXTRAPOLATION_RESUME=0."
        )
    checkpoint_df = checkpoint_df[checkpoint_df["model"].isin(VALIDATION_MODELS)]
    return checkpoint_df.drop(columns=["data_start_year", "data_end_year"]).to_dict("records")


def main():
    paths = load_paths()
    checkpoint_path = os.getenv(
        "EXTRAPOLATION_CHECKPOINT_PATH",
        os.path.join(paths["RESULTS_DIR"], "extrapolation_validation_checkpoint.csv"),
    )

    print("Preparing datasets for extrapolation validation...")
    nasa_df, nasa_features = primary.prepare_nasa_only_data()
    fused_df, fused_features = primary.prepare_fused_data()
    nasa_df, fused_df = restrict_common_support(nasa_df, fused_df)

    datasets = [
        ("NASA-Only", nasa_df, nasa_features),
        ("Aggregated", fused_df, fused_features),
    ]
    models = VALIDATION_MODELS

    all_rows = load_checkpoint(checkpoint_path)
    completed_keys = {
        (row["dataset"], row["model"], row["scheme"], int(row["fold"]))
        for row in all_rows
    }
    if all_rows:
        print(f"Resuming from {len(all_rows)} checkpointed folds")

    def checkpoint_row(row):
        all_rows.append(row)
        completed_keys.add((row["dataset"], row["model"], row["scheme"], int(row["fold"])))
        write_checkpoint(all_rows, checkpoint_path)
        print(f"  Checkpointed fold {int(row['fold'])}: {len(all_rows)} total rows")

    for dataset_name, df, features in datasets:
        for model_name in models:
            print(f"Running temporal extrapolation: {dataset_name} | {model_name}")
            temporal_extrapolation(
                df, features, dataset_name, model_name, completed_keys, checkpoint_row
            )

            print(f"Running spatial extrapolation: {dataset_name} | {model_name}")
            for buffer_deg in SPATIAL_BUFFERS_DEG:
                spatial_extrapolation(
                    df, features, dataset_name, model_name, completed_keys, checkpoint_row,
                    buffer_deg=buffer_deg,
                )

    detail_df = pd.DataFrame(all_rows)
    summary_df = summarize_results(detail_df)
    detail_df.insert(0, "data_start_year", primary.DATA_START_YEAR)
    detail_df.insert(1, "data_end_year", primary.DATA_END_YEAR)
    summary_df.insert(0, "data_start_year", primary.DATA_START_YEAR)
    summary_df.insert(1, "data_end_year", primary.DATA_END_YEAR)

    detail_path = os.getenv(
        "EXTRAPOLATION_DETAIL_OUTPUT_PATH",
        os.path.join(paths["RESULTS_DIR"], "extrapolation_validation_detailed.csv"),
    )
    summary_csv_path = os.getenv(
        "EXTRAPOLATION_SUMMARY_OUTPUT_PATH",
        os.path.join(paths["RESULTS_DIR"], "extrapolation_validation_summary.csv"),
    )
    summary_md_path = os.getenv(
        "EXTRAPOLATION_SUMMARY_MARKDOWN_PATH",
        os.path.join(paths["RESULTS_DIR"], "extrapolation_validation_summary.md"),
    )

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Extrapolation Validation Summary\n\n")
        f.write(summary_df.round(4).to_markdown(index=False))
        f.write("\n")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print("Saved extrapolation validation outputs:")
    print(f"  - {detail_path}")
    print(f"  - {summary_csv_path}")
    print(f"  - {summary_md_path}")


if __name__ == "__main__":
    main()
