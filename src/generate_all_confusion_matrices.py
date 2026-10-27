import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

import run_comparison_experiments as rce

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def _prepare_static_and_seq_sets(frame, features):
    frame = rce.add_next_month_targets(frame)
    frame = frame.loc[frame["target_drought"].notna()].copy()
    target_dates = np.sort(frame["target_date"].unique())
    split_index = max(1, int(len(target_dates) * rce.CHRONOLOGICAL_SPLIT_RATIO))
    split_index = min(split_index, len(target_dates) - 1)
    split_date = target_dates[split_index]
    train_mask = frame["target_date"] < split_date
    test_mask = ~train_mask
    X_train = frame.loc[train_mask, features]
    X_test = frame.loc[test_mask, features]
    y_train = frame.loc[train_mask, "target_drought"].astype(int)
    y_test = frame.loc[test_mask, "target_drought"].astype(int)

    X_train, X_test = rce.apply_imputation_strategy(X_train, X_test)

    constant_cols = X_train.columns[X_train.std() == 0]
    variable_cols = X_train.columns[X_train.std() != 0]

    scaler = StandardScaler()
    X_train_scaled_vars = scaler.fit_transform(X_train[variable_cols])
    X_test_scaled_vars = scaler.transform(X_test[variable_cols])

    X_train_processed = np.hstack((X_train_scaled_vars, X_train[constant_cols].values))
    X_test_processed = np.hstack((X_test_scaled_vars, X_test[constant_cols].values))

    all_train, all_test = rce.apply_imputation_strategy(
        frame.loc[train_mask, features], frame.loc[test_mask, features]
    )
    all_imputed = pd.concat([all_train, all_test]).loc[frame.index]
    all_processed = np.hstack(
        (scaler.transform(all_imputed[variable_cols]), all_imputed[constant_cols].values)
    )
    sequence_result = rce.create_location_sequences(
        all_processed,
        frame["is_drought"].to_numpy(),
        frame["date"].to_numpy(dtype="datetime64[M]"),
        frame["LAT"].to_numpy(),
        frame["LON"].to_numpy(),
        rce.LSTM_SEQUENCE_LENGTH,
    )
    sequence_train = sequence_result["target_dates"] < split_date
    sequence_test = ~sequence_train
    X_train_seq = sequence_result["features"][sequence_train]
    y_train_seq = sequence_result["targets"][sequence_train]
    X_test_seq = sequence_result["features"][sequence_test]
    y_test_seq = sequence_result["targets"][sequence_test]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "X_train_seq": X_train_seq,
        "y_train_seq": y_train_seq,
        "X_test_seq": X_test_seq,
        "y_test_seq": y_test_seq,
    }


def _fit_predict_rf(data):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(data["X_train_processed"], data["y_train"])
    return data["y_test"].values, model.predict(data["X_test_processed"])


def _fit_predict_svm(data):
    X_fit = data["X_train_processed"]
    y_fit = data["y_train"]

    if len(X_fit) > rce.SVM_SUBSAMPLE_SIZE:
        sample_indices = data["X_train"].sample(n=rce.SVM_SUBSAMPLE_SIZE, random_state=42).index
        original_indices = data["X_train"].index.tolist()
        sample_mask = [original_indices.index(i) for i in sample_indices]
        X_fit = X_fit[sample_mask]
        y_fit = y_fit.loc[sample_indices]

    model = SVC(kernel="rbf", random_state=42, class_weight="balanced")
    model.fit(X_fit, y_fit)
    return data["y_test"].values, model.predict(data["X_test_processed"])


def _fit_predict_lstm(data):
    X_train_seq = data["X_train_seq"]
    y_train_seq = data["y_train_seq"]
    X_test_seq = data["X_test_seq"]
    y_test_seq = data["y_test_seq"]

    model = Sequential(
        [
            Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            LSTM(50, return_sequences=True),
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

    y_pred_prob = model.predict(X_test_seq, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    return y_test_seq, y_pred


def _evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    return cm, precision, recall, f1


def _plot_all_confusions(confusion_dict, output_path):
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)

    pipeline_order = ["SOTWIS-Only", "NASA-Only", "Aggregated"]
    model_order = ["Random Forest", "SVM (RBF)", "LSTM"]

    for i, model_name in enumerate(model_order):
        for j, pipeline_name in enumerate(pipeline_order):
            ax = axes[i, j]
            cm = confusion_dict[(model_name, pipeline_name)]
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

            ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks([0, 1], ["Pred ND", "Pred D"])
            ax.set_yticks([0, 1], ["True ND", "True D"])

            if i == 0:
                ax.set_title(pipeline_name, fontsize=11, pad=8)
            if j == 0:
                ax.set_ylabel(model_name, fontsize=11)

            for r_idx in range(2):
                for c_idx in range(2):
                    ax.text(
                        c_idx,
                        r_idx,
                        f"{cm[r_idx, c_idx]:,}\n{cm_norm[r_idx, c_idx]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                    )

    fig.suptitle(
        "Binary Drought Confusion Matrices for All Model-Dataset Combinations\n"
        "(count on first line, row-normalized rate on second line)",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    results_dir = repo_root / "results"
    fig_dir = repo_root.parent / "Science_Direct___Aggregation_of_Soil_and_Environmental_Data_to_Predict_Drought_in_Eastern_and_Central_Europe" / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing datasets...")
    sotwis_frame, sotwis_features = rce.prepare_sotwis_only_data()
    nasa_frame, nasa_features = rce.prepare_nasa_only_data()
    fused_frame, fused_features = rce.prepare_fused_data()
    nasa_frame, fused_frame = rce.restrict_to_common_support(nasa_frame, fused_frame)

    datasets = {
        "SOTWIS-Only": _prepare_static_and_seq_sets(sotwis_frame, sotwis_features),
        "NASA-Only": _prepare_static_and_seq_sets(nasa_frame, nasa_features),
        "Aggregated": _prepare_static_and_seq_sets(fused_frame, fused_features),
    }

    model_fns = {
        "Random Forest": _fit_predict_rf,
        "SVM (RBF)": _fit_predict_svm,
        "LSTM": _fit_predict_lstm,
    }

    metrics_rows = []
    confusion_dict = {}

    for model_name, model_fn in model_fns.items():
        for pipeline_name, data in datasets.items():
            print(f"Running {model_name} on {pipeline_name}...")
            y_true, y_pred = model_fn(data)
            cm, p, r, f1 = _evaluate(y_true, y_pred)
            confusion_dict[(model_name, pipeline_name)] = cm
            metrics_rows.append(
                {
                    "Model": model_name,
                    "Pipeline": pipeline_name,
                    "Precision_Drought": p,
                    "Recall_Drought": r,
                    "F1_Drought": f1,
                    "TN": int(cm[0, 0]),
                    "FP": int(cm[0, 1]),
                    "FN": int(cm[1, 0]),
                    "TP": int(cm[1, 1]),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.sort_values(["Model", "Pipeline"], inplace=True)

    metrics_csv_path = results_dir / "all_models_confusion_metrics.csv"
    metrics_md_path = results_dir / "all_models_confusion_metrics.md"
    matrix_fig_path = fig_dir / "confusion_matrices_all_models.png"

    metrics_df.to_csv(metrics_csv_path, index=False)
    with open(metrics_md_path, "w", encoding="utf-8") as f:
        f.write("# All Model/Dataset Confusion-Matrix Metrics\n\n")
        f.write(metrics_df.round(4).to_markdown(index=False))
        f.write("\n")

    _plot_all_confusions(confusion_dict, matrix_fig_path)

    print("Saved:")
    print(f"  - {metrics_csv_path}")
    print(f"  - {metrics_md_path}")
    print(f"  - {matrix_fig_path}")


if __name__ == "__main__":
    main()
